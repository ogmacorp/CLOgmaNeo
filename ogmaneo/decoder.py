# ----------------------------------------------------------------------------
#  CLOgmaNeo
#  Copyright(c) 2023-2025 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of CLOgmaNeo is licensed to you under the terms described
#  in the CLOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

import numpy as np
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
import math
from dataclasses import dataclass
import io
import struct
from .helpers import *

class Decoder:
    @dataclass
    class VisibleLayerDesc:
        size: (int, int, int, int) = (5, 5, 16) # Width, height, column size
        radius: int = 2
        
    class VisibleLayer:
        weights: cl.array.Array

    def __init__(self, cq: cl.CommandQueue, prog: cl.Program, hidden_size: (int, int, int) = (5, 5, 16), num_dendrites_per_cell=4, vlds: [VisibleLayerDesc] = [], fd: io.IOBase = None):
        if fd is None:
            self.hidden_size = hidden_size
            self.num_dendrites_per_cell = num_dendrites_per_cell

            num_hidden_columns = hidden_size[0] * hidden_size[1]
            num_hidden_cells = num_hidden_columns * hidden_size[2]
            num_dendrites = num_hidden_cells * num_dendrites_per_cell

            self.dendrite_activations = cl.array.zeros(cq, (num_dendrites,), np.float32)
            self.hidden_activations = cl.array.zeros(cq, (num_hidden_cells,), np.float32)
            self.hidden_states = cl.array.zeros(cq, (num_hidden_columns,), np.int32)

            self.vlds = vlds
            self.vls = []

            for i in range(len(vlds)):
                vld = self.vlds[i]
                vl = self.VisibleLayer()

                num_visible_columns = vld.size[0] * vld.size[1]
                num_visible_cells = num_visible_columns * vld.size[2]

                diam = vld.radius * 2 + 1
                area = diam * diam
                num_weights = num_dendrites * area * vld.size[2]

                vl.weights = cl.array.to_device(cq, np.random.randint(122, 132, size=num_weights, dtype=np.uint8))

                self.vls.append(vl)

            # Parameters
            self.scale = 8.0
            self.lr = 0.2

        else: # Load
            self.hidden_size = struct.unpack("iii", fd.read(3 * np.dtype(np.int32).itemsize))
            self.num_dendrites_per_cell = struct.unpack("i", fd.read(np.dtype(np.int32).itemsize))[0]

            num_hidden_columns = self.hidden_size[0] * self.hidden_size[1]
            num_hidden_cells = num_hidden_columns * self.hidden_size[2]
            num_dendrites = num_hidden_cells * self.num_dendrites_per_cell

            self.dendrite_activations = cl.array.empty(cq, (num_dendrites,), np.float32)
            self.hidden_activations = cl.array.empty(cq, (num_hidden_cells,), np.float32)
            self.hidden_states = cl.array.empty(cq, (num_hidden_columns,), np.int32)

            read_into_buffer(fd, self.dendrite_activations)
            read_into_buffer(fd, self.hidden_activations)
            read_into_buffer(fd, self.hidden_states)
            
            num_visible_layers = struct.unpack("i", fd.read(np.dtype(np.int32).itemsize))[0]

            self.vlds = []
            self.vls = []

            for i in range(num_visible_layers):
                vld = self.VisibleLayerDesc()
                vl = self.VisibleLayer()

                vld.size = struct.unpack("iii", fd.read(3 * np.dtype(np.int32).itemsize))
                vld.radius = struct.unpack("i", fd.read(np.dtype(np.int32).itemsize))[0]

                num_visible_columns = vld.size[0] * vld.size[1]
                num_visible_cells = num_visible_columns * vld.size[2]

                diam = vld.radius * 2 + 1
                area = diam * diam
                num_weights = num_dendrites * area * vld.size[2]

                vl.weights = cl.array.empty(cq, (num_weights,), np.uint8)

                read_into_buffer(fd, vl.weights)

                self.vlds.append(vld)
                self.vls.append(vl)

            # Parameters
            self.scale, self.lr = struct.unpack("ff", fd.read(2 * np.dtype(np.float32).itemsize))

        # Kernels
        self.decoder_activate_kernel = prog.decoder_activate
        self.decoder_learn_kernel = prog.decoder_learn

        self.decoder_activate_cache = KernelArgCache(self.decoder_activate_kernel)
        self.decoder_learn_cache = KernelArgCache(self.decoder_learn_kernel)

    def activate(self, cq: cl.CommandQueue, visible_states: [cl.array.Array]):
        assert len(visible_states) == len(self.vls)

        vec_hidden_size = np.array(list(self.hidden_size) + [1], dtype=np.int32)

        # Clear
        self.dendrite_activations.fill(np.float32(0))

        # Accumulate for all visible layers
        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            diam = vld.radius * 2 + 1

            vec_visible_size = np.array(list(vld.size) + [1], dtype=np.int32)

            finish = bool(i == (len(self.vls) - 1))

            self.decoder_activate_cache.set_args(visible_states[i].data, vl.weights.data,
                    self.dendrite_activations.data, self.hidden_activations.data, self.hidden_states.data,
                    vec_visible_size, vec_hidden_size, np.int32(self.num_dendrites_per_cell), np.int32(vld.radius), np.int32(diam),
                    np.array([vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1]], dtype=np.float32),
                    np.float32(1.0 / len(self.vls)), np.uint8(finish), np.float32(self.scale))

            cl.enqueue_nd_range_kernel(cq, self.decoder_activate_kernel, self.hidden_size, (1, 1, self.hidden_size[2]))

    def learn(self, cq: cl.CommandQueue, visible_states: [cl.array.Array], target_hidden_states: cl.array.Array):
        assert len(visible_states) == len(self.vls)

        vec_hidden_size = np.array(list(self.hidden_size) + [1], dtype=np.int32)

        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            diam = vld.radius * 2 + 1

            vec_visible_size = np.array(list(vld.size) + [1], dtype=np.int32)

            self.decoder_learn_cache.set_args(visible_states[i].data, target_hidden_states.data,
                    self.dendrite_activations.data, self.hidden_activations.data, vl.weights.data,
                    vec_visible_size, vec_hidden_size, np.int32(self.num_dendrites_per_cell), np.int32(vld.radius), np.int32(diam),
                    np.array([vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1]], dtype=np.float32),
                    np.float32(self.lr))

            cl.enqueue_nd_range_kernel(cq, self.decoder_learn_kernel, self.hidden_size, (1, 1, self.hidden_size[2]))

    def write(self, fd: io.IOBase):
        fd.write(struct.pack("iii", *self.hidden_size))
        fd.write(struct.pack("i", self.num_dendrites_per_cell))

        write_from_buffer(fd, self.dendrite_activations)
        write_from_buffer(fd, self.hidden_activations)
        write_from_buffer(fd, self.hidden_states)

        fd.write(struct.pack("i", len(self.vlds)))

        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            fd.write(struct.pack("iiii", *vld.size, vld.radius))

            write_from_buffer(fd, vl.weights)

        fd.write(struct.pack("ff", self.scale, self.lr))
