# ----------------------------------------------------------------------------
#  CLOgmaNeo
#  Copyright(c) 2023 Ogma Intelligent Systems Corp. All rights reserved.
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
        size: (int, int, int, int) = (4, 4, 16, 1) # Width, height, column size, temporal size
        radius: int = 2
        
    class VisibleLayer:
        weights: cl.array.Array
        visible_states_prev: cl.array.Array

    def __init__(self, cq: cl.CommandQueue, prog: cl.Program, hidden_size: (int, int, int, int) = (4, 4, 16, 1), vlds: [ VisibleLayerDesc ] = [], fd: io.IOBase = None):
        if fd is None:
            self.hidden_size = hidden_size

            num_hidden_columns = hidden_size[0] * hidden_size[1]
            num_hidden_cells = num_hidden_columns * hidden_size[2]

            self.activations = cl.array.zeros(cq, (num_hidden_cells * hidden_size[3],), np.float32)
            self.activations_prev = cl.array.empty(cq, (num_hidden_cells * hidden_size[3],), np.float32)
            self.hidden_states = cl.array.zeros(cq, (num_hidden_columns * hidden_size[3],), np.int32)

            self.vlds = vlds
            self.vls = []

            for i in range(len(vlds)):
                vld = self.vlds[i]
                vl = self.VisibleLayer()

                num_visible_columns = vld.size[0] * vld.size[1]
                num_visible_cells = num_visible_columns * vld.size[2]

                diam = vld.radius * 2 + 1
                area = diam * diam
                num_weights = num_hidden_cells * hidden_size[3] * area * vld.size[2] * vld.size[3]

                vl.weights = cl.clrandom.rand(cq, (num_weights,), np.float32, a=0.0, b=0.01)
                vl.visible_states_prev = cl.array.zeros(cq, (num_visible_columns * vld.size[3],), np.int32)

                self.vls.append(vl)

            # Parameters
            self.lr = 1.0

        else: # Load from h5py group
            self.hidden_size = struct.unpack("iiii", fd.read(4 * np.dtype(np.int32).itemsize))

            num_hidden_columns = self.hidden_size[0] * self.hidden_size[1]
            num_hidden_cells = num_hidden_columns * self.hidden_size[2]

            self.activations = cl.array.empty(cq, (num_hidden_cells * self.hidden_size[3],), np.float32)
            self.activations_prev = cl.array.empty(cq, (num_hidden_cells * self.hidden_size[3],), np.float32)
            self.hidden_states = cl.array.empty(cq, (num_hidden_columns * self.hidden_size[3],), np.int32)

            read_into_buffer(fd, self.activations)
            read_into_buffer(fd, self.hidden_states)
            
            num_visible_layers = struct.unpack("i", fd.read(np.dtype(np.int32).itemsize))[0]

            self.vlds = []
            self.vls = []

            for i in range(num_visible_layers):
                vld = self.VisibleLayerDesc()
                vl = self.VisibleLayer()

                vld.size = struct.unpack("iiii", fd.read(4 * np.dtype(np.int32).itemsize))
                vld.radius = struct.unpack("i", fd.read(np.dtype(np.int32).itemsize))[0]

                num_visible_columns = vld.size[0] * vld.size[1]
                num_visible_cells = num_visible_columns * vld.size[2]

                diam = vld.radius * 2 + 1
                area = diam * diam
                num_weights = num_hidden_cells * self.hidden_size[3] * area * vld.size[2] * vld.size[3]

                vl.weights = cl.array.empty(cq, (num_weights,), np.float32)
                vl.visible_states_prev = cl.array.empty(cq, (num_visible_columns * vld.size[3],), np.int32)

                read_into_buffer(fd, vl.weights)
                read_into_buffer(fd, vl.visible_states_prev)

                self.vlds.append(vld)
                self.vls.append(vl)

            # Parameters
            self.lr = struct.unpack("f", fd.read(np.dtype(np.float32).itemsize))[0]

        # Kernels
        self.decoder_activate_kernel = prog.decoder_activate.clone()

        self.decoder_activate_cache = KernelArgCache(self.decoder_activate_kernel)

    def step(self, cq: cl.CommandQueue, visible_states: [ cl.array.Array ], target_hidden_states: cl.array.Array, history_pos: int, history_pos_prev: int, target_pos: int, target_temporal_horizon: int, learn_enabled: bool = True):
        assert len(visible_states) == len(self.vls)

        vec_hidden_size = np.array(list(self.hidden_size), dtype=np.int32)

        # Clear
        cl.enqueue_copy(cq, self.activations_prev.data, self.activations.data)

        self.activations.fill(np.float32(0))

        # Accumulate for all visible layers
        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            diam = vld.radius * 2 + 1

            vec_visible_size = np.array(list(vld.size), dtype=np.int32)

            finish = bool(i == len(self.vls) - 1)
            lr = float(i == 0 and learn_enabled) * self.lr

            self.decoder_activate_cache.set_args(visible_states[i].data, vl.visible_states_prev.data, target_hidden_states.data, self.activations_prev.data, vl.weights.data, self.activations.data, self.hidden_states.data,
                    vec_visible_size, vec_hidden_size, np.int32(vld.radius), np.int32(diam),
                    np.array([ vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1] ], dtype=np.float32),
                    np.int32(history_pos), np.int32(history_pos_prev), np.int32(target_pos), np.int32(target_temporal_horizon), np.float32(1.0 / len(self.vls)), np.uint8(finish), np.float32(lr))

            cl.enqueue_nd_range_kernel(cq, self.decoder_activate_kernel, (self.hidden_size[0], self.hidden_size[1], self.hidden_size[2] * self.hidden_size[3]), (1, 1, self.hidden_size[2]))

        # Copy to prevs
        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            cl.enqueue_copy(cq, vl.visible_states_prev.data, visible_states[i].data)

    def write(self, fd: io.IOBase):
        fd.write(struct.pack("iiii", *self.hidden_size))

        write_from_buffer(fd, self.activations)
        write_from_buffer(fd, self.hidden_states)

        fd.write(struct.pack("i", len(self.vlds)))

        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            fd.write(struct.pack("iiiii", *vld.size, vld.radius))

            write_from_buffer(fd, vl.weights)
            write_from_buffer(fd, vl.visible_states_prev)

        fd.write(struct.pack("f", self.lr))
