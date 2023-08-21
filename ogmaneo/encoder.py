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

class Encoder:
    @dataclass
    class VisibleLayerDesc:
        size: (int, int, int, int) = (4, 4, 16, 1) # Width, height, column size, temporal size
        radius: int = 2
        importance: float = 1.0

    class VisibleLayer:
        weights: cl.array.Array
        reconstruction: cl.array.Array

    def __init__(self, cq: cl.CommandQueue, prog: cl.Program, hidden_size: (int, int, int) = (4, 4, 16), vlds: [ VisibleLayerDesc ] = [], fd: io.IOBase = None):
        if fd is None:
            self.hidden_size = hidden_size

            num_hidden_columns = hidden_size[0] * hidden_size[1]
            num_hidden_cells = num_hidden_columns * hidden_size[2]

            self.activations = cl.array.empty(cq, (num_hidden_cells,), np.float32)
            self.hidden_states = cl.array.zeros(cq, (num_hidden_columns,), np.int32)

            self.gates = cl.array.empty(cq, (num_hidden_columns,), np.float32)

            self.vlds = vlds
            self.vls = []

            for i in range(len(vlds)):
                vld = self.vlds[i]
                vl = self.VisibleLayer()

                num_visible_columns = vld.size[0] * vld.size[1]
                num_visible_cells = num_visible_columns * vld.size[2]

                diam = vld.radius * 2 + 1
                area = diam * diam
                num_weights = num_hidden_cells * area * vld.size[2] * vld.size[3]

                vl.weights = cl.clrandom.rand(cq, (num_weights,), np.float32, a=0.99, b=1.0)
                vl.reconstruction = cl.array.empty(cq, (num_visible_cells * vld.size[3],), np.float32)

                self.vls.append(vl)

            # Hyperparameters
            self.lr = 0.5
            self.gcurve = 0.1

        else: # Load from h5py group
            self.hidden_size = struct.unpack("iii", fd.read(3 * np.dtype(np.int32).itemsize))
            
            num_hidden_columns = self.hidden_size[0] * self.hidden_size[1]
            num_hidden_cells = num_hidden_columns * self.hidden_size[2]

            self.activations = cl.array.empty(cq, (num_hidden_cells,), np.float32)
            self.hidden_states = cl.array.empty(cq, (num_hidden_columns,), np.int32)

            self.gates = cl.array.empty(cq, (num_hidden_columns,), np.float32)

            read_into_buffer(fd, self.hidden_states)
            
            num_visible_layers = struct.unpack("i", fd.read(np.dtype(np.int32).itemsize))[0]

            self.vlds = []
            self.vls = []

            for i in range(num_visible_layers):
                vld = self.VisibleLayerDesc()
                vl = self.VisibleLayer()

                vld.size = struct.unpack("iiii", fd.read(4 * np.dtype(np.int32).itemsize))
                vld.radius, vld.importance = struct.unpack("if", fd.read(np.dtype(np.int32).itemsize + np.dtype(np.float32).itemsize))

                num_visible_columns = vld.size[0] * vld.size[1]
                num_visible_cells = num_visible_columns * vld.size[2]

                diam = vld.radius * 2 + 1
                area = diam * diam
                num_weights = num_hidden_cells * area * vld.size[2] * vld.size[3]

                vl.weights = cl.array.empty(cq, (num_weights,), np.float32)
                vl.reconstruction = cl.array.empty(cq, (num_visible_cells * vld.size[3],), np.float32)

                read_into_buffer(fd, vl.weights)

                self.vlds.append(vld)
                self.vls.append(vl)

            # Parameters
            self.lr, self.gcurve = struct.unpack("ff", fd.read(np.dtype(np.float32).itemsize))

        # Kernels
        self.encoder_activate_kernel = prog.encoder_activate.clone()
        self.encoder_update_gates_kernel = prog.encoder_update_gates.clone()
        self.encoder_learn_kernel = prog.encoder_learn.clone()

        self.encoder_activate_cache = KernelArgCache(self.encoder_activate_kernel)
        self.encoder_update_gates_cache = KernelArgCache(self.encoder_update_gates_kernel)
        self.encoder_learn_cache = KernelArgCache(self.encoder_learn_kernel)

    def step(self, cq: cl.CommandQueue, visible_states: [ cl.array.Array ], history_pos: int, learn_enabled: bool = True):
        assert len(visible_states) == len(self.vls)

        # Pad 3-vecs to 4-vecs
        vec_hidden_size = np.array(list(self.hidden_size) + [ 1 ], dtype=np.int32)

        # Clear
        self.activations.fill(np.float32(0))

        # Accumulate for all visible layers
        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            diam = vld.radius * 2 + 1

            vec_visible_size = np.array(list(vld.size), dtype=np.int32)
            
            finish = bool(i == len(self.vls) - 1)

            self.encoder_activate_cache.set_args(visible_states[i].data, vl.weights.data, self.activations.data, self.hidden_states.data,
                    vec_visible_size, vec_hidden_size, np.int32(vld.radius), np.int32(diam),
                    np.array([ vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1] ], dtype=np.float32),
                    np.int32(history_pos), np.float32(vld.importance / len(self.vls)), np.uint8(finish))

            cl.enqueue_nd_range_kernel(cq, self.encoder_activate_kernel, self.hidden_size, (1, 1, self.hidden_size[2]))

        if learn_enabled:
            # clear
            self.gates.fill(np.float32(0))

            # Accumulate gates
            for i in range(len(self.vls)):
                vld = self.vlds[i]
                vl = self.vls[i]

                diam = vld.radius * 2 + 1

                vec_visible_size = np.array(list(vld.size), dtype=np.int32)
                
                finish = bool(i == len(self.vls) - 1)

                self.encoder_update_gates_cache.set_args(self.hidden_states.data, vl.weights.data, self.gates.data,
                        vec_visible_size, vec_hidden_size, np.int32(vld.radius), np.int32(diam),
                        np.array([ vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1] ], dtype=np.float32),
                        np.float32(vld.importance / len(self.vls)), np.uint8(finish), np.float32(self.gcurve))

                cl.enqueue_nd_range_kernel(cq, self.encoder_update_gates_kernel, (self.hidden_size[0], self.hidden_size[1]), None)

            for i in range(len(self.vls)):
                vld = self.vlds[i]
                vl = self.vls[i]

                diam = vld.radius * 2 + 1

                vec_visible_size = np.array(list(vld.size), dtype=np.int32)

                self.encoder_learn_cache.set_args(visible_states[i].data, self.hidden_states.data, self.gates.data, vl.weights.data, vl.reconstruction.data,
                        vec_visible_size, vec_hidden_size, np.int32(vld.radius),
                        np.array([ math.ceil(diam * self.hidden_size[0] / vld.size[0] * 0.5), math.ceil(diam * self.hidden_size[1] / vld.size[1] * 0.5) ], np.int32),
                        np.int32(diam),
                        np.array([ vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1] ], dtype=np.float32),
                        np.array([ self.hidden_size[0] / vld.size[0], self.hidden_size[1] / vld.size[1] ], dtype=np.float32),
                        np.int32(history_pos),
                        np.float32(self.lr))

                cl.enqueue_nd_range_kernel(cq, self.encoder_learn_kernel, (vld.size[0], vld.size[1], vld.size[2] * vld.size[3]), (1, 1, vld.size[2]))

    def write(self, fd: io.IOBase):
        fd.write(struct.pack("iii", *self.hidden_size))

        write_from_buffer(fd, self.hidden_states)

        fd.write(struct.pack("i", len(self.vlds)))

        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            fd.write(struct.pack("iiiiif", *vld.size, vld.radius, vld.importance))

            write_from_buffer(fd, vl.weights)

        fd.write(struct.pack("ff", self.lr, self.gcurve))


        
