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
        usages: cl.array.Array
        visible_states_prev: cl.array.Array
        visible_gates: cl.array.Array

    def __init__(self, cq: cl.CommandQueue, prog: cl.Program, hidden_size: (int, int, int, int) = (4, 4, 16, 1), vlds: [ VisibleLayerDesc ] = [], fd: io.IOBase = None):
        if fd is None:
            self.hidden_size = hidden_size

            num_hidden_columns = hidden_size[0] * hidden_size[1] * hidden_size[3]
            num_hidden_cells = num_hidden_columns * hidden_size[2]

            self.activations = cl.array.zeros(cq, (num_hidden_cells,), np.float32)
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
                num_weights = num_hidden_cells * area * vld.size[2] * vld.size[3]

                vl.weights = cl.clrandom.rand(cq, (num_weights,), np.float32, a=-0.01, b=0.01)
                vl.usages = cl.array.zeros(cq, (num_weights,), np.uint8)
                vl.visible_states_prev = cl.array.zeros(cq, (num_visible_columns * vld.size[3],), np.int32)

                vl.visible_gates = cl.array.zeros(cq, (num_visible_columns * vld.size[3],), np.float32)

                self.vls.append(vl)

            # Parameters
            self.lr = 2.0
            self.gcurve = 0.02

        else: # Load from h5py group
            self.hidden_size = struct.unpack("iiii", fd.read(4 * np.dtype(np.int32).itemsize))

            num_hidden_columns = self.hidden_size[0] * self.hidden_size[1] * self.hidden_size[3]
            num_hidden_cells = num_hidden_columns * self.hidden_size[2]

            self.activations = cl.array.empty(cq, (num_hidden_cells,), np.float32)
            self.hidden_states = cl.array.empty(cq, (num_hidden_columns,), np.int32)

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
                num_weights = num_hidden_cells * area * vld.size[2] * vld.size[3]

                vl.weights = cl.array.empty(cq, (num_weights,), np.float32)
                vl.usages = cl.array.empty(cq, (num_weights,), np.uint8)
                vl.visible_states_prev = cl.array.empty(cq, (num_visible_columns * vld.size[3],), np.int32)

                read_into_buffer(fd, vl.weights)
                read_into_buffer(fd, vl.usages)
                read_into_buffer(fd, vl.visible_states_prev)

                vl.visible_gates = cl.array.zeros(cq, (num_visible_columns * vld.size[3],), np.float32)

                self.vlds.append(vld)
                self.vls.append(vl)

            # Parameters
            self.lr, self.gcurve = struct.unpack("ff", fd.read(2 * np.dtype(np.float32).itemsize))

        # Kernels
        self.accum_activations_kernel = prog.accum_activations
        self.inhibit_activations_kernel = prog.inhibit_activations
        self.decoder_activate_gates_kernel = prog.decoder_activate_gates
        self.decoder_learn_kernel = prog.decoder_learn

    def step(self, cq: cl.CommandQueue, visible_states: [ cl.array.Array ], target_hidden_states: cl.array.Array, history_pos: int, target_pos: int, target_temporal_horizon: int, learn_enabled: bool = True):
        assert(len(visible_states) == len(self.vls))

        vec_hidden_size = np.array(list(self.hidden_size), dtype=np.int32)

        if learn_enabled:
            # Activate gates
            for i in range(len(self.vls)):
                vld = self.vlds[i]
                vl = self.vls[i]

                diam = vld.radius * 2 + 1

                vec_visible_size = np.array(list(vld.size), dtype=np.int32)

                self.decoder_activate_gates_kernel(cq, (vld.size[0], vld.size[1], vld.size[3]), None,
                        visible_states[i].data, vl.usages.data, vl.visible_gates.data,
                        vec_visible_size, vec_hidden_size, np.int32(vld.radius),
                        np.array([ math.ceil(diam * self.hidden_size[0] / vld.size[0] * 0.5), math.ceil(diam * self.hidden_size[1] / vld.size[1] * 0.5) ], np.int32),
                        np.int32(diam),
                        np.array([ vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1] ], dtype=np.float32),
                        np.array([ self.hidden_size[0] / vld.size[0], self.hidden_size[1] / vld.size[1] ], dtype=np.float32),
                        np.int32(history_pos),
                        np.float32(self.gcurve))

            for i in range(len(self.vls)):
                vld = self.vlds[i]
                vl = self.vls[i]

                diam = vld.radius * 2 + 1

                vec_visible_size = np.array(list(vld.size), dtype=np.int32)

                self.decoder_learn_kernel(cq, (self.hidden_size[0], self.hidden_size[1], self.hidden_size[2] * self.hidden_size[3]), (1, 1, self.hidden_size[2]),
                        vl.visible_states_prev.data, self.hidden_states.data, target_hidden_states.data, vl.visible_gates.data, self.activations.data, vl.weights.data, vl.usages.data,
                        vec_visible_size, vec_hidden_size, np.int32(vld.radius), np.int32(diam),
                        np.array([ vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1] ], dtype=np.float32),
                        np.int32(history_pos), np.int32(target_pos), np.int32(target_temporal_horizon),
                        np.float32(self.lr))

        # Clear
        self.activations.fill(np.float32(0))

        # Accumulate for all visible layers
        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            diam = vld.radius * 2 + 1

            vec_visible_size = np.array(list(vld.size), dtype=np.int32)

            self.accum_activations_kernel(cq, (self.hidden_size[0], self.hidden_size[1], self.hidden_size[2] * self.hidden_size[3]), (1, 1, self.hidden_size[2]),
                    visible_states[i].data, vl.weights.data, self.activations.data,
                    vec_visible_size, vec_hidden_size, np.int32(vld.radius), np.int32(diam),
                    np.array([ vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1] ], dtype=np.float32),
                    np.int32(history_pos), np.float32(1.0))

        self.inhibit_activations_kernel(cq, (self.hidden_size[0], self.hidden_size[1], self.hidden_size[3]), None, self.activations.data, self.hidden_states.data,
                vec_hidden_size,
                np.float32(1.0 / len(self.vls)))

        # Copy to prevs
        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            vl.visible_states_prev[:] = visible_states[i][:]

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
            write_from_buffer(fd, vl.usages)
            write_from_buffer(fd, vl.visible_states_prev)

        fd.write(struct.pack("ff", self.lr, self.gcurve))
