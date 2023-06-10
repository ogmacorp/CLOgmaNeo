# ----------------------------------------------------------------------------
#  CLOgmaNeo
#  Copyright(c) 2022 Ogma Intelligent Systems Corp. All rights reserved.
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
import h5py
import pickle

MAX_USAGE = 999999

class Encoder:
    @dataclass
    class VisibleLayerDesc:
        size: (int, int, int, int) # Width, height, column size, temporal size
        radius: int
        importance: float

    class VisibleLayer:
        weights: cl.array.Array
        usages: cl.array.Array
        reconstruction: cl.array.Array

    def __init__(self, cq: cl.CommandQueue, prog: cl.Program, hidden_size: (int, int, int) = (4, 4, 16), vlds: [ VisibleLayerDesc ] = [], grp: h5py.Group = None):
        if grp is None:
            self.hidden_size = hidden_size

            num_hidden_columns = hidden_size[0] * hidden_size[1]
            num_hidden_cells = num_hidden_columns * hidden_size[2]

            self.activations = cl.array.empty(cq, (num_hidden_cells,), np.float32)
            self.hidden_states = cl.array.zeros(cq, (num_hidden_columns,), np.int32)

            self.usage_sums = cl.array.empty(cq, (num_hidden_columns,), np.float32)
            self.hidden_gates = cl.array.zeros(cq, (num_hidden_columns,), np.float32)

            self.vlds = vlds
            self.vls = []

            for i in range(len(vlds)):
                vld = self.vlds[i]
                vl = self.VisibleLayer()

                num_visible_columns = vld.size[0] * vld.size[1] * vld.size[3]
                num_visible_cells = num_visible_columns * vld.size[2]

                diam = vld.radius * 2 + 1
                area = diam * diam
                num_weights = num_hidden_cells * area * vld.size[2] * vld.size[3]

                vl.weights = cl.clrandom.rand(cq, (num_weights,), np.float32, a=0.0, b=1.0)
                vl.usages = cl.array.zeros(cq, (num_weights,), np.uint8)
                vl.reconstruction = cl.array.empty(cq, (num_visible_cells,), np.float32)

                self.vls.append(vl)

            # Hyperparameters
            self.lr = 1.0
            self.gcurve = 8.0

        else: # Load from h5py group
            self.hidden_size = pickle.loads(grp.attrs['hidden_size'].tobytes())
            
            num_hidden_columns = self.hidden_size[0] * self.hidden_size[1]
            num_hidden_cells = num_hidden_columns * self.hidden_size[2]

            self.activations = cl.array.empty(cq, (num_hidden_cells,), np.float32)
            self.hidden_states = cl.array.empty(cq, (num_hidden_columns,), np.int32)

            self.usage_sums = cl.array.empty(cq, (num_hidden_columns,), np.float32)
            self.hidden_gates = cl.array.zeros(cq, (num_hidden_columns,), np.float32)

            self.hidden_states.set(np.array(grp['hidden_states'][:], np.int32))
            
            self.vlds = pickle.loads(grp.attrs['vlds'].tobytes())
            self.vls = []

            for i in range(len(self.vlds)):
                vld = self.vlds[i]
                vl = self.VisibleLayer()

                num_visible_columns = vld.size[0] * vld.size[1] * vld.size[3]
                num_visible_cells = num_visible_columns * vld.size[2]

                diam = vld.radius * 2 + 1
                area = diam * diam
                num_weights = num_hidden_cells * area * vld.size[2] * vld.size[3]

                vl.weights = cl.array.empty(cq, (num_weights,), np.float32)
                vl.usages = cl.array.empty(cq, (num_weights,), np.uint8)
                vl.reconstruction = cl.array.empty(cq, (num_visible_cells,), np.float32)

                vl.weights.set(np.array(grp['weights' + str(i)][:], np.float32))
                vl.usages.set(np.array(grp['usages' + str(i)][:], np.uint8))

                self.vls.append(vl)

            # Hyperparameters
            self.lr = pickle.loads(grp.attrs['lr'].tobytes())
            self.gcurve = pickle.loads(grp.attrs['gcurve'].tobytes())

        # Kernels
        self.accum_activations_kernel = prog.accum_activations
        self.inhibit_activations_kernel = prog.inhibit_activations
        self.encoder_accum_usages_kernel = prog.encoder_accum_usages
        self.encoder_activate_gates_kernel = prog.encoder_activate_gates
        self.encoder_learn_kernel = prog.encoder_learn

    def step(self, cq: cl.CommandQueue, visible_states: [ cl.array.Array ], history_pos: int, learn_enabled: bool = True):
        assert(len(visible_states) == len(self.vls))

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
            
            self.accum_activations_kernel(cq, self.hidden_size, (1, 1, self.hidden_size[2]),
                    visible_states[i].data, vl.weights.data, self.activations.data,
                    vec_visible_size, vec_hidden_size, np.int32(vld.radius), np.int32(diam),
                    np.array([ vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1] ], dtype=np.float32),
                    np.int32(history_pos), np.float32(vld.importance))

        self.inhibit_activations_kernel(cq, (self.hidden_size[0], self.hidden_size[1], 1), None, self.activations.data, self.hidden_states.data,
                vec_hidden_size,
                np.float32(1.0 / len(self.vls)))

        if learn_enabled:
            # Clear
            self.usage_sums.fill(np.float32(0))

            # Accumulate gates for all visible layers
            for i in range(len(self.vls)):
                vld = self.vlds[i]
                vl = self.vls[i]

                diam = vld.radius * 2 + 1

                vec_visible_size = np.array(list(vld.size), dtype=np.int32)
                
                self.encoder_accum_usages_kernel(cq, self.hidden_size, (1, 1, self.hidden_size[2]),
                        self.hidden_states.data, vl.usages.data, self.usage_sums.data,
                        vec_visible_size, vec_hidden_size, np.int32(vld.radius), np.int32(diam),
                        np.array([ vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1] ], dtype=np.float32),
                        np.float32(vld.importance))

            self.encoder_activate_gates_kernel(cq, (self.hidden_size[0], self.hidden_size[1], 1), None, self.usage_sums.data, self.hidden_gates.data,
                    vec_hidden_size,
                    np.float32(1.0 / len(self.vls)))

            for i in range(len(self.vls)):
                vld = self.vlds[i]
                vl = self.vls[i]

                diam = vld.radius * 2 + 1

                vec_visible_size = np.array(list(vld.size), dtype=np.int32)

                self.encoder_learn_kernel(cq, (vld.size[0], vld.size[1], vld.size[2] * vld.size[3]), (1, 1, vld.size[2]),
                        visible_states[i].data, self.hidden_states.data, self.hidden_gates.data, vl.weights.data, vl.reconstruction.data,
                        vec_visible_size, vec_hidden_size, np.int32(vld.radius),
                        np.array([ math.ceil(diam * self.hidden_size[0] / vld.size[0] * 0.5), math.ceil(diam * self.hidden_size[1] / vld.size[1] * 0.5) ], np.int32),
                        np.int32(diam),
                        np.array([ vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1] ], dtype=np.float32),
                        np.array([ self.hidden_size[0] / vld.size[0], self.hidden_size[1] / vld.size[1] ], dtype=np.float32),
                        np.int32(history_pos),
                        np.float32(self.lr))

    def write(self, grp: h5py.Group):
        grp.attrs['hidden_size'] = np.void(pickle.dumps(self.hidden_size))

        grp.create_dataset('hidden_states', data=self.hidden_states.get())

        grp.attrs['vlds'] = np.void(pickle.dumps(self.vlds))

        for i in range(len(self.vls)):
            grp.create_dataset('weights' + str(i), data=self.vls[i].weights.get())
            grp.create_dataset('usages' + str(i), data=self.vls[i].usages.get())

        grp.attrs['lr'] = np.void(pickle.dumps(self.lr))
        grp.attrs['gcurve'] = np.void(pickle.dumps(self.gcurve))


        
