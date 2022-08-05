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

class Encoder:
    @dataclass
    class VisibleLayerDesc:
        size: (int, int, int, int) # Width, height, column size, temporal size
        radius: int

    class VisibleLayer:
        weights: cl.array.Array
        reconstruction: cl.array.Array

    def __init__(self, cq: cl.CommandQueue, prog: cl.Program, hidden_size: (int, int, int) = (4, 4, 16), vlds: [ VisibleLayerDesc ] = [], grp: h5py.Group = None):
        if grp is None:
            self.hidden_size = hidden_size

            num_hidden_columns = hidden_size[0] * hidden_size[1]
            num_hidden_cells = num_hidden_columns * hidden_size[2]

            self.activations = cl.array.zeros(cq, (num_hidden_cells,), np.float32)
            self.hidden_states = cl.array.zeros(cq, (num_hidden_columns,), np.int32)

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

                vl.weights = cl.clrandom.rand(cq, (num_weights,), np.float32, a=-1.0, b=1.0)
                vl.reconstruction = cl.array.empty(cq, (num_visible_cells,), np.float32)

                vl.visible_states_prev = cl.array.zeros(cq, (num_visible_columns,), np.int32)

                self.vls.append(vl)

            self.history_pos_prev = 0

            # Hyperparameters
            self.lr = 0.1
            self.reg = 0.01

        else: # Load from h5py group
            self.hidden_size = pickle.loads(grp.attrs['hidden_size'].tobytes())
            
            num_hidden_columns = self.hidden_size[0] * self.hidden_size[1]
            num_hidden_cells = num_hidden_columns * self.hidden_size[2]

            self.activations = cl.array.empty(cq, (num_hidden_cells,), np.float32)
            self.hidden_states = cl.array.empty(cq, (num_hidden_columns,), np.int32)

            self.activations.set(np.array(grp['activations'][:], np.float32))
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
                vl.reconstruction = cl.array.empty(cq, (num_visible_cells,), np.float32)

                vl.weights.set(np.array(grp['weights' + str(i)][:], np.float32))

                vl.visible_states_prev = cl.array.empty(cq, (num_visible_columns,), np.int32)
                vl.visible_states_prev.set(np.array(grp['visible_states_prev' + str(i)][:], np.int32))

                self.vls.append(vl)

            self.history_pos_prev = pickle.loads(grp.attrs['history_pos_prev'].tobytes())

            # Hyperparameters
            self.lr = pickle.loads(grp.attrs['lr'].tobytes())
            self.reg = pickle.loads(grp.attrs['reg'].tobytes())

        # Kernels
        self.accum_sparse_activations_kernel = prog.accum_sparse_activations
        self.sparse_activations_kernel = prog.sparse_activations
        self.dense_tanh_activations_kernel = prog.dense_tanh_activations
        self.encoder_learn_kernel = prog.encoder_learn

    def step(self, cq: cl.CommandQueue, visible_states: [ cl.array.Array ], errors: cl.array.Array, history_pos: int, learn_enabled: bool = True):
        assert(len(visible_states) == len(self.vls))

        # Pad 3-vecs to 4-vecs
        vec_hidden_size = np.array(list(self.hidden_size) + [ 1 ], dtype=np.int32)

        if learn_enabled:
            for i in range(len(self.vls)):
                vld = self.vlds[i]
                vl = self.vls[i]

                diam = vld.radius * 2 + 1

                vec_visible_size = np.array(list(vld.size), dtype=np.int32)

                self.encoder_learn_kernel(cq, self.hidden_size, (1, 1, self.hidden_size[2]),
                        vl.visible_states_prev.data, self.activations.data, errors.data, vl.weights.data, 
                        vec_visible_size, vec_hidden_size, np.int32(vld.radius),
                        np.int32(diam),
                        np.array([ vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1] ], dtype=np.float32),
                        np.int32(self.history_pos_prev),
                        np.float32(self.lr),
                        np.float32(self.reg))

        # Clear
        self.activations.fill(np.float32(0))

        # Accumulate for all visible layers
        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            diam = vld.radius * 2 + 1

            vec_visible_size = np.array(list(vld.size), dtype=np.int32)
            
            self.accum_sparse_activations_kernel(cq, self.hidden_size, (1, 1, self.hidden_size[2]),
                    visible_states[i].data, vl.weights.data, self.activations.data,
                    vec_visible_size, vec_hidden_size, np.int32(vld.radius), np.int32(diam),
                    np.array([ vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1] ], dtype=np.float32),
                    np.int32(history_pos))

        self.sparse_activations_kernel(cq, (self.hidden_size[0], self.hidden_size[1], 1), None, self.activations.data, self.hidden_states.data,
                vec_hidden_size,
                np.float32(1.0 / len(self.vls)))

        self.dense_tanh_activations_kernel(cq, self.hidden_size, None, self.activations.data,
                vec_hidden_size,
                np.float32(1.0)) # No scaling, as we did that in the previous activation step

        # Copy to prevs
        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            vl.visible_states_prev[:] = visible_states[i][:]

        self.history_pos_prev = history_pos

    def write(self, grp: h5py.Group):
        grp.attrs['hidden_size'] = np.void(pickle.dumps(self.hidden_size))

        grp.create_dataset('activations', data=self.activations.get())
        grp.create_dataset('hidden_states', data=self.hidden_states.get())

        grp.attrs['vlds'] = np.void(pickle.dumps(self.vlds))

        for i in range(len(self.vls)):
            grp.create_dataset('weights' + str(i), data=self.vls[i].weights.get())
            grp.create_dataset('visible_states_prev' + str(i), data=self.vls[i].visible_states_prev.get())

        grp.attrs['history_pos_prev'] = np.void(pickle.dumps(self.history_pos_prev))

        grp.attrs['lr'] = np.void(pickle.dumps(self.lr))
        grp.attrs['reg'] = np.void(pickle.dumps(self.reg))

