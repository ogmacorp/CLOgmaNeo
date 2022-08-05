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

class ImageEnc:
    @dataclass
    class VisibleLayerDesc:
        size: (int, int, int) = (32, 32, 3)
        radius: int = 4

    class VisibleLayer:
        weights: cl.array.Array
        reconstruction: cl.array.Array

    def __init__(self, cq: cl.CommandQueue, prog: cl.Program, prog_extra: cl.Program, hidden_size: (int, int, int) = (4, 4, 16), vlds: [ VisibleLayerDesc ] = [], grp: h5py.Group = None):
        if grp is None:
            self.hidden_size = hidden_size

            num_hidden_columns = hidden_size[0] * hidden_size[1]
            num_hidden_cells = num_hidden_columns * hidden_size[2]

            self.activations = cl.array.empty(cq, (num_hidden_cells,), np.float32)
            self.hidden_states = cl.array.zeros(cq, (num_hidden_columns,), np.int32)
            self.hidden_rates = cl.array.empty(cq, (num_hidden_cells,), np.float32)

            self.hidden_rates.fill(np.float32(1))

            self.vlds = vlds
            self.vls = []

            for i in range(len(vlds)):
                vld = self.vlds[i]
                vl = self.VisibleLayer()

                num_visible_columns = vld.size[0] * vld.size[1]
                num_visible_cells = num_visible_columns * vld.size[2]

                diam = vld.radius * 2 + 1
                area = diam * diam
                num_weights = num_hidden_cells * area * vld.size[2]

                vl.weights = cl.clrandom.rand(cq, (num_weights,), np.float32, a=0.0, b=0.01)
                vl.reconstruction = cl.array.zeros(cq, (num_visible_cells,), np.float32)

                self.vls.append(vl)

            # Hyperparameters
            self.lr = 0.01
            self.falloff = 0.1

        else: # Load from h5py group
            self.hidden_size = pickle.loads(grp.attrs['hidden_size'].tobytes())
            
            num_hidden_columns = self.hidden_size[0] * self.hidden_size[1]
            num_hidden_cells = num_hidden_columns * self.hidden_size[2]

            self.activations = cl.array.empty(cq, (num_hidden_cells,), np.float32)
            self.hidden_states = cl.array.empty(cq, (num_hidden_columns,), np.int32)
            self.hidden_rates = cl.array.empty(cq, (num_hidden_cells,), np.float32)

            self.hidden_states.set(np.array(grp['hidden_states'][:], np.int32))
            self.hidden_rates.set(np.array(grp['hidden_rates'][:], np.float32))

            self.vlds = pickle.loads(grp.attrs['vlds'].tobytes())
            self.vls = []

            for i in range(len(self.vlds)):
                vld = self.vlds[i]
                vl = self.VisibleLayer()

                num_visible_columns = vld.size[0] * vld.size[1]
                num_visible_cells = num_visible_columns * vld.size[2]

                diam = vld.radius * 2 + 1
                area = diam * diam
                num_weights = num_hidden_cells * area * vld.size[2]

                vl.weights = cl.array.empty(cq, (num_weights,), np.float32)
                vl.reconstruction = cl.array.empty(cq, (num_visible_cells,), np.float32)

                vl.weights.set(np.array(grp['weights' + str(i)][:], np.float32))
                vl.reconstruction.set(np.array(grp['reconstruction' + str(i)][:], np.float32))

                self.vls.append(vl)

            # Hyperparameters
            self.lr = pickle.loads(grp.attrs['lr'].tobytes())
            self.falloff = pickle.loads(grp.attrs['falloff'].tobytes())

        # Kernels
        self.image_enc_accum_activations_kernel = prog_extra.image_enc_accum_activations
        self.sparse_activations_kernel = prog.sparse_activations
        self.image_enc_learn_kernel = prog_extra.image_enc_learn
        self.image_enc_decay_kernel = prog_extra.image_enc_decay
        self.image_enc_reconstruct_kernel = prog_extra.image_enc_reconstruct

    def step(self, cq: cl.CommandQueue, visible_states: [ cl.array.Array ], learn_enabled: bool = True):
        assert(len(visible_states) == len(self.vls))

        # Clear
        self.activations.fill(np.float32(0))

        # Pad 3-vecs to 4-vecs
        vec_hidden_size = np.array(list(self.hidden_size) + [ 1 ], dtype=np.int32)

        # Accumulate for all visible layers
        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            diam = vld.radius * 2 + 1

            # Pad 3-vecs to 4-vecs
            vec_visible_size = np.array(list(vld.size) + [ 1 ], dtype=np.int32)

            self.image_enc_accum_activations_kernel(cq, self.hidden_size, (1, 1, self.hidden_size[2]),
                    visible_states[i].data, vl.weights.data, self.activations.data,
                    vec_visible_size, vec_hidden_size, np.int32(vld.radius), np.int32(diam),
                    np.array([ vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1] ], dtype=np.float32))

        self.sparse_activations_kernel(cq, (self.hidden_size[0], self.hidden_size[1], 1), None, self.activations.data, self.hidden_states.data,
                vec_hidden_size,
                np.float32(1.0 / len(self.vls)))

        if learn_enabled:
            for i in range(len(self.vls)):
                vld = self.vlds[i]
                vl = self.vls[i]

                diam = vld.radius * 2 + 1

                # Pad 3-vecs to 4-vecs
                vec_visible_size = np.array(list(vld.size) + [ 1 ], dtype=np.int32)

                self.image_enc_learn_kernel(cq, self.hidden_size, (1, 1, self.hidden_size[2]),
                        visible_states[i].data, self.hidden_states.data, self.hidden_rates.data, vl.weights.data,
                        vec_visible_size, vec_hidden_size, np.int32(vld.radius), np.int32(diam),
                        np.array([ vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1] ], dtype=np.float32),
                        np.float32(self.falloff))

        # Decay
        self.image_enc_decay_kernel(cq, self.hidden_size, None,
                self.hidden_states.data, self.hidden_rates.data,
                vec_hidden_size,
                np.float32(self.lr),
                np.float32(self.falloff))

    def reconstruct(self, cq: cl.CommandQueue, hidden_states: cl.array.Array, indices: [ int ] = []):
        if len(indices) == 0: # Empty means all indices
            indices = [ i for i in range(len(self.vls)) ]

        # Pad 3-vecs to 4-vecs
        vec_hidden_size = np.array(list(self.hidden_size) + [ 1 ], dtype=np.int32)

        for i in indices:
            vld = self.vlds[i]
            vl = self.vls[i]

            diam = vld.radius * 2 + 1

            # Pad 3-vecs to 4-vecs
            vec_visible_size = np.array(list(vld.size) + [ 1 ], dtype=np.int32)

            self.image_enc_reconstruct_kernel(cq, vld.size, (1, 1, vld.size[2]),
                    hidden_states.data, vl.weights.data, vl.reconstruction.data,
                    vec_visible_size, vec_hidden_size, np.int32(vld.radius),
                    np.array([ math.ceil(diam * self.hidden_size[0] / vld.size[0] * 0.5), math.ceil(diam * self.hidden_size[1] / vld.size[1] * 0.5) ], np.int32),
                    np.int32(diam),
                    np.array([ vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1] ], dtype=np.float32),
                    np.array([ self.hidden_size[0] / vld.size[0], self.hidden_size[1] / vld.size[1] ], dtype=np.float32))

    def write(self, grp: h5py.Group):
        grp.attrs['hidden_size'] = np.void(pickle.dumps(self.hidden_size))

        grp.create_dataset('hidden_states', data=self.hidden_states.get())
        grp.create_dataset('hidden_rates', data=self.hidden_rates.get())

        grp.attrs['vlds'] = np.void(pickle.dumps(self.vlds))

        for i in range(len(self.vls)):
            grp.create_dataset('weights' + str(i), data=self.vls[i].weights.get())
            grp.create_dataset('reconstruction' + str(i), data=self.vls[i].reconstruction.get())

        grp.attrs['lr'] = np.void(pickle.dumps(self.lr))
        grp.attrs['falloff'] = np.void(pickle.dumps(self.falloff))
