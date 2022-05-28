import numpy as np
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
import math
from dataclasses import dataclass
import h5py

class Decoder:
    @dataclass
    class VisibleLayerDesc:
        size: (int, int, int)
        radius: int

    class VisibleLayer:
        weights: cl.array.Array
        visible_states_prev: cl.array.Array

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

                num_visible_columns = vld.size[0] * vld.size[1]
                num_visible_cells = num_visible_columns * vld.size[2]

                diam = vld.radius * 2 + 1
                area = diam * diam
                num_weights = num_hidden_cells * area * vld.size[2]

                vl.weights = cl.clrandom.rand(cq, (num_weights,), np.float32, a=-0.01, b=0.01)
                vl.visible_states_prev = cl.array.zeros(cq, (num_visible_columns,), np.int32)

                self.vls.append(vl)

            # Hyperparameters
            self.lr = 1.0

        else: # Load from h5py group
            self.hidden_size = grp.attrs['hidden_size']

            self.activations = cl.array.empty(cq, (num_hidden_cells,), np.float32)
            self.hidden_states = cl.array.empty(cq, (num_hidden_columns,), np.int32)

            self.activations.set(grp['activations'])
            self.hidden_states.set(grp['hidden_states'])
            
            self.vlds = grp.attrs['vlds']
            self.vls = []

            for i in range(len(vlds)):
                vld = self.vlds[i]
                vl = self.VisibleLayer()

                num_visible_columns = vld.size[0] * vld.size[1]
                num_visible_cells = num_visible_columns * vld.size[2]

                diam = vld.radius * 2 + 1
                area = diam * diam
                num_weights = num_hidden_cells * area * vld.size[2]

                vl.weights = cl.array.empty(cq, (num_weights,), np.float32)
                vl.visible_states_prev = cl.array.empty(cq, (num_visible_columns,), np.int32)

                vl.weights.set(grp['weights' + str(i)])
                vl.visible_states_prev.set(grp['visible_states_prev' + str(i)])

                self.vls.append(vl)

            self.lr = grp.attrs['lr']

        # Kernels
        self.accum_activations_kernel = prog.accum_activation
        self.inhibit_activations_kernel = prog.inhibit_activations
        self.decoder_learn_kernel = prog.decoder_learn

    def step(self, cq: cl.CommandQueue, visible_states: [ cl.array.Array ], target_hidden_states: cl.array.Array, learn_enabled: bool = True):
        assert(len(visible_states) == len(self.vls))

        # Pad 3-vecs to 4-vecs
        vec_hidden_size = np.array(list(self.hidden_size) + [ 0 ], dtype=np.int32)

        if learn_enabled:
            for i in range(len(self.vls)):
                vld = self.vlds[i]
                vl = self.vls[i]

                diam = vld.radius * 2 + 1

                # Pad 3-vecs to 4-vecs
                vec_visible_size = np.array(list(vld.size) + [ 0 ], dtype=np.int32)

                self.decoder_learn_kernel(cq, self.hidden_size, (1, 1, self.hidden_size[2]),
                        vl.visible_states_prev.data, target_hidden_states.data, self.activations.data, vl.weights.data, 
                        vec_visible_size, vec_hidden_size, np.int32(vld.radius), np.int32(diam),
                        np.array([ vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1] ], dtype=np.float32),
                        np.float32(self.lr))

        # Clear
        self.activations.fill(np.float32(0))

        # Accumulate for all visible layers
        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            diam = vld.radius * 2 + 1

            # Pad 3-vecs to 4-vecs
            vec_visible_size = np.array(list(vld.size) + [ 0 ], dtype=np.int32)

            self.accum_activations_kernel(cq, self.hidden_size, (1, 1, self.hidden_size[2]),
                    visible_states[i].data, vl.weights.data, self.activations.data,
                    vec_visible_size, vec_hidden_size, np.int32(vld.radius), np.int32(diam),
                    np.array([ vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1] ], dtype=np.float32))

        self.inhibit_activations_kernel(cq, (self.hidden_size[0], self.hidden_size[1]), None, self.activations.data, self.hidden_states.data,
                vec_hidden_size,
                np.float32(1.0 / len(self.vls)))

        # Copy to prevs
        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            cl.enqueue_copy(cq, vl.visible_states_prev.data, visible_states[i].data)

    def write(self, grp: h5py.Group):
        grp.attrs['hidden_size'] = hidden_size

        grp.create_dataset('activations', data=self.activations.get())
        grp.create_dataset('hidden_states', data=self.hidden_states.get())

        grp.attrs['vlds'] = self.vlds

        for i in range(len(self.vls)):
            grp.create_dataset('weights' + str(i), data=self.vls[i].weights.get())
            grp.create_dataset('visible_states_prev' + str(i), data=self.vls[i].visible_states_prev.get())

        grp.attrs['lr'] = self.lr
