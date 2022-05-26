import numpy as np
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
import math
from dataclasses import dataclass

class Encoder:
    @dataclass
    class VisibleLayerDesc:
        size: (int, int, int)
        radius: int

    class VisibleLayer:
        weights: cl.array.Array
        reconstruction: cl.array.Array

    def __init__(self, cq: cl.CommandQueue, prog: cl.Program, hidden_size: (int, int, int), vlds: [ VisibleLayerDesc ]):
        self.hidden_size = hidden_size

        num_hidden_columns = hidden_size[0] * hidden_size[1]
        num_hidden_cells = num_hidden_columns * hidden_size[2]

        self.activations = cl.array.empty(cq, (num_hidden_cells,), np.float32)
        self.hidden_states = cl.array.zeros(cq, (num_hidden_columns,), np.int32)

        self.vlds = vlds
        self.vls = len(vlds) * [ self.VisibleLayer() ]

        for i in range(len(vlds)):
            vld = self.vlds[i]
            vl = self.vls[i]

            num_visible_columns = vld.size[0] * vld.size[1]
            num_visible_cells = num_visible_columns * vld.size[2]

            diam = vld.radius * 2 + 1
            area = diam * diam
            num_weights = num_hidden_cells * area * vld.size[2]

            vl.weights = cl.clrandom.rand(cq, (num_weights,), np.float32, a=0.0, b=0.01)
            vl.reconstruction = cl.array.zeros(cq, (num_visible_cells,), np.float32)

        # Kernels
        self.accum_activation_kernel = prog.accum_activation
        self.inhibit_activations_kernel = prog.inhibit_activations
        self.encoder_learn_kernel = prog.encoder_learn

        # Hyperparameters
        self.lr = 0.01

    def step(self, cq: cl.CommandQueue, visible_states: [ cl.array.Array ], learn_enabled: bool = True):
        # Clear
        self.activations.fill(np.float32(0))

        # Pad 3-vecs to 4-vecs
        vec_hidden_size = np.array(list(self.hidden_size) + [ 0 ], dtype=np.int32)

        # Accumulate for all visible layers
        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            diam = vld.radius * 2 + 1

            # Pad 3-vecs to 4-vecs
            vec_visible_size = np.array(list(vld.size) + [ 0 ], dtype=np.int32)

            self.accum_activation_kernel(cq, (self.hidden_size[0], self.hidden_size[1]), None,
                    visible_states[i].data, vl.weights.data, self.activations.data,
                    vec_visible_size, vec_hidden_size, np.int32(vld.radius), np.int32(diam),
                    np.array([ vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1] ], dtype=np.float32))

        self.inhibit_activations_kernel(cq, (self.hidden_size[0], self.hidden_size[1]), None, self.activations.data, self.hidden_states.data,
                vec_hidden_size)

        if learn_enabled:
            for i in range(len(self.vls)):
                vld = self.vlds[i]
                vl = self.vls[i]

                diam = vld.radius * 2 + 1

                # Pad 3-vecs to 4-vecs
                vec_visible_size = np.array(list(vld.size) + [ 0 ], dtype=np.int32)

                self.encoder_learn_kernel(cq, (vld.size[0], vld.size[1]), None,
                        visible_states[i].data, self.hidden_states.data, vl.weights.data, vl.reconstruction.data,
                        vec_visible_size, vec_hidden_size, np.int32(vld.radius),
                        np.array([ math.ceil(diam * self.hidden_size[0] / vld.size[0] * 0.5), math.ceil(diam * self.hidden_size[1] / vld.size[1] * 0.5) ], np.int32),
                        np.int32(diam),
                        np.array([ vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1] ], dtype=np.float32),
                        np.array([ self.hidden_size[0] / vld.size[0], self.hidden_size[1] / vld.size[1] ], dtype=np.float32),
                        np.float32(self.lr))

