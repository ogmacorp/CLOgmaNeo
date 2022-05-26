import numpy as np
import pyopencl as cl
from dataclasses import dataclass

class Encoder:
    @dataclass
    class VisibleLayerDesc:
        size: tuple(int, int, int)
        radius: int

    class VisibleLayer:
        weights: cl.array.Array
        reconstruction: cl.array.Array

    def __init__(self, cq: cl.CommandQueue, prog: cl.Program, hidden_size: tuple(int, int, int), vlds: [ VisibleLayerDesc ]):
        self.hidden_size = hidden_size

        num_hidden_columns = hidden_size[0] * hidden_size[1]
        num_hidden_cells = num_hidden_columns * hidden_size[2]

        self.activations = cl.array.empty(cq, (num_hidden_cells,), np.float32)
        self.hidden_states = cl.array.zeros(cq, (num_hidden_columns,), np.int32)

        self.vlds = vlds
        self.vls = len(vlds) * [ VisibleLayer() ]

        for i in range(len(vlds)):
            vld = self.vlds[i]
            vl = self.vls[i]

            num_visible_columns = vld.size[0] * vld.size[1]
            num_visible_cells = num_visible_columns * vld.size[2]

            diam = vld.radius * 2 + 1
            area = diam * diam
            num_weights = num_hidden_cells * area * vld.size[2]

            vl.weights = cl.clrandom.rand(cq, (num_weights,), np.float32)
            vl.reconstruction = cl.array.zeros(cq, (num_visible_cells,), np.float32)

        # Kernels
        self.accum_activation_kernel = prog.encoder_accum_activation
        self.inhibit_activations_kernel = prog.inhibit_activations
        self.encoder_learn_kernel = prog.encoder_learn

        # Hyperparameters
        self.lr = 0.1

    def step(self, cq: cl.CommandQueue, visible_states: [ cl.array.Array ], learn_enabled: bool = True):
        # Clear
        self.activations.fill(np.float32(0))

        # Accumulate for all visible layers
        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            self.accum_activation_kernel(cq, (self.hidden_size[0], self.hidden_size[1]), None,
                    visible_states[i], vl.weights, self.activations,
                    np.array(vld.size, dtype=np.int32), np.array(self.hidden_size, dtype=np.int32), np.int32(vld.radius), np.int32(vld.radius * 2 + 1),
                    np.array([ vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1] ], dtype=np.float32))

        self.inhibit_activations_kernel(cq, (self.hidden_size[0], self.hidden_size[1]), None, self.activations, self.hidden_states,
                np.array(self.hidden_size, dtype=np.float32))

        if learn_enabled:
            for i in range(len(self.vls)):
                vld = self.vlds[i]
                vl = self.vls[i]

                diam = vld.radius * 2 + 1

                self.encoder_learn_kernel(cq, (vld.size[0], vld.size[1]), None,
                        visible_states[i], self.hidden_states, vl.weights, vl.reconstruction,
                        np.array(vld.size, dtype=np.int32), np.array(self.hidden_size, dtype=np.int32), np.int32(vld.radius),
                        np.array([ ceil(diam * self.hidden_size[0] / vld.size[0] * 0.5), ceil(diam * self.hidden_size[1] / vld.size[1] * 0.5) ], np.int32),
                        np.int32(diam),
                        np.array([ vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1] ], dtype=np.float32),
                        np.array([ self.hidden_size[0] / vld.size[0], self.hidden_size[1] / vld.size[1] ], dtype=np.float32),
                        np.float32(self.lr))

