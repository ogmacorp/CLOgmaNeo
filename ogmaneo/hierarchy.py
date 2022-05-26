import numpy as np
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
import math
from dataclasses import dataclass
from enum import Enum
from encoder import Encoder
from decoder import Decoder

class Hierarchy:
    class IOType(Enum):
        NONE = 0
        PREDICTION = 1

    @dataclass
    class IODesc:
        size: (int, int, int) = (4, 4, 16)
        t: IOType = IOType.PREDICTION
        e_radius: int = 2
        d_radius: int = 2

    @dataclass
    class LayerDesc:
        hidden_size: (int, int, int) = (4, 4, 16)
        e_radius: int = 2
        d_radius: int = 2
        ticks_per_update: int = 2
        temporal_horizon: int = 2

    def __init__(self, cq: cl.CommandQueue, prog: cl.Program, io_descs: [ IODesc ], lds: [ LayerDesc ]):
        self.encoders = []
        self.decoders = []
        self.histories = []

        # Create layers
        for i in range(len(lds)):
            e_vlds = []
            layer_history = []

            if i == 0: # First layer
                io_history = []
                io_decoders = []

                # For each IO layer
                for j in range(len(io_descs)):
                    e_vlds.append(Encoder.VisibleLayerDesc(size=io_descs[j].size, radius=io_descs[j].e_radius))

                    num_io_columns = io_descs[j].size[0] * io_descs[j].size[1]

                    temporal_history = []

                    # For each timestep
                    for k in range(lds[i].temporal_horizon):
                        temporal_history.append(cl.array.zeros(cq, (num_io_columns,), np.int32))

                    io_history.append(temporal_history)

                    if io_descs[j].t == IOType.PREDICTION:
                        d_vld = Decoder.VisibleLayerDesc(size=io_descs[j].size, radius=io_descs[j].d_radius)

                        d_vlds = [ d_vld ] if i < len(lds) - 1 else 2 * [ d_vld ] # 1 visible layer if no higher layer, otherwise 2 (additional for feed-back)

                        io_decoders.append(Decoder(cq, prog, io_descs[j].size, d_vlds))
                    else:
                        io_decoders.append(None) # Mark no decoder

                layer_history.append(io_history)
                
                self.decoders.append(io_decoders)

            else: # Higher layers
                temporal_history = []

                num_prev_columns = lds[i - 1].hidden_size[0] * lds[i - 1].hidden_size[1]

                # For each timestep
                for j in range(lds[i].temporal_horizon):
                    temporal_history.append(cl.array.zeros(cq, (num_prev_columns,), np.int32))

                layer_history.append([ temporal_history ])

                e_vlds = lds[i].ticks_per_update * [ Encoder.VisibleLayerDesc(size=lds[i - 1].hidden_size, radius=lds[i].e_radius) ]

            self.encoders.append(Encoder(cq, prog, lds[i].hidden_size, e_vlds))

            self.histories.append(layer_history)

        self.ticks = len(lds) * [ 0 ]
        self.ticks_per_update = [ lds[i].ticks_per_update for i in range(len(lds)) ]
        self.ticks_per_update[0] = 1 # First layer always 1

        self.updates = len(lds) * [ False ]

        self.io_sizes = [ io_descs[i].size for i in range(len(io_descs)) ]
        self.io_types = [ io_descs[i].t for i in range(len(io_descs)) ]

    def step(self, cq: cl.CommandQueue, input_states: [ cl.array.Array ], learn_enabled: bool = True):
        # Push into first layer history
        for i in range(len(self.io_sizes)):
            back = self.histories[0][i][-1]

            self.histories[0][i].pop()

            cl.enqueue_copy(cq, back.data, input_states[i].data)

            self.histories[0][i].insert(0, back)

        # Up-pass
        for i in range(len(self.encoders)):
            self.updates[i] = False

            if i == 0 or self.ticks[i] >= self.ticks_per_update[i]:
                self.ticks[i] = 0

                self.updates[i] = True

                encoder_visible_states = []

                if i == 0:
                    for j in range(len(self.io_sizes)):
                        encoder_visible_states += self.histories[i][j]
                else:
                    encoder_visible_states = self.histories[i][0]

                self.encoders[i].step(cq, encoder_visible_states, learn_enabled)

                # If there is a higher layer
                if i < len(self.encoders) - 1:
                    i_next = i + 1

                    back = self.histories[i_next][0][-1]

                    self.histories[i_next][0].pop()

                    cl.enqueue_copy(cq, back.data, self.encoders[i].hidden_states.data)

                    self.histories[i_next][0].index(0, back)

                    self.ticks[i_next] += 1

        # Down-pass
        for i in range(len(self.encoders) - 1, -1, -1):
            if self.updates[i]:
                decoder_visible_states = [ self.encoders[i].hidden_states ]

                if i < len(self.encoders) - 1:
                    decoder_visible_states.append(self.decoders[i + 1][self.ticks_per_update[i + 1] - 1 - self.ticks[i + 1]].hidden_states)

                if i == 0:
                    for j in range(len(self.io_sizes)):
                        self.decoders[i][j].step(cq, decoder_visible_states, input_states[j], learn_enabled)
                else:
                    for j in range(self.ticks_per_update[i]):
                        self.decoders[i][j].step(cq, decoder_visible_states, self.histories[i][0][j], learn_enabled)

    def get_predicted_states(self, i) -> cl.array.Array:
        assert(self.decoders[0][i] is not None)

        return self.decoders[0][i].hidden_states

