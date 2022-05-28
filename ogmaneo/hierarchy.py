import numpy as np
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
import math
from dataclasses import dataclass
from enum import Enum
from .encoder import Encoder
from .decoder import Decoder
import h5py

class IOType(Enum):
    NONE = 0
    PREDICTION = 1

class Hierarchy:
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

    def __init__(self, cq: cl.CommandQueue, prog: cl.Program, io_descs: [ IODesc ] = [], lds: [ LayerDesc ] = [], grp: h5py.Group = None):
        if grp is None:
            self.io_descs = io_descs
            self.lds = lds

            self.encoders = []
            self.decoders = []
            self.histories = []

            # Create layers
            for i in range(len(lds)):
                e_vlds = []
                io_history = []

                if i == 0: # First layer
                    io_decoders = []

                    # For each IO layer
                    for j in range(len(io_descs)):
                        num_io_columns = io_descs[j].size[0] * io_descs[j].size[1]

                        temporal_history = []

                        # For each timestep
                        for k in range(lds[i].temporal_horizon):
                            temporal_history.append(cl.array.zeros(cq, (num_io_columns,), np.int32))

                            e_vlds.append(Encoder.VisibleLayerDesc(size=io_descs[j].size, radius=io_descs[j].e_radius))

                        io_history.append(temporal_history)

                        if io_descs[j].t == IOType.PREDICTION:
                            d_vld = Decoder.VisibleLayerDesc(size=lds[i].hidden_size, radius=io_descs[j].d_radius)

                            d_vlds = 2 * [ d_vld ] if i < len(lds) - 1 else [ d_vld ] # 1 visible layer if no higher layer, otherwise 2 (additional for feed-back)

                            io_decoders.append(Decoder(cq, prog, io_descs[j].size, d_vlds))
                        else:
                            io_decoders.append(None) # Mark no decoder

                    self.decoders.append(io_decoders)

                else: # Higher layers
                    temporal_history = []

                    num_prev_columns = lds[i - 1].hidden_size[0] * lds[i - 1].hidden_size[1]

                    # For each timestep
                    for j in range(lds[i].temporal_horizon):
                        temporal_history.append(cl.array.zeros(cq, (num_prev_columns,), np.int32))

                    io_history.append(temporal_history)

                    e_vlds = lds[i].temporal_horizon * [ Encoder.VisibleLayerDesc(size=lds[i - 1].hidden_size, radius=lds[i].e_radius) ]

                    d_vld = Decoder.VisibleLayerDesc(size=lds[i].hidden_size, radius=lds[i].d_radius)

                    d_vlds = 2 * [ d_vld ] if i < len(lds) - 1 else [ d_vld ] # 1 visible layer if no higher layer, otherwise 2 (additional for feed-back)

                    temporal_decoders = []

                    for j in range(lds[i].ticks_per_update):
                        temporal_decoders.append(Decoder(cq, prog, lds[i - 1].hidden_size, d_vlds))

                    self.decoders.append(temporal_decoders)

                self.encoders.append(Encoder(cq, prog, lds[i].hidden_size, e_vlds))

                self.histories.append(io_history)

            self.ticks = len(lds) * [ 0 ]
            self.ticks_per_update = [ lds[i].ticks_per_update for i in range(len(lds)) ]
            self.ticks_per_update[0] = 1 # First layer always 1

            self.updates = len(lds) * [ False ]

        else: # Load from h5py group
            self.io_descs = grp.attrs['io_descs']
            self.lds = grp.attrs['lds']

            self.encoders = []
            self.decoders = []
            self.histories = []

            # Create layers
            for i in range(len(lds)):
                io_history = []

                if i == 0: # First layer
                    io_decoders = []

                    # For each IO layer
                    for j in range(len(io_descs)):
                        num_io_columns = io_descs[j].size[0] * io_descs[j].size[1]

                        temporal_history = []

                        # For each timestep
                        for k in range(lds[i].temporal_horizon):
                            temporal_history.append(cl.array.empty(cq, (num_io_columns,), np.int32))
                            temporal_history[-1].set(grp['histories' + str(i) + '_' + str(j) + '_' + str(k)])

                        io_history.append(temporal_history)

                        if io_descs[j].t == IOType.PREDICTION:
                            io_decoders.append(Decoder(cq, prog, grp=grp['decoders' + str(i) + '_' + str(j)]))
                        else:
                            io_decoders.append(None) # Mark no decoder

                    self.decoders.append(io_decoders)

                else: # Higher layers
                    temporal_history = []

                    num_prev_columns = lds[i - 1].hidden_size[0] * lds[i - 1].hidden_size[1]

                    # For each timestep
                    for j in range(lds[i].temporal_horizon):
                        temporal_history.append(cl.array.empty(cq, (num_prev_columns,), np.int32))
                        temporal_history[-1].set(grp['histories' + str(i) + '_0_' + str(j)])

                    io_history.append(temporal_history)

                    d_vlds = 2 * [ d_vld ] if i < len(lds) - 1 else [ d_vld ] # 1 visible layer if no higher layer, otherwise 2 (additional for feed-back)

                    temporal_decoders = []

                    for j in range(lds[i].ticks_per_update):
                        temporal_decoders.append(Decoder(cq, prog, lds[i - 1].hidden_size, d_vlds))

                    self.decoders.append(temporal_decoders)

                self.encoders.append(Encoder(cq, prog, grp=grp['encoders' + str(i)]))

                self.histories.append(io_history)

            self.ticks = grp.attrs['ticks']
            self.ticks_per_update = grp.attrs['ticks_per_update']

            self.updates = grp.attrs['updates']

    def step(self, cq: cl.CommandQueue, input_states: [ cl.array.Array ], learn_enabled: bool = True):
        # Push into first layer history
        for i in range(len(self.io_descs)):
            back = self.histories[0][i].pop()

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
                    for j in range(len(self.io_descs)):
                        encoder_visible_states += self.histories[i][j]
                else:
                    encoder_visible_states = self.histories[i][0]

                self.encoders[i].step(cq, encoder_visible_states, learn_enabled)

                # If there is a higher layer
                if i < len(self.encoders) - 1:
                    i_next = i + 1

                    back = self.histories[i_next][0].pop()

                    cl.enqueue_copy(cq, back.data, self.encoders[i].hidden_states.data)

                    self.histories[i_next][0].insert(0, back)

                    self.ticks[i_next] += 1

        # Down-pass
        for i in range(len(self.decoders) - 1, -1, -1):
            if self.updates[i]:
                decoder_visible_states = [ self.encoders[i].hidden_states ]

                if i < len(self.encoders) - 1:
                    decoder_visible_states.append(self.decoders[i + 1][self.ticks_per_update[i + 1] - 1 - self.ticks[i + 1]].hidden_states)

                if i == 0:
                    for j in range(len(self.io_descs)):
                        if self.decoders[i][j] is None:
                            continue

                        self.decoders[i][j].step(cq, decoder_visible_states, input_states[j], learn_enabled)
                else:
                    for j in range(self.ticks_per_update[i]):
                        self.decoders[i][j].step(cq, decoder_visible_states, self.histories[i][0][j], learn_enabled)

    def get_predicted_states(self, i) -> cl.array.Array:
        assert(self.decoders[0][i] is not None)

        return self.decoders[0][i].hidden_states

    def write(self, grp: h5py.Group):
        grp.attrs['io_descs'] = self.io_descs
        grp.attrs['lds'] = self.lds

        for i in range(len(self.lds)):
            for j in range(len(self.histories[i])):
                for k in range(len(self.histories[i][j])):
                    grp.create_dataset('histories' + str(i) + '_' + str(j) + '_' + str(k), data=self.histories[i][j][k].get())

            for j in range(len(self.decoders[i])):
                self.decoders[i][j].write(grp['decoders' + str(i) + '_' + str(j)])

            self.encoders[i].write(grp['encoders' + str(i)])
            
        grp.attrs['ticks'] = self.ticks
        grp.attrs['ticks_per_update'] = self.ticks_per_update

        gro.updates['updates'] = self.updates
