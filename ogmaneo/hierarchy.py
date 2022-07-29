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
from enum import Enum
from .encoder import Encoder
from .decoder import Decoder
import h5py
import pickle

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
        temporal_horizon: int = 4

    def __init__(self, cq: cl.CommandQueue, prog: cl.Program, io_descs: [ IODesc ] = [], lds: [ LayerDesc ] = [], grp: h5py.Group = None):
        if grp is None:
            self.io_descs = io_descs
            self.lds = lds

            self.encoders = []
            self.decoders = []
            self.histories = []
            self.errors = []
            self.slices = []

            # Create layers
            for i in range(len(lds)):
                e_vlds = []
                io_history = []

                if i == 0: # First layer
                    io_decoders = []

                    # For each IO layer
                    for j in range(len(io_descs)):
                        num_io_columns = io_descs[j].size[0] * io_descs[j].size[1]

                        # For each timestep
                        e_vlds.append(Encoder.VisibleLayerDesc(size=(io_descs[j].size[0], io_descs[j].size[1], io_descs[j].size[2], lds[i].temporal_horizon), radius=io_descs[j].e_radius))

                        io_history.append(cl.array.zeros(cq, (num_io_columns * lds[i].temporal_horizon,), np.int32))

                        if io_descs[j].t == IOType.PREDICTION:
                            if i < len(lds) - 1:
                                d_vlds = [ Decoder.VisibleLayerDesc(size=lds[i].hidden_size, radius=io_descs[j].d_radius, is_dense=True),
                                        Decoder.VisibleLayerDesc(size=lds[i].hidden_size, radius=io_descs[j].d_radius, is_dense=False) ]
                            else:
                                d_vlds = [ Decoder.VisibleLayerDesc(size=lds[i].hidden_size, radius=io_descs[j].d_radius, is_dense=True) ]

                            io_decoders.append(Decoder(cq, prog, (io_descs[j].size[0], io_descs[j].size[1], io_descs[j].size[2], 1), d_vlds))
                        else:
                            io_decoders.append(None) # Mark no decoder

                    self.decoders.append(io_decoders)

                else: # Higher layers
                    temporal_history = []

                    num_prev_columns = lds[i - 1].hidden_size[0] * lds[i - 1].hidden_size[1]

                    io_history.append(cl.array.zeros(cq, (num_prev_columns * lds[i].temporal_horizon,), np.int32))

                    e_vlds = [ Encoder.VisibleLayerDesc(size=(lds[i - 1].hidden_size[0], lds[i - 1].hidden_size[1], lds[i - 1].hidden_size[2], lds[i].temporal_horizon), radius=lds[i].e_radius) ]

                    if i < len(lds) - 1:
                        d_vlds = [ Decoder.VisibleLayerDesc(size=lds[i].hidden_size, radius=lds[i].d_radius, is_dense=True),
                                Decoder.VisibleLayerDesc(size=lds[i].hidden_size, radius=lds[i].d_radius, is_dense=False) ]
                    else:
                        d_vlds = [ Decoder.VisibleLayerDesc(size=lds[i].hidden_size, radius=lds[i].d_radius, is_dense=True) ]

                    temporal_decoders = [ Decoder(cq, prog, (lds[i - 1].hidden_size[0], lds[i - 1].hidden_size[1], lds[i - 1].hidden_size[2], lds[i].ticks_per_update), d_vlds) ]

                    self.decoders.append(temporal_decoders)

                self.encoders.append(Encoder(cq, prog, lds[i].hidden_size, e_vlds))

                self.histories.append(io_history)

                self.errors.append(cl.array.empty(cq, (self.lds[i].hidden_size[0] * self.lds[i].hidden_size[1] * self.lds[i].hidden_size[2],), dtype=np.float32))
                self.slices.append(cl.array.empty(cq, (self.lds[i].hidden_size[0] * self.lds[i].hidden_size[1],), dtype=np.int32))

            self.ticks = len(lds) * [ 0 ]
            self.ticks_per_update = [ lds[i].ticks_per_update for i in range(len(lds)) ]
            self.ticks_per_update[0] = 1 # First layer always 1

            self.history_pos = len(lds) * [ 0 ]

            self.updates = len(lds) * [ False ]

        else: # Load from h5py group
            self.io_descs = pickle.loads(grp.attrs['io_descs'].tobytes())
            self.lds = pickle.loads(grp.attrs['lds'].tobytes())

            self.encoders = []
            self.decoders = []
            self.histories = []
            self.errors = []
            self.slices = []

            # Create layers
            for i in range(len(self.lds)):
                io_history = []

                if i == 0: # First layer
                    io_decoders = []

                    # For each IO layer
                    for j in range(len(self.io_descs)):
                        num_io_columns = self.io_descs[j].size[0] * self.io_descs[j].size[1]

                        # For each timestep
                        io_history.append(cl.array.empty(cq, (num_io_columns * self.lds[i].temporal_horizon,), np.int32))
                        io_history[-1].set(np.array(grp['histories' + str(i) + '_' + str(j)][:], np.int32))

                        if self.io_descs[j].t == IOType.PREDICTION:
                            io_decoders.append(Decoder(cq, prog, grp=grp['decoders' + str(i) + '_' + str(j)]))
                        else:
                            io_decoders.append(None) # Mark no decoder

                    self.decoders.append(io_decoders)

                else: # Higher layers
                    num_prev_columns = self.lds[i - 1].hidden_size[0] * self.lds[i - 1].hidden_size[1]

                    io_history.append(cl.array.empty(cq, (num_prev_columns * self.lds[i].temporal_horizon,), np.int32))
                    io_history[-1].set(np.array(grp['histories' + str(i) + '_0'][:], np.int32))

                    temporal_decoders = [ Decoder(cq, prog, grp=grp['decoders' + str(i) + '_0']) ]

                    self.decoders.append(temporal_decoders)

                self.encoders.append(Encoder(cq, prog, grp=grp['encoders' + str(i)]))

                self.histories.append(io_history)

                self.errors.append(cl.array.empty(cq, (self.lds[i].hidden_size[0] * self.lds[i].hidden_size[1] * self.lds[i].hidden_size[2],), dtype=np.float32))
                self.slices.append(cl.array.empty(cq, (self.lds[i].hidden_size[0] * self.lds[i].hidden_size[1],), dtype=np.int32))

            self.ticks = pickle.loads(grp.attrs['ticks'].tobytes())
            self.ticks_per_update = pickle.loads(grp.attrs['ticks_per_update'].tobytes())

            self.history_pos = pickle.loads(grp.attrs['history_pos'].tobytes())

            self.updates = pickle.loads(grp.attrs['updates'].tobytes())

    def step(self, cq: cl.CommandQueue, input_states: [ cl.array.Array ], learn_enabled: bool = True):
        # Push front
        self.history_pos[0] -= 1

        if self.history_pos[0] < 0:
            self.history_pos[0] += self.lds[0].temporal_horizon

        # Push into first layer history
        for i in range(len(self.io_descs)):
            num_visible_columns = self.io_descs[i].size[0] * self.io_descs[i].size[1]

            self.histories[0][i][num_visible_columns * self.history_pos[0] : num_visible_columns * (self.history_pos[0] + 1)][:] = input_states[i][:]

        # Up-pass
        for i in range(len(self.encoders)):
            self.updates[i] = False

            if i == 0 or self.ticks[i] >= self.ticks_per_update[i]:
                self.ticks[i] = 0

                self.updates[i] = True

                encoder_visible_states = self.histories[i]

                if learn_enabled:
                    # Accumulate errors
                    self.errors[i].fill(np.float32(0))

                    if i == 0:
                        for j in range(len(self.io_descs)):
                            if self.decoders[i][j] is None:
                                continue

                            self.decoders[i][j].generate_errors(cq, 0, self.errors[i], input_states[j], 0, 1)
                    else:
                        self.decoders[i][0].generate_errors(cq, 0, self.errors[i], self.histories[i][0], self.history_pos[i], self.lds[i].temporal_horizon)

                self.encoders[i].step(cq, encoder_visible_states, self.errors[i], self.history_pos[i], learn_enabled)

                # If there is a higher layer
                if i < len(self.encoders) - 1:
                    i_next = i + 1

                    # Push front
                    self.history_pos[i_next] -= 1

                    if self.history_pos[i_next] < 0:
                        self.history_pos[i_next] += self.lds[i_next].temporal_horizon

                    num_visible_columns = self.lds[i].hidden_size[0] * self.lds[i].hidden_size[1]

                    self.histories[i_next][0][num_visible_columns * self.history_pos[i_next] : num_visible_columns * (self.history_pos[i_next] + 1)][:] = self.encoders[i].hidden_states[:]

                    self.ticks[i_next] += 1

        # Down-pass
        for i in range(len(self.decoders) - 1, -1, -1):
            if self.updates[i]:
                # Copy
                decoder_visible_states = []

                if i < len(self.lds) - 1:
                    num_hidden_columns = self.lds[i].hidden_size[0] * self.lds[i].hidden_size[1]
                    destride_index = self.ticks_per_update[i + 1] - 1 - self.ticks[i + 1]

                    self.slices[i][:] = self.decoders[i + 1][0].hidden_states[num_hidden_columns * destride_index : num_hidden_columns * (destride_index + 1)][:]

                    decoder_visible_states = [ self.encoders[i].activations, self.slices[i] ]
                else:
                    decoder_visible_states = [ self.encoders[i].activations ]

                if i == 0:
                    for j in range(len(self.io_descs)):
                        if self.decoders[i][j] is None:
                            continue

                        self.decoders[i][j].step(cq, decoder_visible_states, input_states[j], 0, 1, learn_enabled)
                else:
                    self.decoders[i][0].step(cq, decoder_visible_states, self.histories[i][0], self.history_pos[i], self.lds[i].temporal_horizon, learn_enabled)

    def get_predicted_states(self, i) -> cl.array.Array:
        assert(self.decoders[0][i] is not None)

        return self.decoders[0][i].hidden_states

    def write(self, grp: h5py.Group):
        grp.attrs['io_descs'] = np.void(pickle.dumps(self.io_descs))
        grp.attrs['lds'] = np.void(pickle.dumps(self.lds))

        for i in range(len(self.lds)):
            for j in range(len(self.histories[i])):
                grp.create_dataset('histories' + str(i) + '_' + str(j), data=self.histories[i][j].get())

            for j in range(len(self.decoders[i])):
                grp.create_group('decoders' + str(i) + '_' + str(j))
                self.decoders[i][j].write(grp['decoders' + str(i) + '_' + str(j)])

            grp.create_group('encoders' + str(i))
            self.encoders[i].write(grp['encoders' + str(i)])
            
        grp.attrs['ticks'] = np.void(pickle.dumps(self.ticks))
        grp.attrs['ticks_per_update'] = np.void(pickle.dumps(self.ticks_per_update))

        grp.attrs['history_pos'] = np.void(pickle.dumps(self.history_pos))

        grp.attrs['updates'] = np.void(pickle.dumps(self.updates))
