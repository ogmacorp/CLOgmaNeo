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
        r_radius: int = 2
        d_radius: int = 2

    def __init__(self, cq: cl.CommandQueue, prog: cl.Program, io_descs: [ IODesc ] = [], lds: [ LayerDesc ] = [], grp: h5py.Group = None):
        if grp is None:
            self.io_descs = io_descs
            self.lds = lds

            self.encoders = []
            self.decoders = []

            # Create layers
            for i in range(len(lds)):
                e_vlds = []

                if i == 0: # First layer
                    io_decoders = []

                    # For each IO layer
                    for j in range(len(io_descs)):
                        num_io_columns = io_descs[j].size[0] * io_descs[j].size[1]

                        # For each timestep
                        e_vlds.append(Encoder.VisibleLayerDesc(size=(io_descs[j].size[0], io_descs[j].size[1], io_descs[j].size[2]), radius=io_descs[j].e_radius))

                        if io_descs[j].t == IOType.PREDICTION:
                            d_vld = Decoder.VisibleLayerDesc(size=(lds[i].hidden_size[0], lds[i].hidden_size[1], lds[i].hidden_size[2]), radius=io_descs[j].d_radius)

                            io_decoders.append(Decoder(cq, prog, (io_descs[j].size[0], io_descs[j].size[1], io_descs[j].size[2]), (2 if i < len(lds) - 1 else 1) * [ d_vld ]))
                        else:
                            io_decoders.append(None) # Mark no decoder

                    self.decoders.append(io_decoders)

                else: # Higher layers
                    e_vlds = [ Encoder.VisibleLayerDesc(size=(lds[i - 1].hidden_size[0], lds[i - 1].hidden_size[1], lds[i - 1].hidden_size[2]), radius=lds[i].e_radius) ]

                    d_vld = Decoder.VisibleLayerDesc(size=(lds[i].hidden_size[0], lds[i].hidden_size[1], lds[i].hidden_size[2]), radius=lds[i].d_radius)

                    self.decoders.append([ Decoder(cq, prog, (lds[i - 1].hidden_size[0], lds[i - 1].hidden_size[1], lds[i - 1].hidden_size[2]), (2 if i < len(lds) - 1 else 1) * [ d_vld ]) ])

                if lds[i].r_radius >= 0:
                    e_vlds.append(Encoder.VisibleLayerDesc(size=(lds[i].hidden_size[0], lds[i].hidden_size[1], lds[i].hidden_size[2]), radius=lds[i].r_radius))

                self.encoders.append(Encoder(cq, prog, lds[i].hidden_size, e_vlds))

        else: # Load from h5py group
            self.io_descs = pickle.loads(grp.attrs['io_descs'].tobytes())
            self.lds = pickle.loads(grp.attrs['lds'].tobytes())

            self.encoders = []
            self.decoders = []

            # Create layers
            for i in range(len(self.lds)):
                if i == 0: # First layer
                    io_decoders = []

                    # For each IO layer
                    for j in range(len(self.io_descs)):
                        if self.io_descs[j].t == IOType.PREDICTION:
                            io_decoders.append(Decoder(cq, prog, grp=grp['decoders' + str(i) + '_' + str(j)]))
                        else:
                            io_decoders.append(None) # Mark no decoder

                    self.decoders.append(io_decoders)

                else: # Higher layers
                    self.decoders.append([ Decoder(cq, prog, grp=grp['decoders' + str(i) + '_0']) ])

                self.encoders.append(Encoder(cq, prog, grp=grp['encoders' + str(i)]))

    def step(self, cq: cl.CommandQueue, input_states: [ cl.array.Array ], learn_enabled: bool = True):
        # Up-pass
        for i in range(len(self.encoders)):
            encoder_visible_states = input_states if i == 0 else [ self.encoders[i - 1].hidden_states ]

            if self.lds[i].r_radius >= 0:
                encoder_visible_states.append(self.encoders[i].hidden_states_prev)

            self.encoders[i].step(cq, encoder_visible_states, learn_enabled)

        # Down-pass
        for i in range(len(self.decoders) - 1, -1, -1):
            decoder_visible_states = []

            if i < len(self.lds) - 1:
                decoder_visible_states = [ self.encoders[i].hidden_states[:], self.decoders[i + 1][0].hidden_states[:] ]
            else:
                decoder_visible_states = [ self.encoders[i].hidden_states ]

            if i == 0:
                for j in range(len(self.io_descs)):
                    if self.decoders[i][j] is None:
                        continue

                    self.decoders[i][j].step(cq, decoder_visible_states, input_states[j], learn_enabled)
            else:
                self.decoders[i][0].step(cq, decoder_visible_states, self.encoders[i - 1].hidden_states, learn_enabled)

    def get_predicted_states(self, i) -> cl.array.Array:
        assert(self.decoders[0][i] is not None)

        return self.decoders[0][i].hidden_states

    def write(self, grp: h5py.Group):
        grp.attrs['io_descs'] = np.void(pickle.dumps(self.io_descs))
        grp.attrs['lds'] = np.void(pickle.dumps(self.lds))

        for i in range(len(self.lds)):
            for j in range(len(self.decoders[i])):
                grp.create_group('decoders' + str(i) + '_' + str(j))
                self.decoders[i][j].write(grp['decoders' + str(i) + '_' + str(j)])

            grp.create_group('encoders' + str(i))
            self.encoders[i].write(grp['encoders' + str(i)])
