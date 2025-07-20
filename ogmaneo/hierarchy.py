# ----------------------------------------------------------------------------
#  CLOgmaNeo
#  Copyright(c) 2023-2025 Ogma Intelligent Systems Corp. All rights reserved.
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
from enum import IntEnum
from .encoder import Encoder
from .decoder import Decoder
import io
import struct
from .helpers import *

class IOType(IntEnum):
    NONE = 0
    PREDICTION = 1

class Hierarchy:
    @dataclass
    class IODesc:
        size: (int, int, int) = (5, 5, 16)
        t: IOType = IOType.PREDICTION
        num_dendrites_per_cell: int = 4
        up_radius: int = 2
        down_radius: int = 2

    @dataclass
    class LayerDesc:
        hidden_size: (int, int, int) = (5, 5, 16)
        num_dendrites_per_cell: int = 4
        up_radius: int = 2
        recurrent_radius: int = 0
        down_radius: int = 2

    def __init__(self, cq: cl.CommandQueue, prog: cl.Program, io_descs: [IODesc] = [], lds: [LayerDesc] = [], fd: io.IOBase = None):
        if fd is None:
            self.io_descs = io_descs
            self.lds = lds

            self.encoders = []
            self.decoders = []
            self.hidden_states_prev = []
            self.feedback_states_prev = []

            # Create layers
            for i in range(len(lds)):
                e_vlds = []

                if i == 0: # First layer
                    io_decoders = []

                    # For each IO layer
                    for j in range(len(io_descs)):
                        num_io_columns = io_descs[j].size[0] * io_descs[j].size[1]

                        # For each timestep
                        e_vlds.append(Encoder.VisibleLayerDesc(size=io_descs[j].size, radius=io_descs[j].up_radius, importance=1.0))

                        if io_descs[j].t == IOType.PREDICTION:
                            d_vld = Decoder.VisibleLayerDesc(size=lds[i].hidden_size, radius=io_descs[j].down_radius)

                            io_decoders.append(Decoder(cq, prog, io_descs[j].size, io_descs[j].num_dendrites_per_cell, (2 if i < len(lds) - 1 else 1) * [d_vld]))
                        else:
                            io_decoders.append(None) # Mark no decoder

                    self.decoders.append(io_decoders)

                else: # Higher layers
                    e_vlds = [Encoder.VisibleLayerDesc(size=lds[i - 1].hidden_size, radius=lds[i].up_radius, importance=1.0)]

                    d_vld = Decoder.VisibleLayerDesc(size=lds[i].hidden_size, radius=lds[i].down_radius)

                    self.decoders.append([Decoder(cq, prog, lds[i - 1].hidden_size, lds[i].num_dendrites_per_cell, (2 if i < len(lds) - 1 else 1) * [d_vld])])

                # Recurrence
                if lds[i].recurrent_radius >= 0:
                    e_vlds.append(Encoder.VisibleLayerDesc(size=lds[i].hidden_size, radius=lds[i].recurrent_radius, importance=0.5))

                self.encoders.append(Encoder(cq, prog, lds[i].hidden_size, e_vlds))

                self.hidden_states_prev.append(cl.array.empty(cq, (lds[i].hidden_size[0] * lds[i].hidden_size[1],), dtype=np.int32))

                if i < len(lds) - 1:
                    self.feedback_states_prev.append(cl.array.empty(cq, (lds[i].hidden_size[0] * lds[i].hidden_size[1],), dtype=np.int32))

            self.anticipation = True

        else: # Load
            num_io, num_layers = struct.unpack("ii", fd.read(2 * np.dtype(np.int32).itemsize))

            self.io_descs = []
            self.lds = []

            self.encoders = []
            self.decoders = []
            self.hidden_states_prev = []
            self.feedback_states_prev = []

            # IO descs
            for i in range(num_io):
                io_desc = self.IODesc()

                io_desc.size = struct.unpack("iii", fd.read(3 * np.dtype(np.int32).itemsize))
                io_desc.t, io_desc.num_dendrites_per_cell, io_desc.up_radius, io_desc.down_radius = struct.unpack("iiii", fd.read(4 * np.dtype(np.int32).itemsize))

                self.io_descs.append(io_desc)

            # Layer descs
            for i in range(num_layers):
                ld = self.LayerDesc()
                
                ld.hidden_size = struct.unpack("iii", fd.read(3 * np.dtype(np.int32).itemsize))
                ld.num_dendrites_per_cell, ld.up_radius, ld.recurrent_radius, ld.down_radius = struct.unpack("iiii", fd.read(4 * np.dtype(np.int32).itemsize))

                self.lds.append(ld)

            # Create layers
            for i in range(len(self.lds)):
                if i == 0: # First layer
                    io_decoders = []

                    for j in range(len(self.io_descs)):
                        if self.io_descs[j].t == IOType.PREDICTION:
                            io_decoders.append(Decoder(cq, prog, fd=fd))
                        else:
                            io_decoders.append(None) # Mark no decoder

                    self.decoders.append(io_decoders)

                else: # Higher layers
                    self.decoders.append([Decoder(cq, prog, fd=fd)])

                self.encoders.append(Encoder(cq, prog, fd=fd))

                self.hidden_states_prev.append(cl.array.empty(cq, (self.lds[i].hidden_size[0] * self.lds[i].hidden_size[1],), dtype=np.int32))

                if i < len(self.lds) - 1:
                    self.feedback_states_prev.append(cl.array.empty(cq, (self.lds[i].hidden_size[0] * self.lds[i].hidden_size[1],), dtype=np.int32))

            self.anticipation = bool(struct.unpack("B", fd.read(np.dtype(np.uint8).itemsize)))

    def step(self, cq: cl.CommandQueue, input_states: [cl.array.Array], learn_enabled: bool = True):
        # Up-pass
        for i in range(len(self.lds)):
            # Keep backup
            cl.enqueue_copy(cq, self.hidden_states_prev[i].data, self.encoders[i].hidden_states.data)

            if i < len(self.lds) - 1:
                cl.enqueue_copy(cq, self.feedback_states_prev[i].data, self.decoders[i + 1][0].hidden_states.data)

            layer_inputs = []

            if i == 0:
                layer_inputs = input_states
            else:
                layer_inputs = [self.encoders[i - 1].hidden_states]

            if self.lds[i].recurrent_radius >= 0:
                layer_inputs.append(self.hidden_states_prev[i])

            self.encoders[i].step(cq, layer_inputs, learn_enabled)

        # Down-pass
        for i in range(len(self.decoders) - 1, -1, -1):
            if learn_enabled:
                if i < len(self.lds) - 1:
                    i_next = i + 1

                    decoder_visible_states = [self.feedback_states_prev[i], self.hidden_states_prev[i]]

                    for j in range(len(self.decoders[i])):
                        if self.decoders[i][j] is None:
                            continue

                        self.decoders[i][j].learn(cq, decoder_visible_states, input_states[j] if i == 0 else self.encoders[i - 1].hidden_states)

                    if self.anticipation:
                        decoder_visible_states = [self.encoders[i].hidden_states, self.hidden_states_prev[i]]

                        for j in range(len(self.decoders[i])):
                            if self.decoders[i][j] is None:
                                continue

                            self.decoders[i][j].activate(cq, decoder_visible_states)
                            self.decoders[i][j].learn(cq, decoder_visible_states, input_states[j] if i == 0 else self.encoders[i - 1].hidden_states)
                else:
                    decoder_visible_states = [self.hidden_states_prev[i]]

                    for j in range(len(self.decoders[i])):
                        if self.decoders[i][j] is None:
                            continue

                        self.decoders[i][j].learn(cq, decoder_visible_states, input_states[j] if i == 0 else self.encoders[i - 1].hidden_states)

            decoder_visible_states = []

            if i < len(self.lds) - 1:
                decoder_visible_states = [self.decoders[i + 1][0].hidden_states, self.encoders[i].hidden_states]
            else:
                decoder_visible_states = [self.encoders[i].hidden_states]

            for j in range(len(self.decoders[i])):
                if self.decoders[i][j] is None:
                    continue

                self.decoders[i][j].activate(cq, decoder_visible_states)

    def get_predicted_states(self, i: int) -> cl.array.Array:
        assert self.decoders[0][i] is not None

        return self.decoders[0][i].hidden_states

    def get_predicted_activations(self, i: int) -> cl.array.Array:
        assert self.decoders[0][i] is not None

        return self.decoders[0][i].hidden_activations

    def sample_prediction(self, i: int, temperature: float = 1.0) -> np.array:
        assert self.decoders[0][i] is not None

        if temperature == 0.0:
            return get_predicted_states(i).get()

        size = self.decoders[0][i].hidden_size

        activations = self.decoders[0][i].hidden_activations.get().reshape((size[0] * size[1], size[2]), order='F')

        if temperature != 1.0:
            # Curve
            activations = np.power(activations, 1.0 / temperature)

            totals = np.sum(activations, axis=1, keepdims=True)

            activations = np.divide(activations, np.repeat(totals, repeats=activations.shape[1], axis=1))

        states = (activations.cumsum(1) > np.random.rand(activations.shape[0])[:, None]).argmax(1).astype(np.int32)

        return states.ravel()

    def write(self, fd: io.IOBase):
        fd.write(struct.pack("ii", len(self.io_descs), len(self.lds)))

        for io_desc in self.io_descs:
            fd.write(struct.pack("iiiiiii", *io_desc.size, io_desc.t, io_desc.num_dendrites_per_cell, io_desc.up_radius, io_desc.down_radius))

        for ld in self.lds:
            fd.write(struct.pack("iiiiiii", *ld.hidden_size, ld.num_dendrites_per_cell, ld.up_radius, ld.recurrent_radius, ld.down_radius))

        for i in range(len(self.lds)):
            if i == 0:
                for j in range(len(self.decoders[i])):
                    if self.decoders[i][j] is not None:
                        self.decoders[i][j].write(fd)
            else:
                self.decoders[i][0].write(fd)

            self.encoders[i].write(fd)
            
        fd.write(struct.pack("B", self.anticipation))

    def set_input_importance(self, i: int, importance: float):
        self.encoders[0].vlds[i].importance = importance

    def get_input_importance(self, i: int):
        return self.encoders[0].vlds[i].importance

    def set_recurrent_importance(self, i: int, importance: float):
        assert self.lds[i].recurrent_radius >= 0

        self.encoders[i].vlds[-1].importance = importance

    def get_recurrent_importance(self, i: int):
        assert self.lds[i].recurrent_radius >= 0

        return self.encoders[i].vlds[-1].importance

    def clear_state(self):
        for i in range(len(self.lds)):
            self.encoders[i].clear_state()

            for j in range(len(self.decoders[i])):
                if self.decoders[i][j] is not None:
                    self.decoders[i][j].clear_state()

            self.hidden_states_prev[i].fill(np.int32(0))

            if i < len(self.lds) - 1:
                self.feedback_states_prev[i].fill(np.int32(0))
