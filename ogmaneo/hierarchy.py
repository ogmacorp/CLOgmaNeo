# ----------------------------------------------------------------------------
#  CLOgmaNeo
#  Copyright(c) 2023 Ogma Intelligent Systems Corp. All rights reserved.
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
        size: (int, int, int) = (4, 4, 16)
        t: IOType = IOType.PREDICTION
        up_radius: int = 2
        down_radius: int = 2

    @dataclass
    class LayerDesc:
        hidden_size: (int, int, int) = (4, 4, 16)
        up_radius: int = 2
        down_radius: int = 2
        ticks_per_update: int = 2
        temporal_horizon: int = 2

    def __init__(self, cq: cl.CommandQueue, prog: cl.Program, io_descs: [ IODesc ] = [], lds: [ LayerDesc ] = [], fd: io.IOBase = None):
        if fd is None:
            self.io_descs = io_descs
            self.lds = lds

            self.encoders = []
            self.decoders = []
            self.histories = []
            self.complete_states = []

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
                        e_vlds.append(Encoder.VisibleLayerDesc(size=(io_descs[j].size[0], io_descs[j].size[1], io_descs[j].size[2], lds[i].temporal_horizon), radius=io_descs[j].up_radius, importance=1.0))

                        io_history.append(cl.array.zeros(cq, (num_io_columns * lds[i].temporal_horizon,), np.int32))

                        if io_descs[j].t == IOType.PREDICTION:
                            d_vld = Decoder.VisibleLayerDesc(size=(lds[i].hidden_size[0], lds[i].hidden_size[1], lds[i].hidden_size[2], 2 if i < len(lds) - 1 else 1), radius=io_descs[j].down_radius)

                            io_decoders.append(Decoder(cq, prog, (io_descs[j].size[0], io_descs[j].size[1], io_descs[j].size[2], 1), [ d_vld ]))
                        else:
                            io_decoders.append(None) # Mark no decoder

                    self.decoders.append(io_decoders)

                else: # Higher layers
                    temporal_history = []

                    num_prev_columns = lds[i - 1].hidden_size[0] * lds[i - 1].hidden_size[1]

                    io_history.append(cl.array.zeros(cq, (num_prev_columns * lds[i].temporal_horizon,), np.int32))

                    e_vlds = [ Encoder.VisibleLayerDesc(size=(lds[i - 1].hidden_size[0], lds[i - 1].hidden_size[1], lds[i - 1].hidden_size[2], lds[i].temporal_horizon), radius=lds[i].up_radius, importance=1.0) ]

                    d_vld = Decoder.VisibleLayerDesc(size=(lds[i].hidden_size[0], lds[i].hidden_size[1], lds[i].hidden_size[2], 2 if i < len(lds) - 1 else 1), radius=lds[i].down_radius)

                    temporal_decoders = [ Decoder(cq, prog, (lds[i - 1].hidden_size[0], lds[i - 1].hidden_size[1], lds[i - 1].hidden_size[2], lds[i].ticks_per_update), [ d_vld ]) ]

                    self.decoders.append(temporal_decoders)

                self.encoders.append(Encoder(cq, prog, lds[i].hidden_size, e_vlds))

                self.histories.append(io_history)

                if i < len(lds) - 1:
                    self.complete_states.append(cl.array.empty(cq, (lds[i].hidden_size[0] * lds[i].hidden_size[1] * 2,), dtype=np.int32))

            self.ticks = len(lds) * [ 0 ]
            self.ticks_per_update = [ lds[i].ticks_per_update for i in range(len(lds)) ]
            self.ticks_per_update[0] = 1 # First layer always 1

            self.history_pos = len(lds) * [ 0 ]

            self.updates = len(lds) * [ False ]

        else: # Load from h5py group
            num_io, num_layers = struct.unpack("ii", fd.read(2 * np.dtype(np.int32).itemsize))

            self.io_descs = []
            self.lds = []

            self.encoders = []
            self.decoders = []
            self.histories = []
            self.complete_states = []

            # IO descs
            for i in range(num_io):
                io_desc = self.IODesc()

                io_desc.size = struct.unpack("iii", fd.read(3 * np.dtype(np.int32).itemsize))
                io_desc.t, io_desc.up_radius, io_desc.down_radius = struct.unpack("iii", fd.read(3 * np.dtype(np.int32).itemsize))

                self.io_descs.append(io_desc)

            # Layer descs
            for i in range(num_layers):
                ld = self.LayerDesc()
                
                ld.hidden_size = struct.unpack("iii", fd.read(3 * np.dtype(np.int32).itemsize))
                ld.up_radius, ld.down_radius, ld.ticks_per_update, ld.temporal_horizon = struct.unpack("iiii", fd.read(4 * np.dtype(np.int32).itemsize))

                self.lds.append(ld)

            # Create layers
            for i in range(len(self.lds)):
                io_history = []

                if i == 0: # First layer
                    io_decoders = []

                    # For each IO layer
                    for j in range(len(self.io_descs)):
                        num_io_columns = self.io_descs[j].size[0] * self.io_descs[j].size[1]

                        io_history.append(cl.array.empty(cq, (num_io_columns * self.lds[i].temporal_horizon,), np.int32))

                        read_into_buffer(fd, io_history[-1])

                    for j in range(len(self.io_descs)):
                        if self.io_descs[j].t == IOType.PREDICTION:
                            io_decoders.append(Decoder(cq, prog, fd=fd))
                        else:
                            io_decoders.append(None) # Mark no decoder

                    self.decoders.append(io_decoders)

                else: # Higher layers
                    num_prev_columns = self.lds[i - 1].hidden_size[0] * self.lds[i - 1].hidden_size[1]

                    io_history.append(cl.array.empty(cq, (num_prev_columns * self.lds[i].temporal_horizon,), np.int32))

                    read_into_buffer(fd, io_history[-1])

                    temporal_decoders = [ Decoder(cq, prog, fd=fd) ]

                    self.decoders.append(temporal_decoders)

                self.encoders.append(Encoder(cq, prog, fd=fd))

                self.histories.append(io_history)

                if i < len(self.lds) - 1:
                    self.complete_states.append(cl.array.empty(cq, (self.lds[i].hidden_size[0] * self.lds[i].hidden_size[1] * 2,), dtype=np.int32))

            self.ticks = read_array(fd, num_layers, np.int32).tolist()
            self.ticks_per_update = read_array(fd, num_layers, np.int32).tolist()

            self.history_pos = read_array(fd, num_layers, np.int32).tolist()
            self.updates = list(map(bool, read_array(fd, num_layers, np.uint8).tolist()))

        self.assign_slice_kernels = []
        self.assign_slice_caches = []

        self.stack_slices_kernels = []
        self.stack_slices_caches = []

        for i in range(len(self.lds)):
            self.assign_slice_kernels.append(prog.assign_slice.clone())
            self.stack_slices_kernels.append(prog.stack_slices.clone())

            self.assign_slice_caches.append(KernelArgCache(self.assign_slice_kernels[-1]))
            self.stack_slices_caches.append(KernelArgCache(self.stack_slices_kernels[-1]))

    def step(self, cq: cl.CommandQueue, input_states: [ cl.array.Array ], learn_enabled: bool = True):
        # Push front
        self.history_pos[0] -= 1

        if self.history_pos[0] < 0:
            self.history_pos[0] += self.lds[0].temporal_horizon

        # Push into first layer history
        for i in range(len(self.io_descs)):
            num_visible_columns = self.io_descs[i].size[0] * self.io_descs[i].size[1]

            self.assign_slice_caches[0].set_args(input_states[i].data, self.histories[0][i].data, np.int32(num_visible_columns * self.history_pos[0]))

            cl.enqueue_nd_range_kernel(cq, self.assign_slice_kernels[0], (num_visible_columns,), None)

        # Up-pass
        for i in range(len(self.encoders)):
            self.updates[i] = False

            if i == 0 or self.ticks[i] >= self.ticks_per_update[i]:
                self.ticks[i] = 0

                self.updates[i] = True

                self.encoders[i].step(cq, self.histories[i], self.history_pos[i], learn_enabled)

                # If there is a higher layer
                if i < len(self.lds) - 1:
                    i_next = i + 1

                    # Push front
                    self.history_pos[i_next] -= 1

                    if self.history_pos[i_next] < 0:
                        self.history_pos[i_next] += self.lds[i_next].temporal_horizon

                    num_visible_columns = self.lds[i].hidden_size[0] * self.lds[i].hidden_size[1]

                    self.assign_slice_caches[i_next].set_args(self.encoders[i].hidden_states.data, self.histories[i_next][0].data, np.int32(num_visible_columns * self.history_pos[i_next]))

                    cl.enqueue_nd_range_kernel(cq, self.assign_slice_kernels[i_next], (num_visible_columns,), None)

                    self.ticks[i_next] += 1

        # Down-pass
        for i in range(len(self.decoders) - 1, -1, -1):
            if self.updates[i]:
                # Copy
                decoder_visible_states = []

                if i < len(self.lds) - 1:
                    i_next = i + 1

                    num_hidden_columns = self.lds[i].hidden_size[0] * self.lds[i].hidden_size[1]
                    destride_index = self.ticks_per_update[i_next] - 1 - self.ticks[i_next]

                    self.stack_slices_caches[i].set_args(self.encoders[i].hidden_states.data, self.decoders[i_next][0].hidden_states.data, self.complete_states[i].data, np.int32(num_hidden_columns), np.int32(num_hidden_columns * destride_index))

                    cl.enqueue_nd_range_kernel(cq, self.stack_slices_kernels[i], (num_hidden_columns,), None)

                    decoder_visible_states = [ self.complete_states[i] ]
                else:
                    decoder_visible_states = [ self.encoders[i].hidden_states ]

                if i == 0:
                    for j in range(len(self.io_descs)):
                        if self.decoders[i][j] is None:
                            continue

                        self.decoders[i][j].step(cq, decoder_visible_states, input_states[j], 0, 0, 0, 1, learn_enabled)
                else:
                    self.decoders[i][0].step(cq, decoder_visible_states, self.histories[i][0], 0, 0, self.history_pos[i], self.lds[i].temporal_horizon, learn_enabled)

    def get_predicted_states(self, i: int) -> cl.array.Array:
        assert self.decoders[0][i] is not None

        return self.decoders[0][i].hidden_states

    def get_predicted_activations(self, i: int) -> cl.array.Array:
        assert self.decoders[0][i] is not None

        return self.decoders[0][i].activations

    def sample_prediction(self, i: int, temperature: float = 1.0) -> np.array:
        assert self.decoders[0][i] is not None

        if temperature == 0.0:
            return get_predicted_states(i).get()

        size = self.decoders[0][i].hidden_size

        activations = self.decoders[0][i].activations.get().reshape((size[0] * size[1], size[2]), order='F')

        if temperature != 1.0:
            # Curve
            activations = np.power(activations, 1.0 / temperature)

            totals = np.sum(activations, axis=1, keepdims=True)

            activations = np.divide(activations, np.repeat(totals, repeats=activations.shape[1], axis=1))

        states = (activations.cumsum(1) > np.random.rand(activations.shape[0])[:,None]).argmax(1)

        return states.ravel()

    def write(self, fd: io.IOBase):
        fd.write(struct.pack("ii", len(self.io_descs), len(self.lds)))

        for io_desc in self.io_descs:
            fd.write(struct.pack("iiiiii", *io_desc.size, io_desc.t, io_desc.up_radius, io_desc.down_radius))

        for ld in self.lds:
            fd.write(struct.pack("iiiiiii", *ld.hidden_size, ld.up_radius, ld.down_radius, ld.ticks_per_update, ld.temporal_horizon))

        for i in range(len(self.lds)):
            if i == 0:
                for j in range(len(self.histories[i])):
                    write_from_buffer(fd, self.histories[i][j])

                for j in range(len(self.decoders[i])):
                    if self.decoders[i][j] is not None:
                        self.decoders[i][j].write(fd)
            else:
                write_from_buffer(fd, self.histories[i][0])

                self.decoders[i][0].write(fd)

            self.encoders[i].write(fd)
            
        write_array(fd, np.array(self.ticks, dtype=np.int32))
        write_array(fd, np.array(self.ticks_per_update, dtype=np.int32))
        write_array(fd, np.array(self.history_pos, dtype=np.int32))
        write_array(fd, np.array(self.updates, dtype=np.uint8))

    def set_input_importance(self, i: int, importance: float):
        self.encoders[0].vlds[i].importance = importance

    def get_input_importance(self, i: int):
        return self.encoders[0].vlds[i].importance
