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

        for i in range(len(lds)):
            e_vlds = []
            layer_history = []

            if i == 0: # First layer
                io_history = []
                io_decoders = []

                for j in range(len(io_descs)):
                    e_vlds.append(Encoder.VisibleLayerDesc(size=io_descs[j].size, radius=io_descs[j].e_radius))

                    num_io_columns = io_descs[j].size[0] * io_descs[j].size[1]

                    temporal_history = []

                    for k in range(lds[i].temporal_horizon):
                        temporal_history.append(cl.array.zeros(cq, (num_io_columns,), np.int32))

                    io_history.append(temporal_history)

                    if io_descs[j].t == IOType.PREDICTION:
                        d_vld = Decoder.VisibleLayerDesc(size=io_descs[j].size, radius=io_descs[j].d_radius)

                        d_vlds = [ d_vld ] if i < len(lds) else 2 * [ d_vld ] # 1 visible layer if no higher layer, otherwise 2 (additional for feed-back)

                        io_decoders.append(Decoder(cq, prog, io_descs[j].size, d_vlds))

                layer_history.append(io_history)
                
                self.decoders.append(io_decoders)

            else: # Higher layers
                temporal_history = []

                num_prev_columns = lds[i - 1].hidden_size[0] * lds[i - 1].hidden_size[1]

                for j in range(lds[i].temporal_horizon):
                    temporal_history.append(cl.array.zeros(cq, (num_prev_columns,), np.int32))

                layer_history.append([ temporal_history ])

                e_vlds = lds[i].ticks_per_update * [ Encoder.VisibleLayerDesc(size=lds[i - 1].hidden_size, radius=lds[i].e_radius) ]

            self.encoders.append(Encoder(cq, prog, lds[i].hidden_size, e_vlds))

            self.histories.append(layer_history)

        self.ticks = len(lds) * [ 0 ]
        self.ticks_per_update = [ lds[i].ticks_per_update for i in range(len(lds)) ]
        self.ticks_per_update[0] = 1 # First layer always 1

        self.io_sizes = [ io_descs[i].size for i in range(len(io_descs)) ]
        self.io_types = [ io_descs[i].t for i in range(len(io_descs)) ]
