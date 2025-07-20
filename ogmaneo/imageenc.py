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
import io
import struct
from .helpers import *

class ImageEnc:
    @dataclass
    class VisibleLayerDesc:
        size: (int, int, int) = (32, 32, 3)
        radius: int = 4

    class VisibleLayer:
        protos: cl.array.Array
        weights: cl.array.Array
        reconstruction: cl.array.Array

    def __init__(self, cq: cl.CommandQueue, prog_extra: cl.Program, hidden_size: (int, int, int) = (5, 5, 16), vlds: [VisibleLayerDesc] = [], recon_enabled: bool = False, fd: io.IOBase = None):
        if fd is None:
            self.hidden_size = hidden_size
            self.recon_enabled = recon_enabled

            num_hidden_columns = hidden_size[0] * hidden_size[1]
            num_hidden_cells = num_hidden_columns * hidden_size[2]

            self.activations = cl.array.empty(cq, (num_hidden_cells,), np.float32)
            self.hidden_states = cl.array.zeros(cq, (num_hidden_columns,), np.int32)
            self.hidden_rates = cl.array.empty(cq, (num_hidden_cells,), np.float32)

            self.hidden_rates.fill(np.float32(0.5))

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

                vl.protos = cl.array.to_device(cq, np.random.randint(0, 256, size=num_weights, dtype=np.uint8))

                if self.recon_enabled:
                    vl.weights = cl.clrandom.rand(cq, (num_weights,), np.float32, a=0.0, b=1.0)
                    vl.reconstruction = cl.array.zeros(cq, (num_visible_cells,), np.uint8)

                self.vls.append(vl)

            # Hyperparameters
            self.lr = 0.1
            self.rr = 0.05
            self.falloff = 0.9
            self.n_radius = 1

        else: # Load from h5py group
            self.hidden_size = struct.unpack("iii", fd.read(3 * np.dtype(np.int32).itemsize))
            self.recon_enabled = bool(struct.unpack("B", fd.read(np.dtype(np.uint8).itemsize))[0])
            
            num_hidden_columns = self.hidden_size[0] * self.hidden_size[1]
            num_hidden_cells = num_hidden_columns * self.hidden_size[2]

            self.activations = cl.array.empty(cq, (num_hidden_cells,), np.float32)
            self.hidden_states = cl.array.empty(cq, (num_hidden_columns,), np.int32)
            self.hidden_rates = cl.array.empty(cq, (num_hidden_cells,), np.float32)

            read_into_buffer(fd, self.hidden_states)
            read_into_buffer(fd, self.hidden_rates)
            
            num_visible_layers = struct.unpack("i", fd.read(np.dtype(np.int32).itemsize))[0]

            self.vlds = []
            self.vls = []

            for i in range(num_visible_layers):
                vld = self.VisibleLayerDesc()
                vl = self.VisibleLayer()

                vld.size = struct.unpack("iii", fd.read(3 * np.dtype(np.int32).itemsize))
                vld.radius = struct.unpack("i", fd.read(np.dtype(np.int32).itemsize))[0]

                num_visible_columns = vld.size[0] * vld.size[1]
                num_visible_cells = num_visible_columns * vld.size[2]

                diam = vld.radius * 2 + 1
                area = diam * diam
                num_weights = num_hidden_cells * area * vld.size[2]

                vl.protos = cl.array.empty(cq, (num_weights,), np.uint8)

                read_into_buffer(fd, vl.protos)

                if self.recon_enabled:
                    vl.weights = cl.array.empty(cq, (num_weights,), np.float32)
                    vl.reconstruction = cl.array.empty(cq, (num_visible_cells,), np.uint8)

                    read_into_buffer(fd, vl.weights)
                    read_into_buffer(fd, vl.reconstruction)

                self.vlds.append(vld)
                self.vls.append(vl)

            # Hyperparameters
            self.lr, self.rr, self.falloff, self.n_radius = struct.unpack("fffi", fd.read(3 * np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize))

        # Kernels
        self.image_enc_activate_kernel = prog_extra.image_enc_activate.clone()
        self.image_enc_learn_recons_kernel = prog_extra.image_enc_learn_recons.clone()
        self.image_enc_reconstruct_kernel = prog_extra.image_enc_reconstruct.clone()
        
        self.image_enc_activate_cache = KernelArgCache(self.image_enc_activate_kernel)
        self.image_enc_learn_recons_cache = KernelArgCache(self.image_enc_learn_recons_kernel)
        self.image_enc_reconstruct_cache = KernelArgCache(self.image_enc_reconstruct_kernel)

    def step(self, cq: cl.CommandQueue, visible_states: [cl.array.Array], learn_enabled: bool = True, learn_recon: bool = True):
        assert len(visible_states) == len(self.vls)

        # Pad 3-vecs to 4-vecs
        vec_hidden_size = np.array(list(self.hidden_size) + [1], dtype=np.int32)

        # Clear
        self.activations.fill(np.float32(0))

        # Accumulate for all visible layers
        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            diam = vld.radius * 2 + 1

            # Pad 3-vecs to 4-vecs
            vec_visible_size = np.array(list(vld.size) + [1], dtype=np.int32)

            finish = bool(i == len(self.vls) - 1)
            lr = float(finish and learn_enabled) * self.lr

            self.image_enc_activate_cache.set_args(visible_states[i].data, vl.protos.data, self.activations.data, self.hidden_states.data, self.hidden_rates.data,
                    vec_visible_size, vec_hidden_size, np.int32(vld.radius), np.int32(diam),
                    np.array([vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1]], dtype=np.float32),
                    np.uint8(finish), np.float32(lr), np.float32(self.falloff), np.int32(self.n_radius))

            cl.enqueue_nd_range_kernel(cq, self.image_enc_activate_kernel, self.hidden_size, (1, 1, self.hidden_size[2]))

        if self.recon_enabled and learn_enabled and learn_recon:
            for i in range(len(self.vls)):
                vld = self.vlds[i]
                vl = self.vls[i]

                diam = vld.radius * 2 + 1

                # Pad 3-vecs to 4-vecs
                vec_visible_size = np.array(list(vld.size) + [1], dtype=np.int32)

                self.image_enc_learn_recons_cache.set_args(
                        visible_states[i].data, self.hidden_states.data, vl.weights.data, 
                        vec_visible_size, vec_hidden_size, np.int32(vld.radius),
                        np.array([math.ceil(diam * self.hidden_size[0] / vld.size[0] * 0.5), math.ceil(diam * self.hidden_size[1] / vld.size[1] * 0.5)], np.int32),
                        np.int32(diam),
                        np.array([vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1]], dtype=np.float32),
                        np.array([self.hidden_size[0] / vld.size[0], self.hidden_size[1] / vld.size[1]], dtype=np.float32),
                        np.float32(self.rr))

            cl.enqueue_nd_range_kernel(cq, self.image_enc_learn_recons_kernel, vld.size, (1, 1, vld.size[2]))

    def reconstruct(self, cq: cl.CommandQueue, hidden_states: cl.array.Array, indices: [int] = []):
        assert self.recon_enabled

        if len(indices) == 0: # Empty means all indices
            indices = [i for i in range(len(self.vls))]

        # Pad 3-vecs to 4-vecs
        vec_hidden_size = np.array(list(self.hidden_size) + [1], dtype=np.int32)

        for i in indices:
            vld = self.vlds[i]
            vl = self.vls[i]

            diam = vld.radius * 2 + 1

            # Pad 3-vecs to 4-vecs
            vec_visible_size = np.array(list(vld.size) + [1], dtype=np.int32)

            self.image_enc_reconstruct_cache.set_args(hidden_states.data, vl.weights.data, vl.reconstruction.data,
                    vec_visible_size, vec_hidden_size, np.int32(vld.radius),
                    np.array([math.ceil(diam * self.hidden_size[0] / vld.size[0] * 0.5), math.ceil(diam * self.hidden_size[1] / vld.size[1] * 0.5)], np.int32),
                    np.int32(diam),
                    np.array([vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1]], dtype=np.float32),
                    np.array([self.hidden_size[0] / vld.size[0], self.hidden_size[1] / vld.size[1]], dtype=np.float32))

            cl.enqueue_nd_range_kernel(cq, self.image_enc_reconstruct_kernel, vld.size, (1, 1, vld.size[2]))

    def write(self, fd: io.IOBase):
        fd.write(struct.pack("iiiB", *self.hidden_size, self.recon_enabled))

        write_from_buffer(fd, self.hidden_states)
        write_from_buffer(fd, self.hidden_rates)

        fd.write(struct.pack("i", len(self.vlds)))

        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            fd.write(struct.pack("iiii", *vld.size, vld.radius))

            write_from_buffer(fd, vl.protos)

            if self.recon_enabled:
                write_from_buffer(fd, vl.weights)
                write_from_buffer(fd, vl.reconstruction)

        fd.write(struct.pack("fffi", self.lr, self.rr, self.falloff, self.n_radius))
