# ----------------------------------------------------------------------------
#  CLOgmaNeo
#  Copyright(c) 2023 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of CLOgmaNeo is licensed to you under the terms described
#  in the CLOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

import numpy as np
import io
import pyopencl as cl
import pyopencl.array

# Serialization/deserialization
def read_array(fd: io.IOBase, count: int, dtype: np.dtype):
    return np.frombuffer(fd.read(count * np.dtype(dtype).itemsize), dtype)

def read_into_buffer(fd: io.IOBase, buffer: cl.array.Array):
    buffer.set(read_array(fd, len(buffer), buffer.dtype))

def write_array(fd: io.IOBase, arr: np.array):
    fd.write(arr.tobytes())
    
def write_from_buffer(fd: io.IOBase, buffer: cl.array.Array):
    write_array(fd, buffer.get())

class KernelArgCache:
    def __init__(self, kernel: cl.Kernel):
        self.kernel = kernel

        self.num_args = kernel.get_info(cl.kernel_info.NUM_ARGS)

        self.args_prev = self.num_args * [None]
        self.args = self.num_args * [None]

    def set_args(self, *args):
        assert len(args) == self.num_args

        self.args = list(args).copy()

        self.update_args()

    def update_args(self):
        for i in range(self.num_args):
            arg_set = False

            if self.args[i] is None: # Skip
                self.args_prev[i] = self.args[i]
            elif self.args_prev[i] is None:
                arg_set = True
                self.args_prev[i] = self.args[i]
            elif isinstance(self.args[i], cl.MemoryObject):
                if self.args[i] is not self.args_prev[i]:
                    arg_set = True
                    self.args_prev[i] = self.args[i]
            elif isinstance(self.args[i], np.ndarray):
                if not (self.args[i] == self.args_prev[i]).all():
                    arg_set = True
                    self.args_prev[i] = self.args[i]
            else:
                if self.args[i] != self.args_prev[i]:
                    arg_set = True
                    self.args_prev[i] = self.args[i]

            if arg_set:
                self.kernel.set_arg(i, self.args[i])

