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
