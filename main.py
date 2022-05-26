import numpy as np
import pyopencl as cl

ctx = cl.create_some_context()
q = cl.CommandQueue(ctx)
