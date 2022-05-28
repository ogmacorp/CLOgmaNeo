import numpy as np
import pyopencl as cl
import pyopencl.array
from ogmaneo.encoder import Encoder
from ogmaneo.decoder import Decoder
from ogmaneo.hierarchy import Hierarchy, IOType

ctx = cl.create_some_context()
cq = cl.CommandQueue(ctx)

kernels_src = ''

with open('kernels/core.cl', 'r') as f:
    kernels_src = f.read()

prog = cl.Program(ctx, kernels_src).build()

test_input = cl.array.Array(cq, (1,), dtype=np.int32)

h = Hierarchy(cq, prog, [ Hierarchy.IODesc((1, 1, 16), IOType.PREDICTION) ], 3 * [ Hierarchy.LayerDesc((4, 4, 16)) ])

for t in range(1000):
    index = t % 16

    test_input.set(np.array([ index ], dtype=np.int32))

    h.step(cq, [ test_input ])

    print("Current: " + str(index) + " Prediction: " + str(h.get_predicted_states(0).get()[0]))

print("Done.")
