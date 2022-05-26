import numpy as np
import pyopencl as cl
import pyopencl.array
from ogmaneo.encoder import Encoder
from ogmaneo.decoder import Decoder

ctx = cl.create_some_context()
cq = cl.CommandQueue(ctx)

kernels_src = ''

with open('kernels/kernels.cl', 'r') as f:
    kernels_src = f.read()

prog = cl.Program(ctx, kernels_src).build()

enc = Encoder(cq, prog, (4, 4, 16), [ Encoder.VisibleLayerDesc(size=(4, 4, 16), radius=2) ])
dec = Decoder(cq, prog, (4, 4, 16), [ Decoder.VisibleLayerDesc(size=(4, 4, 16), radius=2) ])

test_input = cl.array.Array(cq, (16,), dtype=np.int32)
test_input.set(np.array([ i % 16 for i in range(16) ], dtype=np.int32))

for i in range(100):
    enc.step(cq, [ test_input ])

    dec.step(cq, [ enc.hidden_states ], test_input)

    print(dec.hidden_states.get())

print("Done.")
