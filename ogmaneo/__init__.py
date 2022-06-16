import pyopencl as cl
import importlib_resources as res

def load_prog(ctx: cl.Context, f: str = 'core.cl'):
    kernels_src = ''

    with res.open_text('kernels', 'core.cl') as f:
        kernels_src = f.read()

    prog = cl.Program(ctx, kernels_src).build()

    return prog

def load_prog_extra(ctx: cl.Context):
    return load_prog(ctx, 'extra.cl')
