import pyopencl as cl
import importlib_resources as res

def load_prog(ctx: cl.Context, kernel_file: str = 'core.cl'):
    kernels_src = ''

    with res.open_text('ogmaneo', 'kernels/' + kernel_file) as f:
        kernels_src = f.read()

    prog = cl.Program(ctx, kernels_src).build()

    return prog

def load_prog_extra(ctx: cl.Context):
    return load_prog(ctx, 'extra.cl')
