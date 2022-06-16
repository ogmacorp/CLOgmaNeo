import pyopencl as cl
import importlib_resources as res

def load_prog(ctx: cl.Context, kernel_file: str = 'core.cl'):
    kernels_src = ''

    with res.open_text('ogmaneo', kernel_file) as f:
        kernels_src = f.read()

    return cl.Program(ctx, kernels_src)

def load_prog_extra(ctx: cl.Context):
    return load_prog(ctx, 'extra.cl')
