import os
import warnings

# Attempt to import CuPy. If not available, set use_gpu to False.
use_gpu = bool(os.environ.get('USE_GPU') == '1')
if use_gpu:
    try:
        import cupy as np
        use_gpu = True
    except ImportError:
        import numpy as np
        use_gpu = False
else:
    import numpy as np


def is_supported_type(x):
    """
    CuPy does not support strings or jagged arrays.
    Note: add any other unsupported types here.
    """
    if isinstance(x, str) or is_jagged_array(x):
        return False
    return True


def is_jagged_array(x):
    """
    Check if an array is jagged.
    """
    if isinstance(x, list):
        # If the lengths of sublists vary, it's a jagged array.
        return len(set(map(len, x))) > 1
    return False

if use_gpu:
    import cupy
    import numpy

    class CuPyReductionKernelWrapper:
        def __init__(self, fn, reduce_fn_1, reduce_fn_2):
            self.fn = fn
            self.reduce_fn_1 = reduce_fn_1
            self.reduce_fn_2 = reduce_fn_2

        def __call__(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

        def reduce(self, x):
            return self.reduce_fn_1(x) if len(x.shape) == 1 else self.reduce_fn_2(x[0], x[1])

    add_reduce_2 = cupy.ElementwiseKernel(
            'T x, T y',
            'T z',
            'z = (x + y)',
            'add_reduce_2')
    np.add = CuPyReductionKernelWrapper(cupy.add, cupy.sum, add_reduce_2)

    def subtract_reduce_1(x):
        return 2*x[0] - cupy.sum(x)

    subtract_reduce_2 = cupy.ElementwiseKernel(
            'T x, T y',
            'T z',
            'z = (x - y)',
            'subtract_reduce_2')
    np.subtract = CuPyReductionKernelWrapper(cupy.subtract, subtract_reduce_1, subtract_reduce_2)

    multiply_reduce_1 = cupy.ReductionKernel(
                'T x',
                'T y',
                'x',
                'a * b',
                'y = a',
                '1',
                'multiply_reduce_1'
             )
    multiply_reduce_2 = cupy.ElementwiseKernel(
            'T x, T y',
            'T z',
            'z = (x * y)',
            'multiply_reduce_2')
    np.multiply = CuPyReductionKernelWrapper(cupy.multiply, multiply_reduce_1, multiply_reduce_2)

    def divide_reduce_1(x):
        raise NotImplementedError()

    divide_reduce_2 = cupy.ElementwiseKernel(
            'T x, T y',
            'T z',
            'z = (x / y)',
            'divide_reduce_2')
    np.divide = CuPyReductionKernelWrapper(cupy.divide, divide_reduce_1, divide_reduce_2)

    np.isarray = lambda x: isinstance(x, (numpy.ndarray, cupy.ndarray))

#    np.hstack = lambda x: cupy.hstack(x) if use_gpu and is_supported_type(x) else numpy.hstack(x)
else:
    np.seterr(divide='ignore')
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    np.isarray = lambda x: isinstance(x, np.ndarray)

np
