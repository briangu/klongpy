import os
import warnings

use_gpu = bool(os.environ.get('USE_GPU') == '1')

if use_gpu:
    import cupy as np
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

    add_reduce_1 = np.ReductionKernel(
                'T x',
                'T y',
                'x',
                'a + b',
                'y = a',
                '0',
                'add_reduce_1'
             )
    add_reduce_2 = np.ElementwiseKernel(
            'T x, T y',
            'T z',
            'z = (x + y)',
            'add_reduce_2')
    np.add = CuPyReductionKernelWrapper(np.add, add_reduce_1, add_reduce_2)

    subtract_reduce_1 = np.ReductionKernel(
                'T x',
                'T y',
                'x',
                'a - b',
                'y = a',
                '0',
                'subtract_reduce_1'
             )
    subtract_reduce_2 = np.ElementwiseKernel(
            'T x, T y',
            'T z',
            'z = (x - y)',
            'subtract_reduce_2')
    np.subtract = CuPyReductionKernelWrapper(np.subtract, subtract_reduce_1, subtract_reduce_2)

    multiply_reduce_1 = np.ReductionKernel(
                'T x',
                'T y',
                'x',
                'a * b',
                'y = a',
                '0',
                'multiply_reduce_1'
             )
    multiply_reduce_2 = np.ElementwiseKernel(
            'T x, T y',
            'T z',
            'z = (x * y)',
            'multiply_reduce_2')
    np.multiply = CuPyReductionKernelWrapper(np.multiply, multiply_reduce_1, multiply_reduce_2)

    np.isarray = lambda x: isinstance(x, (numpy.ndarray, np.ndarray))
else:
    import numpy as np
    np.seterr(divide='ignore')
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    np.isarray = lambda x: isinstance(x, np.ndarray)

np
