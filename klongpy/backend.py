import os
import warnings

use_gpu = bool(os.environ.get('USE_GPU') == '1')

if use_gpu:
    import cupy as np

    class CuPyReductionKernelWrapper:
        def __init__(self, fn, reduce_fn):
            self.fn = fn
            self.reduce_fn = reduce_fn

        def __call__(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

        def reduce(self, *args, **kwargs):
            return self.reduce_fn(*args, **kwargs)

    np.add = CuPyReductionKernelWrapper(np.add, np.ReductionKernel(
                'T x',
                'T y',
                'x',
                'a + b',
                'y = a',
                '0',
                'add_reduce'
             ))
    np.subtract = CuPyReductionKernelWrapper(np.subtract, np.ReductionKernel(
                'T x',
                'T y',
                'x',
                'a - b',
                'y = a',
                '0',
                'subtract_reduce'
             ))
else:
    import numpy as np
    np.seterr(divide='ignore')
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

np