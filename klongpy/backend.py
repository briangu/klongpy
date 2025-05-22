import os
import warnings

use_torch = bool(os.environ.get('USE_TORCH') == '1')

if use_torch:
    try:
        import torch
        import numpy
        use_torch = True
    except ImportError:
        use_torch = False

if not use_torch:
    # Attempt to import CuPy. If not available, fall back to NumPy.
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
else:
    use_gpu = False


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

if use_torch:
    import numpy
    import torch

    class TorchReductionWrapper:
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

        def reduce(self, x):
            r = x[0]
            for a in x[1:]:
                r = self.fn(r, a)
            return r

        def accumulate(self, x):
            res = []
            r = x[0]
            res.append(r)
            for a in x[1:]:
                r = self.fn(r, a)
                res.append(r)
            return torch.stack(res)

    class TorchModule:
        ndarray = torch.Tensor
        integer = numpy.integer
        floating = numpy.floating
        dtype = numpy.dtype
        inf = numpy.inf

        def __init__(self):
            self.add = TorchReductionWrapper(torch.add)
            self.subtract = TorchReductionWrapper(torch.subtract)
            self.multiply = TorchReductionWrapper(torch.multiply)
            self.divide = TorchReductionWrapper(torch.divide)
            self.random = torch.random

        def asarray(self, obj, dtype=None):
            if dtype is object or isinstance(obj, str) or is_jagged_array(obj):
                return numpy.asarray(obj, dtype=dtype)
            return torch.tensor(obj, dtype=self._torch_dtype(dtype))

        array = asarray

        def isarray(self, x):
            return isinstance(x, (torch.Tensor, numpy.ndarray))

        def arange(self, *args, **kwargs):
            return torch.arange(*args, **kwargs)

        def repeat(self, a, repeats, axis=None):
            if axis is None:
                return a.repeat(repeats)
            return a.repeat_interleave(repeats, dim=axis)

        def where(self, *args, **kwargs):
            return torch.where(*args, **kwargs)

        def unique(self, *args, **kwargs):
            return torch.unique(*args, **kwargs)

        def concatenate(self, seq, axis=0):
            if all(isinstance(x, torch.Tensor) for x in seq):
                return torch.cat(seq, dim=axis)
            return numpy.concatenate([x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in seq], axis=axis)

        def _torch_dtype(self, dtype):
            if dtype is None:
                return None
            if isinstance(dtype, str):
                if dtype.startswith('int'):
                    return torch.int64
                if dtype.startswith('float'):
                    return torch.float64
            if isinstance(dtype, numpy.dtype):
                if dtype.kind in ['i', 'u']:
                    return torch.int64
                if dtype.kind == 'f':
                    return torch.float64
            return dtype

    np = TorchModule()
elif use_gpu:
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
            return self.reduce_fn_1(x) if x.ndim == 1 else self.reduce_fn_2(x[0], x[1])

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
    warnings.filterwarnings("error", category=np.VisibleDeprecationWarning)
    np.isarray = lambda x: isinstance(x, np.ndarray)
