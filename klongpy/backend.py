import os
import warnings

use_gpu = bool(os.environ.get('USE_GPU') == '1')

if use_gpu:
    import cupy as cp
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

    add_reduce_2 = cp.ElementwiseKernel(
            'T x, T y',
            'T z',
            'z = (x + y)',
            'add_reduce_2')
    cp.add = CuPyReductionKernelWrapper(cp.add, cp.sum, add_reduce_2)

    # def subtract_reduce_1(x):
    #     return 2*x[0] - cp.sum(x)

    subtract_reduce_kernel_1d = cp.RawKernel(r'''
        extern "C" __global__
        void subtract_reduce_kernel_1d(const float* input, float* output, const int length) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < length) {
                float value = (idx == 0) ? input[idx] : -input[idx];
                atomicAdd(output, value);
            }
        }
        ''', 'subtract_reduce_kernel_1d')

    def subtract_reduce_1(arr):
        input_arr = cp.asarray(arr)
        length = input_arr.size
        output_arr = cp.zeros(1, dtype=input_arr.dtype)

        threads_per_block = 256
        blocks_per_grid = (length + threads_per_block - 1) // threads_per_block

        subtract_reduce_kernel_1d((blocks_per_grid,), (threads_per_block,), (input_arr, output_arr, length))

        return output_arr

    subtract_reduce_2 = cp.ElementwiseKernel(
            'T x, T y',
            'T z',
            'z = (x - y)',
            'subtract_reduce_2')
    np.subtract = CuPyReductionKernelWrapper(cp.subtract, subtract_reduce_1, subtract_reduce_2)

    multiply_reduce_1 = cp.ReductionKernel(
                'T x',
                'T y',
                'x',
                'a * b',
                'y = a',
                '1',
                'multiply_reduce_1'
             )
    multiply_reduce_2 = cp.ElementwiseKernel(
            'T x, T y',
            'T z',
            'z = (x * y)',
            'multiply_reduce_2')
    np.multiply = CuPyReductionKernelWrapper(cp.multiply, multiply_reduce_1, multiply_reduce_2)

    divide_reduce_1 = cp.ReductionKernel(
                'T x',
                'T y',
                'x',
                'a / b',
                'y = a',
                '1',
                'divide_reduce_1'
             )
    divide_reduce_2 = cp.ElementwiseKernel(
            'T x, T y',
            'T z',
            'z = (x / y)',
            'divide_reduce_2')
    np.divide = CuPyReductionKernelWrapper(cp.divide, divide_reduce_1, divide_reduce_2)

    np.isarray = lambda x: isinstance(x, (numpy.ndarray, cp.ndarray))
else:
    import numpy as np
    np.seterr(divide='ignore')
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    np.isarray = lambda x: isinstance(x, np.ndarray)

np
