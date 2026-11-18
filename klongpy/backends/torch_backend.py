"""
PyTorch backend provider for KlongPy.

This backend uses PyTorch tensors for array operations, enabling GPU acceleration.
It does not support object dtype or string operations.
"""
import math
import operator
import numpy
import torch
import torch.autograd.functional as torch_autograd_functional

from .base import BackendProvider, CompiledDecline, UnsupportedDtypeError, is_jagged_array
from .numpy_backend import KGChar, NumpyBackendProvider, _beyond_int64, _has_big_int
from ..autograd import AutogradChainBrokenError, NonScalarLossError, _invoke_fn
from ..errors import KlongSystemError, KlongTypeError

# numpy 2.x moved VisibleDeprecationWarning to numpy.exceptions
from numpy.exceptions import VisibleDeprecationWarning as NumpyVisibleDeprecationWarning


class TorchUnsupportedDtypeError(UnsupportedDtypeError):
    """Raised when an operation requires object dtype which is not supported by PyTorch."""
    pass


class TorchRandomModule:
    """NumPy-compatible random module using PyTorch tensors."""
    def __init__(self, backend):
        self._backend = backend

    def random(self, size=None):
        """Return random floats in the half-open interval [0.0, 1.0)."""
        if size is None:
            return torch.rand(1, device=self._backend.device).item()
        if isinstance(size, int):
            size = (size,)
        return torch.rand(*size, device=self._backend.device)

    def rand(self, *shape):
        if len(shape) == 0:
            return torch.rand(1, device=self._backend.device).item()
        return torch.rand(*shape, device=self._backend.device)

    def randn(self, *shape):
        if len(shape) == 0:
            return torch.randn(1, device=self._backend.device).item()
        return torch.randn(*shape, device=self._backend.device)

    def randint(self, low, high=None, size=None):
        if high is None:
            high = low
            low = 0
        if size is None:
            return torch.randint(low, high, (1,), device=self._backend.device).item()
        if isinstance(size, int):
            size = (size,)
        return torch.randint(low, high, size, device=self._backend.device)

    def choice(self, a, size=None, replace=True):
        if isinstance(a, int):
            a = torch.arange(a, device=self._backend.device)
        elif not isinstance(a, torch.Tensor):
            a = torch.tensor(a, device=self._backend.device)
        n = len(a)
        if size is None:
            idx = torch.randint(0, n, (1,), device=self._backend.device).item()
            return a[idx]
        if isinstance(size, int):
            size = (size,)
        total = 1
        for s in size:
            total *= s
        if replace:
            indices = torch.randint(0, n, (total,), device=self._backend.device)
        else:
            indices = torch.randperm(n, device=self._backend.device)[:total]
        return a[indices].reshape(size)

    def seed(self, seed):
        torch.manual_seed(seed)


class TorchDtype:
    """Wrapper for torch dtype providing numpy-compatible 'kind' attribute."""

    def __init__(self, torch_dtype):
        self._dtype = torch_dtype
        kind_map = {
            torch.float16: 'f',
            torch.float32: 'f',
            torch.float64: 'f',
            torch.bfloat16: 'f',
            torch.int8: 'i',
            torch.int16: 'i',
            torch.int32: 'i',
            torch.int64: 'i',
            torch.uint8: 'u',
            torch.bool: 'b',
            torch.complex64: 'c',
            torch.complex128: 'c',
        }
        self.kind = kind_map.get(torch_dtype, 'f')  # default to float

    def __eq__(self, other):
        if isinstance(other, TorchDtype):
            return self._dtype == other._dtype
        if isinstance(other, str):
            return False  # torch dtype != string like 'O'
        return self._dtype == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return repr(self._dtype)


class TorchBackend:
    """NumPy-compatible interface using PyTorch tensors for GPU acceleration."""

    def __init__(self, device=None):
        self._numpy = numpy
        self._torch = torch
        self._random = None
        self._add = None
        self._subtract = None
        self._multiply = None
        self._divide = None

        # Device priority: explicit > CUDA > MPS (Apple Silicon) > CPU
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

    @property
    def random(self):
        if self._random is None:
            self._random = TorchRandomModule(self)
        return self._random

    def __getattr__(self, name):
        # First check if torch has this attribute
        if hasattr(self._torch, name):
            attr = getattr(self._torch, name)
            if callable(attr):
                return self._wrap_torch_func(attr, name)
            return attr
        # Fall back to numpy for things torch doesn't have
        if hasattr(self._numpy, name):
            return getattr(self._numpy, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _wrap_torch_func(self, func, name):
        # Functions that require tensor inputs (not Python scalars)
        tensor_required_funcs = {
            'abs', 'trunc', 'floor', 'ceil', 'round', 'sign',
            'sin', 'cos', 'tan', 'exp', 'log', 'sqrt',
            'isinf', 'isnan', 'isfinite',
            'minimum', 'maximum', 'fmod',
            'less', 'greater', 'less_equal', 'greater_equal',
        }

        def wrapper(*args, **kwargs):
            converted_args = []
            needs_tensor = name in tensor_required_funcs
            for arg in args:
                if isinstance(arg, numpy.ndarray):
                    # Handle float64 arrays on MPS by converting to float32
                    if arg.dtype == numpy.float64 and self.device.type == 'mps':
                        arg = arg.astype(numpy.float32)
                    converted_args.append(torch.from_numpy(arg).to(self.device))
                elif isinstance(arg, list):
                    try:
                        converted_args.append(torch.tensor(arg, device=self.device))
                    except (ValueError, TypeError):
                        converted_args.append(arg)
                elif needs_tensor and isinstance(arg, (int, float)):
                    # Convert Python scalars to tensors for functions that require it
                    dtype = torch.float32 if isinstance(arg, float) else torch.int64
                    converted_args.append(torch.tensor(arg, dtype=dtype, device=self.device))
                else:
                    converted_args.append(arg)
            return func(*converted_args, **kwargs)
        return wrapper

    def asarray(self, a, dtype=None):
        """Convert input to a torch tensor.

        Note: MPS (Apple Silicon) doesn't support float64, so we convert to float32.
        Object dtypes are not supported - use numpy backend for heterogeneous data.
        """
        if dtype is not None and (dtype == object or (hasattr(dtype, 'kind') and dtype.kind == 'O')):
            raise TorchUnsupportedDtypeError(
                "PyTorch backend does not support object dtype."
            )
        if isinstance(a, torch.Tensor):
            if a.device != self.device:
                return a.to(self.device)
            return a
        if isinstance(a, numpy.ndarray):
            if a.dtype == object:
                raise TorchUnsupportedDtypeError(
                    "PyTorch backend does not support object dtype arrays."
                )
            if a.dtype == numpy.float64 and self.device.type == 'mps':
                a = a.astype(numpy.float32)
            return torch.from_numpy(a).to(self.device)
        # Check if input is a list/tuple of tensors - use stack to preserve gradients
        if isinstance(a, (list, tuple)) and len(a) > 0 and all(isinstance(x, torch.Tensor) for x in a):
            result = torch.stack(a)
            if result.device != self.device:
                result = result.to(self.device)
            if result.dtype == torch.float64 and self.device.type == 'mps':
                result = result.to(torch.float32)
            return result
        # For all other lists/tuples, convert via numpy (faster than torch.tensor for nested/mixed data)
        if isinstance(a, (list, tuple)):
            if any(isinstance(x, torch.Tensor) for x in a):
                # tensors mixed with plain numbers: stack, so the tensors stay in the autograd graph
                # (numpy.asarray would detach them, or fail on one that requires grad)
                if not all(isinstance(x, (torch.Tensor, int, float)) for x in a):
                    raise TorchUnsupportedDtypeError("PyTorch backend does not support object dtype arrays.")
                members = [x if isinstance(x, torch.Tensor) else self.asarray(x) for x in a]
                result = torch.stack([m.to(self.device) for m in members])
                if result.dtype == torch.float64 and self.device.type == 'mps':
                    result = result.to(torch.float32)
                return result
            arr = numpy.asarray(a)
            if arr.dtype == object:
                raise TorchUnsupportedDtypeError(
                    "PyTorch backend does not support object dtype arrays."
                )
            if arr.dtype.kind == 'u' or (arr.dtype.kind == 'f' and type(a) is list and arr.size
                                         and _beyond_int64(arr) and _has_big_int(a)):
                # numpy made uint64 or floats of [2**63 1]: the integers stay exact in an object array
                raise TorchUnsupportedDtypeError(
                    "PyTorch backend does not support integers beyond 64 bits in a tensor."
                )
            # Convert float64 to float32 to match torch.tensor's default behavior
            if arr.dtype == numpy.float64:
                arr = arr.astype(numpy.float32)
            return torch.from_numpy(arr).to(self.device)
        # Scalar or other type
        try:
            t = torch.tensor(a, device=self.device)
            if t.dtype == torch.float64 and self.device.type == 'mps':
                t = t.to(torch.float32)
            return t
        except (ValueError, TypeError, RuntimeError) as e:
            raise TorchUnsupportedDtypeError(
                f"PyTorch backend cannot convert this data: {e}"
            )

    def array(self, a, dtype=None):
        """Create a torch tensor. Like numpy.array this is a copy: the caller may write to it."""
        if dtype is None:
            if isinstance(a, torch.Tensor):
                r = a.clone()
                return r if r.device == self.device else r.to(self.device)
            if self._is_object_ndarray(a):
                return a.copy()
        return self.asarray(a, dtype=dtype)

    def isarray(self, x):
        """Check if x is an array (numpy or torch tensor)."""
        return isinstance(x, (numpy.ndarray, torch.Tensor))

    def zeros(self, shape, dtype=None):
        if dtype is int:
            return torch.zeros(shape, dtype=torch.int64, device=self.device)
        return torch.zeros(shape, device=self.device)

    def ones(self, shape, dtype=None):
        if dtype is int:
            return torch.ones(shape, dtype=torch.int64, device=self.device)
        return torch.ones(shape, device=self.device)

    @staticmethod
    def _is_object_ndarray(x):
        return isinstance(x, numpy.ndarray) and x.dtype.kind not in 'biuf'

    def _host(self, x):
        """x for a numpy call: tensors leave the graph and the device."""
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

    @staticmethod
    def _int_list(x):
        """A shape, repeat count or index list as Python integers."""
        if isinstance(x, (torch.Tensor, numpy.ndarray)):
            x = x.tolist()
        if isinstance(x, (list, tuple)):
            return [int(i) for i in x]
        return int(x)

    def full(self, shape, fill_value, dtype=None):
        """numpy.full: the shape may be a tensor; a tensor fill value stays in the graph."""
        shape = self._int_list(shape)
        shape = tuple(shape) if isinstance(shape, list) else (shape,)
        if isinstance(fill_value, torch.Tensor):
            return fill_value.expand(shape + tuple(fill_value.shape)).clone()
        if isinstance(fill_value, bool) or not isinstance(fill_value, (int, float)):
            # a string, symbol, dictionary, ...: not a tensor value
            r = numpy.empty(shape, dtype=object)
            r.fill(fill_value)
            return r
        t = torch.full(shape, fill_value, device=self.device)
        if t.dtype == torch.float64:
            t = t.to(torch.float32)  # the float width asarray gives a Python float
        return t

    def tile(self, a, reps):
        """numpy.tile: an integer or a list of integers as the count; object arrays stay numpy."""
        reps = self._int_list(reps)
        if not isinstance(a, torch.Tensor):
            if self._is_object_ndarray(a):
                return numpy.tile(a, reps)
            a = self.asarray(a)
        return torch.tile(a, tuple(reps) if isinstance(reps, list) else (reps,))

    def resize(self, a, new_shape):
        """numpy.resize: the elements of a in order, cycled or cut to fill new_shape."""
        new_shape = self._int_list(new_shape)
        new_shape = tuple(new_shape) if isinstance(new_shape, list) else (new_shape,)
        if not isinstance(a, torch.Tensor):
            if self._is_object_ndarray(a):
                return numpy.resize(a, new_shape)
            a = self.asarray(a)
        flat = a.reshape(-1)
        n = 1
        for d in new_shape:
            n *= d
        if n == 0 or flat.numel() == 0:
            return flat[:0].reshape(new_shape) if n == 0 else torch.zeros(new_shape, dtype=a.dtype, device=a.device)
        if n > flat.numel():
            flat = flat.repeat(-(-n // flat.numel()))
        return flat[:n].reshape(new_shape)

    def roll(self, a, shift, axis=None):
        """numpy.roll (without an axis: over the flattened elements)."""
        if not isinstance(a, torch.Tensor):
            if isinstance(a, numpy.ndarray):
                if self._is_object_ndarray(a):
                    return numpy.roll(a, self._int_list(shift), axis)
            a = self.asarray(a)
        shift = self._int_list(shift)
        return torch.roll(a, shift) if axis is None else torch.roll(a, shift, dims=axis)

    def repeat(self, a, repeats, axis=None):
        """numpy.repeat (torch calls it repeat_interleave)."""
        a = a if isinstance(a, torch.Tensor) else self.asarray(a)
        repeats = repeats if isinstance(repeats, torch.Tensor) else self.asarray(repeats)
        if axis is None:
            a = a.reshape(-1)
            return torch.repeat_interleave(a, repeats.to(torch.int64).reshape(-1) if repeats.ndim else int(repeats))
        return torch.repeat_interleave(a, repeats.to(torch.int64), dim=axis)

    def logical_not(self, a):
        if self._is_object_ndarray(a) or not isinstance(a, (torch.Tensor, numpy.ndarray, int, float, list)):
            return numpy.logical_not(a)
        return torch.logical_not(a if isinstance(a, torch.Tensor) else self.asarray(a))

    def arange(self, *args, **kwargs):
        return torch.arange(*args, device=self.device, **kwargs)

    def concatenate(self, arrays, axis=0):
        if not isinstance(arrays, torch.Tensor) and any(self._is_object_ndarray(a) for a in arrays):
            # strings and mixed lists are numpy object arrays on every backend
            return numpy.concatenate([self._host(a) for a in arrays], axis=axis)
        tensors = [self.asarray(a) for a in arrays]
        return torch.cat(tensors, dim=axis)

    def hstack(self, arrays):
        tensors = [self.asarray(a) for a in arrays]
        return torch.hstack(tensors)

    def vstack(self, arrays):
        tensors = [self.asarray(a) for a in arrays]
        return torch.vstack(tensors)

    def stack(self, arrays, axis=0):
        tensors = [self.asarray(a) for a in arrays]
        return torch.stack(tensors, dim=axis)

    @property
    def ndarray(self):
        """Return the tensor class for isinstance checks."""
        return torch.Tensor

    @property
    def integer(self):
        return numpy.integer

    @property
    def floating(self):
        return numpy.floating

    def copy(self, a):
        if isinstance(a, torch.Tensor):
            return a.clone()
        if self._is_object_ndarray(a):
            return a.copy()
        return self.asarray(a).clone()

    def isclose(self, a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        # For scalars, use numpy's isclose to avoid tensor conversion issues
        if not hasattr(a, '__len__') and not hasattr(b, '__len__'):
            return self._numpy.isclose(float(a), float(b), rtol=rtol, atol=atol, equal_nan=equal_nan)
        a_t = self.asarray(a) if not isinstance(a, torch.Tensor) else a
        b_t = self.asarray(b) if not isinstance(b, torch.Tensor) else b
        # torch.isclose requires same dtype, convert to float if needed
        if a_t.dtype != b_t.dtype:
            # Use float32 for MPS compatibility (MPS doesn't support float64)
            a_t = a_t.to(torch.float32)
            b_t = b_t.to(torch.float32)
        return torch.isclose(a_t, b_t, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def array_equal(self, a, b):
        a_t = self.asarray(a) if not isinstance(a, torch.Tensor) else a
        b_t = self.asarray(b) if not isinstance(b, torch.Tensor) else b
        return torch.equal(a_t, b_t)

    def take(self, a, indices, axis=None):
        a_t = self.asarray(a) if not isinstance(a, torch.Tensor) else a
        indices_t = self.asarray(indices) if not isinstance(indices, torch.Tensor) else indices
        if axis is None:
            return a_t.flatten()[indices_t.long()]
        return torch.index_select(a_t, axis, indices_t.long())

    def transpose(self, a, axes=None):
        """Transpose a tensor."""
        if self._is_object_ndarray(a):
            return numpy.transpose(a, axes)
        a_t = self.asarray(a) if not isinstance(a, torch.Tensor) else a
        if axes is None:
            return a_t.T if a_t.ndim >= 2 else a_t
        return a_t.permute(*axes)

    def sum(self, a, axis=None, dtype=None, out=None, keepdims=False):
        """Sum array elements."""
        a_t = self.asarray(a) if not isinstance(a, torch.Tensor) else a
        if axis is None:
            return a_t.sum()
        return a_t.sum(dim=axis, keepdim=keepdims)

    def abs(self, a):
        """Absolute value."""
        if isinstance(a, (int, float)):
            return abs(a)
        a_t = self.asarray(a) if not isinstance(a, torch.Tensor) else a
        return torch.abs(a_t)

    def _binary(self, op, numpy_ufunc, a, b):
        """op on tensors; operands a tensor cannot hold (an integer beyond int64, an
        object array) go to the numpy ufunc, which raises OverflowError for the
        integer so that vec_fn2 repeats the operator on Python integers."""
        try:
            a_t = self.asarray(a) if not isinstance(a, torch.Tensor) else a
            b_t = self.asarray(b) if not isinstance(b, torch.Tensor) else b
        except TorchUnsupportedDtypeError:
            return numpy_ufunc(self._host(a), self._host(b))
        return op(a_t, b_t)

    def minimum(self, a, b):
        """Element-wise minimum."""
        return self._binary(torch.minimum, numpy.minimum, a, b)

    def maximum(self, a, b):
        """Element-wise maximum."""
        return self._binary(torch.maximum, numpy.maximum, a, b)

    # exact on Python integers, see BackendProvider._exact_atomic
    minimum.object_safe = True
    maximum.object_safe = True

    def less(self, a, b):
        return self._binary(torch.less, numpy.less, a, b)

    def greater(self, a, b):
        return self._binary(torch.greater, numpy.greater, a, b)

    def floor(self, a):
        """Floor of input."""
        if isinstance(a, (int, float)):
            return math.floor(a)
        a_t = self.asarray(a) if not isinstance(a, torch.Tensor) else a
        return torch.floor(a_t)

    def ceil(self, a):
        """Ceiling of input."""
        if isinstance(a, (int, float)):
            return math.ceil(a)
        a_t = self.asarray(a) if not isinstance(a, torch.Tensor) else a
        return torch.ceil(a_t)

    def trunc(self, a):
        """Truncate to integer."""
        if isinstance(a, (int, float)):
            return math.trunc(a)
        a_t = self.asarray(a) if not isinstance(a, torch.Tensor) else a
        return torch.trunc(a_t)

    def isinf(self, a):
        """Check for infinity."""
        if isinstance(a, (int, float)):
            return math.isinf(a)
        a_t = self.asarray(a) if not isinstance(a, torch.Tensor) else a
        return torch.isinf(a_t)

    def isnan(self, a):
        """Check for NaN."""
        if isinstance(a, (int, float)):
            return math.isnan(a)
        a_t = self.asarray(a) if not isinstance(a, torch.Tensor) else a
        return torch.isnan(a_t)

    def negative(self, a):
        if isinstance(a, torch.Tensor):
            return torch.negative(a)
        if self._is_object_ndarray(a) or (isinstance(a, int) and not -2 ** 63 <= a < 2 ** 63):
            return numpy.negative(a)  # an integer beyond int64, or a list holding one
        return torch.negative(self.asarray(a))

    def fmod(self, a, b):
        """Truncated remainder. Integer operands with a zero divisor are computed as
        floats, so the result is the IEEE nan (torch raises ZeroDivisionError for integers)."""
        try:
            a_t = a if isinstance(a, torch.Tensor) else self.asarray(a)
            b_t = b if isinstance(b, torch.Tensor) else self.asarray(b)
        except TorchUnsupportedDtypeError:
            return numpy.fmod(self._host(a), self._host(b))  # an integer beyond int64: numpy's OverflowError
        if not (a_t.is_floating_point() or b_t.is_floating_point()) and bool((b_t == 0).any()):
            a_t = a_t.to(torch.float32)
        return torch.fmod(a_t, b_t)

    def sign(self, a):
        """Sign of elements."""
        if isinstance(a, (int, float)):
            return (a > 0) - (a < 0)
        a_t = self.asarray(a) if not isinstance(a, torch.Tensor) else a
        return torch.sign(a_t)

    class TorchUfunc:
        """Wraps torch ops to support numpy ufunc interface (reduce, accumulate).

        Falls back to numpy for object arrays since torch doesn't support them.
        """
        def __init__(self, backend, op, reduce_op, accumulate_op=None, numpy_ufunc=None):
            self._backend = backend
            self._op = op
            self._reduce_op = reduce_op
            self._accumulate_op = accumulate_op
            self._torch = torch
            self._numpy_ufunc = numpy_ufunc
            # exact on Python integers, see BackendProvider._exact_atomic
            self.object_safe = numpy_ufunc in (numpy.add, numpy.subtract, numpy.multiply)

        def _is_object_array(self, x):
            return isinstance(x, numpy.ndarray) and x.dtype == object

        def _to_numpy(self, x):
            if isinstance(x, self._torch.Tensor):
                return x.detach().cpu().numpy()
            return x

        def __call__(self, a, b):
            a_is_tensor = isinstance(a, self._torch.Tensor)
            b_is_tensor = isinstance(b, self._torch.Tensor)
            # Fast path for tensor operations
            if a_is_tensor and b_is_tensor and a.device == b.device:
                return self._op(a, b)
            if (a_is_tensor and isinstance(b, (int, float))) or \
               (b_is_tensor and isinstance(a, (int, float))):
                s = a if b_is_tensor else b
                if type(s) is int and not -9223372036854775808 <= s <= 9223372036854775807:
                    # torch would wrap the integer silently; numpy raises, and
                    # vec_fn2 then repeats the operator on Python integers
                    raise OverflowError("Python integer out of bounds for int64")
                return self._op(a, b)
            # Numpy fallback for object arrays
            if self._numpy_ufunc and (self._is_object_array(a) or self._is_object_array(b)):
                return self._numpy_ufunc(self._to_numpy(a), self._to_numpy(b))
            try:
                return self._op(self._backend.asarray(a), self._backend.asarray(b))
            except TorchUnsupportedDtypeError:
                if self._numpy_ufunc:
                    return self._numpy_ufunc(self._to_numpy(a), self._to_numpy(b))
                raise

        def reduce(self, a, axis=None):
            if self._numpy_ufunc and self._is_object_array(a):
                return self._numpy_ufunc.reduce(self._to_numpy(a), axis=axis)
            try:
                arr = self._backend.asarray(a)
                if getattr(arr, 'ndim', 1) == 0:
                    return arr
                if axis is None:
                    # match numpy's ufunc.reduce: fold over axis 0, so +/ of a
                    # 2-D array yields the column-wise vector (not a scalar) and
                    # stays autograd-safe (torch.sum(dim=0)).
                    axis = 0
                return self._reduce_op(arr, dim=axis)
            except TorchUnsupportedDtypeError:
                if self._numpy_ufunc:
                    return self._numpy_ufunc.reduce(self._to_numpy(a), axis=axis)
                raise

        def accumulate(self, a, axis=0):
            if self._numpy_ufunc and self._is_object_array(a):
                return self._numpy_ufunc.accumulate(self._to_numpy(a), axis=axis)
            try:
                arr = self._backend.asarray(a)
                if self._accumulate_op:
                    return self._accumulate_op(arr, dim=axis)
                result = [arr[0]]
                for i in range(1, len(arr)):
                    result.append(self._op(result[-1], arr[i]))
                return self._torch.stack(result)
            except TorchUnsupportedDtypeError:
                if self._numpy_ufunc:
                    return self._numpy_ufunc.accumulate(self._to_numpy(a), axis=axis)
                raise

    @property
    def add(self):
        if self._add is None:
            self._add = self.TorchUfunc(self, torch.add, torch.sum, torch.cumsum, numpy.add)
        return self._add

    @property
    def subtract(self):
        def cumulative_subtract(a, dim=0):
            result = [a[0]]
            for i in range(1, a.shape[dim]):
                result.append(result[-1] - a[i])
            return torch.stack(result)
        def reduce_subtract(a, dim=None):
            # a-b-c == a-(b+c). Fold over axis 0 (the reduce() default since the
            # autograd-safe over change) so -/ of a 2-D array stays element-wise
            # column-wise; torch.sum keeps it differentiable. (Previously this
            # returned None for any dim other than None, which broke -/ once
            # reduce() started passing dim=0.)
            rest = torch.sum(a[1:]) if dim is None else torch.sum(a[1:], dim=dim)
            return a[0] - rest
        return self.TorchUfunc(
            self, torch.subtract,
            reduce_subtract,
            cumulative_subtract,
            numpy.subtract
        )

    @property
    def multiply(self):
        if self._multiply is None:
            self._multiply = self.TorchUfunc(self, torch.multiply, torch.prod, torch.cumprod, numpy.multiply)
        return self._multiply

    @property
    def divide(self):
        def reduce_divide(a, dim=None):
            # a/b/c == a/(b*c). Same axis-0 fold as the other reduces, vectorized
            # + autograd-safe via torch.prod. (Previously returned None for any
            # dim other than None, which broke %/ once reduce() passed dim=0.)
            rest = torch.prod(a[1:]) if dim is None else torch.prod(a[1:], dim=dim)
            return a[0] / rest
        return self.TorchUfunc(self, torch.divide, reduce_divide, None, numpy.divide)

    @property
    def inf(self):
        return float('inf')

    def seterr(self, **kwargs):
        pass

    @property
    def VisibleDeprecationWarning(self):
        return NumpyVisibleDeprecationWarning


class TorchBackendProvider(BackendProvider):
    """PyTorch-based backend provider."""

    def __init__(self, device=None):
        if device is not None:
            try:
                torch_device = torch.device(device)
            except Exception as exc:
                raise ValueError(f"Invalid torch device '{device}': {exc}")
            if torch_device.type == 'cuda':
                if not torch.cuda.is_available():
                    raise ValueError(f"Torch device '{device}' is not available (cuda not available)")
                if torch_device.index is not None and torch_device.index >= torch.cuda.device_count():
                    raise ValueError(f"Torch device '{device}' is not available (device index out of range)")
            if torch_device.type == 'mps':
                if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    raise ValueError(f"Torch device '{device}' is not available (mps not available)")
            if torch_device.type not in {'cpu', 'cuda', 'mps'}:
                raise ValueError(f"Torch device type '{torch_device.type}' is not supported")
        self._torch_backend = TorchBackend(device)
        self._device = device

    @property
    def name(self) -> str:
        return 'torch'

    @property
    def np(self):
        return self._torch_backend

    @property
    def device(self):
        return self._torch_backend.device

    def list_devices(self):
        """List available torch devices (cpu, cuda, mps)."""
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
            for i in range(torch.cuda.device_count()):
                devices.append(f'cuda:{i}')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append('mps')
        return devices

    def supports_object_dtype(self) -> bool:
        return False

    def supports_strings(self) -> bool:
        return True

    def supports_exact_integers(self) -> bool:
        # an integer beyond int64 is never a tensor: it is a Python integer (or
        # sits in a numpy object array) on this backend too
        return True

    def supports_float64(self) -> bool:
        # MPS device doesn't support float64
        return 'mps' not in str(self.device).lower()

    def is_array(self, x) -> bool:
        return isinstance(x, (numpy.ndarray, torch.Tensor))

    def is_backend_array(self, x) -> bool:
        """Check if x is specifically a torch tensor (not numpy)."""
        return isinstance(x, torch.Tensor)

    def get_dtype_kind(self, arr) -> str:
        if hasattr(arr, 'dtype'):
            dtype = arr.dtype
            # numpy arrays have dtype.kind
            if hasattr(dtype, 'kind'):
                return dtype.kind
            # torch tensors need manual mapping
            kind_map = {
                torch.float16: 'f',
                torch.float32: 'f',
                torch.float64: 'f',
                torch.bfloat16: 'f',
                torch.int8: 'i',
                torch.int16: 'i',
                torch.int32: 'i',
                torch.int64: 'i',
                torch.uint8: 'u',
                torch.bool: 'b',
                torch.complex64: 'c',
                torch.complex128: 'c',
            }
            return kind_map.get(dtype, 'f')
        return None

    def to_numpy(self, x):
        """Convert torch tensor to numpy array."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def to_display(self, x):
        """A tensor as the value that is printed.

        A float32 number (a 0-d tensor, or an element of a float32 list)
        becomes the Python float with the shortest text that reads back as the
        same float32: 1.23 prints as 1.23, where the float32 value widened to
        64 bits would print as 1.2300000190734863.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            if x.ndim > 0:
                return x
        if hasattr(x, 'item') and hasattr(x, 'ndim') and x.ndim == 0:
            if x.dtype == numpy.float32:
                return float(str(x[()] if isinstance(x, numpy.ndarray) else x))
            return x.item()
        return x

    def is_scalar_integer(self, x) -> bool:
        if isinstance(x, torch.Tensor) and x.ndim == 0:
            return x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        if isinstance(x, numpy.ndarray) and x.ndim == 0:
            # an item of a mixed (object) list is a 0-d numpy array on this backend too
            return numpy.issubdtype(x.dtype, numpy.integer)
        return False

    def is_scalar_float(self, x) -> bool:
        if isinstance(x, torch.Tensor) and x.ndim == 0:
            return x.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16)
        if isinstance(x, numpy.ndarray) and x.ndim == 0:
            return numpy.issubdtype(x.dtype, numpy.floating)
        return False

    def argsort(self, a, descending=False):
        """Return indices that would sort the array.

        Ascending is stable. Descending is the reverse of the ascending order,
        as on the numpy backend, so equal elements come out last index first.
        """
        if not isinstance(a, torch.Tensor):
            a = self._torch_backend.asarray(a)
        indices = torch.argsort(a, stable=True)
        return torch.flip(indices, dims=[0]) if descending else indices

    def put(self, r, idx, v):
        """Amend: write v at the flat positions idx of r (a fresh copy, see TorchBackend.array).

        A tensor stays a tensor and stays in the autograd graph (r is a clone, so
        the write is not on a leaf); a real number written into an integer tensor
        makes it a float tensor, and a value a tensor cannot hold (a string, a
        symbol, ...) makes the list a numpy object array.
        """
        if not isinstance(r, torch.Tensor):
            if isinstance(v, torch.Tensor):
                v = v.item() if v.ndim == 0 else v.detach().cpu().numpy()
            if isinstance(r, numpy.ndarray) and r.dtype.kind in 'iub' and isinstance(v, float):
                r = r.astype(float)
            numpy.put(r, numpy.asarray(idx, dtype=int), v)
            return r
        is_number = isinstance(v, torch.Tensor) or (isinstance(v, (int, float, numpy.number)) and not isinstance(v, bool))
        if not is_number:
            o = r.detach().cpu().numpy().astype(object)
            numpy.put(o, numpy.asarray(idx, dtype=int), v)
            return o
        v_is_float = v.is_floating_point() if isinstance(v, torch.Tensor) else isinstance(v, (float, numpy.floating))
        if v_is_float and not r.is_floating_point():
            r = r.to(v.dtype if isinstance(v, torch.Tensor) else torch.float32)
        if isinstance(v, torch.Tensor) and v.dtype != r.dtype:
            v = v.to(r.dtype)  # an integer written into a float tensor (index_put wants one dtype)
        flat = r.reshape(-1)
        flat[torch.as_tensor(idx, dtype=torch.int64, device=r.device)] = v
        return r

    def amend_at(self, a, index, v):
        """Amend-in-Depth: a copy of a with the one element at the index path replaced by v."""
        index = (index.tolist() if hasattr(index, 'tolist') else list(index)) \
            if getattr(index, 'ndim', 1) > 0 else [index.item() if hasattr(index, 'item') else index]
        for i in index:
            if i != int(i):
                # 5.5,[0 1] is one float tensor: whole numbers are indices, anything else is not
                raise KlongTypeError(f":-: index must be an integer, got {i!r}")
        index = tuple(int(i) for i in index)
        is_number = isinstance(v, torch.Tensor) or (isinstance(v, (int, float, numpy.number)) and not isinstance(v, bool))
        if not is_number:
            return super().amend_at(a.detach().cpu().numpy(), index, v)
        r = a.clone()
        v_is_float = v.is_floating_point() if isinstance(v, torch.Tensor) else isinstance(v, (float, numpy.floating))
        if v_is_float and not r.is_floating_point():
            r = r.to(v.dtype if isinstance(v, torch.Tensor) else torch.float32)
        r[index] = v
        return r

    def index_at(self, a, index):
        """Index-in-Depth: the element of a at the index path."""
        if isinstance(a, torch.Tensor):
            if isinstance(index, list) or getattr(index, 'ndim', 0) > 0:
                if not isinstance(index, torch.Tensor):
                    index = self._torch_backend.asarray(index)
                # one index per axis; rows of indices select several elements, as in numpy
                return a[tuple(index.to(torch.int64))]
            return a[int(index)]
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        return super().index_at(a, index)

    def array_split(self, a, sections):
        """numpy.array_split: a number of sections, or the positions to cut before."""
        sections = TorchBackend._int_list(sections)
        if isinstance(a, torch.Tensor):
            return list(torch.tensor_split(a, sections))
        return numpy.array_split(a, sections)

    def find_equal(self, a, b):
        """Positions (along the first axis) where a equals the atom b."""
        if isinstance(b, torch.Tensor) and b.ndim == 0 and not isinstance(a, torch.Tensor):
            b = b.item()
        if isinstance(a, torch.Tensor):
            if not isinstance(b, (torch.Tensor, int, float)):
                return torch.zeros(0, dtype=torch.int64, device=a.device)
            return torch.where(a == b)[0]
        return super().find_equal(a, b)

    def reciprocal(self, x):
        if isinstance(x, torch.Tensor):
            return torch.reciprocal(x if x.is_floating_point() else x.to(torch.float32))
        return super().reciprocal(x)

    def dict_key(self, k):
        """A dictionary key: a 0-d tensor is the number it holds (tensors hash by identity)."""
        if isinstance(k, torch.Tensor) and k.ndim == 0:
            return k.item()
        return k

    def array_size(self, a):
        """Get the total number of elements in an array/tensor."""
        if isinstance(a, torch.Tensor):
            return a.numel()
        if hasattr(a, 'size'):
            size = a.size
            return size if isinstance(size, int) else size()
        return len(a) if hasattr(a, '__len__') else 1

    def safe_equal(self, x, y):
        """Compare two values, handling torch tensors correctly."""
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            # Convert scalars to tensors for comparison
            if isinstance(x, torch.Tensor) and x.dim() == 0:
                x = x.item()
            if isinstance(y, torch.Tensor) and y.dim() == 0:
                y = y.item()
            return x == y
        # Default numpy comparison
        return numpy.asarray(x, dtype=object) == numpy.asarray(y, dtype=object)

    def array_equal(self, a, b) -> bool:
        """Backend-native exact equality for torch tensors."""
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            return False
        try:
            return bool(torch.equal(a, b))
        except RuntimeError:
            # Fall back to CPU comparison if devices mismatch
            return bool(torch.equal(a.cpu(), b.cpu()))

    def detach_if_needed(self, x):
        """Detach tensor if it requires grad, to allow type conversions."""
        if isinstance(x, torch.Tensor) and x.requires_grad:
            return x.detach()
        return x

    def to_int_array(self, a):
        """Convert array/tensor to integer type.

        A float tensor holding inf or nan (a division by zero) stays a float
        tensor: the IEEE value is the result, an integer cast would be garbage.
        """
        if isinstance(a, torch.Tensor):
            if a.requires_grad or (a.is_floating_point() and not bool(torch.isfinite(a).all())):
                return a  # an integer cast would leave the autograd graph
            return a.to(int)
        return numpy.asarray(a, dtype=int) if isinstance(a, numpy.ndarray) else int(a)

    def floor_to_int(self, a):
        """Floor a value and convert to integer."""
        if not isinstance(a, torch.Tensor):
            a = self.kg_asarray(a)
        return torch.floor(a.float()).to(int)

    @staticmethod
    def _mps_integer_power(a, b):
        """a^b for a float tensor on mps and integer-valued exponents, by repeated squaring (and
        a reciprocal for a negative exponent), else None. mps computes pow as exp(b*log(a)) in
        float32, so 2^-1 came out 0.49999994 and 10^3 999.99994 where the cpu gives exactly 0.5
        and 1000; products of float32 values are correctly rounded on every device."""
        if a.device.type != 'mps' or not a.is_floating_point() or a.dtype == torch.float64:
            return None
        if isinstance(b, torch.Tensor):
            if b.requires_grad or b.dtype == torch.bool:
                return None
            bt = b.to(a.device)
        elif isinstance(b, (int, float, numpy.integer, numpy.floating)) and not isinstance(b, bool):
            bt = torch.tensor(float(b) if isinstance(b, (float, numpy.floating)) else int(b), device=a.device)
        else:
            return None
        if bt.numel() == 0:
            return None
        if bt.is_floating_point():
            if not bool(torch.isfinite(bt).all()) or bool((bt != bt.round()).any()):
                return None
        mag = bt.abs()
        if bool((mag > 4096).any()):
            return None
        n = mag.to(torch.int64)
        try:
            n = n.expand(torch.broadcast_shapes(n.shape, a.shape))
        except RuntimeError:
            return None
        base = a
        result = torch.ones_like(a * n)  # the broadcast shape, float
        while True:
            result = torch.where((n & 1) == 1, result * base, result)
            n = n >> 1
            if not bool((n > 0).any()):
                break
            base = base * base
        return torch.where(bt < 0, 1.0 / result, result)

    def power(self, a, b):
        """Compute a^b, handling gradient tracking for torch tensors."""
        if isinstance(a, torch.Tensor):
            # Handle negative exponents - torch doesn't support int^negative
            if isinstance(b, torch.Tensor) and b.dtype in (torch.int8, torch.int16, torch.int32, torch.int64) and (b < 0).any():
                base = a.float() if a.dtype in (torch.int8, torch.int16, torch.int32, torch.int64) else a
                result = base.pow(b.abs()).float()
                return torch.where(b < 0, 1.0 / result, result)
            b_val = b.item() if isinstance(b, torch.Tensor) and b.ndim == 0 else b
            if isinstance(b_val, (int, numpy.integer)) and b_val < 0:
                a = a.float()
            exact = self._mps_integer_power(a, b)
            if exact is not None:
                return exact
            try:
                return a.pow(b)
            except RuntimeError:
                # an exponent beyond float32 on a device without float64 (mps):
                # the float32 it rounds to (inf), which is what the cpu computes
                if type(b) is float and not isinstance(a, bool):
                    return a.pow(torch.tensor(b, dtype=torch.float64).to(torch.float32).to(a.device))
                raise
        if isinstance(b, torch.Tensor) and (b.ndim > 0 or b.requires_grad) \
                and isinstance(a, (int, float)) and not isinstance(a, bool) \
                and (type(a) is float or -9223372036854775808 <= a <= 9223372036854775807):
            # a number to a tensor power: a tensor, in the autograd graph of the exponent
            # (an exponent that is a plain 0-d tensor is a number: the float64 path below)
            return self.power(torch.as_tensor(float(a), dtype=b.dtype if b.is_floating_point() else torch.float32,
                                              device=b.device), b)
        # For numpy arrays or scalars
        a_val = float(a) if isinstance(a, (int, numpy.integer)) else a
        b_val = b.item() if isinstance(b, torch.Tensor) and b.ndim == 0 else (b.cpu().numpy() if isinstance(b, torch.Tensor) else b)
        with numpy.errstate(divide='ignore'):  # 0^-1 is inf, as on the numpy backend
            return numpy.power(a_val, b_val)

    def vec_fn(self, a, f):
        """As the base, but the results are put into the object array one by one:
        numpy.asarray would read a tensor among them as a sequence, which copies
        it out of the graph on the cpu and fails on any other device."""
        if isinstance(a, numpy.ndarray) and a.dtype == object:
            out = numpy.empty(len(a), dtype=object)
            for i, x in enumerate(a):
                out[i] = self.vec_fn(x, f) if self._is_list(x) else f(x)
            return out
        return f(a)

    def has_gradient(self, x) -> bool:
        """Check if x is tracking gradients."""
        return isinstance(x, torch.Tensor) and x.requires_grad

    def supports_autograd(self) -> bool:
        """Torch backend supports automatic differentiation."""
        return True

    def create_grad_tensor(self, x):
        """Create a tensor that tracks gradients."""
        if isinstance(x, torch.Tensor):
            return x.clone().detach().float().requires_grad_(True)
        elif isinstance(x, numpy.ndarray):
            return torch.from_numpy(x.astype(numpy.float64)).float().requires_grad_(True)
        else:
            return torch.tensor(x, dtype=torch.float32, requires_grad=True)

    def compute_autograd(self, func, x):
        """Compute gradient using PyTorch automatic differentiation."""
        x_tensor = self.create_grad_tensor(x)

        # Compute the function value
        y = func(x_tensor)

        # Check result type - must be a tensor for autograd to work
        if not isinstance(y, torch.Tensor):
            if isinstance(y, numpy.ndarray):
                raise AutogradChainBrokenError(
                    "function output",
                    "torch.Tensor",
                    "numpy.ndarray",
                    "Avoid numpy operations. Use torch-compatible functions."
                )
            raise AutogradChainBrokenError(
                "function output",
                "torch.Tensor",
                type(y).__name__,
                "For autograd, use torch-compatible operations."
            )

        # Ensure y is a scalar
        if y.numel() != 1:
            raise NonScalarLossError(tuple(y.shape))

        # Check requires_grad
        if not y.requires_grad:
            raise AutogradChainBrokenError(
                "gradient computation",
                "requires_grad=True",
                "requires_grad=False",
                "Output lost gradient tracking. Avoid .item(), .numpy(), or Python float()."
            )

        # Compute gradient
        y.backward()

        return x_tensor.grad

    def compute_multi_autograd(self, func, params):
        """
        Compute gradients for multiple parameters using torch.autograd.grad().

        Args:
            func: Callable that takes a list of tensors and returns a scalar loss
            params: List of parameter values to compute gradients for

        Returns:
            List of gradients, one per parameter
        """
        # Create grad tensors for all parameters
        grad_tensors = [self.create_grad_tensor(p) for p in params]

        # Compute the function value (loss)
        y = func(grad_tensors)

        # Validate output is a tensor
        if not isinstance(y, torch.Tensor):
            if isinstance(y, numpy.ndarray):
                raise AutogradChainBrokenError(
                    "loss computation",
                    "torch.Tensor",
                    "numpy.ndarray",
                    "Avoid numpy operations in the loss function."
                )
            raise AutogradChainBrokenError(
                "loss computation",
                "torch.Tensor",
                type(y).__name__,
                "For autograd, use torch-compatible operations."
            )

        # Ensure y is a scalar
        if y.numel() != 1:
            raise NonScalarLossError(tuple(y.shape))

        # Compute all gradients in one backward pass using torch.autograd.grad
        grads = torch.autograd.grad(y, grad_tensors, create_graph=False)

        return list(grads)

    def compute_jacobian(self, func, x):
        """
        Compute Jacobian matrix using torch.autograd.functional.jacobian().

        Args:
            func: Callable that takes x and returns a vector
            x: Input point

        Returns:
            Jacobian matrix J where J[i,j] = df_i/dx_j
        """
        x_tensor = self.create_grad_tensor(x)

        # torch.autograd.functional.jacobian expects func(inputs) -> outputs
        jacobian = torch_autograd_functional.jacobian(func, x_tensor)

        return jacobian

    def str_to_char_array(self, s):
        """A string as a list of characters: a numpy object array, as on the numpy backend
        (text is never a tensor)."""
        return numpy.asarray([KGChar(x) for x in s], dtype=object)

    def compile_function(self, func, example_input, output_path=None, mode="default",
                         backend="inductor", fullgraph=False, dynamic=None):
        """
        Compile a function using torch.compile with configurable options.

        Args:
            func: Callable to compile
            example_input: Example input for tracing the function
            output_path: Optional path to save the exported graph (.pt2 file)
            mode: Compilation mode - affects speed/quality tradeoff
                - "default": Balanced compilation (default)
                - "reduce-overhead": Faster compile, less optimization
                - "max-autotune": Slower compile, maximum runtime performance
            backend: Compilation backend
                - "inductor": Default backend with C++/Triton codegen
                - "eager": No compilation (for debugging)
                - "aot_eager": Ahead-of-time eager (debugging with autograd)
                - "cudagraphs": CUDA graphs for GPU (reduces launch overhead)
            fullgraph: If True, requires entire function to compile as one graph
            dynamic: If True, enables dynamic shapes; if False, assumes static shapes

        Returns:
            If output_path is None: compiled function
            If output_path is provided: dict with compiled function and export info

        Compilation Modes Comparison:
            | Mode            | Compile Time | Runtime Speed | Best For           |
            |-----------------|--------------|---------------|---------------------|
            | default         | Medium       | Good          | General use         |
            | reduce-overhead | Fast         | Moderate      | Quick iteration     |
            | max-autotune    | Slow         | Best          | Production/training |
            | (eager backend) | None         | Baseline      | Debugging           |
        """
        # Convert example input to tensor if needed
        if not isinstance(example_input, torch.Tensor):
            example_input = self.create_grad_tensor(example_input)

        # Build compile options
        compile_kwargs = {
            'mode': mode,
            'backend': backend,
            'fullgraph': fullgraph,
        }
        if dynamic is not None:
            compile_kwargs['dynamic'] = dynamic

        # Compile the function with specified options
        compiled_fn = torch.compile(func, **compile_kwargs)

        # Warm up the compiled function (triggers actual compilation)
        _ = compiled_fn(example_input)

        if output_path is None:
            # Wrap with Klong-convention parameter name (x) so that
            # KGLambda introspection binds the argument correctly when
            # the compiled function is stored via ::
            def klong_compiled(x):
                if not isinstance(x, torch.Tensor):
                    x = self.create_grad_tensor(x)
                return compiled_fn(x)
            return klong_compiled

        # Export the function graph for inspection
        try:
            # Use torch.export for graph capture
            # torch.export takes a torch.nn.Module (torch 2.15 rejects a plain function)
            class _Exported(torch.nn.Module):
                def __init__(self, fn):
                    super().__init__()
                    self.fn = fn

                def forward(self, x):
                    return self.fn(x)

            exported = torch.export.export(_Exported(func), (example_input,))

            # Save the exported program
            torch.export.save(exported, output_path)

            # Get graph representation for inspection
            graph_str = str(exported.graph_module.graph)

            return {
                'compiled_fn': compiled_fn,
                'export_path': output_path,
                'graph': graph_str,
                'graph_module': exported.graph_module,
                'mode': mode,
                'backend': backend,
            }
        except Exception as e:
            # If export fails, still return the compiled function
            return {
                'compiled_fn': compiled_fn,
                'export_path': None,
                'export_error': str(e),
                'mode': mode,
                'backend': backend,
            }

    def get_compile_modes(self):
        """
        Return information about available compilation modes.

        Returns:
            Dict with mode descriptions and recommendations
        """
        return {
            'modes': {
                'default': 'Balanced compilation - good for most cases',
                'reduce-overhead': 'Faster compile time, less optimization - good for development',
                'max-autotune': 'Maximum optimization - best for production/training loops',
            },
            'backends': {
                'inductor': 'Default backend with C++/Triton code generation',
                'eager': 'No compilation - runs original Python (for debugging)',
                'aot_eager': 'Ahead-of-time eager - captures autograd graph (debugging)',
                'cudagraphs': 'CUDA graphs - reduces kernel launch overhead (GPU only)',
            },
            'recommendations': {
                'development': {'mode': 'reduce-overhead', 'backend': 'inductor'},
                'production': {'mode': 'max-autotune', 'backend': 'inductor'},
                'debugging': {'mode': 'default', 'backend': 'eager'},
                'gpu_inference': {'mode': 'max-autotune', 'backend': 'cudagraphs'},
            }
        }

    def gradcheck(self, func, inputs, eps=1e-6, atol=1e-5, rtol=1e-3):
        """
        Check gradients computed by autograd against numeric gradients.

        Uses torch.autograd.gradcheck to verify correctness.

        Args:
            func: Function to check (should return scalar or tensor)
            inputs: Tuple of input tensors (must have requires_grad=True)
            eps: Step size for numeric differentiation
            atol: Absolute tolerance
            rtol: Relative tolerance

        Returns:
            True if gradients match, raises GradcheckError otherwise
        """
        # Ensure inputs are tensors with gradients
        tensor_inputs = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                if not inp.requires_grad:
                    inp = inp.clone().detach().float().requires_grad_(True)
                tensor_inputs.append(inp)
            else:
                tensor_inputs.append(self.create_grad_tensor(inp))

        return torch.autograd.gradcheck(
            func,
            tuple(tensor_inputs),
            eps=eps,
            atol=atol,
            rtol=rtol,
            raise_exception=True
        )

    def klong_gradcheck(self, klong, fn, inputs):
        """
        Check gradients for a Klong function.

        Handles wrapping the Klong function, dtype selection based on device,
        and tolerance adjustment for float32 (MPS).

        Args:
            klong: KlongInterpreter instance
            fn: Klong function to check
            inputs: Input value or list of inputs

        Returns:
            1 if gradients are correct, raises error otherwise
        """
        # Gradcheck requires float64 which is only supported on CPU
        if self.device.type != 'cpu':
            raise KlongSystemError(
                f".gradcheck() requires CPU device, got '{self.device.type}'",
                hint="run with kgpy --backend torch --device cpu")

        dtype = torch.float64

        # Wrap the Klong function
        def wrapped_fn(v):
            result = _invoke_fn(klong, fn, [v])
            # Ensure result is a scalar tensor for gradcheck
            if isinstance(result, torch.Tensor) and result.numel() > 1:
                result = result.sum()
            return result

        # Convert inputs to tensor on CPU for gradcheck
        if isinstance(inputs, (list, tuple)) and not isinstance(inputs[0], torch.Tensor):
            tensor_inputs = torch.tensor(inputs, dtype=dtype, device='cpu', requires_grad=True)
        elif not isinstance(inputs, torch.Tensor):
            tensor_inputs = torch.tensor([inputs], dtype=dtype, device='cpu', requires_grad=True)
        else:
            tensor_inputs = inputs.detach().cpu().to(dtype=dtype).requires_grad_(True)

        result = self.gradcheck(wrapped_fn, (tensor_inputs,))

        return 1 if result else 0

    def compile_expr_ir(self, ir, var_syms):
        """Compile IR tree to a callable using torch tensor operations."""
        source = self._ir_to_source(ir)
        if source is None:
            return None

        param_names = list(self._collect_params(ir))
        fn_source = f"def _expr({', '.join(param_names)}): return {source}"
        # The compiled operators are the ones the interpreter applies:
        # the same ufunc objects through vec_fn2, the interpreter's own ^ and
        # negate, and reduce/scan with the interpreter's treatment of atoms and
        # empty lists. Both paths then give the same values, the same types
        # (tensors, also for two plain numbers) and the same autograd graph.
        xp = self._torch_backend
        divide = xp.divide

        def _div(x, y):
            # the interpreter gives :undefined for atom%0; leave that to it
            if getattr(y, 'ndim', 0) == 0 and getattr(x, 'ndim', 0) == 0 and y == 0:
                raise ZeroDivisionError("division by zero")
            return divide(x, y)

        def _reducing(reduce):
            def over(x):
                shape = getattr(x, 'shape', ())
                if len(shape) and shape[0]:
                    if not isinstance(x, torch.Tensor):
                        raise CompiledDecline("not a tensor")  # an object array: the interpreter folds it
                    return reduce(x)
                return x
            return over

        ns = {
            '_atomic': self.vec_fn2,
            '_add': xp.add, '_sub': xp.subtract, '_mul': xp.multiply,
            '_div': _div, '_pow': self.interpreter_power(), '_neg': self.interpreter_negate(),
            '_eq': self.safe_equal, '_gt': xp.greater, '_lt': xp.less,
            '_rsum': _reducing(xp.add.reduce), '_rprod': _reducing(xp.multiply.reduce),
            '_rmax': _reducing(lambda x: x.amax(0)), '_rmin': _reducing(lambda x: x.amin(0)),
            '_cumsum': _reducing(lambda x: x.cumsum(0)), '_cumprod': _reducing(lambda x: x.cumprod(0)),
            '_cummax': _reducing(lambda x: x.cummax(0).values), '_cummin': _reducing(lambda x: x.cummin(0).values),
        }
        try:
            exec(fn_source, ns)
        except Exception:
            return None
        return (ns['_expr'], var_syms)

    def _ir_to_source(self, ir):
        """Convert IR tree to Python source string with torch operations."""
        node_type = ir[0]

        if node_type == 'literal':
            return repr(ir[1])

        if node_type == 'var':
            return ir[1]

        if node_type == 'binop':
            op, left, right = ir[1], ir[2], ir[3]
            l = self._ir_to_source(left)
            r = self._ir_to_source(right)
            if l is None or r is None:
                return None
            py_op = {'+': '_add', '-': '_sub', '*': '_mul', '%': '_div', '^': '_pow'}.get(op)
            if py_op is None:
                return None
            return f'_atomic({l}, {r}, {py_op})'

        if node_type == 'cmp':
            op, left, right = ir[1], ir[2], ir[3]
            l = self._ir_to_source(left)
            r = self._ir_to_source(right)
            if l is None or r is None:
                return None
            py_cmp = {'=': '_eq', '>': '_gt', '<': '_lt'}.get(op)
            if py_cmp is None:
                return None
            return f'(_atomic({l}, {r}, {py_cmp})*1)'

        if node_type == 'negate':
            child = self._ir_to_source(ir[1])
            if child is None:
                return None
            return f'_neg({child})'

        if node_type == 'reduce':
            op, arg = ir[1], ir[2]
            arg_src = self._ir_to_source(arg)
            if arg_src is None:
                return None
            method = {'+': '_rsum', '*': '_rprod', '|': '_rmax', '&': '_rmin'}.get(op)
            if method is None:
                return None
            return f'{method}({arg_src})'

        if node_type == 'scan':
            op, arg = ir[1], ir[2]
            arg_src = self._ir_to_source(arg)
            if arg_src is None:
                return None
            method = {'+': '_cumsum', '*': '_cumprod', '|': '_cummax', '&': '_cummin'}.get(op)
            if method is None:
                return None
            return f'{method}({arg_src})'

        return None

    def kg_asarray(self, a):
        """
        Converts input data into a PyTorch tensor for KlongPy.

        For data that can't be converted to tensors (strings, heterogeneous
        types, jagged arrays), falls back to numpy object arrays to maintain
        compatibility with Klong's list semantics.
        """
        if isinstance(a, str):
            # Strings become numpy character arrays like in numpy backend
            return self.str_to_char_array(a)
        try:
            # Check for jagged arrays early - torch converts them incorrectly
            if is_jagged_array(a):
                raise TorchUnsupportedDtypeError("Jagged arrays not supported")
            arr = self._torch_backend.asarray(a)
            if hasattr(arr, 'dtype'):
                # For torch tensors, dtype doesn't have .kind attribute
                if hasattr(arr.dtype, 'kind'):
                    if arr.dtype.kind not in ['O', 'i', 'f']:
                        raise ValueError
            return arr
        except (NumpyVisibleDeprecationWarning, ValueError, TypeError, RuntimeError, TorchUnsupportedDtypeError):
            return self._object_array(a)

    def _object_array(self, a):
        """
        The numpy object array for data a tensor cannot hold (strings, symbols,
        mixed or ragged lists). Inner Python lists are converted as the numpy
        backend converts them (numeric ones become numpy arrays, symbols and
        strings are kept as they are); tensors that are members stay tensors,
        except that a 0-d tensor outside the autograd graph becomes the number
        it holds.
        """
        if isinstance(a, (list, tuple)) and any(isinstance(x, torch.Tensor) for x in a):
            out = numpy.empty(len(a), dtype=object)
            for i, x in enumerate(a):
                if isinstance(x, torch.Tensor):
                    out[i] = x.item() if x.ndim == 0 and not x.requires_grad else x
                else:
                    out[i] = _HOST.kg_asarray(x) if isinstance(x, list) else x
            return out
        try:
            arr = numpy.asarray(a, dtype=object)
            if arr.ndim == 0:
                return arr
            # Recursively convert inner lists to numpy arrays
            return numpy.asarray(
                [_HOST.kg_asarray(x) if isinstance(x, list) else x for x in arr],
                dtype=object
            )
        except (ValueError, TypeError):
            try:
                return _HOST.kg_asarray(a)
            except (ValueError, TypeError):
                # Last resort: keep as list
                return a


# numpy conversions of the data tensors cannot hold
_HOST = NumpyBackendProvider()
