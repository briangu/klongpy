"""
PyTorch backend provider for KlongPy.

This backend uses PyTorch tensors for array operations, enabling GPU acceleration.
It does not support object dtype or string operations.
"""
import math
import numpy
import torch

from .base import BackendProvider, UnsupportedDtypeError


class TorchUnsupportedDtypeError(UnsupportedDtypeError):
    """Raised when an operation requires object dtype which is not supported by PyTorch."""
    pass


def is_supported_type(x):
    """
    PyTorch does not support strings or jagged arrays.
    """
    if isinstance(x, str) or is_jagged_array(x):
        return False
    return True


def is_jagged_array(x):
    """
    Check if an array is jagged (nested lists with varying lengths).
    """
    if isinstance(x, list) and len(x) > 0:
        if all(isinstance(item, (list, tuple)) for item in x):
            return len(set(map(len, x))) > 1
    return False


class TorchRandomModule:
    """
    A NumPy-compatible random module using PyTorch.
    """
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
    """
    A wrapper around torch dtype that provides numpy-compatible attributes.
    """
    def __init__(self, torch_dtype):
        self._dtype = torch_dtype
        # Map torch dtypes to numpy dtype kinds
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


class TorchTensor:
    """
    A wrapper around torch.Tensor that provides numpy-compatible dtype attribute.
    """
    pass  # We'll use monkey-patching instead for simplicity


class TorchBackend:
    """
    A wrapper that provides a NumPy-compatible interface using PyTorch tensors.
    """
    def __init__(self, device=None):
        self._numpy = numpy
        self._torch = torch
        self._random = None  # Lazy init
        # Cached ufuncs - initialized lazily
        self._add = None
        self._subtract = None
        self._multiply = None
        self._divide = None

        # Determine the best available device
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
        """Convert input to a torch tensor."""
        if dtype is not None and (dtype == object or (hasattr(dtype, 'kind') and dtype.kind == 'O')):
            raise TorchUnsupportedDtypeError(
                "PyTorch backend does not support object dtype."
            )
        if isinstance(a, torch.Tensor):
            # Move to correct device if needed
            if a.device != self.device:
                return a.to(self.device)
            return a
        if isinstance(a, numpy.ndarray):
            if a.dtype == object:
                raise TorchUnsupportedDtypeError(
                    "PyTorch backend does not support object dtype arrays."
                )
            # Handle float64 on MPS by converting to float32
            if a.dtype == numpy.float64 and self.device.type == 'mps':
                a = a.astype(numpy.float32)
            return torch.from_numpy(a).to(self.device)
        # Check if input is a list/tuple of tensors - use stack to preserve gradients
        if isinstance(a, (list, tuple)) and len(a) > 0 and all(isinstance(x, torch.Tensor) for x in a):
            # torch.stack preserves requires_grad, torch.tensor does not
            result = torch.stack(a)
            if result.device != self.device:
                result = result.to(self.device)
            # Handle float64 on MPS
            if result.dtype == torch.float64 and self.device.type == 'mps':
                result = result.to(torch.float32)
            return result
        # Check if input contains any arrays/tensors mixed with non-arrays - these need object dtype
        if isinstance(a, (list, tuple)) and len(a) > 0:
            has_array = any(isinstance(x, (torch.Tensor, numpy.ndarray, list)) for x in a)
            has_scalar = any(isinstance(x, (int, float)) and not isinstance(x, bool) for x in a)
            if has_array and has_scalar:
                # Mixed array/scalar list - can't represent in torch without losing structure
                raise TorchUnsupportedDtypeError(
                    "PyTorch backend cannot convert mixed array/scalar lists without losing structure."
                )
        try:
            t = torch.tensor(a, device=self.device)
            # Handle float64 on MPS
            if t.dtype == torch.float64 and self.device.type == 'mps':
                t = t.to(torch.float32)
            return t
        except (ValueError, TypeError, RuntimeError) as e:
            raise TorchUnsupportedDtypeError(
                f"PyTorch backend cannot convert this data: {e}"
            )

    def array(self, a, dtype=None):
        """Create a torch tensor."""
        return self.asarray(a, dtype=dtype)

    def isarray(self, x):
        """Check if x is an array (numpy or torch tensor)."""
        return isinstance(x, (numpy.ndarray, torch.Tensor))

    def zeros(self, shape, dtype=None):
        return torch.zeros(shape, device=self.device)

    def ones(self, shape, dtype=None):
        return torch.ones(shape, device=self.device)

    def arange(self, *args, **kwargs):
        return torch.arange(*args, device=self.device, **kwargs)

    def concatenate(self, arrays, axis=0):
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
        return self.asarray(a).clone()

    def isclose(self, a, b, rtol=1e-05, atol=1e-08):
        # For scalars, use numpy's isclose to avoid tensor conversion issues
        if not hasattr(a, '__len__') and not hasattr(b, '__len__'):
            return self._numpy.isclose(float(a), float(b), rtol=rtol, atol=atol)
        a_t = self.asarray(a) if not isinstance(a, torch.Tensor) else a
        b_t = self.asarray(b) if not isinstance(b, torch.Tensor) else b
        # torch.isclose requires same dtype, convert to float if needed
        if a_t.dtype != b_t.dtype:
            # Use float32 for MPS compatibility (MPS doesn't support float64)
            a_t = a_t.to(torch.float32)
            b_t = b_t.to(torch.float32)
        return torch.isclose(a_t, b_t, rtol=rtol, atol=atol)

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

    def minimum(self, a, b):
        """Element-wise minimum."""
        a_t = self.asarray(a) if not isinstance(a, torch.Tensor) else a
        b_t = self.asarray(b) if not isinstance(b, torch.Tensor) else b
        return torch.minimum(a_t, b_t)

    def maximum(self, a, b):
        """Element-wise maximum."""
        a_t = self.asarray(a) if not isinstance(a, torch.Tensor) else a
        b_t = self.asarray(b) if not isinstance(b, torch.Tensor) else b
        return torch.maximum(a_t, b_t)

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

    def sign(self, a):
        """Sign of elements."""
        if isinstance(a, (int, float)):
            return (a > 0) - (a < 0)
        a_t = self.asarray(a) if not isinstance(a, torch.Tensor) else a
        return torch.sign(a_t)

    # Ufunc-like wrapper for operations that need .reduce and .accumulate
    class TorchUfunc:
        def __init__(self, backend, op, reduce_op, accumulate_op=None, numpy_ufunc=None):
            self._backend = backend
            self._op = op
            self._reduce_op = reduce_op
            self._accumulate_op = accumulate_op
            self._torch = torch
            self._numpy_ufunc = numpy_ufunc  # Fallback for object arrays

        def _is_object_array(self, x):
            """Check if x is a numpy object array that torch can't handle."""
            if isinstance(x, numpy.ndarray) and x.dtype == object:
                return True
            return False

        def _to_numpy(self, x):
            """Convert tensor to numpy array for fallback operations."""
            if isinstance(x, self._torch.Tensor):
                return x.detach().cpu().numpy()
            return x

        def __call__(self, a, b):
            # Fast path: both are tensors on the same device, or tensor + scalar
            a_is_tensor = isinstance(a, self._torch.Tensor)
            b_is_tensor = isinstance(b, self._torch.Tensor)
            if a_is_tensor and b_is_tensor:
                if a.device == b.device:
                    return self._op(a, b)
            elif a_is_tensor and isinstance(b, (int, float)):
                return self._op(a, b)
            elif b_is_tensor and isinstance(a, (int, float)):
                return self._op(a, b)
            # Fall back to numpy for object arrays
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
                if axis is None:
                    return self._reduce_op(arr)
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
        return self.TorchUfunc(
            self, torch.subtract,
            lambda a, dim=None: a[0] - torch.sum(a[1:]) if dim is None else None,
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
            if dim is None:
                result = a.flatten()[0]
                for x in a.flatten()[1:]:
                    result = result / x
                return result
            return None
        return self.TorchUfunc(self, torch.divide, reduce_divide, None, numpy.divide)

    @property
    def inf(self):
        return float('inf')

    def seterr(self, **kwargs):
        pass

    @property
    def VisibleDeprecationWarning(self):
        return numpy.VisibleDeprecationWarning


class TorchBackendProvider(BackendProvider):
    """PyTorch-based backend provider."""

    def __init__(self, device=None):
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

    def supports_object_dtype(self) -> bool:
        return False

    def supports_strings(self) -> bool:
        return False

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

    def is_scalar_integer(self, x) -> bool:
        if isinstance(x, torch.Tensor) and x.ndim == 0:
            return x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        return False

    def is_scalar_float(self, x) -> bool:
        if isinstance(x, torch.Tensor) and x.ndim == 0:
            return x.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16)
        return False

    def argsort(self, a, descending=False):
        """Return indices that would sort the array."""
        if not isinstance(a, torch.Tensor):
            a = self._torch_backend.asarray(a)
        return torch.argsort(a, descending=descending)

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

        # Convert result to tensor if needed (e.g., if function used numpy/python math)
        if not isinstance(y, torch.Tensor):
            # Function broke the autograd chain - result is not a tensor
            # This happens when using numpy or math functions instead of torch
            raise ValueError(
                f"Function returned {type(y).__name__} instead of Tensor. "
                "For autograd, use torch-compatible functions (e.g., .bkf('sin') instead of .pyf('math';'sin'))"
            )

        # Ensure y is a scalar
        if y.numel() != 1:
            raise ValueError(f"Function must return a scalar, got shape {y.shape}")

        # Compute gradient
        y.backward()

        return x_tensor.grad

    def str_to_char_array(self, s):
        """Not supported in torch backend."""
        raise TorchUnsupportedDtypeError(
            "PyTorch backend does not support string-to-character array conversion."
        )

    def kg_asarray(self, a):
        """
        Converts input data into a PyTorch tensor for KlongPy.

        For data that can't be converted to tensors (strings, heterogeneous
        types, jagged arrays), falls back to numpy object arrays to maintain
        compatibility with Klong's list semantics.
        """
        if isinstance(a, str):
            # Strings become numpy character arrays like in numpy backend
            return numpy.array(list(a))
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
        except (numpy.VisibleDeprecationWarning, ValueError, TypeError, RuntimeError, TorchUnsupportedDtypeError):
            # Fall back to numpy object array for heterogeneous/unsupported data
            # Use numpy for inner conversions to avoid MPS tensor issues
            def _numpy_convert(x):
                if isinstance(x, list):
                    try:
                        return numpy.asarray(x)
                    except (ValueError, TypeError):
                        return numpy.asarray([_numpy_convert(i) for i in x], dtype=object)
                return x
            try:
                arr = numpy.asarray(a, dtype=object)
                # Recursively convert inner lists to numpy arrays
                arr = numpy.asarray(
                    [_numpy_convert(x) if isinstance(x, list) else x for x in arr],
                    dtype=object
                )
                return arr
            except (ValueError, TypeError):
                # Last resort: keep as list
                return a
