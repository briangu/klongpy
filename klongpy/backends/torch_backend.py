"""
PyTorch backend provider for KlongPy.

This backend uses PyTorch tensors for array operations, enabling GPU acceleration.
It does not support object dtype or string operations.
"""
import math
import numpy
import torch
import torch.autograd.functional as torch_autograd_functional

from .base import BackendProvider, UnsupportedDtypeError, is_jagged_array
from ..autograd import AutogradChainBrokenError, NonScalarLossError, _invoke_fn

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
            arr = numpy.asarray(a)
            if arr.dtype == object:
                raise TorchUnsupportedDtypeError(
                    "PyTorch backend does not support object dtype arrays."
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
        """Convert array/tensor to integer type."""
        if isinstance(a, torch.Tensor):
            return a.to(int)
        return numpy.asarray(a, dtype=int) if isinstance(a, numpy.ndarray) else int(a)

    def floor_to_int(self, a):
        """Floor a value and convert to integer."""
        if not isinstance(a, torch.Tensor):
            a = self.kg_asarray(a)
        return torch.floor(a.float()).to(int)

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
            return a.pow(b)
        # For numpy arrays or scalars
        a_val = float(a) if isinstance(a, (int, numpy.integer)) else a
        b_val = b.item() if isinstance(b, torch.Tensor) and b.ndim == 0 else (b.cpu().numpy() if isinstance(b, torch.Tensor) else b)
        return numpy.power(a_val, b_val)

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
        """Not supported in torch backend."""
        raise TorchUnsupportedDtypeError(
            "PyTorch backend does not support string-to-character array conversion."
        )

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
            exported = torch.export.export(func, (example_input,))

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
            raise RuntimeError(
                f".gradcheck() requires CPU device, got '{self.device.type}'. "
                "Run with: kgpy --backend torch --device cpu"
            )

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
        except (NumpyVisibleDeprecationWarning, ValueError, TypeError, RuntimeError, TorchUnsupportedDtypeError):
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
