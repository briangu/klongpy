import os
import warnings

# Attempt to import PyTorch. If not available, set use_torch to False.
use_torch = bool(os.environ.get('USE_TORCH') == '1')
if use_torch:
    try:
        import torch
        import numpy
        use_torch = True
    except ImportError:
        import numpy as np
        use_torch = False
else:
    import numpy as np


class TorchUnsupportedDtypeError(Exception):
    """Raised when an operation requires object dtype which is not supported by PyTorch."""
    pass


def is_supported_type(x):
    """
    PyTorch does not support strings or jagged arrays.
    Note: add any other unsupported types here.
    """
    if isinstance(x, str) or is_jagged_array(x):
        return False
    return True


def is_jagged_array(x):
    """
    Check if an array is jagged (nested lists with varying lengths).
    """
    if isinstance(x, list) and len(x) > 0:
        # Check if elements are lists/sequences that can have length
        if all(isinstance(item, (list, tuple)) for item in x):
            # If the lengths of sublists vary, it's a jagged array.
            return len(set(map(len, x))) > 1
    return False


if use_torch:
    class TorchBackend:
        """
        A wrapper that provides a NumPy-compatible interface using PyTorch tensors.
        """
        def __init__(self):
            self._numpy = numpy
            self._torch = torch
            # Determine the best available device
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')

        def __getattr__(self, name):
            # First check if torch has this attribute
            if hasattr(self._torch, name):
                attr = getattr(self._torch, name)
                # If it's callable, wrap it to handle tensor/array conversion
                if callable(attr):
                    return self._wrap_torch_func(attr, name)
                return attr
            # Fall back to numpy for things torch doesn't have
            if hasattr(self._numpy, name):
                return getattr(self._numpy, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        def _wrap_torch_func(self, func, name):
            def wrapper(*args, **kwargs):
                # Convert numpy arrays to torch tensors
                converted_args = []
                for arg in args:
                    if isinstance(arg, numpy.ndarray):
                        converted_args.append(torch.from_numpy(arg).to(self.device))
                    elif isinstance(arg, list):
                        try:
                            converted_args.append(torch.tensor(arg, device=self.device))
                        except (ValueError, TypeError):
                            # Can't convert to tensor, use as-is
                            converted_args.append(arg)
                    else:
                        converted_args.append(arg)
                return func(*converted_args, **kwargs)
            return wrapper

        def asarray(self, a, dtype=None):
            """Convert input to a torch tensor or numpy array."""
            if dtype is not None and (dtype == object or (hasattr(dtype, 'kind') and dtype.kind == 'O')):
                raise TorchUnsupportedDtypeError(
                    "PyTorch backend does not support object dtype. "
                    "This operation requires heterogeneous data types which are not supported."
                )
            if isinstance(a, torch.Tensor):
                return a
            if isinstance(a, numpy.ndarray):
                if a.dtype == object:
                    raise TorchUnsupportedDtypeError(
                        "PyTorch backend does not support object dtype arrays."
                    )
                return torch.from_numpy(a).to(self.device)
            try:
                result = torch.tensor(a, device=self.device)
                return result
            except (ValueError, TypeError, RuntimeError) as e:
                raise TorchUnsupportedDtypeError(
                    f"PyTorch backend cannot convert this data: {e}"
                )

        def array(self, a, dtype=None):
            """Create a torch tensor or numpy array."""
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
            a_t = self.asarray(a) if not isinstance(a, torch.Tensor) else a
            b_t = self.asarray(b) if not isinstance(b, torch.Tensor) else b
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

        # Ufunc-like wrapper for operations that need .reduce and .accumulate
        class TorchUfunc:
            def __init__(self, backend, op, reduce_op, accumulate_op=None):
                self._backend = backend
                self._op = op
                self._reduce_op = reduce_op
                self._accumulate_op = accumulate_op

            def __call__(self, a, b):
                return self._op(self._backend.asarray(a), self._backend.asarray(b))

            def reduce(self, a, axis=None):
                arr = self._backend.asarray(a)
                if axis is None:
                    return self._reduce_op(arr)
                return self._reduce_op(arr, dim=axis)

            def accumulate(self, a, axis=0):
                arr = self._backend.asarray(a)
                if self._accumulate_op:
                    return self._accumulate_op(arr, dim=axis)
                # Fallback implementation
                result = [arr[0]]
                for i in range(1, len(arr)):
                    result.append(self._op(result[-1], arr[i]))
                return torch.stack(result)

        @property
        def add(self):
            return self.TorchUfunc(self, torch.add, torch.sum, torch.cumsum)

        @property
        def subtract(self):
            def cumulative_subtract(a, dim=0):
                result = [a[0]]
                for i in range(1, a.shape[dim]):
                    result.append(result[-1] - a[i])
                return torch.stack(result)
            return self.TorchUfunc(self, torch.subtract,
                                   lambda a, dim=None: a[0] - torch.sum(a[1:]) if dim is None else None,
                                   cumulative_subtract)

        @property
        def multiply(self):
            return self.TorchUfunc(self, torch.multiply, torch.prod, torch.cumprod)

        @property
        def divide(self):
            def reduce_divide(a, dim=None):
                if dim is None:
                    result = a.flatten()[0]
                    for x in a.flatten()[1:]:
                        result = result / x
                    return result
                return None
            return self.TorchUfunc(self, torch.divide, reduce_divide)

        @property
        def inf(self):
            return float('inf')

        def seterr(self, **kwargs):
            # PyTorch doesn't have this, just ignore
            pass

        @property
        def VisibleDeprecationWarning(self):
            return numpy.VisibleDeprecationWarning

    np = TorchBackend()

else:
    np.seterr(divide='ignore')
    warnings.filterwarnings("error", category=np.VisibleDeprecationWarning)
    np.isarray = lambda x: isinstance(x, np.ndarray)

np
