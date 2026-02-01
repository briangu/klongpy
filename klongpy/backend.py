"""
Backend compatibility module for KlongPy.

This module provides backward compatibility with the old global backend system.
New code should use the backends package directly:

    from klongpy.backends import get_backend, BackendProvider

For per-interpreter backends, use:

    klong = KlongInterpreter(backend='torch')
"""
import numpy as real_np

from .backends import (
    get_backend,
    register_backend,
    list_backends,
    BackendProvider,
    UnsupportedDtypeError,
    TorchUnsupportedDtypeError,
    NumpyBackendProvider,
    TorchBackendProvider,
    KGChar,
    is_jagged_array,
    is_supported_type,
)


# Global backend state for backward compatibility
# This is used by modules that import `np` and `use_torch` directly
_default_backend = get_backend()

# Backward compatibility: expose np and use_torch at module level
np = _default_backend.np
use_torch = _default_backend.name == 'torch'


def get_default_backend():
    """Get the default backend provider."""
    return _default_backend


def to_numpy(x):
    """Convert tensor/array to numpy, handling device transfers and 0-dim arrays."""
    result = _default_backend.to_numpy(x)
    if isinstance(result, real_np.ndarray) and result.ndim == 0:
        return result.item()
    return result


def to_display(x):
    """Convert value to display-friendly format."""
    return _default_backend.to_display(x)


def array_size(a):
    """
    Get the total number of elements in an array/tensor.

    Works with both numpy arrays and torch tensors.
    """
    return _default_backend.array_size(a)


def safe_equal(x, y):
    """Compare two values for equality, handling backend-specific array types."""
    return _default_backend.safe_equal(x, y)


def detach_if_needed(x):
    """Detach array from computation graph if needed."""
    return _default_backend.detach_if_needed(x)


def to_int_array(a):
    """Convert array to integer type."""
    return _default_backend.to_int_array(a)


def power(a, b):
    """Compute a^b, handling gradient tracking if applicable."""
    return _default_backend.power(a, b)


def has_gradient(x):
    """Check if x is tracking gradients (for autograd)."""
    return _default_backend.has_gradient(x)


def kg_asarray(a):
    """Convert input to array using the default backend's kg_asarray method."""
    return _default_backend.kg_asarray(a)


def is_integer(x):
    """Check if x is an integer type using the default backend."""
    return _default_backend.is_integer(x)


def is_float(x):
    """Check if x is a float type using the default backend."""
    return _default_backend.is_float(x)


def is_number(a):
    """Check if a is a number (integer or float) using the default backend."""
    return _default_backend.is_number(a)


def get_dtype_kind(arr):
    """Get the dtype 'kind' character for an array using the default backend."""
    return _default_backend.get_dtype_kind(arr)


def str_to_chr_arr(s):
    """Convert string to character array using the default backend."""
    return _default_backend.str_to_chr_arr(s)


def kg_argsort(a, descending=False):
    """Argsort array using the default backend."""
    from .core import kg_argsort as core_kg_argsort
    return core_kg_argsort(a, _default_backend, descending=descending)


def vec_fn(a, f):
    """Apply a function f to an array a, with support for nested arrays."""
    from .core import vec_fn as core_vec_fn
    return core_vec_fn(a, f, _default_backend)


def kg_equal(a, b):
    """Compare two values or arrays for equality using the default backend."""
    from .core import kg_equal as core_kg_equal
    return core_kg_equal(a, b, _default_backend)


__all__ = [
    'np',
    'use_torch',
    'get_backend',
    'get_default_backend',
    'register_backend',
    'list_backends',
    'BackendProvider',
    'UnsupportedDtypeError',
    'TorchUnsupportedDtypeError',
    'NumpyBackendProvider',
    'TorchBackendProvider',
    'KGChar',
    'is_supported_type',
    'is_jagged_array',
    'to_numpy',
    'to_display',
    'array_size',
    'safe_equal',
    'detach_if_needed',
    'to_int_array',
    'power',
    'has_gradient',
    'kg_asarray',
    'is_integer',
    'is_float',
    'is_number',
    'get_dtype_kind',
    'str_to_chr_arr',
    'kg_argsort',
    'vec_fn',
    'kg_equal',
]
