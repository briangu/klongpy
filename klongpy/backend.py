"""
Backend compatibility module for KlongPy.

This module provides backward compatibility with the old global backend system.
New code should use the backends package directly:

    from klongpy.backends import get_backend, BackendProvider

For per-interpreter backends, use:

    klong = KlongInterpreter(backend='torch')
"""
import os
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
    'array_size',
    'safe_equal',
    'detach_if_needed',
    'to_int_array',
    'power',
    'has_gradient',
]
