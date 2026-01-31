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
]
