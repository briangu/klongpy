"""
Backend compatibility module for KlongPy.

This module provides backward compatibility with the old global backend system.
New code should use the backends package directly:

    from klongpy.backends import get_backend, BackendProvider

For per-interpreter backends, use:

    klong = KlongInterpreter(backend='torch')
"""
import os

from .backends import (
    get_backend,
    register_backend,
    list_backends,
    BackendProvider,
    UnsupportedDtypeError,
    NumpyBackendProvider,
    KGChar,
)

# Try to import torch-specific items
try:
    from .backends.torch_backend import (
        TorchBackendProvider,
        TorchUnsupportedDtypeError,
        TorchBackend,
        is_supported_type,
        is_jagged_array,
    )
except ImportError:
    TorchBackendProvider = None
    TorchUnsupportedDtypeError = UnsupportedDtypeError

    def is_supported_type(x):
        return True

    def is_jagged_array(x):
        if isinstance(x, list) and len(x) > 0:
            if all(isinstance(item, (list, tuple)) for item in x):
                return len(set(map(len, x))) > 1
        return False


# Global backend state for backward compatibility
# This is used by modules that import `np` and `use_torch` directly
_default_backend = get_backend()

# Backward compatibility: expose np and use_torch at module level
np = _default_backend.np
use_torch = _default_backend.name == 'torch'


def get_default_backend():
    """Get the default backend provider."""
    return _default_backend


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
]
