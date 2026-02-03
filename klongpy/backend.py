"""
Backend compatibility module for KlongPy.

This module provides backward compatibility with the old backend import style.
New code should use the backends package directly:

    from klongpy.backends import get_backend, BackendProvider

For per-interpreter backends, use:

    klong = KlongInterpreter(backend='torch')
"""
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

# Module-level np for backward compatibility with existing imports
# This uses the default numpy backend
_default_np_backend = get_backend('numpy')
np = _default_np_backend.np
use_torch = False  # Deprecated: each interpreter has its own backend

__all__ = [
    'np',
    'use_torch',
    'get_backend',
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
