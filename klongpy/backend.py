"""
Backend module for KlongPy.

Prefer using the backends package directly:

    from klongpy.backends import get_backend, BackendProvider

For per-interpreter backends, use:

    klong = KlongInterpreter(backend='torch')
"""
from .backends.base import (
    BackendProvider,
    UnsupportedDtypeError,
    is_jagged_array,
    is_supported_type,
)
from .backends.numpy_backend import KGChar, NumpyBackendProvider
from .backends.registry import get_backend, list_backends, register_backend, TorchBackendProvider

_default_np_backend = get_backend('numpy')
np = _default_np_backend.np
bknp = np

__all__ = [
    'np',
    'bknp',
    'get_backend',
    'register_backend',
    'list_backends',
    'BackendProvider',
    'UnsupportedDtypeError',
    'NumpyBackendProvider',
    'TorchBackendProvider',
    'KGChar',
    'is_supported_type',
    'is_jagged_array',
]
