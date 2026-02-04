"""
Backend public API for KlongPy.

Re-exports registry helpers and core backend types.
"""
from .base import (
    BackendProvider,
    UnsupportedDtypeError,
    is_jagged_array,
    is_supported_type,
)
from .numpy_backend import NumpyBackendProvider, KGChar
from .registry import get_backend, list_backends, register_backend, TorchBackendProvider

__all__ = [
    'BackendProvider',
    'UnsupportedDtypeError',
    'NumpyBackendProvider',
    'TorchBackendProvider',
    'KGChar',
    'get_backend',
    'register_backend',
    'list_backends',
    'is_jagged_array',
    'is_supported_type',
]
