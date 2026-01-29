"""
Backend registry for KlongPy.

Provides a unified interface for registering and retrieving array backends.
The default backend is 'numpy'.
"""
import os

from .base import BackendProvider, UnsupportedDtypeError
from .numpy_backend import NumpyBackendProvider, KGChar

# Registry of available backends
_BACKENDS = {}

# Default backend name
_DEFAULT_BACKEND = 'numpy'


def register_backend(name: str, provider_class):
    """Register a backend provider class."""
    _BACKENDS[name] = provider_class


def get_backend(name: str = None, **kwargs) -> BackendProvider:
    """
    Get a backend provider instance.

    Parameters
    ----------
    name : str, optional
        Backend name ('numpy' or 'torch'). If None, uses default.
    **kwargs
        Additional arguments passed to the backend provider constructor.

    Returns
    -------
    BackendProvider
        The backend provider instance.
    """
    if name is None:
        # Check environment variable for default
        env_backend = os.environ.get('KLONG_BACKEND', '').lower()
        if env_backend == 'torch' or os.environ.get('USE_TORCH') == '1':
            name = 'torch'
        else:
            name = _DEFAULT_BACKEND

    if name not in _BACKENDS:
        available = ', '.join(_BACKENDS.keys())
        raise ValueError(f"Unknown backend: '{name}'. Available: {available}")

    return _BACKENDS[name](**kwargs)


def list_backends():
    """Return list of available backend names."""
    return list(_BACKENDS.keys())


def set_default_backend(name: str):
    """Set the default backend name."""
    global _DEFAULT_BACKEND
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend: '{name}'")
    _DEFAULT_BACKEND = name


# Register built-in backends
register_backend('numpy', NumpyBackendProvider)

# Try to register torch backend if available
try:
    from .torch_backend import TorchBackendProvider, TorchUnsupportedDtypeError
    register_backend('torch', TorchBackendProvider)
except ImportError:
    # Torch not available
    TorchBackendProvider = None
    TorchUnsupportedDtypeError = UnsupportedDtypeError


__all__ = [
    'BackendProvider',
    'UnsupportedDtypeError',
    'TorchUnsupportedDtypeError',
    'NumpyBackendProvider',
    'TorchBackendProvider',
    'KGChar',
    'get_backend',
    'register_backend',
    'list_backends',
    'set_default_backend',
]
