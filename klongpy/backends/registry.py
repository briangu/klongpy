"""
Backend registry for KlongPy.

Owns backend registration, lookup, and lazy torch loading.
"""
import importlib
import importlib.util

from .base import BackendProvider
from .numpy_backend import NumpyBackendProvider

# Registry of available backends
_BACKENDS = {}

# Default backend name
_DEFAULT_BACKEND = 'numpy'

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_TORCH_BACKEND_LOADED = False
TorchBackendProvider = None


def register_backend(name: str, provider_class):
    """Register a backend provider class."""
    _BACKENDS[name] = provider_class


def _load_torch_backend():
    global _TORCH_BACKEND_LOADED, TorchBackendProvider
    if _TORCH_BACKEND_LOADED or not _TORCH_AVAILABLE:
        return
    _torch_backend = importlib.import_module("klongpy.backends.torch_backend")
    TorchBackendProvider = _torch_backend.TorchBackendProvider
    register_backend('torch', TorchBackendProvider)
    _TORCH_BACKEND_LOADED = True


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
        name = _DEFAULT_BACKEND

    if name == 'torch':
        _load_torch_backend()

    if name not in _BACKENDS:
        available = ', '.join(_BACKENDS.keys())
        raise ValueError(f"Unknown backend: '{name}'. Available: {available}")

    return _BACKENDS[name](**kwargs)


def list_backends():
    """Return list of available backend names."""
    backends = list(_BACKENDS.keys())
    if _TORCH_AVAILABLE and 'torch' not in backends:
        backends.append('torch')
    return backends


# Register built-in backends
register_backend('numpy', NumpyBackendProvider)
