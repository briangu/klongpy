from .interpreter import KlongInterpreter, KlongException
from .backends import (
    get_backend,
    register_backend,
    list_backends,
    BackendProvider,
    UnsupportedDtypeError,
)

# Fallback for uninstalled source trees; pyproject.toml's [project] version is
# the source of truth and tests/test_version.py asserts the two agree.
__version__ = "0.7.0"

__all__ = [
    "__version__",
    "KlongInterpreter",
    "KlongException",
    "UnsupportedDtypeError",
    "get_backend",
    "register_backend",
    "list_backends",
    "BackendProvider",
]
