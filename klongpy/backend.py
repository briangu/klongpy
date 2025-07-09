"""Backend selection utilities for klongpy."""

from importlib import import_module
from typing import Any
import numpy as _np

# default numpy compatibility shim for legacy modules
_np.seterr(divide="ignore")
_np.isarray = lambda x: isinstance(x, _np.ndarray)
np = _np

BACKEND = "numpy"


def current():
    """Return the currently selected backend module."""
    return import_module(f"klongpy.backends.{BACKEND}_backend")


def set_backend(name: str) -> None:
    """Select the computation backend.

    Parameters
    ----------
    name: str
        Either ``"numpy"`` or ``"torch"``.
    """
    global BACKEND
    name = name.lower()
    if name not in {"numpy", "torch"}:
        raise ValueError(f"unknown backend '{name}'")
    if name == "torch":
        import_module("klongpy.backends.torch_backend")
    BACKEND = name


def array(obj: Any, *, dtype: Any | None = None, requires_grad: bool = False) -> Any:
    """Create an array or tensor using the active backend."""
    return current().array(obj, dtype=dtype, requires_grad=requires_grad)


def add(a: Any, b: Any) -> Any:
    """Element-wise addition via the active backend."""
    return current().add(a, b)


def mul(a: Any, b: Any) -> Any:
    """Element-wise multiplication via the active backend."""
    return current().mul(a, b)


def matmul(a: Any, b: Any) -> Any:
    """Matrix multiplication via the active backend."""
    return current().matmul(a, b)


def sum(a: Any, axis: int | None = None) -> Any:
    """Sum elements of ``a`` via the active backend."""
    return current().sum(a, axis=axis)


def grad(fn: Any, wrt: int = 0) -> Any:
    """Return gradient function via the active backend."""
    return current().grad(fn, wrt=wrt)


def stop(x: Any) -> Any:
    """Detach ``x`` from the autograd graph via the active backend."""
    return current().stop(x)
