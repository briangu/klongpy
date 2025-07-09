"""Torch backend using NumPy fallback for strings."""

from __future__ import annotations

from typing import Any, Callable
import os

try:
    import torch
except Exception as e:  # pragma: no cover - optional dependency
    raise ImportError("torch backend requires the 'torch' package") from e

import numpy as np
from . import numpy_backend as npb

# when STRICT is true, string checks are skipped for performance
STRICT = os.environ.get("KLONGPY_TORCH_STRICT", "0") == "1"


def _contains_strings(x: Any) -> bool:
    if isinstance(x, str):
        return True
    if isinstance(x, (list, tuple)):
        return any(_contains_strings(i) for i in x)
    if isinstance(x, np.ndarray):
        return x.dtype.kind in {"U", "S", "O"}
    return False


def _has_strings(*objs: Any) -> bool:
    """Return True if any object contains strings unless STRICT mode is on."""
    if STRICT:
        return False
    return any(_contains_strings(o) for o in objs)


def _to_numpy(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def array(obj: Any, *, dtype: Any | None = None, requires_grad: bool = False) -> Any:
    """Create a torch tensor or numpy array."""
    if _has_strings(obj):
        return npb.array(obj, dtype=dtype, requires_grad=requires_grad)
    return torch.tensor(obj, dtype=dtype, requires_grad=requires_grad)


def _torchify(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x)


def add(a: Any, b: Any) -> Any:
    if _has_strings(a, b):
        return npb.add(_to_numpy(a), _to_numpy(b))
    return _torchify(a) + _torchify(b)


def mul(a: Any, b: Any) -> Any:
    if _has_strings(a, b):
        return npb.mul(_to_numpy(a), _to_numpy(b))
    return _torchify(a) * _torchify(b)


def matmul(a: Any, b: Any) -> Any:
    if _has_strings(a, b):
        return npb.matmul(_to_numpy(a), _to_numpy(b))
    return _torchify(a) @ _torchify(b)


def sum(a: Any, axis: int | None = None) -> Any:
    if _has_strings(a):
        return npb.sum(_to_numpy(a), axis=axis)
    return torch.sum(_torchify(a), dim=axis)


def stop(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach()
    return npb.stop(x)


def grad(fn: Callable[..., Any], wrt: int = 0) -> Callable[..., Any]:
    """Return a function computing ``∂fn/∂arg[wrt]`` using torch.autograd."""

    def _grad_fn(*args: Any) -> Any:
        if _has_strings(*args):
            raise RuntimeError("not differentiable")
        targs = []
        for i, a in enumerate(args):
            if isinstance(a, torch.Tensor):
                t = a.clone().detach().requires_grad_(i == wrt)
            else:
                t = torch.tensor(a, dtype=torch.float64, requires_grad=(i == wrt))
            targs.append(t)
        try:
            out = fn(*targs)
        except Exception as e:  # fallbacks may fail with type errors
            raise RuntimeError("not differentiable") from e
        if not isinstance(out, torch.Tensor):
            raise RuntimeError("not differentiable")
        g, = torch.autograd.grad(out, targs[wrt])
        return g

    return _grad_fn
