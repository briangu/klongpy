"""Torch backend using NumPy fallback for strings."""

from __future__ import annotations

from typing import Any, Callable

try:
    import torch
except Exception as e:  # pragma: no cover - optional dependency
    raise ImportError("torch backend requires the 'torch' package") from e

import numpy as np
from . import numpy_backend as npb

def _to_numpy(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def array(obj: Any, *, dtype: Any | None = None, requires_grad: bool = False) -> Any:
    """Create a torch tensor or numpy array."""
    t = _to_tensor(obj, dtype=dtype, requires_grad=requires_grad)
    if t is None:
        return npb.array(obj, dtype=dtype, requires_grad=requires_grad)
    return t


def _to_tensor(x: Any, *, dtype: Any | None = None, requires_grad: bool = False) -> torch.Tensor | None:
    """Return a ``torch.Tensor`` if possible else ``None``."""
    if isinstance(x, torch.Tensor):
        t = x if dtype is None else x.to(dtype)
        if requires_grad:
            t = t.clone().detach().requires_grad_(True)
        return t
    try:
        return torch.tensor(x, dtype=dtype, requires_grad=requires_grad)
    except Exception:
        return None


def add(a: Any, b: Any) -> Any:
    ta = _to_tensor(a)
    tb = _to_tensor(b)
    if ta is None or tb is None:
        return npb.add(_to_numpy(a), _to_numpy(b))
    return ta + tb


def mul(a: Any, b: Any) -> Any:
    ta = _to_tensor(a)
    tb = _to_tensor(b)
    if ta is None or tb is None:
        return npb.mul(_to_numpy(a), _to_numpy(b))
    return ta * tb


def matmul(a: Any, b: Any) -> Any:
    ta = _to_tensor(a)
    tb = _to_tensor(b)
    if ta is None or tb is None:
        return npb.matmul(_to_numpy(a), _to_numpy(b))
    return ta @ tb


def sum(a: Any, axis: int | None = None) -> Any:
    ta = _to_tensor(a)
    if ta is None:
        return npb.sum(_to_numpy(a), axis=axis)
    return torch.sum(ta, dim=axis)


def stop(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach()
    return npb.stop(x)


def grad(fn: Callable[..., Any], wrt: int = 0) -> Callable[..., Any]:
    """Return a function computing ``∂fn/∂arg[wrt]`` using torch.autograd."""

    def _grad_fn(*args: Any) -> Any:
        targs = []
        for i, a in enumerate(args):
            t = _to_tensor(a, dtype=torch.float64, requires_grad=(i == wrt))
            if t is None:
                raise RuntimeError("not differentiable")
            if not isinstance(a, torch.Tensor):
                t = t  # type: ignore[assignment]
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
