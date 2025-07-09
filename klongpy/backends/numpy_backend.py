"""NumPy backend with minimal reverse-mode autodiff."""

from __future__ import annotations

import numpy as np
from typing import Any, Callable, Iterable, Optional


class Tensor:
    """Simple tensor wrapper for reverse-mode autodiff."""

    def __init__(self, data: np.ndarray, _children: Iterable[Tensor] = (), requires_grad: bool = False):
        self.data = np.asarray(data)
        if self.data.dtype == object:
            raise TypeError("object dtype not supported")
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = np.zeros_like(self.data, dtype=self.data.dtype) if requires_grad else None
        self._prev = set(_children)
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Tensor(data={self.data}, grad={self.grad})"


def array(obj: Any, *, dtype: Any | None = None, requires_grad: bool = False) -> Any:
    """Create an array or tensor."""
    if isinstance(obj, Tensor):
        data = obj.data if dtype is None else obj.data.astype(dtype)
        return Tensor(data, requires_grad=requires_grad or obj.requires_grad)
    arr = np.asarray(obj, dtype=dtype)
    if arr.dtype == object:
        raise TypeError("object dtype not supported")
    return Tensor(arr, requires_grad=requires_grad) if requires_grad else arr


def _ensure_tensor(x: Any) -> Tensor:
    if isinstance(x, Tensor):
        return x
    arr = np.asarray(x)
    if arr.dtype == object:
        raise TypeError("object dtype not supported")
    return Tensor(arr)


def _broadcast_grad(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    g = grad
    while len(g.shape) > len(shape):
        g = g.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1:
            g = g.sum(axis=i, keepdims=True)
    return g


def add(a: Any, b: Any) -> Any:
    """Element-wise addition."""
    if not isinstance(a, Tensor) and not isinstance(b, Tensor):
        return np.add(a, b)
    ta, tb = _ensure_tensor(a), _ensure_tensor(b)
    out = Tensor(ta.data + tb.data, (ta, tb), requires_grad=ta.requires_grad or tb.requires_grad)

    def _backward() -> None:
        if out.grad is None:
            return
        if ta.requires_grad:
            ta.grad += _broadcast_grad(out.grad, ta.data.shape)
        if tb.requires_grad:
            tb.grad += _broadcast_grad(out.grad, tb.data.shape)

    out._backward = _backward
    return out


def mul(a: Any, b: Any) -> Any:
    """Element-wise multiplication."""
    if not isinstance(a, Tensor) and not isinstance(b, Tensor):
        return np.multiply(a, b)
    ta, tb = _ensure_tensor(a), _ensure_tensor(b)
    out = Tensor(ta.data * tb.data, (ta, tb), requires_grad=ta.requires_grad or tb.requires_grad)

    def _backward() -> None:
        if out.grad is None:
            return
        if ta.requires_grad:
            ta.grad += _broadcast_grad(out.grad * tb.data, ta.data.shape)
        if tb.requires_grad:
            tb.grad += _broadcast_grad(out.grad * ta.data, tb.data.shape)

    out._backward = _backward
    return out


def matmul(a: Any, b: Any) -> Any:
    """Matrix multiplication."""
    if not isinstance(a, Tensor) and not isinstance(b, Tensor):
        return np.matmul(a, b)
    ta, tb = _ensure_tensor(a), _ensure_tensor(b)
    out = Tensor(ta.data @ tb.data, (ta, tb), requires_grad=ta.requires_grad or tb.requires_grad)

    def _backward() -> None:
        if out.grad is None:
            return
        if ta.requires_grad:
            ta.grad += out.grad @ tb.data.T
        if tb.requires_grad:
            tb.grad += ta.data.T @ out.grad

    out._backward = _backward
    return out


def sum(a: Any, axis: int | None = None) -> Any:
    """Sum of elements."""
    if not isinstance(a, Tensor):
        return np.sum(a, axis=axis)
    out = Tensor(np.sum(a.data, axis=axis), (a,), requires_grad=a.requires_grad)

    def _backward() -> None:
        if out.grad is None or not a.requires_grad:
            return
        grad = out.grad
        if axis is None:
            grad = np.broadcast_to(grad, a.data.shape)
        else:
            grad = np.expand_dims(grad, axis)
            grad = np.broadcast_to(grad, a.data.shape)
        a.grad += grad

    out._backward = _backward
    return out


def stop(x: Any) -> Any:
    """Detach ``x`` from the autograd graph."""
    if isinstance(x, Tensor):
        return Tensor(x.data.copy(), requires_grad=False)
    return x


def grad(fn: Callable[..., Any], wrt: int = 0) -> Callable[..., Any]:
    """Return a function computing ``∂fn/∂arg[wrt]``."""

    def _grad_fn(*args: Any) -> Any:
        targs = []
        for i, a in enumerate(args):
            t = array(a, requires_grad=(i == wrt))
            targs.append(t)
        out = fn(*targs)
        if not isinstance(out, Tensor):
            raise RuntimeError("not differentiable")
        out.grad = np.ones_like(out.data)
        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build(v: Tensor) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(out)
        for node in reversed(topo):
            node._backward()
        return targs[wrt].grad

    return _grad_fn
