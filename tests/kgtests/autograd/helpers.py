from klongpy import backend
import numpy as np
from tests.utils import to_numpy


# simple function used by the ∂ example
def square(x):
    return x * x


def _apply_grad(fn, x, backend_name="numpy"):
    backend.set_backend(backend_name)
    b = backend.current()
    g = b.grad(fn)
    out = g(b.array(x, requires_grad=True))
    out = to_numpy(out)
    return float(out) if np.ndim(out) == 0 else out


# expose a del-symbol helper for Klong tests
globals()["∂"] = _apply_grad


def scalarSquareGrad(x, backend_name="numpy"):
    backend.set_backend(backend_name)
    b = backend.current()

    def f(t):
        return b.mul(t, t)

    g = b.grad(f)
    out = g(b.array(x, requires_grad=True))
    out = to_numpy(out)
    return float(out) if np.ndim(out) == 0 else out


def vectorElemwiseGrad(x, backend_name="numpy"):
    backend.set_backend(backend_name)
    b = backend.current()

    def f(t):
        return b.sum(b.mul(b.add(t, 1), b.add(t, 2)))

    g = b.grad(f)
    out = g(b.array(x, requires_grad=True))
    out = to_numpy(out)
    return out.tolist() if isinstance(out, np.ndarray) else out


def mixedGradX(x, y, backend_name="numpy"):
    backend.set_backend(backend_name)
    b = backend.current()

    def f(a, b_):
        return b.sum(b.mul(a, b_))

    g = b.grad(f, wrt=0)
    out = g(b.array(x, requires_grad=True), b.array(y, requires_grad=True))
    out = to_numpy(out)
    return out.tolist() if isinstance(out, np.ndarray) else out


def mixedGradY(x, y, backend_name="numpy"):
    backend.set_backend(backend_name)
    b = backend.current()

    def f(a, b_):
        return b.sum(b.mul(a, b_))

    g = b.grad(f, wrt=1)
    out = g(b.array(x, requires_grad=True), b.array(y, requires_grad=True))
    out = to_numpy(out)
    return out.tolist() if isinstance(out, np.ndarray) else out


def stopGrad(x, backend_name="numpy"):
    backend.set_backend(backend_name)
    b = backend.current()

    def f(t):
        return b.sum(b.mul(b.stop(t), t))

    g = b.grad(f)
    out = g(b.array(x, requires_grad=True))
    out = to_numpy(out)
    return out.tolist() if isinstance(out, np.ndarray) else out
