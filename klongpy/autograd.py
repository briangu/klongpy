import numpy as np
from .core import KGLambda, KGCall, KGSym, KGFn


def numeric_grad(func, x, eps=1e-6):
    """Compute numeric gradient of scalar-valued function."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig = float(x[idx])
        x[idx] = orig + eps
        f_pos = func(x)
        x[idx] = orig - eps
        f_neg = func(x)
        grad[idx] = (f_pos - f_neg) / (2 * eps)
        x[idx] = orig
        it.iternext()
    return grad


def grad_of_fn(klong, fn, x):
    """Return gradient of Klong or Python function ``fn`` at ``x``."""
    def call_fn(v):
        if isinstance(fn, (KGSym, KGLambda)):
            return klong.call(KGCall(fn, [v], 1))
        elif isinstance(fn, KGCall):
            return klong.call(KGCall(fn.a, [v], fn.arity))
        elif isinstance(fn, KGFn):
            return klong.call(KGCall(fn.a, [v], fn.arity))
        else:
            return fn(v)
    return numeric_grad(call_fn, x)
