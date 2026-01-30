import numpy as np
from .core import KGLambda, KGCall, KGSym, KGFn
from .backend import get_default_backend, to_numpy


def _get_float_dtype(backend=None):
    """Get the appropriate float dtype for the current backend."""
    if backend is None:
        backend = get_default_backend()
    # MPS doesn't support float64
    if hasattr(backend, 'supports_float64') and not backend.supports_float64():
        return np.float32
    return np.float64


def _scalar_value(x, backend=None):
    """Extract scalar value from various array/tensor types."""
    if backend is None:
        backend = get_default_backend()
    x = backend.to_numpy(x) if backend.is_backend_array(x) else x
    if isinstance(x, np.ndarray):
        return float(x.item()) if x.ndim == 0 else float(x)
    return float(x)


def _to_func_input(x, backend=None):
    """Convert numpy array to appropriate input type for function call."""
    if backend is None:
        backend = get_default_backend()
    if backend.supports_autograd():
        return backend.create_grad_tensor(x)
    return x


def numeric_grad(func, x, eps=None, backend=None):
    """Compute numeric gradient of scalar-valued function."""
    if backend is None:
        backend = get_default_backend()

    # Get appropriate float dtype
    float_dtype = _get_float_dtype(backend)

    # Use larger epsilon for float32 to maintain precision
    if eps is None:
        eps = 1e-4 if float_dtype == np.float32 else 1e-6

    # Convert backend tensors to numpy for gradient computation
    if backend.is_backend_array(x):
        x = backend.to_numpy(x)
    x = np.asarray(x, dtype=float_dtype)

    grad = np.zeros_like(x, dtype=float_dtype)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig = float(x[idx])
        x[idx] = orig + eps
        f_pos = _scalar_value(func(_to_func_input(x.copy(), backend)), backend)
        x[idx] = orig - eps
        f_neg = _scalar_value(func(_to_func_input(x.copy(), backend)), backend)
        grad[idx] = (f_pos - f_neg) / (2 * eps)
        x[idx] = orig
        it.iternext()
    return grad


def grad_of_fn(klong, fn, x):
    """
    Return gradient of Klong or Python function ``fn`` at ``x``.

    Uses PyTorch autograd when available (USE_TORCH=1), otherwise
    falls back to numeric differentiation.
    """
    backend = klong._backend

    def call_fn(v):
        if isinstance(fn, (KGSym, KGLambda)):
            return klong.call(KGCall(fn, [v], 1))
        elif isinstance(fn, KGCall):
            return klong.call(KGCall(fn.a, [v], fn.arity))
        elif isinstance(fn, KGFn):
            return klong.call(KGCall(fn.a, [v], fn.arity))
        else:
            return fn(v)

    if backend.supports_autograd():
        return backend.compute_autograd(call_fn, x)
    else:
        return numeric_grad(call_fn, x, backend=backend)


def torch_autograd(func, x):
    """
    Compute gradient using PyTorch autograd.

    This is a convenience wrapper around the backend's compute_autograd method.

    Parameters
    ----------
    func : callable
        A function that takes a tensor and returns a scalar tensor.
    x : array-like
        The point at which to compute the gradient.

    Returns
    -------
    Tensor
        The gradient of func at x.
    """
    backend = get_default_backend()
    if not backend.supports_autograd():
        raise RuntimeError("PyTorch autograd requires torch backend (USE_TORCH=1)")
    return backend.compute_autograd(func, x)


def autograd_of_fn(klong, fn, x):
    """
    Compute gradient using PyTorch autograd when available.

    This function uses PyTorch's automatic differentiation when available,
    otherwise falls back to numeric gradient computation.

    Parameters
    ----------
    klong : KlongInterpreter
        The Klong interpreter instance.
    fn : KGFn, KGLambda, KGSym, or callable
        The function to differentiate.
    x : array-like
        The point at which to compute the gradient.

    Returns
    -------
    array-like
        The gradient of fn at x.
    """
    backend = klong._backend

    def call_fn(v):
        if isinstance(fn, (KGSym, KGLambda)):
            return klong.call(KGCall(fn, [v], 1))
        elif isinstance(fn, KGCall):
            return klong.call(KGCall(fn.a, [v], fn.arity))
        elif isinstance(fn, KGFn):
            return klong.call(KGCall(fn.a, [v], fn.arity))
        else:
            return fn(v)

    if backend.supports_autograd():
        return backend.compute_autograd(call_fn, x)
    else:
        return numeric_grad(call_fn, x, backend=backend)
