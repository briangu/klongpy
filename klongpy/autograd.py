import numpy as np
from .core import KGLambda, KGCall, KGSym, KGFn
from .backend import use_torch, TorchUnsupportedDtypeError, to_numpy

if use_torch:
    import torch


def _get_float_dtype():
    """Get the appropriate float dtype for the current backend."""
    if use_torch:
        from .backend import get_default_backend
        backend = get_default_backend()
        # MPS doesn't support float64
        if hasattr(backend, 'supports_float64') and not backend.supports_float64():
            return np.float32
    return np.float64


def _scalar_value(x):
    """Extract scalar value from various array/tensor types."""
    x = to_numpy(x)
    if isinstance(x, np.ndarray):
        return float(x.item()) if x.ndim == 0 else float(x)
    return float(x)


def _to_func_input(x):
    """Convert numpy array to appropriate input type for function call."""
    if use_torch:
        # Always convert to tensor when in torch mode
        return torch.tensor(x, dtype=torch.float32)
    return x


def numeric_grad(func, x, eps=None):
    """Compute numeric gradient of scalar-valued function."""
    # Get appropriate float dtype
    float_dtype = _get_float_dtype()

    # Use larger epsilon for float32 to maintain precision
    if eps is None:
        eps = 1e-4 if float_dtype == np.float32 else 1e-6

    # Convert torch tensors to numpy for gradient computation
    x = to_numpy(x) if use_torch else x
    x = np.asarray(x, dtype=float_dtype)

    grad = np.zeros_like(x, dtype=float_dtype)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig = float(x[idx])
        x[idx] = orig + eps
        f_pos = _scalar_value(func(_to_func_input(x.copy())))
        x[idx] = orig - eps
        f_neg = _scalar_value(func(_to_func_input(x.copy())))
        grad[idx] = (f_pos - f_neg) / (2 * eps)
        x[idx] = orig
        it.iternext()
    return grad


def torch_autograd(func, x):
    """
    Compute gradient using PyTorch autograd.

    Parameters
    ----------
    func : callable
        A function that takes a tensor and returns a scalar tensor.
    x : array-like
        The point at which to compute the gradient.

    Returns
    -------
    torch.Tensor
        The gradient of func at x.
    """
    if not use_torch:
        raise RuntimeError("PyTorch autograd requires USE_TORCH=1")

    # Convert input to torch tensor with gradient tracking
    if isinstance(x, torch.Tensor):
        x_tensor = x.clone().detach().float().requires_grad_(True)
    elif isinstance(x, np.ndarray):
        x_tensor = torch.from_numpy(x.astype(np.float64)).float().requires_grad_(True)
    else:
        x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    # Compute the function value
    y = func(x_tensor)

    # Ensure y is a scalar
    if y.numel() != 1:
        raise ValueError(f"Function must return a scalar, got shape {y.shape}")

    # Compute gradient
    y.backward()

    return x_tensor.grad


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


def autograd_of_fn(klong, fn, x):
    """
    Compute gradient using PyTorch autograd when available.

    This function uses PyTorch's automatic differentiation when USE_TORCH=1,
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
    def call_fn(v):
        if isinstance(fn, (KGSym, KGLambda)):
            return klong.call(KGCall(fn, [v], 1))
        elif isinstance(fn, KGCall):
            return klong.call(KGCall(fn.a, [v], fn.arity))
        elif isinstance(fn, KGFn):
            return klong.call(KGCall(fn.a, [v], fn.arity))
        else:
            return fn(v)

    if use_torch:
        return torch_autograd(call_fn, x)
    else:
        return numeric_grad(call_fn, x)
