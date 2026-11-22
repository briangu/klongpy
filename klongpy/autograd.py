import contextlib

import numpy as np
from .core import KGLambda, KGCall, KGSym, KGFn, is_list


class AutogradError(Exception):
    """Base class for autograd-related errors."""
    pass


class AutogradChainBrokenError(AutogradError):
    """Raised when the gradient computation chain is broken."""

    def __init__(self, context, expected, actual, suggestion=None):
        self.context = context
        self.expected = expected
        self.actual = actual
        self.suggestion = suggestion
        msg = f"Autograd chain broken at {context}: expected {expected}, got {actual}."
        if suggestion:
            msg += f" {suggestion}"
        super().__init__(msg)


class NonScalarLossError(AutogradError):
    """Raised when the loss function returns a non-scalar value."""

    def __init__(self, shape):
        self.shape = shape
        super().__init__(
            f"Loss function must return a scalar, got shape {shape}. "
            "Use sum (+/) or mean (%#) to reduce to a scalar."
        )


def _get_float_dtype(backend):
    """Get the appropriate float dtype for the current backend."""
    # MPS doesn't support float64
    if hasattr(backend, 'supports_float64') and not backend.supports_float64():
        return np.float32
    return np.float64


def _scalar_value(x, backend):
    """Extract scalar value from various array/tensor types.

    Raises:
        NonScalarLossError: If x is not a scalar value.
    """
    x = backend.to_numpy(x) if backend.is_backend_array(x) else x
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return float(x.item())
        elif x.size == 1:
            return float(x.flat[0])
        else:
            raise NonScalarLossError(tuple(x.shape))
    return float(x)


def _to_func_input(x, backend, require_grad=False):
    """Convert numpy array to appropriate input type for function call.

    Args:
        x: Input array (numpy)
        backend: Backend provider
        require_grad: If True and backend supports autograd, create grad tensor.
                      For numeric gradient, this should be False.
    """
    if require_grad and backend.supports_autograd():
        return backend.create_grad_tensor(x)
    return x


def _invoke_fn(klong, fn, args):
    """Invoke a Klong function with the given arguments.

    Handles all function types uniformly:
    - KGSym, KGLambda: applied to args
    - KGFn, KGCall: extract inner function, apply it to args; a projection is
      applied as it is, so the arguments it already holds are kept
    - callable: call directly with args

    args are values and are not evaluated again.
    """
    if callable(fn) and not isinstance(fn, (KGSym, KGLambda, KGFn)):
        return fn(*args)
    if isinstance(fn, KGFn) and type(fn.args) is list and any(q is None for q in fn.args):
        # a projection, f(;3): its bound arguments belong to the call, so apply the projection itself
        return klong.apply(fn, list(args))
    inner = fn.a if isinstance(fn, KGFn) else fn
    return klong.apply(inner, list(args))


def _wide(klong):
    """The scope in which a numeric derivative is computed: the interpreter's floats are as wide as the backend's widest (float64
    on the CPU), whatever its own width is. A step of 1e-6 is lost in float32."""
    w = getattr(klong, "wide_floats", None)
    return w() if w is not None else contextlib.nullcontext()


def _derivative_result(value, backend, klong):
    # Return Klong's computed array after wide_floats restores the provider's
    # device and compute dtype. Direct NumPy helper calls retain float64.
    if klong is not None and value.ndim and backend.name != "numpy":
        dtype = getattr(backend, "_host_float", value.dtype)
        return backend.kg_asarray(value.astype(dtype, copy=False))
    return value


def numeric_grad(func, x, backend, eps=None, klong=None):
    """Compute numeric gradient of scalar-valued function (in float64 where the backend has it, see `_wide`)."""
    with _wide(klong):
        result = _numeric_grad(func, x, backend, eps)
    return _derivative_result(result, backend, klong)


def _numeric_grad(func, x, backend, eps=None):
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

    Uses PyTorch autograd when available (the torch backend), otherwise
    falls back to numeric differentiation.
    """
    backend = klong._backend
    call_fn = lambda v: _invoke_fn(klong, fn, [v])

    if backend.supports_autograd():
        return backend.compute_autograd(call_fn, x)
    else:
        return numeric_grad(call_fn, x, backend, klong=klong)


def numeric_jacobian(func, x, backend, eps=None, klong=None):
    """
    Compute Jacobian matrix of func at point x using finite differences.

    For f: R^n -> R^m, returns m x n matrix where J[i,j] = df_i/dx_j.

    Args:
        func: Callable that takes an array and returns an array
        x: Input point (array)
        backend: Backend provider
        eps: Step size for finite differences (default: 1e-6 or 1e-4 for float32)

    Returns:
        A NumPy matrix, or the selected provider's array for a Klong call.
    """
    with _wide(klong):
        result = _numeric_jacobian(func, x, backend, eps)
    return _derivative_result(result, backend, klong)


def _numeric_jacobian(func, x, backend, eps=None):
    float_dtype = _get_float_dtype(backend)
    if eps is None:
        eps = 1e-4 if float_dtype == np.float32 else 1e-6

    # Convert to numpy
    if backend.is_backend_array(x):
        x = backend.to_numpy(x)
    x = np.asarray(x, dtype=float_dtype).flatten()

    # Evaluate function at x to get output shape
    f0 = func(_to_func_input(x.copy(), backend))
    if backend.is_backend_array(f0):
        f0 = backend.to_numpy(f0)
    f0 = np.asarray(f0, dtype=float_dtype).flatten()

    n = len(x)  # Input dimension
    m = len(f0)  # Output dimension
    jacobian = np.zeros((m, n), dtype=float_dtype)

    for j in range(n):
        x_plus = x.copy()
        x_plus[j] += eps
        x_minus = x.copy()
        x_minus[j] -= eps

        f_plus = func(_to_func_input(x_plus, backend))
        f_minus = func(_to_func_input(x_minus, backend))

        if backend.is_backend_array(f_plus):
            f_plus = backend.to_numpy(f_plus)
        if backend.is_backend_array(f_minus):
            f_minus = backend.to_numpy(f_minus)

        f_plus = np.asarray(f_plus, dtype=float_dtype).flatten()
        f_minus = np.asarray(f_minus, dtype=float_dtype).flatten()

        jacobian[:, j] = (f_plus - f_minus) / (2 * eps)

    return jacobian


def jacobian_of_fn(klong, fn, x):
    """
    Compute Jacobian matrix of Klong function fn at point x.

    For f: R^n -> R^m, returns m x n matrix where J[i,j] = df_i/dx_j.

    Args:
        klong: KlongInterpreter instance
        fn: Function (KGSym, KGLambda, KGFn, KGCall, or callable)
        x: Input point

    Returns:
        Jacobian matrix
    """
    backend = klong._backend
    call_fn = lambda v: _invoke_fn(klong, fn, [v])

    if backend.supports_autograd():
        try:
            return backend.compute_jacobian(call_fn, x)
        except Exception:
            # Fall back to numeric if torch jacobian fails
            return numeric_jacobian(call_fn, x, backend=backend, klong=klong)
    else:
        return numeric_jacobian(call_fn, x, backend=backend, klong=klong)


def multi_jacobian_of_fn(klong, fn, param_syms):
    """
    Compute Jacobians for multiple parameters in one call.

    Args:
        klong: KlongInterpreter instance
        fn: Function (KGSym, KGLambda, KGFn, KGCall, or callable)
             Should be a niladic function that references the parameters
        param_syms: List of KGSym parameter symbols to differentiate with respect to

    Returns:
        List of Jacobian matrices, one per parameter
    """
    backend = klong._backend
    param_values = [klong[sym] for sym in param_syms]
    call_fn = lambda: _invoke_fn(klong, fn, [])

    jacobians = []
    for sym, val in zip(param_syms, param_values):
        original = klong[sym]

        def single_param_fn(v, s=sym, orig=original):
            """Wrapper that sets param to v, calls fn, restores param."""
            klong[s] = v
            try:
                return call_fn()
            finally:
                klong[s] = orig

        if backend.supports_autograd():
            try:
                jac = backend.compute_jacobian(single_param_fn, val)
            except Exception:
                jac = numeric_jacobian(single_param_fn, val, backend=backend, klong=klong)
        else:
            jac = numeric_jacobian(single_param_fn, val, backend=backend, klong=klong)

        # Restore original value after jacobian computation
        klong[sym] = original
        jacobians.append(jac)

    return jacobians


def multi_grad_of_fn(klong, fn, param_syms):
    """
    Compute gradients for multiple parameters in one call.

    Args:
        klong: KlongInterpreter instance
        fn: Loss function (KGSym, KGLambda, KGFn, KGCall, or callable)
             Should be a niladic function that references the parameters
        param_syms: List of KGSym parameter symbols to differentiate with respect to

    Returns:
        List of gradients, one per parameter
    """
    backend = klong._backend
    # Access context directly to avoid KGFnWrapper wrapping
    param_values = [klong[sym] for sym in param_syms]

    def call_fn_with_tensors(tensors):
        """Call the loss function with tensor values temporarily bound to symbols."""
        originals = {sym: klong[sym] for sym in param_syms}
        try:
            for sym, tensor in zip(param_syms, tensors):
                klong[sym] = tensor
            return _invoke_fn(klong, fn, [])
        finally:
            for sym, orig in originals.items():
                klong[sym] = orig

    if backend.supports_autograd():
        return backend.compute_multi_autograd(call_fn_with_tensors, param_values)
    else:
        # Fallback: compute numeric gradients one at a time
        grads = []
        for i, sym in enumerate(param_syms):
            def single_param_fn(v, idx=i):
                vals = list(param_values)
                vals[idx] = v
                return call_fn_with_tensors(vals)
            grads.append(numeric_grad(single_param_fn, param_values[i], backend, klong=klong))
        return grads


# ---- the gradient operators (what the host table's HostOps calls, and .grad / .jacobian) ----

def eval_dyad_grad(klong, a, b):
    """

        a∇b                                                    [Grad]

        Compute the numeric gradient of the monadic function ``b`` at ``a``
        using finite differences. Always uses numeric differentiation.

        For automatic differentiation, use the :> operator instead.
        ASCII form: .grad(b;a).

    """
    def call_fn(v):
        if isinstance(b, (KGSym, KGLambda, KGFn, KGCall)):
            return klong.apply(b, [v])
        return b(v)

    if isinstance(a, KGSym):
        orig = klong[a]

        def func(v):
            klong[a] = v
            try:
                return call_fn(v)
            finally:
                klong[a] = orig

        return numeric_grad(func, orig, klong._backend, klong=klong)
    else:
        return numeric_grad(call_fn, a, klong._backend, klong=klong)


def eval_dyad_jacobian(klong, a, b):
    """

        a∂b                                                  [Jacobian]

        Compute Jacobian matrix of function ``b`` at point ``a``.
        For f: R^n -> R^m, returns m x n matrix where J[i,j] = df_i/dx_j.

        Two modes based on what ``a`` contains:
        1. Single point: [1 2]∂f -> Jacobian at that point
        2. List of symbols: [w b]∂f -> [J_w J_b] (multi-param mode)

        In multi-param mode, ``b`` should be a niladic function
        that references the parameter symbols.

        Examples:
            [1 2]∂{[x@0^2 x@1^2]}  -->  [[2 0] [0 4]]
            [w b]∂f               -->  [J_w J_b] (multi-param mode)

    """
    if is_list(a) and len(a) > 0 and all(isinstance(p, KGSym) for p in a):
        return multi_jacobian_of_fn(klong, b, list(a))
    return jacobian_of_fn(klong, b, a)  # a is the point, b the function


def eval_dyad_autograd(klong, a, b):
    """

        a:>b                                                 [Autograd]

        Compute gradient of function ``a`` with respect to ``b``.

        Two modes based on what ``b`` contains:
        1. Single param/point: a:>x or a:>[1 2 3] -> gradient at that point
        2. List of symbols: a:>[w b] -> [grad_w grad_b] (multi-param mode)

        In multi-param mode, ``a`` should be a niladic function (loss)
        that references the parameter symbols.

        Examples:
            {x^2}:>3.0       -->  6.0      (derivative of x^2 at x=3)
            {x^3}:>2.0       -->  12.0     (derivative of x^3 at x=2)
            {+/x^2}:>[1 2 3] -->  [2 4 6]  (gradient of sum of squares)
            loss:>[w b]      -->  [grad_w grad_b]  (multi-param mode)

    """
    if is_list(b) and len(b) > 0 and all(isinstance(p, KGSym) for p in b):
        return multi_grad_of_fn(klong, a, list(b))
    return grad_of_fn(klong, a, b)
