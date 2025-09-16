import numbers
import numpy as np
from .core import KGLambda, KGCall, KGSym, KGFn


class AutodiffNotSupported(Exception):
    """Raised when forward-mode autodiff cannot handle an operation."""


_UNARY_DERIVATIVES = {
    np.negative: lambda x: -1.0,
    np.positive: lambda x: 1.0,
    np.sin: np.cos,
    np.cos: lambda x: -np.sin(x),
    np.tan: lambda x: 1.0 / (np.cos(x) ** 2),
    np.exp: np.exp,
    np.expm1: np.exp,
    np.log: lambda x: 1.0 / x,
    np.log1p: lambda x: 1.0 / (1.0 + x),
    np.sqrt: lambda x: 0.5 / np.sqrt(x),
    np.square: lambda x: 2.0 * x,
    np.reciprocal: lambda x: -1.0 / (x ** 2),
    np.sinh: np.cosh,
    np.cosh: np.sinh,
    np.tanh: lambda x: 1.0 / (np.cosh(x) ** 2),
    np.abs: np.sign,
    np.log10: lambda x: 1.0 / (x * np.log(10.0)),
    np.log2: lambda x: 1.0 / (x * np.log(2.0)),
}

_BINARY_DERIVATIVES = {
    np.add: lambda x, y: (1.0, 1.0),
    np.subtract: lambda x, y: (1.0, -1.0),
    np.multiply: lambda x, y: (y, x),
    np.divide: lambda x, y: (1.0 / y, -x / (y * y)),
    np.true_divide: lambda x, y: (1.0 / y, -x / (y * y)),
}

_POWER_UFUNCS = {np.power, np.float_power}


class Dual:
    """Simple dual number for forward-mode autodiff on scalar inputs."""

    __slots__ = ("value", "grad")
    __array_priority__ = 1000

    def __init__(self, value, grad=0.0):
        self.value = float(value)
        self.grad = float(grad)

    @staticmethod
    def _coerce(other):
        if isinstance(other, Dual):
            return other
        if np.isarray(other):
            raise AutodiffNotSupported("array operands are not supported by Dual")
        if isinstance(other, numbers.Real):
            return Dual(other, 0.0)
        try:
            return Dual(float(other), 0.0)
        except (TypeError, ValueError) as exc:
            raise AutodiffNotSupported(
                f"unsupported operand of type {type(other)!r}"
            ) from exc

    def __array__(self, dtype=None):
        return np.array(self.value, dtype=dtype)

    def __float__(self):
        return float(self.value)

    def __neg__(self):
        return Dual(-self.value, -self.grad)

    def __pos__(self):
        return Dual(+self.value, +self.grad)

    def __add__(self, other):
        other = self._coerce(other)
        return Dual(self.value + other.value, self.grad + other.grad)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._coerce(other)
        return Dual(self.value - other.value, self.grad - other.grad)

    def __rsub__(self, other):
        other = self._coerce(other)
        return Dual(other.value - self.value, other.grad - self.grad)

    def __mul__(self, other):
        other = self._coerce(other)
        grad = self.grad * other.value + other.grad * self.value
        return Dual(self.value * other.value, grad)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = self._coerce(other)
        if other.value == 0.0:
            raise AutodiffNotSupported("division by zero is not supported")
        grad = (self.grad * other.value - other.grad * self.value) / (other.value ** 2)
        return Dual(self.value / other.value, grad)

    def __rtruediv__(self, other):
        other = self._coerce(other)
        return other.__truediv__(self)

    def __pow__(self, other):
        other = self._coerce(other)
        result = self.value ** other.value
        grad = self.grad * other.value * (self.value ** (other.value - 1))
        if other.grad != 0.0:
            if self.value <= 0.0:
                raise AutodiffNotSupported(
                    "differentiating w.r.t. the exponent requires a positive base"
                )
            grad += other.grad * result * np.log(self.value)
        return Dual(result, grad)

    def __rpow__(self, other):
        other = self._coerce(other)
        return other.__pow__(self)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            raise AutodiffNotSupported(f"ufunc method {method!r} is not supported")
        if kwargs.get("out") is not None:
            raise AutodiffNotSupported("ufunc out parameter is not supported")
        if "where" in kwargs and not np.all(kwargs["where"]):
            raise AutodiffNotSupported("where argument is not supported")

        values = [x.value if isinstance(x, Dual) else x for x in inputs]
        grads = [x.grad if isinstance(x, Dual) else 0.0 for x in inputs]

        try:
            result = getattr(ufunc, method)(*values, **kwargs)
        except TypeError as exc:
            raise AutodiffNotSupported from exc

        try:
            if ufunc in _UNARY_DERIVATIVES:
                derivative = _UNARY_DERIVATIVES[ufunc](values[0])
                grad = grads[0] * derivative
            elif ufunc in _BINARY_DERIVATIVES:
                dfdx, dfdy = _BINARY_DERIVATIVES[ufunc](values[0], values[1])
                grad = grads[0] * dfdx + grads[1] * dfdy
            elif ufunc in _POWER_UFUNCS:
                base, exponent = values
                dfdx = exponent * (base ** (exponent - 1))
                grad = grads[0] * dfdx
                if grads[1] != 0.0:
                    if base <= 0.0:
                        raise AutodiffNotSupported(
                            "differentiating power w.r.t. exponent requires positive base"
                        )
                    grad += grads[1] * (base ** exponent) * np.log(base)
            else:
                raise AutodiffNotSupported(f"ufunc {ufunc.__name__} is not supported")
        except ZeroDivisionError as exc:
            raise AutodiffNotSupported from exc

        return Dual(result, grad)


def numeric_grad(func, x, eps=1e-6):
    """Compute numeric gradient of scalar-valued function."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
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


def autodiff_grad(func, x):
    """Compute gradient using forward-mode autodiff for scalar ``x``."""
    if np.ndim(x) != 0:
        raise AutodiffNotSupported("autodiff only supports scalar inputs")
    value = float(np.asarray(x))
    dual = Dual(value, 1.0)
    try:
        result = func(dual)
    except AutodiffNotSupported:
        raise
    except Exception as exc:
        raise AutodiffNotSupported from exc
    if isinstance(result, Dual):
        return np.asarray(result.grad, dtype=float)
    raise AutodiffNotSupported("function did not return a Dual value")


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

    try:
        return autodiff_grad(call_fn, x)
    except AutodiffNotSupported:
        return numeric_grad(call_fn, x)
