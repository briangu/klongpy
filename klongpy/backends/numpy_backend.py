"""
NumPy backend provider for KlongPy.

This is the default backend that supports all Klong operations including
string manipulation and object dtype arrays.
"""
import operator
import types
import numpy as np

from .base import BackendProvider

# numpy 2.x moved VisibleDeprecationWarning to numpy.exceptions
from numpy.exceptions import VisibleDeprecationWarning as NumpyVisibleDeprecationWarning


class KGChar(str):
    """Character type for Klong."""
    pass


def _make_numpy_namespace():
    """The numpy module as seen through the numpy backend.

    A separate module object holding numpy's own attributes (including its lazy
    ``__getattr__`` for submodules such as ``random``) plus ``isarray``, which
    the backend API needs.  Lookups cost the same as on numpy itself, and the
    real ``numpy`` module is never modified.
    """
    ns = types.ModuleType('numpy')
    ns.__dict__.update(vars(np))
    ns.isarray = lambda x: isinstance(x, np.ndarray)
    return ns


_NUMPY_NAMESPACE = _make_numpy_namespace()

_NUMPY_VALUES = (np.ndarray, np.generic)


def _compiled_div(a, b):
    """Division for compiled expressions.

    A zero divisor raises ZeroDivisionError, for arrays as well as scalars, so
    the compiled path falls back to the interpreter, which returns :undefined
    per element.  Without a zero the result is np.divide, which
    is what the interpreter computes.
    """
    if isinstance(b, _NUMPY_VALUES):
        if np.any(b == 0):
            raise ZeroDivisionError("division by zero")
    elif b == 0:
        raise ZeroDivisionError("division by zero")
    return np.divide(a, b)


_EXACT_FLOAT_INT = 2 ** 53


def _kind_of(x):
    dt = getattr(x, 'dtype', None)
    if dt is None:
        return 'i' if type(x) is int else 'f'
    return dt.kind


def _small_ints(x):
    if isinstance(x, np.ndarray):
        return x.size == 0 or (x.max() <= _EXACT_FLOAT_INT and x.min() >= -_EXACT_FLOAT_INT)
    return -_EXACT_FLOAT_INT <= x <= _EXACT_FLOAT_INT


def _equal(x, y):
    """`x=y` for atoms, as the interpreter's: the truth value is made per atom, so lists stay integer
    lists, and an integer and a float are compared exactly (numpy would round the integer to a float)."""
    dx = getattr(x, 'dtype', None)
    dy = getattr(y, 'dtype', None)
    if dx is not dy or (dx is None and type(x) is not type(y)):
        kx = _kind_of(x)
        ky = _kind_of(y)
        if kx != ky and (kx == 'i' and ky == 'f' and not _small_ints(x) or kx == 'f' and ky == 'i' and not _small_ints(y)):
            return (np.asarray(x, dtype=object) == np.asarray(y, dtype=object)) * 1
    return (x == y) * 1


_INT64_MIN = -2 ** 63
_INT64_MAX = 2 ** 63 - 1
_PY_NUMBERS = (int, float)


def _arithmetic(py_op, ufunc, atomic):
    """`x+y`, `x-y` or `x*y` for a compiled expression.

    Arrays and anything else go through the atomic dyad with the numpy ufunc,
    as in the interpreter. Two Python numbers (the loop counters of compiled
    scalar code) are combined directly; an integer result that does not fit
    int64 is left to the ufunc, which wraps it as the interpreter does.
    """
    def f(x, y):
        if x.__class__ in _PY_NUMBERS and y.__class__ in _PY_NUMBERS:
            r = py_op(x, y)
            if r.__class__ is float or _INT64_MIN <= r <= _INT64_MAX:
                return r  # a Python number again, so a loop counter stays eligible for the compiled path
        return atomic(x, y, ufunc)
    return f


def _negation(negate, ufunc):
    """`-x` for a compiled expression: np.negative for numbers and numeric arrays, as the interpreter does;
    the interpreter's own function for anything else (object arrays)."""
    def f(x):
        c = x.__class__
        if c is float or (c is int and _INT64_MIN <= x <= _INT64_MAX) or (c is np.ndarray and x.dtype.kind != 'O'):
            return ufunc(x)
        return negate(x)
    return f


def _accumulating(ufunc):
    """`op\\x` for a compiled expression: the interpreter returns an atom unchanged."""
    accumulate = ufunc.accumulate

    def scan(x):
        return x if np.ndim(x) == 0 else accumulate(x)
    return scan


def _reducing(ufunc):
    """`op/x` for a compiled expression: the interpreter returns an atom or an empty list unchanged."""
    reduce = ufunc.reduce

    def over(x):
        shape = getattr(x, 'shape', ())
        return reduce(x) if shape and shape[0] else x
    return over


_INT64_LIMIT = float(2 ** 63)


def _beyond_int64(arr):
    """True if a float array has a value of magnitude 2**63 or more (the only floats that can come from big integers)."""
    return arr.max() >= _INT64_LIMIT or arr.min() <= -_INT64_LIMIT


def _has_big_int(a):
    """A list of integers, one of them outside int64 (a float in the list makes it a float list, as usual)."""
    return all(type(x) is int or isinstance(x, np.integer) for x in a) and any(not -2 ** 63 <= x < 2 ** 63 for x in a)


class NumpyBackendProvider(BackendProvider):
    """NumPy-based backend provider."""

    def __init__(self, device=None):
        if device is not None:
            raise ValueError("Backend 'numpy' does not support device selection")
        self._np = _NUMPY_NAMESPACE

    @property
    def name(self) -> str:
        return 'numpy'

    @property
    def np(self):
        return self._np

    def supports_object_dtype(self) -> bool:
        return True

    def supports_strings(self) -> bool:
        return True

    def supports_float64(self) -> bool:
        return True

    def is_array(self, x) -> bool:
        return isinstance(x, np.ndarray)

    def is_backend_array(self, x) -> bool:
        return False  # numpy arrays are the base case, not a "backend" type

    def get_dtype_kind(self, arr) -> str:
        if hasattr(arr, 'dtype') and hasattr(arr.dtype, 'kind'):
            return arr.dtype.kind
        return None

    def to_numpy(self, x):
        # Already numpy, just return as-is
        return x

    def is_scalar_integer(self, x) -> bool:
        if isinstance(x, np.ndarray) and x.ndim == 0:
            return np.issubdtype(x.dtype, np.integer)
        return False

    def is_scalar_float(self, x) -> bool:
        if isinstance(x, np.ndarray) and x.ndim == 0:
            return np.issubdtype(x.dtype, np.floating)
        return False

    def argsort(self, a, descending=False):
        """Return indices that would sort the array."""
        indices = np.argsort(a)
        if descending:
            indices = indices[::-1].copy()
        return indices

    def array_size(self, a):
        """Get the total number of elements in an array."""
        if hasattr(a, 'size'):
            return a.size
        return len(a) if hasattr(a, '__len__') else 1

    def safe_equal(self, x, y):
        """Compare two values for equality."""
        return np.asarray(x, dtype=object) == np.asarray(y, dtype=object)

    def to_int_array(self, a):
        """Convert array to integer type."""
        return np.asarray(a, dtype=int) if self.is_array(a) else int(a)

    def power(self, a, b):
        """Compute a^b, returning integer if result is whole number."""
        with np.errstate(divide='ignore'):
            r = np.power(float(a) if isinstance(a, (int, np.integer)) else a, b)
        return r

    def str_to_char_array(self, s):
        """Convert string to character array."""
        return self._np.asarray([KGChar(x) for x in s], dtype=object)

    def compile_expr_ir(self, ir, var_syms):
        """Compile IR tree to a callable using numpy operations."""
        source = self._ir_to_source(ir)
        if source is None:
            return None

        param_names = list(self._collect_params(ir))
        fn_source = f"def _expr({', '.join(param_names)}): return {source}"
        # The compiled operators are the ones the interpreter applies (np.add,
        # ... and the interpreter's own ^ and negate), so both paths give the
        # same values, dtypes and errors.
        xp = self._np
        ns = {
            'np': xp,
            '_atomic': self.vec_fn2,
            '_add': _arithmetic(operator.add, xp.add, self.vec_fn2),
            '_sub': _arithmetic(operator.sub, xp.subtract, self.vec_fn2),
            '_mul': _arithmetic(operator.mul, xp.multiply, self.vec_fn2),
            '_div': _compiled_div, '_pow': self.interpreter_power(), '_neg': _negation(self.interpreter_negate(), xp.negative),
            '_eq': _equal, '_gt': xp.greater, '_lt': xp.less,
            '_cumsum': _accumulating(xp.add), '_cumprod': _accumulating(xp.multiply),
            '_rsum': _reducing(xp.add), '_rprod': _reducing(xp.multiply),
            '_rmax': _reducing(xp.maximum), '_rmin': _reducing(xp.minimum),
        }
        try:
            exec(fn_source, ns)
        except Exception:
            return None
        return (ns['_expr'], var_syms)

    def _ir_to_source(self, ir):
        """Convert IR tree to Python source string with numpy operations."""
        node_type = ir[0]

        if node_type == 'literal':
            return repr(ir[1])

        if node_type == 'var':
            return ir[1]

        if node_type == 'binop':
            op, left, right = ir[1], ir[2], ir[3]
            l = self._ir_to_source(left)
            r = self._ir_to_source(right)
            if l is None or r is None:
                return None
            py_op = {'+': '_add', '-': '_sub', '*': '_mul', '%': '_div', '^': '_pow'}.get(op)
            if py_op is None:
                return None
            if py_op in ('_add', '_sub', '_mul'):
                return f'{py_op}({l}, {r})'
            return f'_atomic({l}, {r}, {py_op})'

        if node_type == 'cmp':
            op, left, right = ir[1], ir[2], ir[3]
            l = self._ir_to_source(left)
            r = self._ir_to_source(right)
            if l is None or r is None:
                return None
            py_cmp = {'=': '_eq', '>': '_gt', '<': '_lt'}.get(op)
            if py_cmp is None:
                return None
            return f'(_atomic({l}, {r}, {py_cmp})*1)'

        if node_type == 'negate':
            child = self._ir_to_source(ir[1])
            if child is None:
                return None
            return f'_neg({child})'

        if node_type == 'reduce':
            op, arg = ir[1], ir[2]
            arg_src = self._ir_to_source(arg)
            if arg_src is None:
                return None
            method = {'+': '_rsum', '*': '_rprod', '|': '_rmax', '&': '_rmin'}.get(op)
            if method is None:
                return None
            return f'{method}({arg_src})'

        if node_type == 'scan':
            op, arg = ir[1], ir[2]
            arg_src = self._ir_to_source(arg)
            if arg_src is None:
                return None
            method = {'+': '_cumsum', '*': '_cumprod'}.get(op)
            if method is None:
                return None  # |\ and &\ not supported in numpy
            return f'{method}({arg_src})'

        return None

    def kg_asarray(self, a):
        """Convert input to numpy array, handling strings and jagged/nested data."""
        if isinstance(a, str):
            return self.str_to_char_array(a)
        try:
            arr = self._np.asarray(a)
            if arr.dtype.kind not in ['O', 'i', 'f']:
                raise ValueError
            if arr.dtype.kind == 'f' and type(a) is list and arr.size and _beyond_int64(arr) and _has_big_int(a):
                raise ValueError  # numpy made floats of [2**63 1]: keep the integers exact
        except (NumpyVisibleDeprecationWarning, ValueError):
            try:
                arr = self._np.asarray(a, dtype=object)
            except ValueError:
                arr = [x.tolist() if self.is_array(x) else x for x in a]
                arr = self._np.asarray(arr, dtype=object)
            if arr.ndim == 0:
                # A scalar outside int64 (for example 2**63): keep the Python value.
                return arr
            arr = self._np.asarray(
                [self.kg_asarray(x) if isinstance(x, list) else x for x in arr],
                dtype=object
            )
        return arr
