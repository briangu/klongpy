"""
NumPy backend provider for KlongPy.

This is the default backend that supports all Klong operations including
string manipulation and object dtype arrays.
"""
import warnings
import numpy as np

from .base import BackendProvider

# numpy 2.x moved VisibleDeprecationWarning to numpy.exceptions
from numpy.exceptions import VisibleDeprecationWarning as NumpyVisibleDeprecationWarning


class KGChar(str):
    """Character type for Klong."""
    pass


class NumpyBackendProvider(BackendProvider):
    """NumPy-based backend provider."""

    def __init__(self, device=None):
        if device is not None:
            raise ValueError("Backend 'numpy' does not support device selection")
        self._np = np
        np.seterr(divide='ignore')
        warnings.filterwarnings("error", category=NumpyVisibleDeprecationWarning)
        # Add isarray method to numpy module reference
        self._np.isarray = lambda x: isinstance(x, np.ndarray)

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
        ns = {'np': np}
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
            py_op = {'+': '+', '-': '-', '*': '*', '%': '/', '^': '**'}.get(op)
            if py_op is None:
                return None
            return f'({l}{py_op}{r})'

        if node_type == 'cmp':
            op, left, right = ir[1], ir[2], ir[3]
            l = self._ir_to_source(left)
            r = self._ir_to_source(right)
            if l is None or r is None:
                return None
            py_cmp = {'=': '==', '>': '>', '<': '<'}.get(op)
            if py_cmp is None:
                return None
            return f'(({l}{py_cmp}{r})*1)'

        if node_type == 'negate':
            child = self._ir_to_source(ir[1])
            if child is None:
                return None
            return f'(-{child})'

        if node_type == 'reduce':
            op, arg = ir[1], ir[2]
            arg_src = self._ir_to_source(arg)
            if arg_src is None:
                return None
            method = {'+': 'np.add.reduce', '*': 'np.multiply.reduce', '|': 'np.maximum.reduce', '&': 'np.minimum.reduce'}.get(op)
            if method is None:
                return None
            return f'{method}({arg_src})'

        if node_type == 'scan':
            op, arg = ir[1], ir[2]
            arg_src = self._ir_to_source(arg)
            if arg_src is None:
                return None
            method = {'+': 'np.cumsum', '*': 'np.cumprod'}.get(op)
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
        except (NumpyVisibleDeprecationWarning, ValueError):
            try:
                arr = self._np.asarray(a, dtype=object)
            except ValueError:
                arr = [x.tolist() if self.is_array(x) else x for x in a]
                arr = self._np.asarray(arr, dtype=object)
            arr = self._np.asarray(
                [self.kg_asarray(x) if isinstance(x, list) else x for x in arr],
                dtype=object
            )
        return arr
