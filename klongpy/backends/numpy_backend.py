"""
NumPy backend provider for KlongPy.

This is the default backend that supports all Klong operations including
string manipulation and object dtype arrays.
"""
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
        # a float beyond int64 wraps, as the integer operators do, without a cast warning
        with np.errstate(invalid='ignore'):
            return np.asarray(a, dtype=int) if self.is_array(a) else int(a)

    def power(self, a, b):
        """Compute a^b, returning integer if result is whole number."""
        # overflow to inf is answered by the caller (a range error), not by a warning
        with np.errstate(divide='ignore', over='ignore'):
            r = np.power(float(a) if isinstance(a, (int, np.integer)) else a, b)
        return r

    def str_to_char_array(self, s):
        """Convert string to character array."""
        return self._np.asarray([KGChar(x) for x in s], dtype=object)

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
