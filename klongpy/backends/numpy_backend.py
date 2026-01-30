"""
NumPy backend provider for KlongPy.

This is the default backend that supports all Klong operations including
string manipulation and object dtype arrays.
"""
import warnings
import numpy as np

from .base import BackendProvider


class KGChar(str):
    """Character type for Klong."""
    pass


class NumpyBackendProvider(BackendProvider):
    """NumPy-based backend provider."""

    def __init__(self, device=None):
        # device parameter is ignored for numpy backend (accepted for API consistency)
        self._np = np
        np.seterr(divide='ignore')
        warnings.filterwarnings("error", category=np.VisibleDeprecationWarning)
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
        # For numpy backend, this is the same as is_array
        # (no separate tensor type like torch)
        return False  # numpy arrays are handled as the base case

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

    def str_to_char_array(self, s):
        """Convert string to character array."""
        return self._np.asarray([KGChar(x) for x in s], dtype=object)

    def kg_asarray(self, a):
        """
        Converts input data into a NumPy array for KlongPy.

        KlongPy treats NumPy arrays as data and Python lists as "programs".
        This function ensures all elements and sub-arrays are converted into
        NumPy arrays, handling complex and jagged data structures.
        """
        if isinstance(a, str):
            return self.str_to_char_array(a)
        try:
            arr = self._np.asarray(a)
            if arr.dtype.kind not in ['O', 'i', 'f']:
                raise ValueError
        except (np.VisibleDeprecationWarning, ValueError):
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
