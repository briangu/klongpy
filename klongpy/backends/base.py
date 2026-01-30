"""
Base interface for array backends.

All backends must implement the BackendProvider interface to ensure
consistent behavior across numpy, torch, and any future backends.
"""
from abc import ABC, abstractmethod


class BackendProvider(ABC):
    """Abstract interface for array backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name."""
        pass

    @property
    @abstractmethod
    def np(self):
        """Return numpy-compatible array module."""
        pass

    @abstractmethod
    def supports_object_dtype(self) -> bool:
        """Whether this backend supports object dtype."""
        pass

    @abstractmethod
    def supports_strings(self) -> bool:
        """Whether this backend supports string operations."""
        pass

    @abstractmethod
    def supports_float64(self) -> bool:
        """Whether this backend supports float64 (double precision)."""
        pass

    @abstractmethod
    def str_to_char_array(self, s):
        """Convert string to character array."""
        pass

    @abstractmethod
    def kg_asarray(self, a):
        """
        Klong-specific array conversion.

        Converts input data into an array suitable for Klong operations.
        For backends that don't support object dtype, this should raise
        an appropriate exception for unsupported data types.
        """
        pass

    @abstractmethod
    def is_array(self, x) -> bool:
        """Check if x is an array type for this backend."""
        pass

    @abstractmethod
    def is_backend_array(self, x) -> bool:
        """Check if x is specifically this backend's array type (not numpy)."""
        pass

    @abstractmethod
    def get_dtype_kind(self, arr) -> str:
        """
        Get the dtype 'kind' character for an array.

        Returns:
            'O' for object dtype
            'i' for integer types
            'f' for float types
            'u' for unsigned integer
            'b' for boolean
            'c' for complex
            None if not an array
        """
        pass

    @abstractmethod
    def to_numpy(self, x):
        """
        Convert backend array to numpy array.

        Handles device transfers (e.g., GPU to CPU) and gradient detachment.
        """
        pass

    @abstractmethod
    def is_scalar_integer(self, x) -> bool:
        """Check if x is a 0-dim integer array/tensor."""
        pass

    @abstractmethod
    def is_scalar_float(self, x) -> bool:
        """Check if x is a 0-dim float array/tensor."""
        pass

    def scalar_to_python(self, x):
        """Convert a 0-dim array/tensor to Python scalar."""
        if hasattr(x, 'item'):
            return x.item()
        return x

    @abstractmethod
    def argsort(self, a, descending=False):
        """Return indices that would sort the array."""
        pass

    def supports_autograd(self) -> bool:
        """Whether this backend supports automatic differentiation."""
        return False

    def create_grad_tensor(self, x):
        """Create a tensor that tracks gradients. Raises if not supported."""
        raise NotImplementedError("This backend does not support autograd")

    def compute_autograd(self, func, x):
        """Compute gradient using automatic differentiation. Raises if not supported."""
        raise NotImplementedError("This backend does not support autograd")


class UnsupportedDtypeError(Exception):
    """Raised when an operation requires a dtype not supported by the backend."""
    pass
