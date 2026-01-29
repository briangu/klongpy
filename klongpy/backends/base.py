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


class UnsupportedDtypeError(Exception):
    """Raised when an operation requires a dtype not supported by the backend."""
    pass
