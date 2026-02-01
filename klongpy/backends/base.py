"""
Base interface for array backends.

All backends must implement the BackendProvider interface to ensure
consistent behavior across numpy, torch, and any future backends.
"""
from abc import ABC, abstractmethod


def is_jagged_array(x):
    """Check if x is a jagged (ragged) array - a list of lists with different lengths."""
    if isinstance(x, list) and len(x) > 0:
        if all(isinstance(item, (list, tuple)) for item in x):
            return len(set(map(len, x))) > 1
    return False


def is_supported_type(x):
    """Check if x can be converted to a tensor/array by the current backend.

    Default implementation returns True for everything except strings and jagged arrays.
    """
    return not (isinstance(x, str) or is_jagged_array(x))


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

    def to_display(self, x):
        """
        Convert backend array to display-friendly format.

        For display purposes, converts arrays to numpy for consistent formatting.
        0-dim arrays are converted to Python scalars.
        Override in subclasses if different behavior is needed.
        """
        if self.is_backend_array(x):
            x = self.to_numpy(x)
        # Convert 0-dim arrays to Python scalars
        if hasattr(x, 'item') and hasattr(x, 'ndim') and x.ndim == 0:
            return x.item()
        return x

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

    def is_integer(self, x) -> bool:
        """Check if x is an integer type (scalar, numpy integer, or 0-dim integer tensor)."""
        import numpy as np
        if issubclass(type(x), (int, np.integer)):
            return True
        return self.is_scalar_integer(x)

    def is_float(self, x) -> bool:
        """Check if x is a float type (scalar, numpy float, int, or 0-dim float tensor)."""
        import numpy as np
        if issubclass(type(x), (float, np.floating, int)):
            return True
        return self.is_scalar_float(x) or self.is_scalar_integer(x)

    def is_number(self, a) -> bool:
        """Check if a is a number (integer or float)."""
        return self.is_float(a) or self.is_integer(a)

    def str_to_chr_arr(self, s):
        """Convert string to character array (alias for str_to_char_array)."""
        return self.str_to_char_array(s)

    @abstractmethod
    def array_size(self, a):
        """
        Get the total number of elements in an array/tensor.

        Works with both numpy arrays and torch tensors.

        Returns:
            int: Total element count (product of all dimensions)
        """
        pass

    def safe_equal(self, x, y):
        """
        Compare two values for equality, handling backend-specific array types.

        Returns a truth value (0 or 1) suitable for Klong.
        """
        import numpy as np
        return np.asarray(x, dtype=object) == np.asarray(y, dtype=object)

    def detach_if_needed(self, x):
        """
        Detach array from computation graph if needed.

        For backends without autograd, this is a no-op.
        """
        return x

    def to_int_array(self, a):
        """
        Convert array to integer type.
        """
        import numpy as np
        return np.asarray(a, dtype=int) if self.is_array(a) else int(a)

    def power(self, a, b):
        """
        Compute a^b, handling gradient tracking if applicable.

        Returns integer result if the result is a whole number.
        """
        import numpy as np
        r = np.power(float(a) if isinstance(a, (int, np.integer)) else a, b)
        return r

    def has_gradient(self, x) -> bool:
        """Check if x is tracking gradients (for autograd)."""
        return False

    def supports_autograd(self) -> bool:
        """Whether this backend supports automatic differentiation."""
        return False

    def create_grad_tensor(self, x):
        """Create a tensor that tracks gradients. Raises if not supported."""
        raise NotImplementedError("This backend does not support autograd")

    def compute_autograd(self, func, x):
        """Compute gradient using automatic differentiation. Raises if not supported."""
        raise NotImplementedError("This backend does not support autograd")

    def compute_multi_autograd(self, func, params):
        """
        Compute gradients for multiple parameters in one backward pass.

        Args:
            func: Callable that takes a list of tensors and returns a scalar loss
            params: List of parameter values to compute gradients for

        Returns:
            List of gradients, one per parameter
        """
        raise NotImplementedError("This backend does not support multi-parameter autograd")

    def compute_jacobian(self, func, x):
        """
        Compute Jacobian matrix of func at point x.

        Args:
            func: Callable that takes x and returns a vector
            x: Input point (tensor/array)

        Returns:
            Jacobian matrix J where J[i,j] = df_i/dx_j
        """
        raise NotImplementedError("This backend does not support Jacobian computation")

    def compile_function(self, func, example_input, output_path=None, mode="default",
                         backend="inductor", fullgraph=False, dynamic=None):
        """
        Compile a function for optimized execution and optionally export for inspection.

        Args:
            func: Callable to compile
            example_input: Example input for tracing the function
            output_path: Optional path to export the compiled graph
            mode: Compilation mode ("default", "reduce-overhead", "max-autotune")
            backend: Compilation backend ("inductor", "eager", "cudagraphs")
            fullgraph: If True, requires entire function to compile as one graph
            dynamic: If True, enables dynamic shapes

        Returns:
            Compiled function or export info dict
        """
        raise NotImplementedError("This backend does not support function compilation")

    def get_compile_modes(self):
        """
        Return information about available compilation modes.

        Returns:
            Dict with mode descriptions and recommendations
        """
        raise NotImplementedError("This backend does not support function compilation")

    def gradcheck(self, func, inputs, eps=1e-6, atol=1e-5, rtol=1e-3):
        """
        Check gradients computed by autograd against numeric gradients.

        Args:
            func: Function to check
            inputs: Tuple of input tensors
            eps: Step size for numeric differentiation
            atol: Absolute tolerance
            rtol: Relative tolerance

        Returns:
            True if gradients match, raises error otherwise
        """
        raise NotImplementedError("This backend does not support gradcheck")

    def klong_gradcheck(self, klong, fn, inputs):
        """
        Check gradients for a Klong function.

        This is a higher-level interface that handles wrapping the Klong function
        and converting inputs appropriately for the backend.

        Args:
            klong: KlongInterpreter instance
            fn: Klong function to check
            inputs: Input value or list of inputs

        Returns:
            1 if gradients are correct, raises error otherwise
        """
        raise RuntimeError(
            ".gradcheck() requires PyTorch backend. "
            "Run with USE_TORCH=1 environment variable."
        )


class UnsupportedDtypeError(Exception):
    """Raised when an operation requires a dtype not supported by the backend."""
    pass
