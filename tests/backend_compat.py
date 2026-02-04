"""
Backend compatibility utilities for tests.

This module provides decorators and utilities to conditionally run tests
based on backend capabilities (numpy vs torch).

Usage:
    from tests.backend_compat import (
        requires_object_dtype,
        requires_strings,
        requires_numpy_backend,
        requires_torch_backend,
        numeric_only,
        get_backend_info,
    )

    class TestMyFeature(unittest.TestCase):
        @requires_strings
        def test_string_operations(self):
            # This test only runs when strings are supported
            ...

        @numeric_only
        def test_array_math(self):
            # This test should work on all backends
            ...
"""
import functools
import importlib
import importlib.util
import os
import unittest

import numpy as np

from klongpy.backends import get_backend

_TORCH_SPEC = importlib.util.find_spec("torch")
torch = importlib.import_module("torch") if _TORCH_SPEC else None
TORCH_AVAILABLE = torch is not None


def _get_test_backend_name():
    """Get the backend name from pytest config (set in conftest.py via env var)."""
    return os.environ.get('_KLONGPY_TEST_BACKEND', 'numpy')


def _get_test_backend():
    """Get a backend instance for the current test configuration."""
    return get_backend(_get_test_backend_name())


def is_torch_backend(backend=None):
    """Check if the given backend (or test default) is torch."""
    if backend is None:
        return _get_test_backend_name() == 'torch'
    return backend.name == 'torch'


def get_backend_info(backend=None):
    """Get information about the current backend configuration."""
    if backend is None:
        backend = _get_test_backend()
    return {
        'name': backend.name,
        'is_torch': is_torch_backend(backend),
        'torch_available': TORCH_AVAILABLE,
        'supports_object_dtype': backend.supports_object_dtype(),
        'supports_strings': backend.supports_strings(),
        'supports_float64': backend.supports_float64() if hasattr(backend, 'supports_float64') else True,
        'device': getattr(backend, 'device', None),
    }


def _is_mps_device(backend=None):
    """Check if torch is using MPS device (Apple Silicon)."""
    if backend is None:
        backend = _get_test_backend()
    if not is_torch_backend(backend) or not TORCH_AVAILABLE:
        return False
    device = getattr(backend, 'device', None)
    return device is not None and 'mps' in str(device).lower()


def _supports_float64():
    """Check if backend supports float64."""
    backend = _get_test_backend()
    if hasattr(backend, 'supports_float64'):
        return backend.supports_float64()
    # MPS doesn't support float64
    if _is_mps_device():
        return False
    return True


def to_numpy(x):
    """
    Convert a tensor or array to numpy array.

    This is a convenience function that handles the common pattern of
    converting torch tensors to numpy arrays, including handling
    gradients and device transfers.

    Parameters
    ----------
    x : array-like, torch.Tensor, or scalar
        The value to convert.

    Returns
    -------
    numpy.ndarray or scalar
        The converted value as a numpy array or Python scalar.

    Examples
    --------
    >>> to_numpy(torch.tensor([1, 2, 3]))
    array([1, 2, 3])
    >>> to_numpy(torch.tensor(5.0, requires_grad=True))
    5.0
    """
    if TORCH_AVAILABLE and torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.ndim == 0:
            return x.item()
        return x.numpy()
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        return x
    return x


def _make_skip_decorator(condition_fn, reason_fn):
    """Factory for creating skip decorators."""
    def decorator(test_item):
        if isinstance(test_item, type):
            # Decorating a class
            for attr_name in dir(test_item):
                if attr_name.startswith('test'):
                    attr = getattr(test_item, attr_name)
                    if callable(attr):
                        setattr(test_item, attr_name, decorator(attr))
            return test_item
        else:
            # Decorating a method
            @functools.wraps(test_item)
            def wrapper(*args, **kwargs):
                if condition_fn():
                    raise unittest.SkipTest(reason_fn())
                return test_item(*args, **kwargs)
            return wrapper
    return decorator


# === Skip Decorators ===

def requires_object_dtype(test_item):
    """
    Skip test if the backend doesn't support object dtype arrays.

    Use this for tests that involve:
    - Nested/jagged arrays like [1, [2, 3], 4]
    - Mixed-type arrays
    - Arrays containing non-numeric types
    """
    return _make_skip_decorator(
        lambda: not _get_test_backend().supports_object_dtype(),
        lambda: f"Backend '{_get_test_backend_name()}' does not support object dtype"
    )(test_item)


def requires_strings(test_item):
    """
    Skip test if the backend doesn't support string operations.

    Use this for tests that involve:
    - String literals and manipulation
    - Character arrays
    - String comparison operations
    """
    return _make_skip_decorator(
        lambda: not _get_test_backend().supports_strings(),
        lambda: f"Backend '{_get_test_backend_name()}' does not support strings"
    )(test_item)


def requires_numpy_backend(test_item):
    """
    Skip test if not using the numpy backend.

    Use this for tests that specifically require numpy behavior
    or test numpy-specific features.
    """
    return _make_skip_decorator(
        lambda: is_torch_backend(),
        lambda: "Test requires numpy backend"
    )(test_item)


def requires_torch_backend(test_item):
    """
    Skip test if not using the torch backend.

    Use this for tests that specifically test torch functionality.
    """
    return _make_skip_decorator(
        lambda: not is_torch_backend(),
        lambda: "Test requires torch backend"
    )(test_item)


def requires_torch_available(test_item):
    """
    Skip test if torch is not installed.

    Use this for tests that need torch to be available but don't
    necessarily require it as the active backend.
    """
    return _make_skip_decorator(
        lambda: not TORCH_AVAILABLE,
        lambda: "Test requires torch to be installed"
    )(test_item)


def requires_float64(test_item):
    """
    Skip test if the backend doesn't support float64.

    Use this for tests that require double precision floating point,
    which is not supported on MPS devices.
    """
    return _make_skip_decorator(
        lambda: not _supports_float64(),
        lambda: f"Backend does not support float64 (MPS device: {_is_mps_device()})"
    )(test_item)


def requires_float32(test_item):
    """
    Skip test if float32 is not available.

    This is mainly a marker decorator since float32 is universally supported.
    Useful for tests that specifically test float32 behavior.
    """
    # Float32 is always supported
    test_item._requires_float32 = True
    return test_item


def requires_cpu_device(test_item):
    """
    Skip test if not using CPU device.

    Use this for tests that require CPU-specific operations,
    like certain numpy interop or operations not available on MPS/CUDA.
    """
    return _make_skip_decorator(
        lambda: _is_mps_device() or (is_torch_backend() and TORCH_AVAILABLE and torch.cuda.is_available()),
        lambda: "Test requires CPU device"
    )(test_item)


def skip_mps_autograd(test_item):
    """
    Skip test if using MPS device due to autograd limitations.

    MPS device has some limitations with complex autograd operations
    that can cause "Placeholder storage has not been allocated" errors.
    """
    return _make_skip_decorator(
        lambda: _is_mps_device(),
        lambda: "MPS device has autograd limitations for complex operations"
    )(test_item)


def numeric_only(test_item):
    """
    Marker decorator for tests that only use numeric operations.

    These tests should work on all backends. This decorator doesn't
    skip anything - it's documentation that the test is backend-agnostic.

    Use this for tests that:
    - Only use numeric arrays (int, float)
    - Don't use strings or mixed-type arrays
    - Don't rely on object dtype behavior
    """
    # Just a marker - doesn't modify behavior
    test_item._numeric_only = True
    return test_item


# === Backend-aware assertion helpers ===

class BackendAwareTestCase(unittest.TestCase):
    """
    Base test case with backend-aware utilities.

    Provides helper methods that work correctly with both numpy and torch backends.
    """

    @property
    def backend(self):
        """Get the current backend provider."""
        return get_backend()

    @property
    def backend_name(self):
        """Get the current backend name."""
        return self.backend.name

    @property
    def uses_torch_backend(self):
        """Check if using torch backend."""
        return is_torch_backend(self.backend)

    def assertArrayEqual(self, a, b, msg=None):
        """
        Assert two arrays are equal, handling both numpy and torch.
        """
        if not self.backend.kg_equal(a, b):
            self.fail(msg or f"Arrays not equal:\n  {a}\n  !=\n  {b}")

    def assertArrayClose(self, a, b, rtol=1e-5, atol=1e-8, msg=None):
        """
        Assert two arrays are close, handling both numpy and torch.
        """
        # Convert to numpy for comparison
        if TORCH_AVAILABLE and torch is not None:
            if isinstance(a, torch.Tensor):
                a = a.detach().cpu().numpy()
            if isinstance(b, torch.Tensor):
                b = b.detach().cpu().numpy()
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not np.allclose(a, b, rtol=rtol, atol=atol):
                self.fail(msg or f"Arrays not close:\n  {a}\n  !=\n  {b}")
        else:
            if not np.isclose(a, b, rtol=rtol, atol=atol):
                self.fail(msg or f"Values not close: {a} != {b}")

    def to_python(self, x):
        """
        Convert backend array/tensor to Python native type.

        Useful for assertions that need Python types.
        """
        if TORCH_AVAILABLE and torch is not None and isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            if x.ndim == 0:
                return x.item()
            return x.numpy().tolist()
        if isinstance(x, np.ndarray):
            if x.ndim == 0:
                return x.item()
            return x.tolist()
        return x

    def skipIfNoObjectDtype(self):
        """Skip the current test if object dtype not supported."""
        if not self.backend.supports_object_dtype():
            self.skipTest(f"Backend '{self.backend_name}' does not support object dtype")

    def skipIfNoStrings(self):
        """Skip the current test if strings not supported."""
        if not self.backend.supports_strings():
            self.skipTest(f"Backend '{self.backend_name}' does not support strings")


# === Test categorization utilities ===

def categorize_test_file(test_cases):
    """
    Analyze a test file and categorize tests by their backend requirements.

    Returns a dict with:
    - 'numeric_only': tests that only use numeric operations
    - 'requires_strings': tests that use string operations
    - 'requires_object_dtype': tests that use object dtype
    - 'uncategorized': tests that haven't been categorized
    """
    # This is a placeholder for potential automated categorization
    pass


# Convenience exports
__all__ = [
    'get_backend_info',
    'requires_object_dtype',
    'requires_strings',
    'requires_numpy_backend',
    'requires_torch_backend',
    'requires_torch_available',
    'requires_float64',
    'requires_float32',
    'requires_cpu_device',
    'skip_mps_autograd',
    'numeric_only',
    'BackendAwareTestCase',
    'TORCH_AVAILABLE',
    'is_torch_backend',
    'to_numpy',
]
