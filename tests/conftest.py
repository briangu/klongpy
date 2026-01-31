"""
Pytest configuration and fixtures for KlongPy tests.

Provides fixtures for running tests against multiple backends.

IMPORTANT: Currently, autograd operations use the GLOBAL backend determined
by the USE_TORCH environment variable at import time. Per-interpreter backends
work for basic operations but not for autograd. Tests that need both backends
for autograd must be run separately with/without USE_TORCH=1.
"""
import pytest
import numpy as np
from klongpy.backend import use_torch

# Check torch availability once at module load
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_active_backend():
    """Return the currently active global backend name."""
    return 'torch' if use_torch else 'numpy'


@pytest.fixture
def backend():
    """
    Fixture that returns the active global backend name.

    Note: This returns the GLOBAL backend (based on USE_TORCH env var),
    not a per-test backend, because autograd uses global state.
    """
    return get_active_backend()


@pytest.fixture
def klong():
    """
    Fixture that provides a KlongInterpreter with the active backend.

    Usage:
        def test_something(klong):
            result = klong('1+2')
            assert result == 3
    """
    from klongpy import KlongInterpreter
    return KlongInterpreter()


@pytest.fixture
def klong_numpy():
    """
    Fixture for numpy backend - only works if USE_TORCH is NOT set.

    Note: Due to global backend state, this only works when the process
    was started without USE_TORCH=1.
    """
    if use_torch:
        pytest.skip("Requires numpy backend (USE_TORCH is set)")
    from klongpy import KlongInterpreter
    return KlongInterpreter()


@pytest.fixture
def klong_torch():
    """
    Fixture for torch backend - only works if USE_TORCH IS set.

    Note: Due to global backend state, this only works when the process
    was started with USE_TORCH=1.
    """
    if not use_torch:
        pytest.skip("Requires torch backend (USE_TORCH not set)")
    from klongpy import KlongInterpreter
    return KlongInterpreter()


def to_numpy(x):
    """Convert result to numpy for comparison."""
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def to_scalar(x):
    """Convert result to Python scalar."""
    x = to_numpy(x)
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return x.item()
    return float(x) if np.isscalar(x) else x


# Markers for test categorization
def pytest_configure(config):
    config.addinivalue_line("markers", "numpy_only: test only runs with numpy backend")
    config.addinivalue_line("markers", "torch_only: test only runs with torch backend")
    config.addinivalue_line("markers", "slow: test is slow")
