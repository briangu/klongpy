"""
Pytest configuration and fixtures for KlongPy tests.

Provides fixtures for running tests against multiple backends.

Usage:
    pytest tests/                              # Run with numpy backend (default)
    pytest tests/ --backend torch              # Run with torch backend
    pytest tests/ --backend torch --device cpu # Run torch on CPU
    pytest tests/ --backend torch --device cuda # Run torch on CUDA
"""
import pytest
import numpy as np
import os

# Check torch availability once at module load
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Globals to store selected backend/device for monkeypatching
_TEST_BACKEND = None
_TEST_DEVICE = None

# Environment variable for sharing backend name across imports
_ENV_VAR_BACKEND = '_KLONGPY_TEST_BACKEND'
_ENV_VAR_DEVICE = '_KLONGPY_TEST_DEVICE'


def pytest_addoption(parser):
    """Add --backend and --device command line options."""
    parser.addoption(
        "--backend",
        action="store",
        default="numpy",
        choices=["numpy", "torch"],
        help="Backend to use for tests: numpy (default) or torch"
    )
    parser.addoption(
        "--device",
        action="store",
        default=None,
        help="Device for torch backend: cpu, cuda, mps, or None for auto-detect"
    )


def pytest_configure(config):
    """Configure test session - set up backend monkeypatch."""
    global _TEST_BACKEND, _TEST_DEVICE
    _TEST_BACKEND = config.getoption("--backend")
    _TEST_DEVICE = config.getoption("--device")

    # Set environment variables so other modules can access the test configuration
    os.environ[_ENV_VAR_BACKEND] = _TEST_BACKEND or 'numpy'
    if _TEST_DEVICE:
        os.environ[_ENV_VAR_DEVICE] = _TEST_DEVICE

    # Monkeypatch KlongInterpreter to use selected backend/device by default
    import klongpy
    _original_init = klongpy.KlongInterpreter.__init__

    def _patched_init(self, *args, backend=None, device=None, **kwargs):
        if backend is None:
            backend = _TEST_BACKEND
        if device is None and _TEST_DEVICE is not None:
            device = _TEST_DEVICE
        return _original_init(self, *args, backend=backend, device=device, **kwargs)

    klongpy.KlongInterpreter.__init__ = _patched_init

    # Add markers
    config.addinivalue_line("markers", "numpy_only: test only runs with numpy backend")
    config.addinivalue_line("markers", "torch_only: test only runs with torch backend")
    config.addinivalue_line("markers", "slow: test is slow")


def get_backend_name(config):
    """Get backend name from pytest config."""
    return config.getoption("--backend")


def get_device(config):
    """Get device from pytest config."""
    return config.getoption("--device")


@pytest.fixture(scope="session")
def backend_name(request):
    """
    Session-scoped fixture that returns the backend name from --backend option.
    """
    return get_backend_name(request.config)


@pytest.fixture
def backend(request):
    """
    Fixture that returns the active backend name from --backend option.
    """
    return get_backend_name(request.config)


@pytest.fixture
def device(request):
    """
    Fixture that returns the device from --device option.
    """
    return get_device(request.config)


@pytest.fixture
def klong(request):
    """
    Fixture that provides a KlongInterpreter with the selected backend.

    Usage:
        pytest tests/ --backend numpy              # Use numpy backend
        pytest tests/ --backend torch              # Use torch backend (auto device)
        pytest tests/ --backend torch --device cpu # Use torch on CPU

        def test_something(klong):
            result = klong('1+2')
            assert result == 3
    """
    from klongpy import KlongInterpreter
    backend_name = get_backend_name(request.config)
    device = get_device(request.config)
    return KlongInterpreter(backend=backend_name, device=device)


@pytest.fixture
def klong_numpy(request):
    """
    Fixture for numpy backend tests.

    Skips if --backend torch was specified.
    """
    if get_backend_name(request.config) == 'torch':
        pytest.skip("Requires numpy backend (run without --backend torch)")
    from klongpy import KlongInterpreter
    return KlongInterpreter(backend='numpy')


@pytest.fixture
def klong_torch(request):
    """
    Fixture for torch backend tests.

    Skips if --backend torch was NOT specified.
    """
    if get_backend_name(request.config) != 'torch':
        pytest.skip("Requires torch backend (run with --backend torch)")
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    from klongpy import KlongInterpreter
    device = get_device(request.config)
    return KlongInterpreter(backend='torch', device=device)


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


