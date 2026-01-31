"""
Autograd tests - unified test structure for both backends.

These tests run against the GLOBAL backend (determined by USE_TORCH env var).
Run tests twice to cover both backends:
    python -m pytest tests/test_autograd_parametrized.py -v           # numpy
    USE_TORCH=1 python -m pytest tests/test_autograd_parametrized.py -v  # torch

The same test logic works for both numeric differentiation (numpy) and
exact autograd (torch), with appropriate tolerances.
"""
import pytest
import numpy as np
from conftest import to_numpy, to_scalar, TORCH_AVAILABLE, get_active_backend

# Tolerance varies by backend and operation
ATOL_NUMPY = 1e-3   # Numeric differentiation with float64
ATOL_TORCH = 1e-5   # Exact autograd is precise
ATOL_NUMERIC_FLOAT32 = 0.02  # Numeric diff with float32 (MPS) has more error


def get_atol(backend=None):
    """Get appropriate tolerance for backend."""
    if backend is None:
        backend = get_active_backend()
    return ATOL_TORCH if backend == 'torch' else ATOL_NUMPY


class TestGradientBasics:
    """Basic gradient tests that should work on all backends."""

    def test_square_derivative(self, klong, backend):
        """d/dx(x^2) = 2x"""
        result = to_scalar(klong('{x^2}:>3.0'))
        assert np.isclose(result, 6.0, atol=get_atol(backend))

    def test_cubic_derivative(self, klong, backend):
        """d/dx(x^3) = 3x^2"""
        result = to_scalar(klong('{x^3}:>2.0'))
        assert np.isclose(result, 12.0, atol=get_atol(backend))

    def test_linear_derivative(self, klong, backend):
        """d/dx(3x) = 3"""
        result = to_scalar(klong('{3*x}:>5.0'))
        assert np.isclose(result, 3.0, atol=get_atol(backend))

    def test_polynomial_derivative(self, klong, backend):
        """d/dx(x^3 + 2x^2 + x) = 3x^2 + 4x + 1"""
        # At x=2: 3(4) + 4(2) + 1 = 21
        result = to_scalar(klong('{(x^3)+(2*x^2)+x}:>2.0'))
        assert np.isclose(result, 21.0, atol=0.1)  # Slightly higher tolerance

    def test_sum_of_squares_gradient(self, klong, backend):
        """gradient of sum(x^2) = 2x"""
        result = to_numpy(klong('{+/x^2}:>[1.0 2.0 3.0]'))
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(result, expected, atol=get_atol(backend))

    def test_named_function(self, klong, backend):
        """Test gradient with named Klong function."""
        klong('sq::{x^2}')
        result = to_scalar(klong('sq:>4.0'))
        assert np.isclose(result, 8.0, atol=get_atol(backend))


class TestNablaOperator:
    """Tests for the nabla (numeric gradient) operator."""

    def test_nabla_scalar(self, klong, backend):
        """Test nabla operator with scalar function."""
        klong('.bkf("sin")')
        klong('g::∇{sin(x)+x*x}')
        result = to_scalar(klong('g(3.14)'))
        expected = 2*3.14 + np.cos(3.14)
        assert np.isclose(result, expected, atol=get_atol(backend))

    def test_nabla_array(self, klong, backend):
        """Test nabla operator with array input."""
        klong('x::˙!5')
        klong('loss::{+/x*x}')
        result = to_numpy(klong('x ∇ loss'))
        expected = np.array([0, 2, 4, 6, 8])
        # Nabla is numeric; torch uses float32 which has more error
        atol = ATOL_NUMERIC_FLOAT32 if backend == 'torch' else ATOL_NUMPY
        np.testing.assert_allclose(result, expected, atol=atol)


class TestGradAndAutogradEquivalence:
    """Verify that :> and nabla give equivalent results."""

    def test_scalar_equivalence(self, klong, backend):
        """Test that :> and nabla match for scalar."""
        r_nabla = to_scalar(klong('3.0 ∇ {x^2}'))
        r_autograd = to_scalar(klong('{x^2}:>3.0'))
        # Nabla is numeric; torch uses float32 which has more error
        atol = ATOL_NUMERIC_FLOAT32 if backend == 'torch' else ATOL_NUMPY
        assert np.isclose(r_nabla, r_autograd, atol=atol)

    def test_array_equivalence(self, klong, backend):
        """Test that :> and nabla match for arrays."""
        klong('x::[1.0 2.0 3.0]')
        klong('f::{+/x^2}')
        r_nabla = to_numpy(klong('x ∇ f'))
        r_autograd = to_numpy(klong('f:>x'))
        # Nabla is numeric; torch uses float32 which has more error
        atol = ATOL_NUMERIC_FLOAT32 if backend == 'torch' else ATOL_NUMPY
        np.testing.assert_allclose(r_nabla, r_autograd, atol=atol)


class TestMultiParamGradient:
    """Tests for multi-parameter gradients with :>[w b] syntax."""

    def test_two_params(self, klong, backend):
        """Test loss:>[w b] where loss = w^2 + b^2."""
        klong('w::2.0')
        klong('b::3.0')
        klong('loss::{(w^2)+(b^2)}')
        result = klong('loss:>[w b]')

        assert len(result) == 2
        # grad of w^2 = 2w = 4, grad of b^2 = 2b = 6
        assert np.isclose(to_scalar(result[0]), 4.0, atol=get_atol(backend))
        assert np.isclose(to_scalar(result[1]), 6.0, atol=get_atol(backend))

    def test_three_params(self, klong, backend):
        """Test with three parameters."""
        klong('a::1.0')
        klong('b::2.0')
        klong('c::3.0')
        klong('loss::{(a^2)+(b^2)+(c^2)}')
        result = klong('loss:>[a b c]')

        assert len(result) == 3
        assert np.isclose(to_scalar(result[0]), 2.0, atol=get_atol(backend))
        assert np.isclose(to_scalar(result[1]), 4.0, atol=get_atol(backend))
        assert np.isclose(to_scalar(result[2]), 6.0, atol=get_atol(backend))


class TestJacobian:
    """Tests for Jacobian computation."""

    def test_element_wise_squared(self, klong, backend):
        """Jacobian of x^2 (element-wise) should be diagonal."""
        klong('f::{x^2}')
        result = to_numpy(klong('[1 2]∂f'))
        expected = np.array([[2., 0.], [0., 4.]])
        np.testing.assert_allclose(result, expected, atol=get_atol(backend))

    def test_scalar_function_jacobian(self, klong, backend):
        """Jacobian of scalar function is gradient vector."""
        klong('f::{+/x^2}')
        result = to_numpy(klong('[1 2 3]∂f')).flatten()
        expected = np.array([2., 4., 6.])
        np.testing.assert_allclose(result, expected, atol=get_atol(backend))

    def test_system_function_syntax(self, klong, backend):
        """Test .jacobian(f;x) syntax."""
        klong('h::{x^2}')
        result = to_numpy(klong('.jacobian(h;[1 2])'))
        expected = np.array([[2., 0.], [0., 4.]])
        np.testing.assert_allclose(result, expected, atol=get_atol(backend))


class TestErrorHandling:
    """Tests for error messages."""

    def test_non_scalar_loss_error(self, klong, backend):
        """Test clear error when loss returns a vector."""
        from klongpy.autograd import NonScalarLossError
        klong('f::{x^2}')  # Returns array, not scalar
        with pytest.raises(NonScalarLossError) as exc_info:
            klong('f:>[1 2 3]')
        assert "scalar" in str(exc_info.value)


# Backend-specific tests

class TestTorchSpecific:
    """Tests that only apply to torch backend."""

    def test_exact_gradients(self, klong, backend):
        """Verify torch gives exact (not numeric) gradients."""
        if backend != 'torch':
            pytest.skip("Requires torch backend (run with USE_TORCH=1)")

        klong('w::2.0')
        klong('b::3.0')
        klong('loss::{(w^2)+(b^2)}')
        result = klong('loss:>[w b]')

        # With torch, gradients should be exact
        assert float(result[0]) == 4.0
        assert float(result[1]) == 6.0


class TestNumpySpecific:
    """Tests that only apply to numpy backend."""

    def test_custom_epsilon(self, klong, backend):
        """Test numeric gradient with custom epsilon."""
        if backend != 'numpy':
            pytest.skip("Requires numpy backend (run without USE_TORCH)")

        from klongpy.autograd import numeric_grad
        result = numeric_grad(lambda x: x[0]**2, np.array([3.0]), eps=1e-8)
        assert np.isclose(result[0], 6.0, atol=1e-5)
