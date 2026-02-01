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
        from klongpy.backend import get_default_backend
        _backend = get_default_backend()
        result = numeric_grad(lambda x: x[0]**2, np.array([3.0]), _backend, eps=1e-8)
        assert np.isclose(result[0], 6.0, atol=1e-5)


# === Additional tests for complete coverage ===

class TestNumericGradFunction:
    """Tests for the numeric_grad function directly (numpy only)."""

    def test_scalar_function(self, klong, backend):
        """Test numeric gradient of a scalar function."""
        if backend == 'torch':
            pytest.skip("numeric_grad tests use numpy directly")
        from klongpy.autograd import numeric_grad
        from klongpy.backend import get_default_backend
        _backend = get_default_backend()
        result = numeric_grad(lambda x: x[0]**2, np.array([3.0]), _backend)
        assert np.isclose(result[0], 6.0, atol=ATOL_NUMPY)

    def test_multidimensional_function(self, klong, backend):
        """Test numeric gradient of a multidimensional function."""
        if backend == 'torch':
            pytest.skip("numeric_grad tests use numpy directly")
        from klongpy.autograd import numeric_grad
        from klongpy.backend import get_default_backend
        _backend = get_default_backend()
        result = numeric_grad(lambda x: np.sum(x**2), np.array([1.0, 2.0, 3.0]), _backend)
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(result, expected, atol=ATOL_NUMPY)

    def test_2d_array(self, klong, backend):
        """Test numeric gradient with 2D array."""
        if backend == 'torch':
            pytest.skip("numeric_grad tests use numpy directly")
        from klongpy.autograd import numeric_grad
        from klongpy.backend import get_default_backend
        _backend = get_default_backend()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = numeric_grad(lambda x: np.sum(x**2), x, _backend)
        expected = 2 * x
        np.testing.assert_allclose(result, expected, atol=ATOL_NUMPY)


class TestGradOfFnAPI:
    """Tests for the grad_of_fn function with various input types."""

    def test_with_callable(self, klong, backend):
        """Test grad_of_fn with a plain Python callable."""
        if backend == 'torch':
            pytest.skip("Plain Python callable with np.sum doesn't work with torch")
        from klongpy.autograd import grad_of_fn
        result = to_numpy(grad_of_fn(klong, lambda x: np.sum(x**2), np.array([1.0, 2.0, 3.0])))
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(result, expected, atol=get_atol(backend))

    def test_with_kgsym(self, klong, backend):
        """Test grad_of_fn with a KGSym (symbol reference)."""
        from klongpy.autograd import grad_of_fn
        from klongpy.core import KGSym
        klong('f::{+/x^2}')
        fn_sym = KGSym('f')
        result = to_numpy(grad_of_fn(klong, fn_sym, np.array([1.0, 2.0, 3.0])))
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(result, expected, atol=get_atol(backend))

    def test_with_kgfn(self, klong, backend):
        """Test grad_of_fn with a KGFn."""
        from klongpy.autograd import grad_of_fn
        from klongpy.core import KGSym
        klong('f::{+/x^2}')
        fn = klong._context[KGSym('f')]
        result = to_numpy(grad_of_fn(klong, fn, np.array([1.0, 2.0, 3.0])))
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(result, expected, atol=get_atol(backend))


class TestMatrixGradients:
    """Tests for gradients with matrix inputs."""

    def test_matrix_grad(self, klong, backend):
        """Test gradient with matrix input."""
        klong('A::˙[2 2]:^!4')
        klong('B::[2 2]:^!4')
        result = to_numpy(klong('(A ∇ {+/(+/ (A*B)) })'))
        B = to_numpy(klong('B'))
        # Nabla is numeric; torch float32 has more error
        atol = ATOL_NUMERIC_FLOAT32 if backend == 'torch' else ATOL_NUMPY
        np.testing.assert_allclose(result, B, atol=atol)

    def test_matrix_equivalence(self, klong, backend):
        """Test that :> and ∇ match for matrix inputs."""
        klong('M::˙[2 3]:^!6')
        klong('f::{+/,/x*x}')
        r_nabla = to_numpy(klong('M ∇ f'))
        r_autograd = to_numpy(klong('f:>M'))
        atol = ATOL_NUMERIC_FLOAT32 if backend == 'torch' else ATOL_NUMPY
        np.testing.assert_allclose(r_nabla, r_autograd, atol=atol)


class TestMultiParamAdvanced:
    """Additional multi-parameter gradient tests."""

    def test_array_still_works(self, klong, backend):
        """Test that loss:>wts still works for single array (portfolio pattern)."""
        klong('wts::[0.25 0.25 0.25 0.25]')
        result = to_numpy(klong('{+/x^2}:>wts'))
        expected = np.array([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(result, expected, atol=get_atol(backend))

    def test_numeric_array_not_multiparam(self, klong, backend):
        """Test that loss:>[1 2 3] treats as point, not multi-param."""
        result = to_numpy(klong('{+/x^2}:>[1 2 3]'))
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(result, expected, atol=get_atol(backend))

    def test_linear_regression_gradients(self, klong, backend):
        """Test gradients for a simple linear regression loss."""
        klong('w::1.0')
        klong('b::0.0')
        klong('X::[1 2 3]')
        klong('Y::[3 5 7]')  # Y = 2*X + 1
        klong('loss::{(+/((w*X)+b-Y)^2)%3}')
        result = klong('loss:>[w b]')
        grad_w = to_scalar(result[0])
        grad_b = to_scalar(result[1])
        # Gradients should push w toward 2 and b toward 1
        assert grad_w < 0  # Loss decreases as w increases
        assert grad_b < 0  # Loss decreases as b increases


class TestJacobianAdvanced:
    """Additional Jacobian tests."""

    def test_partial_operator_different_point(self, klong, backend):
        """Test ∂ operator at different points."""
        klong('g::{x^2}')
        result = to_numpy(klong('[3 4]∂g'))
        expected = np.array([[6., 0.], [0., 8.]])
        np.testing.assert_allclose(result, expected, atol=get_atol(backend))

    def test_multi_param_jacobian(self, klong, backend):
        """Test [w b]∂f syntax for multi-parameter Jacobians."""
        klong('w::[1.0 2.0]')
        klong('b::[3.0 4.0]')
        klong('f::{w^2}')
        result = klong('[w b]∂f')

        assert len(result) == 2
        jac_w = to_numpy(result[0])
        expected_w = np.array([[2., 0.], [0., 4.]])
        np.testing.assert_allclose(jac_w, expected_w, atol=get_atol(backend))

        # f doesn't depend on b, so jacobian w.r.t. b should be zero
        jac_b = to_numpy(result[1])
        expected_b = np.array([[0., 0.], [0., 0.]])
        np.testing.assert_allclose(jac_b, expected_b, atol=get_atol(backend))


class TestTorchAutogradFunction:
    """Tests for torch_autograd function directly."""

    def test_torch_autograd_not_in_torch_mode(self, klong, backend):
        """Test that torch_autograd raises error when not in torch mode."""
        if backend == 'torch':
            pytest.skip("Test for numpy backend only")
        from klongpy.autograd import torch_autograd
        with pytest.raises(RuntimeError):
            torch_autograd(lambda x: x**2, np.array([3.0]))

    def test_torch_autograd_with_tensor(self, klong, backend):
        """Test torch_autograd with tensor input."""
        if backend != 'torch':
            pytest.skip("Requires torch backend")
        import torch
        from klongpy.autograd import torch_autograd
        x = torch.tensor([1.0, 2.0, 3.0])
        result = torch_autograd(lambda t: (t**2).sum(), x)
        expected = torch.tensor([2.0, 4.0, 6.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_torch_autograd_non_scalar_output(self, klong, backend):
        """Test that torch_autograd raises error for non-scalar output."""
        if backend != 'torch':
            pytest.skip("Requires torch backend")
        from klongpy.autograd import torch_autograd, NonScalarLossError
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(NonScalarLossError):
            torch_autograd(lambda t: t**2, x)


class TestGradcheck:
    """Tests for torch.autograd.gradcheck verification."""

    def test_gradcheck_basic_scalar(self, klong, backend):
        """Test .gradcheck() with basic scalar function."""
        if backend != 'torch':
            pytest.skip("Requires torch backend")
        klong('f::{x^2}')
        result = klong('.gradcheck(f;3.0)')
        assert result == 1

    def test_gradcheck_vector_input(self, klong, backend):
        """Test .gradcheck() with vector input."""
        if backend != 'torch':
            pytest.skip("Requires torch backend")
        klong('g::{+/x^2}')
        result = klong('.gradcheck(g;[1.0 2.0 3.0])')
        assert result == 1

    def test_gradcheck_polynomial(self, klong, backend):
        """Test .gradcheck() with polynomial function."""
        if backend != 'torch':
            pytest.skip("Requires torch backend")
        klong('p::{(x^3)+(2*x^2)+x}')
        result = klong('.gradcheck(p;2.0)')
        assert result == 1

    def test_gradcheck_backend_direct(self, klong, backend):
        """Test gradcheck() directly on backend."""
        if backend != 'torch':
            pytest.skip("Requires torch backend")
        import torch
        result = klong._backend.gradcheck(
            lambda x: (x**2).sum(),
            (torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True),)
        )
        assert result is True

    def test_gradcheck_fails_on_numpy(self, klong, backend):
        """Test that .gradcheck() raises error on numpy backend."""
        if backend != 'numpy':
            pytest.skip("Test for numpy backend only")
        klong('f::{x^2}')
        with pytest.raises(RuntimeError):
            klong('.gradcheck(f;3.0)')


class TestCompile:
    """Tests for torch.compile functionality."""

    def test_compile_basic(self, klong, backend):
        """Test .compile() returns a compiled function."""
        if backend != 'torch':
            pytest.skip("Requires torch backend")
        klong('f::{x^2}')
        try:
            result = klong('.compile(f;3.0)')
        except Exception as e:
            # torch.compile may fail on some systems (missing C++ compiler, etc.)
            if 'CppCompileError' in str(type(e).__name__) or 'InductorError' in str(type(e).__name__):
                pytest.skip(f"torch.compile not available on this system: {e}")
            raise
        # Compiled function should still work
        import torch
        test_input = torch.tensor(5.0)
        output = result(test_input)
        assert np.isclose(float(output), 25.0, atol=1e-5)

    def test_export_with_path(self, klong, backend):
        """Test .export() returns dict with graph info."""
        if backend != 'torch':
            pytest.skip("Requires torch backend")
        import tempfile
        import os
        klong('f::{x^2}')
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pt2')
            try:
                result = klong(f'.export(f;3.0;"{path}")')
            except Exception as e:
                # torch.compile/export may fail on some systems
                if 'CppCompileError' in str(type(e).__name__) or 'InductorError' in str(type(e).__name__):
                    pytest.skip(f"torch.compile not available on this system: {e}")
                raise
            # Should return a dict
            assert isinstance(result, dict)
            assert 'compiled_fn' in result
            # Export may fail on some systems, but compiled_fn should work
            if result.get('export_path'):
                assert os.path.exists(path)

    def test_compile_fails_on_numpy(self, klong, backend):
        """Test that .compile() raises error on numpy backend."""
        if backend != 'numpy':
            pytest.skip("Test for numpy backend only")
        klong('f::{x^2}')
        with pytest.raises(RuntimeError):
            klong('.compile(f;3.0)')

    def test_export_fails_on_numpy(self, klong, backend):
        """Test that .export() raises error on numpy backend."""
        if backend != 'numpy':
            pytest.skip("Test for numpy backend only")
        klong('f::{x^2}')
        with pytest.raises(RuntimeError):
            klong('.export(f;3.0;"test.pt2")')


class TestCompileModes:
    """Tests for extended compile options."""

    def test_cmodes_returns_dict(self, klong, backend):
        """Test .cmodes() returns mode information."""
        if backend != 'torch':
            pytest.skip("Requires torch backend")
        result = klong('.cmodes()')
        assert isinstance(result, dict)
        assert 'modes' in result
        assert 'backends' in result
        assert 'recommendations' in result

    def test_cmodes_has_expected_modes(self, klong, backend):
        """Test .cmodes() includes standard modes."""
        if backend != 'torch':
            pytest.skip("Requires torch backend")
        result = klong('.cmodes()')
        modes = result['modes']
        assert 'default' in modes
        assert 'reduce-overhead' in modes
        assert 'max-autotune' in modes

    def test_cmodes_has_expected_backends(self, klong, backend):
        """Test .cmodes() includes standard backends."""
        if backend != 'torch':
            pytest.skip("Requires torch backend")
        result = klong('.cmodes()')
        backends = result['backends']
        assert 'inductor' in backends
        assert 'eager' in backends

    def test_compilex_eager_backend(self, klong, backend):
        """Test .compilex() with eager backend (no actual compilation)."""
        if backend != 'torch':
            pytest.skip("Requires torch backend")
        klong('f::{x^2}')
        # Eager backend should work without C++ compiler
        result = klong('.compilex(f;3.0;:{["backend" "eager"]})')
        # Should return a callable
        import torch
        test_input = torch.tensor(5.0)
        output = result(test_input)
        assert np.isclose(float(output), 25.0, atol=1e-5)

    def test_compilex_fails_on_numpy(self, klong, backend):
        """Test that .compilex() raises error on numpy backend."""
        if backend != 'numpy':
            pytest.skip("Test for numpy backend only")
        klong('f::{x^2}')
        with pytest.raises(RuntimeError):
            klong('.compilex(f;3.0;:{["mode" "default"]})')

    def test_cmodes_fails_on_numpy(self, klong, backend):
        """Test that .cmodes() raises error on numpy backend."""
        if backend != 'numpy':
            pytest.skip("Test for numpy backend only")
        with pytest.raises(RuntimeError):
            klong('.cmodes()')
