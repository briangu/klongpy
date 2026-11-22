"""
Autograd tests - unified test structure for both backends.

Run with the --backend option:
    pytest tests/test_autograd_parametrized.py -v               # numpy (default)
    pytest tests/test_autograd_parametrized.py -v --backend torch  # torch

The same test logic works for both numeric differentiation (numpy) and
exact autograd (torch), with appropriate tolerances.
"""
import importlib
import os
import tempfile

import numpy as np
import pytest

from conftest import to_numpy, to_scalar, TORCH_AVAILABLE
from klongpy.autograd import NonScalarLossError, grad_of_fn, numeric_grad
from klongpy.backends import get_backend
from klongpy.core import KGSym
from klongpy.errors import KlongError
from utils import torch_autograd

torch = importlib.import_module("torch") if TORCH_AVAILABLE else None

# Tolerance varies by backend and operation
ATOL_NUMPY = 1e-3   # Numeric differentiation with float64
ATOL_TORCH = 1e-5   # Exact autograd is precise
ATOL_NUMERIC_FLOAT32 = 0.02  # Numeric diff with float32 (MPS) has more error


def get_atol(backend):
    """Get appropriate tolerance for backend."""
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
        _backend = get_backend()
        result = numeric_grad(lambda x: x[0]**2, np.array([3.0]), _backend, eps=1e-8)
        assert np.isclose(result[0], 6.0, atol=1e-5)


# === Additional tests for complete coverage ===

class TestNumericGradFunction:
    """Tests for the numeric_grad function directly (numpy only)."""

    def test_scalar_function(self, klong, backend):
        """Test numeric gradient of a scalar function."""
        if backend == 'torch':
            pytest.skip("numeric_grad tests use numpy directly")
        _backend = get_backend()
        result = numeric_grad(lambda x: x[0]**2, np.array([3.0]), _backend)
        assert np.isclose(result[0], 6.0, atol=ATOL_NUMPY)

    def test_multidimensional_function(self, klong, backend):
        """Test numeric gradient of a multidimensional function."""
        if backend == 'torch':
            pytest.skip("numeric_grad tests use numpy directly")
        _backend = get_backend()
        result = numeric_grad(lambda x: np.sum(x**2), np.array([1.0, 2.0, 3.0]), _backend)
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(result, expected, atol=ATOL_NUMPY)

    def test_2d_array(self, klong, backend):
        """Test numeric gradient with 2D array."""
        if backend == 'torch':
            pytest.skip("numeric_grad tests use numpy directly")
        _backend = get_backend()
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
        result = to_numpy(grad_of_fn(klong, lambda x: np.sum(x**2), np.array([1.0, 2.0, 3.0])))
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(result, expected, atol=get_atol(backend))

    def test_with_kgsym(self, klong, backend):
        """Test grad_of_fn with a KGSym (symbol reference)."""
        klong('f::{+/x^2}')
        fn_sym = KGSym('f')
        result = to_numpy(grad_of_fn(klong, fn_sym, np.array([1.0, 2.0, 3.0])))
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(result, expected, atol=get_atol(backend))

    def test_with_kgfn(self, klong, backend):
        """Test grad_of_fn with a KGFn."""
        klong('f::{+/x^2}')
        fn = klong['f'].fn
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
        with pytest.raises(RuntimeError):
            torch_autograd(lambda x: x**2, np.array([3.0]), klong._backend)

    def test_torch_autograd_with_tensor(self, klong, backend):
        """Test torch_autograd with tensor input."""
        if backend != 'torch':
            pytest.skip("Requires torch backend")
        x = torch.tensor([1.0, 2.0, 3.0])
        result = torch_autograd(lambda t: (t**2).sum(), x, klong._backend)
        expected = torch.tensor([2.0, 4.0, 6.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_torch_autograd_non_scalar_output(self, klong, backend):
        """Test that torch_autograd raises error for non-scalar output."""
        if backend != 'torch':
            pytest.skip("Requires torch backend")
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(NonScalarLossError):
            torch_autograd(lambda t: t**2, x, klong._backend)


class TestGradientForms:
    """Every documented form of the gradient operators, on both backends."""

    def _setup(self, klong):
        klong('w::2.0;b::3.0;loss::{(w^2)+(b^2)};f::{x^2};g::{+/x^2};v::[1.0 2.0 3.0]')

    def test_single_symbol(self, klong, backend):
        """loss:>w differentiates a niladic loss with respect to w."""
        self._setup(klong)
        assert np.isclose(to_scalar(klong('loss:>w')), 4.0, atol=get_atol(backend))
        assert np.isclose(to_scalar(klong('loss:>b')), 6.0, atol=get_atol(backend))

    def test_single_symbol_restores_value(self, klong, backend):
        self._setup(klong)
        klong('loss:>w')
        assert np.isclose(to_scalar(klong('w')), 2.0)

    def test_single_symbol_matches_list_form(self, klong, backend):
        self._setup(klong)
        assert np.isclose(to_scalar(klong('loss:>w')), to_scalar(klong('*loss:>[w]')),
                          atol=get_atol(backend))

    def test_symbol_list(self, klong, backend):
        self._setup(klong)
        r = klong('loss:>[w b]')
        np.testing.assert_allclose([to_scalar(r[0]), to_scalar(r[1])], [4.0, 6.0],
                                   atol=get_atol(backend))

    def test_vector_symbol(self, klong, backend):
        """A symbol that holds a vector, with a niladic loss."""
        klong('u::[1.0 2.0];gg::{+/u^2}')
        np.testing.assert_allclose(to_numpy(klong('gg:>u')), [2.0, 4.0], atol=get_atol(backend))

    def test_point_value(self, klong, backend):
        self._setup(klong)
        assert np.isclose(to_scalar(klong('f:>3')), 6.0, atol=get_atol(backend))
        assert np.isclose(to_scalar(klong('f:>3.0')), 6.0, atol=get_atol(backend))

    def test_point_held_in_a_variable(self, klong, backend):
        """A function of x given a variable takes the variable's value as the point."""
        self._setup(klong)
        klong('p::3.0')
        assert np.isclose(to_scalar(klong('f:>p')), 6.0, atol=get_atol(backend))
        np.testing.assert_allclose(to_numpy(klong('g:>v')), [2.0, 4.0, 6.0], atol=get_atol(backend))

    def test_point_vector(self, klong, backend):
        self._setup(klong)
        np.testing.assert_allclose(to_numpy(klong('g:>[1 2 3]')), [2.0, 4.0, 6.0],
                                   atol=get_atol(backend))

    def test_numeric_nabla_and_ascii_name(self, klong, backend):
        """3∇f and .grad(f;3) are the numeric gradient (coarse in float32)."""
        self._setup(klong)
        for src in ('3∇f', '.grad(f;3)'):
            assert np.isclose(to_scalar(klong(src)), 6.0, atol=0.02), src
        for src in ('[1 2 3]∇g', '.grad(g;[1 2 3])', 'v∇g', '.grad(g;v)'):
            np.testing.assert_allclose(to_numpy(klong(src)), [2.0, 4.0, 6.0],
                                       atol=0.02, err_msg=src)

    def test_jacobian_forms(self, klong, backend):
        self._setup(klong)
        for src in ('[1 2]∂{x^2}', '.jacobian({x^2};[1.0 2.0])'):
            np.testing.assert_allclose(to_numpy(klong(src)), [[2.0, 0.0], [0.0, 4.0]],
                                       atol=0.02, err_msg=src)


class TestGradcheck:
    """Tests for torch.autograd.gradcheck verification."""

    def test_gradcheck_basic_scalar(self, klong, backend):
        """Test .gradcheck() with basic scalar function."""
        if backend != 'torch':
            pytest.skip("Requires torch backend")
        if klong._backend.device.type != 'cpu':
            pytest.skip("gradcheck requires CPU device (float64 support)")
        klong('f::{x^2}')
        result = klong('.gradcheck(f;3.0)')
        assert result == 1

    def test_gradcheck_vector_input(self, klong, backend):
        """Test .gradcheck() with vector input."""
        if backend != 'torch':
            pytest.skip("Requires torch backend")
        if klong._backend.device.type != 'cpu':
            pytest.skip("gradcheck requires CPU device (float64 support)")
        klong('g::{+/x^2}')
        result = klong('.gradcheck(g;[1.0 2.0 3.0])')
        assert result == 1

    def test_gradcheck_polynomial(self, klong, backend):
        """Test .gradcheck() with polynomial function."""
        if backend != 'torch':
            pytest.skip("Requires torch backend")
        if klong._backend.device.type != 'cpu':
            pytest.skip("gradcheck requires CPU device (float64 support)")
        klong('p::{(x^3)+(2*x^2)+x}')
        result = klong('.gradcheck(p;2.0)')
        assert result == 1

    def test_gradcheck_backend_direct(self, klong, backend):
        """Test gradcheck() directly on backend."""
        if backend != 'torch':
            pytest.skip("Requires torch backend")
        if klong._backend.device.type != 'cpu':
            pytest.skip("gradcheck requires CPU device (float64 support)")
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
        with pytest.raises(KlongError) as e:
            klong('.gradcheck(f;3.0)')
        assert 'torch backend' in str(e.value)
        assert '--backend torch' in (e.value.hint or '')
        assert 'USE_TORCH' not in str(e.value) + (e.value.hint or '')


class TestExport:
    """Tests for .export."""

    def test_export_with_path(self, klong, backend):
        """Test .export() returns dict with graph info."""
        if backend != 'torch':
            pytest.skip("Requires torch backend")
        klong('f::{x^2}')
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pt2')
            try:
                result = klong(f'.export(f;3.0;"{path}")')
            except Exception:
                raise
            # Should return a dict
            assert isinstance(result, dict)
            assert 'compiled_fn' in result
            # Export may fail on some systems, but compiled_fn should work
            if result.get('export_path'):
                assert os.path.exists(path)

    def test_export_writes_a_loadable_file(self, klong, backend):
        """torch 2.15 export takes a Module: a Klong function must still export."""
        if backend != 'torch':
            pytest.skip("Requires torch backend")
        klong('f::{x^2}')
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pt2')
            try:
                result = klong(f'.export(f;3.0;"{path}")')
            except Exception:
                raise
            assert not result.get('export_error'), result.get('export_error')
            assert result['export_path'] == path
            assert os.path.exists(path)
            # on MPS a whole-number power is computed by multiplication, so the graph has mul and no pow
            assert 'pow' in result['graph'] or 'mul' in result['graph']
            loaded = torch.export.load(path).module()
            assert np.isclose(float(loaded(torch.tensor(3.0))), 9.0, atol=1e-5)

    def test_compile_is_gone(self, klong, backend):
        """.compile, .compilex and .cmodes were removed: a call is the ordinary undefined-function error."""
        for src in ('.compile(f;3.0)', '.compilex(f;3.0;:{["mode" "default"]})', '.cmodes()'):
            klong('f::{x^2}')
            with pytest.raises(KlongError):
                klong(src)

    def test_export_fails_on_numpy(self, klong, backend):
        """Test that .export() raises error on numpy backend."""
        if backend != 'numpy':
            pytest.skip("Test for numpy backend only")
        klong('f::{x^2}')
        with pytest.raises(KlongError):
            klong('.export(f;3.0;"test.pt2")')


class TestMatrixVectorGradient:
    """each-dyad (f') and over (f/) must support a differentiable matrix-vector
    contraction A . RC (sum over axis 0), on both backends. Regression guard for
    the autograd-safe each-dyad + axis-0 reduce fix."""

    def test_over_2d_reduces_axis0(self, klong, backend):
        # +/ of a 2-D array folds over axis 0 (column-wise sum), matching numpy.
        klong['M'] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_allclose(to_numpy(klong('+/M')), [5.0, 7.0, 9.0],
                                   atol=get_atol(backend))

    def test_each_dyad_scales_rows(self, klong, backend):
        # A *' RC scales each row of RC by A[i]; +/ then sums them -> A . RC.
        klong['RC'] = np.array([[0.1, 0.2, 0.3], [0.0, -0.1, 0.4]])
        val = to_numpy(klong("{+/x*'RC}([0.6 0.4])"))
        np.testing.assert_allclose(val, [0.06, 0.08, 0.34], atol=get_atol(backend))

    def test_matrix_vector_gradient(self, klong, backend):
        # gradient of  +/(A . RC)^2  wrt A must flow through each-dyad + over.
        RC = np.array([[0.1, 0.2, 0.3], [0.0, -0.1, 0.4]])
        klong['RC'] = RC
        grad = to_numpy(klong("{+/(+/x*'RC)^2}:>[0.6 0.4]"))

        def loss(a):
            d = (a[:, None] * RC).sum(0)
            return float((d ** 2).sum())

        a0 = np.array([0.6, 0.4])
        eps = 1e-4
        fd = np.array([(loss(a0 + eps * e) - loss(a0 - eps * e)) / (2 * eps)
                       for e in np.eye(2)])
        np.testing.assert_allclose(grad, fd, atol=get_atol(backend) * 10)

    def test_subtract_over_2d_reduces_axis0(self, klong, backend):
        # -/ of a 2-D array folds over axis 0 element-wise: r0-r1-r2.
        # Regression guard: the axis-0 reduce default must not break -/ on torch
        # (a-b-c == a-(b+c)).
        klong['M'] = np.array([[10.0, 20.0], [1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(to_numpy(klong('-/M')), [6.0, 14.0],
                                   atol=get_atol(backend))

    def test_divide_over_2d_reduces_axis0(self, klong, backend):
        # %/ of a 2-D array folds over axis 0 element-wise: r0/r1/r2.
        # (a/b/c == a/(b*c)) — same regression guard as subtract.
        klong['M'] = np.array([[12.0, 24.0], [2.0, 4.0], [3.0, 2.0]])
        np.testing.assert_allclose(to_numpy(klong('%/M')), [2.0, 3.0],
                                   atol=get_atol(backend))
