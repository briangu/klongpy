"""
Tests for advanced autograd features.

Tests multi-parameter gradients, Jacobians, and error messages.

Note: Optimizer tests are in examples/autograd/ since optimizers
are provided as example code, not core library features.
"""
import unittest
import numpy as np
from klongpy import KlongInterpreter
from klongpy.backend import use_torch
from tests.backend_compat import to_numpy

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Tolerance for numeric gradient comparison
GRAD_ATOL = 1e-2


class TestMultiGrad(unittest.TestCase):
    """Tests for multi-parameter gradient computation via :>[w b] syntax."""

    def test_two_scalar_params(self):
        """Test loss:>[w b] where loss = w^2 + b^2, grads = [2w, 2b]."""
        klong = KlongInterpreter()
        klong('w::2.0')
        klong('b::3.0')
        klong('loss::{(w^2)+(b^2)}')

        result = klong('loss:>[w b]')

        # Verify we get a list of two gradients
        self.assertEqual(len(result), 2)

        grad_w = to_numpy(result[0])
        grad_b = to_numpy(result[1])

        # grad of w^2 = 2w = 4, grad of b^2 = 2b = 6
        self.assertAlmostEqual(grad_w, 4.0, delta=GRAD_ATOL)
        self.assertAlmostEqual(grad_b, 6.0, delta=GRAD_ATOL)

    def test_three_params(self):
        """Test with three parameters."""
        klong = KlongInterpreter()
        klong('a::1.0')
        klong('b::2.0')
        klong('c::3.0')
        klong('loss::{(a^2)+(b^2)+(c^2)}')

        result = klong('loss:>[a b c]')

        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(to_numpy(result[0]), 2.0, delta=GRAD_ATOL)  # 2a
        self.assertAlmostEqual(to_numpy(result[1]), 4.0, delta=GRAD_ATOL)  # 2b
        self.assertAlmostEqual(to_numpy(result[2]), 6.0, delta=GRAD_ATOL)  # 2c

    def test_array_still_works(self):
        """Test that loss:>wts still works for single array (portfolio pattern)."""
        klong = KlongInterpreter()
        klong('wts::[0.25 0.25 0.25 0.25]')

        # Gradient of sum of squares
        result = klong('{+/x^2}:>wts')

        result_np = to_numpy(result)
        expected = np.array([0.5, 0.5, 0.5, 0.5])

        np.testing.assert_allclose(result_np, expected, atol=GRAD_ATOL)

    def test_mixed_not_symbols(self):
        """Test that loss:>[1 2 3] treats as point, not multi-param."""
        klong = KlongInterpreter()

        # This should be gradient at point [1, 2, 3], not multi-param mode
        result = klong('{+/x^2}:>[1 2 3]')

        result_np = to_numpy(result)
        expected = np.array([2.0, 4.0, 6.0])  # Gradient of sum of squares: 2x

        np.testing.assert_allclose(result_np, expected, atol=GRAD_ATOL)

    def test_linear_regression_gradients(self):
        """Test gradients for a simple linear regression loss."""
        klong = KlongInterpreter()
        klong('w::1.0')
        klong('b::0.0')
        klong('X::[1 2 3]')
        klong('Y::[3 5 7]')  # Y = 2*X + 1

        # MSE loss: mean((w*X + b - Y)^2)
        klong('loss::{(+/((w*X)+b-Y)^2)%3}')

        result = klong('loss:>[w b]')

        # At w=1, b=0: predictions = [1,2,3], errors = [-2,-3,-4]
        # dL/dw should push w toward 2
        # dL/db should push b toward 1
        grad_w = to_numpy(result[0])
        grad_b = to_numpy(result[1])

        # The gradients should be non-zero and have correct sign
        self.assertLess(grad_w, 0)  # Loss decreases as w increases
        self.assertLess(grad_b, 0)  # Loss decreases as b increases


class TestMultiGradTorch(unittest.TestCase):
    """Torch-specific tests for multi-parameter gradient."""

    @unittest.skipUnless(TORCH_AVAILABLE and use_torch, "Requires torch backend")
    def test_exact_gradients(self):
        """Test that torch gives exact gradients (not numeric)."""
        klong = KlongInterpreter()
        klong('w::2.0')
        klong('b::3.0')
        klong('loss::{(w^2)+(b^2)}')

        result = klong('loss:>[w b]')

        # With torch, gradients should be exact
        grad_w = float(result[0])
        grad_b = float(result[1])

        self.assertEqual(grad_w, 4.0)  # Exact, not approximate
        self.assertEqual(grad_b, 6.0)


class TestJacobian(unittest.TestCase):
    """Tests for Jacobian computation via ∂ operator and .jacobian() function."""

    def test_element_wise_squared(self):
        """Test Jacobian of x^2 (element-wise)."""
        klong = KlongInterpreter()
        klong('f::{x^2}')

        # Jacobian of x^2 at [1,2] should be diag([2, 4])
        result = klong('[1 2]∂f')
        result_np = to_numpy(result)

        expected = np.array([[2., 0.], [0., 4.]])
        np.testing.assert_allclose(result_np, expected, atol=GRAD_ATOL)

    def test_partial_operator(self):
        """Test ∂ operator syntax (point∂function)."""
        klong = KlongInterpreter()
        klong('g::{x^2}')

        result = klong('[3 4]∂g')
        result_np = to_numpy(result)

        # Jacobian at [3,4] should be diag([6, 8])
        expected = np.array([[6., 0.], [0., 8.]])
        np.testing.assert_allclose(result_np, expected, atol=GRAD_ATOL)

    def test_system_function(self):
        """Test .jacobian(f;x) syntax."""
        klong = KlongInterpreter()
        klong('h::{x^2}')

        result = klong('.jacobian(h;[1 2])')
        result_np = to_numpy(result)

        expected = np.array([[2., 0.], [0., 4.]])
        np.testing.assert_allclose(result_np, expected, atol=GRAD_ATOL)

    def test_scalar_function(self):
        """Test Jacobian of scalar-valued function (gradient vector)."""
        klong = KlongInterpreter()
        klong('f::{+/x^2}')  # sum of squares, scalar output

        result = klong('[1 2 3]∂f')
        result_np = to_numpy(result).flatten()  # Flatten to handle shape differences

        # Gradient of sum of squares: [2, 4, 6]
        expected = np.array([2., 4., 6.])
        np.testing.assert_allclose(result_np, expected, atol=GRAD_ATOL)

    def test_multi_param_jacobian(self):
        """Test [w b]∂f syntax for multi-parameter Jacobians."""
        klong = KlongInterpreter()
        klong('w::[1.0 2.0]')
        klong('b::[3.0 4.0]')
        # f returns w^2 (element-wise), so output is [w0^2, w1^2]
        klong('f::{w^2}')

        result = klong('[w b]∂f')

        # Should get list of two Jacobians
        self.assertEqual(len(result), 2)

        # Jacobian w.r.t. w: df/dw
        # f = [w0^2, w1^2]
        # df0/dw0 = 2*w0 = 2, df0/dw1 = 0
        # df1/dw0 = 0, df1/dw1 = 2*w1 = 4
        jac_w = to_numpy(result[0])
        expected_w = np.array([[2., 0.], [0., 4.]])
        np.testing.assert_allclose(jac_w, expected_w, atol=GRAD_ATOL)

        # Jacobian w.r.t. b: df/db = 0 (f doesn't depend on b)
        jac_b = to_numpy(result[1])
        expected_b = np.array([[0., 0.], [0., 0.]])
        np.testing.assert_allclose(jac_b, expected_b, atol=GRAD_ATOL)

    def test_multi_param_jacobian_scalar_params(self):
        """Test multi-param Jacobian with scalar parameters."""
        klong = KlongInterpreter()
        klong('a::2.0')
        klong('b::3.0')
        # f returns a*b (scalar output)
        klong('f::{a*b}')

        result = klong('[a b]∂f')

        self.assertEqual(len(result), 2)

        # Jacobian w.r.t. a: df/da = b = 3
        jac_a = float(np.asarray(to_numpy(result[0])).flat[0])
        self.assertAlmostEqual(jac_a, 3.0, delta=GRAD_ATOL)

        # Jacobian w.r.t. b: df/db = a = 2
        jac_b = float(np.asarray(to_numpy(result[1])).flat[0])
        self.assertAlmostEqual(jac_b, 2.0, delta=GRAD_ATOL)


class TestErrorMessages(unittest.TestCase):
    """Tests for clear error messages when autograd fails."""

    def test_non_scalar_loss_error(self):
        """Test clear error when loss returns a vector."""
        from klongpy.autograd import NonScalarLossError

        klong = KlongInterpreter()
        klong('f::{x^2}')  # Returns array, not scalar

        with self.assertRaises(NonScalarLossError) as ctx:
            klong('f:>[1 2 3]')

        self.assertIn("scalar", str(ctx.exception))
        self.assertIn("(3,)", str(ctx.exception))

    @unittest.skipUnless(TORCH_AVAILABLE and use_torch, "Requires torch backend")
    def test_error_message_suggests_reduction(self):
        """Test that error message suggests using sum or mean."""
        from klongpy.autograd import NonScalarLossError

        klong = KlongInterpreter()
        klong('f::{x^2}')

        with self.assertRaises(NonScalarLossError) as ctx:
            klong('f:>[1 2 3]')

        # Should suggest reduction operations
        self.assertIn("+/", str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
