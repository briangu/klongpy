import unittest
import numpy as np
from klongpy import KlongInterpreter
from klongpy.backend import use_torch
from klongpy.autograd import numeric_grad, grad_of_fn, autograd_of_fn
from klongpy.core import KGLambda, KGCall, KGSym, KGFn

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False


class TestNumericGrad(unittest.TestCase):
    """Tests for the numeric_grad function."""

    def test_scalar_function(self):
        """Test numeric gradient of a scalar function."""
        # f(x) = x^2, f'(x) = 2x
        result = numeric_grad(lambda x: x[0]**2, np.array([3.0]))
        self.assertTrue(np.isclose(result[0], 6.0, atol=1e-3))

    def test_multidimensional_function(self):
        """Test numeric gradient of a multidimensional function."""
        # f(x) = sum(x^2), gradient = 2*x
        result = numeric_grad(lambda x: np.sum(x**2), np.array([1.0, 2.0, 3.0]))
        expected = np.array([2.0, 4.0, 6.0])
        self.assertTrue(np.allclose(result, expected, atol=1e-3))

    def test_2d_array(self):
        """Test numeric gradient with 2D array."""
        # f(x) = sum(x^2)
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = numeric_grad(lambda x: np.sum(x**2), x)
        expected = 2 * x
        self.assertTrue(np.allclose(result, expected, atol=1e-3))

    def test_custom_epsilon(self):
        """Test numeric gradient with custom epsilon."""
        result = numeric_grad(lambda x: x[0]**2, np.array([3.0]), eps=1e-8)
        self.assertTrue(np.isclose(result[0], 6.0, atol=1e-5))


class TestGradOfFn(unittest.TestCase):
    """Tests for the grad_of_fn function."""

    def test_with_callable(self):
        """Test grad_of_fn with a plain Python callable."""
        klong = KlongInterpreter()
        result = grad_of_fn(klong, lambda x: np.sum(x**2), np.array([1.0, 2.0, 3.0]))
        expected = np.array([2.0, 4.0, 6.0])
        self.assertTrue(np.allclose(result, expected, atol=1e-3))

    def test_with_kgsym(self):
        """Test grad_of_fn with a KGSym (symbol reference)."""
        klong = KlongInterpreter()
        klong('f::{+/x^2}')
        fn_sym = KGSym('f')
        result = grad_of_fn(klong, fn_sym, np.array([1.0, 2.0, 3.0]))
        expected = np.array([2.0, 4.0, 6.0])
        self.assertTrue(np.allclose(result, expected, atol=1e-3))

    def test_with_kglambda(self):
        """Test grad_of_fn with a KGLambda."""
        klong = KlongInterpreter()
        fn = KGLambda(lambda x: np.sum(x**2))
        result = grad_of_fn(klong, fn, np.array([1.0, 2.0, 3.0]))
        expected = np.array([2.0, 4.0, 6.0])
        self.assertTrue(np.allclose(result, expected, atol=1e-3))

    def test_with_kgfn(self):
        """Test grad_of_fn with a KGFn."""
        klong = KlongInterpreter()
        klong('f::{+/x^2}')
        fn = klong._context[KGSym('f')]
        result = grad_of_fn(klong, fn, np.array([1.0, 2.0, 3.0]))
        expected = np.array([2.0, 4.0, 6.0])
        self.assertTrue(np.allclose(result, expected, atol=1e-3))

    def test_with_kgcall(self):
        """Test grad_of_fn with a KGCall."""
        klong = KlongInterpreter()
        klong('f::{+/x^2}')
        fn = klong._context[KGSym('f')]
        fn_call = KGCall(fn.a, [], fn.arity)
        result = grad_of_fn(klong, fn_call, np.array([1.0, 2.0, 3.0]))
        expected = np.array([2.0, 4.0, 6.0])
        self.assertTrue(np.allclose(result, expected, atol=1e-3))


class TestAutogradOfFn(unittest.TestCase):
    """Tests for the autograd_of_fn function (numpy mode)."""

    def test_with_callable(self):
        """Test autograd_of_fn with a plain Python callable."""
        klong = KlongInterpreter()
        result = autograd_of_fn(klong, lambda x: np.sum(x**2), np.array([1.0, 2.0, 3.0]))
        expected = np.array([2.0, 4.0, 6.0])
        self.assertTrue(np.allclose(result, expected, atol=1e-3))

    def test_with_kgsym(self):
        """Test autograd_of_fn with a KGSym."""
        klong = KlongInterpreter()
        klong('f::{+/x^2}')
        fn_sym = KGSym('f')
        result = autograd_of_fn(klong, fn_sym, np.array([1.0, 2.0, 3.0]))
        expected = np.array([2.0, 4.0, 6.0])
        self.assertTrue(np.allclose(result, expected, atol=1e-3))

    def test_with_kglambda(self):
        """Test autograd_of_fn with a KGLambda."""
        klong = KlongInterpreter()
        fn = KGLambda(lambda x: np.sum(x**2))
        result = autograd_of_fn(klong, fn, np.array([1.0, 2.0, 3.0]))
        expected = np.array([2.0, 4.0, 6.0])
        self.assertTrue(np.allclose(result, expected, atol=1e-3))

    def test_with_kgfn(self):
        """Test autograd_of_fn with a KGFn."""
        klong = KlongInterpreter()
        klong('f::{+/x^2}')
        fn = klong._context[KGSym('f')]
        result = autograd_of_fn(klong, fn, np.array([1.0, 2.0, 3.0]))
        expected = np.array([2.0, 4.0, 6.0])
        self.assertTrue(np.allclose(result, expected, atol=1e-3))

    def test_with_kgcall(self):
        """Test autograd_of_fn with a KGCall."""
        klong = KlongInterpreter()
        klong('f::{+/x^2}')
        fn = klong._context[KGSym('f')]
        fn_call = KGCall(fn.a, [], fn.arity)
        result = autograd_of_fn(klong, fn_call, np.array([1.0, 2.0, 3.0]))
        expected = np.array([2.0, 4.0, 6.0])
        self.assertTrue(np.allclose(result, expected, atol=1e-3))


class TestAutograd(unittest.TestCase):
    """Tests for the ∇ operator."""

    def test_scalar_grad(self):
        klong = KlongInterpreter()
        klong['sin'] = lambda x: np.sin(x)
        klong['cos'] = lambda x: np.cos(x)
        klong('g::∇{sin(x)+x*x}')
        r = klong('g(3.14)')
        self.assertTrue(np.isclose(r, 2*3.14 + np.cos(3.14), atol=1e-3))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch required")
    def test_scalar_grad_torch(self):
        klong = KlongInterpreter()
        klong['sin'] = lambda x: np.sin(x)
        klong['cos'] = lambda x: np.cos(x)
        klong('g::∇{sin(x)+x*x}')
        r = klong('g(3.14)')
        x = torch.tensor(3.14, dtype=torch.float64, requires_grad=True)
        f = torch.sin(x) + x * x
        f.backward()
        self.assertTrue(np.isclose(r, x.grad.item(), atol=1e-3))

    def test_array_grad(self):
        klong = KlongInterpreter()
        klong('x::˙!5')
        klong('loss::{+/x*x}')
        r = klong('x ∇ loss')
        self.assertTrue(np.allclose(r, np.array([0,2,4,6,8]), atol=1e-3))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch required")
    def test_array_grad_torch(self):
        klong = KlongInterpreter()
        klong('x::˙!5')
        klong('loss::{+/x*x}')
        r = klong('x ∇ loss')

        x = torch.arange(5, dtype=torch.float64, requires_grad=True)
        loss = (x * x).sum()
        loss.backward()
        self.assertTrue(np.allclose(r, x.grad.numpy(), atol=1e-3))

    def test_matrix_grad(self):
        klong = KlongInterpreter()
        klong('A::˙[2 2]:^!4')
        klong('B::[2 2]:^!4')
        r = klong('(A ∇ {+/(+/ (A*B)) })')
        self.assertTrue(np.allclose(r, klong('B'), atol=1e-3))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch required")
    def test_matrix_grad_torch(self):
        klong = KlongInterpreter()
        klong('A::˙[2 2]:^!4')
        klong('B::[2 2]:^!4')
        r = klong('(A ∇ {+/(+/ (A*B)) })')

        # Use view and retain_grad to properly track gradients
        A_base = torch.arange(4, dtype=torch.float64, requires_grad=True)
        A = A_base.view(2, 2)
        A.retain_grad()
        B = torch.arange(4, dtype=torch.float64).reshape(2,2)
        loss = (A * B).sum()
        loss.backward()
        self.assertTrue(np.allclose(r, A.grad.numpy(), atol=1e-3))


class TestAutogradOperator(unittest.TestCase):
    """Tests for the :> autograd operator."""

    def test_scalar_autograd_numpy(self):
        """Test :> operator with scalar input (numpy mode)."""
        klong = KlongInterpreter()
        # derivative of x^2 at x=3 should be 6
        r = klong('{x^2}:>3.0')
        self.assertTrue(np.isclose(r, 6.0, atol=1e-3))

    def test_scalar_autograd_cubic(self):
        """Test :> operator with cubic function."""
        klong = KlongInterpreter()
        # derivative of x^3 at x=2 should be 12
        r = klong('{x^3}:>2.0')
        self.assertTrue(np.isclose(r, 12.0, atol=1e-3))

    def test_array_autograd(self):
        """Test :> operator with array input."""
        klong = KlongInterpreter()
        # gradient of sum(x^2) at [1,2,3] should be [2,4,6]
        r = klong('{+/x^2}:>[1.0 2.0 3.0]')
        self.assertTrue(np.allclose(r, [2.0, 4.0, 6.0], atol=1e-3))

    def test_autograd_with_named_function(self):
        """Test :> operator with a named Klong function."""
        klong = KlongInterpreter()
        klong('sq::{x^2}')
        r = klong('sq:>4.0')
        self.assertTrue(np.isclose(r, 8.0, atol=1e-3))

    def test_autograd_sum_of_squares(self):
        """Test :> with sum of squares loss function."""
        klong = KlongInterpreter()
        klong('loss::{+/x^2}')
        r = klong('loss:>[1.0 2.0 3.0 4.0]')
        # gradient is 2*x
        expected = np.array([2.0, 4.0, 6.0, 8.0])
        self.assertTrue(np.allclose(r, expected, atol=1e-3))

    def test_autograd_linear_function(self):
        """Test :> with linear function (constant gradient)."""
        klong = KlongInterpreter()
        # derivative of 3*x at any point is 3
        r = klong('{3*x}:>5.0')
        self.assertTrue(np.isclose(r, 3.0, atol=1e-3))

    def test_autograd_polynomial(self):
        """Test :> with polynomial function."""
        klong = KlongInterpreter()
        # f(x) = x^3 + 2*x^2 + x
        # f'(x) = 3*x^2 + 4*x + 1
        # f'(2) = 12 + 8 + 1 = 21
        r = klong('{(x^3)+(2*x^2)+x}:>2.0')
        self.assertTrue(np.isclose(r, 21.0, atol=1e-2))

    @unittest.skipUnless(TORCH_AVAILABLE and use_torch, "torch backend required")
    def test_autograd_torch_exact(self):
        """Test :> operator uses exact PyTorch autograd when in torch mode."""
        klong = KlongInterpreter()

        # Use a more complex function where numeric gradient might differ
        r = klong('{x^2+2*x+1}:>3.0')

        # derivative of x^2 + 2x + 1 at x=3 is 2x + 2 = 8
        x = torch.tensor(3.0, requires_grad=True)
        y = x**2 + 2*x + 1
        y.backward()
        self.assertTrue(np.isclose(r.item() if hasattr(r, 'item') else r, x.grad.item(), atol=1e-5))


class TestAutogradComparison(unittest.TestCase):
    """Compare :> and ∇ operators."""

    def test_grad_and_autograd_equivalent(self):
        """Test that :> and ∇ give the same results for simple functions."""
        klong = KlongInterpreter()

        # Test with x^2
        r_grad = klong('3.0 ∇ {x^2}')
        r_autograd = klong('{x^2}:>3.0')
        self.assertTrue(np.isclose(r_grad, r_autograd, atol=1e-3))

    def test_grad_and_autograd_array(self):
        """Test that :> and ∇ give the same results for array inputs."""
        klong = KlongInterpreter()

        klong('x::[1.0 2.0 3.0]')
        klong('f::{+/x^2}')

        r_grad = klong('x ∇ f')
        r_autograd = klong('f:>x')

        self.assertTrue(np.allclose(r_grad, r_autograd, atol=1e-3))

    def test_grad_and_autograd_matrix(self):
        """Test that :> and ∇ give the same results for matrix inputs."""
        klong = KlongInterpreter()

        klong('M::˙[2 3]:^!6')
        # Use multiplication instead of power to avoid a bug in power with matrices
        klong('f::{+/,/x*x}')

        r_grad = klong('M ∇ f')
        r_autograd = klong('f:>M')

        self.assertTrue(np.allclose(r_grad, r_autograd, atol=1e-3))


class TestTorchAutograd(unittest.TestCase):
    """Tests for torch_autograd function (requires torch)."""

    def test_torch_autograd_not_in_torch_mode(self):
        """Test that torch_autograd raises error when not in torch mode."""
        if not use_torch:
            from klongpy.autograd import torch_autograd
            with self.assertRaises(RuntimeError):
                torch_autograd(lambda x: x**2, np.array([3.0]))

    @unittest.skipUnless(TORCH_AVAILABLE and use_torch, "torch backend required")
    def test_torch_autograd_with_tensor(self):
        """Test torch_autograd with tensor input."""
        from klongpy.autograd import torch_autograd
        x = torch.tensor([1.0, 2.0, 3.0])
        result = torch_autograd(lambda t: (t**2).sum(), x)
        expected = torch.tensor([2.0, 4.0, 6.0])
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    @unittest.skipUnless(TORCH_AVAILABLE and use_torch, "torch backend required")
    def test_torch_autograd_with_numpy(self):
        """Test torch_autograd with numpy array input."""
        from klongpy.autograd import torch_autograd
        x = np.array([1.0, 2.0, 3.0])
        result = torch_autograd(lambda t: (t**2).sum(), x)
        expected = np.array([2.0, 4.0, 6.0])
        self.assertTrue(np.allclose(result.numpy(), expected, atol=1e-5))

    @unittest.skipUnless(TORCH_AVAILABLE and use_torch, "torch backend required")
    def test_torch_autograd_with_list(self):
        """Test torch_autograd with list input."""
        from klongpy.autograd import torch_autograd
        x = [1.0, 2.0, 3.0]
        result = torch_autograd(lambda t: (t**2).sum(), x)
        expected = np.array([2.0, 4.0, 6.0])
        self.assertTrue(np.allclose(result.numpy(), expected, atol=1e-5))

    @unittest.skipUnless(TORCH_AVAILABLE and use_torch, "torch backend required")
    def test_torch_autograd_non_scalar_output(self):
        """Test that torch_autograd raises error for non-scalar output."""
        from klongpy.autograd import torch_autograd
        x = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            torch_autograd(lambda t: t**2, x)  # Returns vector, not scalar


if __name__ == "__main__":
    unittest.main()
