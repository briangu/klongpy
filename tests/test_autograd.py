import unittest
import numpy as np
from klongpy import KlongInterpreter

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

class TestAutograd(unittest.TestCase):
    def test_scalar_grad(self):
        klong = KlongInterpreter()
        klong['sin'] = lambda x: np.sin(x)
        klong['cos'] = lambda x: np.cos(x)
        klong('g::∇{sin(x)+x*x}')
        r = klong('g(3.14)')
        self.assertTrue(np.isclose(r, 2*3.14 + np.cos(3.14), atol=1e-3))

    def test_scalar_grad_high_precision(self):
        """autodiff should provide very accurate gradients for scalars."""
        klong = KlongInterpreter()
        klong['exp'] = lambda x: np.exp(x)
        klong('g::∇{exp(x)}')
        r = klong('g(50)')
        self.assertTrue(np.isclose(r, np.exp(50.0), atol=1e-12))

    def test_scalar_grad_extended_ufuncs(self):
        klong = KlongInterpreter()
        klong['log'] = lambda x: np.log(x)
        klong['sqrt'] = lambda x: np.sqrt(x)
        klong['tanh'] = lambda x: np.tanh(x)
        klong('g::∇{log(x)+sqrt(x)+tanh(x)}')
        value = 2.5
        r = klong(f'g({value})')
        expected = (1.0 / value) + (0.5 / np.sqrt(value)) + (1.0 / np.cosh(value) ** 2)
        self.assertTrue(np.isclose(r, expected, atol=1e-9))

    def test_scalar_grad_python_operators(self):
        klong = KlongInterpreter()
        klong('g::∇{(x*x*x) - 3*x + 5}')
        value = 4.0
        r = klong(f'g({value})')
        expected = 3 * (value ** 2) - 3
        self.assertTrue(np.isclose(r, expected, atol=1e-9))

    def test_scalar_grad_numeric_fallback(self):
        klong = KlongInterpreter()
        klong['relu'] = lambda x: np.maximum(x, 0.0)
        klong('g::∇{relu(x)}')
        pos = klong('g(3.0)')
        neg = klong('g(-3.0)')
        self.assertTrue(np.isclose(pos, 1.0, atol=1e-6))
        self.assertTrue(np.isclose(neg, 0.0, atol=1e-6))

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

        A = torch.arange(4, dtype=torch.float64, requires_grad=True).reshape(2,2)
        B = torch.arange(4, dtype=torch.float64).reshape(2,2)
        loss = (A * B).sum()
        loss.backward()
        self.assertTrue(np.allclose(r, A.grad.numpy(), atol=1e-3))

if __name__ == "__main__":
    unittest.main()
