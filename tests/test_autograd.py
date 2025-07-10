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
