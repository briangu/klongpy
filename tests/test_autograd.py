import unittest
import numpy as np

from klongpy import backend


class TestAutograd(unittest.TestCase):
    def _check_matrix_grad(self, name: str):
        try:
            backend.set_backend(name)
        except ImportError:
            raise unittest.SkipTest(f"{name} backend not available")
        b = backend.current()

        def f(x):
            return b.sum(b.matmul(x, x))

        g = b.grad(f)
        x = b.array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        grad = g(x)
        if hasattr(grad, "detach"):
            grad = grad.detach().cpu().numpy()
        np.testing.assert_allclose(np.array(grad), np.array([[7.0, 11.0], [9.0, 13.0]]))

    def test_matrix_grad_numpy(self):
        self._check_matrix_grad("numpy")

    def test_matrix_grad_torch(self):
        self._check_matrix_grad("torch")


if __name__ == "__main__":
    unittest.main()
