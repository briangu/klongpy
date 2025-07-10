import unittest
import numpy as np
from tests.utils import to_numpy

from klongpy import backend


class TestAutograd(unittest.TestCase):
    """Autograd gradient checks using numpy and torch backends."""
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
        grad = to_numpy(g(x))
        np.testing.assert_allclose(np.array(grad), np.array([[7.0, 11.0], [9.0, 13.0]]))

    def _check_scalar_square_grad(self, name: str):
        """Verify ∂(x²)/∂x = 2x for a scalar input."""
        try:
            backend.set_backend(name)
        except ImportError:
            raise unittest.SkipTest(f"{name} backend not available")
        b = backend.current()

        def f(x):
            return b.mul(x, x)

        g = b.grad(f)
        x = b.array(3.0, requires_grad=True)
        grad = to_numpy(g(x))
        np.testing.assert_allclose(np.array(grad), np.array(6.0))

    def _check_vector_elemwise_grad(self, name: str):
        """Verify gradient of ∑(x+1)(x+2) = 2x+3 via the chain rule."""
        try:
            backend.set_backend(name)
        except ImportError:
            raise unittest.SkipTest(f"{name} backend not available")
        b = backend.current()

        def f(x):
            return b.sum(b.mul(b.add(x, 1), b.add(x, 2)))

        g = b.grad(f)
        x = b.array([0.0, 1.0, 2.0], requires_grad=True)
        grad = to_numpy(g(x))
        expected = 2 * np.array([0.0, 1.0, 2.0]) + 3
        np.testing.assert_allclose(np.array(grad), expected)

    def _check_mixed_args_grad(self, name: str):
        """Verify gradient of the dot product x·y with respect to each argument."""
        try:
            backend.set_backend(name)
        except ImportError:
            raise unittest.SkipTest(f"{name} backend not available")
        b = backend.current()

        def f(x, y):
            return b.sum(b.mul(x, y))

        gx = b.grad(f, wrt=0)
        gy = b.grad(f, wrt=1)
        x = b.array([1.0, 2.0, 3.0], requires_grad=True)
        y = b.array([4.0, 5.0, 6.0], requires_grad=True)
        gradx = to_numpy(gx(x, y))
        grady = to_numpy(gy(x, y))
        np.testing.assert_allclose(np.array(gradx), np.array([4.0, 5.0, 6.0]))
        np.testing.assert_allclose(np.array(grady), np.array([1.0, 2.0, 3.0]))

    def _check_stop_grad(self, name: str):
        """Verify gradients ignore values detached with ``stop``."""
        try:
            backend.set_backend(name)
        except ImportError:
            raise unittest.SkipTest(f"{name} backend not available")
        b = backend.current()

        def f(x):
            return b.sum(b.mul(b.stop(x), x))

        g = b.grad(f)
        x = b.array([2.0, 3.0], requires_grad=True)
        grad = to_numpy(g(x))
        np.testing.assert_allclose(np.array(grad), np.array([2.0, 3.0]))

    def test_matrix_grad_numpy(self):
        self._check_matrix_grad("numpy")

    def test_matrix_grad_torch(self):
        self._check_matrix_grad("torch")

    def test_scalar_grad_numpy(self):
        self._check_scalar_square_grad("numpy")

    def test_scalar_grad_torch(self):
        self._check_scalar_square_grad("torch")

    def test_vector_elemwise_grad_numpy(self):
        self._check_vector_elemwise_grad("numpy")

    def test_vector_elemwise_grad_torch(self):
        self._check_vector_elemwise_grad("torch")

    def test_mixed_args_grad_numpy(self):
        self._check_mixed_args_grad("numpy")

    def test_mixed_args_grad_torch(self):
        self._check_mixed_args_grad("torch")

    def test_stop_grad_numpy(self):
        self._check_stop_grad("numpy")

    def test_stop_grad_torch(self):
        self._check_stop_grad("torch")


if __name__ == "__main__":
    unittest.main()
