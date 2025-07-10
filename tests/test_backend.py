import unittest
import numpy as np

from klongpy import backend


class TestBackend(unittest.TestCase):
    def _check_grad(self, name: str):
        try:
            backend.set_backend(name)
        except ImportError:
            raise unittest.SkipTest(f"{name} backend not available")
        b = backend.current()

        def f(x):
            return b.sum(b.mul(b.add(x, 1), b.add(x, 1)))

        g = b.grad(f)
        x = b.array([1.0, 2.0, 3.0], requires_grad=True)
        grad = g(x)
        if hasattr(grad, "detach"):
            grad = grad.detach().cpu().numpy()
        np.testing.assert_allclose(np.array(grad), np.array([4.0, 6.0, 8.0]))

    def test_grad_numpy(self):
        self._check_grad("numpy")

    def test_grad_torch(self):
        self._check_grad("torch")

    def test_strings_not_differentiable(self):
        try:
            backend.set_backend("torch")
        except ImportError:
            raise unittest.SkipTest("torch backend not available")
        b = backend.current()

        def f(x):
            return b.add(x, ["a", "b", "c"])

        x = b.array([1.0, 2.0, 3.0], requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "not differentiable"):
            b.grad(f)(x)


if __name__ == "__main__":
    unittest.main()
