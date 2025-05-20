import os
import unittest
import importlib

# Skip tests if torch is not available
try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - torch is optional
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class TestTorchBackend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ["USE_TORCH"] = "1"
        import klongpy.backend as backend
        import klongpy.core as core
        importlib.reload(backend)
        importlib.reload(core)
        cls.backend = backend
        cls.core = core

    def test_string_asarray(self):
        arr = self.core.kg_asarray("hello")
        self.assertTrue(self.backend.np.isarray(arr))
        self.assertEqual(arr.dtype, object)
        self.assertEqual("".join(arr), "hello")
        import numpy as np
        self.assertIsInstance(arr, np.ndarray)
        self.assertFalse(isinstance(arr, torch.Tensor))

    def test_numeric_asarray_and_add(self):
        arr = self.core.kg_asarray([1, 2, 3])
        self.assertIsInstance(arr, torch.Tensor)
        res = self.backend.np.add.reduce(arr)
        self.assertEqual(res.item(), 6)


if __name__ == "__main__":  # pragma: no cover - manual run
    unittest.main()
