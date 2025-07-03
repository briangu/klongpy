import unittest
import numpy as np
import klongpy as k

class TestRustDtype(unittest.TestCase):
    def test_set_dtype_f32(self):
        k.set_dtype('f32')
        a = np.arange(16, dtype=np.float32)
        b = np.ones_like(a)
        out = k.add(a, b)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(out, a + b)
        k.set_dtype('f64')

if __name__ == '__main__':
    unittest.main()
