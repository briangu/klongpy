import unittest
import numpy as np
import klongpy as k

class TestRustArithmetic(unittest.TestCase):
    def test_add(self):
        a = np.arange(16, dtype=np.float64)
        b = np.ones_like(a)
        out = k.add(a, b)
        np.testing.assert_array_equal(out, a + b)

    def test_subtract(self):
        a = np.arange(16, dtype=np.float64)
        b = np.ones_like(a)
        out = k.subtract(a, b)
        np.testing.assert_array_equal(out, a - b)

    def test_multiply(self):
        a = np.arange(16, dtype=np.float64)
        b = np.ones_like(a) * 2
        out = k.multiply(a, b)
        np.testing.assert_array_equal(out, a * b)

    def test_divide(self):
        a = np.arange(1, 17, dtype=np.float64)
        b = np.ones_like(a) * 2
        out = k.divide(a, b)
        np.testing.assert_array_equal(out, a / b)

if __name__ == '__main__':
    unittest.main()
