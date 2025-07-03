import unittest
import numpy as np
import klongpy as k

class TestRustMap(unittest.TestCase):
    def test_map_lambda(self):
        a = np.arange(5, dtype=np.float64)
        out = k.map(a, lambda x: x * 2)
        np.testing.assert_array_equal(out, a * 2)

if __name__ == '__main__':
    unittest.main()
