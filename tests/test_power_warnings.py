"""Scalar integer power is exact; where it falls back to floats, no NumPy warning leaks."""
import unittest
import warnings

from klongpy import KlongInterpreter
from klongpy.core import KlongException


class TestPower(unittest.TestCase):
    def test_exact_scalar_integer_power(self):
        klong = KlongInterpreter()
        self.assertEqual(klong("2^64"), 2 ** 64)
        self.assertEqual(klong("10^400"), 10 ** 400)

    def test_overflow_is_a_range_error_without_a_numpy_warning(self):
        klong = KlongInterpreter()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(KlongException) as cm:
                klong("10^100000000")
            self.assertIn("cannot apply ^", str(cm.exception))
            klong("2^[10 64]")      # a list wraps to 64 bits: no invalid-cast warning
            klong("[10 10]^100000000")


if __name__ == "__main__":
    unittest.main()
