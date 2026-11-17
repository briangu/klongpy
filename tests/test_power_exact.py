import unittest

from klongpy import KlongInterpreter


class TestExactIntegerPower(unittest.TestCase):
    """A scalar integer to a non-negative integer power is exact (ticket: pow-scalar)."""

    def setUp(self):
        self.klong = KlongInterpreter()

    def test_scalar_exact_beyond_float_precision(self):
        self.assertEqual(self.klong("7^22"), 3909821048582988049)
        self.assertEqual([int(v) for v in self.klong("[7]^[22]")], [3909821048582988049])

    def test_scalar_exact_beyond_64_bits(self):
        self.assertEqual(self.klong("7^40"), 7 ** 40)
        self.assertEqual(int(self.klong("(-3)^41")), (-3) ** 41)

    def test_compiled_and_interpreted_agree(self):
        self.klong("f::{x^y}")
        self.assertEqual(self.klong("f(7;22)"), 7 ** 22)
        self.assertEqual(self.klong("{x^22}(7)"), 7 ** 22)
        self.assertEqual(self.klong("+/{x^22}'[7 7]"), 2 * 7 ** 22)

    def test_negative_or_fractional_exponent_is_a_float(self):
        self.assertEqual(float(self.klong("2^-2")), 0.25)
        self.assertAlmostEqual(float(self.klong("2^0.5")), 2 ** 0.5)
        self.assertEqual(float(self.klong("2.5^2")), 6.25)


if __name__ == "__main__":
    unittest.main()
