"""Iterate, scan-iterate, split and converge must not loop forever."""
import math
import signal
import unittest

from klongpy import KlongInterpreter
from klongpy.core import KGUndefined  # noqa: F401  (import check only)


class _Timeout(BaseException):
    """BaseException so the compiled-path `except Exception` cannot swallow it."""


def _alarm(*_):
    raise _Timeout()


class TestIterationTerminates(unittest.TestCase):

    def setUp(self):
        self._old = signal.signal(signal.SIGALRM, _alarm)

    def tearDown(self):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._old)

    def run_kg(self, src):
        klong = KlongInterpreter()
        signal.alarm(3)
        try:
            return klong(src)
        except _Timeout:
            self.fail(f"{src!r} did not terminate")
        finally:
            signal.alarm(0)

    def assert_raises_kg(self, src):
        with self.assertRaises(Exception) as cm:
            self.run_kg(src)
        self.assertNotIsInstance(cm.exception, (ZeroDivisionError, AttributeError, TypeError, ValueError, AssertionError), src)

    # iterate
    def test_iterate_negative_count_raises(self):
        self.assert_raises_kg("(-1){x+1}:*0")
        self.assert_raises_kg("a::-1;a{x+1}:*0")

    def test_iterate_fractional_count_raises(self):
        self.assert_raises_kg("0.5{x+1}:*0")
        self.assert_raises_kg("2.5{x+1}:*0")

    def test_iterate_integral_float_count(self):
        self.assertEqual(self.run_kg("2.0{x+1}:*5"), 7)
        self.assertEqual(self.run_kg("0.0{x+1}:*5"), 5)
        self.assertEqual(self.run_kg("3.0{x+1}:*0"), 3)

    def test_iterate_numpy_int_count(self):
        self.assertEqual(self.run_kg("(3+0){x+1}:*0"), 3)
        self.assertEqual(self.run_kg("(6%2){x+1}:*0"), 3)

    def test_iterate_numpy_int_count_nonzero_start(self):
        self.assertEqual(self.run_kg("(3+0){x+1}:*5"), 8)

    def test_iterate_zero_and_positive_unchanged(self):
        self.assertEqual(self.run_kg("0{x+1}:*5"), 5)
        self.assertEqual(self.run_kg("2{x+1}:*0"), 2)
        self.assertEqual(list(self.run_kg("3{1,x}:*[]")), [1, 1, 1])

    # scan-iterate
    def test_scan_iterate_negative_count_raises(self):
        self.assert_raises_kg("(-1){x+1}\\*0")

    def test_scan_iterate_fractional_count_raises(self):
        self.assert_raises_kg("0.5{x+1}\\*0")

    def test_scan_iterate_zero_count_shape(self):
        # 0 f\*b returns b itself (reference suite: 0,\*1 --> 1); decision recorded in the docstring
        self.assertEqual(self.run_kg("0{x+1}\\*5"), 5)
        self.assertEqual(list(self.run_kg("1{x+1}\\*5")), [5, 6])
        self.assertEqual(list(self.run_kg("2.0{x+1}\\*5")), [5, 6, 7])
        self.assertEqual(list(self.run_kg("(1+1){x+1}\\*5")), [5, 6, 7])

    # split
    def test_split_zero_sizes_raises(self):
        self.assert_raises_kg('[0 0]:#"abc"')
        self.assert_raises_kg('0:#"abc"')
        self.assert_raises_kg('[1 -1]:#"abc"')
        self.assert_raises_kg('(-1):#"abc"')
        self.assert_raises_kg('[0]:#"abc"')

    def test_split_valid_unchanged(self):
        self.assertEqual(list(self.run_kg('2:#"abcde"')), ["ab", "cd", "e"])
        self.assertEqual(list(self.run_kg('[1 2]:#[1 2 3 4 5 6]')[1]), [2, 3])

    # converge
    def test_converge_nan_terminates(self):
        r = self.run_kg("{(-1)^0.5}:~1")
        self.assertTrue(math.isnan(r))

    def test_scan_converge_nan_terminates(self):
        r = self.run_kg("{(-1)^0.5}\\~1")
        self.assertTrue(math.isnan(r[-1]))

    def test_converge_normal(self):
        r = self.run_kg("{(x+2%x)%2}:~2")
        self.assertAlmostEqual(float(r), 2 ** 0.5, places=4)


if __name__ == "__main__":
    unittest.main()
