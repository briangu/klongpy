"""
Ticket 029: numeric checks of library functions against values computed
independently with numpy / plain Python.

* nstat.kg `freq`: the n-point frequency distribution of D on [x0 xn]
  (it declared two unused locals, `d` and `c`, and was cleaned up).
* math.kg `sin` of a negative argument raised an undefined-name error inside
  `sin.tlr`; that was also what made `fft` fail at every level.
* math.kg `fft` against `numpy.fft.fft` (the library's `exp`/`sin`/`cos` are
  series approximations, hence the tolerance).
* eigenv.kg `eigenv`: needs two operator fixes (see the probe in the test).
"""
import contextlib
import io
import math
import os
import unittest

import numpy as np

from klongpy import KlongInterpreter
from backend_compat import requires_object_dtype

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "klongpy", "lib")
LEVELS = (0, 1, 2)


def load(*files, level=0):
    k = KlongInterpreter(strict=level)
    k[".helpdb"] = []
    cwd = os.getcwd()
    os.chdir(LIB)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for f in files:
                k(f'.l("{f}")')
    finally:
        os.chdir(cwd)
    return k


def complex_of(pairs):
    a = np.asarray(pairs, dtype=float)
    return a[:, 0] + 1j * a[:, 1]


def normal_pdf(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


class TestNstatFreq(unittest.TestCase):

    @requires_object_dtype
    def test_freq_is_D_at_n_equally_spaced_points(self):
        for level in LEVELS:
            k = load("nstat.kg", level=level)
            with self.subTest(level=level):
                got = np.asarray(k("freq(ndf;5;[0 4])"), dtype=float)
                np.testing.assert_allclose(got, [normal_pdf(x) for x in (0, 1, 2, 3, 4)], rtol=1e-3)
                got = np.asarray(k("freq(ndf;3;[-1 1])"), dtype=float)
                np.testing.assert_allclose(got, [normal_pdf(x) for x in (-1, 0, 1)], rtol=1e-3)
                got = np.asarray(k("freq(ndf;4;[0 3])"), dtype=float)
                np.testing.assert_allclose(got, [normal_pdf(x) for x in np.linspace(0, 3, 4)], rtol=1e-3)

    @requires_object_dtype
    def test_nfreq_is_freq_relative_to_its_maximum(self):
        k = load("nstat.kg")
        f = np.asarray(k("freq(ndf;5;[0 4])"), dtype=float)
        np.testing.assert_allclose(np.asarray(k("nfreq(ndf;5;[0 4])"), dtype=float), f / f.max(), rtol=1e-3)


class TestSinNegative(unittest.TestCase):

    @requires_object_dtype
    def test_sin_is_odd_and_accurate(self):
        for level in LEVELS:
            k = load("math.kg", level=level)
            for x in (-6.0, -4.0, -3.0, -1.57, -1.0, -0.5, 0.5, 1.0, 3.0, 6.0):
                with self.subTest(level=level, x=x):
                    self.assertAlmostEqual(float(k(f"sin({x})")), math.sin(x), places=4)
                    self.assertAlmostEqual(float(k(f"cos({x})")), math.cos(x), places=4)


class TestFft(unittest.TestCase):

    @requires_object_dtype
    def test_fft_matches_numpy(self):
        cases = [[1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8], [3, -1, 4, 1, -5, 9, 2, 6], [1, 0, 0, 0], [5, 7]]
        for level in LEVELS:
            k = load("math.kg", level=level)
            for xs in cases:
                with self.subTest(level=level, x=xs):
                    got = complex_of(k("fft([" + " ".join(map(str, xs)) + "])"))
                    np.testing.assert_allclose(got, np.fft.fft(xs), atol=1e-3)

    @requires_object_dtype
    def test_fft_pads_to_a_power_of_two_and_keeps_n_values(self):
        k = load("math.kg")
        got = complex_of(k("fft([1 2 3])"))
        np.testing.assert_allclose(got, np.fft.fft([1, 2, 3], 4)[:3], atol=1e-3)


class TestEigenv(unittest.TestCase):
    """eigenv needs `:-` to accept a float value joined to integer indices, and
    `,` to join arrays of different rank (see ticket 029). Both operators are
    fixed (tests/test_amend_depth_join_ragged.py); the result must match numpy."""

    @requires_object_dtype
    def test_eigenv_matches_numpy(self):
        k = load("eigenv.kg")
        m = [[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]]
        r = k("eigenv([[2.0 1.0 0.0] [1.0 2.0 1.0] [0.0 1.0 2.0]];4)")
        vals = np.sort(np.asarray(r[0], dtype=float))
        np.testing.assert_allclose(vals, np.sort(np.linalg.eigvalsh(np.array(m))), atol=1e-3)


if __name__ == "__main__":
    unittest.main()
