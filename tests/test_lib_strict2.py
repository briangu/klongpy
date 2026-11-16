"""
Ticket 051: the standard library and the .kg test suites run at every strict
level. A file that relied on a self-defining name, a symbol lookup or a
dropped argument fails here at level 2 (and one that assigns an undeclared
name inside a function fails at level 1).
"""
import contextlib
import glob
import importlib.util
import io
import os
import unittest

import numpy as np

from klongpy import KlongInterpreter
from backend_compat import requires_object_dtype

LEVELS = (0, 1, 2)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "klongpy", "lib")
KGTESTS = os.path.join(ROOT, "tests", "kgtests")
_PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None

# Calls of library functions whose bodies the strict levels reach only when
# they run: (file, expression). The value must be the same at every level.
CALLS = [
    ("math.kg", "sqr(16)"), ("math.kg", "exp(1)"), ("math.kg", "ln(10)"), ("math.kg", "log(2;8)"),
    ("math.kg", "sin(1)"), ("math.kg", "cos(1)"), ("math.kg", "tan(1)"), ("math.kg", "atan2(1;1)"),
    ("math.kg", "gcd(12;18)"), ("math.kg", "lcm(4;6)"), ("math.kg", "rndn(3.14159;2)"),
    ("math.kg", "gamln(3.5)"), ("math.kg", "gam(5)"), ("math.kg", "gamp(2;1.5)"), ("math.kg", "lgam(2;1.5)"),
    ("math.kg", "bisect({(x*x)-2})"), ("math.kg", "regf({(x*x)-2};1;2)"), ("math.kg", "primes(30)"),
    ("math.kg", "nrt(27;3)"), ("math.kg", ".md"),
    ("nstat.kg", "mu([1 2 3 4])"), ("nstat.kg", "var([1 2 3 4])"), ("nstat.kg", "sd([1 2 3 4])"),
    ("nstat.kg", "svar([1 2 3 4])"), ("nstat.kg", "M([3 1 2])"), ("nstat.kg", "mo([1 2 2 3])"),
    ("nstat.kg", "rng([1 5 3])"), ("nstat.kg", "cov([1 2 3];[2 4 7])"), ("nstat.kg", "cor([1 2 3];[2 4 7])"),
    ("nstat.kg", "lr(4;lreg([[1 2] [2 4] [3 7]]))"), ("nstat.kg", "ndf(0)"), ("nstat.kg", "cdf(0)"),
    ("nstat.kg", "qf(0.975)"), ("nstat.kg", "sqr2pi(2)"), ("nstat.kg", "f.mu([[1 2] [3 1]])"),
    ("nstat.kg", "f.var([[1 2] [3 1]])"), ("nstat.kg", "t.cdf(5;1.2)"), ("nstat.kg", "t.pdf(5;1.2)"),
    ("nstat.kg", "x2.pdf(3;1.5)"), ("nstat.kg", "erf(0.5)"),
    ("set.kg", "union([1 2];[2 3])"),
    ("csv.kg", 'csv.split("a,b,""c,d""")'), ("csv.kg", 'csv.make([["a" "b"] ["c" "d"]])'),
    ("eigenv.kg", "eye(2)"), ("eigenv.kg", "diag([[1 2] [3 4]])"), ("eigenv.kg", "msum([[1 2] [3 4]])"),
]


@contextlib.contextmanager
def in_dir(path):
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


def load(level, fname):
    k = KlongInterpreter(strict=level)
    k[".helpdb"] = []
    with contextlib.redirect_stdout(io.StringIO()):
        k(f'.l("{fname}")')
    return k


def plain(v):
    return v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else v


class TestLibraryAtEveryLevel(unittest.TestCase):

    @requires_object_dtype
    def test_every_file_loads(self):
        files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(LIB, "*.kg")))
        self.assertIn("nstat.kg", files)
        with in_dir(LIB):
            for fname in files:
                for level in LEVELS:
                    with self.subTest(file=fname, level=level):
                        k = load(level, fname)
                        self.assertEqual(k.strict, level)

    @requires_object_dtype
    def test_calls_give_the_same_value_at_every_level(self):
        with in_dir(LIB):
            for fname, text in CALLS:
                with self.subTest(file=fname, text=text):
                    values = [plain(load(level, fname)(text)) for level in LEVELS]
                    self.assertEqual(values[1], values[0])
                    self.assertEqual(values[2], values[0])
                    self.assertFalse(isinstance(values[0], str) and values[0] == text)

    @requires_object_dtype
    def test_eigenv_loads_what_it_uses(self):
        # eigenv.kg calls atan2, sin and cos but did not load math.kg
        with in_dir(LIB):
            for level in LEVELS:
                k = load(level, "eigenv.kg")
                self.assertAlmostEqual(k("atan2(1;1)"), 0.7853981633974483)


def _kg_suite(path, level):
    is_db = os.sep + "db" + os.sep in path

    @requires_object_dtype
    def test(self):
        if is_db and not _PANDAS_AVAILABLE:
            raise unittest.SkipTest("requires pandas")
        k = KlongInterpreter(strict=level)
        k["fullpath"] = path
        with in_dir(ROOT), contextlib.redirect_stdout(io.StringIO()) as out:
            k('.l("tests/kgtests/runner.kg")')
        self.assertEqual(k["err"], 0, out.getvalue()[-2000:])
    return test


class TestKgSuitesAtStrictLevels(unittest.TestCase):
    """Every tests/kgtests suite at levels 1 and 2 (tests/test_kgtests.py runs them at level 0)."""


for _dp, _, _names in sorted(os.walk(KGTESTS)):
    for _name in sorted(_names):
        if _name.endswith(".kg") and _name.startswith(("test", "gen")):
            for _level in (1, 2):
                _sub = os.path.basename(_dp)
                setattr(TestKgSuitesAtStrictLevels, f"test_{_sub}_{_name[:-3]}_level_{_level}",
                        _kg_suite(os.path.join(_dp, _name), _level))


if __name__ == "__main__":
    unittest.main()
