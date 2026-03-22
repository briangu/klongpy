import importlib.util
import os
import unittest
from utils import *
from backend_compat import requires_object_dtype


_PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


class TestKgTests(unittest.TestCase):

    @requires_object_dtype
    def test_known_failure(self):
        klong = KlongInterpreter()
        klong['fullpath'] = "tests/kgtests/known_failure.kg"
        klong('.l("tests/kgtests/runner.kg")')
        self.assertEqual(klong['err'], 1)

    def eval_file_by_lines(self, fname):
        klong = create_test_klong()
        with open(fname, "r") as f:
            skip_header = True
            i = 0
            for r in f.readlines():
                if skip_header:
                    if r.startswith("t::"):
                        skip_header = False
                    else:
                        continue
                r = r.strip()
                if len(r) == 0:
                    continue
                i += 1
                klong.exec(r)
                self.assertEqual(klong['err'], 0)
            print(f"executed {i} lines")


def _make_kg_test(filepath, is_gen):
    """Create a test method for a single .kg file."""
    is_db = '/db/' in filepath or '\\db\\' in filepath

    @requires_object_dtype
    def test_method(self):
        if is_db and not _PANDAS_AVAILABLE:
            raise unittest.SkipTest("requires pandas")
        klong = KlongInterpreter()
        klong['fullpath'] = filepath
        klong('.l("tests/kgtests/runner.kg")')
        self.assertEqual(klong['err'], 0)
        if is_gen:
            self.eval_file_by_lines(filepath)

    return test_method


# Discover .kg test files and generate individual test methods
_root_dir = os.path.join(os.getcwd(), "tests", "kgtests")
for _dirpath, _, _filenames in os.walk(_root_dir):
    for _fname in sorted(_filenames):
        if (_fname.startswith("test") or _fname.startswith("gen")) and _fname.endswith(".kg"):
            _fullpath = os.path.join(_dirpath, _fname)
            _is_gen = _fname.startswith("gen")
            _test_name = "test_" + os.path.splitext(_fname)[0]
            # Disambiguate by subdirectory
            _subdir = os.path.basename(_dirpath)
            if _subdir != "kgtests":
                _test_name = f"test_{_subdir}_{os.path.splitext(_fname)[0]}"
            setattr(TestKgTests, _test_name, _make_kg_test(_fullpath, _is_gen))
