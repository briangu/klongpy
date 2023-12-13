import glob
import math
import os
import unittest

from utils import *


class TestKgTests(unittest.TestCase):

    def test_known_failure(self):
        klong = KlongInterpreter()
        klong['fullpath'] = "tests/kgtests/known_failure.kg"
        klong('.l("tests/kgtests/runner.kg")')
        self.assertEqual(klong['err'], 1)

    def eval_file_by_lines(self, fname):
        """
        Test the suite file line by line using our own t()
        """
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
                self.assertEqual(klong['err'],0)
            print(f"executed {i} lines")


    def test_kgtests(self):
        """
        Recursively run all tests under the kgtests folder that begin with "test" and end with ".kg".
        """
        ran_tests = False
        root_dir = os.path.join(os.getcwd(), "tests", "kgtests")

        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if (fname.startswith("test") or fname.startswith("gen")) and fname.endswith(".kg"):
                    ran_tests = True
                    klong = KlongInterpreter()
                    fullpath = os.path.join(dirpath, fname)
                    klong['fullpath'] = fullpath
                    try:
                        klong('.l("tests/kgtests/runner.kg")')
                        if fname.startswith("gen"):
                            print(f"testing (line by line) {fname}...")
                            self.eval_file_by_lines(fullpath)
                    except Exception as e:
                        print(e)
                        self.assertEqual(klong['err'], 1)
                    finally:
                        self.assertEqual(klong['err'], 0)

        self.assertTrue(ran_tests)
