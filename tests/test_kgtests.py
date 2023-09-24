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


    def test_kgtests(self):
        """
        Recursively run all tests under the kgtests folder that begin with "test" and end with ".kg".
        """
        ran_tests = False
        root_dir = os.path.join(os.getcwd(), "tests", "kgtests")

        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.startswith("test") and fname.endswith(".kg"):
                    ran_tests = True
                    print(f"Running {fname}...")
                    klong = KlongInterpreter()
                    klong['fullpath'] = os.path.join(dirpath, fname)
                    try:
                        klong('.l("tests/kgtests/runner.kg")')
                    except Exception as e:
                        print(e)
                        self.assertEqual(klong['err'], 1)
                    finally:
                        self.assertEqual(klong['err'], 0)

        self.assertTrue(ran_tests)
