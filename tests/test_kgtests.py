import glob
import math
import os
import unittest

from utils import *


class TestKgTests(unittest.TestCase):

    def test_kgtests(self):
        """
        Recursively run all tests under the kgtests folder that begin with "test" and end with ".kg".
        """
        ran_tests = False
        root_dir = os.path.join(os.getcwd(), "tests", "kgtests")

        for dirpath, dirnames, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.startswith("test") and fname.endswith(".kg"):
                    ran_tests = True
                    full_path = os.path.join(dirpath, fname)
                    print(f"TESTING: {fname}")
                    klong, _ = run_file(full_path)
                    self.assertEqual(klong['err'], 0)

        self.assertTrue(ran_tests)
