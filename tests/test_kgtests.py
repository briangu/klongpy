import glob
import math
import os
import unittest

from utils import *


class TestKgTests(unittest.TestCase):

    def test_kgtests(self):
        """
        Run all tests under the kgtests folder.
        """
        ran_tests = False
        for x in glob.glob(os.path.join(os.getcwd(), os.path.join("tests", os.path.join("kgtests", "*.kg")))):
            ran_tests = True
            fname = os.path.basename(x)
            print(f"testing: {fname}")
            klong, _ = run_file(x)
            self.assertEqual(klong['err'], 0)
        self.assertTrue(ran_tests)
