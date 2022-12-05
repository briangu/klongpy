import glob
import math
import os
import unittest

from utils import *


class TestExamples(unittest.TestCase):

    def test_load_examples(self):
        """
        Verify we can at least load the example libs.
        """
        for x in glob.glob(os.path.join(os.getcwd(), "*.kg")):
            print(f"loading {x}")
            load_lib_file(x)

    def test_load_nstats(self):
        """
        Test that
        1) we can load the very complex nstat.kg
        2) we can fallback to math.kg for pi
        """
        klong, _ = load_lib_file('nstat.kg')
        r = klong("sqr2pi(2)")
        self.assertTrue(np.isclose(r, math.sqrt(math.pi*2)))
