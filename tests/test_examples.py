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
        ran_nstat = False
        try:
            cwd = os.getcwd()
            os.chdir(os.path.join(cwd, 'examples'))
            for x in glob.glob("*.kg"):
                fname = os.path.basename(x)
                print(f"loading {fname}")
                klong = None
                if fname != "help.kg":
                    klong, _ = run_file("help.kg")
                klong, _ = run_file(x, klong=klong)
                if fname == "nstat.kg":
                    ran_nstat = True
                    r = klong("sqr2pi(2)")
                    self.assertTrue(np.isclose(r, math.sqrt(math.pi*2)))
        finally:
            os.chdir(cwd)
        self.assertTrue(ran_nstat)

