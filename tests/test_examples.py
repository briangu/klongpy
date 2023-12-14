import glob
import math
import os
import unittest

from utils import *
from unittest.mock import MagicMock


class TestExamples(unittest.TestCase):

    def test_load_lib(self):
        """
        Verify we can at least load the library.
        """
        ran_nstat = False
        try:
            cwd = os.getcwd()
            os.chdir(os.path.join("klongpy","lib"))
            for x in glob.glob("./**/*.kg", recursive=True):
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

    @unittest.skip("TODO: need to mock out the web server")
    def test_load_examples(self):
        """
        Verify we can at least load the example examples.
        """
        ran_dfs = False
        try:
            cwd = os.getcwd()
            os.chdir("examples")
            for x in glob.glob("./**/*.kg", recursive=True):
                fname = os.path.basename(x)
                if fname == "update_data.kg":
                    # requires polygon module
                    continue
                print(f"loading example {fname}")
                klong = KlongInterpreter()
                klong['.system'] = {}
                klong['.system']['ioloop'] = MagicMock()
                klong['.system']['klongloop'] = MagicMock()
                klong['.system']['closeEvent'] = MagicMock()
                try:
                    run_file(x, klong=klong)
                except Exception as e:
                    print(f"failed to load example {fname} with error {e}")
                    raise e
                if fname == "dfs.kg":
                    ran_dfs = True
        finally:
            os.chdir(cwd)
        self.assertTrue(ran_dfs)
