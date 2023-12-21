import glob
import math
import os
import unittest

from utils import *

import timeit

def parse_suite_file(fname, number=10):
    klong = KlongInterpreter()
    with open(fname, "r") as f:
        d = f.read()
        b = len(d) * number
        r = timeit.timeit(lambda: klong(d), number=number)
        return b, r, int(b / r), r / number


def test_kgtests():
    """
    Recursively run all tests under the kgtests folder that begin with "test" and end with ".kg".
    """
    root_dir = os.path.join(os.getcwd(), "tests", "kgtests")

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if (fname.startswith("gen")) and fname.endswith(".kg"):
                ran_tests = True
                klong = KlongInterpreter()
                fullpath = os.path.join(dirpath, fname)
                klong['fullpath'] = fullpath
                klong('.l("tests/kgtests/runner.kg")')
                if fname.startswith("gen"):
                    r = parse_suite_file(fullpath)
                    print(f"total: {r[1]} iter: {r[-1]}")

if __name__ == "__main__":
    test_kgtests()