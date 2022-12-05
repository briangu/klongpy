import os
import timeit

import numpy as np
from utils import load_lib_file

from klongpy import KlongInterpreter


def perf_load_lib(number=100):
    r = timeit.timeit(lambda: load_lib_file('nstat.kg'), number=number)
    return r/number


if __name__ == "__main__":
    number = 1000
    perf_load_lib(1000)
