import os
import timeit

import numpy as np

from klongpy import KlongInterpreter


def numpy_vec(number=100):
    r = timeit.timeit(lambda: np.multiply(np.add(np.arange(10000000), 1), 2), number=number)
    return r/number


def klong_vec(number=100):
    klong = KlongInterpreter()
    r = timeit.timeit(lambda: klong.exec("2*1+!10000000"), number=number)
    return r/number


def python_vec(number=100):
    r = timeit.timeit(lambda: [2 * (1 + x) for x in range(10000000)], number=number)
    return r/number


if __name__ == "__main__":
    number = 1000
    print("Python: ", end='')
    pr = python_vec(number=number)
    print(f"{round(pr,6)}s")

    print(f"KlongPy USE_GPU={os.environ.get('USE_GPU')}: ", end='')
    kr = klong_vec(number=number)
    print(f"{round(kr,6)}s")

    print("Numpy: ", end='')
    nr = numpy_vec(number=number)
    print(f"{round(nr,6)}s")

    print(f"Python / KlongPy => {round(pr/kr,6)}")
    print(f"Numpy / KlongPy => {round(nr/kr,6)}")
