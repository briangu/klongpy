import unittest
from klongpy import KlongInterpreter
from utils import *
from klongpy.backend import np
from klongpy.backends import get_backend
from backend_compat import requires_numpy_backend
import numpy

# Get backend for kg_equal (always numpy for these tests)
_backend = get_backend('numpy')


class Executed:
    def __init__(self, fn):
        self.fn = fn
        self.executed = False

    def __call__(self, *args, **kwargs):
        self.executed = True
        return self.fn(*args, **kwargs)


class ExecutedReduce:
    def __init__(self, fn):
        self.fn = fn
        self.executed = False

    def reduce(self, *args, **kwargs):
        self.executed = True
        return self.fn.reduce(*args, **kwargs)


def get_rnd_nested_array():
    return np.asarray([np.random.rand(100), np.random.rand(100)])


def get_rnd_array():
    return np.random.rand(100)


def get_reduce_data(data):
    return data if isinstance(data, numpy.ndarray) else data.get()


# This approach isn't great because any usage of the thunked method will pass
# We need a way to intercept the real call
# Note: These tests use numpy ufunc monkey-patching, so only work with numpy backend


