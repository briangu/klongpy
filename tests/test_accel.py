import unittest
from klongpy import KlongInterpreter
from utils import *
from klongpy.backend import np
import numpy


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
class TestAccelerate(unittest.TestCase):
    """
    Verify that we are actually running the adverb_over accelerated paths for cases that we can.
    """

    # TODO: this is not parallel test safe
    #       add ability to intercept calls in interpeter
    def test_over_add_nested_array(self):
        klong = KlongInterpreter()
        e = ExecutedReduce(np.add)
        data = get_rnd_nested_array()
        try:
            np.add = e
            klong['data'] = data
            r = klong('+/data')
        finally:
            np.add = e.fn
        self.assertTrue(array_equal(r, numpy.add.reduce(get_reduce_data(data))))
        self.assertTrue(e.executed)

    def test_over_add_array(self):
        klong = KlongInterpreter()
        e = ExecutedReduce(np.add)
        data = get_rnd_array()
        try:
            np.add = e
            klong['data'] = data
            r = klong('+/data')
        finally:
            np.add = e.fn
        self.assertTrue(np.isclose(r, numpy.add.reduce(get_reduce_data(data))))
        self.assertTrue(e.executed)

    ####### Subtract

    def test_over_subtract_nested_array(self):
        klong = KlongInterpreter()
        e = ExecutedReduce(np.subtract)
        data = get_rnd_nested_array()
        try:
            np.subtract = e
            klong['data'] = data
            r = klong('-/data')
        finally:
            np.subtract = e.fn
        self.assertTrue(array_equal(r, numpy.subtract.reduce(get_reduce_data(data))))
        self.assertTrue(e.executed)

    def test_over_subtract(self):
        klong = KlongInterpreter()
        e = ExecutedReduce(np.subtract)
        data = get_rnd_array()
        try:
            np.subtract = e
            klong['data'] = data
            r = klong('-/data')
        finally:
            np.subtract = e.fn
        self.assertTrue(numpy.isclose(r, numpy.subtract.reduce(get_reduce_data(data))))
        self.assertTrue(e.executed)

    ####### Multiply

    def test_over_multiply_nested_array(self):
        klong = KlongInterpreter()
        e = ExecutedReduce(np.multiply)
        data = get_rnd_nested_array()
        try:
            np.multiply = e
            klong['data'] = data
            r = klong('*/data')
        finally:
            np.multiply = e.fn
        self.assertTrue(array_equal(r, numpy.multiply.reduce(get_reduce_data(data))))
        self.assertTrue(e.executed)

    def test_over_multipy(self):
        klong = KlongInterpreter()
        e = ExecutedReduce(np.multiply)
        data = get_rnd_array()
        try:
            np.multiply = e
            klong['data'] = data
            r = klong('*/data')
        finally:
            np.multiply = e.fn
        self.assertTrue(numpy.isclose(r, numpy.multiply.reduce(get_reduce_data(data))))
        self.assertTrue(e.executed)

    ####### Divide

    def test_over_divide_nested_array(self):
        klong = KlongInterpreter()
        e = ExecutedReduce(np.divide)
        data = get_rnd_nested_array()
        try:
            np.divide = e
            klong['data'] = data
            r = klong('%/data')
        finally:
            np.divide = e.fn
        self.assertTrue(array_equal(r, numpy.divide.reduce(get_reduce_data(data))))
        self.assertTrue(e.executed)

    def test_over_divide(self):
        if np != numpy:
            return
        if not hasattr(np.divide, "reduce"):
            return
        klong = KlongInterpreter()
        e = ExecutedReduce(np.divide)
        data = get_rnd_array()
        try:
            np.divide = e
            klong['data'] = data
            r = klong('%/data')
        finally:
            np.divide = e.fn
        self.assertTrue(numpy.isclose(r, numpy.divide.reduce(get_reduce_data(data))))
        self.assertTrue(e.executed)

    ####### Min

    def test_over_min_nested_arrays(self):
        klong = KlongInterpreter()
        e = Executed(np.min)
        try:
            np.min = e
            r = klong('&/[[1 2 3] [4 5 6]]')
        finally:
            np.min = e.fn
        self.assertTrue(array_equal(r, [1,2,3]))
        self.assertFalse(e.executed)

    def test_over_min(self):
        klong = KlongInterpreter()
        e = Executed(np.min)
        try:
            np.min = e
            r = klong('&/[1 2 3 4]')
        finally:
            np.min = e.fn
        self.assertEqual(r, 1)
        self.assertTrue(e.executed)

    ####### Max

    def test_over_max_nested_arrays(self):
        klong = KlongInterpreter()
        e = Executed(np.max)
        try:
            np.max = e
            r = klong('|/[[1 2 3] [4 5 6]]')
        finally:
            np.max = e.fn
        self.assertTrue(array_equal(r, [4,5,6]))
        self.assertFalse(e.executed)

    def test_over_max(self):
        klong = KlongInterpreter()
        e = Executed(np.max)
        try:
            np.max = e
            r = klong('|/[1 2 3 4]')
        finally:
            np.max = e.fn
        self.assertEqual(r, 4)
        self.assertTrue(e.executed)

    ####### Join

    @unittest.skip
    def test_over_join(self):
        if np != numpy:
            return
        klong = KlongInterpreter()
        e = Executed(np.concatenate)
        try:
            np.concatenate = e
            r = klong(',/:~[[1] [2] [3]]')
        finally:
            np.concatenate = e.fn
        self.assertTrue(array_equal(r, [1,2,3]))
        self.assertTrue(e.executed)


