import unittest
from klongpy.core import *
from klongpy import KlongInterpreter


class TestVectorization(unittest.TestCase):

    def test_iterative(self):
        klong = KlongInterpreter()
        r = klong("{2*x}'!1000")
        self.assertTrue(kg_equal(r, np.arange(1000)*2))

    def test_vectorized(self):
        klong = KlongInterpreter()
        r = klong("2*!1000")
        self.assertTrue(kg_equal(r, np.arange(1000)*2))
        r = klong("a::!1000;2*a")
        self.assertTrue(kg_equal(r, np.arange(1000)*2))
