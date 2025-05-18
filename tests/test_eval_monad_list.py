import unittest
import numpy as np

from klongpy import KlongInterpreter
from klongpy.core import KGSym
from tests.utils import kg_equal


class TestEvalMonadList(unittest.TestCase):

    def setUp(self):
        self.klong = KlongInterpreter()

    def test_int(self):
        r = self.klong(',1')
        self.assertTrue(kg_equal(r, np.asarray([1])))

    def test_symbol(self):
        r = self.klong(',:foo')
        self.assertTrue(r.dtype == object)
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0], KGSym('foo'))

    def test_string(self):
        r = self.klong(',"xyz"')
        self.assertTrue(kg_equal(r, np.asarray(['xyz'], dtype=object)))

    def test_list(self):
        r = self.klong(',[1]')
        self.assertTrue(kg_equal(r, np.asarray([[1]], dtype=object)))


if __name__ == '__main__':
    unittest.main()
