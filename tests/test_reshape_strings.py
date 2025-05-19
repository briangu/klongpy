import unittest
import numpy as np

from klongpy import KlongInterpreter
from tests.utils import kg_equal

class TestReshapeStrings(unittest.TestCase):
    def setUp(self):
        self.klong = KlongInterpreter()

    def test_reshape_string_len1(self):
        r = self.klong('[2 2]:^"a"')
        self.assertTrue(kg_equal(r, np.asarray(["aa", "aa"], dtype=object)))

    def test_reshape_string_len2(self):
        r = self.klong('[2 2]:^"ab"')
        self.assertTrue(kg_equal(r, np.asarray(["ab", "ab"], dtype=object)))

    def test_reshape_string_len3(self):
        r = self.klong('[2 2]:^"abc"')
        self.assertTrue(kg_equal(r, np.asarray(["ab", "ca"], dtype=object)))

    def test_reshape_string_len4(self):
        r = self.klong('[2 2]:^"abcd"')
        self.assertTrue(kg_equal(r, np.asarray(["ab", "cd"], dtype=object)))

    def test_reshape_string_larger_shape(self):
        r = self.klong('[3 3]:^"abcd"')
        self.assertTrue(kg_equal(r, np.asarray(["abc", "dab", "cda"], dtype=object)))


if __name__ == '__main__':
    unittest.main()
