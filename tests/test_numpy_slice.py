import unittest
from utils import eval_cmp
from klongpy import KlongInterpreter

class TestNumpySliceBehavior(unittest.TestCase):
    def assert_eval_cmp(self, a, b, klong=None):
        self.assertTrue(eval_cmp(a, b, klong=klong))

    def test_reshape_wildcard(self):
        self.assert_eval_cmp('[2 []]:^[1 2 3 4]', '[[1 2] [3 4]]')

    def test_index_in_depth_wildcard(self):
        self.assert_eval_cmp('[[1 2 3] [4 5 6]]:@[[] 1]', '[2 5]')
