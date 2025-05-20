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

    def test_reshape_wildcard_front(self):
        """Wildcard as the first dimension"""
        self.assert_eval_cmp('[[] 2]:^[1 2 3 4 5 6]', '[[1 2] [3 4] [5 6]]')

    def test_index_row_wildcard(self):
        """Select entire first row using wildcard"""
        self.assert_eval_cmp('[[1 2 3] [4 5 6]]:@[0 []]', '[1 2 3]')

    def test_index_column_wildcard(self):
        """Select entire third column using wildcard"""
        self.assert_eval_cmp('[[1 2 3] [4 5 6]]:@[[] 2]', '[3 6]')

    def test_index_negative_row(self):
        """Use negative index for last row"""
        self.assert_eval_cmp('[[1 2 3] [4 5 6]]:@[-1 []]', '[4 5 6]')

    def test_index_negative_column(self):
        """Use negative index for last column"""
        self.assert_eval_cmp('[[1 2 3] [4 5 6]]:@[[] -1]', '[3 6]')

    def test_index_entire_matrix(self):
        """Wildcard for all rows and columns"""
        self.assert_eval_cmp('[[1 2 3] [4 5 6]]:@[[] []]', '[[1 2 3] [4 5 6]]')

    def test_index_3d_wildcard(self):
        """Indexing into 3D array with wildcard"""
        expr = '[[[1 2] [3 4]] [[5 6] [7 8]]]:@[1 0 []]'
        self.assert_eval_cmp(expr, '[5 6]')
