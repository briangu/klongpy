import unittest
import numpy as np
from utils import eval_cmp
from klongpy import KlongInterpreter
from klongpy.core import kg_write

class TestNumpySliceBehavior(unittest.TestCase):
    def assert_eval_cmp(self, a, b, klong=None):
        self.assertTrue(eval_cmp(a, b, klong=klong))

    def test_reshape_wildcard(self):
        """np.array([1,2,3,4]).reshape(2, -1)"""
        expected = np.array([1, 2, 3, 4]).reshape(2, -1)
        self.assert_eval_cmp('[2 []]:^[1 2 3 4]', kg_write(expected))

    def test_index_in_depth_wildcard(self):
        """np.array([[1,2,3],[4,5,6]])[:,1]"""
        expected = np.array([[1,2,3],[4,5,6]])[:,1]
        self.assert_eval_cmp('[[1 2 3] [4 5 6]]:@[[] 1]', kg_write(expected))

    def test_reshape_wildcard_front(self):
        """Wildcard as the first dimension. np.array([1,2,3,4,5,6]).reshape(-1, 2)"""
        expected = np.array([1,2,3,4,5,6]).reshape(-1,2)
        self.assert_eval_cmp('[[] 2]:^[1 2 3 4 5 6]', kg_write(expected))

    def test_index_row_wildcard(self):
        """Select entire first row using wildcard. np.array([[1,2,3],[4,5,6]])[0,:]"""
        expected = np.array([[1,2,3],[4,5,6]])[0,:]
        self.assert_eval_cmp('[[1 2 3] [4 5 6]]:@[0 []]', kg_write(expected))

    def test_index_column_wildcard(self):
        """Select entire third column using wildcard. np.array([[1,2,3],[4,5,6]])[:,2]"""
        expected = np.array([[1,2,3],[4,5,6]])[:,2]
        self.assert_eval_cmp('[[1 2 3] [4 5 6]]:@[[] 2]', kg_write(expected))

    def test_index_negative_row(self):
        """Use negative index for last row. np.array([[1,2,3],[4,5,6]])[-1,:]"""
        expected = np.array([[1,2,3],[4,5,6]])[-1,:]
        self.assert_eval_cmp('[[1 2 3] [4 5 6]]:@[-1 []]', kg_write(expected))

    def test_index_negative_column(self):
        """Use negative index for last column. np.array([[1,2,3],[4,5,6]])[:,-1]"""
        expected = np.array([[1,2,3],[4,5,6]])[:,-1]
        self.assert_eval_cmp('[[1 2 3] [4 5 6]]:@[[] -1]', kg_write(expected))

    def test_index_entire_matrix(self):
        """Wildcard for all rows and columns. np.array([[1,2,3],[4,5,6]])[:, :]"""
        expected = np.array([[1,2,3],[4,5,6]])[:, :]
        self.assert_eval_cmp('[[1 2 3] [4 5 6]]:@[[] []]', kg_write(expected))

    def test_index_3d_wildcard(self):
        """Indexing into 3D array with wildcard. np.array([[[1,2],[3,4]],[[5,6],[7,8]]])[1,0,:]"""
        expr = '[[[1 2] [3 4]] [[5 6] [7 8]]]:@[1 0 []]'
        expected = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])[1,0,:]
        self.assert_eval_cmp(expr, kg_write(expected))
