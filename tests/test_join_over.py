import unittest
from klongpy import KlongInterpreter
from utils import *


class TestJoinOver(unittest.TestCase):

    def assert_eval_cmp(self, a, b, klong=None):
        self.assertTrue(eval_cmp(a, b, klong=klong))

    def test_empty_arrays(self):
        klong = KlongInterpreter()
        r = klong(",/[]")
        self.assertTrue(array_equal(r, np.array([])))
        r = klong(",/[[]]")
        self.assertTrue(array_equal(r, np.array([])))
        r = klong(",/[[] []]")
        self.assertTrue(array_equal(r, np.array([])))
        # r = klong(",/[[] [] [[]]]")
        # self.assertTrue(array_equal(r, np.array([[]])))

    def test_single_element_array(self):
        klong = KlongInterpreter()
        r = klong(",/1")
        self.assertEqual(r, 1)
        r = klong(",/[1]")
        self.assertEqual(r, 1)
        r = klong(",/[[1]]")
        self.assertTrue(array_equal(r, np.array([1])))

    def test_no_join_array(self):
        klong = KlongInterpreter()
        r = klong(",/[1 2 3 4 5 6 7 8 9]")
        self.assertTrue(array_equal(r, np.array([1,2,3,4,5,6,7,8,9])))
    
    def test_1D_arrays(self):
        klong = KlongInterpreter()
        r = klong(",/[[1 2 3] [4 5 6] [7 8 9]]")
        self.assertTrue(array_equal(r, np.array([1,2,3,4,5,6,7,8,9])))

    def test_2D_arrays(self):
        klong = KlongInterpreter()
        r = klong(",/[[[1 2] [3 4]] [[5 6] [7 8]]]")
        self.assertTrue(array_equal(r, np.array([[1,2],[3,4],[5,6],[7,8]])))

    def test_3D_arrays(self):
        klong = KlongInterpreter()
        r = klong(",/[[[[1 2] [3 4]] [[5 6] [7 8]]] [[[9 10] [11 12]] [[13 14] [15 16]]]]")
        self.assertTrue(array_equal(r, np.array([[[1,2],[3,4]], [[5,6],[7,8]], [[9,10],[11,12]], [[13,14],[15,16]]])))

    def test_mixed_dim_arrays(self):
        klong = KlongInterpreter()
        r = klong(",/[[[1 2] [3 4]] [[5 6] [7]]]")
        self.assertTrue(array_equal(r, np.array([[1,2],[3,4],[5,6],[7]])))
        r = klong(",/[[[1] [2] [3 4]] [[5] [6] [7]]]")
        self.assertTrue(array_equal(r, np.array([[1],[2],[3,4],[5],[6],[7]])))
        r = klong(",/[[1] [[2 3]] 4]")
        self.assertTrue(array_equal(r, np.array([1, [2, 3], 4], dtype='object')))
        r = klong(",/[[] [] [[]]]")
        self.assertTrue(array_equal(r, np.array([[]])))

    def test_string_elements(self): 
        klong = KlongInterpreter()
        r = klong(',/[["a" "b"] ["c" "d"]]')
        self.assertTrue(array_equal(r, np.array(['a', 'b', 'c', 'd'])))

    def test_mixed_elements(self):
        klong = KlongInterpreter()
        r = klong(',/[[1 "a"] ["b" 2]]')
        self.assertTrue(array_equal(r, np.array([1, 'a', 'b', 2], dtype='object')))
        r = klong(',/[["a"] [1]]')
        self.assertTrue(array_equal(r, np.array(['a', 1], dtype='object')))
        r = klong(',/["a" [1]]')
        self.assertTrue(array_equal(r, np.array(['a', 1], dtype='object')))

    def test_file_by_lines(self):
        """
        Test the suite file line by line using our own t()
        """
        klong = create_test_klong()
        with open("tests/klong_join_over.kg", "r") as f:
            skip_header = True
            i = 0
            for r in f.readlines():
                if skip_header:
                    if r.startswith("t::"):
                        skip_header = False
                    continue
                r = r.strip()
                if len(r) == 0:
                    continue
                i += 1
                klong.exec(r)
            print(f"executed {i} lines")


    def test_gen_file(self):
        """
        Test the entire suite file.
        """
        self.assertEqual(run_suite_file('klong_join_over.kg'), 0)


if __name__ == "__main__":
    run_suite_file("klong_join_over.kg")


