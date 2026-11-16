"""
Two operator bugs that broke eigenv (klongpy/lib/eigenv.kg; tickets 029, 033):

* a:-v,indices with a float v: joining v to the indices makes one float list,
  so the indices arrived as floats and the amend failed; an integer matrix
  must also become a float matrix to hold v.
* a,b for lists whose members are arrays of different shape (a vector and a
  matrix) failed while building the result.
"""
import unittest

import numpy as np

from klongpy import KlongInterpreter
from backend_compat import requires_object_dtype


def tolist(v):
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v).tolist()


class TestAmendInDepthFloatValue(unittest.TestCase):
    def setUp(self):
        self.k = KlongInterpreter()

    def test_float_value_with_integer_indices(self):
        self.assertEqual(tolist(self.k("[[1 2] [3 4]]:-5.5,[0 1]")), [[1.0, 5.5], [3.0, 4.0]])

    def test_integer_value_is_unchanged(self):
        r = self.k("[[1 2] [3 4]]:-42,[0 1]")
        self.assertEqual(tolist(r), [[1, 42], [3, 4]])
        self.assertEqual(self.k("([[1 2] [3 4]]:-42,[0 1])~[[1 42] [3 4]]"), 1)

    def test_three_levels(self):
        self.assertEqual(tolist(self.k("[[[0 0] [0 0]] [[0 0] [0 0]]]:-0.5,[1 0 1]")),
                         [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.5], [0.0, 0.0]]])

    def test_float_matrix(self):
        self.assertEqual(tolist(self.k("[[1.5 2.5] [3.5 4.5]]:-0.25,[1 0]")), [[1.5, 2.5], [0.25, 4.5]])

    def test_the_original_is_not_changed(self):
        self.k("a::[[1 2] [3 4]];b::a:-5.5,[0 1]")
        self.assertEqual(tolist(self.k("a")), [[1, 2], [3, 4]])

    def test_index_that_is_not_integral_is_an_error(self):
        from klongpy.errors import KlongTypeError
        with self.assertRaises(KlongTypeError):
            self.k("[[1 2] [3 4]]:-5.5,[0.5 1]")

    def test_at_each_strict_level(self):
        for level in (0, 1, 2):
            with self.subTest(level=level):
                k = KlongInterpreter(strict=level)
                self.assertEqual(tolist(k("[[1 2] [3 4]]:-5.5,[1 1]")), [[1.0, 2.0], [3.0, 5.5]])


class TestJoinDifferentShapes(unittest.TestCase):
    def setUp(self):
        self.k = KlongInterpreter()

    @requires_object_dtype
    def test_vector_and_matrix_members(self):
        r = self.k("(,[1 2]),(,[[1 2] [3 4]])")
        self.assertEqual(len(r), 2)
        self.assertEqual(tolist(r[0]), [1, 2])
        self.assertEqual(tolist(r[1]), [[1, 2], [3, 4]])

    @requires_object_dtype
    def test_count_and_members(self):
        self.assertEqual(self.k("#(,[1 2]),(,[[1 2] [3 4]])"), 2)
        self.assertEqual(self.k("((,[1 2]),(,[[1 2] [3 4]]))@1")[1][0], 3)

    @requires_object_dtype
    def test_matrix_then_vector(self):
        r = self.k("(,[[1 2] [3 4]]),(,[5 6])")
        self.assertEqual(tolist(r[0]), [[1, 2], [3, 4]])
        self.assertEqual(tolist(r[1]), [5, 6])

    def test_same_shapes_are_unchanged(self):
        self.assertEqual(tolist(self.k("(,[1 2]),(,[3 4])")), [[1, 2], [3, 4]])
        self.assertEqual(tolist(self.k("[1 2 3],[4 5 6]")), [1, 2, 3, 4, 5, 6])

    @requires_object_dtype
    def test_mixed_members_are_unchanged(self):
        r = self.k('[1 2],"ab"')
        self.assertEqual(len(r), 3)
        self.assertEqual(r[2], "ab")


if __name__ == "__main__":
    unittest.main()
