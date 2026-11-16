"""Ticket 016: range and group keep first-appearance order."""
import unittest

from klongpy import KlongInterpreter


def py(x):
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [py(e) for e in x]
    return x


class Ticket016(unittest.TestCase):

    def run_kg(self, src):
        return py(KlongInterpreter()(src))

    def test_range_string_order(self):
        self.assertEqual(self.run_kg('?"banana"'), "ban")
        self.assertEqual(self.run_kg('?"hello foo"'), "helo f")
        self.assertEqual(self.run_kg('?"cab"'), "cab")
        self.assertEqual(self.run_kg('?"aaabbcccd"'), "abcd")

    def test_range_numeric_regression_guard(self):
        self.assertEqual(self.run_kg('?[3 1 3 2]'), [3, 1, 2])
        self.assertEqual(self.run_kg('?[5 3 5 1]'), [5, 3, 1])

    def test_range_numeric_keeps_dtype(self):
        r = KlongInterpreter()('?[3 1 3 2]')
        self.assertEqual(r.dtype.kind, "i")

    def test_range_object_cases(self):
        self.assertEqual(self.run_kg('?["b" "a" "b"]'), ["b", "a"])
        self.assertEqual(self.run_kg('?[[1 2] [3] [1 2]]'), [[1, 2], [3]])
        self.assertEqual(self.run_kg('?[1 "a" 1 "a"]'), [1, "a"])

    def test_range_does_not_merge_text_and_number(self):
        self.assertEqual(len(self.run_kg('?[10 "10"]')), 2)
        self.assertEqual(self.run_kg('?[10 "10" 10 "10"]'), [10, "10"])

    def test_range_empty(self):
        self.assertEqual(self.run_kg('?[]'), [])
        self.assertEqual(self.run_kg('?""'), "")

    def test_group_first_appearance(self):
        self.assertEqual(self.run_kg('=[3 1 3 2]'), [[0, 2], [1], [3]])
        self.assertEqual(self.run_kg('=[3 1 3 2 1]'), [[0, 2], [1, 4], [3]])
        self.assertEqual(self.run_kg('=["b" "a" "b"]'), [[0, 2], [1]])
        self.assertEqual(self.run_kg('="hello foo"'), [[0], [1], [2, 3], [4, 7, 8], [5], [6]])
        self.assertEqual(self.run_kg('=[1 2 3 4]'), [[0], [1], [2], [3]])

    def test_group_mixed_types(self):
        self.assertEqual(self.run_kg('=[1 "a" 1]'), [[0, 2], [1]])

    def test_group_nested(self):
        self.assertEqual(self.run_kg('=[[1 2] [3] [1 2]]'), [[0, 2], [1]])

    def test_group_empty(self):
        self.assertEqual(self.run_kg('=[]'), [])
        self.assertEqual(self.run_kg('=""'), [])


if __name__ == "__main__":
    unittest.main()
