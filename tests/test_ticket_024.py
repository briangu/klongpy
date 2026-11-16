"""Ticket 024: take on rank-2+ arrays cycles over rows; KGOp equality."""
import unittest

from klongpy import KlongInterpreter
from klongpy.types import KGOp


class TestTicket024Take(unittest.TestCase):

    def setUp(self):
        self.klong = KlongInterpreter()

    def eq(self, expr, expected):
        r = self.klong(expr)
        self.assertTrue(self.klong._backend.kg_equal(r, expected), f"{expr} gave {r!r}")

    def test_take_rows_zero_array(self):
        self.eq("(4)#[[0 0]]", [[0, 0]] * 4)

    def test_take_fewer_rows(self):
        self.eq("1#[[1 2] [3 4]]", [[1, 2]])

    def test_take_more_rows(self):
        self.eq("3#[[1 2] [3 4]]", [[1, 2], [3, 4], [1, 2]])

    def test_take_negative(self):
        self.eq("(-3)#[[1 2] [3 4]]", [[3, 4], [1, 2], [3, 4]])

    def test_take_negative_fewer(self):
        self.eq("(-1)#[[1 2] [3 4]]", [[3, 4]])

    def test_take_zero(self):
        self.eq("0#[[1 2]]", [])

    def test_take_rank3(self):
        self.eq("3#[[[1 2]] [[3 4]]]", [[[1, 2]], [[3, 4]], [[1, 2]]])

    def test_take_ragged(self):
        self.eq("3#[[1] [2 3]]", [[1], [2, 3], [1]])

    def test_take_ragged_negative(self):
        self.eq("(-3)#[[1] [2 3]]", [[2, 3], [1], [2, 3]])

    def test_take_rank1_unchanged(self):
        self.eq("5#[1 2]", [1, 2, 1, 2, 1])
        self.eq("(-3)#[1 2]", [2, 1, 2])


class TestTicket024OpEquality(unittest.TestCase):

    def test_match_operators(self):
        klong = KlongInterpreter()
        self.assertEqual(klong("[:+]~[:+]"), 1)

    def test_match_operators_differ(self):
        klong = KlongInterpreter()
        self.assertEqual(klong("[:+]~[:-]"), 0)

    def test_match_data_list_with_operator(self):
        klong = KlongInterpreter()
        self.assertEqual(klong("[1,2]~[1,2]"), 1)

    def test_kgop_eq_hash(self):
        self.assertEqual(KGOp("+", 0), KGOp("+", 0))
        self.assertEqual(hash(KGOp("+", 0)), hash(KGOp("+", 0)))
        self.assertNotEqual(KGOp("+", 0), KGOp("-", 0))
        self.assertNotEqual(KGOp("+", 1), KGOp("+", 2))
        self.assertNotEqual(KGOp("+", 0), "+")
        self.assertEqual(len({KGOp("+", 0), KGOp("+", 0)}), 1)


if __name__ == "__main__":
    unittest.main()


class TestIntegerLiteralOutsideInt64(unittest.TestCase):
    """Late finding of ticket 015: negating 2**63 raised a TypeError."""

    def test_min_int64_literal(self):
        from klongpy import KlongInterpreter
        k = KlongInterpreter()
        self.assertEqual(int(k('-9223372036854775808')), -9223372036854775808)
        self.assertEqual(int(k('-9223372036854775809')), -9223372036854775809)
