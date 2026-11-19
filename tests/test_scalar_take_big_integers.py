"""
Scalar take, integers beyond 64 bits, and each on a symbol.

Decision on integers beyond 64 bits: exact. An atomic dyad (+ - * % < > ^ ...)
with an operand that is an integer outside int64 is computed again with Python
integers, so `1+9223372036854775808` is `9223372036854775809`. Operands inside
int64 keep numpy semantics (a result that does not fit wraps).
"""
import random
import unittest
import warnings
from unittest import mock

import numpy as np

from klongpy import KlongInterpreter
from klongpy.errors import KlongRangeError
from tests.backend_compat import TORCH_AVAILABLE, is_torch_backend, plain

if TORCH_AVAILABLE:
    import torch


BIG = 9223372036854775808          # 2**63
HUGE = 100000000000000000000       # 10**20


def values(r):
    """A result as plain Python data, numpy scalars and arrays (and tensors) unwrapped."""
    if TORCH_AVAILABLE and isinstance(r, torch.Tensor):
        return r.detach().cpu().tolist()
    if isinstance(r, np.ndarray):
        return r.tolist()
    if isinstance(r, np.generic):
        return r.item()
    return r


class TestScalarTake(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        self.k = KlongInterpreter()

    def test_take_number(self):
        self.assertEqual(values(self.k("3#7")), [7, 7, 7])

    def test_take_negative_count(self):
        self.assertEqual(values(self.k("(-3)#7")), [7, 7, 7])

    def test_take_zero(self):
        self.assertEqual(values(self.k("0#7")), [])

    def test_take_one(self):
        self.assertEqual(values(self.k("1#7")), [7])

    def test_take_float(self):
        self.assertEqual(values(self.k("2#1.5")), [1.5, 1.5])

    def test_take_computed_scalar(self):
        self.assertEqual(values(self.k("3#(2+5)")), [7, 7, 7])
        self.k("x::7")
        self.assertEqual(values(self.k("3#x")), [7, 7, 7])

    def test_take_symbol(self):
        r = self.k("2#:foo")
        self.assertEqual(len(r), 2)
        self.assertEqual([str(s) for s in r], ["foo", "foo"])

    def test_take_big_integer(self):
        self.assertEqual(values(self.k("2#9223372036854775808")), [BIG, BIG])

    def test_take_list_and_string_unchanged(self):
        self.assertEqual(values(self.k("5#[1 2 3]")), [1, 2, 3, 1, 2])
        self.assertEqual(values(self.k("(-2)#[1 2 3]")), [2, 3])
        self.assertEqual(self.k('3#"abcdef"'), "abc")
        self.assertEqual(self.k('3#"a"'), "aaa")
        self.assertEqual(values(self.k("0#[]")), [])
        self.assertEqual(self.k('0#""'), "")


class TestBigIntegers(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        self.k = KlongInterpreter()

    def test_scalar_add(self):
        self.assertEqual(values(self.k("1+9223372036854775808")), BIG + 1)
        self.assertEqual(values(self.k("9223372036854775808+1")), BIG + 1)

    def test_list_add(self):
        self.assertEqual(values(self.k("[1 2]+9223372036854775808")), [BIG + 1, BIG + 2])
        self.assertEqual(values(self.k("9223372036854775808+[1 2]")), [BIG + 1, BIG + 2])

    def test_other_operators(self):
        k = self.k
        self.assertEqual(values(k("9223372036854775808-1")), BIG - 1)
        self.assertEqual(values(k("1-9223372036854775808")), 1 - BIG)
        self.assertEqual(values(k("2*9223372036854775808")), 2 * BIG)
        self.assertEqual(values(k("[1 2]*100000000000000000000")), [HUGE, 2 * HUGE])
        self.assertEqual(values(k("9223372036854775808+9223372036854775808")), 2 * BIG)
        self.assertAlmostEqual(values(k("100000000000000000000%2")) / (HUGE / 2), 1, places=6)     # float32 on the torch backend

    def test_result_back_in_range(self):
        self.assertEqual(values(self.k("9223372036854775808-9223372036854775807")), 1)
        r = self.k("[9223372036854775808 9223372036854775809]-9223372036854775808")
        self.assertEqual(values(r), [0, 1])
        self.assertNotEqual(r.dtype, object)

    def test_comparisons(self):
        k = self.k
        self.assertEqual(values(k("1<9223372036854775808")), 1)
        self.assertEqual(values(k("9223372036854775808>1")), 1)
        self.assertEqual(values(k("9223372036854775808<1")), 0)
        self.assertEqual(values(k("9223372036854775808=9223372036854775808")), 1)
        self.assertEqual(values(k("[1 2 3]<9223372036854775808")), [1, 1, 1])

    def test_negate_and_reduce(self):
        k = self.k
        self.assertEqual(values(k("x::9223372036854775808;-x")), -BIG)
        self.assertEqual(values(k("+/[9223372036854775808 1]")), BIG + 1)

    def test_list_literal_keeps_big_integers(self):
        # numpy turns [2**63 1] into floats; the list must stay exact
        self.assertEqual(values(self.k("[9223372036854775808 1]")), [BIG, 1])
        self.assertEqual(values(self.k("[9223372036854775808]")), [BIG])
        self.assertEqual(values(self.k("[18446744073709551616 1]")), [2 * BIG, 1])
        self.assertEqual(values(self.k("[1.5 2]")), [1.5, 2.0])

    def test_int64_operands_keep_numpy_semantics(self):
        self.assertEqual(values(self.k("9223372036854775807+1")), -BIG)
        self.assertEqual(values(self.k("[1 2 3]+1")), [2, 3, 4])

    def test_variables(self):
        k = self.k
        k("x::9223372036854775808")
        self.assertEqual(values(k("x+1")), BIG + 1)
        self.assertEqual(values(k("[1 2]+x")), [BIG + 1, BIG + 2])

    def test_power_beyond_float_is_exact(self):
        # an integer to a non-negative integer power is exact (a scalar)
        self.assertEqual(int(self.k("10^400")), 10 ** 400)

    def test_power_beyond_a_million_bits_is_a_range_error(self):
        with self.assertRaises(KlongRangeError):
            self.k("10^100000000")

    def test_unsupported_operator_is_a_range_error(self):
        # ! (remainder) has no object-dtype loop: still an error, in the library classes
        with self.assertRaises(KlongRangeError):
            self.k("9223372036854775808!7")


class TestEachOnSymbol(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        self.k = KlongInterpreter()

    def test_each_on_symbol_returns_the_symbol(self):
        r = self.k("s:::foo;{x}'s")
        self.assertIsInstance(r, type(self.k(":foo")))
        self.assertEqual(r, self.k(":foo"))
        self.assertNotIsInstance(r, np.ndarray)

    def test_each_on_symbol_applies_f_once(self):
        self.assertEqual(self.k(":foo"), self.k("{x}':foo"))
        self.assertEqual(values(self.k("{1}':foo")), 1)

    def test_each_on_string_unchanged(self):
        self.assertEqual(self.k("""{x}'"abc\""""), "abc")
