"""
Scalar take, integers beyond 64 bits, each on a symbol, and the
compiled path agreeing with the interpreter.

Decision on integers beyond 64 bits: exact. An atomic dyad (+ - * % < > ^ ...)
with an operand that is an integer outside int64 is computed again with Python
integers, so `1+9223372036854775808` is `9223372036854775809`. Operands inside
int64 keep numpy semantics (a result that does not fit wraps), and the
compiled path declines any expression whose variable holds an integer outside
int64, so both paths give the same answer.
"""
import random
import unittest
import warnings
from unittest import mock

import numpy as np

import klongpy.interpreter as interp_mod
from klongpy import KlongInterpreter
from klongpy.errors import KlongRangeError
from tests.backend_compat import TORCH_AVAILABLE, is_torch_backend, plain, requires_python_engine

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
        self.assertEqual(values(k("100000000000000000000%2")), HUGE / 2)

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


# --- compiled path versus interpreter ---------------------------------------

def _guard(ast, klong):
    if getattr(klong, "_nocompile", False):
        return None
    return _guard.original(ast, klong)


_guard.original = interp_mod.compile_guarded


SETUP = [
    "i1::3", "i2::-2", "i0::0", "f1::2.5", "f0::0.0", "fm::-1.5",
    "imax::9223372036854775807", "imin::-9223372036854775808",
    "big::9223372036854775808", "huge::100000000000000000000", "fh::1e300",
    "ia::[1 2 3]", "imaxa::[9223372036854775807 1]", "fa::[1.5 2.5 0.5]",
    "fz::[0.0 1.0 2.0]", "iz::[0 1 2]", "ba::[9223372036854775808 1]",
    "m2::[[1 2][3 4]]", "em::[]",
]
VARS = [s.split("::")[0] for s in SETUP]


def _norm(r):
    if TORCH_AVAILABLE and isinstance(r, torch.Tensor):
        # the type is part of the comparison: a tensor is not a numpy array or a Python number
        return ("tensor", tuple(r.shape), str(r.dtype), repr(r.detach().cpu().tolist()), r.requires_grad)
    if isinstance(r, np.ndarray):
        l = values(r.tolist())
        return ("array", r.shape, repr(l), repr(_types(l)))
    v = values(r)
    return ("scalar", repr(v), type(v).__name__)


def _types(x):
    return [_types(y) for y in x] if isinstance(x, list) else type(x).__name__


def _run(k, src):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _norm(k(src))
    except Exception as e:  # noqa: BLE001 - the class is part of the comparison
        return ("error", type(e).__name__)


class CompiledMatchesInterpreter(unittest.TestCase):
    """Run an expression with and without the compiler; the answers must be equal."""

    @classmethod
    def setUpClass(cls):
        cls._patch = mock.patch.object(interp_mod, "compile_guarded", _guard)
        cls._patch.start()

    @classmethod
    def tearDownClass(cls):
        cls._patch.stop()

    def pair(self):
        out = []
        for compiled in (True, False):
            k = KlongInterpreter()
            k._nocompile = not compiled
            for s in SETUP:
                k(s)
            out.append(k)
        return out

    def check(self, kc, ki, src):
        self.assertEqual(_run(kc, src), _run(ki, src), src)

    def assert_same(self, *sources):
        kc, ki = self.pair()
        for src in sources:
            self.check(kc, ki, src)

    @requires_python_engine("reads the private _compiled_cache of the Python interpreter")
    def test_compiler_is_in_use(self):
        kc, ki = self.pair()
        kc("ia+ia")
        self.assertTrue(kc._compiled_cache[("ia+ia", kc._module)])
        ki("ia+ia")
        self.assertFalse(ki._compiled_cache[("ia+ia", ki._module)])

    def test_integer_valued_power(self):
        # 017 notes: 0^x and x^2 with a 0.0 base gave floats compiled, integers interpreted
        self.assert_same("f0^i1", "0^f1", "f0^f0", "f0^ia", "fz^2", "fz^iz", "0.0^fa", "(-f0)^2",
                         "f1^2", "fa^2", "f0^(-f0)", "ia^2", "2^ia", "0^iz", "fz^fz", "1e300^0")

    def test_power_out_of_range(self):
        # 017 notes: 10^400 raised in the interpreter and gave a Python int compiled
        self.assert_same("imax^100", "10^huge", "i1^400", "fh^2", "fh^fh", "ia^1e300", "2^big",
                         "2^imax", "big^2", "ba^2")

    def test_scalar_scan(self):
        # 017 notes: +\a is [a] compiled and a interpreted
        self.assert_same("+\\i1", "+\\f1", "*\\i1", "+\\(i1+1)", "ia+(+\\i1)", "+\\(+/ia)",
                         "+\\ia", "*\\ia", "+\\fa", "*\\m2", "+\\m2", "+\\em", "+\\iz=iz", "+\\(i1>2)")

    def test_integers_beyond_int64(self):
        # 017 notes: Python ints instead of int64 wraparound or an error
        self.assert_same("imax+1", "imax*2", "imin-1", "-imin", "imax+imax", "imin*3", "imax-imin",
                         "big+0", "big+1", "1+big", "big-big", "huge*i1", "ba+1", "ba*2", "-big",
                         "big<i1", "i1<big", "big=big", "+/ba", "|/ba", "&/ba", "+\\ba", "-ba",
                         "imaxa+1", "+/imaxa", "*/imaxa", "imaxa*imaxa", "imaxa%2", "i0*big",
                         "big%i0", "imax+9223372036854775808", "ia+9223372036854775808",
                         "ia*9223372036854775807", "ia^100", "big%2")

    def test_float_overflow_and_zero(self):
        self.assert_same("fh*fh", "fh*fh*fh", "fh+fh", "-fh", "f0*(-1)", "-f0", "fa%f0", "ia%i0",
                         "i1%i0", "f1%f0", "iz%iz", "fz%fz", "em%1")

    def test_ordinary_arithmetic(self):
        self.assert_same("ia+ia", "ia*2+iz", "fa-fz", "(ia*2+iz)%(ia+1)", "+/ia*ia", "|/ia-iz", "&/fa",
                         "ia>iz", "ia<2", "fa=fa", "i1<i2", "-ia", "-fa", "(-ia)*(-ia)", "+\\ia*2",
                         "+/em", "*/em", "m2+1", "m2*m2", "+/m2", "|/m2", "&/m2", "ia+m2")

    def test_randomised(self):
        rnd = random.Random(2025)
        lits = ["0", "1", "2", "3", "0.0", "2.5", "9223372036854775808", "9223372036854775807", "1e300", "100"]

        def gen(depth):
            r = rnd.random()
            if depth == 0 or r < 0.3:
                return rnd.choice(VARS) if rnd.random() < 0.8 else rnd.choice(lits)
            if r < 0.4:
                return "(-" + gen(depth - 1) + ")"
            if r < 0.55:
                return rnd.choice("+*|&") + rnd.choice("/\\") + gen(depth - 1)
            return "(" + gen(depth - 1) + rnd.choice(["+", "-", "*", "%", "^", "<", ">", "="]) + gen(depth - 1) + ")"

        kc, ki = self.pair()
        for _ in range(500):
            self.check(kc, ki, gen(rnd.choice([1, 2, 2, 3])))


if __name__ == "__main__":
    unittest.main()
