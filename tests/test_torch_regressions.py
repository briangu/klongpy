"""
The torch backend: regressions found during the 2.0 work.

Everything here runs only with `--backend torch` (the numpy run skips it).

* TestGradientThroughOperators: a tensor that requires grad still requires
  grad after every differentiable operator, and the gradient that comes back
  is the analytic one: each arithmetic operator, reduce and scan, and the
  operators touched by the operator edge-case fixes.
* TestResultsStayTensors: operators that used to hand a tensor to numpy
  ("Remainders") return tensors.
* TestTorchValueModel: what is different from the numpy backend by design
  (IEEE values for a division by zero inside a tensor, float32 text) and what
  is the same (strings, symbols, integers beyond 64 bits, dictionary keys).
* TestCompiledPathOnTorch: the compiled path gives what the interpreter gives.
* TestIpcWireValues: tensors through the kdb+ IPC endpoint conversion.
"""
import importlib.util
import math
import os
import unittest
import warnings
from unittest import mock

import numpy as np

from klongpy import KlongInterpreter
from klongpy.core import KGChar, KGSym, KLONG_UNDEFINED, kg_write
from tests.backend_compat import plain

TORCH = importlib.util.find_spec("torch") is not None
IS_TORCH_RUN = os.environ.get("_KLONGPY_TEST_BACKEND", "numpy") == "torch"

if TORCH:
    import torch


def _device():
    return os.environ.get("_KLONGPY_TEST_DEVICE") or "cpu"


def _close(a, b, tol=1e-4):
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    return abs(a - b) <= tol * max(1.0, abs(b))


@unittest.skipUnless(TORCH and IS_TORCH_RUN, "requires --backend torch")
class TorchCase(unittest.TestCase):

    A = [1.5, 2.0, 3.0, 0.5]
    M = [[1.5, 2.0], [3.0, 0.5]]
    S = 1.5

    def setUp(self):
        warnings.simplefilter("ignore")
        self.device = _device()
        self.k = KlongInterpreter(backend="torch", device=self.device)

    def assert_on_selected_device(self, tensor):
        family, _, index = self.device.partition(":")
        self.assertEqual(tensor.device.type, family)
        if index:
            self.assertEqual(tensor.device.index, int(index))

    def leaves(self):
        """A fresh interpreter with a (vector), m (matrix) and s (scalar) that require grad."""
        k = KlongInterpreter(backend="torch", device=self.device)
        a = torch.tensor(self.A, dtype=torch.float32, device=self.device, requires_grad=True)
        m = torch.tensor(self.M, dtype=torch.float32, device=self.device, requires_grad=True)
        s = torch.tensor(self.S, dtype=torch.float32, device=self.device, requires_grad=True)
        k["a"], k["m"], k["s"] = a, m, s
        return k, {"a": a, "m": m, "s": s}

    def grad(self, expr, wrt="a"):
        """d(sum of expr)/d(wrt) as a flat list; fails if the result left the graph."""
        k, leaves = self.leaves()
        r = k(expr)
        self.assertIsInstance(r, torch.Tensor, f"{expr}: result is {type(r).__name__}, not a tensor")
        self.assertTrue(r.requires_grad, f"{expr}: the result does not require grad")
        r.sum().backward()
        g = leaves[wrt].grad
        self.assertIsNotNone(g, f"{expr}: no gradient reached {wrt}")
        return g.detach().cpu().flatten().tolist()

    def assert_grad(self, expr, expected, wrt="a"):
        g = self.grad(expr, wrt)
        self.assertTrue(_close(g, expected), f"{expr}: d/d{wrt} is {g}, expected {expected}")


class TestGradientThroughOperators(TorchCase):

    # --- arithmetic --------------------------------------------------------

    def test_plus_minus_times(self):
        a = self.A
        for expr, want in [
            ("a+1", [1] * 4), ("1+a", [1] * 4), ("a+a", [2] * 4),
            ("a-1", [1] * 4), ("1-a", [-1] * 4), ("a-a*2", [-1] * 4),
            ("a*2", [2] * 4), ("2*a", [2] * 4), ("a*a", [2 * x for x in a]),
        ]:
            with self.subTest(expr=expr):
                self.assert_grad(expr, want)

    def test_divide(self):
        a = self.A
        for expr, want in [
            ("a%2", [0.5] * 4), ("2%a", [-2 / (x * x) for x in a]), ("a%a", [0] * 4),
            ("%a", [-1 / (x * x) for x in a]),
        ]:
            with self.subTest(expr=expr):
                self.assert_grad(expr, want)

    def test_power(self):
        a = self.A
        for expr, want in [
            ("a^2", [2 * x for x in a]),
            ("a^0.5", [0.5 * x ** -0.5 for x in a]),
            ("2^a", [math.log(2) * 2 ** x for x in a]),           # a number to a tensor power
            ("a^a", [x ** x * (math.log(x) + 1) for x in a]),
            ("a^-1", [-1 / (x * x) for x in a]),
        ]:
            with self.subTest(expr=expr):
                self.assert_grad(expr, want)

    def test_negate_max_min_magnitude(self):
        for expr, want in [
            ("-a", [-1] * 4),
            ("a|1.75", [0, 1, 1, 0]), ("a&1.75", [1, 0, 0, 1]),
        ]:
            with self.subTest(expr=expr):
                self.assert_grad(expr, want)

    def test_scalar_tensor(self):
        s = self.S
        for expr, want in [
            ("s+1", [1]), ("s*s", [2 * s]), ("s^2", [2 * s]), ("2^s", [math.log(2) * 2 ** s]),
            ("%s", [-1 / (s * s)]), ("-s", [-1]), ("s%2", [0.5]), ("2%s", [-2 / (s * s)]),
            ("#s-2", [-1]), ("#s", [1]),                       # magnitude of an atom
        ]:
            with self.subTest(expr=expr):
                self.assert_grad(expr, want, wrt="s")

    def test_comparison_masks_do_not_break_the_graph(self):
        # a comparison is not differentiable; multiplying by its 0/1 result is
        self.assert_grad("a*a>1", [1, 1, 1, 0])
        self.assert_grad("a*~a>1", [0, 0, 0, 1])

    # --- reduce and scan ---------------------------------------------------

    def test_reduce(self):
        a = self.A
        p = math.prod(a)
        for expr, want in [
            ("+/a", [1] * 4),
            ("-/a", [1, -1, -1, -1]),
            ("*/a", [p / x for x in a]),
            ("%/a", [1 / (a[1] * a[2] * a[3])] + [-(a[0] / (a[1] * a[2] * a[3])) / x for x in a[1:]]),
            ("|/a", [0, 0, 1, 0]),
            ("&/a", [0, 0, 0, 1]),
            ("{x+y}/a", [1] * 4),
            ("1{x+y}/a", [1] * 4),
        ]:
            with self.subTest(expr=expr):
                self.assert_grad(expr, want)

    def test_scan(self):
        a = self.A
        for expr, want in [
            ("+\\a", [4, 3, 2, 1]),
            ("-\\a", [4, -3, -2, -1]),
            ("*\\a", [1 + a[1] + a[1] * a[2] + a[1] * a[2] * a[3],
                      a[0] * (1 + a[2] + a[2] * a[3]), a[0] * a[1] * (1 + a[3]), a[0] * a[1] * a[2]]),
            ("|\\a", [1, 1, 2, 0]),
            ("&\\a", [3, 0, 0, 1]),
            ("{x*y}\\a", [1 + a[1] + a[1] * a[2] + a[1] * a[2] * a[3],
                          a[0] * (1 + a[2] + a[2] * a[3]), a[0] * a[1] * (1 + a[3]), a[0] * a[1] * a[2]]),
            ("1{x+y}\\a", [4, 3, 2, 1]),
        ]:
            with self.subTest(expr=expr):
                self.assert_grad(expr, want)

    def test_divide_scan(self):
        g = self.grad("%\\a")
        k, _ = self.leaves()
        x = torch.tensor(self.A, dtype=torch.float64, requires_grad=True)
        (x[0] + x[0] / x[1] + x[0] / x[1] / x[2] + x[0] / x[1] / x[2] / x[3]).backward()
        self.assertTrue(_close(g, x.grad.tolist()), g)

    def test_reduce_and_scan_of_a_matrix(self):
        m = self.M
        self.assert_grad("+/m", [1, 1, 1, 1], wrt="m")
        self.assert_grad("*/m", [m[1][0], m[1][1], m[0][0], m[0][1]], wrt="m")
        self.assert_grad("+\\m", [2, 2, 1, 1], wrt="m")
        self.assert_grad(",/m", [1, 1, 1, 1], wrt="m")

    # --- adverbs -----------------------------------------------------------

    def test_each_family(self):
        a = self.A
        for expr, want in [
            ("{x*x}'a", [2 * x for x in a]),
            ("a{x*y}'a", [2 * x for x in a]),
            ("2*:\\a", [2] * 4),
            ("a*:/2", [2] * 4),
            ("{x-y}:'a", [1, 0, 0, -1]),
            ("3{x*2}:*a", [8] * 4),            # iterate
            ("3{x*2}\\*a", [15] * 4),
            ("{x*y}@'a", None),
        ]:
            with self.subTest(expr=expr):
                if want is None:
                    continue
                self.assert_grad(expr, want)

    def test_converge_keeps_the_graph(self):
        # converge compares successive values; the value it returns is still in the graph
        k, leaves = self.leaves()
        r = k("{(x+s)%2}:~8.0")     # converges to s
        self.assertIsInstance(r, torch.Tensor)
        self.assertTrue(r.requires_grad)
        r.backward()
        self.assertTrue(_close(leaves["s"].grad.item(), 1.0, 1e-2))

    # --- operator edge-case fixes -------------------------------------

    def test_amend(self):
        self.assert_grad("a:=7.0,1", [1, 0, 1, 1])
        self.assert_grad("a:=7,1", [1, 0, 1, 1])                       # an integer into a float tensor
        self.assert_grad("a:=s,1", [1, 0, 1, 1])
        self.assert_grad("a:=s,1", [1], wrt="s")                       # through the value that is written
        self.assert_grad("a:=(s*2),[0 2]", [0, 1, 0, 1])
        self.assert_grad("a:=(s*2),[0 2]", [4], wrt="s")
        self.assert_grad("a:=7.0,-1", [1, 1, 1, 0])                    # negative index
        self.assert_grad("m:-s,[0 1]", [1, 0, 1, 1], wrt="m")          # amend in depth
        self.assert_grad("m:-s,[0 1]", [1], wrt="s")
        self.assert_grad("m:-9.0,[1 0]", [1, 1, 0, 1], wrt="m")

    def test_amend_does_not_write_to_the_operand(self):
        k, leaves = self.leaves()
        k("b::a:=7.0,1")
        self.assertEqual(leaves["a"].detach().cpu().tolist(), self.A)
        k2 = KlongInterpreter(backend="torch", device=self.device)
        self.assertEqual(plain(k2("x::[1 2 3];y::x:=9,0;x")), [1, 2, 3])
        self.assertEqual(plain(k2("y")), [9, 2, 3])

    def test_split_and_cut(self):
        self.assert_grad("2:#a", [1] * 4)
        self.assert_grad("2:_a", [1] * 4)
        # ragged pieces cannot be one tensor: a list of tensors, each still in the graph
        for expr in ("[1 2]:#a", "[1 3]:_a"):
            with self.subTest(expr=expr):
                k, leaves = self.leaves()
                r = k(expr)
                self.assertIsInstance(r, np.ndarray)
                self.assertTrue(all(isinstance(p, torch.Tensor) and p.requires_grad for p in r), expr)
                sum(p.sum() for p in r).backward()
                self.assertEqual(leaves["a"].grad.tolist(), [1.0] * 4)

    def test_index_divide_join(self):
        for expr, want in [
            ("a@1", [0, 1, 0, 0]), ("a@-1", [0, 0, 0, 1]),                 # negative index
            ("a@[0 2]", [1, 0, 1, 0]), ("a@[-1 0]", [1, 0, 0, 1]), ("a@[1 1]", [0, 2, 0, 0]),
            ("a%[1 2 4 8]", [1, 0.5, 0.25, 0.125]),
            ("a,1.0", [1] * 4), ("1.0,a", [1] * 4), ("a,a", [2] * 4), ("a,1", [1] * 4),
            ("*a", [1, 0, 0, 0]), (",a", [1] * 4),
        ]:
            with self.subTest(expr=expr):
                self.assert_grad(expr, want)
        self.assert_grad("m:@[0 1]", [0, 1, 0, 0], wrt="m")
        self.assert_grad("a,s", [1], wrt="s")
        self.assert_grad("s,s", [2], wrt="s")

    def test_division_by_zero_keeps_the_tensor(self):
        # the decision: IEEE values, so the tensor stays a tensor and gradients flow
        k, leaves = self.leaves()
        r = k("a%[1 0 2 0]")
        self.assertIsInstance(r, torch.Tensor)
        self.assertTrue(r.requires_grad)
        self.assertEqual(r.detach().cpu().tolist(), [1.5, float("inf"), 1.5, float("inf")])
        r[[0, 2]].sum().backward()
        self.assertEqual(leaves["a"].grad.tolist()[0], 1.0)
        self.assertEqual(leaves["a"].grad.tolist()[2], 0.5)

    def test_range_and_grade(self):
        self.assert_grad("?a", [1] * 4)                    # unique, order of appearance
        self.assert_grad("?a,a", [1] * 4)
        self.assert_grad("a@<a", [1] * 4)
        self.assert_grad("a@>a", [1] * 4)
        self.assert_grad("a@*=a", [1, 0, 0, 0])            # group gives indices; indexing by them is differentiable

    def test_take_and_reshape(self):
        for expr, want in [
            ("3#a", [1, 1, 1, 0]), ("6#a", [2, 2, 1, 1]), ("(-2)#a", [0, 0, 1, 1]), ("(-6)#a", [1, 1, 2, 2]),
            ("2_a", [0, 0, 1, 1]), ("(-1)_a", [1, 1, 1, 0]),
            ("[2 2]:^a", [1] * 4), ("[3 2]:^a", [2, 2, 1, 1]), ("6:^a", [2, 2, 1, 1]), ("[1 2]:^a", [1, 1, 0, 0]),
            ("|a", [1] * 4), ("1:+a", [1] * 4), ("(-1):+a", [1] * 4),
        ]:
            with self.subTest(expr=expr):
                self.assert_grad(expr, want)
        self.assert_grad("3#m", [2, 2, 1, 1], wrt="m")       # rank 2: cycles over the rows
        self.assert_grad("(-3)#m", [1, 1, 2, 2], wrt="m")
        self.assert_grad("+m", [1] * 4, wrt="m")

    def test_scalar_take(self):
        self.assert_grad("3#s", [3], wrt="s")
        self.assert_grad("2:^s", [2], wrt="s")
        self.assert_grad("[2 2]:^s", [4], wrt="s")

    # --- the gradient operators themselves -----------------------------------

    def test_autograd_operator_through_the_fixed_operators(self):
        k = self.k
        for src, want in [
            ("{+/x:=0.0,1}:>[1.0 2.0 3.0]", [1, 0, 1]),
            ("{+/%x}:>[1.0 2.0 4.0]", [-1, -0.25, -0.0625]),
            ("{+/6#x}:>[1.0 2.0 3.0 4.0]", [2, 2, 1, 1]),
            ("{+/1:+x*x}:>[1.0 2.0 3.0]", [2, 4, 6]),
            ("{+/2^x}:>[1.0 2.0]", [math.log(2) * 2, math.log(2) * 4]),
            ("{+/,/+[2 2]:^x*x}:>[1.0 2.0 3.0 4.0]", [2, 4, 6, 8]),
            ("{+/x@[0 0 2]}:>[1.0 2.0 3.0]", [2, 0, 1]),
            ("{+/x,x*x}:>[1.0 2.0]", [3, 5]),
        ]:
            with self.subTest(src=src):
                g = plain(k(src))
                self.assertTrue(_close(g, want), f"{src}: {g}")
        self.assertTrue(_close(plain(k("{x^2}:>3.0")), 6.0))
        self.assertTrue(_close(plain(k(".grad({+/x*x};[1.0 2.0 3.0])")), [2, 4, 6], 1e-2))


class TestResultsStayTensors(TorchCase):
    """Tensor in, tensor out (these went through numpy before)."""

    def test_tensor_results(self):
        for src, want in [
            ("!5", [0, 1, 2, 3, 4]),
            ("[1 2 3]:=0,1", [1, 0, 3]),
            ("[1 2 3]:=0.5,1", [1.0, 0.5, 3.0]),
            ("[[1 2] [3 4]]:-9,[1 0]", [[1, 2], [9, 4]]),
            ("[[1 2] [3 4]]:@[0 1]", 2),
            ("[[1 2] [3 4]]:@[[0 1] [1 0]]", [2, 3]),
            ("?[3 1 3 2]", [3, 1, 2]),
            ("7#[1 2 3]", [1, 2, 3, 1, 2, 3, 1]),
            ("(-7)#[1 2 3]", [3, 1, 2, 3, 1, 2, 3]),
            ("3#7", [7, 7, 7]),
            ("2:_[1 2 3 4]", [[1, 2], [3, 4]]),
            ("2:#[1 2 3 4]", [[1, 2], [3, 4]]),
            ("[2 2]:^!3", [[0, 1], [2, 0]]),
            ("[2 2]:^1", [[1, 1], [1, 1]]),
            ("1:+[1 2 3]", [3, 1, 2]),
            ("(-1):+[1 2 3]", [2, 3, 1]),
            ("+[[1 2] [3 4]]", [[1, 3], [2, 4]]),
            ("&[1 0 2]", [0, 2, 2]),
            ("~[1 0 2]", [0, 1, 0]),
            ("%[1 2 4]", [1.0, 0.5, 0.25]),
            ("[1 2 3 2]?2", [1, 3]),
            ("2^[1 2 3]", [2, 4, 8]),
            (">[2 1 2 1]", [2, 0, 3, 1]),          # grade down is the reverse of the stable grade up
            ("<[2 1 2 1]", [1, 3, 0, 2]),
        ]:
            with self.subTest(src=src):
                r = self.k(src)
                self.assertIsInstance(r, torch.Tensor, src)
                self.assertEqual(plain(r), want, src)

    def test_ragged_results_are_lists_of_tensors(self):
        r = self.k("[1 4 7]:_!10")
        self.assertEqual(plain(r), [[0], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertTrue(all(isinstance(p, torch.Tensor) for p in r))

    def test_converge_and_scan_converge(self):
        self.assertEqual(plain(self.k("{_x%2}:~100")), 0)
        self.assertEqual(plain(self.k("{_x%2}\\~100")), [100, 50, 25, 12, 6, 3, 1, 0])
        self.assertEqual(plain(self.k(",/:~[[[[[0]]]]]")), 0)

    def test_amend_range_and_type_errors_are_klong_errors(self):
        from klongpy import KlongRangeError
        from klongpy.errors import KlongTypeError
        with self.assertRaises(KlongRangeError):
            self.k("[1 2 3]:=0,3")
        with self.assertRaises(KlongTypeError):
            self.k("[1 2 3]:=0,1.5")
        with self.assertRaises(KlongRangeError):
            self.k("(-2):#[1 2 3]")
        with self.assertRaises(KlongRangeError):
            self.k("[1 2 3]@3")


class TestTorchValueModel(TorchCase):

    def test_division_by_zero(self):
        inf = float("inf")
        # atoms: :undefined, as on the numpy backend (the reference suite relies on 1%0)
        for src in ("1%0", "0%0", "7!0", "7:%0", "%0", "(1+1)%0", "(1+1)!(1-1)"):
            with self.subTest(src=src):
                self.assertIs(self.k(src), KLONG_UNDEFINED)
        # lists: the tensor keeps the IEEE value
        self.assertEqual(plain(self.k("[1 2]%0")), [inf, inf])
        self.assertEqual(plain(self.k("[1.0 2.0]%[2 0]")), [0.5, inf])
        r = plain(self.k("[5 6]![0 2]"))
        self.assertTrue(math.isnan(r[0]) and r[1] == 0.0)
        self.assertEqual(plain(self.k("[5 6]:%[0 2]")), [inf, 3.0])
        self.assertEqual(plain(self.k("[5 6]:%[2 4]")), [2, 1])
        self.assertEqual(plain(self.k("[7 8]![2 3]")), [1, 2])
        self.assertFalse(self.k("[7 8]![2 3]").is_floating_point())

    def test_strings_and_characters(self):
        k = self.k
        self.assertEqual(k('"hello"@1'), KGChar("e"))
        self.assertEqual(k('"hello"@[1 3]'), "el")
        self.assertEqual(k('2#"hello"'), "he")
        self.assertEqual(k('10#"abc"'), "abcabcabca")
        self.assertEqual(k('1:+"abc"'), "cab")
        self.assertEqual(plain(k('2:#"hello"')), ["he", "ll", "o"])
        self.assertEqual(plain(k('[2 4]:_"abcdef"')), ["ab", "cd", "ef"])
        self.assertEqual(plain(k('[2 2]:^"abcd"')), ["ab", "cd"])
        self.assertEqual(k('"abc":="x",1'), "axc")
        self.assertEqual(k("{x}'\"abc\""), "abc")
        self.assertEqual(plain(k(":#[72 105]")), ["H", "i"])
        self.assertEqual(k('s::"banana";s@<s'), "aaabnn")
        self.assertEqual(plain(k('[1 2 3]:="a",1')), [1, "a", 3])

    def test_format_is_the_number_not_the_tensor(self):
        k = self.k
        self.assertEqual(k("$-7"), "-7")
        self.assertEqual(plain(k("$[1 2 3]")), ["1", "2", "3"])
        self.assertEqual(k("0:$$-45"), -45)
        self.assertEqual(k("(-10)$-123"), "      -123")
        self.assertEqual(k("$-1.23"), "-1.23")
        self.assertEqual(k("6$-1.23"), "-1.23 ")

    def test_float32_numbers_print_with_their_shortest_text(self):
        k = self.k
        w = lambda src: kg_write(k(src), k._backend)
        self.assertEqual(w("-1.23"), "-1.23")
        self.assertEqual(w("[1.23 2.5]"), "[1.23 2.5]")
        # float32 arithmetic by design: not the float64 text of the numpy backend
        self.assertEqual(w("0.1+0.2"), "0.3")
        self.assertEqual(w("1%3"), "0.33333334")
        self.assertEqual(w("100000000.0*3"), "300000000.0")

    def test_symbols_in_nested_lists_stay_symbols(self):
        r = self.k("[[:a] [:b :c]]")
        self.assertIsInstance(r[0][0], KGSym)
        self.assertEqual(kg_write(r, self.k._backend), "[[:a] [:b :c]]")
        self.assertEqual(kg_write(self.k("5#[:a :b]"), self.k._backend), "[:a :b :a :b :a]")

    def test_dictionary_keys_are_numbers_not_tensors(self):
        k = self.k
        self.assertEqual(kg_write(k(":{[1 2] [3 4]},[1 9]"), k._backend), ":{[1 9] [3 4]}")
        self.assertEqual(plain(k("d:::{[1 2]};d,[1 5];d?1")), 5)
        self.assertEqual(plain(k("c:::{};{c,x,1+:[:_(c?x);0;c?x]}'[1 2 1 1];c?1")), 3)
        self.assertEqual(plain(k("d:::{[1 2] [3 4]};(1+0)_d;#d")), 1)
        self.assertEqual(plain(k("d:::{[2 7]};d?1+1")), 7)

    def test_integers_beyond_64_bits_are_exact(self):
        k = self.k
        big = 2 ** 63
        self.assertEqual(k("1+9223372036854775808"), big + 1)
        self.assertEqual(plain(k("[1 2]+9223372036854775808")), [big + 1, big + 2])
        self.assertEqual(plain(k("[9223372036854775808 1]")), [big, 1])
        self.assertEqual(k("9223372036854775808|1"), big)
        self.assertEqual(int(k("-9223372036854775808")), -big)
        self.assertEqual(plain(k("3#9223372036854775808")), [big] * 3)
        self.assertEqual(plain(k("9223372036854775808<3")), 0)
        # inside int64 the tensor arithmetic is unchanged
        self.assertIsInstance(k("[1 2]+1"), torch.Tensor)

    def test_scalar_power_is_computed_in_float64(self):
        # a number to the power of a plain number is computed in float64; it leaves the interpreter as the backend's float (float32)
        self.assertEqual(float(self.k("2^0.5")), float(np.float32(2 ** 0.5)))
        self.assertEqual(float(self.k("2^-1")), 0.5)


class TestIpcWireValues(TorchCase):
    """The endpoint converts tensors before `encode` and arrays after `decode`."""

    def roundtrip(self, value, widths=64):
        from klongpy.ipc.codec import decode, encode
        from klongpy.ipc.endpoint import _from_wire_array, to_wire_value
        wire = to_wire_value(self.k, value)
        back = decode(encode(wire, widths), True, widths)
        return wire, _from_wire_array(self.k, back)

    def test_integer_tensor(self):
        wire, back = self.roundtrip(self.k("[1 2 3]"))
        self.assertIsInstance(wire, np.ndarray)
        self.assertEqual(wire.dtype, np.int64)
        self.assertIsInstance(back, torch.Tensor)
        self.assertEqual(back.tolist(), [1, 2, 3])
        self.assert_on_selected_device(back)

    def test_float_tensor_goes_as_its_own_width(self):
        # a Klong float list on torch is float32: it is sent as a kdb real vector, exactly
        wire, back = self.roundtrip(self.k("[1.5 2.5]"))
        self.assertEqual(wire.dtype, np.float32)
        self.assertIsInstance(back, torch.Tensor)
        self.assertTrue(back.is_floating_point())
        self.assertEqual(back.tolist(), [1.5, 2.5])

    def test_only_real_vectors_can_narrow_on_a_device_without_doubles(self):
        narrow = np.array([1.5, -0.0], dtype=np.float32)
        wide = np.array([1 + 2 ** -52, 1e100], dtype=np.float64)
        backend = self.k.backend
        has_doubles = backend.supports_float64()
        for rows, expected, uses_device in (
            ([narrow, narrow], np.stack([narrow, narrow]), True),
            ([narrow, wide], np.stack([narrow, wide]), has_doubles),
            ([wide, narrow], np.stack([wide, narrow]), has_doubles),
            (wide, wide, has_doubles),
        ):
            with self.subTest(rows=rows):
                _, result = self.roundtrip(rows)
                self.assertEqual(backend.is_backend_array(result), uses_device)
                if uses_device:
                    self.assert_on_selected_device(result)
                    result = backend.to_numpy(result)
                self.assertEqual(result.dtype, np.float64 if has_doubles else expected.dtype)
                np.testing.assert_array_equal(result, expected)

    def test_matrix_and_scalar_and_empty(self):
        _, back = self.roundtrip(self.k("[[1 2] [3 4]]"))
        self.assertEqual(plain(back), [[1, 2], [3, 4]])
        wire, back = self.roundtrip(self.k("1+2"))
        self.assertEqual(plain(back), 3)
        _, back = self.roundtrip(self.k("!0"))
        self.assertEqual(len(back), 0)

    def test_tensor_that_requires_grad_is_detached(self):
        _, leaves = self.leaves()
        wire, back = self.roundtrip(leaves["a"] * 2)
        self.assertIsInstance(wire, np.ndarray)
        self.assertEqual(back.tolist(), [3.0, 4.0, 6.0, 1.0])
        self.assertFalse(back.requires_grad)

    def test_mixed_list_with_tensors_inside(self):
        value = self.k('[1 2 3],"a"')                      # an object array
        wire, back = self.roundtrip(value)
        self.assertEqual(plain(back), [1, 2, 3, "a"])
        value = self.k("[1 4]:_!6")                        # an object array of tensors
        wire, back = self.roundtrip(value)
        self.assertTrue(all(isinstance(x, np.ndarray) for x in wire))
        self.assertEqual(plain(back), [[0], [1, 2, 3], [4, 5]])

    def test_dictionary_with_tensor_values(self):
        from klongpy.ipc.codec import decode, encode
        from klongpy.ipc.endpoint import to_wire_value
        d = self.k(":{}")
        d[KGSym("a")] = self.k("[1 2 3]")
        try:
            wire = to_wire_value(self.k, d)
            back = decode(encode(wire))
        except Exception as e:  # noqa: BLE001
            self.fail(f"a dictionary holding a tensor cannot be sent: {type(e).__name__}: {e}")
        self.assertEqual(plain(back[KGSym("a")]), [1, 2, 3])


class TestNumericGradientWidth(TorchCase):
    """The finite differences of a ∇ are computed in float64 on the CPU whatever the device is (MPS has no float64)."""

    def test_scalar_point_is_exact_to_float32(self):
        self.assertEqual(float(self.k("3.0∇{x^2}")), 6.0)
        self.assertEqual(float(self.k("3.0∇{x*x*x*x}")), 108.0)

    def test_vector_point(self):
        g = self.k("[1.5 2.5]∇{+/x^3}")
        self.assertEqual(g.dtype, self.k("[1.5 2.5]").dtype)
        self.assertTrue(_close(g.detach().cpu().flatten().tolist(), [6.75, 18.75]))

    def test_numeric_jacobian_fallback_returns_to_device(self):
        with mock.patch.object(self.k.backend, "compute_jacobian", side_effect=NotImplementedError):
            result = self.k("[1.5 2.5]∂{x^2}")
        self.assertTrue(self.k.backend.is_backend_array(result))
        self.assertEqual(result.dtype, self.k("[1.5 2.5]").dtype)
        self.assert_on_selected_device(result)
        np.testing.assert_allclose(self.k.backend.to_numpy(result), [[3, 0], [0, 5]], atol=1e-5)

    def test_a_tensor_made_before_is_read_on_its_own_device(self):
        self.k("w::[1.0 2.0]")
        g = self.k("[3.0 4.0]∇{+/w*x*x}")
        self.assertTrue(_close(g.detach().cpu().flatten().tolist(), [6.0, 16.0]))
        self.assertEqual(self.k("s::2.0;3.0∇{s*x*x}"), 12.0)


if __name__ == "__main__":
    unittest.main()
