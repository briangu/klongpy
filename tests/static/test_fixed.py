"""
Fixed-point numbers, ticket 113 part 1: one reference semantics (klongpy/static/fixed.py), three implementations that
agree bit for bit. The interpreter stand-in (the typed-IR evaluator behind call_entry), the C emitter, and (in the
remote tier, test_hls_remote.py) the C simulation of the hardware kernel.

  * the reference functions against a brute-force oracle written here with Python integers;
  * fixed<8,4>: every pair of values for every operator, interpreter against C;
  * fixed<32,16> and fixed<64,32>: random vectors, interpreter against C;
  * the conversions, the checker's rules, and the host API.
"""
import unittest

import numpy as np

from klongpy.errors import KlongStaticError
from klongpy.static import call_entry, fixed as fx, load_program, set_param
from klongpy.static.checker import check_source

from .harness import Build, find_cc

CC = find_cc()


def wrap(v, w):
    m = 1 << w
    v %= m
    return v - m if v >= m // 2 else v


def oracle(op, a, b, w, f):
    """Python-integer definition of the semantics of fixed.py."""
    if op == "add":
        return wrap(a + b, w)
    if op == "sub":
        return wrap(a - b, w)
    if op == "mul":
        return wrap((a * b) // (1 << f), w)
    if op == "div":
        return wrap((a * (1 << f)) // b, w)
    if op == "max":
        return max(a, b)
    if op == "min":
        return min(a, b)
    return int({"lt": a < b, "gt": a > b, "eq": a == b}[op])


class TestReference(unittest.TestCase):
    def test_every_pair_of_fixed_8_4_matches_the_oracle(self):
        spec = fx.Spec(8, 4)
        v = np.arange(-128, 128, dtype=np.int64)
        a, b = np.meshgrid(v, v, indexing="ij")
        a, b = a.reshape(-1), b.reshape(-1)
        for op, fn in (("add", fx.add), ("sub", fx.sub), ("mul", fx.mul)):
            got = fn(a, b, spec)
            want = np.array([oracle(op, int(x), int(y), 8, 4) for x, y in zip(a, b)])
            self.assertTrue(np.array_equal(got, want), op)
        keep = b != 0
        got = fx.div(a[keep], b[keep], spec)
        want = np.array([oracle("div", int(x), int(y), 8, 4) for x, y in zip(a[keep], b[keep])])
        self.assertTrue(np.array_equal(got, want))

    def test_wide_multiply_matches_the_oracle(self):
        rng = np.random.default_rng(3)
        for w, i in ((40, 8), (53, 20), (64, 32), (64, 1), (64, 64)):
            spec = fx.Spec(w, i)
            a = rng.integers(-(1 << (w - 1)), 1 << (w - 1), 200, dtype=np.int64)
            b = rng.integers(-(1 << (w - 1)), 1 << (w - 1), 200, dtype=np.int64)
            got = fx.mul(a, b, spec)
            want = [oracle("mul", int(x), int(y), w, w - i) for x, y in zip(a, b)]
            self.assertEqual([int(g) for g in got], want, (w, i))

    def test_conversions(self):
        s = fx.Spec(8, 4)
        self.assertEqual(int(fx.from_float(1.99, s)), 31)          # truncation toward minus infinity
        self.assertEqual(int(fx.from_float(-1.99, s)), -32)
        self.assertEqual(int(fx.from_float(9.0, s)), wrap(9 * 16, 8))     # wraps
        self.assertEqual(int(fx.from_float(float("nan"), s)), 0)
        self.assertEqual(int(fx.from_int(3, s)), 48)
        self.assertEqual(int(fx.from_int(9, s)), wrap(9 * 16, 8))
        self.assertEqual(int(fx.convert(np.int64(48), fx.Spec(8, 4), fx.Spec(16, 4))), 48 << 8)
        self.assertEqual(int(fx.convert(np.int64(-1), fx.Spec(16, 4), fx.Spec(8, 4))), -1)      # floor shift
        self.assertEqual(int(fx.floor_int(np.int64(-1), s, 32)), -1)
        self.assertEqual(int(fx.floor_int(np.int64(17), s, 32)), 1)


OPS = {
    "add": "x+y", "sub": "x-y", "mul": "x*y", "div": "x%y", "max": "x|y", "min": "x&y",
    "lt": "x<y", "gt": "x>y", "eq": "x=y",
}


def program(t, n, body, result=None):
    result = result or t
    return f'f::{{{body}}}\n.entry("f";"{t}[{n}] {t}[{n}] -> {result}[{n}]")\n'


@unittest.skipIf(CC is None, "no C compiler")
class TestInterpreterAgainstC(unittest.TestCase):
    def pair(self, text, args_raw, spec, out_elem=None):
        """Run f on the evaluator and in C; returns (host raw, C raw)."""
        host = load_program("t.kg", text=text, strict=2, backend="numpy")
        fa = [fx.FixedArray(a, spec) for a in args_raw]
        h = call_entry(host, "f", *fa, raw=True)
        b = Build(text, name="t")
        kind, c = b.call("f", *args_raw)
        self.assertEqual(kind, "ok")
        return np.asarray(h), np.asarray(c)

    def test_fixed_8_4_every_pair_every_operator(self):
        v = np.arange(-128, 128, dtype=np.int64)
        a, b = (x.reshape(-1) for x in np.meshgrid(v, v, indexing="ij"))
        t = "fixed<8,4>"
        for name, body in OPS.items():
            with self.subTest(op=name):
                bb = np.where(b == 0, 1, b) if name == "div" else b
                text = program(t, 65536, body, "int" if name in ("lt", "gt", "eq") else t)
                h, c = self.pair(text, [a, bb], fx.Spec(8, 4))
                self.assertTrue(np.array_equal(h, c.astype(np.int64)), name)
                want = np.array([oracle(name, int(x), int(y), 8, 4) for x, y in zip(a, bb)])
                self.assertTrue(np.array_equal(h, want), name)

    def test_fixed_8_4_unary_operators(self):
        v = np.arange(-128, 128, dtype=np.int64)
        t = "fixed<8,4>"
        for name, body, res in (("neg", "-x", t), ("floor", "_x", "int"), ("not", "~x", "int")):
            with self.subTest(op=name):
                text = f'f::{{{body}}}\n.entry("f";"{t}[256] -> {res}[256]")\n'
                host = load_program("t.kg", text=text, strict=2, backend="numpy")
                h = call_entry(host, "f", fx.FixedArray(v, fx.Spec(8, 4)), raw=True)
                kind, c = Build(text, name="t").call("f", v)
                self.assertEqual(kind, "ok")
                self.assertTrue(np.array_equal(np.asarray(h), c.astype(np.int64)), name)
        self.assertEqual(int(fx.neg(np.int64(-128), fx.Spec(8, 4))), -128)      # wraps

    def random_check(self, w, i, ops, n=4000, seed=0):
        t = f"fixed<{w},{i}>"
        rng = np.random.default_rng(seed)
        lo, hi = -(1 << (w - 1)), (1 << (w - 1))
        a = rng.integers(lo, hi, n, dtype=np.int64)
        b = rng.integers(lo, hi, n, dtype=np.int64)
        a[:4] = [lo, hi - 1, 0, -1]
        b[:4] = [lo, hi - 1, -1, lo]
        for name in ops:
            with self.subTest(w=w, i=i, op=name):
                bb = np.where(b == 0, 1, b) if name == "div" else b
                text = program(t, n, OPS[name], "int" if name in ("lt", "gt", "eq") else t)
                h, c = self.pair(text, [a, bb], fx.Spec(w, i))
                self.assertTrue(np.array_equal(h, c.astype(np.int64)), name)
                want = [oracle(name, int(x), int(y), w, w - i) for x, y in zip(a, bb)]
                self.assertEqual([int(q) for q in h], want)

    def test_fixed_32_16_random(self):
        self.random_check(32, 16, list(OPS))

    def test_fixed_32_8_and_32_31_random(self):
        self.random_check(32, 8, ["mul", "div", "add"], seed=1)
        self.random_check(32, 31, ["mul", "div", "add"], seed=2)

    def test_fixed_64_32_random(self):
        self.random_check(64, 32, [k for k in OPS if k != "div"], n=1500)

    def test_fixed_24_and_48_random(self):
        self.random_check(24, 12, list(OPS), n=1500, seed=5)
        self.random_check(48, 20, [k for k in OPS if k != "div"], n=1500, seed=6)

    def test_conversions_in_a_program(self):
        text = ('f::{.fixed("fixed<16,8>";x)}\ng::{.float(x)}\nh::{.fixed("fixed<32,16>";x)}\nk::{.fixed("fixed<16,8>";x)}\n'
                '.entry("f";"float[6] -> fixed<16,8>[6]")\n.entry("g";"fixed<16,8>[6] -> float[6]")\n'
                '.entry("h";"fixed<16,8>[6] -> fixed<32,16>[6]")\n.entry("k";"fixed<32,16>[6] -> fixed<16,8>[6]")\n')
        host = load_program("t.kg", text=text, strict=2, backend="numpy")
        b = Build(text, name="t")
        xs = np.array([1.5, -1.5, 0.00390625, 127.99609375, -128.0, 200.0])
        h = call_entry(host, "f", xs, raw=True)
        kind, c = b.call("f", xs.astype(np.float32))
        self.assertTrue(np.array_equal(h, c.astype(np.int64)))
        raw = np.array([-32768, -1, 0, 1, 255, 32767], dtype=np.int64)
        fa = fx.FixedArray(raw, fx.Spec(16, 8))
        self.assertTrue(np.array_equal(call_entry(host, "g", fa), b.call("g", raw)[1].astype(np.float64)))
        self.assertTrue(np.array_equal(call_entry(host, "h", fa, raw=True), b.call("h", raw)[1].astype(np.int64)))
        wide = fx.FixedArray(np.array([-70000, -1, 0, 1, 255, 70000], dtype=np.int64), fx.Spec(32, 16))
        self.assertTrue(np.array_equal(call_entry(host, "k", wide, raw=True), b.call("k", wide.raw)[1].astype(np.int64)))

    def test_reductions_and_scans(self):
        t = "fixed<16,8>"
        text = (f'r::{{+/x}}\ns::{{+\\x}}\nm::{{|/x}}\nn::{{&/x}}\nq::{{0+/x}}\n'
                f'.entry("r";"{t}[9] -> {t}")\n.entry("s";"{t}[9] -> {t}[9]")\n.entry("m";"{t}[9] -> {t}")\n'
                f'.entry("n";"{t}[9] -> {t}")\n.entry("q";"{t}[9] -> {t}")\n')
        host = load_program("t.kg", text=text, strict=2, backend="numpy")
        b = Build(text, name="t")
        rng = np.random.default_rng(9)
        for _ in range(50):
            raw = rng.integers(-32768, 32768, 9, dtype=np.int64)
            for name in "rsmnq":
                h = np.asarray(call_entry(host, name, fx.FixedArray(raw, fx.Spec(16, 8)), raw=True))
                c = np.asarray(b.call(name, raw)[1]).astype(np.int64)
                self.assertTrue(np.array_equal(h, c), name)

    def test_state_persists_and_division_by_zero_traps_in_both(self):
        text = ('acc::.fixed("fixed<32,16>";0)\nstep::{acc::acc+x;acc}\nq::{x%y}\n'
                '.entry("step";"fixed<32,16> -> fixed<32,16>")\n.entry("q";"fixed<32,16> fixed<32,16> -> fixed<32,16>")\n')
        host = load_program("t.kg", text=text, strict=2, backend="numpy")
        b = Build(text, name="t")
        for raw in (65536, 100, -3, 2 ** 31 - 1):
            h = call_entry(host, "step", fx.FixedArray(raw, fx.Spec(32, 16)), raw=True)
            kind, c = b.call("step", raw)
            self.assertEqual(int(h), c)
        self.assertEqual(b.call("q", 5, 0), ("trap", 2))
        from klongpy.errors import KlongError
        with self.assertRaises(KlongError):
            call_entry(host, "q", fx.FixedArray(5, fx.Spec(32, 16)), fx.FixedArray(0, fx.Spec(32, 16)))


class TestChecker(unittest.TestCase):
    def rules(self, text):
        return sorted({d.rule for d in check_source(text, "t.kg").diagnostics})

    def sig(self, body, sig="fixed<32,16> fixed<32,16> -> fixed<32,16>", pre=""):
        return f'{pre}f::{{{body}}}\n.entry("f";"{sig}")\n'

    def test_operators_that_are_accepted(self):
        for body in ("x+y", "x-y", "x*y", "x%y", "x|y", "x&y"):
            with self.subTest(body=body):
                self.assertEqual(self.rules(self.sig(body, "fixed<32,16>[4] fixed<32,16>[4] -> fixed<32,16>[4]")), [])
        for body in ("-x", "x*2", "2*x", "x+(-3)", "x%2", "x|0", "+/x", "|/x", "+\\x", "0+/x"):
            with self.subTest(body=body):
                res = "fixed<32,16>[4]" if "\\" in body else "fixed<32,16>"
                if body in ("-x", "x*2", "2*x", "x+(-3)", "x%2", "x|0"):
                    res = "fixed<32,16>[4]"
                self.assertEqual(self.rules(self.sig(body, f"fixed<32,16>[4] -> {res}")), [])
        self.assertEqual(self.rules(self.sig("x<y", sig="fixed<32,16> fixed<32,16> -> int")), [])
        self.assertEqual(self.rules(self.sig("_x", sig="fixed<32,16> -> int")), [])

    def test_mixing_is_an_error(self):
        for body, sig in (("x+y", "fixed<32,16> fixed<16,8> -> fixed<32,16>"), ("x+1.5", "fixed<32,16> fixed<32,16> -> fixed<32,16>"),
                          ("x*y", "fixed<32,16> int -> fixed<32,16>"), ("x+40000", "fixed<32,16> int -> fixed<32,16>"),
                          ("x:%y", "fixed<32,16> fixed<32,16> -> fixed<32,16>"), ("x^2", "fixed<32,16> fixed<32,16> -> fixed<32,16>"),
                          ("x!y", "fixed<32,16> fixed<32,16> -> fixed<32,16>"), ("%x", "fixed<32,16> fixed<32,16> -> float"),
                          ("x%y", "fixed<64,32> fixed<64,32> -> fixed<64,32>"), ("x%0", "fixed<32,16> -> fixed<32,16>")):
            with self.subTest(body=body, sig=sig):
                self.assertTrue(self.rules(self.sig(body, sig)), body)

    def test_an_int_constant_converts_only_where_it_fits(self):
        one = "fixed<32,16> -> fixed<32,16>"
        self.assertEqual(self.rules(self.sig("x+32767", one)), [])
        self.assertIn("fixed", self.rules(self.sig("x+32768", one)))

    def test_entry_result_must_match(self):
        self.assertIn("entry", self.rules(self.sig("x+y", "fixed<32,16> fixed<32,16> -> fixed<16,8>")))
        self.assertIn("entry", self.rules(self.sig("x+y", "fixed<32,16> fixed<32,16> -> float")))

    def test_signature_grammar(self):
        self.assertIn("signature", self.rules('f::{x}\n.entry("f";"fixed<1,1> -> int")\n'))
        self.assertIn("signature", self.rules('f::{x}\n.entry("f";"fixed<8,9> -> int")\n'))
        self.assertIn("signature", self.rules('f::{x}\n.entry("f";"fixed<65,9> -> int")\n'))
        self.assertEqual(self.rules('f::{x}\n.entry("f";"fixed<2,1> -> fixed<64,64>")\n'), ["entry"])


class TestParams(unittest.TestCase):
    def test_param_in_the_interpreter_and_in_c(self):
        text = '.param("gain";"fixed<32,16>";2)\n.param("n";"int";3)\nacc::0\nf::{acc::acc+n;x*gain}\n' \
               '.entry("f";"fixed<32,16> -> fixed<32,16>")\n'
        klong = load_program("t.kg", text=text, strict=2, backend="numpy")
        self.assertEqual(float(call_entry(klong, "f", 1.5)), 3.0)
        set_param(klong, "gain", 0.5)
        self.assertEqual(float(call_entry(klong, "f", 1.5)), 0.75)
        from klongpy.errors import KlongError
        with self.assertRaises(KlongError):
            set_param(klong, "nope", 1)
        if CC:
            b = Build(text, name="t")
            self.assertEqual(b.call("f", 98304), ("ok", 196608))
            b.lib["t_set_gain"].argtypes = [__import__("ctypes").c_void_p, __import__("ctypes").c_int32]
            b.lib["t_set_gain"](b.mem, 32768)
            self.assertEqual(b.call("f", 98304), ("ok", 49152))


if __name__ == "__main__":
    unittest.main()


class TestParamDeclarations(unittest.TestCase):
    def rules(self, text):
        return [(d.line, d.rule) for d in check_source(text, "t.kg").diagnostics]

    def test_declaration_errors(self):
        self.assertEqual(self.rules('.param("p";"int[3]")\nf::{x}\n.entry("f";"int -> int")'), [(1, "param")])
        self.assertEqual(self.rules('.param("p";"str")\nf::{x}\n.entry("f";"int -> int")'), [(1, "param")])
        self.assertEqual(self.rules('.param("while";"int")\nf::{x}\n.entry("f";"int -> int")'), [(1, "param")])
        self.assertEqual(self.rules('.param("p";"int";1.5)\nf::{x}\n.entry("f";"int -> int")'), [(1, "param")])
        self.assertEqual(self.rules('.param("p";"int")\n.param("p";"int")\nf::{x}\n.entry("f";"int -> int")'), [])
        self.assertEqual(self.rules('.param("p";"int")\n.param("p";"float")\nf::{x}\n.entry("f";"int -> int")'), [(2, "param")])
        self.assertEqual(self.rules('.param("p";"int")\np::3\nf::{x}\n.entry("f";"int -> int")'), [(2, "redefinition")])
        self.assertEqual(self.rules('.param(p;"int")\nf::{x}\n.entry("f";"int -> int")'), [(1, "decl-literal")])

    def test_a_parameter_cannot_be_assigned_or_used_as_a_shape(self):
        pre = '.param("p";"int";2)\n'
        self.assertEqual(self.rules(pre + 'f::{p::x;x}\n.entry("f";"int -> int")'), [(2, "param")])
        self.assertEqual(self.rules(pre + 'f::{p#x}\n.entry("f";"float[4] -> float[2]")'), [(2, "const-count")])
        self.assertEqual(self.rules(pre + 'f::{+/x*(!4)<p}\n.entry("f";"float[4] -> float")'), [])    # the first p of a maximum
        self.assertEqual(self.rules(pre + 'f::{p{x*2}:*x}\n.entry("f";"int -> int")'), [(2, "param")])    # a loop bound
        self.assertEqual(self.rules(pre + 'f::{3{x*p}:*x}\n.entry("f";"int -> int")'), [])

    def test_the_interpreter_side(self):
        from klongpy.errors import KlongError
        k = load_program("t.kg", text='.param("p";"int";2)\nf::{x*p}\n.entry("f";"int -> int")\n', strict=2, backend="numpy")
        self.assertEqual(call_entry(k, "f", 3), 6)
        set_param(k, "p", 5)
        self.assertEqual(call_entry(k, "f", 3), 15)
        with self.assertRaises(KlongError):
            set_param(k, "p", 1.5)               # an int parameter takes an int
        with self.assertRaises(KlongError):
            k('.param("p";"float")')            # already defined with another type
        with self.assertRaises(KlongError):
            k('.param("q";"int[2]")')
        self.assertEqual(str(k.externs.params["p"].ty), "int")
        k2 = load_program("t.kg", text='.param("g";"float";0.5)\nf::{x*g}\n.entry("f";"float -> float")\n', strict=2, backend="numpy")
        set_param(k2, "g", 2)                     # an int converts to a float parameter
        self.assertEqual(call_entry(k2, "f", 1.5), 3.0)
