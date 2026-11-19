"""
Differential tests: each program runs under KlongPy and as
compiled C, and the two must give the same answers. See tests/static/harness.py
for how the C side runs and how values are compared.

    TestReferencePrograms    mavg, pid (extern call logs), fir, and the .expect files, through C
    TestOperatorRows         every example of every allowed row and form, random and edge inputs
    TestGeneratedPrograms    a seeded generator of small static programs (see Gen)
    TestEvaluationOrder      side effects happen in Klong's order (extern logs, state)
    TestAliasing             assignments that read their own target
    TestBounds, TestDivisionByZero, TestIntegerOverflow, TestFloor, ...   the policies of the C emitter

Every program runs in two configurations: the default (int32_t, float) and
--int64 --float64. Where the two worlds are specified to differ (see
docs/static-profile.md) the test says so and checks the specified C behaviour instead.
All tests skip when there is no C compiler.
"""
import math
import json
import os
import random
import unittest

import numpy as np

from klongpy.static import FORMS, OPTABLE, parse_signature
from klongpy.static.signature import Ty

from .harness import Build, Host, find_cc, same, wrap
from .test_reference_programs import DIR, read_expect

CONFIGS = ((32, 32), (64, 64))
NAN, INF = float("nan"), float("inf")
I32_MAX, I32_MIN = 2**31 - 1, -2**31
I64_MAX, I64_MIN = 2**63 - 1, -2**63


@unittest.skipIf(find_cc() is None, "no C compiler on the path (cc, clang, gcc or $KG_TEST_CC)")
class Differential(unittest.TestCase):
    maxDiff = None

    def agree(self, h, b, name, args, ty, rtol=None, where="", zero_sign=True):
        """Call entry name with args in both worlds; fail unless both succeed with equal values."""
        hr = h.call(name, *args)
        cr = b.call(name, *args)
        self.assertEqual((hr[0], cr[0]), ("ok", "ok"), f"{where}{name}{args!r}: host {hr!r}, C {cr!r}")
        d = same(hr[1], cr[1], ty, b.float_bits, rtol, zero_sign=zero_sign)
        self.assertIsNone(d, f"{where}{name}{args!r}: {d}")
        return cr[1]

    def state_agrees(self, h, b, names, rtol=None, where="", zero_sign=True):
        for g in b.program.globals:
            if g.state and g.name in names:
                d = same(h.state(g.name), b.state(g.name), g.ty, b.float_bits, rtol, zero_sign=zero_sign)
                self.assertIsNone(d, f"{where}state {g.name}: {d}")


def read(name):
    with open(os.path.join(DIR, name + ".kg")) as f:
        return f.read()


class Adc:
    """Stand-ins for the externs of pid.kg that log every call; one object per world."""
    def __init__(self, samples):
        self.samples = list(samples)
        self.i = 0
        self.log = []

    def read_adc(self, channel):
        v = self.samples[self.i % len(self.samples)]
        self.i += 1
        self.log.append(("read.adc", channel, v))
        return v

    def write_pwm(self, channel, duty):
        self.log.append(("write.pwm", channel, duty))
        return len(self.log)

    def externs(self):
        return {"read.adc": self.read_adc, "write.pwm": self.write_pwm}


class TestReferencePrograms(Differential):
    def test_mavg_over_a_long_run_with_edge_inputs(self):
        rng = random.Random(1)
        runs = [
            [rng.randrange(-4000, 4000) * 0.25 for _ in range(40)],
            [0.0, -0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.5],
            [1e30, 1e30, 1e30, 1e-30, 1.0, 2.0, 3.0, 4.0, 5.0],        # (a cancelling 1e30 is in TestKnownDifferences)
            [-3.5, -7.25, -1e6, -2.0, -0.0, -0.0, -0.0, -0.0, -0.0],
            [1.0, NAN, 2.0, 3.0, 4.0, 5.0, 6.0, INF, 1.0, -INF, 2.0, 3.0, 4.0, 5.0, 6.0],
            [rng.uniform(-1, 1) for _ in range(40)],
        ]
        for cfg in CONFIGS:
            b = Build(read("mavg"), name="mavg", int_bits=cfg[0], float_bits=cfg[1])
            h = Host(read("mavg"))
            for k, run in enumerate(runs):
                b.reset()
                h.reset()
                for i, x in enumerate(run):
                    self.agree(h, b, "step", (x,), Ty("float"), where=f"{cfg} run {k} call {i}: ", zero_sign=False)
                self.state_agrees(h, b, ("buf", "pos"), where=f"{cfg} run {k}: ")

    def test_pid_results_state_and_the_log_of_extern_calls(self):
        rng = random.Random(2)
        samples = [rng.randrange(0, 4096) / 4096.0 for _ in range(50)] + [0.0, 1.0, 0.5, 0.5]
        setpoints = [rng.randrange(0, 1024) / 1024.0 for _ in range(60)] + [0.0, 0.0, 1.0, -5.0, 1e6, -1e6, 0.5]
        for cfg in CONFIGS:
            ha, ca = Adc(samples), Adc(samples)
            b = Build(read("pid"), name="pidc", int_bits=cfg[0], float_bits=cfg[1], externs=ca.externs())
            h = Host(read("pid"), externs=ha.externs())
            for i, sp in enumerate(setpoints):
                self.agree(h, b, "step", (sp,), Ty("float"), rtol=None if cfg[1] == 64 else 2e-4, where=f"{cfg} call {i}: ")
            self.state_agrees(h, b, ("integ", "prev"), rtol=None if cfg[1] == 64 else 2e-4)
            self.assertEqual(len(ha.log), 2 * len(setpoints))
            self.assertEqual([e[:2] for e in ha.log], [e[:2] for e in ca.log], "the same externs, in the same order")
            for he, ce in zip(ha.log, ca.log):
                self.assertIsNone(same(he[2], ce[2], Ty("float"), cfg[1], None if cfg[1] == 64 else 2e-4))

    def test_fir_blocks(self):
        rng = np.random.default_rng(3)
        blocks = [rng.integers(-2048, 2048, 256) / 16.0 for _ in range(6)]
        blocks += [np.zeros(256), -np.ones(256), np.full(256, 1e20), np.arange(256) * 0.5, rng.uniform(-1, 1, 256)]
        special = rng.uniform(-1, 1, 256)
        special[[3, 100, 255]] = [NAN, INF, -INF]
        blocks.append(special)
        for cfg in CONFIGS:
            b = Build(read("fir"), name="fir", int_bits=cfg[0], float_bits=cfg[1])
            h = Host(read("fir"))
            for k, x in enumerate(blocks):
                self.agree(h, b, "run", (x,), Ty("float", (241,)), where=f"{cfg} block {k}: ")

    def test_the_expect_files_through_c(self):
        """The entry calls recorded in each .expect file give the recorded values in C too."""
        from klongpy import KlongInterpreter
        literal = KlongInterpreter(backend="numpy")
        for name in ("mavg", "pid", "fir"):
            externs, steps = read_expect(name)
            for cfg in CONFIGS:
                b = Build(read(name), name=name + "x", int_bits=cfg[0], float_bits=cfg[1], externs=externs)
                h = Host(read(name), externs=externs)
                checked = 0
                for expr, want in steps:
                    if expr.startswith(("step(", "run(")) and expr.endswith(")"):
                        entry, _, arg = expr[:-1].partition("(")
                        value = literal(arg)
                        got = b.call(entry, value)
                        self.assertEqual(got[0], "ok")
                        np.testing.assert_allclose(np.asarray(got[1], dtype=float), np.asarray(literal(want), dtype=float),
                                                   rtol=1e-9 if cfg[1] == 64 else 1e-5, atol=1e-9 if cfg[1] == 64 else 1e-6,
                                                   err_msg=f"{name} {cfg}: {expr}")
                        h.call(entry, value)
                        checked += 1
                    elif expr in [g.name for g in b.program.globals if g.state]:
                        np.testing.assert_allclose(np.asarray(b.state(expr), dtype=float), np.asarray(literal(want), dtype=float),
                                                   rtol=1e-9 if cfg[1] == 64 else 1e-5, atol=1e-9 if cfg[1] == 64 else 1e-6)
                        checked += 1
                self.assertGreaterEqual(checked, 1, name)

    def test_two_instances_do_not_share_state(self):
        b = Build(read("mavg"), name="mavg2")
        first = b.mem
        b.call("step", 8.0)
        b.reset()
        second = b.mem
        self.assertEqual(b.call("step", 4.0), ("ok", 1.0))
        b.mem = first
        self.assertEqual(b.call("step", 8.0), ("ok", 4.0))
        b.mem = second
        self.assertEqual(b.call("step", 4.0), ("ok", 2.0))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

# every float below is exact in float32, so both widths see the same inputs
F_EDGES = [0.0, -0.0, 1.0, -1.0, 0.5, 2.0 ** -10, 123456.75, -1e6, 3.0, -2.5]
F_SPECIAL = [NAN, INF, -INF]
I_EDGES = [0, 1, -1, 2, -2, 7, -7, 1000, -1000]
# at 32 bits KlongPy (64-bit ints) and compiled code differ on overflow by specification: TestIntegerOverflow
I_EXTREME = {32: [], 64: [I64_MAX, I64_MIN, I32_MAX, I32_MIN, 2**40]}


def rand_value(rng, ty):
    def one():
        return rng.randrange(-32, 33) * 0.25 if ty.elem == "float" else rng.randrange(-9, 10)
    if not ty.shape:
        return one()
    return np.array([one() for _ in range(ty.size)], dtype=np.float64 if ty.elem == "float" else np.int64).reshape(ty.shape)


def edge_value(rng, ty, pool):
    if not ty.shape:
        return rng.choice(pool)
    flat = [rng.choice(pool) for _ in range(ty.size)]
    return np.array(flat, dtype=np.float64 if ty.elem == "float" else np.int64).reshape(ty.shape)


def fresh(v):
    """A distinct object with the same value: KlongPy's Match answers 1 for one object compared with itself, even a nan."""
    return v.copy() if isinstance(v, np.ndarray) else (float(np.float64(v)) if isinstance(v, float) else v)


# rows whose operands cannot take every edge value, and why
NO_SPECIAL = {
    "pow": "KlongPy raises a range error for some infinite operands (it converts a whole result to an int)",
    "floor": "the int a nan or an infinity floors to is the host platform's conversion; TestFloor checks the C values",
    "iterate": "the count must be a small non-negative int",
    "while": "the loop must end",
    "each2": "", "over": "", "scan": "",
}
DIVISOR_ROWS = {"div", "idiv", "rem", "recip"}
WRAPPING_ROWS = {"add", "sub", "mul", "neg", "max", "min", "lt", "gt", "eq", "not"}

CUSTOM_INPUTS = {
    ("iterate", "3{x*2}:*x"): [(v,) for v in (0, 1, -1, 5, -9, 1000)],
    ("iterate", "y{x*0.5}:*x"): [(8.0, 2), (1.0, 0), (-3.5, 5), (123456.75, 1), (0.0, 3)],
    ("while", "{x<100}{x*2}:~x"): [(v,) for v in (3, 1, 99, 100, 101, 46340, 7)],
    ("pow", "x^y"): [(x, y) for x in (-3, -1, 1, 2, 3, 5) for y in (-2, -1, 0, 1, 2, 3, 7)] + [(0, 0), (0, 3)],
    ("pow", "x^2"): None,
}


def examples_of(row):
    """The examples of a table row as (body, signature, inputs): inputs is a list of arguments, or None."""
    return [(e.body, e.signature, json.loads(e.inputs) if e.inputs else None) for e in row.examples]


def inputs_for(rng, row, ex, sig, int_bits):
    """The argument tuples one example is called with."""
    body = ex[0]
    custom = CUSTOM_INPUTS.get((row, body))
    if custom:
        return custom
    given = ex[2] if len(ex) > 2 else None
    out = []
    if given is not None:
        # an example with its own inputs has indices that must stay in range: vary the floats only
        base = [np.asarray(a) if isinstance(a, list) else a for a in given]
        out.append(tuple(base))
        for _ in range(6):
            out.append(tuple(rand_value(rng, p) if p.elem == "float" else a for p, a in zip(sig.params, base)))
        return out
    for _ in range(8):
        out.append(tuple(rand_value(rng, p) for p in sig.params))
    fpool = list(F_EDGES) + ([] if row in NO_SPECIAL else F_SPECIAL)
    if row == "pow":
        fpool = [0.5, 1.0, 2.0, 3.0, 1.5, 0.25]
    ipool = list(I_EDGES) + (I_EXTREME[int_bits] if row in WRAPPING_ROWS else [])
    if row == "pow":
        ipool = [0, 1, 2, -1, 3]
    for _ in range(14):
        out.append(tuple(edge_value(rng, p, fpool if p.elem == "float" else ipool) for p in sig.params))
    out.append(tuple(edge_value(rng, p, [0.0] if p.elem == "float" else [0]) for p in sig.params))
    return out


def divides_by_zero(row, body, args):
    """Does this call of a divide row have a zero divisor? (the divisor is the last operand written)"""
    if row not in DIVISOR_ROWS:
        return False
    if body in ("x:%2",):
        return False
    return bool(np.any(np.asarray(args[-1]) == 0))


class TestOperatorRows(Differential):
    """One generated test per example of every allowed row, in both configurations."""

    def rows(self):
        for op in OPTABLE:
            for j, ex in enumerate(examples_of(op)):
                yield op.id, ex, parse_signature(ex[1])

    def test_every_row_agrees_with_klongpy_on_random_and_edge_inputs(self):
        rows = list(self.rows())
        text = "\n".join(f'r{i}::{{{ex[0]}}}\n.entry("r{i}";"{ex[1]}")' for i, (_, ex, _) in enumerate(rows))
        calls = 0
        covered = set()
        for cfg in CONFIGS:
            rng = random.Random(40 + cfg[0])
            b = Build(text, int_bits=cfg[0], float_bits=cfg[1])
            h = Host(text)
            for i, (row, ex, sig) in enumerate(rows):
                # float32 cannot reproduce float64 results of inexact operations: within 1e-5 there, bit for bit otherwise
                exact = row not in ("div", "recip", "pow", "over", "over-seed", "iterate") and "%" not in ex[0]
                rtol = None if cfg[1] == 64 else (0 if exact else 1e-5)
                if row == "pow" and cfg[1] == 64:
                    rtol = 1e-15        # the C library's pow and numpy's may differ in the last bit
                for args in inputs_for(rng, row, ex, sig, cfg[0]):
                    args = tuple(fresh(a) for a in args)
                    where = f"{cfg} row {row} {{{ex[0]}}} "
                    if divides_by_zero(row, ex[0], args):
                        continue        # specified to differ: TestDivisionByZero
                    if row in ("max", "min") and any(np.isnan(np.asarray(a, dtype=float)).any() for a in args):
                        continue        # engine needs: | and & give the other operand for a nan, compiled code a nan (KNOWN_DIFFERENCES)
                    self.agree(h, b, f"r{i}", args, sig.result, rtol, where, zero_sign=row not in ("max", "min"))
                    calls += 1
                covered.add(row)
        self.assertEqual({op.id for op in OPTABLE if op.status == "allowed"} - covered, {"define"})
        self.assertGreater(calls, 3000)

    def test_every_form_agrees_with_klongpy(self):
        stubs = {"read.adc": lambda ch: 0.5 + ch, "log.msg": lambda s, n: None}
        for form in FORMS:
            for prog, sigtext, _ in examples_of(form):
                sig = parse_signature(sigtext)
                text = f'{prog}\n.entry("f";"{sigtext}")'
                for cfg in CONFIGS:
                    rng = random.Random(7)
                    b = Build(text, int_bits=cfg[0], float_bits=cfg[1], externs=stubs)
                    h = Host(text, externs=stubs)
                    for args in inputs_for(rng, form.id, (prog,), sig, cfg[0]):
                        exact = "%" not in prog
                        self.agree(h, b, "f", args, sig.result, None if cfg[1] == 64 else (0 if exact else 1e-5),
                                   f"{cfg} form {form.id}: ")
                    self.state_agrees(h, b, ("acc",))


# ---------------------------------------------------------------------------
# A generator of small static programs
# ---------------------------------------------------------------------------

class Gen:
    """
    Random functions of the static profile over four parameters

        a: float    b: int    c: float[4]    d: int[4]

    three state globals (gs: float, gi: int, gbuf: float[4]) and two constant
    tables (tab: float[4], mat: float[2,3]).

    Every float it computes is a multiple of a power of two small enough to
    be exact in float32 as well as float64 (each expression carries a bound
    on its magnitude and on its number of fraction bits, and a production
    that would exceed 2^22 is replaced by a leaf), and no int leaves 32
    bits, so the host and both C configurations must agree bit for bit (but
    for the sign of a zero: KlongPy's +/ of negative zeros is +0.0, see
    TestSpecifiedDifferences). The values written to the state globals are
    reduced first, which keeps that true over any number of calls.

    A conditional or a [;a;b] list is never the direct operand of a monad:
    KlongPy applies the monad to the condition or the first item instead
    (TestKnownInterpreterDefects).
    """
    PRELUDE = ("gs::0.5\ngi::3\ngbuf::[1.0 -2.0 0.25 4.0]\ntab::[0.5 -1.5 2.0 8.0]\nmat::[[1.0 2.0 3.0] [-4.0 0.5 6.0]]\n"
               "half::{x*0.5}\nmix::{(x*2.0)-y}\nbump::{gi::(gi+x)!16;gi}\n")
    PARAMS = "a;b;c;d"
    SIG = "float int float[4] int[4]"
    LIMIT = 2.0 ** 22

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.locals = []        # (name, kind, bound, frac)
        self.stateful = True

    # each generator returns (source, bound, fraction bits); ints have 0 fraction bits
    def pick(self, options):
        return self.rng.choice(options)()

    def ok(self, bound, frac):
        return bound * (2.0 ** frac) <= self.LIMIT

    def lit_f(self):
        v = self.rng.randrange(-16, 17) * 0.25
        return (f"({v!r})" if v < 0 else repr(v)), max(abs(v), 0.25), 2

    def lit_i(self):
        v = self.rng.randrange(-9, 10)
        return (f"({v})" if v < 0 else str(v)), max(abs(v), 1), 0

    def idx4(self, d):
        """An int in -3..3: a valid index into four items (negative counts from the end)."""
        s, _, _ = self.I(d - 1)
        return f"({s}!4)", 3, 0

    def F(self, d):
        """A float scalar."""
        loc = [(n, b, k) for n, kind, b, k in self.locals if kind == "F"]
        leaves = [lambda: ("a", 8, 2), lambda: ("gs", 4, 2), self.lit_f, lambda: (f"(tab@{self.rng.randrange(-4, 4)})", 8, 1)]
        leaves += [lambda v=v: v for v in loc]
        if d <= 0:
            return self.pick(leaves)

        def binop(op):
            x, bx, kx = self.F(d - 1)
            y, by, ky = self.F(d - 1)
            return f"({x}{op}{y})", (bx + by if op in "+-" else max(bx, by)), max(kx, ky)

        def scale():
            x, b, k = self.F(d - 1)
            m, mb, mk = self.pick([lambda: ("2.0", 2, 0), lambda: ("0.5", 0.5, 1), lambda: ("(-1.0)", 1, 0), lambda: ("3", 3, 0),
                                   lambda: ("0.25", 0.25, 2)])
            return (f"({x}*{m})" if self.rng.random() < 0.5 else f"({m}*{x})"), max(b * mb, 0.25), k + mk

        def div():
            x, b, k = self.F(d - 1)
            return self.pick([lambda: (f"({x}%2.0)", b, k + 1), lambda: (f"({x}%4)", b, k + 2)])

        def of_int():
            x, b, _ = self.I(d - 1)
            return self.pick([lambda: (f"({x}*0.5)", b, 1), lambda: (f"({x}%4)", b, 2), lambda: (f"(0.25+{x})", b + 1, 2)])

        def reduce():
            v, b, k = self.V(d - 1)
            return self.pick([lambda: (f"(+/{v})", 4 * b, k), lambda: (f"(|/{v})", b, k), lambda: (f"(&/{v})", b, k),
                              lambda: (f"({{x-y}}/{v})", 4 * b, k), lambda: (f"(0.5+/{v})", 4 * b + 1, max(k, 1)),
                              lambda: (f"(*{v})", b, k), lambda: (f"({{(x*0.5)+y}}/{v})", 2 * b, k + 3)])

        def index():
            v, b, k = self.V(d - 1)
            i, _, _ = self.idx4(d)
            return f"({v}@{i})", b, k

        def cond():
            c = self.C(d - 1)
            x, bx, kx = self.F(d - 1)
            y, by, ky = self.F(d - 1)
            return f"(0.0+:[{c};{x};{y}])", max(bx, by), max(kx, ky)

        def call():
            x, bx, kx = self.F(d - 1)
            y, by, ky = self.F(d - 1)
            return self.pick([lambda: (f"half({x})", bx, kx + 1), lambda: (f"mix({x};{y})", 2 * bx + by, max(kx, ky))])

        def loop():
            x, b, k = self.F(d - 1)
            return self.pick([lambda: (f"(2{{x*0.5}}:*{x})", b, k + 2), lambda: (f"(3{{x+1.5}}:*{x})", b + 5, max(k, 1)),
                              lambda: (f"({{x<20}}{{(x|0.5)*2}}:~{x})", max(b, 40), k)])

        def rem():
            x, b, k = self.F(d - 1)
            return self.pick([lambda: (f"({x}!2.0)", 2, k), lambda: (f"({x}!3)", 3, k), lambda: (f"(-{x})", b, k)])

        def matrix():
            return f"((mat@{self.rng.randrange(-2, 2)})@{self.rng.randrange(-3, 3)})", 6, 1

        for _ in range(8):
            s, b, k = self.pick([lambda: binop("+"), lambda: binop("-"), lambda: binop("|"), lambda: binop("&"), scale, div, of_int,
                                 reduce, index, cond, call, loop, rem, matrix])
            if self.ok(b, k):
                return s, b, k
        return self.pick(leaves)

    def I(self, d):
        """An int scalar."""
        loc = [(n, b, k) for n, kind, b, k in self.locals if kind == "I"]
        leaves = [lambda: ("b", 9, 0), lambda: ("gi", 16, 0), self.lit_i, lambda: (f"(d@{self.rng.randrange(-4, 4)})", 9, 0)]
        leaves += [lambda v=v: v for v in loc]
        if d <= 0:
            return self.pick(leaves)

        def binop(op):
            x, bx, _ = self.I(d - 1)
            y, by, _ = self.I(d - 1)
            return f"({x}{op}{y})", (bx + by if op in "+-" else max(bx, by)), 0

        def scale():
            x, b, _ = self.I(d - 1)
            m = self.rng.choice([2, 3, -1, -2, 5])
            return f"({x}*{m if m > 0 else f'({m})'})", b * abs(m), 0

        def divide():
            x, b, _ = self.I(d - 1)
            m = self.rng.choice([2, 3, -2, 7, -5])
            op = self.rng.choice(["!", ":%"])
            return f"({x}{op}{m if m > 0 else f'({m})'})", b, 0

        def floor():
            x, b, _ = self.F(d - 1)
            return f"(_{x})", b + 1, 0

        def compare():
            op = self.rng.choice(["<", ">", "="])
            if self.rng.random() < 0.5:
                x, _, _ = self.F(d - 1)
                y, _, _ = self.F(d - 1)
            else:
                x, _, _ = self.I(d - 1)
                y, _, _ = self.I(d - 1)
            return f"({x}{op}{y})", 1, 0

        def logic():
            x, _, _ = self.pick([lambda: self.I(d - 1), lambda: self.F(d - 1)])
            return f"(~{x})", 1, 0

        def match():
            def vectors():
                x, _, _ = self.V(d - 1)
                y, _, _ = self.V(d - 1)
                return x, y

            def ints():
                x, _, _ = self.I(d - 1)
                y, _, _ = self.I(d - 1)
                return x, y
            x, y = self.pick([vectors, ints])
            return f"({x}~{y})", 1, 0

        def reduce():
            v, b, _ = self.W(d - 1)
            return self.pick([lambda: (f"(+/{v})", 4 * b, 0), lambda: (f"(|/{v})", b, 0), lambda: (f"(#{v})", 4, 0),
                              lambda: (f"({v}@{self.idx4(d)[0]})", b, 0), lambda: (f"(1+/{v})", 4 * b + 1, 0)])

        def cond():
            c = self.C(d - 1)
            x, bx, _ = self.I(d - 1)
            y, by, _ = self.I(d - 1)
            return f"(0+:[{c};{x};{y}])", max(bx, by), 0

        def loop():
            x, b, _ = self.I(d - 1)
            return self.pick([lambda: (f"({{x<50}}{{x+7}}:~{x})", max(b, 57), 0), lambda: (f"(3{{x+2}}:*{x})", b + 6, 0),
                              lambda: (f"(-{x})", b, 0)])

        for _ in range(8):
            s, b, k = self.pick([lambda: binop("+"), lambda: binop("-"), lambda: binop("|"), lambda: binop("&"), scale, divide, floor,
                                 compare, logic, match, reduce, cond, loop])
            if b < 2 ** 20:
                return s, b, k
        return self.pick(leaves)

    def C(self, d):
        """A condition: any scalar."""
        return self.pick([lambda: self.I(d), lambda: self.F(d)])[0]

    def V(self, d):
        """A float[4]."""
        loc = [(n, b, k) for n, kind, b, k in self.locals if kind == "V"]
        leaves = [lambda: ("c", 8, 2), lambda: ("gbuf", 4, 2), lambda: ("tab", 8, 1)] + [lambda v=v: v for v in loc]
        if d <= 0:
            return self.pick(leaves)

        def binop(op):
            x, bx, kx = self.V(d - 1)
            y, by, ky = self.pick([lambda: self.V(d - 1), lambda: self.F(d - 1)])
            if self.rng.random() < 0.3:
                x, y = y, x
            return f"({x}{op}{y})", (bx + by if op in "+-" else max(bx, by)), max(kx, ky)

        def scale():
            x, b, k = self.V(d - 1)
            return self.pick([lambda: (f"({x}*0.5)", b, k + 1), lambda: (f"(2*{x})", 2 * b, k), lambda: (f"(-{x})", b, k),
                              lambda: (f"({x}%2)", b, k + 1)])

        def structure():
            x, b, k = self.V(d - 1)
            f, bf, kf = self.F(d - 1)
            n, _, _ = self.I(d - 1)
            both = (max(b, bf), max(k, kf))
            return self.pick([
                lambda: (f"(|{x})", b, k),
                lambda: (f"({n}:+{x})", b, k),
                lambda: (f"({self.rng.choice([1, -1, 5, -6])}:+{x})", b, k),
                lambda: (f"(4#1_{x})", b, k),
                lambda: (f"((-4)#(-1)_{x})", b, k),
                lambda: (f"({f},1_{x})",) + both,
                lambda: (f"((3#{x}),{f})",) + both,
                lambda: (f"((2#{x}),(-2)#{x})", b, k),
                lambda: (f"({x}:={f},{self.idx4(d)[0]})",) + both,
                lambda: (f"({x}:={f},[0 -1])",) + both,
                lambda: (f"({x}@(d!4))", b, k),
                lambda: (f"({x}@[3 -4 1 1])", b, k),
                lambda: (f"(4:^{f})", bf, kf),
                lambda: (f"([4]:^1_{x})", b, k),
                lambda: (f"((([2 3]:^{x})@1),{f})",) + both,
                lambda: (f"(0.0+[;{f};{x}@1;{f};*{x}])",) + both,
                lambda: (f"([;{x};{x}*2]@1)", 2 * b, k),
                lambda: (f"(+/[;{x};|{x}])", 2 * b, k),
                lambda: (f"(((+[2 4]:^{x},|{x})@1)@[0 1 0 1])", b, k),
                lambda: (f"(*,{x})", b, k),
            ])

        def adverb():
            x, b, k = self.V(d - 1)
            y, by, ky = self.V(d - 1)
            f, bf, kf = self.F(d - 1)
            return self.pick([
                lambda: (f"({{x*2.0}}'{x})", 2 * b, k),
                lambda: (f"({{(x*0.5)+{f}}}'{x})", b + bf, max(k + 1, kf)),
                lambda: (f"(half'{x})", b, k + 1),
                lambda: (f"({x}{{x-y}}'{y})", b + by, max(k, ky)),
                lambda: (f"(({x})mix'{y})", 2 * b + by, max(k, ky)),
                lambda: (f"(+\\{x})", 4 * b, k),
                lambda: (f"(1_{f}+\\{x})", 4 * b + bf, max(k, kf)),
                lambda: (f"({{x|y}}\\{x})", b, k),
                lambda: (f"(-:'{x},{f})", 2 * max(b, bf), max(k, kf)),
                lambda: (f"(2{{|x*0.5}}:*{x})", b, k + 2),
                lambda: (f"({{(*x)<16}}{{x+4}}:~{x})", b + 20, k),
                lambda: (f"(0.25*_'{x})", b / 4 + 1, 2),
            ])

        def of_ints():
            w, b, _ = self.W(d - 1)
            return self.pick([lambda: (f"({w}*0.5)", b, 1), lambda: (f"({w}%4)", b, 2)])

        def cond():
            c = self.C(d - 1)
            x, bx, kx = self.V(d - 1)
            y, by, ky = self.V(d - 1)
            return f"(0.0+:[{c};{x};{y}])", max(bx, by), max(kx, ky)

        for _ in range(8):
            s, b, k = self.pick([lambda: binop("+"), lambda: binop("-"), lambda: binop("|"), lambda: binop("&"), scale, structure,
                                 structure, adverb, adverb, of_ints, cond])
            if self.ok(b, k):
                return s, b, k
        return self.pick(leaves)

    def W(self, d):
        """An int[4]."""
        leaves = [lambda: ("d", 9, 0), lambda: ("(!4)", 3, 0), lambda: ("[3 -1 0 2]", 3, 0)]
        if d <= 0:
            return self.pick(leaves)

        def binop(op):
            x, bx, _ = self.W(d - 1)
            y, by, _ = self.pick([lambda: self.W(d - 1), lambda: self.I(d - 1)])
            return f"({x}{op}{y})", (bx + by if op in "+-" else max(bx, by)), 0

        def other():
            x, b, _ = self.W(d - 1)
            v, bv, _ = self.V(d - 1)
            y, _, _ = self.V(d - 1)
            i, bi, _ = self.I(d - 1)
            return self.pick([
                lambda: (f"(_{v})", bv + 1, 0), lambda: (f"({v}<{y})", 1, 0), lambda: (f"({v}={y})", 1, 0), lambda: (f"(~{x})", 1, 0),
                lambda: (f"(|{x})", b, 0), lambda: (f"({i}:+{x})", b, 0), lambda: (f"({x}@({x}!4))", b, 0),
                lambda: (f"({x}*3)", 3 * b, 0), lambda: (f"({x}!3)", b, 0), lambda: (f"({x}:%2)", b, 0),
                lambda: (f"({x}:={i},1)", max(b, bi), 0), lambda: (f"(+\\{x})", 4 * b, 0), lambda: (f"(-{x})", b, 0),
                lambda: (f"((#'[;{x};{x}]),^[2 4]:^{x},{x})", 4, 0), lambda: (f"({{x*x}}'{x})", b * b, 0),
                lambda: (f"({x}{{x>y}}'{x}+1)", 1, 0),
            ])

        for _ in range(8):
            s, b, k = self.pick([lambda: binop("+"), lambda: binop("-"), lambda: binop("|"), lambda: binop("&"), other, other])
            if b < 2 ** 20:
                return s, b, k
        return self.pick(leaves)

    # -- functions ----------------------------------------------------------------
    def reduced_f(self, d):
        """A float in -4..4 with two fraction bits: what may be stored in the state."""
        x, _, _ = self.F(d)
        return f"(0.25*(_{x})!16)"

    def function(self, depth):
        """(body, signature result) of one function."""
        self.locals = []
        stmts = []
        names = []
        for name, kind in (("t", "F"), ("u", "I"), ("v", "V")):
            if self.rng.random() < 0.45:
                s, b, k = {"F": self.F, "I": self.I, "V": self.V}[kind](depth - 1)
                stmts.append(f"{name}::{s}")
                names.append(name)
                self.locals.append((name, kind, b, k))
        if self.rng.random() < 0.5:
            f = self.reduced_f(depth - 1)
            i = self.idx4(depth)[0]
            stmts.append(self.rng.choice([
                lambda: f"gs::{f}",
                lambda: f"gi::({self.I(depth - 1)[0]}!16)",
                lambda: f"gbuf::(1_gbuf),{f}",
                lambda: f"gbuf::gbuf:={f},{i}",
                lambda: "gbuf::|gbuf",
                lambda: f"gbuf::({self.I(depth - 1)[0]}:+gbuf)",
                lambda: f"gbuf::0.25*(_{self.V(depth - 1)[0]})!16",
                lambda: f"gbuf::gbuf@[1 0 3 2]",
                lambda: f"u2::bump({self.I(depth - 1)[0]})",
            ])())
            if stmts[-1].startswith("u2::"):
                names.append("u2")
                self.locals.append(("u2", "I", 16, 0))
        kind = self.rng.choice("FFIVVW")
        s, _, _ = {"F": self.F, "I": self.I, "V": self.V, "W": self.W}[kind](depth)
        stmts.append(s)
        head = f"[{' '.join(names)}];" if names else ""
        return head + ";".join(stmts), {"F": "float", "I": "int", "V": "float[4]", "W": "int[4]"}[kind]

    def program(self, count, depth):
        fns = []
        text = self.PRELUDE
        for i in range(count):
            body, result = self.function(depth)
            text += f'f{i}({self.PARAMS})::{{{body}}}\n.entry("f{i}";"{self.SIG} -> {result}")\n'
            fns.append((f"f{i}", body, parse_signature(f"{self.SIG} -> {result}")))
        return text, fns

    def inputs(self, n):
        rng = self.rng
        out = []
        for k in range(n):
            if k == 0:
                out.append((0.0, 0, np.zeros(4), np.zeros(4, dtype=np.int64)))
            elif k == 1:
                out.append((-8.0, -9, np.array([-8.0, -0.25, -4.0, -1.5]), np.array([-9, -1, -4, -7])))
            elif k == 2:
                out.append((8.0, 9, np.array([8.0, 7.75, -8.0, 0.25]), np.array([9, -9, 9, 0])))
            else:
                out.append((rng.randrange(-32, 33) * 0.25, rng.randrange(-9, 10),
                            np.array([rng.randrange(-32, 33) * 0.25 for _ in range(4)]),
                            np.array([rng.randrange(-9, 10) for _ in range(4)], dtype=np.int64)))
        return out


class TestGeneratedPrograms(Differential):
    PROGRAMS = int(os.environ.get("KG_TEST_PROGRAMS", "12"))
    FUNCTIONS = 14
    CALLS = 9

    def test_the_generator_is_deterministic(self):
        self.assertEqual(Gen(5).program(4, 3)[0], Gen(5).program(4, 3)[0])
        self.assertNotEqual(Gen(5).program(4, 3)[0], Gen(6).program(4, 3)[0])

    def test_generated_programs_agree_bit_for_bit_in_both_configurations(self):
        calls = 0
        for seed in range(self.PROGRAMS):
            g = Gen(1000 + seed)
            text, fns = g.program(self.FUNCTIONS, 2 + seed % 3)
            inputs = g.inputs(self.CALLS)
            for cfg in CONFIGS:
                try:
                    b = Build(text, int_bits=cfg[0], float_bits=cfg[1])
                except Exception as e:
                    self.fail(f"seed {seed}: the checker or the emitter refused a generated program: {e}\n{text}")
                h = Host(text)
                for args in inputs:
                    for name, body, sig in fns:
                        hr = h.call(name, *args)
                        cr = b.call(name, *args)
                        shown = f"seed {seed} {cfg}\n{g.PRELUDE}{name}({g.PARAMS})::{{{body}}}\n{name}{args!r}"
                        self.assertEqual((hr[0], cr[0]), ("ok", "ok"), f"{shown}\nhost {hr!r}\nC {cr!r}")
                        d = same(hr[1], cr[1], sig.result, cfg[1], rtol=0, zero_sign=False)
                        self.assertIsNone(d, f"{shown}\n{d}")
                        calls += 1
                    self.state_agrees(h, b, ("gs", "gi", "gbuf"), rtol=0, where=f"seed {seed} {cfg} ", zero_sign=False)
        self.assertEqual(calls, self.PROGRAMS * len(CONFIGS) * self.FUNCTIONS * self.CALLS)


# ---------------------------------------------------------------------------
# Evaluation order and aliasing
# ---------------------------------------------------------------------------

class Tracer:
    def __init__(self):
        self.log = []

    def tr(self, v):
        self.log.append(v)
        return v

    def note(self, s, v):
        self.log.append((s, v))


ORDER = [
    ("tr(1)+tr(2)", "int"),
    ("tr(1)-tr(2)*tr(3)", "int"),
    ("[;tr(1);tr(2);tr(3)]", "int[3]"),
    ("add(tr(1);tr(2))", "int"),
    ("tr(1),tr(2)", "int[2]"),
    ("[1 2 3]:=tr(9),tr(0)", "int[3]"),
    ("[1 2 3]:=tr(9),[;tr(0);tr(1)]", "int[3]"),
    (":[tr(1);tr(2);tr(3)]", "int"),
    (":[tr(0);tr(2);tr(3)]", "int"),
    ("(tr(1))+/[;tr(1);tr(2)]", "int"),
    ("[1 2 3]@tr(1)", "int"),
    ("[;tr(3);tr(4)]{x+y}'[;tr(5);tr(6)]", "int[2]"),
    ("tr(2){x+1}:*tr(7)", "int"),
    ("{tr(x)<3}{tr(10)+x}:~tr(1)", "int"),
    ("tr'[1 2 3]", "int[3]"),
    ("{tr(x)+y}/[1 2 3]", "int"),
    ("{tr(x)+y}\\[1 2 3]", "int[3]"),
    ("{tr(x)+y}:'[1 2 3]", "int[2]"),
    ("[7 8]{tr(x)+y}'[1 2]", "int[2]"),
    ("-tr(1)", "int"),
    ("tr(1)~tr(2)", "int"),
    ("tr(1):+[1 2 3]", "int[3]"),
    ("tr(1)%tr(2)", "float"),
    ("tr(5)!tr(2)", "int"),
    ("tr(1)<tr(2)", "int"),
    ("tr(2)^tr(3)", "float"),
    ("(tr(1),tr(2)),tr(3)", "int[3]"),
    ("[a b]::[;tr(3);tr(4)];a-b", "int"),
    ("note(\"first\";tr(1));note(\"second\";tr(2)+tr(3));tr(4)", "int"),
    ("g::5;g+g::7", "int"),
    ("g::5;(g::7)+g", "int"),
    ("g::5;(g::g+1)*g::g+2", "int"),
    ("[a];a::1;(a::5)+a", "int"),
    ("[a];a::1;a+a::5", "int"),
    ("[a];a::1;[;a;a::5;a]", "int[3]"),
    ("[a];a::[1 2 3];(a::a*2)+a", "int[3]"),
    ("[a];a::[1 2 3];a+a::a*2", "int[3]"),
    ("[a b];a::[1 2 3];b::a;a::a:=9,0;b", "int[3]"),
    ("[a];a::[1 2 3];a::(|a)+a;a", "int[3]"),
    ("[a];a::[1 2 3];(a@0)+*a::|a", "int"),
    ("g::5;setg(1)+g", "int"),
    ("g::5;g+setg(1)", "int"),
    ("g::5;{x+g}'[;setg(1);setg(2)]", "int[2]"),
    ("g::5;{setg(x)+g}'[1 2 3]", "int[3]"),
    ("g::5;{setg(x+y)}/[1 2 3]", "int"),
    ("#tr'[1 2 3]", "int"),
]


class TestEvaluationOrder(Differential):
    def test_side_effects_happen_in_klongs_order(self):
        text = ('.extern("tr";"int -> int")\n.extern("note";"str int -> void")\ng::5\nadd::{x+y}\nsetg::{g::g+x;g}\n'
                + "\n".join(f'e{i}::{{{body}}}\n.entry("e{i}";"-> {result}")' for i, (body, result) in enumerate(ORDER)))
        for cfg in CONFIGS:
            ht, ct = Tracer(), Tracer()
            b = Build(text, int_bits=cfg[0], float_bits=cfg[1], externs={"tr": ct.tr, "note": ct.note})
            h = Host(text, externs={"tr": ht.tr, "note": ht.note})
            for i, (body, result) in enumerate(ORDER):
                ht.log.clear()
                ct.log.clear()
                self.agree(h, b, f"e{i}", (), parse_signature("-> " + result).result, 1e-6 if cfg[1] == 32 else None,
                           f"{cfg} {{{body}}}: ")
                self.assertEqual(ct.log, ht.log, f"{cfg} {{{body}}}: the externs were called in another order")
                self.state_agrees(h, b, ("g",))


ALIASING = [
    "buf::(1_buf),x",
    "buf::x,(-1)_buf",
    "buf::|buf",
    "buf::buf+|buf",
    "buf::buf@[3 2 1 0]",
    "buf::(-1):+buf",
    "buf::y:+buf",
    "buf::buf:=x,0",
    "buf::buf:=x,y",
    "buf::buf:=(buf@1),[0 2]",
    "buf::buf*buf@0",
    "buf::{x+buf@0}'buf",
    "buf::+\\buf",
    "buf::{x-y}\\|buf",
    "buf::swap(buf)",
    "buf::2{(|x)+buf}:*buf",
    "buf::buf{x+y*buf@1}'|buf",
    "buf:::[x>0;|buf;buf*2]",
    "buf::buf+x;buf::buf-|buf",
    "buf::4#1_buf",
    "buf::[4]:^|buf",
    "buf::[;buf@3;buf@0;x;buf@1]",
    "buf::(+/buf)+buf",
    "buf::neg(buf)",
    "buf::shift(buf;x)",
    "buf::(buf<|buf)*buf",
    "ibuf::ibuf@|ibuf",
    "ibuf::ibuf:=y,ibuf",
    "ibuf::(ibuf+1)!4",
    "[a b];a::buf;b::|a;buf::b;a::a+b;buf::buf+a",
    "[a];a::buf;buf::buf*2;a::a+buf;buf::a",
    "buf::[;|buf;buf]@0",
    "buf::(2#buf),2#|buf",
    "buf::+/[;buf;|buf;buf@[1 1 0 0]]",
]


class TestAliasing(Differential):
    def test_an_assignment_that_reads_its_target(self):
        text = ("buf::[1.0 2.0 4.0 8.0]\nibuf::[3 0 2 1]\nswap::{buf::|x;x+1}\nneg::{-x}\nshift::{(1_x),y}\n"
                + "\n".join(f'f{i}::{{{body};(+/buf)+0*y}}\n.entry("f{i}";"float int -> float")' for i, body in enumerate(ALIASING)))
        rng = random.Random(9)
        for cfg in CONFIGS:
            b = Build(text, int_bits=cfg[0], float_bits=cfg[1])
            h = Host(text)
            for i, body in enumerate(ALIASING):
                b.reset()
                h.reset()
                for call in range(4):
                    args = (rng.randrange(-8, 9) * 0.5, rng.randrange(-3, 4))
                    # some bodies grow beyond what float32 holds exactly
                    rtol = 0 if cfg[1] == 64 else 1e-5
                    self.agree(h, b, f"f{i}", args, Ty("float"), rtol, f"{cfg} {{{body}}} call {call}: ", zero_sign=False)
                    self.state_agrees(h, b, ("buf", "ibuf"), rtol=rtol, where=f"{cfg} {{{body}}} call {call}: ", zero_sign=False)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

BOUNDS = ('buf::[1.0 2.0 3.0 4.0]\n'
          'at::{x@y}\n.entry("at";"float[4] int -> float")\n'
          'rows::{x@y}\n.entry("rows";"int[3,2] int -> int[2]")\n'
          'gather::{x@y}\n.entry("gather";"float[4] int[3] -> float[3]")\n'
          'amend::{x:=9.5,y}\n.entry("amend";"float[4] int -> float[4]")\n'
          'amends::{x:=9.5,y}\n.entry("amends";"float[4] int[2] -> float[4]")\n'
          'store::{buf::buf:=x,y;+/buf}\n.entry("store";"float int -> float")\n')


class TestBounds(Differential):
    def cases(self):
        v = np.array([1.5, 2.5, 3.5, 4.5])
        m = np.array([[1, 2], [3, 4], [5, 6]])
        for i in range(-7, 8):
            yield "at", (v, i), Ty("float"), -4 <= i < 4
            yield "rows", (m, i), Ty("int", (2,)), -3 <= i < 3
            yield "amend", (v, i), Ty("float", (4,)), -4 <= i < 4
            yield "store", (0.5, i), Ty("float"), -4 <= i < 4
            yield "gather", (v, np.array([0, i, -1])), Ty("float", (3,)), -4 <= i < 4
            yield "amends", (v, np.array([i, 1])), Ty("float", (4,)), -4 <= i < 4
        yield "at", (v, I32_MAX), Ty("float"), False
        yield "at", (v, I32_MIN), Ty("float"), False

    def test_checked_by_default_an_index_out_of_range_traps_where_klongpy_raises(self):
        for cfg in CONFIGS:
            b = Build(BOUNDS, int_bits=cfg[0], float_bits=cfg[1])
            h = Host(BOUNDS)
            traps = 0
            for name, args, ty, inside in self.cases():
                hr = h.call(name, *args)
                cr = b.call(name, *args)
                where = f"{cfg} {name}{args!r}"
                if inside:
                    self.assertEqual((hr[0], cr[0]), ("ok", "ok"), where)
                    self.assertIsNone(same(hr[1], cr[1], ty, cfg[1]), where)
                else:
                    self.assertEqual(hr[0], "error", where)
                    self.assertEqual(type(hr[1]).__name__, "KlongRangeError", where)
                    self.assertEqual(cr, ("trap", 1), where)        # KG_TRAP_BOUNDS
                    traps += 1
                    b.reset()       # after a trap the instance is unspecified
                    h.reset()
            self.assertGreater(traps, 30)

    def test_unchecked_builds_agree_on_every_index_in_range_and_negative_indices_still_wrap(self):
        for cfg in CONFIGS:
            b = Build(BOUNDS, int_bits=cfg[0], float_bits=cfg[1], defines={"KG_BOUNDS": 0})
            h = Host(BOUNDS)
            n = 0
            for name, args, ty, inside in self.cases():
                if inside:      # out of range is undefined behaviour with KG_BOUNDS=0: not run
                    self.agree(h, b, name, args, ty, where=f"{cfg} KG_BOUNDS=0 ")
                    n += 1
            self.assertGreater(n, 40)

    def test_a_constant_index_out_of_range_is_a_compile_error(self):
        from klongpy.errors import KlongStaticError
        with self.assertRaises(KlongStaticError) as cm:
            Build('f::{x@4}\n.entry("f";"float[4] -> float")')
        self.assertEqual(cm.exception.rule, "index-range")


DIV = ('fdiv::{x%y}\n.entry("fdiv";"float float -> float")\n'
       'idivf::{x%y}\n.entry("idivf";"int int -> float")\n'
       'vdiv::{x%y}\n.entry("vdiv";"float[4] float[4] -> float[4]")\n'
       'recip::{%x}\n.entry("recip";"float[4] -> float[4]")\n'
       'frem::{x!y}\n.entry("frem";"float float -> float")\n'
       'idiv::{x:%y}\n.entry("idiv";"int int -> int")\n'
       'irem::{x!y}\n.entry("irem";"int int -> int")\n'
       'vidiv::{x:%y}\n.entry("vidiv";"int[3] int[3] -> int[3]")\n'
       'virem::{x!y}\n.entry("virem";"int[3] int -> int[3]")\n')


class TestDivisionByZero(Differential):
    """
    Specified to differ. KlongPy answers :undefined for x%0, x!0 and x:%0
    (an entry declared to return a number then fails). Compiled code:
    float divide and float remainder by zero follow IEEE under KG_DIV0=ieee
    (the default) and call kg_trap under KG_DIV0=trap; an integer :% or !
    by zero always calls kg_trap.
    """
    FLOATS = [0.0, -0.0, 1.0, -2.5, 7.5, 1e-3, INF, -INF, NAN]

    def float_cases(self):
        for x in self.FLOATS:
            for y in self.FLOATS:
                yield "fdiv", (x, y), Ty("float"), y == 0
                yield "frem", (x, y), Ty("float"), y == 0
        for x in (-7, 0, 3):
            for y in (-2, 0, 5):
                yield "idivf", (x, y), Ty("float"), y == 0
        v = np.array([1.0, -2.0, 0.0, NAN])
        for w in (np.array([2.0, 4.0, -0.5, 1.0]), np.array([0.0, 4.0, 0.0, -0.0]), np.array([1.0, 0.0, 3.0, INF])):
            yield "vdiv", (v, w), Ty("float", (4,)), bool(np.any(w == 0))
            yield "recip", (w,), Ty("float", (4,)), bool(np.any(w == 0))

    def ieee(self, name, args, float_bits):
        """The IEEE result at the width of kg_float."""
        ft = np.float64 if float_bits == 64 else np.float32
        with np.errstate(all="ignore"):
            if name in ("fdiv", "idivf", "vdiv"):
                return np.divide(np.asarray(args[0], dtype=ft), np.asarray(args[1], dtype=ft))
            if name == "recip":
                return np.divide(ft(1.0), np.asarray(args[0], dtype=ft))
            return np.fmod(ft(args[0]), ft(args[1]))

    def test_ieee_by_default(self):
        for cfg in CONFIGS:
            b = Build(DIV, int_bits=cfg[0], float_bits=cfg[1])
            h = Host(DIV)
            zero = 0
            for name, args, ty, by_zero in self.float_cases():
                args = tuple(fresh(a) for a in args)
                cr = b.call(name, *args)
                where = f"{cfg} {name}{args!r}"
                self.assertEqual(cr[0], "ok", where)
                # the IEEE result, whether or not the divisor is zero
                self.assertIsNone(same(self.ieee(name, args, cfg[1]), cr[1], ty, cfg[1], rtol=0), where)
                hr = h.call(name, *args)
                if not by_zero:
                    self.assertIsNone(same(hr[1], cr[1], ty, cfg[1], rtol=None if cfg[1] == 64 else 1e-6), where)
                elif name == "recip":
                    self.assertEqual(hr[0], "ok")           # KlongPy's %a on an array is already IEEE
                    self.assertIsNone(same(hr[1], cr[1], ty, cfg[1]), where)
                    zero += 1
                else:
                    self.assertEqual(hr[0], "error", where)    # :undefined is not a float
                    self.assertFalse(np.all(np.isfinite(np.asarray(cr[1], dtype=float))), where)
                    zero += 1
            self.assertGreater(zero, 30)

    def test_trap_when_asked(self):
        for cfg in CONFIGS:
            b = Build(DIV, int_bits=cfg[0], float_bits=cfg[1], defines={"KG_DIV0": "trap"})
            h = Host(DIV)
            for name, args, ty, by_zero in self.float_cases():
                args = tuple(fresh(a) for a in args)
                cr = b.call(name, *args)
                where = f"{cfg} KG_DIV0=trap {name}{args!r}"
                if by_zero:
                    self.assertEqual(cr, ("trap", 2), where)        # KG_TRAP_DIV0
                else:
                    self.agree(h, b, name, args, ty, None if cfg[1] == 64 else 1e-6, where)

    def test_integer_divide_and_remainder_by_zero_always_trap(self):
        v = np.array([7, -8, 9])
        for cfg in CONFIGS:
            for defines in ({}, {"KG_DIV0": "trap"}, {"KG_DIV0": "ieee"}):
                b = Build(DIV, int_bits=cfg[0], float_bits=cfg[1], defines=defines)
                h = Host(DIV)
                for x in (-7, -1, 0, 1, 7, 46340, I32_MIN, I32_MAX):
                    for y in (-7, -2, -1, 0, 1, 2, 3, I32_MAX):
                        for name in ("idiv", "irem"):
                            where = f"{cfg} {defines} {name}({x};{y})"
                            hr = h.call(name, x, y)
                            cr = b.call(name, x, y)
                            if y == 0:
                                self.assertEqual(hr[0], "error", where)
                                self.assertEqual(cr, ("trap", 2), where)
                            elif cfg[0] == 32 and x == I32_MIN and y == -1 and name == "idiv":
                                self.assertEqual(hr[1], 2**31)                  # KlongPy does not overflow
                                self.assertEqual(cr, ("ok", I32_MIN), where)    # compiled code wraps
                            else:
                                self.assertEqual((hr[0], cr[0]), ("ok", "ok"), where)
                                self.assertEqual(int(hr[1]), cr[1], where)
                for w in (np.array([1, 0, 2]), np.array([3, -2, 5])):
                    hr, cr = h.call("vidiv", v, w), b.call("vidiv", v, w)
                    if 0 in w:
                        self.assertEqual((hr[0], cr), ("error", ("trap", 2)))
                    else:
                        self.assertIsNone(same(hr[1], cr[1], Ty("int", (3,)), cfg[1]))
                self.assertEqual(b.call("virem", v, 0), ("trap", 2))
                self.assertIsNone(same(h.call("virem", v, -2)[1], b.call("virem", v, -2)[1], Ty("int", (3,)), cfg[1]))

    def test_the_four_sign_cases_of_the_design(self):
        """7!-2, (-7)!2, 7:%-2 and (-7):%2 are 1, -1, -3 and -3, as in KlongPy."""
        for cfg in CONFIGS:
            b = Build(DIV, int_bits=cfg[0], float_bits=cfg[1])
            h = Host(DIV)
            for name, x, y, want in (("irem", 7, -2, 1), ("irem", -7, 2, -1), ("idiv", 7, -2, -3), ("idiv", -7, 2, -3)):
                self.assertEqual(b.call(name, x, y), ("ok", want))
                self.assertEqual(int(h.call(name, x, y)[1]), want)
            for x, y, want in ((7.5, 2.0, 1.5), (-7.5, 2.0, -1.5), (7.5, -2.0, 1.5), (-6.0, 3.0, -0.0), (5.0, INF, 5.0)):
                self.assertIsNone(same(want, b.call("frem", x, y)[1], Ty("float"), cfg[1], rtol=0))
                self.assertIsNone(same(h.call("frem", x, y)[1], b.call("frem", x, y)[1], Ty("float"), cfg[1], rtol=0))

    def test_a_constant_integer_division_by_zero_is_a_compile_error(self):
        from klongpy.errors import KlongStaticError
        with self.assertRaises(KlongStaticError) as cm:
            Build('f::{x:%0}\n.entry("f";"int -> int")')
        self.assertEqual(cm.exception.rule, "div-zero")


WRAP = ('add::{x+y}\n.entry("add";"int int -> int")\nsub::{x-y}\n.entry("sub";"int int -> int")\n'
        'mul::{x*y}\n.entry("mul";"int int -> int")\nneg::{-x}\n.entry("neg";"int -> int")\n'
        'p31::{x^31}\n.entry("p31";"int -> int")\np32::{x^32}\n.entry("p32";"int -> int")\n'
        'p63::{x^63}\n.entry("p63";"int -> int")\np64::{x^64}\n.entry("p64";"int -> int")\n'
        'sum::{+/x}\n.entry("sum";"int[4] -> int")\nprod::{*/x}\n.entry("prod";"int[4] -> int")\n'
        'vadd::{x+y}\n.entry("vadd";"int[3] int -> int[3]")\n'
        'acc::0\nstep::{acc::acc+x;acc}\n.entry("step";"int -> int")\n')


class TestIntegerOverflow(Differential):
    """
    Specified to differ at 32 bits: KlongPy computes in 64 bits (and in
    unbounded Python ints for ^ and :%), compiled code wraps at the width
    of kg_int. The exact result, wrapped, is the oracle.
    """
    EXACT = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y, "neg": lambda x: -x,
             "p31": lambda x: x ** 31, "p32": lambda x: x ** 32, "p63": lambda x: x ** 63, "p64": lambda x: x ** 64}

    def test_expected_values_written_by_hand_at_32_bits(self):
        b = Build(WRAP)
        for name, args, want in (
                ("add", (I32_MAX, 1), I32_MIN), ("add", (I32_MIN, -1), I32_MAX), ("sub", (I32_MIN, 1), I32_MAX),
                ("sub", (0, I32_MIN), I32_MIN), ("mul", (65536, 65536), 0), ("mul", (46341, 46341), -2147479015),
                ("mul", (I32_MIN, -1), I32_MIN), ("neg", (I32_MIN,), I32_MIN), ("neg", (I32_MAX,), -I32_MAX),
                ("p31", (2,), I32_MIN), ("p32", (2,), 0), ("p31", (3,), wrap(3 ** 31, 32)), ("p31", (-2,), I32_MIN),
                ("sum", (np.array([2**30] * 4),), 0), ("prod", (np.array([65536, 65536, 3, 5]),), 0),
                ("prod", (np.array([46341, 46341, 1, -1]),), 2147479015)):
            self.assertEqual(b.call(name, *args), ("ok", want), f"{name}{args!r}")
        self.assertEqual(list(b.call("vadd", np.array([I32_MAX, 0, I32_MIN]), 1)[1]), [I32_MIN, 1, I32_MIN + 1])
        b.call("step", I32_MAX)
        self.assertEqual(b.call("step", 1), ("ok", I32_MIN))        # state wraps too

    def test_compiled_arithmetic_is_the_exact_result_wrapped(self):
        rng = random.Random(11)
        for cfg in CONFIGS:
            bits = cfg[0]
            lo, hi = -2 ** (bits - 1), 2 ** (bits - 1) - 1
            b = Build(WRAP, int_bits=cfg[0], float_bits=cfg[1])
            h = Host(WRAP)
            pool = [lo, hi, lo + 1, hi - 1, 0, 1, -1, 2, 3, -3, 2 ** (bits // 2), 2 ** (bits // 2) + 1, -(2 ** (bits // 2))]
            pool += [rng.randrange(lo, hi) for _ in range(20)]
            differ = 0
            for name, fn in self.EXACT.items():
                arity = fn.__code__.co_argcount
                for _ in range(60):
                    args = tuple(rng.choice(pool) for _ in range(arity))
                    exact = fn(*args)
                    cr = b.call(name, *args)
                    self.assertEqual(cr, ("ok", wrap(exact, bits)), f"{cfg} {name}{args!r}")
                    hr = h.call(name, *args)
                    if lo <= exact <= hi:
                        self.assertEqual(int(hr[1]), cr[1], f"{cfg} {name}{args!r}: no overflow, so KlongPy agrees")
                    elif bits == 64 and name in ("add", "sub", "mul", "neg"):
                        self.assertEqual(int(hr[1]), cr[1], f"{cfg} {name}{args!r}: KlongPy's int64 wraps the same way")
                    else:
                        differ += 1
                        if hr[0] == "ok":
                            self.assertNotEqual(int(hr[1]), cr[1])
            self.assertGreater(differ, 20)


FLOOR = 'fl::{_x}\n.entry("fl";"float -> int")\nvfl::{_x}\n.entry("vfl";"float[4] -> int[4]")\n'


class TestFloor(Differential):
    def test_floor_rounds_toward_minus_infinity_and_agrees_in_range(self):
        for cfg in CONFIGS:
            b = Build(FLOOR, int_bits=cfg[0], float_bits=cfg[1])
            h = Host(FLOOR)
            for x in (3.7, -3.7, -3.5, 3.5, 0.0, -0.0, -0.25, 0.999, -1.0, 1e6, -1e6, 2147483520.0, -2147483648.0, 123456.75):
                self.agree(h, b, "fl", (x,), Ty("int"), where=f"{cfg} ")
            self.assertEqual(b.call("fl", -3.5), ("ok", -4))
            self.agree(h, b, "vfl", (np.array([1.5, -1.5, 2.0, -0.5]),), Ty("int", (4,)))

    def test_floor_saturates_at_the_range_of_kg_int(self):
        """Specified to differ: KlongPy converts to int64 (and nan by the host's conversion); compiled code saturates, nan is 0."""
        for cfg in CONFIGS:
            b = Build(FLOOR, int_bits=cfg[0], float_bits=cfg[1])
            hi, lo = (I32_MAX, I32_MIN) if cfg[0] == 32 else (I64_MAX, I64_MIN)
            for x, want in ((1e30, hi), (-1e30, lo), (INF, hi), (-INF, lo), (NAN, 0), (float(2 ** 63), hi), (-float(2 ** 63), lo),
                            (2147483648.0, hi if cfg[0] == 32 else 2147483648), (-2147483904.0, lo if cfg[0] == 32 else -2147483904)):
                self.assertEqual(b.call("fl", x), ("ok", want), f"{cfg} _{x!r}")
            self.assertEqual(list(b.call("vfl", np.array([1e30, -1e30, NAN, 2.5]))[1]), [hi, lo, 0, 2])


SPECIAL = ('mx::{x|y}\n.entry("mx";"float float -> float")\nmn::{x&y}\n.entry("mn";"float float -> float")\n'
           'vmx::{x|y}\n.entry("vmx";"float[4] float -> float[4]")\n'
           'lt::{x<y}\n.entry("lt";"float float -> int")\ngt::{x>y}\n.entry("gt";"float float -> int")\n'
           'eq::{x=y}\n.entry("eq";"float float -> int")\nm::{x~y}\n.entry("m";"float float -> int")\n'
           'vm::{x~y}\n.entry("vm";"float[3] float[3] -> int")\nim::{x~y}\n.entry("im";"int int -> int")\n'
           'nt::{~x}\n.entry("nt";"float -> int")\ntruth::{:[x;1;2]}\n.entry("truth";"float -> int")\n'
           'mixed::{x<y}\n.entry("mixed";"int float -> int")\n')


class TestKnownDifferences(Differential):
    """Where the native interpreter and compiled code give different answers (each an "engine needs" item): the tests are expected
    to fail until the engines agree, then they say so."""

    @unittest.expectedFailure
    def test_a_nan_in_max_and_min(self):
        b = Build(SPECIAL)
        h = Host(SPECIAL)
        for name in ("mx", "mn"):
            self.agree(h, b, name, (1000.0, NAN), Ty("float"), 0)

    @unittest.expectedFailure
    def test_match_of_two_close_integers(self):
        b = Build(SPECIAL)
        h = Host(SPECIAL)
        self.agree(h, b, "im", (100000, 100001), Ty("int"))

    @unittest.expectedFailure
    def test_a_sum_that_loses_a_term_to_order(self):
        """+/ of [1.0 -1e30 1e30 1e-30]: compiled code adds left to right (1e-30), the interpreter's sum gives 1.0."""
        b = Build(read("mavg"), name="mavg", int_bits=32, float_bits=64)
        h = Host(read("mavg"))
        for x in (1e30, -1e30, 1e30, 1e-30, 1.0):
            self.agree(h, b, "step", (x,), Ty("float"), 0, zero_sign=False)


class TestSpecialValues(Differential):
    VALUES = [0.0, -0.0, 1.0, -1.0, 2.5, NAN, INF, -INF, 1e30, 2.0 ** -20]

    def test_min_max_comparisons_not_and_truth_with_nan_infinities_and_signed_zeros(self):
        for cfg in CONFIGS:
            b = Build(SPECIAL, int_bits=cfg[0], float_bits=cfg[1])
            h = Host(SPECIAL)
            for x in self.VALUES:
                self.agree(h, b, "nt", (x,), Ty("int"))
                self.agree(h, b, "truth", (x,), Ty("int"))
                if not math.isnan(x):
                    self.agree(h, b, "vmx", (np.array(self.VALUES[:4]) * 1.0, x), Ty("float", (4,)), 0, zero_sign=False)
                for y in self.VALUES:
                    for name in ("mx", "mn"):
                        if math.isnan(x) or math.isnan(y):
                            continue        # engine needs: KNOWN_DIFFERENCES
                        self.agree(h, b, name, (x, fresh(y)), Ty("float"), 0, f"{cfg} ", zero_sign=False)
                    for name in ("lt", "gt", "eq", "m"):
                        self.agree(h, b, name, (x, fresh(y)), Ty("int"), where=f"{cfg} ")
            for i in (-3, 0, 2, 46340):
                for y in (-3.0, 0.5, 2.0, 46340.0, NAN, INF):
                    self.agree(h, b, "mixed", (i, y), Ty("int"))

    def test_match_is_tolerant_for_two_numbers_and_exact_for_arrays(self):
        for cfg in CONFIGS:
            b = Build(SPECIAL, int_bits=cfg[0], float_bits=cfg[1])
            h = Host(SPECIAL)
            pairs = [(1.0, 1.0), (1.0, 1.000005), (1.0, 1.0001), (100000.0, 100000.5), (100000.0, 100002.0), (0.0, 1e-9), (0.0, 1e-7),
                     (1e10, 1e10 + 1e4), (1e10, 1e10 + 1e6), (-2.5, 2.5), (INF, INF), (INF, -INF), (INF, 1e30), (NAN, 1.0)]
            for x, y in pairs:
                self.agree(h, b, "m", (x, y), Ty("int"), where=f"{cfg} ")
            self.assertEqual(b.call("m", 1.0, 1.000001), ("ok", 1))
            self.assertEqual(b.call("m", 1.0, 1.1), ("ok", 0))
            for x, y in ((5, 5), (5, 6), (-7, 7), (0, 0)):
                self.agree(h, b, "im", (x, y), Ty("int"), where=f"{cfg} ")
            v = np.array([1.0, 2.0, 3.0])
            for w in (v.copy(), v + 1e-7 if cfg[1] == 64 else v + 1e-3, np.array([1.0, 2.0, NAN]), np.array([1.0, -2.0, 3.0])):
                self.agree(h, b, "vm", (v, w), Ty("int"), where=f"{cfg} ")

    def test_nan_never_matches_in_compiled_code(self):
        """A nan matches nothing, compiled or interpreted."""
        b = Build(SPECIAL)
        self.assertEqual(b.call("m", NAN, NAN), ("ok", 0))
        self.assertEqual(b.call("vm", np.array([1.0, NAN, 2.0]), np.array([1.0, NAN, 2.0])), ("ok", 0))
        h = Host(SPECIAL)
        self.assertEqual(h.call("m", NAN, NAN)[1], 0)      # the engine's `~` of two nans is 0 as well


POWER = ('p::{x^y}\n.entry("p";"float float -> float")\nip::{x^y}\n.entry("ip";"int int -> float")\n'
         'sq::{x^2}\n.entry("sq";"float[3] -> float[3]")\ncube::{x^3}\n.entry("cube";"int -> int")\n'
         'r::{x!y}\n.entry("r";"float float -> float")\n')


class TestPowerAndTheFreestandingRuntime(Differential):
    CASES = [(2.0, 0.5), (2.0, 10.0), (2.5, 2.0), (10.0, -3.0), (0.5, 7.0), (9.0, 0.5), (1.5, 2.5), (3.0, 4.0), (7.0, 0.0), (0.0, 2.0),
             (1.0, 100.0), (-2.0, 3.0), (-2.0, 2.0), (100.0, 0.25), (0.001, 3.0), (5.0, 1.0), (-8.0, 1.0 / 3.0)]

    def run_cases(self, b, h, rtol):
        for x, y in self.CASES:
            self.agree(h, b, "p", (x, y), Ty("float"), rtol, where=f"{x}^{y} ")
        for x, y in ((2, 10), (3, -2), (-2, 3), (7, 0), (0, 0), (10, 9), (2, -1)):
            self.agree(h, b, "ip", (x, y), Ty("float"), rtol)
        self.agree(h, b, "sq", (np.array([1.5, -2.5, 100.0]),), Ty("float", (3,)), rtol)
        for x in (-3, 0, 2, 11):
            self.agree(h, b, "cube", (x,), Ty("int"))
        for x, y in ((7.5, 2.0), (-7.5, 2.0), (1e6, 3.0), (5.0, 0.375), (2.0 ** 70, 7.0), (3.0, 2.0 ** 70), (-0.0, 2.0)):
            self.agree(h, b, "r", (x, y), Ty("float"), 0)       # the remainder is exact, with or without a C library

    def test_power_with_the_c_library(self):
        for cfg in CONFIGS:
            b = Build(POWER, int_bits=cfg[0], float_bits=cfg[1])
            self.run_cases(b, Host(POWER), 1e-15 if cfg[1] == 64 else 1e-6)

    def test_power_and_remainder_without_a_c_library(self):
        """KG_FREESTANDING: kg_pow is computed in klong_rt.h, to about 1e-6 for float and 1e-13 for double."""
        for cfg in CONFIGS:
            b = Build(POWER, int_bits=cfg[0], float_bits=cfg[1], freestanding=True)
            self.run_cases(b, Host(POWER), 1e-13 if cfg[1] == 64 else 2e-6)

    def test_every_operator_row_under_the_freestanding_runtime_and_at_O2(self):
        rows = [(op.id, ex, parse_signature(ex[1])) for op in OPTABLE for ex in examples_of(op)]
        text = "\n".join(f'r{i}::{{{ex[0]}}}\n.entry("r{i}";"{ex[1]}")' for i, (_, ex, _) in enumerate(rows))
        for cfg in CONFIGS:
            rng = random.Random(77)
            b = Build(text, int_bits=cfg[0], float_bits=cfg[1], freestanding=True, opt="-O2")
            h = Host(text)
            for i, (row, ex, sig) in enumerate(rows):
                for args in inputs_for(rng, row, ex, sig, cfg[0])[:8]:
                    if row in ("max", "min") and any(np.isnan(np.asarray(a, dtype=float)).any() for a in args):
                        continue        # KNOWN_DIFFERENCES
                    if not divides_by_zero(row, ex[0], args):
                        self.agree(h, b, f"r{i}", args, sig.result, 1e-13 if cfg[1] == 64 else 1e-5, f"{cfg} row {row} ",
                                   zero_sign=row not in ("max", "min"))


class TestMixedWidths(Differential):
    def test_int64_with_float32_and_int32_with_float64(self):
        text = Gen(4242).program(8, 3)
        inputs = Gen(4242).inputs(5)
        for cfg in ((64, 32), (32, 64)):
            b = Build(text[0], int_bits=cfg[0], float_bits=cfg[1])
            h = Host(text[0])
            for args in inputs:
                for name, body, sig in text[1]:
                    self.agree(h, b, name, args, sig.result, 0, f"{cfg} {{{body}}} ", zero_sign=False)


class TestSpecifiedDifferences(Differential):
    def test_the_sum_of_negative_zeros(self):
        """numpy's add.reduce of two or more negative zeros is +0.0; compiled code adds left to right from the first item: -0.0."""
        text = 's::{+/x}\n.entry("s";"float[4] -> float")\nl::{{x+y}/x}\n.entry("l";"float[4] -> float")'
        z = np.array([-0.0] * 4)
        for cfg in CONFIGS:
            b = Build(text, int_bits=cfg[0], float_bits=cfg[1])
            h = Host(text)
            self.assertFalse(np.signbit(h.call("s", z)[1]))
            self.assertTrue(np.signbit(b.call("s", z)[1]))
            self.agree(h, b, "l", (z,), Ty("float"), 0)        # written as a function, both add left to right
            self.agree(h, b, "s", (z,), Ty("float"), 0, zero_sign=False)

    def test_the_rounding_of_a_long_float_sum(self):
        """
        +/ of eight or more floats: numpy sums pairwise, compiled code adds
        left to right, so the two round differently (both within n ulps of
        the sum of magnitudes). Written as {x+y}/a, or as +\\a, KlongPy adds
        left to right too and the results are bit for bit the same.
        """
        text = ('s::{+/x}\n.entry("s";"float[256] -> float")\nd::{+/x*y}\n.entry("d";"float[256] float[256] -> float")\n'
                'l::{{x+y}/x}\n.entry("l";"float[256] -> float")\nc::{+\\x}\n.entry("c";"float[256] -> float[256]")')
        b = Build(text, int_bits=64, float_bits=64)
        h = Host(text)
        rng = np.random.default_rng(1)
        differ = 0
        for _ in range(30):
            x, y = rng.uniform(-1, 1, 256), rng.uniform(-1, 1, 256)
            self.agree(h, b, "l", (x,), Ty("float"), 0)
            self.agree(h, b, "c", (x,), Ty("float", (256,)), 0)
            for name, args, scale in (("s", (x,), np.abs(x).sum()), ("d", (x, y), np.abs(x * y).sum())):
                hv, cv = h.call(name, *args)[1], b.call(name, *args)[1]
                self.assertLessEqual(abs(hv - cv), 256 * 2.3e-16 * scale)
                differ += hv != cv
        self.assertGreater(differ, 0, "numpy no longer sums pairwise: compare bit for bit")

    def test_a_negative_iteration_count(self):
        """KlongPy raises for n{f}:*x with n below zero; compiled code runs the loop zero times."""
        text = 'it::{x{x*2}:*y}\n.entry("it";"int int -> int")'
        b = Build(text)
        h = Host(text)
        self.agree(h, b, "it", (3, 5), Ty("int"))
        self.agree(h, b, "it", (0, 5), Ty("int"))
        self.assertEqual(h.call("it", -1, 5)[0], "error")
        self.assertEqual(b.call("it", -1, 5), ("ok", 5))


class TestFormerInterpreterDefects(Differential):
    """Cases the harness found where KlongPy was wrong; fixed, and now agreement tests."""

    def test_a_monad_applied_to_a_conditional_or_an_expression_list(self):
        # the interpreter used to give a monad the condition (or the first item) as its operand,
        # because KGCond and KGExprArray are list subclasses
        text = ('n::{-:[x;2;3]}\n.entry("n";"int -> int")\nc::{#[;x;x;x]}\n.entry("c";"int -> int")\n'
                'ok::{[t];t:::[x;2;3];-t}\n.entry("ok";"int -> int")')
        b = Build(text)
        h = Host(text)
        self.assertEqual([b.call("n", v)[1] for v in (0, 1, 5)], [-3, -2, -2])
        self.assertEqual(b.call("c", 7), ("ok", 3))
        for v in (0, 1, 5):
            self.agree(h, b, "n", (v,), Ty("int"))
            self.agree(h, b, "ok", (v,), Ty("int"))
        self.agree(h, b, "c", (7,), Ty("int"))


class TestHarness(unittest.TestCase):
    """The comparison itself."""

    def test_same(self):
        f, i = Ty("float"), Ty("int")
        self.assertIsNone(same(1.5, 1.5, f, 64))
        self.assertIsNotNone(same(1.5, 1.5000000000000002, f, 64))
        self.assertIsNone(same(NAN, NAN, f, 64))
        self.assertIsNotNone(same(0.0, -0.0, f, 64))
        self.assertIsNone(same(0.1, float(np.float32(0.1)), f, 32))
        self.assertIsNotNone(same(0.1, 0.1001, f, 32))
        self.assertIsNone(same(0.1, float(np.float32(0.1)), f, 32, rtol=0))
        self.assertIsNone(same(3, 3, i, 32))
        self.assertIsNotNone(same(3, 4, i, 32))
        self.assertIsNone(same(np.float64(4.0), 4, i, 32))      # KlongPy's ^ may answer a whole float
        self.assertIsNotNone(same(np.array([1, 2]), np.array([1, 2, 3]), Ty("int", (3,)), 32))
        self.assertTrue(math.isnan(NAN))

    def test_wrap(self):
        self.assertEqual(wrap(2**31, 32), -2**31)
        self.assertEqual(wrap(-2**31 - 1, 32), 2**31 - 1)
        self.assertEqual(wrap(2**64 + 5, 64), 5)


if __name__ == "__main__":
    unittest.main()
