"""
The reference programs of the static profile, examples/embedded/*.kg. Each one

  * is inside the profile (the checker accepts it without running it),
  * runs under KlongPy at strict level 2 with Python stand-ins for its
    externs and gives the values of its .expect file,
  * and has, once its compile-time definitions are evaluated and handed to
    the checker, a fully known type for every global and every function.

The .expect format: a line ``expression => result`` is evaluated in order in
one session and compared with the result (a Klong literal or expression,
within 1e-9); ``# extern NAME = VALUE`` binds a stand-in that returns VALUE;
other ``#`` lines are comments.
"""
import glob
import os
import re
import unittest

import numpy as np

from klongpy import KlongInterpreter
from klongpy.static import call_entry, check_file, load_program

from .hostutil import host_rtol, to_np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.path.join(ROOT, "examples", "embedded")
PROGRAMS = ["mavg", "pid", "fir"]
EXTERN = re.compile(r"#\s*extern\s+(\S+)\s*=\s*(\S+)\s*$")


def read_expect(name):
    """(stand-ins, [(expression, expected text)]) of examples/embedded/<name>.expect."""
    externs, steps = {}, []
    with open(os.path.join(DIR, name + ".expect")) as f:
        for line in f:
            line = line.rstrip("\n")
            m = EXTERN.match(line)
            if m:
                value = float(m.group(2)) if ("." in m.group(2) or "e" in m.group(2)) else int(m.group(2))
                externs[m.group(1)] = (lambda v: lambda *args: v)(value)
            elif line.strip() and not line.startswith("#"):
                expr, _, want = line.partition(" => ")
                steps.append((expr.strip(), want.strip()))
    return externs, steps


def load(name):
    externs, steps = read_expect(name)
    return load_program(os.path.join(DIR, name + ".kg"), externs=externs), steps


class TestReferencePrograms(unittest.TestCase):
    def test_the_directory_holds_the_three_programs(self):
        have = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(DIR, "*.kg")))
        self.assertEqual(have, sorted(PROGRAMS + ["board"]))         # board.kg: the module of host functions pid.kg loads
        for name in PROGRAMS:
            self.assertTrue(os.path.isfile(os.path.join(DIR, name + ".expect")))

    def test_each_is_inside_the_profile(self):
        for name in PROGRAMS:
            with self.subTest(program=name):
                r = check_file(os.path.join(DIR, name + ".kg"))
                self.assertEqual([str(d) for d in r.diagnostics], [])
                self.assertEqual(len(r.entries), 1)
                self.assertEqual(r.host_only, [])

    def test_each_runs_under_klongpy_with_the_expected_values(self):
        for name in PROGRAMS:
            with self.subTest(program=name):
                klong, steps = load(name)
                self.assertEqual(klong.strict, 2)
                self.assertTrue(steps)
                literal = KlongInterpreter()
                for expr, want in steps:
                    got = klong(expr)
                    np.testing.assert_allclose(np.asarray(to_np(klong, got), dtype=float), np.asarray(to_np(literal, literal(want)), dtype=float),
                                               rtol=host_rtol(), atol=1e-9 if host_rtol() == 0 else 1e-5, err_msg=f"{name}: {expr}")

    def test_every_type_is_known_once_the_definitions_are_evaluated(self):
        want = {
            "mavg": ({"buf": "float[4]", "pos": "int"}, {"step": {("float",): "float"}}),
            "pid": ({"kp": "float", "ki": "float", "kd": "float", "dt": "float", "lo": "float", "hi": "float",
                     "integ": "float", "prev": "float"},
                    {"clamp": {("float",): "float"}, "step": {("float",): "float"}}),
            "fir": ({"raw": "float[16]", "taps": "float[16]", "idx": "int[16,241]"}, {"run": {("float[256]",): "float[241]"}}),
        }
        for name in PROGRAMS:
            with self.subTest(program=name):
                path = os.path.join(DIR, name + ".kg")
                externs, _ = read_expect(name)
                klong = load_program(path, externs=externs)
                first = check_file(path)
                r = check_file(path, globals={g: klong(g) for g in first.globals})
                self.assertEqual([str(d) for d in r.diagnostics], [])
                self.assertEqual(r.unresolved, [])
                self.assertEqual({g: str(i.ty) for g, i in r.globals.items()}, want[name][0])
                got = {f: {tuple(str(a) for a in k): str(v) for k, v in i.specializations.items()} for f, i in r.functions.items()}
                self.assertEqual(got, want[name][1])

    def test_mavg(self):
        klong, _ = load("mavg")
        self.assertEqual([call_entry(klong, "step", v) for v in (4, 8, 12, 16, 20)], [1.0, 3.0, 6.0, 10.0, 14.0])
        r = check_file(os.path.join(DIR, "mavg.kg"))
        self.assertEqual({n: g.assigned for n, g in r.globals.items()}, {"buf": True, "pos": True})    # both are state

    def test_pid(self):
        # acceptance: read_adc fixed at 0.3; setpoints 0.31, 0.31, 0.5
        writes = []
        klong = KlongInterpreter(strict=2)
        klong.externs["read_adc"] = lambda c: 0.3
        klong.externs["write_pwm"] = lambda channel, duty: writes.append((channel, duty)) or 0
        klong(f'.l("{os.path.join(DIR, "pid.kg")}")')
        out = [float(to_np(klong, klong(f"step({v})"))) for v in (0.31, 0.31, 0.5)]
        np.testing.assert_allclose(out, [0.12005, 0.0201, 1.0], rtol=host_rtol(), atol=1e-12 if host_rtol() == 0 else 1e-6)
        np.testing.assert_allclose([float(to_np(klong, klong("integ"))), float(to_np(klong, klong("prev")))], [0.0011, 0.2], rtol=host_rtol(), atol=1e-12 if host_rtol() == 0 else 1e-6)
        self.assertEqual([w[0] for w in writes], [0, 0, 0])
        np.testing.assert_allclose([w[1] for w in writes], out, rtol=host_rtol(), atol=1e-12 if host_rtol() == 0 else 1e-6)
        self.assertEqual(sorted(klong.externs.declared), ["read_adc", "write_pwm"])
        self.assertEqual([(e.name, str(e.signature)) for e in klong.externs.entries], [("step", "float -> float")])
        r = check_file(os.path.join(DIR, "pid.kg"))
        self.assertEqual(sorted(n for n, g in r.globals.items() if g.assigned), ["integ", "prev"])     # the rest is constant
        self.assertEqual([(x.name, x.c_name) for x in r.externs.values()], [("board.adc", "read_adc"), ("board.pwm", "write_pwm")])

    def test_pid_without_stand_ins_names_the_extern(self):
        from klongpy.errors import KlongSystemError
        klong = load_program(os.path.join(DIR, "pid.kg"))
        with self.assertRaises(KlongSystemError) as cm:
            call_entry(klong, "step", 0.31)
        self.assertIn("read_adc", cm.exception.message)
        self.assertEqual(cm.exception.source.name, os.path.join(DIR, "pid.kg"))

    def test_fir(self):
        klong, _ = load("fir")
        ones = call_entry(klong, "run", np.ones(256))
        self.assertEqual(ones.shape, (241,))
        np.testing.assert_allclose(ones, 1.0, rtol=host_rtol(), atol=1e-12 if host_rtol() == 0 else 1e-6)       # the taps sum to 1
        ramp = call_entry(klong, "run", np.arange(256) * 0.5)
        np.testing.assert_allclose([ramp[0], ramp[-1]], [3.75, 123.75], rtol=host_rtol(), atol=1e-9 if host_rtol() == 0 else 1e-5)
        r = check_file(os.path.join(DIR, "fir.kg"))
        self.assertEqual(r.unresolved, ["raw", "taps", "idx"])          # built on the host
        self.assertEqual([n for n, g in r.globals.items() if g.assigned], [])


if __name__ == "__main__":
    unittest.main()
