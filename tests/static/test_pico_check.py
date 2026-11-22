"""The Pico firmware's board check, on the desktop.

`native/boards/pico-bench/klong_c.c` registers the compiled examples of `examples/embedded/` as system functions (`.c.mavg`, `.c.pid`,
`.c.fir`) and `.c.check()` runs each of them against the interpreter on fixed inputs. This file builds the same two firmware sources
(`klong_c.c` and the C that `native/boards/pico-bench/tools/gen_c_examples.py` generates, which is how the firmware build gets it too)
into one program with the host build of the native runtime (`native/klong-ffi`, feature `cfun`), drives its console, and compares:

  * the compiled functions with KlongPy's `call_entry` on the same inputs (float32 against double: 1e-5 of 1+|value|, the
    tolerance of the board check; mavg and pid also bit for bit with the harness's own build of the C, which shows that the boundary
    of the firmware changes nothing);
  * the native interpreter, loading the Klong source the firmware carries, with KlongPy (both double: 1e-12);
  * the three pinned `.expect` files;
  * `.c.check()` itself: every row passes, with a heap as large as you like (the interpreter runs fir) and with a heap of the board's
    pool, 92,000 bytes (the table of fir.kg does not fit in it: that row says so, -1, and passes against KlongPy's results);
  * the arguments of the functions (a wrong type or length is a Klong error), and that the check leaves no growth behind.

What the host cannot show: the clock (the times are those of this machine), soft-float speed, the Pico's stack and the size of the
runtime's objects on a 32-bit core. The test needs cargo, a C compiler and numpy; it skips without them.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from .harness import WARN, Build, find_cc
from .common import EMBEDDED, ROOT

NATIVE = os.path.join(ROOT, "native")
BOARD = os.path.join(NATIVE, "boards", "pico-bench")
CC = find_cc()
TOL32 = 1e-5   # float32 against double: about 150 steps of 6e-8, the tolerance of klong_c.c
TOL64 = 1e-12  # double against double
BACKEND_ENV = "_KLONGPY_TEST_BACKEND"  # set by tests/conftest.py

_built = {}


def run_generator(script, output):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("KLONGPY_NATIVE_PATH", None)
    env["KLONG"] = os.path.abspath(env.get("KLONG") or os.path.join(NATIVE, "target", "release", "kg"))
    return subprocess.run([sys.executable, "-S", script, output], env=env, capture_output=True, text=True)


def setUpModule():
    if not (CC and shutil.which("cargo")):
        raise unittest.SkipTest("needs cargo and a C compiler")
    work = tempfile.mkdtemp(prefix="kgpico_")
    _built["work"] = work
    gen = os.path.join(work, "gen")
    r = run_generator(os.path.join(BOARD, "tools", "gen_c_examples.py"), gen)
    if r.returncode:
        raise AssertionError("gen_c_examples.py failed:\n" + r.stdout + r.stderr)
    _built["gen"] = gen
    r = subprocess.run(["cargo", "build", "--release", "-p", "klong-ffi", "--features", "cfun", "--target-dir", "target/pico-check"], cwd=NATIVE, capture_output=True, text=True)
    if r.returncode:
        raise unittest.SkipTest("cannot build klong-ffi with the cfun feature: " + r.stderr[-300:])
    lib = os.path.join(NATIVE, "target", "pico-check", "release", "libklong_ffi.a")
    inc = ["-I" + os.path.join(NATIVE, "klong-ffi", "include"), "-I" + BOARD, "-I" + gen]
    objs = []

    def cc(args, out):
        r = subprocess.run([CC] + args + ["-c", "-o", out], capture_output=True, text=True)
        if r.returncode:
            raise AssertionError("compile failed: %s\n%s" % (" ".join(args), r.stderr))
        objs.append(out)

    for name in ("mavg", "pidc", "fir"):  # as the firmware's build compiles them: C99, pedantic, no warnings, no default trap
        cc(["-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic", "-DKG_NO_DEFAULT_TRAP", "-I" + gen, os.path.join(gen, name + ".c")], os.path.join(work, name + ".o"))
    cc(["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I" + gen, os.path.join(gen, "klong_c_data.c")], os.path.join(work, "data.o"))
    cc(["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror"] + inc + [os.path.join(BOARD, "klong_c.c")], os.path.join(work, "klong_c.o"))
    cc(["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror"] + inc + [os.path.join(BOARD, "tools", "host_driver.c")], os.path.join(work, "driver.o"))
    exe = os.path.join(work, "pico_check")
    r = subprocess.run([CC] + objs + [lib, "-o", exe], capture_output=True, text=True)
    if r.returncode:
        raise AssertionError("link failed:\n" + r.stderr)
    _built["exe"] = exe


def tearDownModule():
    if _built.get("work"):
        shutil.rmtree(_built["work"], ignore_errors=True)


def run(lines, heap=None):
    """Type the lines at the console of the program; the outputs, one per line."""
    cmd = [_built["exe"]] + (["--heap-limit", str(heap)] if heap else [])
    r = subprocess.run(cmd, input="".join(l + "\n" for l in lines), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.returncode
    parts = r.stdout.split("klong> ")
    assert len(parts) == len(lines) + 2, (len(parts), r.stdout[:300])
    return [p.rstrip("\n") for p in parts[1:-1]]


def floats(text):
    return np.array([float(t) for t in text.strip("[]").split()])


def ints(text):
    return [int(t) for t in re.findall(r"-?\d+", text)]


def inputs(name):
    """The fixed inputs of the board check, as the generated data file holds them."""
    with open(os.path.join(_built["gen"], "klong_c_data.c")) as f:
        src = f.read()
    m = re.search(r"static const double in_%s\[\d+\] = \{([^}]*)\}" % name, src)
    return np.array([float(t) for t in m.group(1).split(",")])


def klongpy_results(name):
    from klongpy.static import call_entry, load_program

    path = os.path.join(EMBEDDED, name + ".kg")
    xs = inputs(name)
    externs = {"read_adc": lambda channel: 0.3, "write_pwm": lambda channel, duty: 0} if name == "pid" else None
    k = load_program(path, externs=externs)
    if name == "fir":
        return np.asarray(call_entry(k, "run", xs), dtype=float)
    return np.array([float(call_entry(k, "step", x)) for x in xs])


def klist(xs):
    return "[" + " ".join(repr(float(x)) for x in xs) + "]"


def assert_close(test, got, want, tol, what):
    err = np.abs(got - want) / (1.0 + np.abs(want))
    test.assertEqual(len(got), len(want), what)
    test.assertTrue(np.all(err <= tol), "%s: worst relative error %g at %d (%g against %g)" % (what, err.max(), err.argmax(), got[err.argmax()], want[err.argmax()]))


class FixtureGeneration(unittest.TestCase):
    def test_a_wrong_count_pin_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            script = os.path.join(directory, "native", "boards", "pico-bench", "tools", "gen_c_examples.py")
            os.makedirs(os.path.dirname(script))
            shutil.copyfile(os.path.join(BOARD, "tools", "gen_c_examples.py"), script)
            embedded = os.path.join(directory, "examples", "embedded")
            shutil.copytree(EMBEDDED, embedded)
            pin = os.path.join(embedded, "fir.expect")
            with open(pin) as f:
                text = f.read()
            self.assertIn("#run(256:^1.0) => 241\n", text)
            with open(pin, "w") as f:
                f.write(text.replace("#run(256:^1.0) => 241\n", "#run(256:^1.0) => 242\n"))
            result = run_generator(script, os.path.join(directory, "generated"))
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("fir.expect: #run(256:^1.0) gave 241, expected 242", result.stderr)


class CompiledAgainstKlongPy(unittest.TestCase):
    def test_mavg_and_pid(self):
        for name, fn, load in (("mavg", ".c.mavg", "step"), ("pid", ".c.pid", "step")):
            xs = inputs(name)
            want = klongpy_results(name)
            got = floats(run(['.c.reset("%s")' % name, "%s'%s" % (fn, klist(xs))])[1])
            assert_close(self, got, want, TOL32, name + " compiled against KlongPy")

    def test_fir(self):
        want = klongpy_results("fir")
        got = floats(run([".c.fir(0.5*!256)"])[0])
        assert_close(self, got, want, TOL32, "fir compiled against KlongPy")

    def test_the_boundary_changes_nothing(self):
        """The compiled value through the firmware's functions is the value of the same C called directly: bit for bit."""
        for name, prefix, fn in (("mavg", "mavg", ".c.mavg"), ("pid", "pidc", ".c.pid")):
            with open(os.path.join(EMBEDDED, name + ".kg")) as f:
                text = f.read()
            externs = {"read_adc": lambda channel: 0.3, "write_pwm": lambda channel, duty: 0} if name == "pid" else None
            b = Build(text, name=prefix, externs=externs, path=os.path.join(EMBEDDED, name + ".kg"))   # pid.kg loads board.kg next to it
            xs = inputs(name)
            direct = np.array([float(np.float32(b.call("step", float(np.float32(x)))[1])) for x in xs])
            got = floats(run(['.c.reset("%s")' % name, "%s'%s" % (fn, klist(xs))])[1])
            self.assertTrue(np.array_equal(got, direct), name)

    def test_the_expect_files(self):
        got = floats(run(['.c.reset("mavg")', ".c.mavg'[4.0 8.0 12.0 16.0 20.0]"])[1])
        self.assertEqual(list(got), [1.0, 3.0, 6.0, 10.0, 14.0])  # exact in float32: mavg.expect
        got = floats(run(['.c.reset("pid")', ".c.pid'[0.31 0.31 0.5]"])[1])
        assert_close(self, got, np.array([0.12005, 0.0201, 1.0]), TOL32, "pid.expect")
        got = floats(run([".c.fir(0.5*!256)"])[0])
        self.assertEqual(len(got), 241)
        self.assertEqual(got[0], 3.75)
        # GCC's last compiled output is one float32 step above the fir.expect pin.
        np.testing.assert_array_max_ulp(np.float32(got[-1]), np.float32(123.75), maxulp=1)


class InterpreterAgainstKlongPy(unittest.TestCase):
    def test_the_source_the_firmware_carries(self):
        """.c.load loads the text of the image into the native interpreter; it answers what KlongPy does (doubles on both sides)."""
        for name, expr in (("mavg", "step'%s"), ("pid", "step'%s"), ("fir", "run(%s)")):
            xs = inputs(name)
            arg = "0.5*!256" if name == "fir" else klist(xs)
            out = run(['.c.load("%s")' % name, expr % arg])
            self.assertEqual(out[0], "1", out[0])
            # KlongPy on the torch backend computes these in float32
            tol = TOL64 if os.environ.get(BACKEND_ENV, "numpy") == "numpy" else TOL32
            assert_close(self, floats(out[1]), klongpy_results(name), tol, name + " native against KlongPy")

    def test_the_text_is_the_file(self):
        for name in ("mavg", "pid", "fir"):
            with open(os.path.join(EMBEDDED, name + ".kg")) as f:
                text = f.read()
            got = run(['.c.src("%s")' % name])[0]
            self.assertTrue(got.startswith('":"') and text.splitlines()[-1].replace('"', '""') in got, name)


class BoardCheck(unittest.TestCase):
    def rows(self, heap=None, times=1):
        lines = []
        for _ in range(times):
            lines += [".c.check()", ".mem()"]
        out = run([".mem()"] + lines, heap)
        return [ints(o) for o in out[1::2]], [ints(o) for o in out[0::2]]

    def test_every_row_passes(self):
        rows, mem = self.rows()
        self.assertEqual(len(rows[0]), 15)
        for (ok, n, interp_us, c_us, ppb), name, want_n in zip((rows[0][i:i + 5] for i in range(0, 15, 5)), ("mavg", "pid", "fir"), (64, 32, 241)):
            self.assertEqual((ok, n), (1, want_n), name)
            self.assertGreaterEqual(interp_us, 0, name + ": the interpreter ran")
            self.assertGreaterEqual(c_us, 0)
            self.assertLessEqual(ppb, int(TOL32 * 1e9), name)

    def test_with_the_boards_pool(self):
        """A heap of the board's size: the interpreter cannot hold fir.kg's table (-1), the row still passes against KlongPy."""
        rows, _ = self.rows(heap=92000)
        mavg, pid, fir = (rows[0][i:i + 5] for i in range(0, 15, 5))
        self.assertEqual((mavg[0], pid[0], fir[0]), (1, 1, 1))
        self.assertGreaterEqual(mavg[2], 0)
        self.assertGreaterEqual(pid[2], 0)
        self.assertEqual(fir[2], -1)
        self.assertLessEqual(fir[4], int(TOL32 * 1e9))

    def test_no_growth(self):
        """The check lets go of what it defined: after the first the in-use heap stops growing (the interpreter keeps the parse trees of
        the last functions until it is idle for 300 ms, which this program never is), and a failed load leaves nothing."""
        _, mem = self.rows(times=8)
        used = [m[0] for m in mem]
        self.assertEqual(used[-1], used[-2])
        self.assertLess(used[-1] - used[0], 20000)

    def test_names_cannot_collide(self):
        """Nothing the board defines is a name of the language's own: `.c.` has none, `.entry` and `.extern` are declarations of KlongPy's."""
        out = run([".c.us()>0", ".entry(\"f\";\"int -> int\")", ".extern(\"g\";\"int -> int\")"])
        self.assertEqual(out[0], "1")


class Arguments(unittest.TestCase):
    def test_wrong_type_or_length_is_a_klong_error(self):
        cases = [(".c.mavg(\"a\")", "type error"), (".c.mavg([1 2])", "type error"), (".c.pid(:s)", "type error"),
                 (".c.fir(!10)", "range error"), (".c.fir(!257)", "range error"), (".c.fir(2.0)", "type error"), (".c.fir(\"abc\")", "type error"),
                 (".c.fir(2 3:^!6)", "type error"), (".c.fir(,[1 2])", "type error"),
                 (".c.reset(\"nope\")", "range error"), (".c.reset(3)", "type error"), (".c.src(:mavg)", "type error"), (".c.load(\"x\")", "range error")]
        out = run([c for c, _ in cases] + ["1+1"])
        for (call, kind), got in zip(cases, out):
            self.assertIn(kind, got, call)
        self.assertEqual(out[-1], "2", "the runtime goes on")

    def test_integers_are_numbers(self):
        got = run(['.c.reset("mavg")', ".c.mavg(8)", ".c.fir(!256)@0"])
        self.assertEqual(got[1], "2.0")
        self.assertAlmostEqual(float(got[2]), 7.5, places=5)  # taps . [0 1 ... 15]


if __name__ == "__main__":
    unittest.main()
