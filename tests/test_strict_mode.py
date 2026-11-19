"""
Strict levels (docs/strict.md).

One integer, 0 (the default), 1 or 2, set by KlongInterpreter(strict=n), the
`strict` property, `kgpy --strict[=n]`, .strict(), .strict(n) and .unsafe().
This file has the plumbing and the level 1 rules; the level 2
rules are in tests/test_strict_level2.py. Every rule is run at all three levels.
"""
import importlib.util
import inspect
import os
import subprocess
import sys
import unittest
import warnings

from klongpy import KlongInterpreter
from klongpy.core import KGLambda, KGSym, KlongException
from klongpy.errors import KlongError, KlongNameError, KlongSyntaxError
from tests.backend_compat import on_rust_engine, requires_python_engine
from klongpy.interpreter import KlongContext

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
LEVELS = (0, 1, 2)


def run(level, *lines, parser=None):
    k = KlongInterpreter(strict=level, parser=parser)
    r = None
    for line in lines:
        r = k(line)
    return r


class TestPlumbing(unittest.TestCase):
    def test_discarding_an_interpreter_leaves_the_process_streams_open(self):
        # .cin/.cout/.cerr wrap the process streams and do not own them: an
        # interpreter that is collected (or never finished construction, as
        # with a bad level) must not close sys.stdout.
        import gc
        import sys
        k = KlongInterpreter()
        k("1+1")
        del k
        with self.assertRaises(ValueError):
            KlongInterpreter(strict=3)
        gc.collect()
        self.assertFalse(sys.stdout.closed)
        self.assertFalse(sys.stderr.closed)
        self.assertFalse(sys.stdin.closed)

    def test_defaults_are_zero_and_equal(self):
        self.assertEqual(KlongInterpreter().strict, 0)
        self.assertEqual(inspect.signature(KlongContext.__init__).parameters["strict_mode"].default, 0)
        self.assertEqual(KlongInterpreter()._context._strict_mode, 0)

    def test_constructor_argument_and_property(self):
        for level in LEVELS:
            k = KlongInterpreter(strict=level)
            self.assertEqual(k.strict, level)
            self.assertEqual(k._context._strict_mode, level)        # one stored level
        k = KlongInterpreter()
        k.strict = 2
        self.assertEqual((k.strict, k._context._strict_mode), (2, 2))
        k.strict = 0
        self.assertEqual(k.strict, 0)

    def test_invalid_levels(self):
        for bad in (3, -1, "1", 1.0, None, True):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    KlongInterpreter(strict=bad)
                k = KlongInterpreter()
                with self.assertRaises(ValueError):
                    k.strict = bad
                self.assertEqual(k.strict, 0)

    def test_dot_strict_and_unsafe(self):
        k = KlongInterpreter()
        self.assertEqual(k(".strict()"), 1)
        self.assertEqual(k.strict, 1)
        self.assertEqual(k(".strict(2)"), 2)
        self.assertEqual(k.strict, 2)
        self.assertEqual(k(".strict(0)"), 0)
        self.assertEqual(k.strict, 0)
        self.assertEqual(k(".strict(1)"), 1)
        self.assertEqual(k(".unsafe()"), 0)
        self.assertEqual(k.strict, 0)

    def test_dot_strict_rejects_other_values(self):
        for bad in ("3", "-1", '"1"', "1.5", "[1]", ":a"):
            with self.subTest(bad=bad):
                k = KlongInterpreter(strict=1)
                with self.assertRaises(KlongError) as c:
                    k(f".strict({bad})")
                self.assertIn("0, 1 or 2", str(c.exception))
                self.assertEqual(k.strict, 1)                       # unchanged

    def test_dot_strict_takes_zero_or_one_argument(self):
        k = KlongInterpreter()
        if not on_rust_engine():        # the Python engine's system function object
            self.assertIsInstance(k._context[KGSym(".strict")], KGLambda)
        # inside a function the caller's x is not taken for the level
        self.assertEqual(k("{x;.strict()}(2)"), 1)
        self.assertEqual(k("{.strict(x)}(2)"), 2)
        self.assertEqual(k("{x;.unsafe()}(2)"), 0)

    def test_the_level_in_force_when_a_statement_is_parsed_applies(self):
        # the same text at level 0 and then at level 1 in one interpreter (the parse cache is per level)
        k = KlongInterpreter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(k("1::5"), 5)
            k.strict = 1
            with self.assertRaises(KlongSyntaxError):
                k("1::5")
            k.strict = 0
            self.assertEqual(k("1::5"), 5)

    def test_run_time_checks_go_by_the_level_in_force_when_the_code_runs(self):
        k = KlongInterpreter()
        k("f::{c::1}")                      # defined at level 0
        self.assertEqual(k("f(0)"), 1)
        k(".strict()")
        with self.assertRaises(KlongNameError):
            k("f(0)")
        k(".unsafe()")
        self.assertEqual(k("f(0)"), 1)

    def test_loaded_files_run_at_the_session_level(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "strictprobe.kg")
            with open(path, "w") as f:
                f.write("g::{newname::1}\nr::g(0)\n")
            k = KlongInterpreter()
            k(f'.l("{path}")')
            self.assertEqual(k("r"), 1)
            k = KlongInterpreter(strict=1)
            with self.assertRaises(KlongNameError):
                k(f'.l("{path}")')

    def test_no_global_prefix_is_promised(self):
        import klongpy
        base = os.path.dirname(klongpy.__file__)
        found = []
        for dp, _, fns in os.walk(base):
            for fn in fns:
                if fn.endswith((".py", ".kg")):
                    with open(os.path.join(dp, fn), encoding="utf-8") as f:
                        if "global:" in f.read():
                            found.append(fn)
        self.assertEqual(found, [])


class TestAssignmentInsideAFunction(unittest.TestCase):
    def test_level_0_creates_a_frame_local(self):
        k = KlongInterpreter()
        self.assertEqual(k("g::{b::5};g(0)"), 5)
        self.assertEqual(k("b"), KGSym("b"))                        # no global was made

    def test_undeclared_name(self):
        self.assertEqual(run(0, "f::{c::1}", "f(0)"), 1)
        for level in (1, 2):
            with self.assertRaises(KlongNameError) as c:
                run(level, "f::{c::1}", "f()")
            e = c.exception
            self.assertEqual(str(e), "undefined variable: c")
            self.assertIn("declare it as a local in the parameter list", e.hint)
            self.assertIsInstance(e, KlongException)

    def test_existing_global_declared_local_and_top_level(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(run(level, "n::0", "inc::{n::n+1}", "inc();inc();n"), 2)
                self.assertEqual(run(level, "{[a];a::1}()"), 1)
                self.assertEqual(run(level, "{[a];a::x;a+1}(2)"), 3)
                self.assertEqual(run(level, "fresh::7", "fresh"), 7)     # top level may create names

    def test_unsafe_restores(self):
        k = KlongInterpreter()
        k("f::{c::1}")
        k(".strict()")
        with self.assertRaises(KlongNameError):
            k("f(0)")
        k(".unsafe()")
        self.assertEqual(k("f(0)"), 1)

    def test_dot_e_creates_a_global_from_a_function(self):
        for level in LEVELS:
            self.assertEqual(run(level, 'f::{.E("made::",$x)}', "f(4)", "made"), 4)


class TestReadInsideAFunction(unittest.TestCase):
    """Level 1: a name error (it was one before, worded as advice for an assignment)."""

    def test_read_of_an_undefined_name(self):
        # level 0: the name is defined as its own symbol, and + fails on it
        with self.assertRaises(KlongError) as c:
            run(0, "f::{a+1}", "f(0)")
        self.assertEqual(c.exception.kind, "type")
        self.assertEqual(run(0, "f::{a;1}", "f(0)"), 1)
        for level in (1, 2):
            with self.assertRaises(KlongNameError) as c:
                run(level, "f::{a+1}", "f()")
            e = c.exception
            self.assertEqual(str(e), "undefined: a")
            self.assertNotIn("parameter list", (e.hint or "") + str(e))

    def test_top_level_read_is_legacy_at_level_1(self):
        self.assertEqual(run(0, "a"), KGSym("a"))
        self.assertEqual(run(1, "a"), KGSym("a"))
        with self.assertRaises(KlongNameError):
            run(2, "a")

    def test_quoted_symbol_inside_a_function(self):
        # a quoted symbol is not a read of a variable: no error at any level
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(run(level, "f::{1;:foo}", "f()"), KGSym("foo"))
                self.assertEqual(run(level, "f::{x,:tag}", "f(1)").tolist(), [1, KGSym("tag")])


class TestAssignmentToANonName(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        self.addCleanup(warnings.resetwarnings)

    def test_number_and_string(self):
        # ("[a b]::[1 2]" was one of these until destructuring made it one)
        for text in ("1::5", '"a"::5', "f(1)::5", "0cx::1", "(1+2)::3", "a::1::5"):
            with self.subTest(text=text):
                run(0, text)                                         # legacy: accepted
                for level in (1, 2):
                    with self.assertRaises(KlongSyntaxError) as c:
                        run(level, text)
                    self.assertIn("left side of ::", str(c.exception))
        self.assertEqual(run(0, "1::5"), 5)
        self.assertEqual(run(0, '"a"::5'), 5)

    def test_position(self):
        with self.assertRaises(KlongSyntaxError) as c:
            run(1, "b::2;1::5")
        e = c.exception
        self.assertEqual(e.span, (5, 6))
        self.assertIn("1:6", e.render())

    def test_names_stay_legal(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(run(level, "(a)::5;a"), 5)
                self.assertEqual(run(level, "a::5;a"), 5)
                self.assertEqual(run(level, "a::b::5;a+b"), 10)
                self.assertEqual(run(level, "a.b::5;a.b"), 5)
                self.assertEqual(run(level, "f::{[t];t::x;t}", "f(3)"), 3)
                self.assertEqual(run(level, "d:::{};d"), {})

    def test_inside_a_function_body_it_is_found_when_the_text_is_parsed(self):
        self.assertIsNotNone(run(0, "f::{1::5}"))
        for level in (1, 2):
            with self.assertRaises(KlongSyntaxError):
                run(level, "f::{1::5}")

    @requires_python_engine("the legacy parser of the Python engine")
    def test_legacy_parser_is_frozen(self):
        for level in LEVELS:
            self.assertEqual(run(level, "1::5", parser="legacy"), 5)


@unittest.skipIf(importlib.util.find_spec("colorama") is None, "colorama not installed")
class TestCommandLine(unittest.TestCase):
    def kgpy(self, *args, stdin=""):
        env = dict(os.environ, KGPY_HISTORY="", PYTHONPATH=ROOT, NO_COLOR="1")
        return subprocess.run([sys.executable, "-W", "ignore", "-m", "klongpy.cli", *args], input=stdin,
                              capture_output=True, text=True, env=env, timeout=120, cwd=ROOT)

    def test_strict_flag_on_an_expression(self):
        r = self.kgpy("--strict", "-e", "f::{c::1};f(0)")
        self.assertEqual(r.returncode, 1)
        self.assertIn("name error: undefined variable: c", r.stderr)
        self.assertIn("declare it as a local in the parameter list", r.stderr)
        r = self.kgpy("-e", "f::{c::1};f(0)")
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "1"))

    def test_levels(self):
        self.assertEqual(self.kgpy("-e", ".strict(0);1::5").stdout.strip(), "5")
        self.assertEqual(self.kgpy("--strict=0", "-e", ".strict(0);1::5").stdout.strip(), "5")
        for flag in ("--strict", "--strict=1", "--strict=2"):
            r = self.kgpy(flag, "-e", "1::5")
            self.assertEqual(r.returncode, 1, flag)
            self.assertIn("syntax error", r.stderr)
        # the level is the one the flag names
        self.assertEqual(self.kgpy("--strict=2", "-e", "f::{x};f(1)").stdout.strip(), "1")
        self.assertEqual(self.kgpy("--strict", "-e", "a").stdout.strip(), ":a")

    def test_space_separated_level(self):
        # --strict 2 is level 2, not level 1 with 2 left over as a file name
        r = self.kgpy("--strict", "2", "-e", "a")
        self.assertEqual(r.returncode, 1)
        self.assertIn("undefined: a", r.stderr)
        self.assertEqual(self.kgpy("--strict", "0", "-e", "1::5").stdout.strip(), "5")
        self.assertEqual(self.kgpy("--strict=1", "-e", "a").stdout.strip(), ":a")

    def test_invalid_level(self):
        for flag in ("--strict=3", "--strict=x", "--strict="):
            r = self.kgpy(flag, "-e", "1")
            self.assertEqual(r.returncode, 2, flag)
            self.assertIn("--strict", r.stderr)

    def test_bare_flag_before_a_file_name(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.kg")
            with open(path, "w") as f:
                f.write('f::{c::1}\n.p("before")\nf(0)\n.p("after")\n')
            r = self.kgpy(path)
            self.assertEqual(r.returncode, 0)
            self.assertIn("after", r.stdout)
            r = self.kgpy("--strict", path)
            self.assertEqual(r.returncode, 1)
            self.assertIn("before", r.stdout)
            self.assertNotIn("after", r.stdout)
            self.assertIn("undefined variable: c", r.stderr)
            r = self.kgpy("--strict=1", "-t", path)
            self.assertEqual(r.returncode, 1)

    def test_repl_takes_the_level(self):
        r = self.kgpy("--strict", stdin="f::{c::1}\nf(0)\n1+1\n")
        self.assertIn("undefined variable: c", r.stdout + r.stderr)
        self.assertIn("2", r.stdout)
        r = self.kgpy(stdin="f::{c::1}\nf(0)\n")
        self.assertNotIn("undefined variable", r.stdout + r.stderr)

    def test_help_names_the_flag(self):
        self.assertIn("--strict", self.kgpy("--help").stdout)


if __name__ == "__main__":
    unittest.main()
