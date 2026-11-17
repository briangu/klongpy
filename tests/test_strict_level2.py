"""
Strict level 2 (docs/strict.md): reading an undefined name, reading a local before it is assigned
and passing too many arguments are errors, and a quoted symbol is always the
symbol. Every rule is run at levels 0, 1 and 2; levels 0 and 1 keep what they
did before.
"""
import importlib.util
import os
import subprocess
import sys
import unittest

from klongpy import KlongInterpreter
from klongpy.core import KGCall, KGSym, KlongException
from klongpy.errors import KlongArityError, KlongError, KlongNameError, KlongTypeError

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
LEVELS = (0, 1, 2)


def run(level, *lines, parser=None):
    k = KlongInterpreter(strict=level, parser=parser)
    r = None
    for line in lines:
        r = k(line)
    return r


def error(level, *lines):
    try:
        run(level, *lines)
    except KlongError as e:
        return e
    raise AssertionError(f"no error at level {level}: {lines}")


class TestUndefinedRead(unittest.TestCase):
    def test_top_level(self):
        for level in (0, 1):
            self.assertEqual(run(level, "a"), KGSym("a"))               # defined as its own symbol
            self.assertEqual(run(level, "a;a"), KGSym("a"))
            self.assertIsInstance(error(level, "a+1"), KlongTypeError)
        e = error(2, "a")
        self.assertIsInstance(e, KlongNameError)
        self.assertEqual((str(e), e.kind), ("undefined: a", "name"))
        self.assertIsInstance(e, KlongException)
        e = error(2, "a+1")
        self.assertIsInstance(e, KlongNameError)
        self.assertEqual(str(e), "undefined: a")

    def test_nothing_is_defined_by_the_failed_read(self):
        k = KlongInterpreter(strict=2)
        for _ in range(2):
            with self.assertRaises(KlongNameError):
                k("a")
        k("a::1")
        self.assertEqual(k("a+1"), 2)

    def test_inside_a_function(self):
        self.assertEqual(run(0, "f::{a;1}", "f()"), 1)
        for level in (1, 2):                                            # level 1 has this rule too
            self.assertEqual(str(error(level, "f::{a+1}", "f()")), "undefined: a")

    def test_as_an_argument_and_in_other_positions(self):
        for text in ("f(a)", "{x}'a", "[;a;1]", ":[a;1;2]", ":[1;a;2]", "b::a", "1,a", "#a", "g(1;a)", "+/a"):
            with self.subTest(text=text):
                e = error(2, "f::{x};g::{x,y}", text)
                self.assertIsInstance(e, KlongNameError)
                self.assertEqual(str(e), "undefined: a")
                run(0, "f::{x};g::{x,y}", text)
                run(1, "f::{x};g::{x,y}", text)

    def test_position(self):
        e = error(2, "b::1;b+a")
        self.assertEqual(e.span, (7, 8))
        e = error(2, "b::1\na")                                         # a lone name reports its own position
        self.assertEqual(e.span, (5, 6))
        self.assertIn("2:1: name error: undefined: a", e.render())
        e = error(2, "f::{x+y}", "f(1;zz)")
        self.assertEqual(e.span, (4, 6))

    def test_did_you_mean(self):
        e = error(2, "lenght::3", "lenth")
        self.assertEqual(e.hint, "did you mean lenght?")
        self.assertIn("hint: did you mean lenght?", e.render())
        # the cutoff: a difflib ratio of at least 0.7 ("lenth" for "lenght" is 0.73)
        self.assertEqual(KlongInterpreter.DID_YOU_MEAN_CUTOFF, 0.7)
        self.assertEqual(error(2, "counter::0", "countr+1").hint, "did you mean counter?")   # 0.92
        self.assertEqual(error(2, "abcd::0", "abcx").hint, "did you mean abcd?")             # 0.75
        self.assertIsNone(error(2, "abc::0", "abd").hint)                                    # 0.67
        self.assertIsNone(error(2, "abcd::0", "abxy").hint)                                  # 0.5
        self.assertIsNone(error(2, "foo::1", "zzz").hint)
        self.assertIsNone(error(2, "zzz").hint)
        # system functions are candidates; a local of the running function too
        self.assertEqual(error(2, ".rnn()").hint, None)                  # a call: "undefined function", no hint rule
        self.assertEqual(error(2, "f::{[total];total::x;totl}", "f(1)").hint, "did you mean total?")

    def test_reserved_names_outside_a_function(self):
        for level in (0, 1):
            self.assertEqual(run(level, "x"), KGSym("x"))
        self.assertEqual(str(error(2, "x")), "undefined: x")
        self.assertEqual(str(error(2, "1+y")), "undefined: y")
        for level in LEVELS:
            self.assertEqual(run(level, "{x}(1)"), 1)

    def test_symbols_are_still_data(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(run(level, "[a b]").tolist(), [KGSym("a"), KGSym("b")])
                self.assertEqual(run(level, ":a"), KGSym("a"))
                self.assertEqual(run(level, "d:::{[a 1]};d?:a"), 1)
                self.assertEqual(run(level, "f::{x}", "f(:a)"), KGSym("a"))
                self.assertEqual(run(level, "f::{x}", "f'[a b]").tolist(), [KGSym("a"), KGSym("b")])
                self.assertEqual(run(level, "s::[a b];s@0"), KGSym("a"))
                self.assertEqual(run(level, 'x::.rs(":foo");x'), KGSym("foo"))

    def test_calling_an_undefined_function_is_an_error_at_every_level(self):
        for level in LEVELS:
            e = error(level, "nosuch(1)")
            self.assertIsInstance(e, KlongNameError)
            self.assertEqual(str(e), "undefined function: nosuch")

    def test_python_lookup_is_unchanged(self):
        for level in LEVELS:
            with self.assertRaises(KeyError):
                KlongInterpreter(strict=level)["nosuch"]


class TestQuotedSymbols(unittest.TestCase):
    def test_quoted_symbol_when_a_variable_exists(self):
        for level in (0, 1):
            self.assertEqual(run(level, "foo::5", ":foo"), 5)           # legacy: looked up
            self.assertEqual(run(level, "foo::5", "1,:foo").tolist(), [1, 5])
        self.assertEqual(run(2, "foo::5", ":foo"), KGSym("foo"))
        self.assertEqual(run(2, "foo::5", "1,:foo").tolist(), [1, KGSym("foo")])
        self.assertIs(type(run(2, "foo::5", ":foo")), KGSym)            # never a KGQSym

    def test_quoted_symbol_without_a_variable(self):
        for level in LEVELS:
            self.assertEqual(run(level, ":foo"), KGSym("foo"))
            self.assertEqual(run(level, "f::{1;:foo}", "f()"), KGSym("foo"))
        # level 0 defines the name as a side effect; above that nothing is defined
        self.assertEqual(run(0, ":foo", "foo"), KGSym("foo"))
        with self.assertRaises(KlongNameError):
            run(2, ":foo", "foo")

    def test_symbols_through_adverbs(self):
        for level in LEVELS:                                            # fixed at every level
            self.assertEqual(run(level, "foo::5", "{x}'[:foo :bar]").tolist(), [KGSym("foo"), KGSym("bar")])

    def test_the_name_still_reads_the_variable(self):
        for level in LEVELS:
            self.assertEqual(run(level, "foo::5", "foo"), 5)
        self.assertEqual(run(2, "foo::5", "d:::{};d,:foo,foo;d?:foo"), 5)    # key :foo, value 5

    def test_quoted_reserved_names(self):
        self.assertEqual(run(2, "{:x}(1)"), KGSym("x"))
        self.assertEqual(run(2, "{1;:x}(1)"), KGSym("x"))
        self.assertEqual(run(0, "{1;:x}(1)"), 1)
        self.assertEqual(run(1, "{1;:x}(1)"), 1)

    def test_a_function_whose_body_is_a_quoted_symbol(self):
        self.assertEqual(run(2, "f::{:foo}", "f()"), KGSym("foo"))
        self.assertEqual(run(2, "foo::5", "f::{:foo}", "f()"), KGSym("foo"))
        for level in (0, 1):                                            # legacy: the body is a name, looked up
            self.assertEqual(run(level, "foo::5", "f::{:foo}", "f()"), 5)

    def test_a_quoted_name_in_call_position_is_a_call(self):
        for level in LEVELS:
            self.assertEqual(run(level, "foo::{x+1}", ":foo(1)"), 2)

    def test_assignment_through_a_quoted_name(self):
        for level in LEVELS:
            self.assertEqual(run(level, ":a::5", "a"), 5)


class TestLocalReadBeforeAssignment(unittest.TestCase):
    def test_read(self):
        for level in (0, 1):
            self.assertEqual(run(level, "{[a];a}()"), KGSym("a"))       # legacy: its own symbol
            self.assertEqual(run(level, "{[a b];b::1;a}()"), KGSym("a"))
        for text in ("{[a];a}()", "{[a b];b::1;a}()", "{[a];a+1}()", "{[a];a::a+1}()", "{[a];{x+a}'[1 2]}()"):
            with self.subTest(text=text):
                e = error(2, text)
                self.assertIsInstance(e, KlongNameError)
                self.assertEqual(str(e), "local 'a' read before assignment")

    def test_assigned_locals_work(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(run(level, "{[a];a::1;a}()"), 1)
                self.assertEqual(run(level, "{[a b];a::x;b::a+1;a+b}(1)"), 3)
                self.assertEqual(run(level, "{[a];a::x;{x+a}'[1 2]}(10)").tolist(), [11, 12])
                self.assertEqual(run(level, "f::{[a];a::x;:[x<2;1;a*.f(x-1)]}", "f(5)"), 120)
                self.assertEqual(run(level, "a::7", "f::{[a];a::x;a}", "f(1);a"), 7)
                self.assertEqual(run(level, "{[x];x}(3)"), 3)          # a local named like a parameter
                self.assertEqual(run(level, "{[a];:[x;a::1;a::2];a}(0)"), 2)

    def test_the_sentinel_is_never_a_value(self):
        k = KlongInterpreter(strict=2)
        seen = []
        k["probe"] = lambda: seen.append(sorted(str(n) for n, _ in k._context)) or 0
        k["vals"] = lambda: seen.append([v for _, v in k._context]) or 0
        k("{[a b];b::2;probe();vals();b}()")
        names, values = seen
        self.assertNotIn("a", names)                                    # an unassigned local is not listed
        self.assertIn("b", names)
        self.assertTrue(all(type(v).__name__ != "_Unset" for v in values))
        # a dictionary or a list built from a local cannot hold it: the read fails first
        for text in ("{[a];:{},:k,a}()", "{[a];[;a;1]}()", "{[a];.p(a)}()", "{[a];,a}()"):
            with self.subTest(text=text):
                self.assertEqual(str(error(2, text)), "local 'a' read before assignment")

    def test_level_change_inside_a_call_does_not_leak(self):
        # the frame of a call made at level 2 keeps its rule after .unsafe(); nothing escapes
        k = KlongInterpreter(strict=2)
        with self.assertRaises(KlongNameError):
            k("{[a];.unsafe();a}()")
        self.assertEqual(k.strict, 0)
        self.assertEqual(k("{[a];a}()"), KGSym("a"))


class TestTooManyArguments(unittest.TestCase):
    def test_extras_are_dropped_below_level_2(self):
        for level in (0, 1):
            self.assertEqual(run(level, "{x+y}(1;2;3)"), 3)
            self.assertEqual(run(level, "{x}(1;2)"), 1)
            self.assertEqual(run(level, "f::{1}", "f(9)"), 1)
            self.assertEqual(run(level, "{x+1}/[1 2 3]"), 3)             # y is dropped at each step

    def test_written_calls(self):
        e = error(2, "{x+y}(1;2;3)")
        self.assertIsInstance(e, KlongArityError)
        self.assertEqual((str(e), e.kind), ("the function takes 2 arguments, got 3", "arity"))
        self.assertEqual(str(error(2, "{x}(1;2)")), "the function takes 1 argument, got 2")
        self.assertEqual(str(error(2, "f::{x+y}", "f(1;2;3)")), "f takes 2 arguments, got 3")
        self.assertEqual(str(error(2, "f::{1}", "f(9)")), "f takes no arguments, got 1")
        self.assertEqual(str(error(2, "f::{-x}", "f(1;2)")), "f takes 1 argument, got 2")
        e = error(2, "f::{x}", "1+f(1;2)")
        self.assertEqual(e.span, (2, 8))

    def test_takes_is_the_highest_argument_used(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(run(level, "{x+z}(1;2;3)"), 4)
                self.assertEqual(run(level, "f::{y}", "f(1;2)"), 2)
                self.assertEqual(run(level, "{x+y}(1;2)"), 3)
                self.assertEqual(run(level, "f::{1}", "f()"), 1)

    def test_fewer_arguments_is_a_projection_at_every_level(self):
        for level in LEVELS:
            self.assertIsInstance(run(level, "f::{x+y}", "f(1)"), KGCall)
            self.assertEqual(run(level, "f::{x+y}", "g::f(1;)", "g(2)"), 3)
            self.assertEqual(run(level, "f::{x+y+z}", "g::f(1;;3)", "g(2)"), 6)

    def test_projections(self):
        for level in (0, 1):
            self.assertEqual(run(level, "f::{x+y}", "g::f(1;)", "g(2;3)"), 3)
        e = error(2, "f::{x+y}", "g::f(1;)", "g(2;3)")
        self.assertIsInstance(e, KlongArityError)
        self.assertEqual(str(e), "g takes 1 argument, got 2")

    def test_adverbs(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(run(level, "{x+1}'[1 2]").tolist(), [2, 3])
                self.assertEqual(run(level, "{x+y}/[1 2 3]"), 6)
                self.assertEqual(run(level, "[1 2]{x+y}'[3 4]").tolist(), [4, 6])
                self.assertEqual(run(level, "{x<5}{x+1}:~0"), 5)
                self.assertEqual(run(level, "3{x+1}:*0"), 3)
                self.assertEqual(run(level, "+/[1 2 3]"), 6)
        self.assertEqual(str(error(2, "{x+1}/[1 2 3]")), "the function takes 1 argument, got 2")
        self.assertEqual(str(error(2, "f::{1}", "f'[1 2]")), "f takes no arguments, got 1")
        self.assertEqual(run(0, "f::{1}", "f'[1 2]").tolist(), [1, 1])

    def test_apply_with_at(self):
        for level in (0, 1):
            self.assertEqual(run(level, "f::{x}", "f@[1 2]"), 1)
        self.assertIsInstance(error(2, "f::{x}", "f@[1 2]"), KlongArityError)
        for level in LEVELS:
            self.assertEqual(run(level, "f::{x+y}", "f@[1 2]"), 3)

    def test_system_functions(self):
        for level in (0, 1):
            self.assertEqual(run(level, '.rs("1";2)'), 1)
        e = error(2, '.rs("1";2)')
        self.assertIsInstance(e, KlongArityError)
        self.assertEqual(str(e), ".rs takes 1 argument, got 2")

    def test_wildcard_functions_are_exempt(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = KlongInterpreter(strict=level)
                self.assertEqual(k(".strict(%d)" % level), level)
                from klongpy.core import KGLambda
                k["anyargs"] = KGLambda(lambda *args: len(args), args=None, wildcard=True)
                self.assertEqual(k("anyargs()"), 0)
                self.assertEqual(k("anyargs(1;2;3)"), 3)

    def test_python_callers(self):
        k = KlongInterpreter(strict=2)
        k("f::{x+y}")
        self.assertEqual(k["f"](1, 2), 3)
        with self.assertRaises(KlongArityError):
            k["f"](1, 2, 3)                                              # KGFnWrapper checks at every level
        with self.assertRaises(KlongArityError):
            k.apply(k._context[KGSym("f")], [1, 2, 3])
        self.assertEqual(KlongInterpreter().apply(KlongInterpreter()("{x+y}"), [1, 2, 3]), 3)


class TestLevelTwoIncludesLevelOne(unittest.TestCase):
    def test_syntax_checks(self):
        from klongpy.errors import KlongSyntaxError
        for text in ("1 2 3+4 5", '"abc', "1+", "1::5", "f (1)"):
            with self.subTest(text=text):
                with self.assertRaises(KlongSyntaxError):
                    run(2, text)

    def test_assignment_inside_a_function(self):
        self.assertEqual(str(error(2, "f::{c::1}", "f()")), "undefined variable: c")


@unittest.skipIf(importlib.util.find_spec("colorama") is None, "colorama not installed")
class TestCommandLine(unittest.TestCase):
    def kgpy(self, *args):
        env = dict(os.environ, KGPY_HISTORY="", PYTHONPATH=ROOT, NO_COLOR="1")
        return subprocess.run([sys.executable, "-W", "ignore", "-m", "klongpy.cli", *args], input="",
                              capture_output=True, text=True, env=env, timeout=120, cwd=ROOT)

    def test_undefined_name(self):
        r = self.kgpy("--strict=2", "-e", "a+1")
        self.assertEqual(r.returncode, 1)
        self.assertIn("1:1: name error: undefined: a", r.stderr)
        r = self.kgpy("-e", "a+1")                                      # without the flag it fails as before
        self.assertEqual(r.returncode, 1)
        self.assertIn("type error: cannot apply + (Plus) to symbol and integer", r.stderr)
        r = self.kgpy("--strict", "-e", "a+1")
        self.assertIn("type error", r.stderr)

    def test_quoted_symbol_and_arity(self):
        self.assertEqual(self.kgpy("--strict=2", "-e", "foo::5;:foo").stdout.strip(), ":foo")
        self.assertEqual(self.kgpy("-e", "foo::5;:foo").stdout.strip(), "5")
        r = self.kgpy("--strict=2", "-e", "{x}(1;2)")
        self.assertEqual(r.returncode, 1)
        self.assertIn("arity error: the function takes 1 argument, got 2", r.stderr)


if __name__ == "__main__":
    unittest.main()
