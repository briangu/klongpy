"""
Ticket 033: interpreter and tooling bugs found while writing the documentation.

Row 1   a call with fewer arguments than the function takes is a projection
        over the missing trailing arguments (it was a value that could never
        be completed); projections display by the arguments they still take
Row 2   x, y and z are assignable inside a function at every strict level
Row 3   modules: the reference behaviour, pinned (see docs/operators.md)
Row 4   .l("math.kg") in a session made by create_repl
Row 5   a variable holding :undefined matches :undefined
Row 6   lint W019 does not fire on a top-level assignment to x
Row 7   kgpy --lint takes its options in any order
Row 8   .pyf functions with optional positional parameters get every argument
Row 9   a q lambda received over IPC prints as its text
"""
import contextlib
import io
import os
import subprocess
import sys
import unittest

import numpy as np

from klongpy import KlongInterpreter
from klongpy.core import KGCall, KGFn, KGSym
from klongpy.errors import KlongArityError, KlongError
from klongpy.writer import kg_write
from backend_compat import requires_object_dtype, requires_strings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVELS = (0, 1, 2)


def tolist(v):
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def show(k, v):
    return kg_write(v, k._backend, display=False)


class Levels(unittest.TestCase):
    """check(source, expected) runs source in a fresh interpreter at each strict level."""

    def check(self, src, expected, levels=LEVELS):
        for level in levels:
            with self.subTest(level=level, src=src):
                k = KlongInterpreter(strict=level)
                self.assertEqual(tolist(k(src)), expected)

    def shows(self, src, expected, levels=LEVELS):
        for level in levels:
            with self.subTest(level=level, src=src):
                k = KlongInterpreter(strict=level)
                self.assertEqual(show(k, k(src)), expected)


# ---------------------------------------------------------------------------
# Row 1

class TestShortCallIsAProjection(Levels):
    def test_the_ticket_rows(self):
        self.check("r::{x+y};s::r(1);s(10)", 11)
        self.check("f(a;b)::{a+b};g::f(1);g(2)", 3)

    def test_same_as_the_empty_slot(self):
        self.check("r::{x-y};s::r(10);t::r(10;);s(3),t(3)", [7, 7])

    def test_y_only_function(self):
        self.check("f::{y};p::f(5);p(6)", 6)

    def test_literal_called_short(self):
        self.check("p::{x+y}(1);p(2)", 3)

    def test_triad_one_argument(self):
        self.check("g::{x,y,z};h::g(1);h(2;3)", [1, 2, 3])

    def test_triad_two_arguments(self):
        self.check("g::{x,y,z};h::g(1;2);h(3)", [1, 2, 3])

    def test_triad_step_by_step(self):
        self.check("g::{x,y,z};h::g(1);i::h(2);i(3)", [1, 2, 3])

    def test_empty_slot_and_missing_trailing_argument(self):
        # g(1;) of a triad leaves y and z open; g(;2) leaves x and z open
        self.check("g::{x,y,z};h::g(1;);h(2;3)", [1, 2, 3])
        self.check("g::{x,y,z};h::g(;2);h(1;3)", [1, 2, 3])

    def test_no_arguments_at_all(self):
        self.check("g::{x,y};h::g();h(1;2)", [1, 2])

    def test_projection_of_a_projection_fills_the_first_open_slot(self):
        self.check("f::{x,y,z};p::f(1);q::p(;3);q(2)", [1, 2, 3])
        self.check("f::{x,y,z};p::f(;;3);q::p(1);q(2)", [1, 2, 3])

    def test_more_projections_than_arguments(self):
        self.check("f::{x,y,z};a::f();b::a(1);c::b(2);d::c();d(3)", [1, 2, 3])

    def test_a_projection_called_short_does_not_run_the_function(self):
        # it used to run g with the open slot unbound
        self.shows("g::{x,y,z};h::g(1;;);h(2;)", ":monad")
        self.shows("g::{x,y,z};h::g(1;;);h(2)", ":monad")

    def test_the_function_is_looked_up_when_completed(self):
        self.check("r::{x+y};p::r(1);r::{x-y};p(10)", -9)

    def test_forward_reference_at_top_level(self):
        self.check("g::f(1;);f::{x+y};g(2)", 3)

    def test_top_level_arguments_are_expressions(self):
        # legacy, kept: evaluated when the projection is completed
        self.check("r::{x+y};a::1;p::r(a;);a::5;p(10)", 15)
        self.check("r::{x+y};a::1;p::r(a);a::5;p(10)", 15)

    def test_complete_calls_are_unchanged(self):
        self.check("r::{x+y};r(1;2)", 3)
        self.check("{x}(7)", 7)
        self.check("f::{42};f()", 42)
        self.check("f::{x+y};1 f 2", 3)

    def test_function_values_are_not_calls(self):
        for level in LEVELS:
            k = KlongInterpreter(strict=level)
            v = k("{x+y}")
            self.assertIsInstance(v, KGFn)
            self.assertIsNone(v.args)
            self.assertEqual(show(k, v), ":dyad")


class TestProjectionMadeInsideAFunction(Levels):
    """It holds the values of its arguments: the frame is gone when it is completed."""

    def test_argument_of_the_enclosing_function(self):
        self.check("f::{x+y};mk::{f(x)};p::mk(5);p(10)", 15)
        self.check("f::{x+y};mk::{[a];a::x;f(a;)};p::mk(5);p(10)", 15)

    def test_a_body_that_is_only_a_projection_passes_the_call_on(self):
        # ticket 026, unchanged: mk(5) is f(1;5), not a function that makes projections
        self.check("f::{x+y};mk::{f(1;)};mk(5)", 6)

    def test_local_of_the_enclosing_function(self):
        self.check("f::{x+y};mk::{[a];a::x*2;f(a)};p::mk(5);p(10)", 20)

    def test_each_makes_one_projection_per_element(self):
        self.check("f::{x+y};{x(10)}'{f(x)}'[1 2 3]", [11, 12, 13])
        self.check("f::{x-y};{x(10)}'{[a];a::x;f(;a)}'[1 2 3]", [9, 8, 7])

    def test_callee_held_in_an_argument(self):
        self.check("f::{x+y};p::{x(1)}(f);p(5)", 6)

    def test_symbol_argument_stays_a_symbol(self):
        for level in LEVELS:
            k = KlongInterpreter(strict=level)
            k("foo::5;f::{x,y};mk::{f(x)}")
            r = k("p::mk(*[:foo]);p(1)")
            self.assertEqual(r[0], KGSym("foo"))

    def test_named_parameters(self):
        self.check("f(a;b)::{a-b};mk(c)::{f(c)};p::mk(10);p(3)", 7)

    def test_recursion_through_dot_f(self):
        self.check("fact::{:[y<2;x;.f(x*y;y-1)]};p::fact(1);p(5)", 120)


class TestShortCallNamedParameters(Levels):
    def test_trailing(self):
        self.check("f(a;b;c;d;e)::{a,b,c,d,e};p::f(1;2);p(3;4;5)", [1, 2, 3, 4, 5])

    def test_levels(self):
        self.check("f(a;b;c;d;e)::{a,b,c,d,e};p::f(1;2);q::p(3);r::q(4);r(5)", [1, 2, 3, 4, 5])

    def test_slot_then_trailing(self):
        self.check("f(a;b;c;d;e)::{a,b,c,d,e};p::f(1;;3);q::p(2);q(4;5)", [1, 2, 3, 4, 5])

    def test_middle_slot_of_a_projection(self):
        self.check("f(a;b;c;d)::{a,b,c,d};p::f(1;;;);q::p(2;;4);q(3)", [1, 2, 3, 4])

    def test_too_many_is_still_an_error_at_every_level(self):
        for level in LEVELS:
            k = KlongInterpreter(strict=level)
            with self.assertRaises(KlongArityError):
                k("f(a;b)::{a+b};f(1;2;3)")
            with self.assertRaises(KlongArityError):
                k("f(a;b)::{a+b};p::f(1);p(2;3)")


class TestShortCallAndStrictLevel(Levels):
    def test_short_call_is_allowed_at_level_2(self):
        k = KlongInterpreter(strict=2)
        self.assertEqual(k("r::{x+y};s::r(1);s(10)"), 11)

    def test_too_many_arguments(self):
        for level, src in ((0, None), (1, None), (2, KlongArityError)):
            k = KlongInterpreter(strict=level)
            if src is None:
                self.assertEqual(k("f::{x+y};f(1;2;3)"), 3)
                self.assertEqual(k("f::{x+y};g::f(1);g(2;3)"), 3)
            else:
                with self.assertRaises(KlongArityError) as c:
                    k("f::{x+y};f(1;2;3)")
                self.assertEqual(str(c.exception), "f takes 2 arguments, got 3")
                with self.assertRaises(KlongArityError) as c:
                    k("f::{x+y};g::f(1);g(2;3)")
                self.assertEqual(str(c.exception), "g takes 1 argument, got 2")


class TestProjectionDisplay(Levels):
    def test_by_remaining_arity(self):
        self.shows("f::{y};f(5)", ":monad")
        self.shows("g::{x+y};g(1;)", ":monad")
        self.shows("g::{x+y};g(;1)", ":monad")
        self.shows("g::{x+y};p::g(1;);p", ":monad")
        self.shows("g::{x+y+z};g(1)", ":dyad")
        self.shows("g::{x+y+z};g(1;)", ":dyad")
        self.shows("g::{x+y+z};g(1;;)", ":dyad")
        self.shows("g::{x+y+z};g(1;2)", ":monad")
        self.shows("g::{x+y+z};g()", ":triad")
        self.shows("g::{x+y+z};p::g(1);p(2)", ":monad")

    def test_functions_are_unchanged(self):
        self.shows("{x+y}", ":dyad")
        self.shows("g::{x+y+z};g", ":triad")
        self.shows("f(a;b)::{a+b};f", ":dyad")

    def test_named(self):
        self.shows("f(a;b;c;d)::{a+b+c+d};f(1;2)", ":dyad")
        self.shows("f(a;b;c;d)::{a+b+c+d};f(1;2;3)", ":monad")

    @requires_strings
    def test_format_and_print(self):
        for level in LEVELS:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                k = KlongInterpreter(strict=level)
                self.assertEqual(k("g::{x+y};$g(1)"), ":monad")
                k(".p(g(1))")
            self.assertEqual(out.getvalue().strip(), ":monad")


class TestShortCallUnderAdverbsAndAt(Levels):
    def test_projection_under_each(self):
        self.check("r::{x+y};s::r(1);s'[1 2 3]", [2, 3, 4])

    def test_projection_under_over_and_scan(self):
        self.check("g::{x+y+z};p::g(100);p/[1 2 3]", 206)
        self.check("g::{x+y+z};p::g(100);p\\[1 2 3]", [1, 103, 206])

    def test_projection_under_each2(self):
        self.check("g::{x+y+z};p::g(100);[1 2]p'[3 4]", [104, 106])

    def test_each_of_a_dyad_gives_projections(self):
        self.check("f::{x+y};{x(10)}'f'[1 2 3]", [11, 12, 13])
        self.shows("f::{x+y};f'[1 2]", "[:monad :monad]")

    def test_at_with_one_value(self):
        self.check("f::{x-y};p::f@10;p(3)", 7)
        self.shows("f::{x-y};f@10", ":monad")

    def test_at_with_a_list_spreads(self):
        self.check("f::{x-y};f@[10 3]", 7)

    def test_at_on_a_projection(self):
        self.check("f::{x-y};p::f(10);p@3", 7)
        self.check("g::{x,y,z};p::g(1);q::p@2;q@3", [1, 2, 3])

    def test_values_are_not_evaluated_again(self):
        for level in LEVELS:
            k = KlongInterpreter(strict=level)
            k("foo::5;f::{x,y}")
            r = k("p::f@*[:foo];p(1)")
            self.assertEqual(r[0], KGSym("foo"))

    def test_named_parameters_under_each_stay_an_arity_error(self):
        # ticket 061: values cannot express a slot for a named-parameter function
        for level in LEVELS:
            k = KlongInterpreter(strict=level)
            with self.assertRaises(KlongArityError):
                k("f(a;b)::{a+b};f'[1 2]")


class TestShortCallFromPython(unittest.TestCase):
    def test_wrapper_with_too_few_is_an_arity_error(self):
        for level in LEVELS:
            k = KlongInterpreter(strict=level)
            k("r::{x+y}")
            with self.assertRaises(KlongArityError):
                k["r"](1)
            self.assertEqual(k["r"](1, 2), 3)

    def test_wrapper_of_a_projection_takes_the_open_slots(self):
        for level in LEVELS:
            k = KlongInterpreter(strict=level)
            k("r::{x-y};s::r(10);t::r(;10);g::{x,y,z};u::g(1)")
            self.assertEqual(k["s"](3), 7)
            self.assertEqual(k["t"](3), -7)
            self.assertEqual(tolist(k["u"](2, 3)), [1, 2, 3])
            with self.assertRaises(KlongArityError):
                k["s"]()
            with self.assertRaises(KlongArityError):
                k["s"](1, 2)

    def test_apply_with_too_few_values(self):
        for level in LEVELS:
            k = KlongInterpreter(strict=level)
            k("r::{x-y}")
            p = k.apply(KGSym("r"), [10])
            self.assertIsInstance(p, KGCall)
            self.assertEqual(str(p), ":monad")
            self.assertEqual(k.apply(p, [3]), 7)
            k["p"] = p
            self.assertEqual(k("p(4)"), 6)

    def test_apply_a_value_that_is_a_symbol(self):
        k = KlongInterpreter()
        k("foo::5;r::{x,y}")
        p = k.apply(KGSym("r"), [KGSym("foo")])
        self.assertEqual(k.apply(p, [1])[0], KGSym("foo"))

    def test_named_wrapper_too_few(self):
        k = KlongInterpreter()
        k("f(a;b)::{a+b}")
        with self.assertRaises(KlongArityError):
            k["f"](1)


class TestShortCallOverIPC(unittest.TestCase):
    """A remote call with too few arguments returns a projection (the server does not fail)."""

    def test_request_with_too_few_arguments(self):
        from klongpy.ipc.endpoint import evaluate_request
        for level in LEVELS:
            k = KlongInterpreter(strict=level)
            k("r::{x-y}")
            p = evaluate_request(k, [KGSym("r"), 10])
            self.assertEqual(str(p), ":monad")
            self.assertEqual(k.apply(p, [3]), 7)
            self.assertEqual(evaluate_request(k, [KGSym("r"), 10, 3]), 7)

    def test_a_projection_goes_out_by_its_remaining_arity(self):
        from klongpy.ipc import KdbLambda, decode, encode
        k = KlongInterpreter()
        p = k("g::{x+y+z};g(1;2)")
        self.assertEqual(decode(encode(p)), KdbLambda("kg", ":monad"))
        self.assertEqual(decode(encode(k("g(1)"))), KdbLambda("kg", ":dyad"))
        self.assertEqual(decode(encode(k("g"))).context, "kg")


# ---------------------------------------------------------------------------
# Row 2

class TestParametersAreAssignable(Levels):
    def test_x(self):
        self.check("h::{x::7};h(1)", 7)
        self.check("h::{x::7;x+1};h(1)", 8)

    def test_y_and_z(self):
        self.check("h::{y::x+1;z::y+1;x,y,z};h(1;0;0)", [1, 2, 3])

    def test_the_callers_argument_is_untouched(self):
        self.check("h::{x::7;x};g::{h(x);x};g(1)", 1)

    def test_no_global_is_made(self):
        for level in (0, 1):
            k = KlongInterpreter(strict=level)
            k("h::{x::7};h(1)")
            self.assertEqual(k("x"), KGSym("x"))

    def test_named_parameters(self):
        self.check("f(a;b)::{a::a+1;b::b*2;a,b};f(1;2)", [2, 4])

    def test_undeclared_name_is_still_an_error(self):
        from klongpy.errors import KlongNameError
        for level in (1, 2):
            k = KlongInterpreter(strict=level)
            with self.assertRaises(KlongNameError):
                k("h::{c::7};h()")


# ---------------------------------------------------------------------------
# Row 3: the reference behaviour (tests/kgtests/language/test_suite.kg, "Modules")

class TestModules(Levels):
    def test_a_module_definition_is_reachable_by_its_plain_name(self):
        self.check(".module(:t);helper::{x+1};.module(0);helper(4)", 5)

    def test_a_module_assignment_shadows_the_global_of_that_name(self):
        self.check("v::1;.module(:t);v::5;.module(0);v", 5)

    def test_functions_of_the_module_keep_the_module_variable(self):
        self.check(".module(:t);v::5;g::{v};.module(0);v::9;v,g()", [9, 5])

    def test_redefinition_outside_does_not_reach_into_the_module(self):
        self.check(".module(:t);g::{1};f::{g()};.module(0);g::{2};g(),f()", [2, 1])

    def test_two_modules_with_the_same_name_inside(self):
        self.check(".module(:t);g::{1};f::{g()};.module(0);.module(:u);g::{2};h::{g()};.module(0);f(),h()", [1, 2])


# ---------------------------------------------------------------------------
# Row 4

class TestCreateReplLoadsTheLibrary(unittest.TestCase):
    def test_math(self):
        from klongpy.repl import cleanup_repl, create_repl
        for level in LEVELS:
            with self.subTest(level=level):
                klong, loops = create_repl(strict=level)
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        klong('.l("math.kg")')
                    self.assertAlmostEqual(float(klong("sqr(16)")), 4.0, places=3)
                finally:
                    cleanup_repl(loops)


# ---------------------------------------------------------------------------
# Row 5

class TestUndefinedMatches(Levels):
    def test_variable_holding_undefined(self):
        self.check("u::1%0;u~:undefined", 1)

    def test_direct(self):
        self.check("(1%0)~:undefined", 1)
        self.check(":undefined~1%0", 1)
        self.check(":undefined~:undefined", 1)

    @requires_object_dtype
    def test_missing_dictionary_key(self):
        self.check("d:::{[1 2]};u::d?9;u~:undefined", 1)

    def test_other_values_do_not_match(self):
        self.check("u::1%0;u~0", 0)
        self.check("u::5;u~:undefined", 0)
        self.check(":foo~:undefined", 0)


# ---------------------------------------------------------------------------
# Rows 6 and 7

class TestLint(unittest.TestCase):
    def test_w019_not_at_top_level(self):
        from klongpy.lint import lint_source
        self.assertEqual([d.code for d in lint_source("x::1\n")], [])
        self.assertEqual([d.code for d in lint_source("y::1;z::2\n")], [])

    def test_w019_still_inside_a_function(self):
        from klongpy.lint import lint_source
        self.assertEqual([d.code for d in lint_source("f::{x::5;x}\n")], ["W019"])

    def kgpy(self, *argv, stdin=None):
        return subprocess.run([sys.executable, "-W", "ignore", "-m", "klongpy.cli", *argv], input=stdin,
                              capture_output=True, text=True, timeout=120, cwd=ROOT)

    def test_stdin_top_level_assignment(self):
        r = self.kgpy("--lint", "-", stdin="x::1\n")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("W019", r.stdout)

    def test_options_in_any_order(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.kg")
            with open(p, "w") as f:
                f.write("f::{x::5;x}\n")
            outs = []
            for argv in (["--lint", p, "-v"], ["--lint", "-v", p], ["-v", "--lint", p],
                         ["--lint", "--select", "W019", "-v", p], ["--select=W019", "--lint", "-v", p]):
                r = self.kgpy(*argv)
                self.assertEqual(r.returncode, 0, (argv, r.stderr))
                self.assertIn("W019", r.stdout, argv)
                self.assertIn("hint:", r.stdout, argv)
                outs.append(r.stdout)
            self.assertEqual(len(set(outs)), 1)

    def test_several_files_after_an_option(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a.kg"), os.path.join(d, "b.kg")
            for p in (a, b):
                with open(p, "w") as f:
                    f.write("f::{x::5;x}\n")
            r = self.kgpy("--lint", a, "-v", b)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.count("W019"), 2)
            self.assertIn("in 2 files", r.stdout)

    def test_lint_without_a_file_is_a_usage_error(self):
        r = self.kgpy("--lint")
        self.assertEqual(r.returncode, 2)


# ---------------------------------------------------------------------------
# Row 8

class TestPyfPassesEveryArgument(unittest.TestCase):
    def test_linspace_and_arange(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = KlongInterpreter(strict=level)
                k('.pyf("numpy";"linspace")')
                k('.pyf("numpy";"arange")')
                self.assertEqual(len(k("linspace(0;1;5)")), 5)
                self.assertEqual(len(k("linspace(0;1)")), 50)
                self.assertEqual(tolist(k("arange(2;8)")), [2, 3, 4, 5, 6, 7])
                self.assertEqual(tolist(k("arange(3)")), [0, 1, 2])
                self.assertEqual(tolist(k("arange(2;8;2)")), [2, 4, 6])

    def test_python_function_with_defaults(self):
        import types
        m = types.ModuleType("t033mod")
        exec("def scale(a, b=10, c=0):\n    return a * b + c\n"
             "def two(a, b):\n    return a - b\n", m.__dict__)
        sys.modules["t033mod"] = m
        try:
            for level in LEVELS:
                with self.subTest(level=level):
                    k = KlongInterpreter(strict=level)
                    k('.pyf("t033mod";"scale")')
                    k('.pyf("t033mod";"two")')
                    self.assertEqual(k("scale(2)"), 20)
                    self.assertEqual(k("scale(2;3)"), 6)
                    self.assertEqual(k("scale(2;3;1)"), 7)
                    self.assertEqual(tolist(k("scale'[1 2]")), [10, 20])
                    self.assertEqual(k("two(5;3)"), 2)          # no optional parameter: as before
        finally:
            del sys.modules["t033mod"]

    def test_py_import_of_a_module(self):
        import types
        m = types.ModuleType("t033mod2")
        exec("def scale(a, b=10):\n    return a * b\n", m.__dict__)
        sys.modules["t033mod2"] = m
        try:
            k = KlongInterpreter()
            k('.py("t033mod2")')
            self.assertEqual(k("scale(2;3)"), 6)
            self.assertEqual(k("scale(2)"), 20)
        finally:
            del sys.modules["t033mod2"]


# ---------------------------------------------------------------------------
# Row 9

class TestQLambdaDisplay(unittest.TestCase):
    def test_prints_as_its_text(self):
        from klongpy.ipc import KdbLambda
        k = KlongInterpreter()
        lam = KdbLambda("", "{x*2}")
        self.assertEqual(show(k, lam), "{x*2}")
        self.assertEqual(kg_write(lam, k._backend, display=True), "{x*2}")
        self.assertEqual(str(lam), "{x*2}")

    def test_repr_and_equality_are_unchanged(self):
        from klongpy.ipc import KdbLambda, decode, encode
        lam = KdbLambda("", "{x*2}")
        self.assertIn("KdbLambda", repr(lam))
        self.assertEqual(decode(encode(lam)), lam)

    def test_received_value(self):
        from klongpy.ipc.endpoint import from_wire
        from klongpy.ipc import KdbLambda
        k = KlongInterpreter()
        v = from_wire(k, KdbLambda("", "{x*2}"))
        self.assertEqual(show(k, v), "{x*2}")


if __name__ == "__main__":
    unittest.main()


class TestMonadOnConditional(unittest.TestCase):
    """A monad applied to a conditional or an evaluated list takes the whole result as its operand."""

    def test_monad_on_conditional_and_expression_list(self):
        from klongpy import KlongInterpreter
        k = KlongInterpreter()
        self.assertEqual(k('-:[0;2;3]'), -3)
        self.assertEqual(k('-:[1;2;3]'), -2)
        self.assertEqual(k('#[;1;2;3]'), 3)
        self.assertEqual(k('*[;4;5]'), 4)

    def test_inside_a_function_and_on_the_compiled_path(self):
        from klongpy import KlongInterpreter
        k = KlongInterpreter()
        self.assertEqual(k('n::{-:[x;2;3]};n(0)'), -3)
        self.assertEqual(k('c::{#[;x;x;x]};c(7)'), 3)
        self.assertEqual(k('a::2;-:[a=2;10;20]'), -10)


class TestConditionOnDictionaryValue(unittest.TestCase):
    """A zero stored in a dictionary is false in a conditional on every backend."""

    def test_zero_from_a_dictionary_is_false(self):
        from klongpy import KlongInterpreter
        k = KlongInterpreter()
        k('d:::{};d,"w",,0;d,"v",,5')
        self.assertEqual(k(':[d?"w";1;2]'), 2)
        self.assertEqual(k(':[d?"v";1;2]'), 1)
