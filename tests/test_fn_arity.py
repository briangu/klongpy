"""
The arity of a function literal is the highest of x, y, z its body
mentions (max_arg), wherever the mention is: under a monad, inside a call
argument, after ",,". Before, arity was a count of the names the old scan
happened to see, so {-x} was a nilad and {y} a monad.
"""
import unittest

import numpy as np

from klongpy import KlongInterpreter
from klongpy.core import KGCall, KGFn, KGSym, KlongArityError
from klongpy.sys_fn_ipc import TcpServerConnectionHandler
from klongpy.types import KGFnWrapper


def fn(text):
    """The function value that the literal text is (the names it calls, f and g, are defined: the engine resolves them)."""
    k = KlongInterpreter()
    k("f::{x};g::{x+y+z}")
    k("h::" + text)
    return k["h"].fn


def run(*lines):
    k = KlongInterpreter()
    r = None
    for line in lines:
        r = k(line)
    return r


class TestTheTable(unittest.TestCase):
    def test_display(self):
        for text, shown in [("{-x}", ":monad"), ("{#x}", ":monad"), ("{y}", ":dyad"), ("{x,-y}", ":dyad"),
                            ("{g(x+1)}", ":monad"), ("{.f(x-1)}", ":monad"), ("{seen::seen,,x}", ":monad"),
                            ("{err::y}", ":dyad"), ("{z}", ":triad"), ("{x+z}", ":triad"), ("{1}", ":nilad"),
                            ("{x}", ":monad"), ("{x+y}", ":dyad"), ("{a::#x;a}", ":monad")]:
            with self.subTest(text=text):
                self.assertEqual(str(run(f"f::{text}", "f")), shown)

    def test_arity_is_max_arg_on_every_literal(self):
        for text, arity in [("{-x}", 1), ("{y}", 2), ("{x,-y}", 2), ("{z}", 3), ("{1}", 0), ("{g(x;z)}", 3),
                            ("{g(x;)}", 1), ("{{y}(1)}", 2), ("{{x+y}'x}", 1), ("{a::{x+y};a(x;1)}", 1),
                            ("{[a];a::-y;a}", 2), ("{[1 2 3]}", 0), ("{:{[1 2]}}", 0), ("{+/x}", 1),
                            ("{x{-y}'x}", 1), ("{x'y}", 2), ("{{+/x*y}(z;)'x}", 3), ("{:[#x;y;0]}", 2)]:
            with self.subTest(text=text):
                f = fn(text)
                self.assertEqual(f.arity, arity)

    def test_python_call(self):
        k = KlongInterpreter()
        k("f::{-x}")
        self.assertIsInstance(k["f"], KGFnWrapper)
        self.assertEqual(k["f"](5), -5)
        k("g::{#x}")
        self.assertEqual(k["g"]([1, 2, 3]), 3)
        k("h::{y}")
        self.assertEqual(k["h"](1, 2), 2)
        with self.assertRaises(KlongArityError):
            k["h"](1)
        with self.assertRaises(KlongArityError):
            k["f"]()

    def test_y_only_is_a_projection_when_given_one_argument(self):
        r = run("h::{y}", "h(1)")
        self.assertIsInstance(r, KGCall)                                # as {x+y}(1)
        self.assertIsInstance(run("{y}(1)"), KGFn)
        self.assertEqual(run("h::{y}", "h(1;2)"), 2)
        self.assertEqual(run("h::{y}", "p::h(1;)", "p(7)"), 7)
        self.assertEqual(run("{y}(1;2)"), 2)
        self.assertEqual(run("f::{x+z}", "f(1;2;3)"), 4)
        self.assertIsInstance(run("f::{x+z}", "f(1;2)"), KGCall)

    def test_written_calls_are_unchanged(self):
        self.assertEqual(run("f::{-x}", "f(5)"), -5)
        self.assertEqual(run("{-x}(5)"), -5)
        self.assertEqual(run("g::{x*2}", "f::{g(x+1)}", "f(1)"), 4)
        self.assertEqual(run("f::{:[x<2;1;x*.f(x-1)]}", "f(5)"), 120)
        self.assertEqual(run("f::{x,-y}", "f(1;2)").tolist(), [1, -2])
        self.assertEqual(run("f::{{x}(1)}", "f(5)"), 1)

    def test_a_nested_literal_keeps_its_arguments_to_itself(self):
        self.assertEqual(run("f::{{x+0}(1)}", "f()"), 1)                # was a monad: f() an incomplete call
        self.assertIsInstance(run("f::{{x}(1)}", "f()"), KGCall)        # {x}(1) is the same tree as x(1): x counts
        self.assertEqual(run("f::{a::{x+y};a(x;1)}", "f(2)"), 3)        # was a dyad: f(2) a projection
        self.assertEqual(run("f::{{x*y}@x}", "f'[[2 3] [4 5]]").tolist(), [6, 20])
        self.assertEqual(run("s::{{x+y}/x}", "s([1 2 3])"), 6)
        self.assertEqual(str(run("s::{{x+y}/x}", "s")), ":monad")
        self.assertEqual(str(run("{[f];f::{x+y+z};f(x;1;2)}")), ":monad")

    def test_a_body_that_is_a_function_value_takes_its_arity(self):
        # Kept as it was: a call of such a function is passed on to the value.
        self.assertEqual(run("g::{x+y}", "f::{g(1;)}", "f(5)"), 6)
        self.assertEqual(run("mk::{{x+y}}", "mk(1;2)"), 3)
        self.assertEqual(run("{{x+1}}(5)"), 6)
        self.assertEqual(run("{{-x}}(5)"), -5)
        self.assertEqual(run("f::{{x+1}}", "g::f()", "g(5)"), 6)
        self.assertEqual(run("f::{{x+y}(1;)}", "f(5)"), 6)
        for text, arity in [("{g(1;)}", 1), ("{g(;;)}", 3), ("{g(x;)}", 1), ("{g(;y)}", 2), ("{{x+y}}", 2),
                            ("{{-x}}", 1), ("{{x+y}(1;)}", 1), ("{{x+y}(1;2)}", 0), ("{{1}}", 0)]:
            with self.subTest(text=text):
                self.assertEqual(fn(text).arity, arity)


class TestAdverbs(unittest.TestCase):
    def test_each(self):
        self.assertEqual(run("{-x}'[1 2]").tolist(), [-1, -2])
        self.assertEqual(run("f::{#x}", "f'[[1] [1 2]]").tolist(), [1, 2])

    def test_over(self):
        self.assertEqual(run("{y}/[1 2 3]"), 3)
        self.assertEqual(run("{x,-y}/[1 2 3]").tolist(), [1, -2, -3])
        self.assertEqual(run("0{y}/[1 2 3]"), 3)

    def test_monad_under_each2_and_over(self):
        # a monad given two arguments ignores the second, as before
        self.assertEqual(run("[1 2]{-x}'[3 4]").tolist(), [-1, -2])
        self.assertEqual(run("{-x}/[1 2 3]"), 1)

    def test_scan_converge_while_iterate(self):
        self.assertEqual(run("{y}\\[1 2 3]").tolist(), [1, 2, 3])
        self.assertEqual(run("{-x}:~0"), 0)
        self.assertEqual(run("3{-x}:*1"), -1)
        self.assertEqual(run("{x<5}{-(-x)-1}:~0"), 5)

    def test_each_of_a_dyad_gives_projections_as_it_did_for_a_counted_dyad(self):
        before = run("{x+y}'[1 2]")
        after = run("{x,-y}'[1 2]")
        self.assertTrue(all(isinstance(q, KGCall) for q in before))
        self.assertTrue(all(isinstance(q, KGCall) for q in after))

    def test_apply_at(self):
        self.assertEqual(run("f::{-x}", "f@5"), -5)
        self.assertEqual(run("f::{y}", "f@[1 2]"), 2)


class TestDotF(unittest.TestCase):
    """.f is the function with its locals: a recursive call gets fresh ones."""

    def test_recursion_through_dot_f_has_its_own_locals(self):
        self.assertEqual(run("f::{[a];a::x;:[x<2;1;a*.f(x-1)]}", "f(5)"), 120)
        self.assertEqual(run("f::{[a];a::x;:[x>0;.f(x-1);0];a}", "f(3)"), 3)
        self.assertEqual(run("{[a];a::x;:[x<2;1;a*.f(x-1)]}(5)"), 120)

    def test_without_locals_as_before(self):
        self.assertEqual(run("f::{:[x<2;1;x*.f(x-1)]}", "f(5)"), 120)
        self.assertEqual(run("f::{:[x<2;x;.f(x-1)+.f(x-2)]}", "f(15)"), 610)

    def test_dot_f_under_an_adverb(self):
        self.assertEqual(run("f::{[a];a::x;:[x<1;0;a+(+/.f'[0],x-1)]}", "f(3)"), 6)


class TestCallbacks(unittest.TestCase):
    def test_wrapper_arity(self):
        k = KlongInterpreter()
        k("seen::[]")
        k("hook::{seen::seen,,x}")
        k["hook"](7)                                                    # was: called with 1 but expected 0
        k["hook"](8)
        self.assertEqual(k("seen").tolist(), [7, 8])

    def test_timer_callback_using_only_y_is_not_a_nilad(self):
        # .timer calls its function with no argument; a function of y is a dyad, and says so
        k = KlongInterpreter()
        k("cb::{y}")
        w = KGFnWrapper(k, k["cb"].fn)
        with self.assertRaises(KlongArityError) as c:
            w()
        self.assertIn("expected 2", str(c.exception))

    def test_ipc_error_hook_using_only_y(self):
        class Now:
            def call_soon_threadsafe(self, f):
                f()
        k = KlongInterpreter()
        k('err::""')
        k(".srv.e::{err::y}")
        h = TcpServerConnectionHandler(None, Now(), k)
        with self.assertNoLogs(level="WARNING"):
            h._call_hook(".srv.e", 1, "boom")
        self.assertEqual(k("err"), "boom")
        k("opened::[]")
        k(".srv.o::{opened::opened,,x}")
        with self.assertNoLogs(level="WARNING"):
            h._call_hook(".srv.o", 3)
        self.assertEqual(k._backend.to_numpy(k("opened")).tolist(), [3])  # to_numpy: a tensor on any device


if __name__ == "__main__":
    unittest.main()
