"""
Function metadata of a function value: arity (the highest argument mentioned), locals, parameters. A function is called
as before whatever it declares, and reading the metadata evaluates nothing.
"""
import unittest

import numpy as np

from klongpy import KlongInterpreter
from tests.backend_compat import plain
from klongpy.types import KGCall, KGFnWrapper, KGSym


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


class TestBehaviourIsUnchanged(unittest.TestCase):
    """Step 1: these passed before the change and pass after it."""

    def test_arity_is_the_highest_argument(self):
        # (was: a count of names, f(5) gave :y and {x+z} was a dyad)
        self.assertIsInstance(run("f::{y}", "f(5)"), KGCall)            # arity 2: f(5) is incomplete
        self.assertEqual(run("f::{x+z}", "f(1;2;3)"), 4)
        self.assertEqual(run("f::{{x}(1)}", "f(5)"), 1)
        self.assertIsInstance(run("f::{{x}(1)}", "f()"), KGCall)        # {x}(1) is the tree of x(1): x counts
        self.assertEqual(run("f::{x+y}", "g::f(1;)", "g(2)"), 3)
        self.assertIsInstance(run("f::{x+y}", "f(1)"), KGCall)

    def test_locals(self):
        self.assertEqual(run("{[a b];a::1;b::2;a+b}()"), 3)
        self.assertEqual(run("{[a b];a::x;a+x}(2)"), 4)
        self.assertEqual(run("{[a;b];a}()"), KGSym("a"))                # ';' in the list; a local is its own symbol
        self.assertEqual(run("{[a 1];x}(3)"), 3)                        # [a 1] is data, not a declaration
        self.assertEqual(run("f::{[a];a::x;a}", "f(1);a"), KGSym("a"))  # the local did not leak
        self.assertEqual(run("a::7", "f::{[a];a::x;a}", "f(1);a"), 7)
        self.assertEqual(run("f::{[a];a::x;a+1}", "g::f;g(2)"), 3)
        self.assertEqual(run("f::{[a];a::x;a+y}", "g::f(;10)", "g(2)"), 12)
        self.assertEqual(plain(run("{[a];a::x;a+1}'!5")), [1, 2, 3, 4, 5])  # plain: a tensor on any device

    def test_a_local_named_like_a_parameter_does_not_shadow_it(self):
        self.assertEqual(run("{[x];x}(3)"), 3)
        self.assertEqual(run("{[x y];x+y}(3;4)"), 7)

    def test_declaration_only_and_odd_bodies(self):
        # a body that is only a list declares nothing: at least one statement must follow
        self.assertTrue(np.array_equal(run("{[a]}()"), np.array([KGSym("a")], dtype=object)))
        self.assertEqual(run("{[a];1}()"), 1)

    def test_nested_functions(self):
        self.assertEqual(run("{{x}(x)}(7)"), 7)
        self.assertEqual(run("{[a];a::x;{[a];a::x*2;a}(a)+a}(3)"), 9)
        self.assertEqual(run("f::{x{x+y}'y}", "f([1 2];[3 4])").tolist(), [4, 6])

    def test_recursion_through_dot_f(self):
        self.assertEqual(run("f::{:[x<2;1;x*.f(x-1)]}", "f(5)"), 120)
        self.assertEqual(run("{:[x<2;1;x*.f(x-1)]}(5)"), 120)

    def test_dot_f_call_declares_its_locals(self):
        # Fixed (this test pinned the defect: 0, 1 and a bare statement list).
        # .f is the function with its declaration, so a recursive call has its own locals.
        self.assertEqual(run("f::{[a];a::x;:[x>0;.f(x-1);0];a}", "f(3)"), 3)
        self.assertEqual(run("f::{[a];a::x;:[x<2;1;a*.f(x-1)]}", "f(5)"), 120)

    def test_python_callers(self):
        k = KlongInterpreter()
        k("f::{[a];a::x;a+1}")
        w = k["f"]
        self.assertIsInstance(w, KGFnWrapper)
        self.assertEqual(w(2), 3)
        node = k["f"].fn
        # IPC and timer callbacks hand apply() the bare body: it carries its locals
        self.assertEqual(k.apply(node.a, [2], node.arity), 3)
        self.assertEqual(k.apply(node, [2]), 3)
        self.assertEqual(k("a"), KGSym("a"))


class TestMetadata(unittest.TestCase):
    def test_acceptance(self):
        f = fn("{[a b];a+x}")
        self.assertEqual(tuple(f.locals), (KGSym("a"), KGSym("b")))
        self.assertEqual((f.arity, f.params), (1, None))
        f = fn("{y}")
        self.assertEqual((f.arity, tuple(f.locals)), (2, ()))       # arity is the highest argument
        self.assertEqual(fn("{x+z}").arity, 3)
        self.assertEqual(fn("{1}").arity, 0)

    def test_nothing_is_evaluated_to_read_them(self):
        k = KlongInterpreter()
        k("h::{[a];a::undefined(x);.p(a)}")
        f = k["h"].fn
        self.assertEqual((tuple(f.locals), f.arity), ((KGSym("a"),), 1))
        with self.assertRaises(KeyError):
            k["a"]

    def test_literal_that_is_called_or_projected(self):
        f = fn("{x+y}(1;)")
        self.assertIsInstance(f, KGCall)
        self.assertEqual(f.arity, 1)


class TestArityCrashesFixed(unittest.TestCase):
    """The wrong cases: get_fn_arity failed on these bodies."""

    def test_function_returning_a_function_that_is_a_name(self):
        self.assertEqual(fn("{{f}}").arity, 0)      # TypeError: 'NoneType' object is not iterable

    def test_a_body_that_is_one_call(self):
        # no special rule any more, except that a projection takes an argument
        # per hole. Changed from the old count: {g(;;)} 1, {g(x+1)} 0.
        for text, arity in [("{g(x)}", 1), ("{g(x;x)}", 1), ("{g(x;y)}", 2), ("{g(1;)}", 1), ("{g(;;)}", 3),
                            ("{g(;x)}", 1), ("{g(x+1)}", 1), ("{g(:x;x)}", 1)]:
            with self.subTest(text=text):
                self.assertEqual(fn(text).arity, arity)


if __name__ == "__main__":
    unittest.main()
