"""
Function metadata computed once when a function literal is parsed:
arity, locals, max_arg and uses_self are on the node; the locals declaration is
not part of the stored body; calling a function no longer inspects its body.

The locals/arity metadata changed nothing a program can observe, except that two crashes of
get_fn_arity are gone. The first class pins that; the corpus class compares
every function literal of the repository with the code as it was.

A later change made arity the highest argument mentioned (max_arg) and bound
.f to the function with its locals. The tests that pinned the old count say
so where they changed; tests/test_fn_arity.py has the new behaviour.
"""
import copy
import os
import pickle
import re
import unittest

import numpy as np

from klongpy import KlongInterpreter
from tests.backend_compat import plain
from klongpy.dump import dump, dump_text
from klongpy.lexer import Source
from klongpy.parser import KGExprArray
from klongpy.types import (KGAdverb, KGBody, KGCall, KGCond, KGFn, KGFnWrapper, KGOp, KGSym, fn_metadata,
                           get_fn_arity, in_map, is_list, reserved_fn_symbols, split_locals)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
x, y, z = reserved_fn_symbols


def fn(text, parser=None):
    """The node of the function literal that text is."""
    return KlongInterpreter(parser=parser).prog(text)[1][0]


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
        # legacy accident, kept: a body that is one nested data list loses its first row
        r = run("{[[a b] [c d]]}()")
        self.assertEqual(r.shape, (1, 2))
        with self.assertRaises(ValueError):
            run("f::{[[a b] [c d]];1}", "f()")      # the declaration test itself fails, when called

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
        r = run("{[a];.f}()")
        self.assertIsInstance(r, KGBody)
        self.assertEqual((list(r), r.locals), ([KGSym(".f")], (KGSym("a"),)))

    def test_python_callers(self):
        k = KlongInterpreter()
        k("f::{[a];a::x;a+1}")
        w = k["f"]
        self.assertIsInstance(w, KGFnWrapper)
        self.assertEqual(w(2), 3)
        node = k._context[KGSym("f")]
        # IPC and timer callbacks hand apply() the bare body: it carries its locals
        self.assertEqual(k.apply(node.a, [2], node.arity), 3)
        self.assertEqual(k.apply(node, [2]), 3)
        self.assertEqual(k("a"), KGSym("a"))

    def test_legacy_parser_runs_the_same(self):
        k = KlongInterpreter(parser="legacy")
        self.assertEqual(k("{[a b];a::1;b::2;a+b}()"), 3)
        k("f::{[a];a::x;a+1}")
        self.assertEqual(k("f(1)"), 2)
        self.assertEqual(k("f(2)"), 3)
        self.assertEqual(k("a"), KGSym("a"))


class TestMetadata(unittest.TestCase):
    def test_acceptance(self):
        f = fn("{[a b];a+x}")
        self.assertEqual(f.locals, (KGSym("a"), KGSym("b")))
        self.assertEqual((f.max_arg, f.arity, f.uses_self, f.params), (1, 1, False, None))
        f = fn("{y}")
        self.assertEqual((f.arity, f.max_arg, f.locals), (2, 2, ()))     # arity is max_arg
        f = fn("{x+z}")
        self.assertEqual((f.arity, f.max_arg), (3, 3))
        self.assertEqual(fn("{1}").max_arg, 0)

    def test_nothing_is_evaluated_to_read_them(self):
        k = KlongInterpreter()
        f = k.prog("{[a];a::undefined(x);.p(a)}")[1][0]
        self.assertEqual((f.locals, f.max_arg, f.arity), ((KGSym("a"),), 1, 1))
        with self.assertRaises(KeyError):
            k["a"]

    def test_the_declaration_is_not_in_the_stored_body(self):
        f = fn("{[a b];a::x;a+b}")
        self.assertIsInstance(f.a, KGBody)
        self.assertEqual(len(f.a), 2)
        self.assertEqual(f.a.locals, f.locals)
        self.assertEqual(len(f.a.full()), 3)
        self.assertEqual(dump(f), '(fn 1 (locals a b) (def (sym a) (arg x)) (d "+" (sym a) (sym b)))\n')
        f = fn("{[a];a}")
        self.assertEqual(list(f.a), [KGSym("a")])           # still a statement list
        self.assertNotIsInstance(fn("{[a 1];x}").a, KGBody)
        self.assertNotIsInstance(fn("{x;y}").a, KGBody)
        self.assertEqual(fn("{[a;b];a}").locals, (KGSym("a"), KGSym("b")))

    def test_dump_is_the_same_on_both_parsers(self):
        for text in ("{[a b];a::x;a+b}", "{[a];1}", "{[a 1];1}", "{[a;b];a}", "{[x];x}", "f::{[a];{[b];b::a;b}(1)}"):
            with self.subTest(text=text):
                self.assertEqual(dump_text(text, parser="token"), dump_text(text, parser="legacy"))

    def test_spans(self):
        f = fn("{[a b];a::x;a+b}")
        self.assertEqual(f.locals_span, (1, 6))
        self.assertEqual(f.body_spans, ((7, 11), (12, 15)))

    def test_literal_that_is_called_or_projected(self):
        f = fn("{[a];a::x;a+y}(1;2)")
        self.assertIsInstance(f, KGCall)
        self.assertEqual((f.locals, f.max_arg, f.arity), ((KGSym("a"),), 2, 2))
        f = fn("{x+y}(1;)")
        self.assertEqual((f.locals, f.max_arg), ((), 2))

    def test_max_arg_looks_everywhere_and_arity_is_max_arg(self):
        # the second column is what arity was before arity became max_arg
        for text, _, max_arg in [("{-x}", 0, 1), ("{#x}", 0, 1), ("{x,-y}", 1, 2), ("{a::#x;a}", 0, 1),
                                 ("{:[x;y;z]}", 3, 3), ("{[;x;y]}", 2, 2), ("{+/x}", 1, 1), ("{x+/y}", 2, 2),
                                 ("{z{x+y}'x}", 2, 3), ("{:[0;1;-z]}", 0, 3), ("{[x y z]}", 0, 0),
                                 ("{:{[1 2]}}", 0, 0), ("{[1 2 3]}", 0, 0)]:
            arity = max_arg
            with self.subTest(text=text):
                f = fn(text)
                self.assertEqual((f.arity, f.max_arg), (arity, max_arg))

    def test_nested_function_bodies_do_not_count(self):
        # the x, y, z of a nested literal are its own. (Before, arity counted
        # them except in the verb of an adverb, and max_arg counted all of them.)
        for text, n in [("{{x+y}'x}", 1), ("{a::{x+y+z};a(x;1;2)}", 1), ("{{x*y}@x}", 1), ("{:[x;{y};0]}", 1),
                        ("{{x+y}(1;2)}", 0), ("{{x+y}(z;2)}", 3), ("{x{y}'x}", 1)]:
            with self.subTest(text=text):
                f = fn(text)
                self.assertEqual((f.arity, f.max_arg), (n, n))
        # {x}(1) and x(1) are one tree; the name in call position counts
        f = fn("{{x}(1)}")
        self.assertEqual((f.arity, f.max_arg), (1, 1))
        f = fn("{{y}(1)}")
        self.assertEqual((f.arity, f.max_arg), (2, 2))

    def test_uses_self(self):
        self.assertTrue(fn("{:[x<2;1;x*.f(x-1)]}").uses_self)
        self.assertTrue(fn("{[a];a::.f;a}").uses_self)
        self.assertTrue(fn("{-.f(x)}").uses_self)
        self.assertFalse(fn("{x+1}").uses_self)
        self.assertFalse(fn("{{.f(x)}(x)}").uses_self)      # .f there is the inner function
        self.assertFalse(fn("{[.f];1}").uses_self is None)

    def test_other_nodes_have_none(self):
        k = KlongInterpreter()
        for text in ("1+2", "f(1)", "+/x", "-x", "f(1;)"):
            n = k.prog(text)[1][0]
            self.assertEqual((n.locals, n.max_arg, n.uses_self, n.params), (None, None, None, None))
        self.assertIsNone(KGFn(x, None, 1).locals)

    def test_pickled_and_copied_functions_keep_their_locals(self):
        k = KlongInterpreter()
        k("f::{[a b];a::x;b::a+1;b}")
        node = k._context[KGSym("f")]
        for clone in (pickle.loads(pickle.dumps(node)), copy.deepcopy(node), copy.copy(node)):
            self.assertIsInstance(clone.a, KGBody)
            self.assertEqual((clone.locals, clone.max_arg, clone.uses_self, clone.arity), ((KGSym("a"), KGSym("b")), 1, False, 1))
            k["g"] = clone
            self.assertEqual(k("g(5)"), 6)
            self.assertEqual(k("a"), KGSym("a"))

    def test_have_locals_scan_is_gone(self):
        with open(os.path.join(ROOT, "klongpy", "interpreter.py")) as f:
            self.assertNotIn("have_locals", f.read())


class TestHandBuiltFunctions(unittest.TestCase):
    """A KGFn made without metadata: computed on first use and kept on the node."""

    def body(self):
        a = KGSym("a")
        return [np.array([a], dtype=object), KGFn(KGOp("::", 2), [a, x], 2), KGFn(KGOp("+", 2), [a, 1], 2)]

    def test_first_call_computes_and_caches(self):
        k = KlongInterpreter()
        f = KGFn(self.body(), None, 1)
        self.assertIsNone(f.locals)
        k["f"] = f
        self.assertEqual(k("f(2)"), 3)
        self.assertEqual(f.locals, (KGSym("a"),))
        self.assertIsInstance(f.a, KGBody)
        self.assertEqual((f.max_arg, f.uses_self), (1, False))
        self.assertEqual(k("f(5)"), 6)
        self.assertEqual(k("f'[1 2]").tolist(), [2, 3])
        self.assertEqual(k("a"), KGSym("a"))

    def test_fn_metadata_on_demand(self):
        f = fn_metadata(KGFn(self.body(), None, 1))
        self.assertEqual((f.locals, f.max_arg, f.uses_self, len(f.a)), ((KGSym("a"),), 1, False, 2))
        self.assertIs(fn_metadata(f), f)
        g = fn_metadata(KGFn(KGFn(KGOp("+", 2), [x, z], 2), None, 2))
        self.assertEqual((g.locals, g.max_arg), ((), 3))

    def test_bare_body_passed_to_apply(self):
        k = KlongInterpreter()
        self.assertEqual(k.apply(self.body(), [2], 1), 3)
        self.assertEqual(k("a"), KGSym("a"))

    def test_wrapper_of_a_hand_built_function(self):
        k = KlongInterpreter()
        w = KGFnWrapper(k, KGFn(self.body(), None, 1))
        self.assertEqual(w(4), 5)
        self.assertEqual(w(5), 6)


class TestArityCrashesFixed(unittest.TestCase):
    """The wrong cases: get_fn_arity failed on these bodies. Both parsers share the fix."""

    def test_call_with_a_list_argument(self):
        for parser in ("token", "legacy"):
            k = KlongInterpreter(parser=parser)
            k("nt::{x}")
            k('newt::{nt([["1" 2] ["3" 4] ["5" 6]])}')      # TypeError: unhashable type: 'numpy.ndarray'
            self.assertEqual(k._context[KGSym("newt")].arity, 0)
            self.assertEqual(k("newt()").shape, (3, 2))
            self.assertEqual(fn("{g([1 2];x)}", parser).arity, 1)
            self.assertEqual(fn("{g(:{[1 2]})}", parser).arity, 0)

    def test_function_returning_a_function_that_is_a_name(self):
        for parser in ("token", "legacy"):
            self.assertEqual(fn("{{f}}", parser).arity, 0)      # TypeError: 'NoneType' object is not iterable
            self.assertEqual(dump_text("{{f}}", parser=parser), "(fn 0 (fn 0 (sym f)))\n")

    def test_a_body_that_is_one_call(self):
        # no special rule any more, except that a projection takes an argument
        # per hole. Changed from the old count: {g(;;)} 1, {g(x+1)} 0.
        for text, arity in [("{g(x)}", 1), ("{g(x;x)}", 1), ("{g(x;y)}", 2), ("{g(1;)}", 1), ("{g(;;)}", 3),
                            ("{g(;x)}", 1), ("{g(x+1)}", 1), ("{g(1;2)}", 0), ("{g()}", 0), ("{g(:x;x)}", 1),
                            ('{g("x")}', 0), ("{g(0cx)}", 0)]:
            with self.subTest(text=text):
                self.assertEqual(fn(text).arity, arity)
                self.assertEqual(fn(text, "legacy").arity, arity)


# get_fn_arity and the locals test of _run_fn as they were at commit a96ca38,
# the oracle for the comparison below.

def old_get_fn_arity(f):
    if isinstance(f, KGFn) and isinstance(f.a, KGSym) and not in_map(f.a, reserved_fn_symbols):
        return sum(1 for q in set(f.args) if in_map(q, reserved_fn_symbols) or (q is None))
    def _e(f, level=0):
        if isinstance(f, KGFn):
            r = _e(f.a, level=1)
            if isinstance(f.args, list):
                for q in f.args:
                    r.update(_e(q, level=1))
        elif isinstance(f, list):
            r = set()
            for q in f:
                r.update(_e(q, level=1))
        elif isinstance(f, KGSym):
            r = set([f]) if f in reserved_fn_symbols else set()
        else:
            r = set()
        return r if level else len(r)
    return _e(f)


def old_locals(f):
    """(locals, body after them) by the scan _run_fn made on every call."""
    if is_list(f) and len(f) > 1 and is_list(f[0]) and len(f[0]) > 0:
        non_sep = [q for q in f[0] if q != ';']
        params = [q for q in non_sep if isinstance(q, KGSym)]
        if len(params) > 0 and len(params) == len(non_sep):
            return tuple(params), f[1:]
    return (), f


def literals(tree, out):
    """Every function literal node in a parse tree of the token parser (the nodes _read_fn made)."""
    if isinstance(tree, KGFn):
        if tree.body_spans is not None:
            out.append(tree)
        a = tree.a
        if isinstance(a, list) and a and isinstance(a[0], KGAdverb):
            literals(a[0].a, out)
            literals(a[-1], out)
        else:
            literals(a, out)
        if not isinstance(tree.args, dict):
            literals(tree.args, out)
    elif isinstance(tree, (list, KGCond, KGExprArray)):
        for q in tree:
            literals(q, out)
    return out


class TestCorpus(unittest.TestCase):
    """Every function literal in tests/kgtests, klongpy/lib and examples against the old code."""

    # Written with the syntax of destructuring, named parameters and the evaluated dictionary (destructuring, named parameters,
    # evaluated dictionary literal), which the old code and the legacy parser do not have.
    NEW_SYNTAX = ("tests/kgtests/language/test_destructuring.kg", "tests/kgtests/language/test_named_params.kg",
                  "tests/kgtests/language/test_dict.kg")

    # The examples are written in the new forms to a large extent, so they are not listed: an example is left out
    # when its text has one of the forms the old code and the legacy parser do not have: a destructuring
    # (]::), a named-parameter head (name(a;b)::), an evaluated dictionary (:{[ or :::{;). The same pattern is
    # checked against the legacy parser in test_parser_tokens (TestSameTreeAsLegacy.test_new_form_pattern_*).
    NEW_FORMS = re.compile(r"\]\s*::|^[ \t]*[\w.]+\([^()\n]*\)[ \t]*::|(?<!:):\{\s*\[|:::?\{\s*;", re.M)

    def sources(self):
        skip = {os.path.join(ROOT, *q.split("/")) for q in self.NEW_SYNTAX}
        for d in ("tests/kgtests", "klongpy/lib", "examples"):
            for dp, _, fns in os.walk(os.path.join(ROOT, d)):
                for name in sorted(fns):
                    if name.endswith(".kg") and os.path.join(dp, name) not in skip:
                        path = os.path.join(dp, name)
                        with open(path, newline="") as f:
                            text = f.read()
                        if d == "examples" and self.NEW_FORMS.search(text):
                            continue
                        yield os.path.relpath(path, ROOT), text

    def test_arity_and_locals_are_what_the_old_code_computed(self):
        # Locals: identical to the old scan. Arity: it is max_arg; the
        # literals where that is not the old count are counted here by (old, new).
        total = with_locals = crashes = 0
        changed = {}
        k = KlongInterpreter()
        for rel, text in self.sources():
            saved = k._module
            try:
                tree = k.prog(Source(rel, text))[1]
            finally:
                k._module = saved
            for f in literals(tree, []):
                total += 1
                where = f"{rel}:{f.source.line_col(f.span[0])[0]}"
                body = f.a.full() if type(f.a) is KGBody else f.a
                try:
                    want = old_get_fn_arity(body)
                except TypeError:
                    # the listed wrong case: the old code could not parse this function at all
                    crashes += 1
                    continue
                self.assertEqual(f.arity, f.max_arg, where)
                if f.arity != want:
                    changed[(want, f.arity)] = changed.get((want, f.arity), 0) + 1
                loc, rest = old_locals(body)
                self.assertEqual(f.locals, loc, where)
                stored = f.a
                if loc:
                    with_locals += 1
                    self.assertIsInstance(stored, KGBody, where)
                    self.assertEqual(len(stored), len(rest), where)
                    self.assertTrue(all(p is q for p, q in zip(stored, rest)), where)
                    self.assertTrue(all(p is q for p, q in zip(stored.plain, rest)), where)
                else:
                    self.assertIs(stored, rest, where)
                self.assertGreaterEqual(f.max_arg, 0)
                self.assertLessEqual(f.max_arg, 3)
        self.assertGreater(total, 1500)
        self.assertGreater(with_locals, 100)
        self.assertEqual(crashes, 0)        # no file in the repository has one: they could not be loaded
        # Undercounts fixed (an argument under a monad, inside a call argument, or a gap
        # such as {y}), and seven overcounts (a nested literal's arguments were counted).
        self.assertTrue(all(new > old for old, new in changed if (old, new) not in ((1, 0), (2, 1))), changed)
        self.assertEqual(changed.get((1, 0), 0) + changed.get((2, 1), 0), 7)
        self.assertGreater(sum(changed.values()), 200)

    def test_arity_equals_the_legacy_parser_tree(self):
        """The same, against the frozen legacy parser: the dumps print each function's arity."""
        for rel, text in self.sources():
            with self.subTest(path=rel):
                self.assertEqual(dump_text(text, parser="token"), dump_text(text, parser="legacy"))

    def test_arity_is_max_arg_everywhere(self):
        """(Before the change, 258 literals had arity != max_arg.)"""
        k = KlongInterpreter()
        total = differ = 0
        for rel, text in self.sources():
            saved = k._module
            try:
                tree = k.prog(text)[1]
            finally:
                k._module = saved
            for f in literals(tree, []):
                total += 1
                differ += f.arity != f.max_arg
        self.assertEqual(differ, 0)
        self.assertGreater(total, 1500)


if __name__ == "__main__":
    unittest.main()
