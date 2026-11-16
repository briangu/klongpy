"""
Destructuring assignment, [a b c]::v (ticket 060; design:
tickets/design/language-additions.md section 1).

A list of plain names on the left of :: binds each name to one element of the
value. As the first statement of a function body the names are also locals of
the function (the head form). Every rule is run at strict levels 0, 1 and 2.
The syntax exists on the token parser only; the legacy parser keeps what it
did (the value is returned and nothing is bound).
"""
import os
import re
import subprocess
import unittest
import warnings

import numpy as np

from klongpy import KlongInterpreter
from klongpy.core import KGChar, KGSym
from klongpy.dump import dump, dump_text, node_spans, parse_program
from klongpy.errors import (KlongError, KlongNameError, KlongShapeError, KlongSyntaxError, KlongSyntaxWarning,
                            KlongTypeError)

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


def num(v):
    """A number as a Python number, whatever the backend holds it in."""
    return v.item() if hasattr(v, "item") else v


def nums(v):
    return [num(q) for q in v]


class TestTopLevel(unittest.TestCase):
    def test_binds_each_name(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(num(run(level, "[a b c]::[1 2 3];a+b+c")), 6)
                self.assertEqual(num(run(level, "[a b c]::[1 2 3];b")), 2)

    def test_value_of_the_statement(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(nums(run(level, "[a b c]::[1 2 3]")), [1, 2, 3])
                self.assertEqual(run(level, '[a b]::"xy"'), "xy")
                self.assertEqual(nums(run(level, "v::[p q]::[4 5];v")), [4, 5])

    def test_one_name(self):
        for level in LEVELS:
            self.assertEqual(num(run(level, "[a]::[5];a")), 5)

    def test_the_right_side_is_an_expression(self):
        for level in LEVELS:
            self.assertEqual(num(run(level, "v::[1 2 3];[a b]::2#v;a+b")), 3)
            self.assertEqual(num(run(level, "[a b]::[;1+1;2*3];a*b")), 12)

    def test_string_gives_chars(self):
        for level in LEVELS:
            k = KlongInterpreter(strict=level)
            k('[a b]::"xy"')
            self.assertEqual(k("a"), KGChar("x"))
            self.assertIsInstance(k("a"), KGChar)
            self.assertEqual(k("b"), KGChar("y"))

    def test_matrix_gives_rows(self):
        for level in LEVELS:
            k = KlongInterpreter(strict=level)
            k("[r s]::[[1 2 3] [4 5 6]]")
            self.assertEqual(nums(k("r")), [1, 2, 3])
            self.assertEqual(nums(k("s")), [4, 5, 6])

    def test_mixed_list(self):
        for level in LEVELS:
            k = KlongInterpreter(strict=level)
            k('[n s l]::[;1;"two";[3 4]]')
            self.assertEqual(num(k("n")), 1)
            self.assertEqual(k("s"), "two")
            self.assertEqual(nums(k("l")), [3, 4])

    def test_same_as_one_assignment_per_name(self):
        # an existing variable is assigned where it lives, a new one is created
        for level in LEVELS:
            self.assertEqual(num(run(level, "a::1;[a b]::[10 20];a+b")), 30)

    def test_parentheses_are_not_part_of_the_tree(self):
        for level in LEVELS:
            self.assertEqual(num(run(level, "([a b])::[1 2];a+b")), 3)

    def test_newlines_inside_the_target_list(self):
        self.assertEqual(num(run(0, "[a\n b]::[1 2];a+b")), 3)

    def test_a_list_on_the_right_is_untouched(self):
        for level in LEVELS:
            k = KlongInterpreter(strict=level)
            r = k("v::[a b]")
            self.assertEqual(list(r), [KGSym("a"), KGSym("b")])
            self.assertEqual(list(k("[a b]")), [KGSym("a"), KGSym("b")])

    def test_module(self):
        k = KlongInterpreter()
        k('.module(:m);[p q]::[1 2];f::{p+q};.module(0)')
        self.assertEqual(num(k("f()")), 3)


class TestErrors(unittest.TestCase):
    def test_length_mismatch_is_a_shape_error(self):
        for level in LEVELS:
            with self.subTest(level=level):
                e = error(level, "[a b]::[1 2 3]")
                self.assertIsInstance(e, KlongShapeError)
                self.assertEqual(str(e), "cannot destructure 3 values into 2 names")
                e = error(level, "[a b c]::[1 2]")
                self.assertIsInstance(e, KlongShapeError)
                self.assertEqual(str(e), "cannot destructure 2 values into 3 names")
                self.assertIsInstance(error(level, '[a b]::"xyz"'), KlongShapeError)
                e = error(level, "[a]::[]")
                self.assertEqual(str(e), "cannot destructure 0 values into 1 name")

    def test_not_a_list_is_a_type_error(self):
        for level in LEVELS:
            with self.subTest(level=level):
                e = error(level, "[a b]::5")
                self.assertIsInstance(e, KlongTypeError)
                self.assertEqual(str(e), "cannot destructure integer into 2 names")
                e = error(level, "[a b c]::1.5")
                self.assertEqual(str(e), "cannot destructure float into 3 names")
                e = error(level, "[a b]:::{[1 2] [3 4]}")
                self.assertIsInstance(e, KlongTypeError)
                self.assertEqual(str(e), "cannot destructure dict into 2 names")
                self.assertIsInstance(error(level, "[a b]:::foo"), KlongTypeError)
                self.assertIsInstance(error(level, "[a b]::0cx"), KlongTypeError)
                self.assertIsInstance(error(level, "[a b]::{x}"), KlongTypeError)

    def test_nothing_is_bound_when_it_fails(self):
        k = KlongInterpreter()
        k("a::7")
        with self.assertRaises(KlongShapeError):
            k("[a b]::[1 2 3]")
        self.assertEqual(k("a"), 7)
        self.assertEqual(k("b"), KGSym("b"))

    def test_position_is_the_assignment(self):
        e = error(0, "q::1;[a b]::5")
        self.assertEqual(e.span, (10, 12))
        self.assertIn("type error: cannot destructure integer into 2 names", e.render())
        self.assertIn("^^", e.render())

    def test_hint(self):
        self.assertIn("list", error(0, "[a b]::5").hint)


class TestNotADestructuring(unittest.TestCase):
    """Any other list on the left of :: is an error at levels 1 and 2 and, at level 0, a warning (nothing is bound)."""

    CASES = {
        "[a b a]::[1 2 3]": "is named twice",
        "[x y]::[1 2]": "x, y and z",
        "[a z]::[1 2]": "x, y and z",
        "[a [b c]]::[1 [2 3]]": "nested",
        "[a 1]::[1 2]": "only names",
        "[a :b]::[1 2]": "only names",
        '[a "b"]::[1 2]': "only names",
        "[a _ c]::[1 2 3]": "only names",
        "[a;b]::[1 2]": "only names",
        "[]::[]": "empty",
        "[1 2]::[1 2]": "only names",
        "[.f a]::[1 2]": ".f",
    }

    def test_levels_1_and_2_are_syntax_errors(self):
        for level in (1, 2):
            for text, word in self.CASES.items():
                with self.subTest(level=level, text=text):
                    e = error(level, text)
                    self.assertIsInstance(e, KlongSyntaxError)
                    self.assertIn("cannot destructure", str(e))
                    self.assertIn(word, str(e) + " " + (e.hint or ""))
                    self.assertIsNotNone(e.span)
                    self.assertEqual(e.source.text[e.span[0]], "[")

    def test_level_0_warns_and_binds_nothing(self):
        for text, word in self.CASES.items():
            with self.subTest(text=text):
                k = KlongInterpreter()
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    r = k(text)
                w = [q for q in w if issubclass(q.category, KlongSyntaxWarning)]
                self.assertEqual(len(w), 1, text)
                msg = str(w[0].message)
                self.assertIn("cannot destructure", msg)
                self.assertIn("nothing is assigned", msg)
                self.assertIn(word, msg + " " + (w[0].message.hint or ""))
                self.assertEqual(w[0].message.code, "not-a-destructuring")
                # what it always did: the value, and no name is bound
                self.assertIsNotNone(r)
                self.assertEqual(k("a"), KGSym("a"))

    def test_nested_says_nesting_is_not_supported(self):
        e = error(1, "[a [b c]]::[1 [2 3]]")
        self.assertIn("nested", str(e))
        self.assertIn("not supported", str(e))

    def test_acceptance_duplicate(self):
        self.assertIsInstance(error(1, "[a b a]::[1 2 3]"), KlongSyntaxError)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", KlongSyntaxWarning)
            k = KlongInterpreter()
            k("[a b a]::[1 2 3]")
            self.assertEqual(k("a"), KGSym("a"))
            self.assertEqual(k("b"), KGSym("b"))


class TestInFunctions(unittest.TestCase):
    def test_head_form_declares_locals(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(num(run(level, "a::7;{[p q]::x;p+q}([1 2])")), 3)
                self.assertEqual(num(run(level, "{[p q]::x;p}([1 2])")), 1)
                # a global of the same name is untouched, and no global is made
                k = KlongInterpreter(strict=level)
                k("p::7;f::{[p q]::x;p+q}")
                self.assertEqual(num(k("f([1 2])")), 3)
                self.assertEqual(k("p"), 7)
                if level < 2:
                    self.assertEqual(k("q"), KGSym("q"))
                else:
                    with self.assertRaises(KlongNameError):
                        k("q")

    def test_acceptance_global_untouched(self):
        for level in LEVELS:
            self.assertEqual(run(level, "a::7;{[p q]::x;p+q}([1 2]);a"), 7)
            self.assertEqual(run(level, "a::7;{[a q]::x;a+q}([1 2]);a"), 7)

    def test_one_statement_body(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = KlongInterpreter(strict=level)
                k("a::7;f::{[a b]::x}")
                self.assertEqual(nums(k("f([1 2])")), [1, 2])
                self.assertEqual(k("a"), 7)

    def test_head_form_after_a_locals_list(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = KlongInterpreter(strict=level)
                k("t::0;s::0;f::{[t];[s u]::x;t::s+u;t*2}")
                self.assertEqual(num(k("f([1 2])")), 6)
                self.assertEqual((k("t"), k("s")), (0, 0))
                self.assertEqual(tuple(k._context[KGSym("f")].locals), (KGSym("t"), KGSym("s"), KGSym("u")))

    def test_metadata(self):
        k = KlongInterpreter()
        k("f::{[s k r]::x;s+k+r}")
        f = k._context[KGSym("f")]
        self.assertEqual(tuple(f.locals), (KGSym("s"), KGSym("k"), KGSym("r")))
        self.assertEqual(f.arity, 1)
        self.assertEqual(f.max_arg, 1)

    def test_acceptance_atom_on_the_right_is_a_type_error(self):
        # (at level 2 fn(0) is one argument too many for a function that uses none)
        for level, call in ((0, "fn(0)"), (1, "fn(0)"), (2, "fn()")):
            e = error(level, "fn::{[a b c]::1;a};" + call)
            self.assertIsInstance(e, KlongTypeError)
            self.assertEqual(str(e), "cannot destructure integer into 3 names")

    def test_elsewhere_in_a_body_is_ordinary_assignment(self):
        # level 0: the names become locals of the call (as any assignment does)
        self.assertEqual(num(run(0, "f::{1;[p q]::x;p+q};f([1 2])")), 3)
        self.assertEqual(run(0, "f::{1;[p q]::x;p+q};f([1 2]);p"), KGSym("p"))
        for level in (1, 2):
            with self.subTest(level=level):
                # undeclared: a name error, as for p::1
                e = error(level, "f::{1;[p q]::x;p+q};f([1 2])")
                self.assertIsInstance(e, KlongNameError)
                # declared: fine
                self.assertEqual(num(run(level, "f::{[p q];1;[p q]::x;p+q};f([1 2])")), 3)
                # existing globals are assigned
                self.assertEqual(num(run(level, "p::0;q::0;f::{1;[p q]::x;0};f([1 2]);p+q")), 3)

    def test_recursion_and_dot_f(self):
        for level in LEVELS:
            self.assertEqual(num(run(level, "f::{[n acc]::x;:[n<1;acc;.f((n-1),acc*n)]};f([5 1])")), 120)

    def test_under_adverbs_and_from_python(self):
        for level in LEVELS:
            k = KlongInterpreter(strict=level)
            k("f::{[p q]::x;p*q}")
            self.assertEqual(nums(k("f'[[1 2] [3 4] [5 6]]")), [2, 12, 30])
            self.assertEqual(num(k["f"]([3, 4])), 12)

    def test_strict_2_locals_are_unset_until_the_destructuring_runs(self):
        self.assertEqual(num(run(2, "{[p q]::x;p+q}([1 2])")), 3)

    def test_arguments_cannot_be_targets_in_a_function_either(self):
        self.assertIsInstance(error(1, "f::{[x y]::x;1}"), KlongSyntaxError)


class TestTree(unittest.TestCase):
    def D(self, text):
        return dump(parse_program(KlongInterpreter(), text)).rstrip("\n")

    def test_dump(self):
        self.assertEqual(self.D("[a b c]::v"), "(destructure (a b c) (sym v))")
        self.assertEqual(self.D("[a]::[5]"), "(destructure (a) (list (num 5)))")
        self.assertEqual(self.D("{[p q]::x;p+q}"),
                         '(fn 1 (locals p q) (destructure (p q) (arg x)) (d "+" (sym p) (sym q)))')
        self.assertEqual(self.D("{[p q]::x}"), "(fn 1 (locals p q) (destructure (p q) (arg x)))")
        self.assertEqual(self.D("{[t];[p q]::x;t}"), "(fn 1 (locals t p q) (destructure (p q) (arg x)) (sym t))")
        self.assertEqual(self.D("{1;[p q]::x}"), "(fn 1 (num 1) (destructure (p q) (arg x)))")
        self.assertEqual(self.D(".module(:m);[a b]::v"),
                         "(call (sym .module) (qsym m))\n(destructure (a`m b`m) (sym v`m))")

    def test_unchanged_forms(self):
        self.assertEqual(self.D("x::[a b]"), "(def (arg x) (list (sym a) (sym b)))")
        self.assertEqual(self.D("[a b]"), "(list (sym a) (sym b))")
        self.assertEqual(self.D("{[a b];a::x;a}"), '(fn 1 (locals a b) (def (sym a) (arg x)) (sym a))')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", KlongSyntaxWarning)
            self.assertEqual(self.D("[a [b]]::v"), "(def (list (sym a) (list (sym b))) (sym v))")
            self.assertEqual(self.D("[a 1]::v"), "(def (list (sym a) (num 1)) (sym v))")

    def test_spans(self):
        k = KlongInterpreter()
        tree = parse_program(k, "q;[a b]::v+1")
        (node, span, op_span, arg_spans), *_ = node_spans(tree)
        self.assertEqual(span, (2, 12))
        self.assertEqual(op_span, (7, 9))
        self.assertEqual(arg_spans, ((2, 7), (9, 12)))

    def test_legacy_parser_keeps_its_behaviour(self):
        k = KlongInterpreter(parser="legacy")
        self.assertEqual(nums(k("[a b]::[1 2]")), [1, 2])
        self.assertEqual(k("a"), KGSym("a"))
        self.assertEqual(dump_text("[a b]::[1 2]", parser="legacy"),
                         "(def (list (sym a) (sym b)) (list (num 1) (num 2)))\n")

    def test_compiled_path_is_not_involved(self):
        # a destructuring inside an arithmetic expression, evaluated twice
        k = KlongInterpreter()
        k("v::[1 2]")
        self.assertEqual(num(k("1+*[a b]::v")), 2)
        k("v::[5 6]")
        self.assertEqual(num(k("1+*[a b]::v")), 6)
        self.assertEqual(num(k("a+b")), 11)

    def test_function_survives_copy_and_pickle(self):
        import copy
        import pickle
        k = KlongInterpreter()
        k("f::{[p q]::x;p+q}")
        f = k._context[KGSym("f")]
        for g in (copy.deepcopy(f), pickle.loads(pickle.dumps(f))):
            k._context[KGSym("g")] = g
            self.assertEqual(num(k("g([1 2])")), 3)
            self.assertEqual(k("p"), KGSym("p"))


class TestRepositoryGate(unittest.TestCase):
    def test_no_tracked_kg_file_gains_a_meaning_by_accident(self):
        """
        ']' followed by '::' in a tracked .kg file is either a destructuring
        written on purpose (listed here) or a silent change of meaning.
        """
        allowed = {
            "examples/stocks/options/black-scholes.kg",
            "examples/1brc/worker.kg",
            "tests/kgtests/language/test_destructuring.kg",
            "tests/golden/fixtures/s2_destructuring_slot.kg",
        }
        try:
            out = subprocess.run(["git", "ls-files", "*.kg"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("not a git checkout")
        hits = []
        for rel in out.split():
            path = os.path.join(ROOT, rel)
            if not os.path.exists(path) or rel.startswith("tests/golden/fixtures/t060_"):
                continue
            with open(path, "r", errors="replace") as f:
                if re.search(r"\]\s*::", f.read()) and rel not in allowed:
                    hits.append(rel)
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
