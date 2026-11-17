"""
A string or char literal whose content is a bracket (or another
piece of punctuation) is that string or char, never the punctuation.

The legacy parser decided what a lexeme was by comparing the value its reader
returned, so the literal "{" started a function. The token parser knows the
kind of every token. The legacy parser (parser="legacy") keeps the defect and
is not tested here.
"""
import unittest
import warnings

import numpy as np

from klongpy.errors import KlongSyntaxWarning
from klongpy import KlongInterpreter as _KlongInterpreter
from klongpy.core import KGChar, KlongException
from klongpy.dump import dump_text as _dump_text

def KlongInterpreter():
    return _KlongInterpreter(parser="token")


def dump_text(text):
    return _dump_text(text, parser="token")


PUNCT = ["{", "}", "(", ")", "[", "]", ";", ":[", ":(", ":{", ":|"]


def lit(s):
    return '"' + s + '"'


class TestBracketStrings(unittest.TestCase):
    def test_top_level(self):
        for p in PUNCT:
            with self.subTest(p=p):
                r = KlongInterpreter()(lit(p))
                self.assertEqual(r, p)
                self.assertIs(type(r), str)

    def test_assignment_right_side(self):
        for p in PUNCT:
            with self.subTest(p=p):
                self.assertEqual(KlongInterpreter()(f"x::{lit(p)};x"), p)
                k = KlongInterpreter()
                k(f"v::{lit(p)}")
                self.assertEqual(k["v"], p)

    def test_function_argument(self):
        for p in PUNCT:
            with self.subTest(p=p):
                k = KlongInterpreter()
                k("f::{x}")
                self.assertEqual(k(f"f({lit(p)})"), p)
                k("g::{y}")
                self.assertEqual(k(f"g(1;{lit(p)})"), p)
                self.assertEqual(k(f"g({lit(p)};{lit(p)})"), p)

    def test_inside_a_function_body(self):
        for p in PUNCT:
            with self.subTest(p=p):
                self.assertEqual(KlongInterpreter()("{" + lit(p) + "}()"), p)
                self.assertEqual(KlongInterpreter()("{x;" + lit(p) + "}(1)"), p)
                self.assertEqual(KlongInterpreter()("f::{[a];a::" + lit(p) + ";a};f()"), p)

    def test_as_operand(self):
        for p in PUNCT:
            with self.subTest(p=p):
                self.assertEqual(KlongInterpreter()(f"#{lit(p)}"), len(p))
                self.assertEqual(KlongInterpreter()(f'{lit(p)},"a"'), p + "a")
                self.assertEqual(KlongInterpreter()(f'"a",{lit(p)}'), "a" + p)
                self.assertEqual(KlongInterpreter()(f"({lit(p)})"), p)

    def test_in_conditional_and_evaluated_list(self):
        for p in PUNCT:
            with self.subTest(p=p):
                self.assertEqual(KlongInterpreter()(f":[1;{lit(p)};0]"), p)
                self.assertEqual(KlongInterpreter()(f":[0;0;{lit(p)}]"), p)
                r = KlongInterpreter()(f"[;{lit(p)};1]")
                self.assertEqual(list(r), [p, 1])

    def test_inside_a_list(self):
        for p in PUNCT:
            with self.subTest(p=p):
                r = KlongInterpreter()(f"[{lit(p)} 1]")
                self.assertEqual(list(r), [p, 1])
        # the legacy list reader started a nested list at the string "["
        self.assertEqual(dump_text('["[" 1 2]'), '(list (str "[") (num 1) (num 2))\n')

    def test_inside_a_dictionary(self):
        for p in PUNCT:
            with self.subTest(p=p):
                r = KlongInterpreter()(":{[" + lit(p) + " 1]}")
                self.assertEqual(r, {p: 1})

    def test_statement_after_a_literal_semicolon(self):
        # the literal ";" is a value: it is neither an empty statement nor a separator
        self.assertEqual(KlongInterpreter()('";"'), ";")
        self.assertEqual(KlongInterpreter()('1;";"'), ";")
        self.assertEqual(dump_text('";";2'), '(str ";")\n(num 2)\n')

    def test_fmt_template(self):
        # the template "{" reaches .fmt, which rejects it: a template error, not a parse error
        with self.assertRaises(KlongException) as cm:
            KlongInterpreter()('.fmt("{";1)')
        self.assertIn(".fmt", str(cm.exception))
        self.assertEqual(KlongInterpreter()('.fmt("{}";1)'), "1")

    def test_unchanged_inputs(self):
        self.assertEqual(KlongInterpreter()('"a{b"'), "a{b")
        self.assertEqual(KlongInterpreter()('"}"'), "}")
        r = KlongInterpreter()('["{"]')
        self.assertEqual(list(r), ["{"])


class TestBracketChars(unittest.TestCase):
    def test_chars(self):
        for c in "{}()[];":
            with self.subTest(c=c):
                r = KlongInterpreter()("0c" + c)
                self.assertEqual(r, c)
                self.assertIsInstance(r, KGChar)

    def test_char_as_argument_and_in_body(self):
        for c in "{(;":
            with self.subTest(c=c):
                k = KlongInterpreter()
                k("f::{x}")
                r = k(f"f(0c{c})")
                self.assertEqual(r, c)
                self.assertIsInstance(r, KGChar)
                r = KlongInterpreter()("{0c" + c + "}()")
                self.assertIsInstance(r, KGChar)
                self.assertEqual(r, c)
                self.assertEqual(KlongInterpreter()(f"x::0c{c};x"), c)

    def test_chars_in_a_list(self):
        r = KlongInterpreter()("[0c{ 0c[ 0c;]")
        self.assertEqual(list(r), ["{", "[", ";"])
        self.assertTrue(all(isinstance(q, KGChar) for q in r))
        self.assertIsInstance(r, np.ndarray)


class TestPunctuationIsStillPunctuation(unittest.TestCase):
    def setUp(self):
        # these inputs are malformed on purpose: the parser warns about them at strict level 0
        cw = warnings.catch_warnings()
        cw.__enter__()
        self.addCleanup(cw.__exit__, None, None, None)
        warnings.simplefilter("ignore", KlongSyntaxWarning)

    def test_real_punctuation(self):
        k = KlongInterpreter()
        self.assertEqual(k("{x+1}(1)"), 2)
        self.assertEqual(k("(1+2)*3"), 9)
        self.assertEqual(k(":[1;2;3]"), 2)
        self.assertEqual(k("1;2"), 2)
        self.assertEqual(k(":[0;1:|1;2;3]"), 2)
        self.assertEqual(k("f::{x-y};f:(3;1)"), 2)
        self.assertEqual(k(":{[1 2]}"), {1: 2})

    def test_stray_punctuation_keeps_its_legacy_meaning(self):
        # a closer where a statement is expected is still the string it is written as
        self.assertEqual(KlongInterpreter()("1;)"), ")")
        self.assertEqual(dump_text("1;)"), '(num 1)\n(str ")")\n')
        self.assertEqual(dump_text("[1;2]"), '(list (num 1) (str ";") (num 2))\n')


if __name__ == "__main__":
    unittest.main()
