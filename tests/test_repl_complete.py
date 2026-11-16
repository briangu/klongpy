"""Table-driven tests for check_complete (ticket 070)."""
import unittest

from klongpy.lexer import Source, tokenize
from klongpy.repl_input import check_complete, check_text

C, I, E = "complete", "incomplete", "error"

CASES = [
    # empty and plain
    ("", C), ("   ", C), ("1+2", C), ("f(1;2)", C), ("\n", C),
    # every opener, left open
    ("(", I), ("[", I), ("{", I), (":[", I), (":{", I), (":(", I),
    ("(1+2", I), ("arr::[1", I), ("f::{", I), ("f::{x+1", I), (":[1;2", I),
    ("{x}(", I), ("d:{[1 2]", I), ("f:(1", I), ("x:[1;2;", I),
    # closed
    ("()", C), ("[]", C), ("{}", C), (":[1;2;3]", C), (":{[1 2]}", C), ("f::{x+1}", C),
    ("[1 2 3]", C), ("(1+2)", C), ("{x}(1)", C), (":[a;b;c]", C),
    # nesting
    ("f::{[a];a::(1+[2", I), ("f::{[a];a::(1+[2])}", C), ("{(}", E), ("{[}", E),
    ("f({x}(", I), (":[1;:[2;3]", I), (":[1;:[2;3]]", C), ("[[1 2][3", I),
    # mismatched or stray closers
    ("(1]", E), ("[1)", E), ("{1)", E), (")", E), ("]", E), ("}", E),
    ("1)", E), ("(1))", E), (":[1;2)", E), (":{1]", E), ("(1]\n", E), ("(}", E),
    # list data mode counts brackets like any other
    ("[ ( ]", E), ("[ (", I), ("[1 (2) 3]", C), ("[1 [2 3] 4", I),
    # strings
    ('"abc', I), ('"abc"', C), ('"a[b"', C), ('"a(b"', C), ('"a""b"', C), ('"a""', I),
    ('"a{"', C), ('f("(', I), ('f("(")', C), ('"', I), ('""', C), ('"a\nb', I),
    ('"a\nb"', C),
    # character literals
    ("0c[", C), ("0c(", C), ("0c{", C), ("0c]", C), ("[0c[ 0c]]", C), ("0c\"", C), ("x::0c[", C),
    # comments
    (':"a { b"', C), (':"a ( b" 1', C), ('1 :"x', I), ('1+2 :"unfinished', I), (':"', I),
    ('1 :"x[" ', C), ('(1 :"ok ) bracket"', I), ('(1 :"ok ) bracket")', C),
    ('f::{x :"{"', I), ('f::{x :"{"}', C),
    # .comment("X")
    ('.comment("X")', I), ('.comment("X"); 1', I), ('.comment("X")\nbody (\nX', C),
    ('.comment("X")\nbody\nX\n1+2', C), ('.comment("END")\n[ ( {\n', I),
    ('.comment("END")\n[ ( {\nEND', C), ('.comment("END")\n[ ( {\nEND\n(1', I),
    ('1+.comment("X")abc', I), ('.comment("X") abc X', C),
    # trailing dyad and operators are complete
    ("1+", C), ("x@", C), ("f::", C), ("1+2*", C), ("-", C), ("#", C), ("a,", C), ("1 2 3+", C),
    ("1;", C), ("a::", C), ("+/", C),
    # a trailing newline does not matter
    ("f::{x}\n", C), ("f::{\n", I), ("f::{\nx\n", I), ("f::{\nx\n}\n", C),
    # an unterminated string inside an open bracket is incomplete either way
    ('f("abc', I), ('{"abc', I),
]


class TestCheckComplete(unittest.TestCase):
    def test_table(self):
        for text, want in CASES:
            with self.subTest(text=text):
                self.assertEqual(check_text(text), want)

    def test_tokens_only_form(self):
        # the documented signature works with tokens alone for brackets and strings
        for text, want in [("(1", I), ("(1]", E), ('"abc', I), ("1+", C), ("", C)]:
            with self.subTest(text=text):
                self.assertEqual(check_complete(tokenize(Source("<t>", text))), want)

    def test_tokens_only_cannot_see_comments(self):
        # comments are trivia and .comment bodies are in the text, so these need `source`
        toks = tokenize(Source("<t>", '1 :"x'))
        self.assertEqual(check_complete(toks), C)
        self.assertEqual(check_complete(toks, '1 :"x'), I)

    def test_source_object_or_text(self):
        src = Source("<t>", '1 :"x')
        self.assertEqual(check_complete(tokenize(src), src), I)

    def test_joined_lines_become_complete(self):
        text = "f::{"
        self.assertEqual(check_text(text), I)
        text += "\n x+1"
        self.assertEqual(check_text(text), I)
        text += "\n}"
        self.assertEqual(check_text(text), C)

    def test_never_raises_on_noise(self):
        import random
        rng = random.Random(7)
        alphabet = list('()[]{}:"\\ ;\n+-0c.ab1') + [':[', ':{', ':(', '.comment("X")']
        for _ in range(2000):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 14)))
            self.assertIn(check_text(text), (C, I, E), text)


if __name__ == "__main__":
    unittest.main()
