"""The lexer: Source, token kinds, flags, trivia, the pull API."""
import copy
import os
import pickle
import random
import sys
import unittest

from klongpy.lexer import ERR_CHAR_EOF, Lexer, Source, Tok, tokenize
from klongpy.types import KGChar

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def kg_files():
    for d in ("tests/kgtests", "klongpy/lib", "examples", "tests/golden/fixtures"):
        for dp, _, fns in os.walk(os.path.join(ROOT, d)):
            for fn in sorted(fns):
                if fn.endswith(".kg"):
                    yield os.path.join(dp, fn)


def kv(text):
    """(kind, value) of every token but the final EOF."""
    toks = Lexer(text).toks
    assert toks[-1].kind == "EOF"
    return [(t.kind, t.value) for t in toks[:-1]]


def kinds(text):
    return [k for k, _ in kv(text)]


class TestSource(unittest.TestCase):
    def test_empty(self):
        s = Source("<string>", "")
        self.assertEqual(s.line_col(0), (1, 1))
        self.assertEqual(s.line_starts(), [0])

    def test_no_final_newline(self):
        s = Source("f.kg", "ab\ncd")
        self.assertEqual(s.line_col(0), (1, 1))
        self.assertEqual(s.line_col(1), (1, 2))
        self.assertEqual(s.line_col(2), (1, 3))     # the newline itself ends line 1
        self.assertEqual(s.line_col(3), (2, 1))
        self.assertEqual(s.line_col(5), (2, 3))     # end of text

    def test_final_newline(self):
        s = Source("f.kg", "ab\n")
        self.assertEqual(s.line_col(3), (2, 1))
        self.assertEqual(s.line_starts(), [0, 3])

    def test_offsets_are_clamped(self):
        s = Source("f.kg", "ab\ncd")
        self.assertEqual(s.line_col(-5), (1, 1))
        self.assertEqual(s.line_col(99), (2, 3))

    def test_line_text(self):
        s = Source("f.kg", "ab\ncd\n")
        self.assertEqual(s.line_text(1), "ab")
        self.assertEqual(s.line_text(2), "cd")
        self.assertEqual(s.line_text(3), "")
        self.assertEqual(s.line_text(9), "")

    def test_immutable_equal_hashable(self):
        s = Source("n", "t")
        with self.assertRaises(AttributeError):
            s.text = "u"
        self.assertEqual(s, Source("n", "t"))
        self.assertNotEqual(s, Source("m", "t"))
        self.assertEqual(len({s, Source("n", "t")}), 1)

    def test_copy_and_pickle(self):
        s = Source("n", "a\nb")
        s.line_col(2)
        self.assertIs(copy.deepcopy(s), s)
        self.assertEqual(pickle.loads(pickle.dumps(s)), s)


class TestTokenKinds(unittest.TestCase):
    def test_num(self):
        self.assertEqual(kv("1 23 1.5 1e3 1e-3 1e+3 1."), [("NUM", 1), ("NUM", 23), ("NUM", 1.5), ("NUM", 1000.0),
                                                           ("NUM", 0.001), ("NUM", 1000.0), ("NUM", 1.0)])
        self.assertIsInstance(Lexer("1").toks[0].value, int)
        self.assertIsInstance(Lexer("1e3").toks[0].value, float)

    def test_colon_number_includes_the_colon(self):
        t = Lexer(":123").toks[0]
        self.assertEqual((t.kind, t.value, t.start, t.end), ("NUM", 123, 0, 4))

    def test_char(self):
        self.assertEqual(kv("0cx 0c0 0c;"), [("CHAR", "x"), ("CHAR", "0"), ("CHAR", ";")])
        t = Lexer("0cx").toks[0]
        self.assertIsInstance(t.value, KGChar)
        self.assertEqual(kv("0c "), [("CHAR", " ")])      # any character, a space too
        self.assertEqual(kv("0c\n"), [("CHAR", "\n")])
        self.assertEqual(kv(":0cx"), [("CHAR", "x")])     # ':' before a digit is ignored, as for :123
        self.assertEqual(kv("10c"), [("NUM", 10), ("NAME", "c")])   # 0c is a prefix only at token start

    def test_char_at_end_of_input(self):
        t = Lexer("0c").toks[0]
        self.assertEqual((t.kind, t.value, t.start, t.end), ("ERROR", "0c", 0, 2))
        self.assertIs(t.message, ERR_CHAR_EOF)
        self.assertEqual(kinds("1+0c"), ["NUM", "OP", "ERROR"])

    def test_str(self):
        self.assertEqual(kv('"" "a" "a""b" "a b" """"'), [("STR", ""), ("STR", "a"), ("STR", 'a"b'), ("STR", "a b"),
                                                          ("STR", '"')])
        t = Lexer('"ab"').toks[0]
        self.assertEqual((t.start, t.end, t.unterminated), (0, 4, False))
        self.assertEqual(kv('"a\nb"'), [("STR", "a\nb")])

    def test_unterminated_string(self):
        t = Lexer('"abc').toks[0]
        self.assertEqual((t.kind, t.value, t.end, t.unterminated), ("STR", "abc", 4, True))
        t = Lexer('"abc""').toks[0]                         # ends in an escaped quote, not a closing one
        self.assertEqual((t.value, t.unterminated), ('abc"', True))
        t = Lexer('"').toks[0]
        self.assertEqual((t.value, t.unterminated), ("", True))

    def test_name(self):
        self.assertEqual(kv("a foo a1 a.b .f .5 x"), [("NAME", "a"), ("NAME", "foo"), ("NAME", "a1"), ("NAME", "a.b"),
                                                      ("NAME", ".f"), ("NAME", ".5"), ("NAME", "x")])
        self.assertEqual(kv("my_var"), [("NAME", "my"), ("OP", "_"), ("NAME", "var")])   # _ is the Drop operator

    def test_qname(self):
        self.assertEqual(kv(":foo :a.b :.f :x"), [("QNAME", "foo"), ("QNAME", "a.b"), ("QNAME", ".f"), ("QNAME", "x")])
        t = Lexer(":foo").toks[0]
        self.assertEqual((t.start, t.end), (0, 4))

    def test_op(self):
        self.assertEqual(kv("+ - * % , @ # ' / \\ ~ ! $ & | < > = ? ^ _"),
                         [("OP", c) for c in "+ - * % , @ # ' / \\ ~ ! $ & | < > = ? ^ _".split()])
        self.assertEqual(kv("\\~ \\* \\"), [("OP", "\\~"), ("OP", "\\*"), ("OP", "\\")])

    def test_cop(self):
        self.assertEqual(kv(":: := :# :~ :_ :' :/ :\\ :* :! :& :, :< :? :- :+"),
                         [("COP", c) for c in ":: := :# :~ :_ :' :/ :\\ :* :! :& :, :< :? :- :+".split()])
        self.assertEqual(kv(": 1"), [("COP", ": "), ("NUM", 1)])    # ':' takes any next character
        self.assertEqual(kv(":\n1"), [("COP", ":\n"), ("NUM", 1)])

    def test_colon_at_end_of_input_is_an_op(self):
        self.assertEqual(kv("x:"), [("NAME", "x"), ("OP", ":")])

    def test_punctuation(self):
        self.assertEqual(kinds("[ ] { } ( ) ;"), ["LBRACK", "RBRACK", "LBRACE", "RBRACE", "LPAREN", "RPAREN", "SEMI"])
        self.assertEqual(kv(":[ :| :{ :("), [("COND", ":["), ("ELIF", ":|"), ("DICT", ":{"), ("CALLP", ":(")])

    def test_newline_and_eof(self):
        toks = Lexer("a\nb\n").toks
        self.assertEqual([t.kind for t in toks], ["NAME", "NEWLINE", "NAME", "NEWLINE", "EOF"])
        self.assertEqual((toks[-1].start, toks[-1].end), (4, 4))
        self.assertEqual([t.kind for t in Lexer("").toks], ["EOF"])
        self.assertEqual([t.kind for t in Lexer("  \t ").toks], ["EOF"])
        self.assertEqual(Lexer("  \t ").toks[0].start, 4)

    def test_other_line_breaks_are_whitespace(self):
        self.assertEqual(kinds("a\r\nb"), ["NAME", "NEWLINE", "NAME"])
        self.assertEqual(kinds("a\rb\x0cc\u2028d\xa0e"), ["NAME"] * 5)


class TestLexingRules(unittest.TestCase):
    def test_negative_numbers_are_not_fused(self):
        self.assertEqual(kv("-1"), [("OP", "-"), ("NUM", 1)])
        self.assertEqual(kv("[1 -2]"), [("LBRACK", "["), ("NUM", 1), ("OP", "-"), ("NUM", 2), ("RBRACK", "]")])
        self.assertEqual(kv("1-2"), [("NUM", 1), ("OP", "-"), ("NUM", 2)])

    def test_adjacency_flag(self):
        toks = Lexer("f(1) g (2)\n-3 [;").toks
        self.assertEqual([(t.value, t.ws) for t in toks[:-1]],
                         [("f", False), ("(", False), (1, False), (")", False), ("g", True), ("(", True), (2, False),
                          (")", False), ("\n", False), ("-", False), (3, False), ("[", True), (";", False)])
        self.assertTrue(Lexer(" a").toks[0].ws)
        self.assertFalse(Lexer("a").toks[0].ws)
        self.assertTrue(Lexer("a ").toks[1].ws)      # EOF after whitespace

    def test_a_comment_counts_as_whitespace(self):
        toks = Lexer('a:"c"b').toks
        self.assertEqual([(t.kind, t.ws) for t in toks[:-1]], [("NAME", False), ("NAME", True)])

    def test_comments_are_trivia(self):
        lx = Lexer('a :"one" + :"two ""x"" " b')
        self.assertEqual([t.kind for t in lx.toks], ["NAME", "OP", "NAME", "EOF"])
        self.assertEqual([(t.kind, t.value, t.unterminated) for t in lx.trivia],
                         [("COMMENT", ':"one"', False), ("COMMENT", ':"two ""x"" "', False)])
        self.assertEqual((lx.trivia[0].start, lx.trivia[0].end), (2, 8))

    def test_unterminated_comment(self):
        lx = Lexer('a :"runs to the end\nb')
        self.assertEqual([t.kind for t in lx.toks], ["NAME", "EOF"])
        self.assertEqual(len(lx.trivia), 1)
        self.assertTrue(lx.trivia[0].unterminated)
        self.assertEqual(lx.trivia[0].end, 21)

    def test_comment_after_an_operator_colon(self):
        # '::' is read first, so the '"' starts a string; ':::"' leaves a ':"' that is a comment
        self.assertEqual(kv('a::"s"'), [("NAME", "a"), ("COP", "::"), ("STR", "s")])
        lx = Lexer('a:::"c"')
        self.assertEqual([t.kind for t in lx.toks], ["NAME", "COP", "EOF"])
        self.assertEqual(len(lx.trivia), 1)

    def test_nl_comment_flag(self):
        toks = Lexer('a\nb :"c"\nd :"c" :"d"  \n\ne').toks
        nls = [t.nl_comment for t in toks if t.kind == "NEWLINE"]
        self.assertEqual(nls, [False, True, True, False])
        self.assertFalse(any(t.nl_comment for t in toks if t.kind != "NEWLINE"))

    def test_error_tokens_for_bad_numbers(self):
        for text in ("1.2.3", "3e", "1e+", "1ee5", "1e-)"):
            with self.subTest(text=text):
                toks = Lexer(text).toks
                self.assertEqual([t.kind for t in toks], ["ERROR", "EOF"])
                self.assertEqual(toks[0].value, text)
                self.assertEqual((toks[0].start, toks[0].end), (0, len(text)))
                self.assertIsInstance(toks[0].message, str)
        t = Lexer(":1.2.3").toks[0]
        self.assertEqual((t.kind, t.value, t.end), ("ERROR", "1.2.3", 6))

    def test_exponent_sign_takes_the_next_character(self):
        # legacy read_num: after e+ or e- one more character is consumed unseen
        self.assertEqual(kv("1e-5x"), [("NUM", 1e-5), ("NAME", "x")])
        self.assertEqual(kinds("1e-x 2"), ["ERROR", "NUM"])

    def test_dot_five_is_a_name(self):
        self.assertEqual(kv(".5"), [("NAME", ".5")])
        self.assertEqual(kv("0.5"), [("NUM", 0.5)])

    def test_non_ascii_letters_are_name_characters(self):
        self.assertEqual(kv("é"), [("NAME", "é")])
        self.assertEqual(kv("naïve.x1"), [("NAME", "naïve.x1")])
        self.assertEqual(kv("aλ::1"), [("NAME", "aλ"), ("COP", "::"), ("NUM", 1)])
        self.assertEqual(kv(":λx"), [("QNAME", "λx")])
        self.assertEqual(kv("日本"), [("NAME", "日本")])

    def test_operator_glyphs(self):
        self.assertEqual(kv("∇f"), [("OP", "∇"), ("NAME", "f")])
        self.assertEqual(kv("x∂y"), [("NAME", "x"), ("OP", "∂"), ("NAME", "y")])
        self.assertEqual(kv("˙"), [("OP", "˙")])
        self.assertEqual(kv(":∇"), [("COP", ":∇")])

    def test_non_ascii_digits(self):
        self.assertEqual(kv("٣"), [("NUM", 3)])                 # a Unicode decimal digit: int() accepts it
        self.assertEqual(kinds("²"), ["ERROR"])                 # numeric but not a digit for int()
        self.assertEqual(kv("1٣"), [("NUM", 13)])
        self.assertEqual(kinds(":²"), ["ERROR"])


class TestLexerApi(unittest.TestCase):
    def test_next_and_peek(self):
        lx = Lexer("a+1")
        self.assertEqual(lx.peek().value, "a")
        self.assertEqual(lx.peek(1).value, "+")
        self.assertEqual(lx.peek(99).kind, "EOF")
        self.assertEqual(lx.next().value, "a")
        self.assertEqual(lx.peek().value, "+")
        self.assertEqual([lx.next().kind for _ in range(4)], ["OP", "NUM", "EOF", "EOF"])
        self.assertEqual(lx.peek().kind, "EOF")

    def test_accepts_text_or_source(self):
        self.assertEqual(Lexer("a").source, Source("<string>", "a"))
        src = Source("f.kg", "a")
        self.assertIs(Lexer(src).source, src)

    def test_start_position(self):
        lx = Lexer("a b c", 2)
        self.assertEqual([t.value for t in lx.toks[:-1]], ["b", "c"])
        self.assertFalse(lx.toks[0].ws)
        self.assertEqual(lx.offset(), 2)

    def test_offset_and_end(self):
        lx = Lexer("ab  cd")
        self.assertEqual(lx.offset(), 0)
        lx.next()
        self.assertEqual((lx.offset(), lx.end()), (2, 2))      # the whitespace is not consumed
        lx.next()
        self.assertEqual(lx.offset(), 6)

    def test_seek_skips_raw_text(self):
        text = 'a "never closed  b c'
        lx = Lexer(text)
        self.assertEqual(lx.next().value, "a")
        lx.seek(text.index("b"))
        self.assertEqual([t.value for t in lx.toks[lx.pos:-1]], ["b", "c"])
        self.assertEqual(lx.offset(), text.index("b"))
        self.assertEqual(lx.next().value, "b")

    def test_seek_gives_the_tokens_of_a_fresh_lexer(self):
        text = 'a b :"c1" c\n :"c2"\n d "x" e [1 2] f :"c3"'
        for off in range(len(text) + 1):
            for consumed in (0, 1, 2):
                with self.subTest(off=off, consumed=consumed):
                    lx = Lexer(text)
                    for _ in range(consumed):
                        lx.next()
                    if off < lx.offset():
                        continue
                    lo = lx.offset()
                    lx.seek(off)
                    fresh = Lexer(text, off)
                    self.assertEqual(lx.toks[lx.pos:], fresh.toks)
                    self.assertEqual([c for c in lx.trivia if c.start >= lo], fresh.trivia)
                    self.assertEqual(lx.offset(), off)

    def test_seek_recomputes_the_adjacency_of_the_first_token(self):
        lx = Lexer("a  b")
        lx.next()
        lx.seek(3)
        self.assertFalse(lx.peek().ws)
        lx = Lexer("a  b")
        lx.next()
        lx.seek(2)
        self.assertTrue(lx.peek().ws)

    def test_tok_is_a_named_tuple(self):
        t = Lexer("ab").toks[0]
        self.assertIsInstance(t, Tok)
        self.assertEqual(t, ("NAME", 0, 2, "ab", False, False, False, None))
        self.assertEqual(Tok("NAME", 0, 2, "ab"), t)


class TestTokenize(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual([t.kind for t in tokenize("a::1")], ["NAME", "COP", "NUM", "EOF"])
        self.assertEqual([t.kind for t in tokenize(Source("f", "a"))], ["NAME", "EOF"])

    def test_tokenize_skips_a_comment_body(self):
        text = '1\n.comment("END")\nnot "tokens ( at all\nEND\n2'
        toks = tokenize(text)
        self.assertEqual([t.value for t in toks if t.kind in ("NUM", "NAME", "STR")], [1, ".comment", "END", 2])
        # the marker is repeated in the legacy way: the body ends at the last overlapping repeat
        text = '.comment("**")\nx\n*****y'
        self.assertEqual([t.value for t in tokenize(text) if t.kind == "NAME"], [".comment", "y"])

    def test_tokenize_without_the_end_marker(self):
        toks = tokenize('.comment("END")\nfoo bar')
        self.assertEqual([t.value for t in toks if t.kind == "NAME"], [".comment", "foo", "bar"])

    def test_tokenize_only_the_call_form_is_a_comment(self):
        toks = tokenize('.comment ("END") a END b')
        self.assertEqual([t.value for t in toks if t.kind == "NAME"], [".comment", "a", "END", "b"])


class TestLexerTotality(unittest.TestCase):
    def check_cover(self, text):
        """Tokens and comments are in order, do not overlap, and what lies between them is whitespace."""
        lx = Lexer(text)
        self.assertEqual(lx.toks[-1].kind, "EOF")
        self.assertEqual(sum(1 for t in lx.toks if t.kind == "EOF"), 1)
        items = sorted(lx.toks[:-1] + lx.trivia, key=lambda t: t.start)
        pos = 0
        rebuilt = []
        for t in items:
            self.assertGreaterEqual(t.start, pos)
            self.assertGreater(t.end, t.start)
            gap = text[pos:t.start]
            self.assertTrue(gap == "" or (gap.isspace() and "\n" not in gap), repr(gap))
            rebuilt.append(gap)
            rebuilt.append(text[t.start:t.end])
            pos = t.end
        tail = text[pos:]
        self.assertTrue(tail == "" or (tail.isspace() and "\n" not in tail))
        rebuilt.append(tail)
        self.assertEqual("".join(rebuilt), text)
        prev_end = 0
        for t in lx.toks:
            self.assertEqual(t.ws, t.start > prev_end, (t, prev_end))
            prev_end = t.end

    def test_every_kg_file(self):
        n = 0
        for path in kg_files():
            with open(path, newline="") as f:
                text = f.read()
            with self.subTest(path=os.path.relpath(path, ROOT)):
                self.check_cover(text)
                tokenize(text)
            n += 1
        self.assertGreater(n, 150)

    def test_never_raises_on_random_text(self):
        rnd = random.Random(30)
        alphabet = list('ab x1.:;()[]{}"\'\\~*+-/@0c9e \n\t\r') + ["é", "∇", "²", "٣", "\u2028", '""', ':"', "0c", "1e-"]
        for _ in range(3000):
            text = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 30)))
            try:
                self.check_cover(text)
                tokenize(text)
            except Exception:
                print("input:", repr(text), file=sys.stderr)
                raise

    def test_never_raises_on_random_code_points(self):
        rnd = random.Random(31)
        for _ in range(500):
            text = "".join(chr(rnd.choice((rnd.randrange(0, 256), rnd.randrange(0, 0x3000), rnd.randrange(0, 0x110000))))
                           for _ in range(rnd.randint(0, 20)))
            text = "".join(c for c in text if not 0xD800 <= ord(c) <= 0xDFFF)
            self.check_cover(text)
            tokenize(text)


if __name__ == "__main__":
    unittest.main()
