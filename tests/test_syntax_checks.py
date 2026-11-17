"""
Syntax checks: what the parser used to accept without a word.

At strict level 0 (the default) each condition is a KlongSyntaxWarning and the
program means what it always meant; at level 1 and above it is a
KlongSyntaxError. A few things changed at every level: whitespace before the
first ';' of a conditional is allowed, an adverb that does not follow its verb
is a syntax error, and a dictionary literal whose entries are not pairs is a
syntax error instead of a crash of the reader.
"""
import os
import subprocess
import sys
import tempfile
import unittest
import warnings

import numpy as np

from klongpy import KlongInterpreter, KlongSyntaxError, KlongSyntaxWarning
from klongpy.dump import dump_text
from klongpy.errors import UnexpectedChar
from klongpy.lexer import Lexer, Source
from klongpy.types import KGFn
from tests.backend_compat import to_numpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def interp(level=0, setup=()):
    k = KlongInterpreter()
    for s in setup:
        k(s)
    k._context._strict_mode = level
    return k


def run(text, level=0, setup=()):
    """(result, [warnings]) of running text on a fresh interpreter, every warning recorded."""
    k = interp(level, setup)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = k(text)
    return r, [w.message for w in caught if isinstance(w.message, KlongSyntaxWarning)]


def same(a, b):
    a, b = to_numpy(a), to_numpy(b)  # a tensor on the torch backend
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


# input, setup, code of the warning, (line, column), check of the level 0 result
TABLE = [
    ("f (1)", ("f::{x}",), "trailing-input", (1, 3), lambda r: isinstance(r, KGFn)),
    ("1 2 3+4 5", (), "trailing-input", (1, 3), lambda r: r == 1),
    ("{x}(1)(2)", (), "trailing-input", (1, 7), lambda r: r == 1),
    ('1+1 "x"', (), "trailing-input", (1, 5), lambda r: r == 2),
    ("q[1]", ("q::[7 8 9]",), "trailing-input", (1, 2), lambda r: same(r, [7, 8, 9])),
    ("1+1 )", (), "stray-closer", (1, 5), lambda r: r == 2),
    ("1+1 ]", (), "stray-closer", (1, 5), lambda r: r == 2),
    ("1+1 }", (), "stray-closer", (1, 5), lambda r: r == 2),
    ("1;)", (), "stray-closer", (1, 3), lambda r: r == ")"),
    ("x@", (), "missing-operand", (1, 2), lambda r: True),
    ("f::", (), "missing-operand", (1, 2), lambda r: True),
    ("1 f", ("f::{x+y}",), "missing-operand", (1, 3), lambda r: True),
    ('"abc', (), "unterminated-string", (1, 1), lambda r: r == "abc"),
    ("[1 2 3", (), "unterminated-list", (1, 1), lambda r: same(r, [1, 2, 3])),
    ("[;1;2", (), "unterminated-list", (1, 1), lambda r: same(r, [1, 2])),
    (":{[1 2]", (), "unterminated-dict", (1, 1), lambda r: r == {1: 2}),
    (':" never ends 1+1', (), "unterminated-comment", (1, 1), lambda r: r is None),
    ('1+1 :" comment ends', (), "unterminated-comment", (1, 5), lambda r: r == 2),
    ('x::1 :"abc', (), "unterminated-comment", (1, 6), lambda r: r == 1),
    ("a::1\nb::2 3", (), "trailing-input", (2, 6), lambda r: r == 2),
]


class TestLevel0Warns(unittest.TestCase):
    def test_one_warning_and_the_legacy_result(self):
        for text, setup, code, pos, check in TABLE:
            with self.subTest(text=text):
                r, ws = run(text, 0, setup)
                self.assertEqual([w.code for w in ws], [code])
                w = ws[0]
                self.assertEqual(w.position()[1:], pos)
                self.assertEqual(w.source.text, text)
                self.assertTrue(check(r), repr(r))

    def test_level_1_and_2_raise(self):
        for level in (1, 2):
            for text, setup, code, pos, _ in TABLE:
                with self.subTest(text=text, level=level):
                    k = interp(level, setup)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        with self.assertRaises(KlongSyntaxError) as cm:
                            k(text)
                    self.assertEqual(caught, [])
                    e = cm.exception
                    self.assertEqual(e.position()[1:], pos)
                    self.assertEqual(e.kind, "syntax")

    def test_warning_and_error_share_the_message(self):
        for text, setup, _, _, _ in TABLE:
            with self.subTest(text=text):
                _, ws = run(text, 0, setup)
                with self.assertRaises(KlongSyntaxError) as cm:
                    interp(1, setup)(text)
                # the warning adds what happens to the input at level 0
                self.assertTrue(str(ws[0]).startswith(str(cm.exception)), (str(ws[0]), str(cm.exception)))
                self.assertEqual(ws[0].hint, cm.exception.hint)

    def test_messages(self):
        want = {
            "1 2 3+4 5": "unexpected '2' after the end of the expression; the rest of the input is not evaluated",
            "1+1 )": "unexpected ')': nothing is open that it could close; it is ignored",
            "1+1 ) 2": "unexpected ')': nothing is open that it could close; the rest of the input is not evaluated",
            "1;)": "unexpected ')'; it is read as the string \")\"",
            "x@": "missing right operand for @",
            "f::": "missing right operand for ::",
            "+/": "missing operand for +/",
            "1+/": "missing right operand for +/",
            '"abc': "unterminated string: the closing \" is missing; it runs to the end of the input",
            "[1 2 3": "unterminated list: the closing ] is missing; it runs to the end of the input",
            ":{[1 2]": "unterminated dictionary: the closing } is missing; it runs to the end of the input",
            ':"abc': "unterminated comment: the closing \" is missing; it runs to the end of the input",
        }
        for text, msg in want.items():
            with self.subTest(text=text):
                # prog: some of these fail when evaluated, as they always did
                self.assertEqual([str(w) for w in interp(0).prog(text).warnings], [msg])

    def test_hints(self):
        for text, setup, hint in [
            ("f (1)", ("f::{x}",), "no space may separate a function from its argument list: write f(1)"),
            ("{x} (1)", (), "no space may separate a function from its argument list: write {...}(1)"),
            ("q[1]", ("q::[7 8 9]",), "use q@1 to index"),
            ("q[0 1]", ("q::[7 8 9]",), "use q@[0 1] to index"),
            ("q[i]", ("q::[7 8 9]", "i::1"), "use q@i to index"),
            ("{x}(1)(2)", (), "only a name or a function literal can be followed by an argument list; "
                              "assign the value to a name first"),
            ("1 2 3", (), "a list is written in brackets, [1 2 3]; statements are separated by ;"),
        ]:
            with self.subTest(text=text):
                _, ws = run(text, 0, setup)
                self.assertEqual([w.hint for w in ws], [hint])

    def test_arguments_without_a_separator(self):
        # f(1 2) passed two arguments without a word; f(q[0]) is how a Python user indexes
        k = interp(0, ("f::{x+y}", "q::[5 6]", "g::{x}"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(k("f(1 2)"), 3)
            self.assertTrue(same(k("g(q[0])"), [5, 6]))
            self.assertEqual(k("f(1;\n2)"), 3)
            self.assertEqual(k("f(1\n;2)"), 3)
        ws = [w.message for w in caught]
        self.assertEqual([(w.code, w.hint) for w in ws], [("trailing-input", "arguments are separated by ;"),
                                                          ("trailing-input", "use q@0 to index")])
        self.assertEqual(str(ws[0]), "unexpected '2' after the end of the argument; it is passed as one more argument")
        k._context._strict_mode = 1
        for text in ("f(1 2)", "g(q[0])", "{x}(1 [2])"):
            with self.assertRaises(KlongSyntaxError):
                k(text)
        self.assertEqual(k("f(1;\n2)"), 3)

    def test_index_hint_on_the_bracket_errors(self):
        for text in ("(q[0])", "{q[0]}"):
            for level in (0, 1):
                with self.assertRaises(UnexpectedChar) as cm:
                    interp(level, ("q::[5 6]",))(text)
                self.assertEqual(cm.exception.hint, "use q@0 to index")
        with self.assertRaises(UnexpectedChar) as cm:
            interp(0)("(1+2")
        self.assertIsNone(cm.exception.hint)

    def test_nested_unterminated_lists_warn_once(self):
        r, ws = run("[1 [2 3")
        self.assertEqual([w.code for w in ws], ["unterminated-list"])
        self.assertEqual(ws[0].span, (0, 1))

    def test_two_conditions_two_warnings_in_source_order(self):
        _, ws = run('1+1 "x')
        self.assertEqual([w.code for w in ws], ["trailing-input", "unterminated-string"])
        _, ws = run('[1 2 :"c')
        self.assertEqual([w.code for w in ws], ["unterminated-list", "unterminated-comment"])

    def test_a_list_that_swallowed_its_bracket(self):
        # the legacy quirk: after a comment and a newline the next lexeme is data, even "]"
        r, ws = run('[1 2 :"c"\n]')
        self.assertEqual([w.code for w in ws], ["unterminated-list"])
        self.assertEqual(r[2], "]")
        # at level 1 the newline is an ordinary one, so the list is closed
        r, ws = run('[1 2 :"c"\n]', 1)
        self.assertTrue(same(r, [1, 2]))
        self.assertEqual(ws, [])

    def test_missing_operand_before_a_separator(self):
        # the ';' was taken as the operand (the string ";")
        for text, codes in (("1+;", ["missing-operand"]), ("f::;", ["missing-operand"]), ("{-;}", ["missing-operand"]),
                            ("1+;2", ["missing-operand", "trailing-input"])):      # and then the 2 is left over
            with self.subTest(text=text):
                self.assertEqual([w.code for w in interp(0).prog(text).warnings], codes)
                with self.assertRaises(KlongSyntaxError) as cm:
                    interp(1)(text)
                self.assertIn("missing", str(cm.exception))
        self.assertEqual(run("f::;")[0], ";")

    def test_missing_operand_that_fails_is_the_syntax_error(self):
        # the error half: "1+" and "-" alone fail at level 0 too, as the syntax error
        for text, msg in (("1+", "missing right operand for +"), ("-", "missing operand for -"),
                          ("1+;2", "missing right operand for +")):
            for level in (0, 1):
                with self.subTest(text=text, level=level):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with self.assertRaises(KlongSyntaxError) as cm:
                            interp(level)(text)
                    self.assertEqual(str(cm.exception), msg)

    def test_eval_and_load_go_through_the_checks(self):
        r, ws = run('.E("1+1 2+2")')
        self.assertEqual(r, 2)
        self.assertEqual([(w.code, w.source.text) for w in ws], [("trailing-input", "1+1 2+2")])
        with self.assertRaises(KlongSyntaxError):
            interp(1)('.E("1+1 2+2")')
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.kg")
            with open(path, "w") as f:
                f.write("a::1\nb::2 3 4+5 6 7 )\nc::3\n")
            k = interp(0)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                k(f'.l("{path}")')
            self.assertEqual(len(caught), 1)
            w = caught[0]
            self.assertEqual((w.filename, w.lineno), (path, 2))
            self.assertEqual(w.message.position(), (path, 2, 6))
            self.assertEqual(k("b"), 2)
            with self.assertRaises(KlongSyntaxError) as cm:
                interp(1)(f'.l("{path}")')
            self.assertEqual(cm.exception.position(), (path, 2, 6))

    def test_exec_warns_too(self):
        k = interp(0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(k.exec("1;2 3"), [1, 2])
        self.assertEqual([w.message.code for w in caught], ["trailing-input"])

    def test_prog_returns_the_warnings_and_issues_none(self):
        k = interp(0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            p = k.prog("1 2")
        self.assertEqual(caught, [])
        self.assertEqual([w.code for w in p.warnings], ["trailing-input"])
        self.assertEqual(p[0], 1)       # the offset where parsing stopped, as before
        self.assertEqual(k.prog("1+2").warnings, ())

    def test_legacy_parser_is_unchanged(self):
        for level in (0, 1):
            k = KlongInterpreter(parser="legacy")
            k._context._strict_mode = level
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertEqual(k("1 2 3+4 5"), 1)
                self.assertEqual(k('"abc'), "abc")
                with self.assertRaises(UnexpectedChar):
                    k(":[1 ;2;3]")
            self.assertEqual(caught, [])


class TestNoWarning(unittest.TestCase):
    def test_trailing_separators_and_comments(self):
        for text, want in [("1+1;", 2), ("1+1;;", 2), ('1 :"c" ', 1), ("1+1\n", 2), ("1+1  \n\n", 2), ('1+1 :"c"\n', 2),
                           ('a::1\n\n :"c"\n', 1), ("", None), ("  ", None), (";", None), ("1;2", 2),
                           ('.comment("END")\n "unterminated\nEND\n1', 1), ("{x}\n(1)", 1), ("[1 2 3]", None),
                           ("f::{[a];a::x;a}", None), (':{[1 2]}', None), ('"a""b"', 'a"b'), ("x::[1 2];x@0", None)]:
            for level in (0, 1):
                with self.subTest(text=text, level=level):
                    k = interp(level)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        r = k(text)
                    self.assertEqual(caught, [])
                    if want is not None:
                        self.assertEqual(r, want)

    def test_brackets_that_already_failed_still_fail_the_same_way(self):
        for text in ("(1+2", "{x+1", ":[1;2;3", "f(1", "f(1 )"):
            for level in (0, 1):
                with self.subTest(text=text, level=level):
                    with self.assertRaises(UnexpectedChar):
                        interp(level)(text)


class TestNewlineAfterComment(unittest.TestCase):
    def test_in_a_body(self):
        text = '{x :"c"\n+2}(1)'
        self.assertEqual(run(text, 0), (2, []))         # legacy: the newline is a ';'
        self.assertEqual(run(text, 1), (3, []))         # an ordinary newline
        self.assertEqual(run(text, 2), (3, []))
        self.assertEqual(run("{x\n+2}(1)", 0)[0], 3)
        self.assertEqual(run("{x\n+2}(1)", 1)[0], 3)

    def test_in_a_conditional_and_arguments(self):
        text = ':[1 :"c"\n;2;3]'
        with self.assertRaises(UnexpectedChar):
            interp(0)(text)                             # legacy: not accepted where ';' is expected
        self.assertEqual(interp(1)(text), 2)
        text = 'f(1 :"c"\n;2)'
        self.assertEqual(dump_text(text), "(proj (sym f) (num 1) _ (num 2))\n")
        k = interp(1, ("f::{x+y}",))
        self.assertEqual(k(text), 3)

    def test_in_a_list_it_was_skipped_already(self):
        for level in (0, 1):
            self.assertTrue(same(interp(level)('[1 2 :"c"\n3]'), [1, 2, 3]))

    def test_lexer_switch(self):
        text = '1 :"c"\n2'
        self.assertTrue(Lexer(text).toks[1].nl_comment)
        lx = Lexer(text, 0, False)
        self.assertFalse(lx.toks[1].nl_comment)
        lx.seek(1)
        self.assertFalse(any(t.nl_comment for t in lx.toks))


class TestConditionalWhitespace(unittest.TestCase):
    def test_accepted_at_every_level(self):
        for text, want in [(":[ 1 ; 2 ; 3 ]", 2), (":[1 ;2;3]", 2), (":[0 ;2;3]", 3), (":[1\n;2;3]", 2),
                           (":[1\n ;\n2\n;\n3\n]", 2), (":[1;2 ;3]", 2), (":[1;2;3 ]", 2),
                           (":[0 ;1:|1 ;2;3]", 2), (":[0\n;1\n:|0\n;2\n;3]", 3), ("{:[x ;1;2]}(0)", 2)]:
            for level in (0, 1, 2):
                with self.subTest(text=text, level=level):
                    self.assertEqual(run(text, level), (want, []))

    def test_what_is_still_an_error(self):
        for text in (":[1 2;3]", ":[1]", ":[1;2 3]"):
            for level in (0, 1):
                with self.subTest(text=text, level=level):
                    with self.assertRaises(UnexpectedChar):
                        interp(level)(text)


class TestAdverbNotAfterVerb(unittest.TestCase):
    def test_spaced_adverb_is_a_syntax_error_with_a_hint(self):
        for text, col in [("{x} '[1 2]", 5), ("f '[1 2]", 3), ("+ /[1 2]", None), ("[1 2] /x", 7), ("a :~b", 3)]:
            for level in (0, 1):
                with self.subTest(text=text, level=level):
                    if col is None:
                        continue    # "+ /x": the adverb is where an operand is expected, not an operator
                    with self.assertRaises(KlongSyntaxError) as cm:
                        interp(level)(text)
                    e = cm.exception
                    self.assertIn("must directly follow its verb", str(e))
                    self.assertIn("no space", e.hint)
                    self.assertEqual(e.position()[1:], (1, col))

    def test_adverb_after_something_that_is_not_a_verb(self):
        for text in ("(f)'[1 2]", "1'2", "[1 2]/x"):
            with self.subTest(text=text):
                with self.assertRaises(KlongSyntaxError) as cm:
                    interp(0)(text)
                self.assertIn("does not follow a verb", str(cm.exception))
                self.assertIsNotNone(cm.exception.hint)

    def test_it_is_raised_when_parsed_not_when_run(self):
        with self.assertRaises(KlongSyntaxError):
            interp(0)("f::{x '[1 2]}")

    def test_adverbs_in_their_place_are_untouched(self):
        self.assertEqual(run("+/[1 2 3]"), (6, []))
        self.assertTrue(same(run("{x+1}'[1 2]")[0], [2, 3]))
        self.assertTrue(same(run("[1 2],'[3 4]")[0], [[1, 3], [2, 4]]))
        self.assertTrue(same(run("[1 2]{x+y}'[3 4]")[0], [4, 6]))


class TestDictionaryEntries(unittest.TestCase):
    def test_not_pairs_is_a_syntax_error_at_every_level(self):
        # (":{;1;2}" was one of these until it became the evaluated literal)
        for text, also in [(":{ ;1;2}", IndexError), (":{1 2", TypeError), (":{[1]}", IndexError), (":{1}", TypeError),
                           (":{[[1 2] 3]}", TypeError), ("[1 :{[1]}]", IndexError)]:
            for level in (0, 1):
                with self.subTest(text=text, level=level):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with self.assertRaises(KlongSyntaxError) as cm:
                            interp(level)(text)
                    e = cm.exception
                    if level == 0:
                        self.assertIsInstance(e, also)      # what the reader raised before
                        self.assertIn("dictionary", str(e))
                        self.assertIsNotNone(e.span)
                        self.assertIsNotNone(e.hint)

    def test_goldens_of_the_two_reader_crashes(self):
        self.assertEqual(dump_text(":{ ;1;2}"), "(error KlongSyntaxError)\n")
        self.assertEqual(dump_text(":{;1;2}"), "(edict (num 1) (num 2))\n")     
        self.assertEqual(dump_text(":{1 2"), "(error KlongSyntaxError)\n")


class TestWarningObject(unittest.TestCase):
    def test_fields_and_render(self):
        k = interp(0, ("q::[7 8 9]",))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            k(Source("demo.kg", "a::1\nq[1]\n"))
        self.assertEqual(len(caught), 1)
        rec = caught[0]
        w = rec.message
        self.assertIs(rec.category, KlongSyntaxWarning)
        self.assertEqual((rec.filename, rec.lineno), ("demo.kg", 2))
        self.assertEqual(w.span, (6, 7))
        self.assertEqual(w.position(), ("demo.kg", 2, 2))
        self.assertEqual(w.code, "trailing-input")
        self.assertEqual(w.message, str(w))
        self.assertEqual(w.render(), "demo.kg:2:2: warning: unexpected '[' after the end of the expression; "
                                     "the rest of the input is not evaluated\n"
                                     "  q[1]\n"
                                     "   ^\n"
                                     "  hint: use q@1 to index")
        self.assertTrue(w.render(color=True).startswith("\x1b[33m"))

    def test_error_render_at_level_1(self):
        k = interp(1, ("q::[7 8 9]",))
        with self.assertRaises(KlongSyntaxError) as cm:
            k("q[1]")
        self.assertEqual(cm.exception.render(), "<string>:1:2: syntax error: unexpected '[' after the end of the expression\n"
                                                "  q[1]\n"
                                                "   ^\n"
                                                "  hint: use q@1 to index")

    def test_is_exported(self):
        import klongpy
        import klongpy.errors
        self.assertIs(klongpy.KlongSyntaxWarning, klongpy.errors.KlongSyntaxWarning)
        self.assertTrue(issubclass(KlongSyntaxWarning, Warning))
        self.assertFalse(issubclass(KlongSyntaxWarning, Exception) and issubclass(KlongSyntaxWarning, KlongSyntaxError))


class TestWarningsAndTheParseCache(unittest.TestCase):
    """
    The decision: syntax warnings belong to the program text. They are issued
    each time the text is run through __call__, whether the tree comes from
    the parse cache or not, and never when a function defined by that text is
    called. Python's warning filters then decide (by default: shown once per
    location).
    """

    def test_issued_on_every_run_of_the_text_cached_or_not(self):
        k = interp(0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(k("1 2"), 1)
            self.assertEqual(k("1 2"), 1)       # from the parse cache
            self.assertEqual(len(k._parse_cache), 1)
            k._parse_cache.clear()
            self.assertEqual(k("1 2"), 1)
        self.assertEqual(len(caught), 3)

    def test_default_filter_shows_a_location_once(self):
        k = interp(0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            for _ in range(3):
                k("1 2")
            k("1 2 ")       # another text, same message and line: another source, shown again
        self.assertEqual(len(caught), 1 + 1)

    def test_not_issued_when_the_function_is_called(self):
        k = interp(0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            k('f::{x+1};g::{f(x)} "junk"')
            self.assertEqual(len(caught), 1)
            for i in range(5):
                self.assertEqual(k["g"](i), i + 1)
            self.assertEqual(k("g'[1 2 3]").tolist(), [2, 3, 4])
        self.assertEqual(len(caught), 1)

    def test_error_filter_raises_every_time_and_nothing_runs(self):
        k = interp(0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", KlongSyntaxWarning)
            for _ in range(2):
                with self.assertRaises(KlongSyntaxWarning) as cm:
                    k("a::5 6")
                self.assertEqual(cm.exception.code, "trailing-input")
            with self.assertRaises(KeyError):
                k["a"]              # the statement before the junk did not run
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(k("a::5 6"), 5)
        self.assertEqual(len(caught), 1)

    def test_the_level_is_part_of_the_cache_key(self):
        k = interp(0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(k("1 2"), 1)
            k._context._strict_mode = 1
            with self.assertRaises(KlongSyntaxError):
                k("1 2")
            with self.assertRaises(KlongSyntaxError):
                k("1 2")
            k._context._strict_mode = 0
            self.assertEqual(k("1 2"), 1)
        self.assertEqual(len(caught), 2)

    def test_strict_and_unsafe_switch_the_checks(self):
        k = KlongInterpreter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(k("1 2"), 1)
            k(".strict()")
            with self.assertRaises(KlongSyntaxError):
                k("1 2")
            k(".unsafe()")
            self.assertEqual(k("1 2"), 1)
        self.assertEqual(len(caught), 2)

    def test_command_line_W_error(self):
        code = ("from klongpy import KlongInterpreter\n"
                "k = KlongInterpreter()\n"
                "print(k('1+1;'))\n"
                "k('1 2')\n")
        r = subprocess.run([sys.executable, "-W", "error", "-c", code], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.stdout.strip(), "2")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("KlongSyntaxWarning: unexpected '2' after the end of the expression", r.stderr)

    def test_command_line_W_with_the_category_is_a_python_limit(self):
        # CPython resolves the category of a -W option (and of PYTHONWARNINGS) while it
        # starts, before site-packages can be imported, and klongpy needs numpy: the
        # option is ignored with a message. Naming the category works wherever the
        # filter is set after start-up: warnings.simplefilter("error", KlongSyntaxWarning)
        # (see the tests above) and pytest's own -W option.
        code = "from klongpy import KlongInterpreter\nKlongInterpreter()('1 2')\n"
        r = subprocess.run([sys.executable, "-W", "error::klongpy.KlongSyntaxWarning", "-c", code],
                           capture_output=True, text=True, cwd=ROOT)
        if "Invalid -W option ignored" in r.stderr:
            self.assertEqual(r.returncode, 0)
        else:
            self.assertNotEqual(r.returncode, 0)    # a Python that can: then it must raise

    def test_default_python_display(self):
        code = ("from klongpy import KlongInterpreter\n"
                "k = KlongInterpreter()\n"
                "for i in range(3): k('1 2')\n")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stderr.count("KlongSyntaxWarning"), 1)
        self.assertIn("<string>:1: KlongSyntaxWarning: unexpected '2'", r.stderr)


class TestRepositorySourcesAreClean(unittest.TestCase):
    """Step 7: no .kg file of the repository trips a check, at level 0 or 1."""

    def test_no_warning_and_no_level_1_error(self):
        n = 0
        for d in ("tests/kgtests", "klongpy/lib", "examples"):
            for dp, _, fns in os.walk(os.path.join(ROOT, d)):
                for fn in sorted(fns):
                    if not fn.endswith(".kg"):
                        continue
                    path = os.path.join(dp, fn)
                    with open(path, newline="") as f:
                        text = f.read()
                    n += 1
                    for level in (0, 1):
                        with self.subTest(path=os.path.relpath(path, ROOT), level=level):
                            k = interp(level)
                            p = k.prog(Source(path, text))
                            self.assertEqual([w.render() for w in p.warnings], [])
        self.assertGreater(n, 100)


if __name__ == "__main__":
    unittest.main()
