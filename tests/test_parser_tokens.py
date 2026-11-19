"""
The reader of Klong text, which is the native engine's: the trees of a table of the legacy language's quirks (as `kgpy ast`
prints them, tests/treedump.py) and the classes and messages of the errors a text that does not parse gives.
"""
import unittest

from klongpy import KlongInterpreter
from tests.treedump import dump_text


class TestQuirks(unittest.TestCase):
    def test_quirks_have_the_recorded_results(self):
        """A table of the legacy quirks of the syntax, pinned to the tree each gives."""
        want = {
            "f (1)": "(sym f)\n",
            "1;)": '(num 1)\n(str ")")\n',
            "[-0cx]": "(error KlongSyntaxError)\n",
            'f(1 :"c"\n;2)': "(proj (sym f) (num 1) _ (num 2))\n",
            "f( ;1)": "(call (sym f) (num 1))\n",
            '[1 2 :"c"\n]': '(list (num 1) (num 2) (str "]"))\n',
            '[1 :"c1"\n :"c2"\n 2]': '(list (num 1) (str ";") (num 2))\n',
            '{x :"c"\n}': "(error UnexpectedChar)\n",
            "{x;}": "(error UnexpectedChar)\n",
            "{+/\nx}": "(error UnexpectedChar)\n",
            ":0cx": '(char "x")\n',
            "f(1 [2])": "(call (sym f) (num 1) (list (num 2)))\n",
            "(;)1": "(num 1)\n",
            "f(0c\n )": '(call (sym f) (char "\\n"))\n',
            "f(1 )": "(error UnexpectedChar)\n",
            "1 :foo 2": "(infix (sym foo) (num 1) (num 2))\n",
            "{{f}}": "(fn 0 (fn 0 (sym f)))\n",
            ".comment()": "(error KlongSyntaxError)\n",
            ".comment(1)": "(error KlongSyntaxError)\n",
            '.comment("END") 1': "(error KlongSyntaxError)\n",   # also a RuntimeError
            "1.2.3": "(error KlongSyntaxError)\n",               # also a ValueError
            "0c": "(error UnexpectedEOF)\n",
        }
        for text, sexp in want.items():
            with self.subTest(text=text):
                self.assertEqual(dump_text(text), sexp)

    def test_error_messages_and_positions(self):
        for text, cls, msg in [("(1+2", "UnexpectedChar", "t: (1+2 pos: 4 char: )"),
                               ("(  ", "UnexpectedChar", "t: (   pos: 3 char: )"),
                               ("{x  \n ", "UnexpectedChar", "t:   \n  pos: 6 char: }"),
                               (":[1 2;3]", "UnexpectedChar", "t: 1 2;3] pos: 4 char: ;"),
                               ("f(1 )", "UnexpectedChar", "t: f(1 ) pos: 3 char: )"),
                               ("1+0c", "UnexpectedEOF", "t: 1+0c pos: 4"),
                               ("3e", "KlongSyntaxError", "could not convert string to float: '3e'"),
                               ("[-1.2.3]", "KlongSyntaxError", "could not convert string to float: '-1.2.3'"),
                               ('.comment("E")', "KlongSyntaxError", "end of comment not found")]:
            with self.subTest(text=text):
                with self.assertRaises(Exception) as cm:
                    KlongInterpreter()(text)
                e = cm.exception
                self.assertEqual(type(e).__name__, cls)
                self.assertIn(msg, getattr(e, "legacy_text", None) or str(e))


if __name__ == "__main__":
    unittest.main()
