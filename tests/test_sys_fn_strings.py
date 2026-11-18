"""The string functions .split .join .words .trim .num."""
import time
import unittest

from klongpy import KlongInterpreter
from klongpy.errors import KlongRangeError, KlongTypeError
from klongpy.types import KGUndefined
from tests.backend_compat import requires_strings


@requires_strings
class TestStringFunctions(unittest.TestCase):
    def setUp(self):
        self.klong = KlongInterpreter()

    def run_(self, s):
        return self.klong(s)

    def strings(self, s):
        r = self.run_(s)
        self.assertTrue(all(type(e) is str for e in r), r)
        return [str(e) for e in r]

    def test_split(self):
        self.assertEqual(self.strings('.split("a,,b";",")'), ["a", "", "b"])
        self.assertEqual(self.strings('.split("";",")'), [""])
        self.assertEqual(self.strings('.split(",";",")'), ["", ""])
        self.assertEqual(self.strings('.split("a::b::";"::")'), ["a", "b", ""])
        self.assertEqual(self.strings('.split("a,b";0c,)'), ["a", "b"])
        self.assertEqual(self.strings('.split("abc";",")'), ["abc"])

    def test_split_unicode(self):
        self.assertEqual(self.strings('.split("hé,é";",")'), ["hé", "é"])
        self.assertEqual(self.strings('.split("aébéc";"é")'), ["a", "b", "c"])
        self.assertEqual(self.run_('#"héllo"'), 5)

    def test_split_errors(self):
        with self.assertRaisesRegex(KlongRangeError, r"\.split: y must not be empty"):
            self.run_('.split("abc";"")')
        with self.assertRaisesRegex(KlongTypeError, r"\.split: x must be a string, got integer"):
            self.run_('.split(5;",")')
        with self.assertRaisesRegex(KlongTypeError, r"\.split: x must be a string, got char"):
            self.run_('.split(0ca;",")')
        with self.assertRaisesRegex(KlongTypeError, r"\.split: y must be a string or a char, got integer"):
            self.run_('.split("a";5)')

    def test_join(self):
        self.assertEqual(self.run_('.join(["a" "b"];", ")'), "a, b")
        self.assertEqual(self.run_('.join([];",")'), "")
        self.assertEqual(self.run_('.join(["x"];",")'), "x")
        self.assertEqual(self.run_('.join([0ca "bc" 0cd];"-")'), "a-bc-d")
        self.assertEqual(self.run_('.join(["a" "b"];0c-)'), "a-b")
        self.assertEqual(self.run_('.join(.split(",a,,b,";",");",")'), ",a,,b,")
        self.assertIs(type(self.run_('.join(["a" "b"];"-")')), str)

    def test_join_errors(self):
        for src in ('.join(["a" 1];",")', '.join([1 2];",")', '.join("ab";",")', '.join(["a"];1)',
                    '.join([["a"] "b"];",")', '.join(5;",")'):
            with self.assertRaisesRegex(KlongTypeError, r"\.join: "):
                self.run_(src)

    def test_words(self):
        self.assertEqual(self.strings('.words("  a  bc d ")'), ["a", "bc", "d"])
        self.assertEqual(self.strings('.words("a\tb\r\nc")'), ["a", "b", "c"])
        self.assertEqual(len(self.run_('.words("   ")')), 0)
        self.assertEqual(len(self.run_('.words("")')), 0)
        with self.assertRaises(KlongTypeError):
            self.run_('.words(1)')

    def test_trim(self):
        self.assertEqual(self.run_('.trim("  a b  ")'), "a b")
        self.assertEqual(self.run_('.trim("   ")'), "")
        self.assertEqual(self.run_('.trim("\t x \r\n")'), "x")
        # only the four white space characters: a no-break space stays
        self.assertEqual(self.run_('.trim("  x ")'), " x")
        with self.assertRaises(KlongTypeError):
            self.run_('.trim(:a)')

    def test_num(self):
        self.assertEqual(self.run_('.num("12")'), 12)
        self.assertIsInstance(self.run_('.num("12")'), int)
        self.assertEqual(self.run_('.num("-3.5")'), -3.5)
        self.assertEqual(self.run_('.num("  7 ")'), 7)
        self.assertEqual(self.run_('.num("1e3")'), 1000.0)
        for text in ('7 8', 'x', '', ' ', '12ab', '1e', '-', '--3', '1-2', '[1 2]', '"a"', ':a', '\u0663', '1e999'):
            self.klong['v'] = text
            self.assertIsInstance(self.run_('.num(v)'), KGUndefined, text)
        with self.assertRaises(KlongTypeError):
            self.run_('.num(5)')

    def test_split_large(self):
        s = ",".join(["abcdefghi"] * 100001)
        self.klong['s'] = s
        t = time.time()
        r = self.run_('#.split(s;",")')
        self.assertEqual(r, 100001)
        self.assertLess(time.time() - t, 1.0)

    def test_lint_knows_the_names(self):
        from klongpy.lint import lint_source
        src = 'a::.split("a,b";",");b::.join(a;"-");c::.words(b);d::.trim(b);e::.num(d)\n'
        self.assertEqual([d.code for d in lint_source(src) if d.code == "W002"], [])


if __name__ == "__main__":
    unittest.main()
