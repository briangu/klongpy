"""Source (klongpy/source.py): the text of a program with its name and the positions errors and warnings point at."""
import copy
import pickle
import unittest

from klongpy.source import Source


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


if __name__ == "__main__":
    unittest.main()
