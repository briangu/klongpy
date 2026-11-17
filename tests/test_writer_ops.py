"""
An operator value prints as its glyph, never as a Python
object repr.

Step 6 (the reserved glyphs `:! :& :, :< :?` as a syntax error when used as an
operator, still data inside lists) needs a change in the parser and the
interpreter; the data-list half is tested here because the writer owns it.
"""
import unittest
import warnings

from klongpy import KlongInterpreter
from klongpy.types import KGOp
from klongpy.writer import kg_write


class TestWriteOperators(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        self.k = KlongInterpreter()

    def show(self, src, display=False):
        return kg_write(self.k(src), self.k._backend, display=display)

    def test_assignment_operator(self):
        self.assertEqual(self.show("::"), "::")

    def test_list_of_operators(self):
        self.assertEqual(self.show("[+ -]"), "[+ -]")

    def test_nested_and_mixed_lists(self):
        self.assertEqual(self.show("[+ [- *]]"), "[+ [- *]]")
        self.assertEqual(self.show('[+ 1 "a" :b]'), '[+ 1 "a" :b]')

    def test_two_character_operators(self):
        self.assertEqual(self.show("[:: :+ :- :* :%]"), "[:: :+ :- :* :%]")
        self.assertEqual(self.show("[\\~ \\*]"), "[\\~ \\*]")

    def test_reserved_glyphs_stay_data_in_a_list(self):
        self.assertEqual(self.show("[:? 1]"), "[:? 1]")
        self.assertEqual(self.show("[:! :& :, :< :?]"), "[:! :& :, :< :?]")

    def test_display_form_is_the_same(self):
        self.assertEqual(self.show("[+ -]", display=True), "[+ -]")

    def test_no_object_repr(self):
        for src in ("::", "[+ -]", "[:: [+ -]]"):
            self.assertNotIn("object at", self.show(src))
            self.assertNotIn("KGOp", self.show(src))

    def test_operator_object_directly(self):
        be = self.k._backend
        self.assertEqual(kg_write(KGOp("+", 2), be), "+")
        self.assertEqual(kg_write(KGOp(":!", 1), be), ":!")
        self.assertEqual(kg_write(KGOp("\\~", 0), be), "\\~")

    def test_other_values_unchanged(self):
        self.assertEqual(self.show("[1 2.5 :foo]"), "[1 2.5 :foo]")
        self.assertEqual(self.show('"a"'), '"a"')
        self.assertEqual(self.show("1%0"), ":undefined")


if __name__ == "__main__":
    unittest.main()
