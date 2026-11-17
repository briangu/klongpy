"""Form (:$) returns :undefined on a failed conversion."""
import unittest

from klongpy import KlongInterpreter
from klongpy.core import KLONG_UNDEFINED


class TestFormUndefined(unittest.TestCase):

    def run_kg(self, src):
        return KlongInterpreter()(src)

    def test_form_int_bad_string(self):
        self.assertIs(self.run_kg('1:$"abc"'), KLONG_UNDEFINED)

    def test_form_float_bad_string(self):
        self.assertIs(self.run_kg('1.0:$"abc"'), KLONG_UNDEFINED)

    def test_form_list_partial(self):
        r = self.run_kg('[1 1]:$["12" "x"]')
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], 12)
        self.assertIs(r[1], KLONG_UNDEFINED)
        r = self.run_kg('1.0:$["1" "x"]')
        self.assertEqual(r[0], 1.0)
        self.assertIs(r[1], KLONG_UNDEFINED)

    def test_form_undefined_is_testable(self):
        self.assertEqual(self.run_kg(':_(1:$"abc")'), 1)

    def test_form_single_element_list(self):
        r = self.run_kg('1:$[5]')
        self.assertEqual(len(r), 1)
        self.assertEqual(list(self.run_kg('1.0:$[5]')), [5.0])

    def test_form_regression_guards(self):
        self.assertIs(self.run_kg('1:$"1.5"'), KLONG_UNDEFINED)
        self.assertIs(self.run_kg('0c0:$"xy"'), KLONG_UNDEFINED)
        self.assertIs(self.run_kg('1:$""'), KLONG_UNDEFINED)
        self.assertIs(self.run_kg('1.0:$""'), KLONG_UNDEFINED)
        self.assertIs(self.run_kg('1:$5'), KLONG_UNDEFINED)
        self.assertEqual(self.run_kg('1:$"12"'), 12)
        self.assertEqual(self.run_kg('1.0:$"1.5"'), 1.5)
        self.assertEqual(self.run_kg('1.0:$5'), 5.0)


if __name__ == "__main__":
    unittest.main()
