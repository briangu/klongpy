"""Ticket 010: amend raises range errors and does not truncate floats."""
import unittest

import numpy as np

from klongpy import KlongInterpreter
from klongpy.types import KlongException
from tests.backend_compat import dtype_kind


class Ticket010(unittest.TestCase):

    def run_kg(self, src):
        return KlongInterpreter()(src)

    def test_amend_string_index_past_end_raises(self):
        for src in ['"abc":="x",9', '"abc":="xy",5', '"abc":="xy",4']:
            with self.assertRaises(KlongException, msg=src):
                self.run_kg(src)

    def test_amend_string_negative_index(self):
        self.assertEqual(self.run_kg('"abc":="x",-1'), "abx")
        with self.assertRaises(KlongException):
            self.run_kg('"abc":="x",-4')

    def test_amend_string_growth_and_docstring_examples(self):
        self.assertEqual(self.run_kg('"abc":="def",3'), "abcdef")
        self.assertEqual(self.run_kg('"abc":="x",3'), "abcx")
        self.assertEqual(self.run_kg('"aa":="bc",1'), "abc")
        self.assertEqual(self.run_kg('"-----":=0cx,[1 3]'), "-x-x-")
        self.assertEqual(self.run_kg('"-------":="xx",[1 4]'), "-xx-xx-")

    def test_amend_float_into_int_array(self):
        r = self.run_kg('[1 2 3]:=0.5,1')
        self.assertEqual(list(r), [1, 0.5, 3])
        r = self.run_kg('[1 2 3]:=1.5,0')
        self.assertEqual(list(r), [1.5, 2, 3])

    def test_amend_int_into_int_array_keeps_dtype(self):
        r = self.run_kg('[1 2 3]:=0,1')
        self.assertEqual(list(r), [1, 0, 3])
        self.assertEqual(dtype_kind(r), "i")

    def test_amend_float_array_control(self):
        self.assertEqual(list(self.run_kg('[1 2 3.0]:=0.5,1')), [1.0, 0.5, 3.0])

    def test_amend_string_into_int_array_does_not_raise(self):
        r = self.run_kg('[1 2 3]:="a",1')
        self.assertEqual(list(r), [1, "a", 3])

    def test_amend_list_out_of_range_error_type(self):
        for src in ['[1 2 3]:=1,5', '[1 2 3]:=1,3', '[1 2 3]:=1,-4']:
            with self.assertRaises(KlongException, msg=src):
                self.run_kg(src)

    def test_amend_list_negative_index_wraps(self):
        self.assertEqual(list(self.run_kg('[1 2 3]:=9,-1')), [1, 2, 9])

    def test_amend_list_value_out_of_range(self):
        with self.assertRaises(KlongException):
            self.run_kg('[[1 2] [3 4]]:=[7 7],5')


if __name__ == "__main__":
    unittest.main()
