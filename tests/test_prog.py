import unittest
from klongpy import KlongInterpreter
from klongpy.core import KGSym
from utils import *
from backend_compat import requires_strings

class TestProg(unittest.TestCase):

    @requires_strings
    def test_read_string_vs_op(self):
        """
        This test makes sure that we can parse a string that's the same character string as an operation.
        """
        klong = KlongInterpreter()
        r = klong("""
        bar::{x}
lin2::{bar("+")}
lin2()
        """)
        self.assertEqual(r,'+')

    @requires_strings
    def test_read_string_arg(self):
        """
        This test makes sure that we can parse a string that's the same character string as an operation.
        """
        klong = KlongInterpreter()
        r = klong("""
lin2::{.d("+");{.d((2+#x):^"-");.d("+")}'x;.p("");x}
lin2(1)
            """)
        self.assertEqual(r, 1)


class TestEvaluatedArrayConstructors(unittest.TestCase):
    """Tests for the [;expr1;expr2;...] evaluated array syntax."""

    def test_basic_expressions(self):
        klong = KlongInterpreter()
        r = klong('[;1+1;2+2;3+3]')
        self.assertEqual(list(r), [2, 4, 6])

    def test_with_variables(self):
        klong = KlongInterpreter()
        klong('a::10')
        klong('b::20')
        r = klong('[;a;b;a+b]')
        self.assertEqual(list(r), [10, 20, 30])

    def test_with_function_calls(self):
        klong = KlongInterpreter()
        klong('avg::{(+/x)%#x}')
        klong('arr::[1 2 3 4 5]')
        r = klong('[;avg(arr);+/arr;#arr]')
        values = [float(x) for x in r]
        self.assertEqual(values, [3.0, 15.0, 5.0])

    def test_empty_array(self):
        klong = KlongInterpreter()
        r = klong('[;]')
        self.assertEqual(len(r), 0)

    def test_single_element(self):
        klong = KlongInterpreter()
        r = klong('[;42]')
        self.assertEqual(list(r), [42])

    def test_nested_arrays(self):
        klong = KlongInterpreter()
        klong('x::[1 2 3]')
        klong('y::[4 5 6]')
        r = klong('[;x;y]')
        self.assertEqual(list(r[0]), [1, 2, 3])
        self.assertEqual(list(r[1]), [4, 5, 6])


class TestEachIndexAdverb(unittest.TestCase):
    """Tests for the @' each-index adverb."""

    def test_index_extraction(self):
        klong = KlongInterpreter()
        r = klong("{x@0}@'[10 20 30]")
        self.assertEqual(list(r), [0, 1, 2])

    def test_value_extraction(self):
        klong = KlongInterpreter()
        r = klong("{x@1}@'[10 20 30]")
        self.assertEqual(list(r), [10, 20, 30])

    def test_index_times_value(self):
        klong = KlongInterpreter()
        r = klong("{(x@0)*(x@1)}@'[10 20 30]")
        self.assertEqual(list(r), [0, 20, 60])

    def test_empty_array(self):
        klong = KlongInterpreter()
        r = klong("{x}@'[]")
        self.assertEqual(len(r), 0)

    def test_enumerate_behavior(self):
        klong = KlongInterpreter()
        r = klong("{x}@'[100 200 300]")
        self.assertEqual(list(r[0]), [0, 100])
        self.assertEqual(list(r[1]), [1, 200])
        self.assertEqual(list(r[2]), [2, 300])
