import unittest

import numpy as real_np
from klongpy.core import *
from klongpy.core import kg_argsort
from klongpy.backends import get_backend
from utils import die
from backend_compat import requires_object_dtype, requires_strings

# Get backend for type checking functions
_backend = get_backend()
is_integer = _backend.is_integer
is_float = _backend.is_float
is_number = _backend.is_number

class TestUtil(unittest.TestCase):

    def test_die(self):
        with self.assertRaises(RuntimeError):
            die("test")


    def test_is_list(self):
        self.assertTrue(is_list([]))
        self.assertTrue(is_list(bknp.array([])))
        self.assertFalse(is_list(()))
        self.assertFalse(is_list(1))
        self.assertFalse(is_list(None))

    def test_is_iterable(self):
        self.assertTrue(is_iterable(""))
        self.assertTrue(is_iterable(''))
        self.assertTrue(is_iterable([]))
        self.assertFalse(is_iterable(None))
        self.assertFalse(is_iterable(1))

    def test_is_empty(self):
        self.assertTrue(is_empty(""))
        self.assertTrue(is_empty(''))
        self.assertTrue(is_empty([]))
        self.assertTrue(is_empty(bknp.array([])))
        self.assertFalse(is_empty(bknp.array([1])))
        self.assertFalse(is_empty([1]))
        self.assertFalse(is_empty(None))
        self.assertFalse(is_empty(1))

    def test_to_list(self):
        self.assertTrue(isinstance(to_list([]), list))
        self.assertEqual(to_list([]), [])
        self.assertTrue(isinstance(to_list(1), list))
        self.assertEqual(to_list([1]), [1])
        self.assertTrue(isinstance(to_list(bknp.array([])), list))
        self.assertEqual(to_list(bknp.array([])), [])
        self.assertTrue(isinstance(to_list(bknp.array([1])), list))
        self.assertEqual(to_list(bknp.array([1])), [1])  # Compare to list, not array

    def test_is_float(self, fn=is_float):
        self.assertFalse(fn(None))
        self.assertFalse(fn([]))
        self.assertFalse(fn([1]))
        self.assertFalse(fn({}))
        self.assertFalse(fn(""))
        self.assertFalse(fn("a"))
        self.assertFalse(fn("0"))
        self.assertFalse(fn("1"))
        self.assertFalse(fn("0.0"))
        self.assertFalse(fn("1.0"))
        self.assertTrue(fn(0))
        self.assertTrue(fn(1))
        self.assertTrue(fn(-1))
        self.assertTrue(fn(0.0))
        self.assertTrue(fn(-0.0))
        self.assertTrue(fn(1.0))
        self.assertTrue(fn(-1.0))

    def test_is_int(self, fn=is_integer, allow_floats=False):
        self.assertFalse(fn(None))
        self.assertFalse(fn([]))
        self.assertFalse(fn([1]))
        self.assertFalse(fn({}))
        self.assertFalse(fn(""))
        self.assertFalse(fn("a"))
        self.assertFalse(fn("0"))
        self.assertFalse(fn("1"))
        self.assertFalse(fn("0.0"))# if not allow_floats else self.assertTrue(fn("0.0"))
        self.assertFalse(fn("1.0"))# if not allow_floats else self.assertTrue(fn("1.0"))
        self.assertTrue(fn(0))
        self.assertTrue(fn(1))
        self.assertTrue(fn(-1))
        self.assertTrue(fn(0))
        self.assertTrue(fn(-0))
        self.assertTrue(fn(1))
        self.assertTrue(fn(-1))

    def test_is_number(self):
        self.test_is_float(fn=is_number)
        self.test_is_int(fn=is_number, allow_floats=True)

    def test_in_map(self):
        self.assertFalse(in_map(1,None))
        self.assertFalse(in_map(1,{}))
        self.assertTrue(in_map(1,{1:2}))

    def test_is_char(self):
        self.assertFalse(is_char(None))
        self.assertFalse(is_char(0))
        self.assertFalse(is_char([]))
        self.assertFalse(is_char({}))
        self.assertFalse(is_char(()))
        self.assertTrue(is_char(KGChar('1')))
        self.assertTrue(is_char(KGChar("1")))
        self.assertTrue(is_char(KGChar("-")))
        self.assertTrue(is_char(KGChar('a')))
        self.assertTrue(is_char(KGChar("a")))
        self.assertFalse(is_char("ab"))


    def test_safe_eq(self):
        self.assertFalse(safe_eq(1,[]))
        self.assertTrue(safe_eq(1,1))

    @requires_object_dtype

    def test_is_adverb(self):
        a = ["'",':\\',":'",':/','/',':~',':*','\\','\\~','\\*']
        for x in a:
            self.assertTrue(is_adverb(x))
        self.assertFalse(is_adverb(":"))
        self.assertFalse(is_adverb("dog"))

    def test_argsort(self):
        a = [1,3,4,2]
        self.assertTrue(_backend.kg_equal(kg_argsort(a, _backend), [0, 3, 1, 2]))
        self.assertTrue(_backend.kg_equal(kg_argsort(a, _backend, descending=True), [2, 1, 3, 0]))
        a = [[1],[2],[],[3]]
        self.assertTrue(_backend.kg_equal(kg_argsort(a, _backend, descending=True), [3, 1, 0, 2]))


if __name__ == '__main__':
  unittest.main()
