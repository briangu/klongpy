import unittest
from klongpy import KlongInterpreter
from klongpy.core import rec_flatten, rec_fn2, KGChar, KGSym, is_integer, is_float
from utils import *
import time

# add tests not included in the original kg suite
class TestExtraCoreSuite(unittest.TestCase):

    def assert_eval_cmp(self, a, b, klong=None):
        self.assertTrue(eval_cmp(a, b, klong=klong))

    def test_fn_nilad(self):
        klong = KlongInterpreter()
        klong("F::{1}")
        r = klong('F(1)')
        self.assertEqual(r, 1)

    def test_fn_nilad_nested_monad(self):
        klong = KlongInterpreter()
        klong("F::{.p(1)}")
        r = klong('F(1)')
        self.assertEqual(r, '1')

    def test_fn_monad(self):
        klong = KlongInterpreter()
        klong("F::{x}")
        r = klong('F(1)')
        self.assertEqual(r, 1)

    def test_fn_monad_2(self):
        klong = KlongInterpreter()
        klong("F::{x,x}")
        r = klong('F(1)')
        self.assertTrue(array_equal(r, [1,1]))

    def test_fn_nested_monad(self):
        klong = KlongInterpreter()
        klong('G::{x};F::{G(x)}')
        r = klong('F(1)')
        self.assertEqual(r, 1)

    def test_fn_nilad_then_nested_monad(self):
        klong = KlongInterpreter()
        r = klong('bar::{x};foo::{bar("+")};foo()')
        self.assertEqual(r,'+')

    def test_fn_nested_monad_w_xform(self):
        klong = KlongInterpreter()
        klong('G::{x};F::{G(4_x)}')
        r = klong('F("hello")')
        self.assertEqual(r, 'o')

    def test_fn_nested_x_scope(self):
        klong = KlongInterpreter()
        klong("FL:::{};FL,0,{.p(,x@1)};F::{f::FL?0;f(x)}")
        r = klong('F("hello")')
        self.assertEqual(r, "e")

    def test_nested_x_scope_3(self):
        klong = KlongInterpreter()
        klong("G::{.p(,x@0)};F::{G(7_x)}")
        r = klong('F("Monkey 0:")')
        self.assertEqual(r, "0")

    def test_nested_x_scope_compact(self):
        klong = KlongInterpreter()
        klong("F::{{.p(,x@0)}(7_x)}")
        r = klong('F("Monkey 0:")')
        self.assertEqual(r, "0")

    def test_nested_x_scope_4(self):
        klong = KlongInterpreter()
        r = klong('{.p(,x@0)}(7_"Monkey 0:")')
        self.assertEqual(r, "0")

    def test_nested_x_scope_projection(self):
        klong = KlongInterpreter()
        klong('UM::{x};G::UM;F::{G(4_x)}')
        r = klong('F("hello")')
        self.assertEqual(r, "o")

    def test_nested_x_scope_dyad_projection(self):
        klong = KlongInterpreter()
        klong('UM::{x;y};G::UM("A";);F::{G(4_x)}')
        r = klong('F("hello")')
        self.assertEqual(r, "o")
