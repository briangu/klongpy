import unittest

from utils import *
from klongpy import KlongInterpreter

#
# Deprecated: now in test_fn.kg
#
class TestFunctionsSuite(unittest.TestCase):

    def assert_eval_cmp(self, a, b, klong=None):
        self.assertTrue(eval_cmp(a, b, klong=klong))

    def test_fn_monad_2(self):
        klong = KlongInterpreter()
        klong("F::{x,x}")
        r = klong('F(1)')
        self.assertTrue(kg_equal(r, [1,1]))

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

