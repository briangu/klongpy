import unittest

from klongpy import KlongInterpreter
from klongpy.core import KGCall, KGFnWrapper, KGSym


class TestKGFnWrapper(unittest.TestCase):
    def test_find_symbol_skips_reserved(self):
        klong = KlongInterpreter()
        klong("f::{x+1}")
        fn = klong._context[KGSym("f")]
        del klong["f"]
        klong._context[KGSym("x")] = fn

        wrapper = KGFnWrapper(klong, fn)

        self.assertIsNone(wrapper._sym)

    def test_find_symbol_non_kgfn(self):
        klong = KlongInterpreter()
        wrapper = KGFnWrapper(klong, lambda x: x)

        self.assertIsNone(wrapper._sym)

    def test_find_symbol_kgcall(self):
        klong = KlongInterpreter()
        call = KGCall(KGSym("f"), [], 1)

        wrapper = KGFnWrapper(klong, call)

        self.assertIsNone(wrapper._sym)

    def test_dynamic_resolution_arity_mismatch(self):
        klong = KlongInterpreter()
        klong("f::{x+1}")
        wrapper = klong["f"]
        klong("f::{1}")

        with self.assertRaises(RuntimeError):
            wrapper(2)

    def test_fallback_when_symbol_deleted(self):
        klong = KlongInterpreter()
        klong("f::{x+1}")
        wrapper = klong["f"]
        del klong["f"]

        self.assertEqual(wrapper(2), 3)
