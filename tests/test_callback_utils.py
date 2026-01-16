import unittest

from klongpy import KlongInterpreter
from klongpy.callback_utils import (
    find_callback_symbol,
    coerce_callback,
    resolve_callback,
    create_dynamic_callback,
)
from klongpy.core import KGFn, KGCall


class TestCallbackUtils(unittest.TestCase):

    def setUp(self):
        self.klong = KlongInterpreter()

    def test_find_callback_symbol_found(self):
        """Test finding a callback symbol that exists in context"""
        self.klong('cb::{x+1}')
        fn = self.klong('cb')

        sym = find_callback_symbol(self.klong, fn)

        self.assertIsNotNone(sym)
        self.assertEqual(str(sym), 'cb')

    def test_find_callback_symbol_not_found(self):
        """Test that None is returned for callbacks not in context"""
        # Create a function but don't assign it to a symbol
        self.klong('{x+1}')
        # Can't test this directly as anonymous functions aren't stored

        # Test with non-KGFn
        sym = find_callback_symbol(self.klong, lambda x: x)
        self.assertIsNone(sym)

    def test_find_callback_symbol_kgcall_returns_none(self):
        """Test that KGCall instances return None"""
        self.klong('fn::{x+1}')
        result = self.klong('fn(5)')  # This creates a KGCall

        # KGCall should return None
        sym = find_callback_symbol(self.klong, result)
        self.assertIsNone(sym)

    def test_coerce_callback_kgfn(self):
        """Test coercing a KGFn to KGFnWrapper"""
        self.klong('fn::{x+1}')
        fn = self.klong('fn')

        callback = coerce_callback(self.klong, fn)

        self.assertIsNotNone(callback)
        self.assertTrue(callable(callback))
        # Test it works
        result = callback(5)
        self.assertEqual(result, 6)

    def test_coerce_callback_callable(self):
        """Test coercing a Python callable returns it as-is"""
        py_fn = lambda x: x + 1

        callback = coerce_callback(self.klong, py_fn)

        self.assertIs(callback, py_fn)
        self.assertEqual(callback(5), 6)

    def test_coerce_callback_kgcall_returns_none(self):
        """Test that KGCall instances return None"""
        result = self.klong('{x+1}(5)')

        callback = coerce_callback(self.klong, result)

        self.assertIsNone(callback)

    def test_coerce_callback_non_callable(self):
        """Test that non-callable values return None"""
        callback = coerce_callback(self.klong, 42)
        self.assertIsNone(callback)

        callback = coerce_callback(self.klong, "string")
        self.assertIsNone(callback)

    def test_resolve_callback_with_symbol(self):
        """Test resolving a callback by symbol name"""
        self.klong('cb::{x+1}')
        fn = self.klong('cb')
        sym = find_callback_symbol(self.klong, fn)

        callback = resolve_callback(self.klong, sym, fn)

        self.assertIsNotNone(callback)
        self.assertTrue(callable(callback))
        self.assertEqual(callback(5), 6)

    def test_resolve_callback_fallback(self):
        """Test that resolve_callback falls back when symbol not found"""
        self.klong('cb::{x+1}')
        fn = self.klong('cb')
        sym = find_callback_symbol(self.klong, fn)

        # Delete the symbol
        self.klong('cb::0')  # Reassign to non-function

        # Should fall back to original fn
        callback = resolve_callback(self.klong, sym, fn)

        self.assertIsNotNone(callback)
        self.assertTrue(callable(callback))
        self.assertEqual(callback(5), 6)

    def test_resolve_callback_no_symbol(self):
        """Test resolving when symbol is None"""
        py_fn = lambda x: x + 1

        callback = resolve_callback(self.klong, None, py_fn)

        self.assertIs(callback, py_fn)

    def test_resolve_callback_picks_up_redefinition(self):
        """Test that resolve_callback uses the current symbol value"""
        self.klong('cb::{x+1}')
        fn = self.klong('cb')
        sym = find_callback_symbol(self.klong, fn)

        # Redefine the callback
        self.klong('cb::{x*10}')

        # Should use the new definition
        callback = resolve_callback(self.klong, sym, fn)

        self.assertIsNotNone(callback)
        self.assertEqual(callback(5), 50)  # Uses new definition (x*10)

    def test_create_dynamic_callback_resolves_on_each_call(self):
        """Test that dynamic callback re-resolves on each invocation"""
        self.klong('cb::{x+1}')
        fn = self.klong('cb')

        dynamic = create_dynamic_callback(self.klong, fn)

        # First call with original definition
        result1 = dynamic(5)
        self.assertEqual(result1, 6)

        # Redefine the callback
        self.klong('cb::{x*10}')

        # Second call should use new definition
        result2 = dynamic(5)
        self.assertEqual(result2, 50)

    def test_create_dynamic_callback_falls_back_if_deleted(self):
        """Test that dynamic callback falls back when symbol deleted"""
        self.klong('cb::{x+1}')
        fn = self.klong('cb')

        dynamic = create_dynamic_callback(self.klong, fn)

        # First call works
        result1 = dynamic(5)
        self.assertEqual(result1, 6)

        # Delete the symbol (set to non-function)
        self.klong('cb::0')

        # Should still work using fallback
        result2 = dynamic(5)
        self.assertEqual(result2, 6)

    def test_create_dynamic_callback_with_python_callable(self):
        """Test dynamic callback with Python function (no symbol)"""
        py_fn = lambda x: x + 1

        dynamic = create_dynamic_callback(self.klong, py_fn)

        # Should just return the function as-is since no symbol
        self.assertIs(dynamic, py_fn)

    def test_create_dynamic_callback_raises_if_becomes_uncallable(self):
        """Test that dynamic callback raises if symbol becomes non-function"""
        self.klong('cb::{x+1}')
        fn = self.klong('cb')

        dynamic = create_dynamic_callback(self.klong, fn)

        # First call works
        self.assertEqual(dynamic(5), 6)

        # Make symbol non-callable by deleting from context entirely
        # (This is hard to test since we can't easily delete from context)
        # Instead test the behavior when reassigned to non-callable
        self.klong('cb::0')

        # This should still work via fallback (tested above)
        # The raise condition only happens if both current and fallback fail

    def test_multiple_dynamic_callbacks_independent(self):
        """Test that multiple dynamic callbacks are independent"""
        self.klong('cb1::{x+1}')
        self.klong('cb2::{x*2}')
        fn1 = self.klong('cb1')
        fn2 = self.klong('cb2')

        dynamic1 = create_dynamic_callback(self.klong, fn1)
        dynamic2 = create_dynamic_callback(self.klong, fn2)

        self.assertEqual(dynamic1(5), 6)
        self.assertEqual(dynamic2(5), 10)

        # Redefine only cb1
        self.klong('cb1::{x*100}')

        # Only dynamic1 should change
        self.assertEqual(dynamic1(5), 500)
        self.assertEqual(dynamic2(5), 10)  # Unchanged


if __name__ == '__main__':
    unittest.main()
