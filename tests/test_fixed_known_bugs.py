"""Tests that began life as skipped entries in test_known_bugs.py and now pass (ticket 018)."""
import unittest
import warnings

from klongpy import KlongInterpreter, KlongSyntaxError, KlongSyntaxWarning


class TestFixedKnownBugs(unittest.TestCase):

    def test_extra_spaces(self):
        # was test_known_bugs.test_extra_spaces; fixed by the parser/lexer work (030/031 era)
        klong = KlongInterpreter()
        r = klong("a::{ 1 + 1 };a()")
        self.assertEqual(r, 2)

    def test_wrap_join(self):
        # was test_known_bugs.test_wrap_join. Klong indexes with `@`, not `q[i]`; negative
        # indices wrap per the ticket 015 decision, so the first and last element joined
        # give the whole two-element list.
        klong = KlongInterpreter()
        klong("q::[3 8]")
        r = klong("(q@0),q@-1")
        self.assertTrue(klong._backend.kg_equal(r, [3, 8]))
        r = klong("((q@0),(q@-1))")
        self.assertTrue(klong._backend.kg_equal(r, [3, 8]))


if __name__ == '__main__':
    unittest.main()

    def test_semicolon_string_arg(self):
        # was test_known_bugs.test_semicolon_string_arg; fixed by the token parser (ticket 023):
        # a string that holds ";" is a string
        klong = KlongInterpreter()
        klong('f::{x,y}')
        r = klong('f("hello";";")')
        self.assertEqual(r, "hello;")

    def test_define_nilad_with_subcall(self):
        # was test_known_bugs.test_define_nilad_with_subcall; ticket 032: computing the arity
        # of a function hashed the arguments of a call in its body (unhashable numpy.ndarray)
        klong = KlongInterpreter()
        klong("nt::{x}")
        klong('newt::{nt([["1" 2] ["3" 4] ["5" 6]])}')
        self.assertEqual(klong("newt()").shape, (3, 2))

    # The next four were "accepted without a word" (ticket 031). At the default strict
    # level 0 a program keeps meaning what it meant, so they are still accepted, now with a
    # KlongSyntaxWarning; at strict level 1 they are the syntax errors the tests asked for.

    def _warns_then_raises(self, text, code, setup=()):
        klong = KlongInterpreter()
        for s in setup:
            klong(s)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = klong(text)
        self.assertEqual([w.message.code for w in caught if isinstance(w.message, KlongSyntaxWarning)], [code])
        klong(".strict()")
        with self.assertRaises(KlongSyntaxError):
            klong(text)
        return r

    def test_extra_chars_ignored(self):
        # was test_known_bugs.test_extra_chars_ignored
        self._warns_then_raises("aggs::{[a];a:::{}}}}", "stray-closer")

    def test_fail_non_terminated_string(self):
        # was test_known_bugs.test_fail_non_terminated_string
        self.assertEqual(self._warns_then_raises('a::"T', "unterminated-string"), "T")

    def test_dict_inner_create_syntax(self):
        # was test_known_bugs.test_dict_inner_create_syntax (it asked for a RuntimeError; a
        # KlongSyntaxError is not one)
        self._warns_then_raises(":{[1 :{[2 3]}}", "unterminated-dict")

    def test_call_of_a_call_result_is_dropped_with_a_warning(self):
        # test_known_bugs.test_monad_argument_returned stays skipped: `foo(fn)(2)` is not
        # application. The dropped `(2)` is no longer silent.
        r = self._warns_then_raises("foo(fn)(2)", "trailing-input", setup=("fn::{x+10}", "foo::{x}"))
        klong = KlongInterpreter()
        klong("fn::{x+10}")
        klong("foo::{x}")
        self.assertEqual(klong("g::foo(fn);g(2)"), 12)
