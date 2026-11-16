"""Tests that began life as skipped entries in test_known_bugs.py and now pass (ticket 018)."""
import unittest

from klongpy import KlongInterpreter


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
