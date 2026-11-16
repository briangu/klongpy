import unittest

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from utils import *

from klongpy import KlongInterpreter
from klongpy.core import KGChar, KGSym

requires_pandas = unittest.skipUnless(HAS_PANDAS, "requires pandas")


class TestKnownBugsSuite(unittest.TestCase):
    """Real bugs and not-yet-implemented behaviour. Every skip names the ticket that owns it.

    Triaged under ticket 018. Passing tests moved to test_fixed_known_bugs.py; intended
    differences from reference Klong live in TestDocumentedDifferences below.
    """

    @requires_pandas
    @unittest.skip("ticket 063: @ has no dict branch, so T@\"col1\" returns the table, not the column")
    def test_table_access_with_at(self):
        data = {'col1': np.arange(10)}
        df = pd.DataFrame(data)
        klong = KlongInterpreter()
        klong['df'] = df
        klong('.py("klongpy.db")')
        klong('T::.table(df)')
        # @ should work the same as a dictionary
        r = klong('T@"col1"')
        self.assertTrue(klong._backend.kg_equal(r, data['col1']))

    @unittest.skip("ticket 031 left this to a language ticket: `foo(fn)(2)` does not apply the result of a call "
                   "(`g::foo(fn);g(2)` works). The dropped `(2)` is now a KlongSyntaxWarning at strict level 0 "
                   "and a syntax error at level 1 (tests/test_fixed_known_bugs.py)")
    def test_monad_argument_returned(self):
        """
        Test that a monadic lambda can be passed as an argument, returned, and then called.
        """
        klong = KlongInterpreter()
        klong('fn::{x+10}')
        klong('foo::{x}')
        r = klong('foo(fn)(2)')
        self.assertEqual(r, 12)

    @unittest.skip("ticket 063: join of two dictionaries rejected (dyads `,` only takes a pair)")
    def test_join_two_dict(self):
        klong = KlongInterpreter()
        klong("b:::{[1 2]}")
        klong("c:::{[3 4]}")
        r = klong("b,c")
        self.assertEqual(r, {1: 2, 3: 4})

    @unittest.skip("ticket 063: nested dict literal is a deferred call, so `?` on it fails (ValueError)")
    def test_nested_dict(self):
        klong = KlongInterpreter()
        klong('c:::{["GET" :{["/" 2]}]}')
        r = klong('(c?"GET")?"/"')
        self.assertEqual(r, 2)

    @unittest.skip("ticket 063: the inner literal stays a deferred KGCall instead of a dict")
    def test_dict_inner_create(self):
        klong = KlongInterpreter()
        r = klong(":{[1 :{[2 3]}]}")
        self.assertEqual(r[1], {2: 3})

    def test_take_nested_array(self):
        klong = KlongInterpreter()
        r = klong("(4)#[[0 0]]")
        self.assertTrue(klong._backend.kg_equal(r,[[0,0],[0,0],[0,0],[0,0]]))

    @unittest.skip("ticket 051/040: calling an undefined function raises KlongException (not a "
                   "RuntimeError subclass) today; strict level 2 owns the error contract")
    def test_fall_call_undefined_fn(self):
        klong = KlongInterpreter()
        with self.assertRaises(RuntimeError):
            klong('R(1)')


class TestDocumentedDifferences(unittest.TestCase):
    """Intended KlongPy behaviour that differs from reference Klong (ticket 018, listed by ticket 081)."""

    def test_array_substitution(self):
        # Differs: a list literal takes tokens, so operators become KGOp objects and names become
        # symbols; use `[;...]` for an evaluated list. Reference Klong gives [:! 10].
        klong = KlongInterpreter()
        r = klong('[!10]')
        self.assertEqual(len(r), 2)
        self.assertEqual(type(r[0]).__name__, 'KGOp')
        self.assertEqual(r[1], 10)
        r = klong('[3*a^2]')
        self.assertEqual(len(r), 5)
        self.assertEqual(r[0], 3)
        self.assertEqual(type(r[1]).__name__, 'KGOp')
        self.assertEqual(r[2], KGSym('a'))
        self.assertEqual(type(r[3]).__name__, 'KGOp')
        self.assertEqual(r[4], 2)

    def test_match_nested_array(self):
        # Differs: `,` is not a list separator; `[7,7,7]` is a five-element token list
        # (7 , 7 , 7 with `,` as an operator object), not [7 7 7].
        klong = KlongInterpreter()
        self.assertEqual(klong('#[7,7,7]'), 5)
        self.assertEqual(klong('[7 7 7]~[7 7 7]'), 1)

    def test_append_empty_dictionaries(self):
        # Differs: `list,dict` follows the pair rule (ticket 063): a two-element list joined with
        # a dict is read as a key/value pair, so the third append fails.
        klong = KlongInterpreter()
        r = klong("A::[];A::A,:{}")
        self.assertEqual(len(r), 1)
        with self.assertRaises(Exception):
            klong("A::A,:{};A::A,:{}")

    def test_triad_as_arguments_with_currying(self):
        # Differs: a projection like x(2;;) needs the arity of x at parse time, which a lambda
        # parameter does not have, so currying a function passed as an argument does not work.
        # Currying a named function directly does.
        klong = KlongInterpreter()
        klong('fn::{x+10+y*z}')
        klong('p::fn(2;;)')
        self.assertEqual(klong('p(3;5)'), 2+10+3*5)
        klong('foo::{x(2;;)}')
        klong('w::foo(fn;;)')
        self.assertNotEqual(klong('w(;3;5)'), 2+10+3*5)

    def test_at_in_depth_strings(self):
        # Differs: `@` with a list of indices does not index into depth (strings), as the
        # original comment says; it raises (a raw IndexError today, to be wrapped by ticket 041).
        klong = KlongInterpreter()
        with self.assertRaises(Exception):
            klong('["1234","5678"]:@[0 1]')


if __name__ == '__main__':
    unittest.main()
