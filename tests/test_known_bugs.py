import unittest

import pandas as pd
from utils import *

from klongpy import KlongInterpreter
from klongpy.core import KGChar


class TestKnownBugsSuite(unittest.TestCase):

    @unittest.skip
    def test_table_access_with_at(self):
        data = {'col1': np.arange(10)}
        df = pd.DataFrame(data)
        klong = KlongInterpreter()
        klong['df'] = df
        klong('.py("klongpy.db")')
        klong('T::.table(df)')
        # @ should work the same as a dictionary
        r = klong('T@"col1"')
        self.assertTrue(kg_equal(r, data['col1']))

    @unittest.skip
    def test_join_nested_array(self):
        klong = KlongInterpreter()
        r = klong("a::!10;k::3;c::(,,,1#a),k")
        # currently this flattens to [[0], 3]
        self.assertTrue(kg_equal(r,[[[[0]]],3]))

    @unittest.skip
    def test_extra_spaces(self):
        klong = KlongInterpreter()
        r = klong("a::{ 1 + 1 };a()")
        self.assertEqual(r, 2)

    @unittest.skip
    def test_semicolon_string_arg(self):
        klong = KlongInterpreter()
        klong('f::{x,y}')
        r = klong('f("hello";";")')
        self.assertEqual(r, "hello;")

    @unittest.skip
    def test_wrap_join(self):
        klong = KlongInterpreter()
        klong("q::[3 8]")
        r = klong("q[0],q[-1]")
        self.assertTrue(kg_equal(r, [3,8]))
        r = klong("(q[0],q[-1])")
        self.assertTrue(kg_equal(r, [3,8]))

    @unittest.skip
    def test_append_empty_dictionaries(self):
        klong = KlongInterpreter()
        r = klong("A::[];A::A,:{};A::A,:{};A::A,:{};A")
        self.assertTrue(kg_equal(r, [{}, {}, {}]))

    @unittest.skip
    def test_extra_chars_ignored(self):
        klong = KlongInterpreter()
        with self.assertRaises(Exception):
            klong("aggs::{[a];a:::{}}}}")

    @unittest.skip
    def test_tested_arity(self):
        # inner x is not seen in arity calculation
        #        {.pyc(x,"ticker";[];:{})}'1#symbols
        pass

    @unittest.skip
    def test_monad_argument_returned(self):
        """
        Test that a monadic lambda can be passed as an argument, returned, and then called.
        """
        klong = KlongInterpreter()
        klong('fn::{x+10}')
        klong('foo::{x}')
        r = klong('foo(fn)(2)')
        self.assertEqual(r, 12)

    @unittest.skip
    def test_triad_as_arguments_with_currying(self):
        """
        Test that two monads can be passed as arguments to a klong function.
        """
        klong = KlongInterpreter()
        klong('fn::{x+10+y*z}')
        klong('foo::{x(2;;)}')
        r = klong('w::foo(fn;;)')
        r = klong('w(;3;5)')
        self.assertEqual(r, 2+10+3*5)

    @unittest.skip
    def test_fail_non_terminated_string(self):
        klong = KlongInterpreter()
        with self.assertRaises(Exception):
            klong('a::"T')

    @unittest.skip
    def test_define_nilad_with_subcall(self):
        klong = KlongInterpreter()
        klong("nt::{x}")
        klong('newt::{nt([["1" 2] ["3" 4] ["5" 6]])}')

    @unittest.skip
    def test_join_two_dict(self):
        klong = KlongInterpreter()
        klong("b:::{[1 2]}")
        klong("c:::{[3 4]}")
        r = klong("b,c")
        self.assertEqual(r, {1: 2, 3: 4})

    @unittest.skip
    def test_nested_dict(self):
        klong = KlongInterpreter()
        klong('c:::{["GET" :{["/" 2]}]}')
        r = klong('(c?"GET")?"/"')
        self.assertEqual(r, 2)

    @unittest.skip
    def test_dict_inner_create_syntax(self):
        klong = KlongInterpreter()
        with self.assertRaises(RuntimeError):
            # should fail to parse
            r = klong(":{[1 :{[2 3]}}")

    @unittest.skip
    def test_dict_inner_create(self):
        # this creates a KGCall to wrap the inner dict, which is generally correct for
        klong = KlongInterpreter()
        r = klong(":{[1 :{[2 3]}]}")
        self.assertEqual(r[1], {2: 3})

    @unittest.skip
    def test_match_nested_array(self):
        klong = KlongInterpreter()
        r = klong('[7,7,7]~[7,7,7]')
        self.assertEqual(r,1)

    @unittest.skip
    def test_monad_not_getting_called(self):
        klong = KlongInterpreter()
        klong("""
BESTF::10^9;RESETF::{BESTF::10^9}
ITER::{1};
RUNX::{{x;ITER()}{x}:~ITER()}
SCAN::{RESETF();RUNX();BESTF}
        """)
        r = klong("SCAN()")
        self.assertEqual(r,99)

    @unittest.skip
    def test_take_nested_array(self):
        klong = KlongInterpreter()
        r = klong("(4)#[[0 0]]")
        self.assertTrue(kg_equal(r,[[0,0],[0,0],[0,0],[0,0]]))

    @unittest.skip
    def test_fall_call_undefined_fn(self):
        klong = KlongInterpreter()
        with self.assertRaises(RuntimeError):
            klong('R(1)')

    @unittest.skip
    def test_at_in_depth_strings(self):
        # DIFF: this isn't yet supported in Klong
        klong = KlongInterpreter()
        r = klong('["1234","5678"]:@[0 1]')
        self.assertEqual(r,KGChar('2'))

    @unittest.skip
    def test_dot_f_complex(self):
        """ TODO: this is from the ref doc but doesn't work in klong either """
        klong = KlongInterpreter()
        klong("fr::{:[@x;0;1+|/{.f(x)}'x]}")
        self.assert_eval_cmp('fr(0)', '[]', klong=klong)
        self.assert_eval_cmp('fr(1)', '[1]', klong=klong)
        self.assert_eval_cmp('fr(3)', '[1 1 1]', klong=klong)
        self.assert_eval_cmp('fr(10)', '[1 1 1 1 1 1 1 1 1 1]', klong=klong)
        klong("fr::{:[@x;0;1+|/.f'x]}")
        self.assert_eval_cmp('fr(0)', '[]', klong=klong)
        self.assert_eval_cmp('fr(1)', '[1]', klong=klong)
        self.assert_eval_cmp('fr(3)', '[1 1 1]', klong=klong)
        self.assert_eval_cmp('fr(10)', '[1 1 1 1 1 1 1 1 1 1]', klong=klong)
