import unittest

from utils import *

from klongpy import KlongInterpreter
from klongpy.core import (KGChar, KGSym, is_float, is_integer, rec_flatten)


# add tests not included in the original kg suite
class TestExtraCoreSuite(unittest.TestCase):

    def assert_eval_cmp(self, a, b, klong=None):
        self.assertTrue(eval_cmp(a, b, klong=klong))

    def test_negate_array_result_type(self):
        """ ensure the result type of negating an array is correct """
        klong = KlongInterpreter()
        r = klong("-1]")
        self.assertTrue(is_integer(r))
        r = klong("-1.0]")
        self.assertTrue(is_float(r))
        r = klong("-[1 2 3]")
        self.assertTrue(r.dtype == int)
        r = klong("-[1.0 2.0 3.0]")
        self.assertTrue(r.dtype == float) 

    def test_match_empty_array_to_undefined_symbol(self):
        """ symbol is undefined so does not match the empty array """
        klong = KlongInterpreter()
        r = klong('[]~.a')
        self.assertEqual(r, 0)

    def test_vectorized(self):
        klong = KlongInterpreter()
        r = klong("2*!1000")
        self.assertTrue(kg_equal(r, np.arange(1000)*2))

    # This is different behavior than Klong, which doesn't allow at/index on dictionaries.
    def test_dict_at_index(self):
        klong = KlongInterpreter()
        klong("D:::{[1 2]}")
        r = klong("D@1")
        self.assertEqual(r, 2)
        with self.assertRaises(KeyError):
            klong("D@2")

    def test_each_dict_with_mixed_types(self):
        klong = KlongInterpreter()
        klong["D"] = {object: [1, 2, 3]}
        klong(".p'D")

    def test_apply_range(self):
        klong = KlongInterpreter()
        r = klong("{x}@,!100")
        self.assertTrue(kg_equal(r, np.arange(100)))
        klong("avg::{(+/x)%#x}")
        r = klong("avg@,!100")
        self.assertEqual(r,49.5)

    def test_eval_quote_string(self):
        klong = KlongInterpreter()
        r = klong(':"hello"')
        self.assertTrue(r is None)

    def test_array_identity(self):
        klong = KlongInterpreter()
        r = klong('[]')
        self.assertTrue(kg_equal(r, np.array([],dtype=object)))
        r = klong('[1]')
        self.assertTrue(kg_equal(r, np.array([1],dtype=object)))
        r = klong('[[1]]')
        self.assertTrue(kg_equal(r, np.array([[1]],dtype=object)))
        r = klong('[[1] [2]]')
        self.assertTrue(kg_equal(r, np.array([[1],[2]],dtype=object)))
        r = klong('[[1] [2 3]]')
        self.assertTrue(kg_equal(r, np.array([[1],[2,3]],dtype=object)))
        r = klong('[[[1]] [2 3]]')
        self.assertTrue(kg_equal(r, np.array([[[1]],[2,3]],dtype=object)))
        r = klong('[[1] [[2 3]]]')
        self.assertTrue(kg_equal(r, np.array([[1],[[2,3]]],dtype=object)))
        r = klong('[[[1]] [[2 3]]]')
        self.assertTrue(kg_equal(r, np.array([[[1]],[[2,3]]],dtype=object)))

    def test_jagged_array_identity(self):
        klong = KlongInterpreter()
        r = klong('[[0] [[1]]]')
        q = np.array([[0],[[1]]],dtype=object)
        self.assertTrue(kg_equal(r, q))

    def test_match_array(self):
        klong = KlongInterpreter()
        r = klong('(^[1])~0')
        self.assertFalse(r is False)
        self.assertEqual(r,0)

    def test_prime(self):
        klong = KlongInterpreter()
        klong('prime::{&/x!:\\2+!_x^1%2}') # note \\ ==> \
        r = klong("prime(251)")
        self.assertTrue(is_integer(r))
        self.assertEqual(r, 1)

    def test_floor_as_int(self):
        klong = KlongInterpreter()
        r = klong('_30%2')
        self.assertTrue(is_integer(r))
        self.assertEqual(r, 15)
        r = klong('_[30 20]%2')
        for x in r:
            self.assertTrue(is_integer(x))
        self.assertTrue(kg_equal(r, [15, 10]))

    def test_drop_string(self):
        klong = KlongInterpreter()
        klong("""
NAMES:::{}
AN::{[k n g];.p(x);k::(x?",")@0;n::.rs(k#x);g::.rs((k+1)_x);NAMES,n,,g}")
        """)
        r = klong('S::\"""John"",""boy""\"')
        self.assertEqual(r,'"John","boy"')
        r = klong('AN(S);NAMES')
        self.assertEqual(r['John'], "boy")

    # read 123456 from "123456 hello" requires parsing by space
    def test_read_number_from_various_strings(self):
        klong = KlongInterpreter()
        # DIFF: Klong will puke on this with undefined
        r = klong('.rs("123456 hello")')
        self.assertEqual(r, 123456)

    def test_join_nested_arrays(self):
        self.assert_eval_cmp('[[0 0] [1 1]],,2,2', '[[0 0] [1 1] [2 2]]')

    def test_range_over_nested_arrays(self):
        self.assert_eval_cmp('?[[0 0] [1 1] 3 3]', '[[0 0] [1 1] 3]')
        self.assert_eval_cmp('?[[0 0] [1 1] [1 1]]', '[[0 0] [1 1]]')
        self.assert_eval_cmp('?[[0 0] [1 1] [1 1] 3 3]', '[[0 0] [1 1] 3]')
        self.assert_eval_cmp('?[[[0 0] [0 0] [1 1]] [1 1] [1 1]]', '[[[0 0] [0 0] [1 1]] [1 1]]')
        self.assert_eval_cmp('?[[[0 0] [0 0] [1 1]] [1 1] [1 1] 3 3]', '[[[0 0] [0 0] [1 1]] [1 1] 3]')
        self.assert_eval_cmp('?[[0 0] [1 0] [2 0] [3 0] [4 1] [4 2] [4 3] [3 4] [2 4] [3 3] [4 3] [3 2] [2 2] [1 2]]', '[[0 0] [1 0] [2 0] [3 0] [4 1] [4 2] [4 3] [3 4] [2 4] [3 3] [3 2] [2 2] [1 2]]')

    def test_range_distinguishes_types(self):
        self.assert_eval_cmp('?[10 "10"]', '[10 "10"]')
        self.assert_eval_cmp('?[:foo ":foo"]', '[:foo ":foo"]')

    def test_sum_over_nested_arrays(self):
        """
        sum over nested arrays should reduce
        """
        self.assert_eval_cmp('+/[1 2 3]', '6')
        self.assert_eval_cmp('+/[[1 2 3]]', '[1 2 3]')
        self.assert_eval_cmp('+/[[1 2 3] [4 5 6]]', '[5 7 9]')
        self.assert_eval_cmp('+/[[1 2 3] [4 5 6] [7 8 9]]', '[12 15 18]')

    def test_power_preserve_type(self):
        klong = KlongInterpreter()
        r = klong("10^5")
        self.assertTrue(is_integer(r))
        r = klong("10.5^5")
        self.assertTrue(is_float(r))

    def test_join_array_dict(self):
        klong = KlongInterpreter()
        klong("""
N::{d:::{[:s 0] [:c []]};d,:p,x;d,:n,y}
D::N(1%0;"/")
n::N(D;"x")
        """)
        klong('(D?:c),n')

    def test_join_sym_string(self):
        klong = KlongInterpreter()
        r = klong(':p,"hello"')
        self.assertTrue(isinstance(r[0],KGSym))
        self.assertEqual(r[1],"hello")

    def test_join_sym_dict(self):
        klong = KlongInterpreter()
        klong("D:::{[1 2]}")
        r = klong(":p,D")
        self.assertTrue(isinstance(r[0],KGSym))
        self.assertTrue(isinstance(r[1],dict))

    def test_complex_join_dict_create(self):
        klong = KlongInterpreter()
        klong("""
N::{d:::{[:s 0] [:c []]};d,:p,x;d,:n,y}
D::N(1%0;"/")
        """)
        r = klong('q::N(D;"hello")')
        self.assertEqual(klong("q?:s"), 0)
        self.assertTrue(kg_equal(klong("q?:c"), []))
        self.assertEqual(klong("q?:n"), "hello")
        self.assertTrue(isinstance(klong("q?:p"), dict))

    def test_join_sym_int(self):
        klong = KlongInterpreter()
        r = klong(":p,43")
        self.assertTrue(isinstance(r[0],KGSym))
        self.assertTrue(isinstance(r[1],int))

    def test_integer_divide_clamp_to_int(self):
        klong = KlongInterpreter()
        r = klong("24:%2")
        self.assertTrue(is_integer(r))
        r = klong("[12 24]:%2")
        self.assertTrue(is_integer(r[0]))
        self.assertTrue(is_integer(r[1]))

    def test_enumerate_float(self):
        klong = KlongInterpreter()
        with self.assertRaises(RuntimeError):
            klong("!(24%2)")
        r = klong("!(24:%2)")
        self.assertEqual(len(r), 12)

    def test_at_index_single_index(self):
        klong = KlongInterpreter()
        def _char_test(x):
            if not isinstance(x,KGChar):
                raise RuntimeError("should be char")
        klong['ischar'] = _char_test
        klong('ischar("hello"@2)')

    def test_at_index_array_index(self):
        klong = KlongInterpreter()
        def _str_test(x):
            if isinstance(x,KGChar):
                raise RuntimeError("should be string")
        klong['isstr'] = _str_test
        klong('isstr("hello"@[2])')
        klong('isstr("hello"@[1 2 2])')

    def test_module_fallback(self):
        klong = KlongInterpreter()
        r = klong("""
        pi::3.14
        .module(:test)
        foo::pi
        .module(0)
        foo
        """)
        self.assertTrue(r,3.14)

    def test_read_enotation(self):
        klong = KlongInterpreter()
        r = klong('t::1.0e8;')
        self.assertEqual(r, 1.0e8)
        r = klong('t::1.0e-8;')
        self.assertEqual(r, 1.0e-8)

    def test_read_list(self):
        klong = KlongInterpreter()
        r = klong('[]')
        self.assertTrue(kg_equal(r,[]))
        klong = KlongInterpreter()
        r = klong('[1 2 3 4]')
        self.assertTrue(kg_equal(r,[1,2,3,4]))
        klong = KlongInterpreter()
        r = klong('[1 2 3 4 ]') # add spaces as found in suite
        self.assertTrue(kg_equal(r,[1,2,3,4]))

    def test_read_list_as_arg(self):
        """
        make list span lines to test whitespace skip
        """
        t = """
        zop::{x}
zop([
    ["hello"]
])
        """
        klong = KlongInterpreter()
        klong('zap::{x}')
        r = klong('zap([])')
        self.assertTrue(kg_equal(r,[]))
        r = klong(t)
        self.assertTrue(kg_equal(r,[["hello"]]))

    def test_sys_comment(self):
        t = """.comment("end-of-comment")
                abcdefghijklmnopqrstuvwxyz
                ABCDEFGHIJKLMNOPQRSTUVWXYZ
                0123456789
                ~!@#$%^&*()_+-=`[]{}:;"'<,>.
                end-of-comment

                A::1;
                A
            """
        klong = KlongInterpreter()
        r = klong.exec(t)
        # verify that we are returning both A::1 and A operation results
        self.assertTrue(kg_equal(r,[1,1]))
        r = klong(t)
        self.assertEqual(r, 1)

    def test_eval_array(self):
        klong = KlongInterpreter()
        r = klong('[[[1] [2] [3]] [1 2 3]]')
        self.assertTrue(kg_equal(r,[[[1],[2],[3]],[1,2,3]]))

    def test_dot_f(self):
        klong = KlongInterpreter()
        klong('fr::{:[x;"hello";.f(1)]}')
        self.assert_eval_cmp('fr(0)', '"hello"', klong=klong)
        self.assert_eval_cmp('fr(1)', '"hello"', klong=klong)
        klong('fr::{:[x<1;x;.f(x%2)]}')
        self.assert_eval_cmp('fr(0)', '0', klong=klong)
        self.assert_eval_cmp('fr(1)', '0.5', klong=klong)
        self.assert_eval_cmp('fr(12)', '0.75', klong=klong)
        klong("fr::{:[0=x;[];1,.f(x-1)]}")
        self.assert_eval_cmp('fr(0)', '[]', klong=klong)
        self.assert_eval_cmp('fr(1)', '[1]', klong=klong)
        self.assert_eval_cmp('fr(2)', '[1 1]', klong=klong)
        self.assert_eval_cmp('fr(3)', '[1 1 1]', klong=klong)

    def test_harness(self):
        klong = KlongInterpreter()
        klong('err::0;')
        klong('wl::{.w(x);.p("")}')
        klong('fail::{err::1;.d("failed: ");.p(x);.d("expected: ");wl(z);.d("got: ");wl(y);[]}')
        klong('t::{:[~y~z;fail(x;y;z);[]]}')
        r = klong("err")
        self.assertEqual(r, 0)
        klong("A::1")
        klong('t("A::1"                 ; A ; 1)')
        r = klong("err")
        self.assertEqual(r, 0)
        r = klong('t("A::1"                 ; A ; 2)')
        r = klong("err")
        self.assertEqual(r, 1)

    def test_vector_math(self):
        klong = KlongInterpreter()
        r = klong("!10000000")
        x = np.arange(10000000)
        self.assertTrue(np.equal(x, r).all())

        r = klong("1+!10000000")
        x = np.add(x, 1)
        self.assertTrue(np.equal(x, r).all())

        r = klong("2*1+!10000000")
        x = np.multiply(x, 2)
        self.assertTrue(np.equal(x, r).all())

        r = klong("3%2*1+!10000000")
        x = np.divide(3, x)
        self.assertTrue(np.equal(x, r).all())

        r = klong("10-3%2*1+!10000000")
        x = np.subtract(10, x)
        self.assertTrue(np.equal(x, r).all())

        r = klong("(1+!10)&3")
        x = np.minimum(np.add(np.arange(10), 1), 3)
        self.assertTrue(np.equal(x, r).all())

        r = klong("(1+!10)!5")
        x = np.fmod(np.add(np.arange(10), 1), 5)
        self.assertTrue(np.equal(x, r).all())

    def test_converge(self):
        klong = KlongInterpreter()
        r = klong('{(x+2%x)%2}:~2')
        self.assertTrue(np.isclose(r,1.41421356237309504))

    def test_scan_converge(self):
        klong = KlongInterpreter()
        r = klong('{(x+2%x)%2}\~2')
        e = np.asarray([2.        , 1.5       , 1.41666667, 1.41421569])
        self.assertTrue(np.isclose(r,e).all())

        r = klong(',/\~["a" ["b"] "c"]')
        e = np.asarray([["a",["b"],"c"],["a","b","c"],"abc"],dtype=object)
        x = r
        self.assertTrue(rec_flatten(rec_fn2(e,x, lambda a,b: a == b)).all())

    def test_converge_as_fn(self):
        klong = KlongInterpreter()
        klong('s::{(x+2%x)%2}:~2')
        r = klong('s')
        self.assertTrue(np.isclose(r,1.41421356237309504))
        r = klong('s()')
        self.assertTrue(np.isclose(r,1.41421356237309504))
