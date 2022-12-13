import unittest
from klongpy import KlongInterpreter
from klongpy.core import rec_flatten, rec_fn2, KGChar, KGSym, is_integer, is_float
from utils import *
import time

# add tests not included in the original kg suite
class TestExtraCoreSuite(unittest.TestCase):

    def assert_eval_cmp(self, a, b, klong=None):
        self.assertTrue(eval_cmp(a, b, klong=klong))

    # TODO: should this be supported?
    # ["1234","5678"]:@[0 1]

    # TODO: why isn't this supported in Klong language?
    # [[1 2 3 4] [5 6 7]]:=[3 4],0
    # the right hand side is interpreter as a join, making it
    # [[1 2 3 4] [5 6 7]]:=[3 4 0]
    # which means 4 becomes the position and 0 is ignored
    # this works:
    # [[1 2 3 4] [5 6 7]]:=[[3 4]],0

    # read 123456 from "123456 hello" requires parsing by space
    # append dict to array value in a dict

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

    def test_dict_find_zero_value(self):
        klong = KlongInterpreter()
        klong('D:::{[:s 0]}')
        self.assertEqual(klong('D?:s'), 0)

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
        self.assertTrue(array_equal(klong("q?:c"), []))
        self.assertEqual(klong("q?:n"), "hello")
        self.assertTrue(isinstance(klong("q?:p"), dict))

    def test_join_sym_int(self):
        klong = KlongInterpreter()
        r = klong(":p,43")
        self.assertTrue(isinstance(r[0],KGSym))
        self.assertTrue(isinstance(r[1],int))

    def test_symbol_dict_key(self):
        klong = KlongInterpreter()
        klong("D:::{[:s 42]}")
        r = klong("D?:s")
        self.assertEqual(r, 42)
        klong("N::{d:::{[:s 0]};d,:p,x;d}")
        klong('P::N(43)')
        r = klong('P?:p')
        self.assertEqual(r,43)

    @unittest.skip
    def test_wrap_fn(self):
        klong = KlongInterpreter()
        klong('L::{(*(x?y))#x};A::L(;",");LL::{.rs(L(x;"-"))}')
        r = klong('A("20-45,13-44")')
        self.assertEqual(r,"20-45")
        r = klong('L(A("20-45,13-44");"-")')
        self.assertEqual(r,"20")
        r = klong('.rs(L(A("20-45,13-44");"-"))')
        self.assertEqual(r,20)
        r = klong('q::L(A("20-45,13-44");"-");.rs(q)')
        self.assertEqual(r,20)
        r = klong('LL(A("20-45,13-44"))')
        self.assertEqual(r,20)

    @unittest.skip
    def test_x_exposure_should_not_collide(self):
        klong = KlongInterpreter()
        klong("I::{{#x}'x}")
        r = klong('I("hello"')
        self.assertTrue(array_equal(r,[104,101,108,108,111]))


    def test_grade_down_with_empty_subarrays(self):
        klong = KlongInterpreter()
        klong("P::{q::y;#x@*>{q?x}'x}")
        r = klong('P("vJrwpWtwJgWr";"hcsFMMfFFhFp")')
        self.assertEqual(r,112)

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

    def test_cond_arr_predicate(self):
        klong = KlongInterpreter()
        r = klong(':["XYZ"?"X";1;0]')
        self.assertTrue(r,1)

    def test_avg_mixed(self):
        data = np.random.rand(10**5)
        klong = KlongInterpreter()
        klong('avg::{(+/x)%#x}')
        klong['data'] = data
        start = time.perf_counter_ns()
        r = klong('avg(data)')
        stop = time.perf_counter_ns()
        print((stop - start) / (10**9))
        self.assertEqual(r, np.average(data))

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

    def test_dyad_uneven_match(self):
        klong = KlongInterpreter()
        r = klong('[]~[1 2 3]')
        self.assertEqual(r,0)
        r = klong('1~[1 2 3]')
        self.assertEqual(r,0)

    def test_read_enotation(self):
        klong = KlongInterpreter()
        r = klong('t::1.0e8;')
        self.assertEqual(r, 1.0e8)
        r = klong('t::1.0e-8;')
        self.assertEqual(r, 1.0e-8)

    def test_read_list(self):
        klong = KlongInterpreter()
        r = klong('[]')
        self.assertTrue(array_equal(r,[]))
        klong = KlongInterpreter()
        r = klong('[1 2 3 4]')
        self.assertTrue(array_equal(r,[1,2,3,4]))
        klong = KlongInterpreter()
        r = klong('[1 2 3 4 ]') # add spaces as found in suite
        self.assertTrue(array_equal(r,[1,2,3,4]))

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
        self.assertTrue(array_equal(r,[]))
        r = klong(t)
        self.assertTrue(array_equal(r,[["hello"]]))

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
        self.assertTrue(array_equal(r,[1,1]))

    def test_read_sym_f0(self):
        klong = KlongInterpreter()
        r = klong("f0::1; f0")
        self.assertEqual(r, 1)

    def test_eval_array(self):
        klong = KlongInterpreter()
        r = klong('[[[1] [2] [3]] [1 2 3]]')
        self.assertTrue(array_equal(r,[[[1],[2],[3]],[1,2,3]]))

    def test_atom(self):
        self.assert_eval_cmp('@0c0', '1')

    def test_join(self):
        self.assert_eval_cmp('[1 2],:{[1 0]}', ':{[1 2]}')
        self.assert_eval_cmp(':{[1 0]},[1 2]', ':{[1 2]}')

    def test_builtin_rn(self):
        klong = KlongInterpreter()
        r = klong(".rn()")
        self.assertTrue(r >= 0 and r <= 1.0)
        r2 = klong(".rn()")
        self.assertTrue(r2 >= 0 and r2 <= 1.0)
        self.assertTrue(r != r2)

    def test_setitem(self):
        klong = KlongInterpreter()
        klong['A'] = 1
        r = klong("A")
        self.assertEqual(r, 1)
        self.assertEqual(klong['A'], 1)

    def test_define_simple(self):
        klong = KlongInterpreter()
        r = klong("A::1; A")
        self.assertEqual(r, 1)
        self.assertEqual(klong['A'], 1)

    def test_define_reassign(self):
        klong = KlongInterpreter()
        klong('A::1')
        r = klong('A')
        self.assertEqual(r, 1)
        klong('A::2')
        r = klong('A')
        self.assertEqual(r, 2)

    def test_fn_nilad(self):
        klong = KlongInterpreter()
        klong("f::{1000}")
        r = klong('f()')
        self.assertEqual(r, 1000)

    def test_fn_monad(self):
        klong = KlongInterpreter()
        klong("f::{x*1000}")
        r = klong('f(3)')
        self.assertEqual(r, 3000)

    def test_fn_dyad(self):
        klong = KlongInterpreter()
        klong("f::{(x*1000) + y}")
        r = klong('f(3;10)')
        self.assertEqual(r, (3 * 1000) + 10)

    def test_fn_triad(self):
        klong = KlongInterpreter()
        klong("f::{((x*1000) + y) - z}")
        r = klong('f(3; 10; 20)')
        self.assertEqual(r, ((3 * 1000) + 10) - 20)

    def test_fn_projection(self):
        klong = KlongInterpreter()
        klong("f::{((x*1000) + y) - z}")
        klong("g::f(3;;)")
        r = klong('g(10; 20)')
        self.assertEqual(r, ((3 * 1000) + 10) - 20)
        klong("h::g(10;)")
        r = klong('h(20)')
        self.assertEqual(r, ((3 * 1000) + 10) - 20)

    def test_lambda_nilad(self):
        klong = KlongInterpreter()
        klong['f'] = lambda: 1000
        r = klong('f()')
        self.assertEqual(r, 1000)

    def test_lambda_monad(self):
        klong = KlongInterpreter()
        klong['f'] = lambda x: x*1000
        r = klong('f(3)')
        self.assertEqual(r, 3000)

    def test_lambda_dyad(self):
        klong = KlongInterpreter()
        klong['f'] = lambda x, y: x*1000 + y
        r = klong('f(3;10)')
        self.assertEqual(r, 3 * 1000 + 10)

    def test_lambda_triad(self):
        klong = KlongInterpreter()
        klong['f'] = lambda x, y, z: x*1000 + y - z
        r = klong('f(3; 10; 20)')
        self.assertEqual(r, 3 * 1000 + 10 - 20)

    def test_lambda_projection(self):
        klong = KlongInterpreter()
        klong['f'] = lambda x, y, z: ((x*1000) + y) - z
        klong("g::f(3;;)") # TODO: can we make the y and z vals implied None?
        r = klong('g(10;20)')
        self.assertEqual(r, ((3 * 1000) + 10) - 20)
        klong("h::g(10;)")
        r = klong('h(20)')
        self.assertEqual(r, ((3 * 1000) + 10) - 20)

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

    def test_over(self):
        klong = KlongInterpreter()
        self.assertEqual(klong('+/!5'), np.add.reduce(np.arange(5)))
        self.assertEqual(klong('-/!5'), np.subtract.reduce(np.arange(5)))
        self.assertEqual(klong('*/1+!5'), np.multiply.reduce(1+np.arange(5)))
        self.assertEqual(klong('%/[1 2 3]'), np.divide.reduce([1,2,3]))

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
        e = np.asarray([["a",["b"],"c"],["a","b","c"],"abc"])
        x = r
        self.assertTrue(rec_flatten(rec_fn2(e,x, lambda a,b: a == b)).all())

    def test_converge_as_fn(self):
        klong = KlongInterpreter()
        klong('s::{(x+2%x)%2}:~2')
        r = klong('s')
        self.assertTrue(np.isclose(r,1.41421356237309504))
        r = klong('s()')
        self.assertTrue(np.isclose(r,1.41421356237309504))
