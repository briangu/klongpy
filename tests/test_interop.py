import random
import unittest
from datetime import datetime

from utils import *
from backend_compat import requires_strings, requires_object_dtype

from klongpy import KlongInterpreter


class TestPythonInterop(unittest.TestCase):

    def test_python_lambda_as_argument(self):
        """
        Test that a python lambda can be passed as an argument to a klong function.
        """
        klong = KlongInterpreter()
        klong['fn'] = lambda x: x+10
        klong('foo::{x(2)}')
        r = klong('foo(fn)')
        self.assertEqual(r, 12)

    def test_ext_var(self):
        klong = KlongInterpreter()
        x = random.random()
        klong['foo'] = x
        self.assertEqual(klong['foo'],x)
        # test reassignment
        x = random.random()
        x = [x]
        klong['foo'] = x
        self.assertTrue(kg_equal(klong['foo'],x))

    def test_del_var(self):
        klong = KlongInterpreter()
        with self.assertRaises(KeyError):
            del klong['foo']
        klong['foo'] = 1
        del klong['foo']
        with self.assertRaises(KeyError):
            del klong['foo']

    def test_python_lambda(self):
        klong = KlongInterpreter()
        klong['fn'] = lambda x: x+10
        self.assertEqual(klong('fn(2)'), 12)

    def test_call_klong_python_lambda(self):
        klong = KlongInterpreter()
        klong['fn'] = lambda x: x+10
        fn = klong['fn']
        self.assertEqual(fn(1),11)

    def test_invalid_call_klong_python_lambda(self):
        klong = KlongInterpreter()
        klong['fn'] = lambda x: x+10
        fn = klong['fn']
        with self.assertRaises(RuntimeError):
            self.assertEqual(fn(1,2),11)

    def test_call_klong_fn_with_scalar(self):
        klong = KlongInterpreter()
        klong('fn::{x+10}')
        r = klong['fn'](2)
        self.assertEqual(r, 12)

    def test_call_klong_fn_with_multiple_scalars(self):
        klong = KlongInterpreter()
        klong = KlongInterpreter()
        klong("fn::{(x*1000) + y - z}")
        r = klong['fn'](3, 10, 20)
        self.assertEqual(r, 2990)

    def test_call_klong_fn_with_array(self):
        klong = KlongInterpreter()
        klong("fn::{{x+10}'x}")
        r = klong['fn']([2])
        self.assertTrue(kg_equal(r, [12]))

    @requires_strings
    def test_datetime_parsing_example(self):
        klong = KlongInterpreter()
        klong['strptime'] = lambda x: datetime.strptime(x, "%d %B, %Y")
        r = klong('a::strptime("21 June, 2018")')
        self.assertEqual(r, datetime(2018, 6, 21, 0, 0))
        r = klong['a']
        self.assertEqual(r, datetime(2018, 6, 21, 0, 0))
        r = klong('d:::{};d,"timestamp",a')
        self.assertEqual(r, {'timestamp': datetime(2018, 6, 21, 0, 0)})

    @requires_strings
    def test_datetime_parsing_example_one_call(self):
        # run everything in one go
        klong = KlongInterpreter()
        klong['strptime'] = lambda x: datetime.strptime(x, "%d %B, %Y")
        r = klong("""a::strptime("21 June, 2018");d:::{};d,"timestamp",a""")
        self.assertEqual(r, {'timestamp': datetime(2018, 6, 21, 0, 0)})
        r = klong['a']
        self.assertEqual(r, datetime(2018, 6, 21, 0, 0))

    @requires_strings
    def test_callback_into_assertion(self):
        def assert_date(x):
            self.assertEqual(x, {'timestamp': datetime(2018, 6, 21, 0, 0)})
            return 42
        klong = KlongInterpreter()
        klong['strptime'] = lambda x: datetime.strptime(x, "%d %B, %Y")
        klong['assert'] = assert_date
        r = klong("""a::strptime("21 June, 2018");d:::{};d,"timestamp",a;assert(d)""")
        self.assertEqual(r, 42)

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

    @requires_object_dtype
    def test_lambda_projection(self):
        klong = KlongInterpreter()
        klong['f'] = lambda x, y, z: ((x*1000) + y) - z
        klong("g::f(3;;)") # TODO: can we make the y and z vals implied None?
        r = klong('g(10;20)')
        self.assertEqual(r, ((3 * 1000) + 10) - 20)
        klong("h::g(10;)")
        r = klong('h(20)')
        self.assertEqual(r, ((3 * 1000) + 10) - 20)

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

    def test_avg_mixed(self):
        data = np.random.rand(10**5)
        klong = KlongInterpreter()
        klong('avg::{(+/x)%#x}')
        klong['data'] = data
        start = time.perf_counter_ns()
        r = klong('avg(data)')
        stop = time.perf_counter_ns()
        # print((stop - start) / (10**9))
        # Use tolerance-based comparison for float32 precision
        r_val = r.item() if hasattr(r, 'item') else r
        self.assertTrue(abs(r_val - np.average(data)) < 1e-3)

    @requires_object_dtype
    def test_join_np_array_and_list(self):
        klong = KlongInterpreter()
        klong("A::[];A::A,:{};A::A,:{}")
        klong['B'] = [{}, {}, {}]
        r = klong("A,B")
        self.assertTrue(kg_equal(r, [{}, {}, {}, {}, {}]))
