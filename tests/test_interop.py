import random
import unittest
from datetime import datetime

from utils import *

from klongpy import KlongInterpreter
from klongpy.core import KGCall


class TestPythonInterop(unittest.TestCase):

    def test_ext_var(self):
        klong = KlongInterpreter()
        x = random.random()
        klong['foo'] = x
        self.assertEqual(klong['foo'],x)
        x = [x]
        klong['foo'] = x
        self.assertTrue(array_equal(klong['foo'],x))

    @unittest.skip
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
        self.assertTrue(array_equal(r, [12]))

    def test_datetime_parsing_example(self):
        klong = KlongInterpreter()
        klong['strptime'] = lambda x: datetime.strptime(x, "%d %B, %Y")
        r = klong('a::strptime("21 June, 2018")')
        self.assertEquals(r, datetime(2018, 6, 21, 0, 0))
        r = klong['a']
        self.assertEquals(r, datetime(2018, 6, 21, 0, 0))
        r = klong('d:::{};d,"timestamp",a')
        self.assertEqual(r, {'timestamp': datetime(2018, 6, 21, 0, 0)})

    def test_datetime_parsing_example_one_call(self):
        # run everything in one go
        klong = KlongInterpreter()
        klong['strptime'] = lambda x: datetime.strptime(x, "%d %B, %Y")
        r = klong("""a::strptime("21 June, 2018");d:::{};d,"timestamp",a""")
        self.assertEqual(r, {'timestamp': datetime(2018, 6, 21, 0, 0)})
        r = klong['a']
        self.assertEquals(r, datetime(2018, 6, 21, 0, 0))

    def test_callback_into_assertion(self):
        def assert_date(x):
            self.assertEqual(x, {'timestamp': datetime(2018, 6, 21, 0, 0)})
            return 42
        klong = KlongInterpreter()
        klong['strptime'] = lambda x: datetime.strptime(x, "%d %B, %Y")
        klong['assert'] = assert_date
        r = klong("""a::strptime("21 June, 2018");d:::{};d,"timestamp",a;assert(d)""")
        self.assertEquals(r, 42)
