"""A Python callable of any parameter names and any count is a Klong function; x, y, z keep the 0.7 mapping."""
import functools
import unittest

import numpy as np

from klongpy import KlongInterpreter
from klongpy.errors import KlongArityError


class TestPythonFunctionParameters(unittest.TestCase):

    def setUp(self):
        self.klong = KlongInterpreter()

    def run_kg(self, src):
        return self.klong(src)

    def test_any_names_any_count(self):
        k = self.klong
        k['f'] = lambda a, b, c, d: a + b + c + d
        self.assertEqual(int(self.run_kg('f(1;2;3;4)')), 10)
        k['g'] = lambda first, second: first - second
        self.assertEqual(int(self.run_kg('g(5;3)')), 2)
        k['h'] = lambda one: one + 1
        self.assertEqual(int(self.run_kg('h(5)')), 6)
        k['w'] = lambda a, b, c, d, e, f, g, h: a + b + c + d + e + f + g + h
        self.assertEqual(int(self.run_kg('w(1;2;3;4;5;6;7;8)')), 36)

    def test_arguments_are_positional(self):
        self.klong['f'] = lambda p, q, r, s: p * 1000 + q * 100 + r * 10 + s
        self.assertEqual(int(self.run_kg('f(1;2;3;4)')), 1234)

    def test_xyz_keeps_by_name_mapping(self):
        k = self.klong
        k['f'] = lambda x, y, z: x * 1000 + y - z
        self.assertEqual(int(self.run_kg('f(3;10;20)')), 2990)
        k['g'] = lambda y, x: x - y
        # the call supplies x then y, passed in that order whatever the names
        self.assertEqual(int(self.run_kg('g(5;3)')), -2)
        k['m'] = lambda x: x + 1
        self.assertEqual(int(self.run_kg('m(1)')), 2)
        k['n'] = lambda: 7
        self.assertEqual(int(self.run_kg('n()')), 7)

    def test_xyz_display(self):
        k = self.klong
        k['m'] = lambda x: x
        k['d'] = lambda x, y: x
        k['t'] = lambda x, y, z: x
        self.assertEqual(self.run_kg('.p(m)'), ':monad')
        self.assertEqual(self.run_kg('.p(d)'), ':dyad')
        self.assertEqual(self.run_kg('.p(t)'), ':triad')

    def test_display_follows_klong_convention(self):
        k = self.klong
        k['one'] = lambda a: a
        k['two'] = lambda a, b: a
        k['three'] = lambda a, b, c: a
        self.assertEqual(self.run_kg('.p(one)'), ':monad')
        self.assertEqual(self.run_kg('.p(two)'), ':dyad')
        self.assertEqual(self.run_kg('.p(three)'), ':triad')

    def test_klong_parameter_is_not_an_argument(self):
        k = self.klong
        k['f'] = lambda klong, a, b: int(klong('1+1')) + a + b
        self.assertEqual(int(self.run_kg('f(10;20)')), 32)
        k['g'] = lambda klong, x: int(klong('1+1')) + x
        self.assertEqual(int(self.run_kg('g(10)')), 12)
        k['h'] = lambda klong, a, b, c, d: a + b + c + d
        self.assertEqual(int(self.run_kg('h(1;2;3;4)')), 10)

    def test_too_few_and_too_many_arguments(self):
        k = self.klong
        k['f'] = lambda a, b, c, d: a + b + c + d
        with self.assertRaises(KlongArityError) as e:
            self.run_kg('f(1;2)')
        self.assertIn('f', str(e.exception))
        self.assertIn('4', str(e.exception))
        self.assertIn('2', str(e.exception))
        with self.assertRaises(KlongArityError):
            self.run_kg('f(1;2;3;4;5)')
        k['g'] = lambda a, b: a
        with self.assertRaises(KlongArityError):
            self.run_kg('g(1)')
        with self.assertRaises(KlongArityError):
            self.run_kg('g(1;2;3)')
        k['h'] = lambda x, y: x
        with self.assertRaises(KlongArityError):
            self.run_kg('h(1)')
        with self.assertRaises(KlongArityError):
            self.run_kg('h(1;2;3)')

    def test_arity_error_from_python(self):
        self.klong['f'] = lambda a, b, c, d: a
        with self.assertRaises(KlongArityError):
            self.klong['f'](1, 2)
        self.klong['g'] = lambda a, b: a
        with self.assertRaises(KlongArityError):
            self.klong['g'](1, 2, 3)

    def test_defaults_take_what_the_call_supplies(self):
        k = self.klong
        k['f'] = lambda a, b, c=100: a + b + c
        self.assertEqual(int(self.run_kg('f(1;2)')), 103)
        self.assertEqual(int(self.run_kg('f(1;2;3)')), 6)
        with self.assertRaises(KlongArityError) as e:
            self.run_kg('f(1)')
        self.assertIn('f', str(e.exception))
        with self.assertRaises(KlongArityError):
            self.run_kg('f(1;2;3;4)')
        k['g'] = lambda a, b=1, c=2, d=3, e=4: a + b + c + d + e
        self.assertEqual(int(self.run_kg('g(0)')), 10)
        self.assertEqual(int(self.run_kg('g(0;0;0;0;0)')), 0)

    def test_varargs(self):
        k = self.klong
        k['f'] = lambda *args: sum(args)
        self.assertEqual(int(self.run_kg('f(1;2;3;4;5)')), 15)
        self.assertEqual(int(self.run_kg('f(1)')), 1)
        k['g'] = lambda a, *rest: a + sum(rest)
        self.assertEqual(int(self.run_kg('g(1;2;3;4;5)')), 15)
        with self.assertRaises(KlongArityError):
            self.run_kg('g()')

    def test_keyword_only_parameters_are_not_arguments(self):
        k = self.klong
        k['f'] = lambda a, b, *, scale=10: (a + b) * scale
        self.assertEqual(int(self.run_kg('f(1;2)')), 30)
        with self.assertRaises(KlongArityError):
            self.run_kg('f(1;2;3)')

    def test_builtin_without_signature_takes_what_is_written(self):
        k = self.klong
        k['mx'] = max
        self.assertEqual(int(self.run_kg('mx(3;9;4;1)')), 9)
        k['ln'] = len
        self.assertEqual(int(self.run_kg('ln("abc")')), 3)

    def test_ufunc_takes_its_inputs(self):
        k = self.klong
        k['add'] = np.add
        self.assertEqual(int(self.run_kg('add(2;3)')), 5)
        with self.assertRaises(KlongArityError):
            self.run_kg('add(2)')

    def test_partial_and_method(self):
        k = self.klong

        class C:
            def m(self, p, q, r, s):
                return p + q + r + s
        k['m'] = C().m
        self.assertEqual(int(self.run_kg('m(1;2;3;4)')), 10)
        k['p'] = functools.partial(lambda a, b, c, d: a + b + c + d, 1)
        self.assertEqual(int(self.run_kg('p(2;3;4)')), 10)

    def test_projection(self):
        k = self.klong
        k['f'] = lambda a, b, c, d: a * 1000 + b * 100 + c * 10 + d
        self.run_kg('g::f(1;;3;4)')
        self.assertEqual(int(self.run_kg('g(2)')), 1234)
        self.run_kg('h::f(;;3;4)')
        self.assertEqual(int(self.run_kg('h(1;2)')), 1234)

    def test_projection_matches_klong_function(self):
        k = self.klong
        k['f'] = lambda a, b, c, d: a * 1000 + b * 100 + c * 10 + d
        self.run_kg('fk(a;b;c;d)::{(a*1000)+(b*100)+(c*10)+d}')
        for suffix in ['(1;;3;4)', '(;2;;4)', '(;;;)']:
            self.run_kg('p1::f' + suffix)
            self.run_kg('p2::fk' + suffix)
            self.assertEqual(self.run_kg('.p(p1)'), self.run_kg('.p(p2)'), suffix)

    def test_adverbs_two_parameters(self):
        k = self.klong
        k['f'] = lambda a, b: a * 10 + b
        kf = self.run_kg
        kf('fk(a;b)::{(a*10)+b}')
        self.assertEqual(int(kf('1 f/[2 3 4]')), int(kf('1 fk/[2 3 4]')))
        self.assertEqual(list(map(int, kf('1 f\\[2 3 4]'))), list(map(int, kf('1 fk\\[2 3 4]'))))
        self.assertEqual(list(map(int, kf('[1 2 3] f\'[4 5 6]'))), list(map(int, kf('[1 2 3] fk\'[4 5 6]'))))

    def test_adverbs_one_parameter(self):
        k = self.klong
        k['f'] = lambda a: a * 2
        self.run_kg('fk(a)::{a*2}')
        self.assertEqual(list(map(int, self.run_kg('f\'[1 2 3]'))), [2, 4, 6])
        self.assertEqual(list(map(int, self.run_kg('f\'[1 2 3]'))), list(map(int, self.run_kg('fk\'[1 2 3]'))))

    def test_dictionary_of_functions(self):
        k = self.klong
        k['d'] = {'sum4': lambda a, b, c, d: a + b + c + d, 'inc': lambda v: v + 1}
        self.run_kg('s::d?"sum4"')
        self.assertEqual(int(self.run_kg('s(1;2;3;4)')), 10)
        self.run_kg('i::d?"inc"')
        self.assertEqual(int(self.run_kg('i(1)')), 2)
        self.assertEqual(self.run_kg('.p(i)'), ':monad')

    def test_dictionary_function_comes_back_as_itself(self):
        fn = lambda a, b: a + b
        self.klong['d'] = {'f': fn}
        self.assertIs(self.klong['d']['f'], fn)

    def test_callback_argument(self):
        k = self.klong
        self.run_kg('ap::{x(1;2;3;4)}')
        self.assertEqual(int(k['ap'](lambda a, b, c, d: a + b + c + d)), 10)
        self.assertEqual(int(k['ap'](lambda x, y, z, w: x + y + z + w)), 10)
        self.assertEqual(int(k['ap'](lambda *a: sum(a))), 10)
        self.run_kg('one::{x(7)}')
        self.assertEqual(int(k['one'](lambda v: v + 1)), 8)
        self.run_kg('pd::{.p(x)}')
        self.assertEqual(k['pd'](lambda a, b: a), ':dyad')

    def test_callback_arity_error(self):
        self.run_kg('ap::{x(1;2)}')
        with self.assertRaises(KlongArityError):
            self.klong['ap'](lambda a, b, c: a)

    def test_callback_adverb(self):
        self.run_kg('m::{x@[1 2 3]}')
        self.run_kg('ev::{x\'[1 2 3]}')
        self.assertEqual(list(map(int, self.klong['ev'](lambda value: value * 3))), [3, 6, 9])
        self.run_kg('fo::{x/[1 2 3 4]}')
        self.assertEqual(int(self.klong['fo'](lambda acc, v: acc + v)), 10)

    def test_callable_returned_by_klong_receives_arguments(self):
        self.run_kg('f(a;b;c;d)::{a+b+c+d}')
        f = self.klong['f']
        self.assertEqual(int(f(1, 2, 3, 4)), 10)

    def test_python_callable_through_klong_roundtrip(self):
        fn = lambda a, b, c, d: a + b + c + d
        self.klong['f'] = fn
        self.run_kg('id::{x}')
        back = self.klong['id'](fn)
        self.assertIs(back, fn)

    def test_module_import(self):
        import os
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, 'pyparams_mod.py'), 'w') as fh:
                fh.write('def add4(a, b, c, d):\n    return a + b + c + d\n'
                         'def sub(p, q):\n    return p - q\n'
                         'def old(x, y):\n    return x - y\n')
            sys.path.insert(0, d)
            try:
                self.run_kg('.pyf("pyparams_mod";"sub")')
                self.run_kg('.pyf("pyparams_mod";"old")')
                self.assertEqual(int(self.run_kg('sub(9;4)')), 5)
                self.assertEqual(int(self.run_kg('old(9;4)')), 5)
                self.run_kg('.py("pyparams_mod")')
                self.assertEqual(int(self.run_kg('sub(9;4)')), 5)
            finally:
                sys.path.remove(d)
                sys.modules.pop('pyparams_mod', None)

    def test_tensor_arguments(self):
        k = self.klong
        k['f'] = lambda a, b, c, d: a + b + c + d
        r = self.run_kg('f([1 2];[3 4];[5 6];[7 8])')
        self.assertEqual([int(v) for v in r], [16, 20])
        self.klong['g'] = lambda xs, n: xs * n
        r = self.run_kg('g([1 2 3];2)')
        self.assertEqual([int(v) for v in r], [2, 4, 6])


if __name__ == '__main__':
    unittest.main()
