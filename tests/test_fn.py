import inspect
import random
import string
import unittest

from utils import *

from klongpy import KlongInterpreter
from klongpy.core import in_map, reserved_fn_symbol_map


def wrap_lambda(arity, fn, fn_dict):
    if arity == 0:
        if len(fn_dict) == 0:
            return lambda: fn()
        elif 'm' in fn_dict and 'd' in fn_dict and 't' in fn_dict:
            return lambda m=fn_dict['m'],d=fn_dict['d'],t=fn_dict['t']: fn(m,d,t)
        elif 'm' in fn_dict and 'd' in fn_dict:
            return lambda m=fn_dict['m'],d=fn_dict['d']: fn(m,d)
        elif 'm' in fn_dict and 't' in fn_dict:
            return lambda m=fn_dict['m'],t=fn_dict['t']: fn(m,t)
        elif 'd' in fn_dict and 't' in fn_dict:
            return lambda d=fn_dict['d'],t=fn_dict['t']: fn(d,t)
        elif 'm' in fn_dict:
            return lambda m=fn_dict['m']: fn(m)
        elif 'd' in fn_dict:
            return lambda d=fn_dict['d']: fn(d)
        elif 't' in fn_dict:
            return lambda t=fn_dict['t']: fn(t)
    elif arity == 1:
        if len(fn_dict) == 0:
            return lambda x: fn(x)
        elif 'm' in fn_dict and 'd' in fn_dict and 't' in fn_dict:
            return lambda x,m=fn_dict['m'],d=fn_dict['d'],t=fn_dict['t']: fn(x,m=m,d=d,t=t)
        elif 'm' in fn_dict and 'd' in fn_dict:
            return lambda x,m=fn_dict['m'],d=fn_dict['d']: fn(x,m=m,d=d)
        elif 'm' in fn_dict and 't' in fn_dict:
            return lambda x,m=fn_dict['m'],t=fn_dict['t']: fn(x,m=m,t=t)
        elif 'd' in fn_dict and 't' in fn_dict:
            return lambda x,d=fn_dict['d'],t=fn_dict['t']: fn(x,d=d,t=t)
        elif 'm' in fn_dict:
            return lambda x,m=fn_dict['m']: fn(x,m=m)
        elif 'd' in fn_dict:
            return lambda x,d=fn_dict['d']: fn(x,d=d)
        elif 't' in fn_dict:
            return lambda x,t=fn_dict['t']: fn(x,t=t)
    elif arity == 2:
        if len(fn_dict) == 0:
            return lambda x,y: fn(x,y)
        elif 'm' in fn_dict and 'd' in fn_dict and 't' in fn_dict:
            return lambda x,y,m=fn_dict['m'],d=fn_dict['d'],t=fn_dict['t']: fn(x,y,m=m,d=d,t=t)
        elif 'm' in fn_dict and 'd' in fn_dict:
            return lambda x,y,m=fn_dict['m'],d=fn_dict['d']: fn(x,y,m=m,d=d)
        elif 'm' in fn_dict and 't' in fn_dict:
            return lambda x,y,m=fn_dict['m'],t=fn_dict['t']: fn(x,y,m=m,t=t)
        elif 'd' in fn_dict and 't' in fn_dict:
            return lambda x,y,d=fn_dict['d'],t=fn_dict['t']: fn(x,y,d=d,t=t)
        elif 'm' in fn_dict:
            return lambda x,y,m=fn_dict['m']: fn(x,y,m=m)
        elif 'd' in fn_dict:
            return lambda x,y,d=fn_dict['d']: fn(x,y,d=d)
        elif 't' in fn_dict:
            return lambda x,y,t=fn_dict['t']: fn(x,y,t=t)
    elif arity == 3:
        if len(fn_dict) == 0:
            return lambda x,y,z: fn(x,y,z)
        elif 'm' in fn_dict and 'd' in fn_dict and 't' in fn_dict:
            return lambda x,y,z,m=fn_dict['m'],d=fn_dict['d'],t=fn_dict['t']: fn(x,y,z,m=m,d=d,t=t)
        elif 'm' in fn_dict and 'd' in fn_dict:
            return lambda x,y,z,m=fn_dict['m'],d=fn_dict['d']: fn(x,y,z,m=m,d=d)
        elif 'm' in fn_dict and 't' in fn_dict:
            return lambda x,y,z,m=fn_dict['m'],t=fn_dict['t']: fn(x,y,z,m=m,t=t)
        elif 'd' in fn_dict and 't' in fn_dict:
            return lambda x,y,z,d=fn_dict['d'],t=fn_dict['t']: fn(x,y,z,d=d,t=t)
        elif 'm' in fn_dict:
            return lambda x,y,z,m=fn_dict['m']: fn(x,y,z,m=m)
        elif 'd' in fn_dict:
            return lambda x,y,z,d=fn_dict['d']: fn(x,y,z,d=d)
        elif 't' in fn_dict:
            return lambda x,y,z,t=fn_dict['t']: fn(x,y,z,t=t)


class SymbolGenerator:
    def __init__(self):
        self.a = [n for n in list(string.ascii_lowercase) if not in_map(n, ['x','y','z','m','d','t'])]
    def __iter__(self):
        for n in self.a:
            yield n
        for n in self.a:
            for m in self.a:
                yield n+m


class TestFunctionsSuite(unittest.TestCase):

    def assert_eval_cmp(self, a, b, klong=None):
        self.assertTrue(eval_cmp(a, b, klong=klong))

    def test_fn_nilad(self):
        klong = KlongInterpreter()
        klong("F::{1}")
        r = klong('F()')
        self.assertEqual(r, 1)

    def test_fn_nilad_nested_monad(self):
        klong = KlongInterpreter()
        klong("F::{.p(1)}")
        r = klong('F()')
        self.assertEqual(r, '1')

    def test_fn_nilad_nested_monad_2(self):
        klong = KlongInterpreter()
        r = klong('bar::{x};foo::{bar("+")};foo()')
        self.assertEqual(r,'+')

    def test_fn_monad_gen(self):
        klong = KlongInterpreter()
        fns = []
        alpha = [n for n in list(string.ascii_lowercase) if not in_map(n, reserved_fn_symbol_map)]
        fns = "a::{x};"+";".join([f"{q}::{p}(x)" for p,q in zip(alpha[:-1],alpha[1:])])
        klong(fns)
        r = klong(f'{alpha[-1]}(1)')
        self.assertEqual(r,1)

    def test_fn_monad(self):
        klong = KlongInterpreter()
        klong("F::{x}")
        r = klong('F(1)')
        self.assertEqual(r, 1)

    def test_fn_monad_2(self):
        klong = KlongInterpreter()
        klong("F::{x,x}")
        r = klong('F(1)')
        self.assertTrue(array_equal(r, [1,1]))

    def test_fn_nested_monad(self):
        klong = KlongInterpreter()
        klong('G::{x};F::{G(x)}')
        r = klong('F(1)')
        self.assertEqual(r, 1)

    def test_fn_nested_monad_w_xform(self):
        klong = KlongInterpreter()
        klong('G::{x};F::{G(4_x)}')
        r = klong('F("hello")')
        self.assertEqual(r, 'o')

    def test_fn_nested_x_scope(self):
        klong = KlongInterpreter()
        klong("FL:::{};FL,0,{.p(,x@1)};F::{f::FL?0;f(x)}")
        r = klong('F("hello")')
        self.assertEqual(r, "e")

    def test_nested_x_scope_3(self):
        klong = KlongInterpreter()
        klong("G::{.p(,x@0)};F::{G(7_x)}")
        r = klong('F("Monkey 0:")')
        self.assertEqual(r, "0")

    def test_nested_x_scope_compact(self):
        klong = KlongInterpreter()
        klong("F::{{.p(,x@0)}(7_x)}")
        r = klong('F("Monkey 0:")')
        self.assertEqual(r, "0")

    def test_nested_x_scope_4(self):
        klong = KlongInterpreter()
        r = klong('{.p(,x@0)}(7_"Monkey 0:")')
        self.assertEqual(r, "0")

    def test_nested_x_scope_projection(self):
        klong = KlongInterpreter()
        klong('UM::{x};G::UM;F::{G(4_x)}')
        r = klong('F("hello")')
        self.assertEqual(r, "o")

    def test_nested_x_scope_dyad_projection(self):
        klong = KlongInterpreter()
        klong('UM::{x;y};G::UM("A";);F::{G(4_x)}')
        r = klong('F("hello")')
        self.assertEqual(r, "o")

    # @unittest.skip
    def test_fn_gen(self):
        def create_symbols():
            return iter(SymbolGenerator())

        def gensym(symbols):
            return next(symbols)

        def fns(e,n,s):
            return e.replace(n,s)

        fn_map = {
            0: [
                ("{0}", lambda: 0),
                ("{m(1)}", lambda m: m(1), 0),
                # ("{d}", lambda d: d, 2),
                # ("{d(1;)}", lambda d: lambda x: d(1,x), 1),
                ("{d(1;2)}", lambda d: d(1,2), 0),
                # ("{t}", lambda t: t, 3),
                # ("{t(1;)}", lambda t: lambda x,y: t(1,x,y), 2),
                # ("{t(1;2;)}", lambda t: lambda x: t(1,2,x), 1),
                ("{t(1;2;3)}", lambda t: t(1,2,3), 0),
            ],
            1: [
                ("{x}", lambda x: x, 0),
                # ("{.p(x)}", lambda x: str(x), 0),
                # ("{d(x;)}", lambda d,x: lambda x1: d(x, x1), 1),
                ("{d(x;x)}", lambda x,d: d(x,x), 0),
                ("{d(1;x)}", lambda x,d: d(1, x), 0),
                ("{d(x;1)}", lambda x,d: d(x, 1), 0),
                # ("{t(x;;)}", lambda t,x: lambda x1,y: t(x,x1,y), 2),
                # ("{t(x;x;)}", lambda t,x: lambda x1: t(x,x,x1), 1),
                ("{t(x;x;x)}", lambda x,t: t(x,x,x), 0),
                ("{t(1;x;x)}", lambda x,t: t(1,x,x), 0),
                ("{t(1;1;x)}", lambda x,t: t(1,1,x), 0),
            ],
            2: [
                ("{x+y}", lambda x,y: x+y),
                ("{x+m(y)}", lambda x,y,m: x+m(y), 0),
                ("{m(x)+m(y)}", lambda x,y,m: m(x)+m(y), 0),
                # ("{t(x;y;)}", lambda t,x,y: lambda x1: t(x,y,x1), 1),
                ("{t(x;y;y)}", lambda x,y,t: t(x,y,y), 0),
            ],
            3: [
                ("{x+y+z}", lambda x,y,z: x+y+z, 0),
                ("{m(x)+d(y;z)}", lambda x,y,z,m,d: m(x) + d(y,z), 0),
                # ("{d(m(x+y+z);)}", lambda x,y,z,m,d: lambda x1: d(m(x)+y+z,x1), 1)
            ]
        }

        def gen_fn(arity,symbols,fn_name=None):
            arr = []
            choices = fn_map[arity]
            q = choices[random.randint(0,len(choices)-1)]
            fnd = q[0]
            # print(fnd)
            fn_name = fn_name or gensym(symbols)
            if fn_name is None:
                return arr
            args = inspect.getargspec(q[1]).args
            fn_dict = {}
            if len(args) == 0:
                pass
            else:
                if 'm' in args:
                    s = gensym(symbols)
                    fnd = fns(fnd,'m',s)
                    _,aa,ff = gen_fn(1,symbols,fn_name=s)
                    fn_dict['m'] = ff
                    arr.extend(aa)
                if 'd' in args:
                    s = gensym(symbols)
                    fnd = fns(fnd,'d',s)
                    _,aa,ff = gen_fn(2,symbols,fn_name=s)
                    fn_dict['d'] = ff
                    arr.extend(aa)
                if 't' in args:
                    s = gensym(symbols)
                    fnd = fns(fnd,'t',s)
                    _,aa,ff = gen_fn(3,symbols,fn_name=s)
                    fn_dict['t'] = ff
                    arr.extend(aa)
            fn = f"{fn_name}::{fnd}"
            arr.append(fn)
            return fn_name,arr,wrap_lambda(arity,q[1],fn_dict)

        random.seed(0)
        seen = set()
        for arity in range(4):
            for _ in range(100):
                # print()
                fn_name, arr, fn = gen_fn(arity,create_symbols())
                klong = KlongInterpreter()
                stmt = "; ".join(arr)
                if stmt in seen:
                    continue
                seen.add(stmt)
                print(stmt,end=' ==> ')
                klong(stmt)
                # [print(x, klong(x)) for x in arr]
                if arity == 0:
                    fr = fn()
                    stmt = f"{fn_name}()"
                    # print(stmt)
                    kr = klong(stmt)
                else:
                    args = [1,2,3][:arity]
                    fr = fn(*args)
                    stmt = f"{fn_name}({';'.join([str(x) for x in args])})"
                    # print(stmt)
                    kr = klong(stmt)
                print(kr,fr)
                self.assertEqual(kr,fr)
        print(f"ran {len(seen)} tests.")