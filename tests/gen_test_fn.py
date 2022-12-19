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
        for n in self.a:
            for m in self.a:
                for o in self.a:
                    yield n+m+o
        for n in self.a:
            for m in self.a:
                for o in self.a:
                    for p in self.a:
                        yield n+m+o+p


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
        ("{d(1;m(2))}", lambda m,d: d(1,m(2)), 0),
        ("{d(m(2);1)}", lambda m,d: d(m(2),1), 0),
        # ("{t}", lambda t: t, 3),
        # ("{t(1;)}", lambda t: lambda x,y: t(1,x,y), 2),
        # ("{t(1;2;)}", lambda t: lambda x: t(1,2,x), 1),
        ("{t(1;2;3)}", lambda t: t(1,2,3), 0),
        ("{t(m(1);2;3)}", lambda m,t: t(m(1),2,3), 0),
        ("{t(1;m(2);3)}", lambda m,t: t(1,m(2),3), 0),
        ("{t(1;2;m(3))}", lambda m,t: t(1,2,m(3)), 0),
        ("{t(1;d(1;2);m(3))}", lambda m,d,t: t(1,d(1,2),m(3)), 0),
        ("{t(d(1;2);m(2);3)}", lambda m,d,t: t(d(1,2),m(2),3), 0),
        ("{t(d(1;2);2;m(3))}", lambda m,d,t: t(d(1,2),2,m(3)), 0),
        ("{t(1;m(2);d(1;2))}", lambda m,d,t: t(1,m(2),d(1,2)), 0),
        ("{t(m(1);d(1;2);3)}", lambda m,d,t: t(m(1),d(1,2),3), 0),
        ("{t(m(1);2;d(1;2))}", lambda m,d,t: t(m(1),2,d(1,2)), 0),
    ],
    1: [
        ("{x}", lambda x: x, 0),
        ("{m(x)}", lambda x,m: m(x), 0),
        ("{x;m(x)}", lambda x,m: m(x), 0),
        ("{x;m(2*x)}", lambda x,m: m(2*x), 0),
        # ("{.p(x)}", lambda x: str(x), 0),
        # ("{d(x;)}", lambda d,x: lambda x1: d(x, x1), 1),
        ("{d(x;x)}", lambda x,d: d(x,x), 0),
        ("{d(1;x)}", lambda x,d: d(1, x), 0),
        ("{d(x;1)}", lambda x,d: d(x, 1), 0),
        ("{d(x;m(x))}", lambda x,m,d: d(x, m(x)), 0),
        ("{d(m(x);x)}", lambda x,m,d: d(m(x), x), 0),
        # ("{t(x;;)}", lambda t,x: lambda x1,y: t(x,x1,y), 2),
        # ("{t(x;x;)}", lambda t,x: lambda x1: t(x,x,x1), 1),
        ("{t(x;x;x)}", lambda x,t: t(x,x,x), 0),
        ("{t(1;x;x)}", lambda x,t: t(1,x,x), 0),
        ("{t(x;1;x)}", lambda x,t: t(x,1,x), 0),
        ("{t(x;x;1)}", lambda x,t: t(x,x,1), 0),
        ("{t(x;1;x)}", lambda x,t: t(x,1,x), 0),
        ("{t(1;2;x)}", lambda x,t: t(1,2,x), 0),
        ("{t(1;x;2)}", lambda x,t: t(1,x,2), 0),
        ("{t(2;x;1)}", lambda x,t: t(2,x,1), 0),
        ("{t(x;1;2)}", lambda x,t: t(x,1,2), 0),
        ("{t(x;2;1)}", lambda x,t: t(x,2,1), 0),
    ],
    2: [
        ("{x+y}", lambda x,y: x+y),
        ("{x+m(y)}", lambda x,y,m: x+m(y), 0),
        ("{m(x)+m(y)}", lambda x,y,m: m(x)+m(y), 0),
        # ("{t(x;y;)}", lambda t,x,y: lambda x1: t(x,y,x1), 1),
        ("{t(x;y;y)}", lambda x,y,t: t(x,y,y), 0),
        ("{t(x;y;m(y))}", lambda x,y,m,t: t(x,y,m(y)), 0),
        ("{t(x;d(x;y);m(y))}", lambda x,y,m,d,t: t(x,d(x,y),m(y)), 0),
    ],
    3: [
        ("{x+y+z}", lambda x,y,z: x+y+z, 0),
        ("{x+y+m(z)}", lambda x,y,z,m: x+y+m(z), 0),
        ("{m(x)+m(y)+m(z)}", lambda x,y,z,m: m(x)+m(y)+m(z), 0),
        ("{x+d(y;z)}", lambda x,y,z,d: x+d(y,z), 0),
        ("{m(x)+d(y;z)}", lambda x,y,z,m,d: m(x) + d(y,z), 0),
        ("{d(x;y)+m(z)}", lambda x,y,z,m,d: d(x,y) + m(z), 0),
        ("{d(x;y)+d(x;z)}", lambda x,y,z,d: d(x,y) + d(x,z), 0),
        ("{d(x;m(y))+d(m(x);z)}", lambda x,y,z,m,d: d(x,m(y)) + d(m(x),z), 0),
        # ("{d(m(x+y+z);)}", lambda x,y,z,m,d: lambda x1: d(m(x)+y+z,x1), 1)
    ]
}

def gen_fn(arity,symbols,fn_name=None,level=0,max_level=100):
    if level > max_level:
        raise RecursionError()
    arr = []
    choices = fn_map[arity]
    q = choices[random.randint(0,len(choices)-1)]
    fnd = q[0]
    fn_name = fn_name or gensym(symbols)
    if fn_name is None:
        return arr
    args = inspect.getfullargspec(q[1]).args
    fn_dict = {}
    if len(args) == 0:
        pass
    else:
        if 'm' in args:
            s = gensym(symbols)
            fnd = fns(fnd,'m',s)
            _,aa,ff = gen_fn(1,symbols,fn_name=s,level=level+1)
            fn_dict['m'] = ff
            arr.extend(aa)
        if 'd' in args:
            s = gensym(symbols)
            fnd = fns(fnd,'d',s)
            _,aa,ff = gen_fn(2,symbols,fn_name=s,level=level+1)
            fn_dict['d'] = ff
            arr.extend(aa)
        if 't' in args:
            s = gensym(symbols)
            fnd = fns(fnd,'t',s)
            _,aa,ff = gen_fn(3,symbols,fn_name=s,level=level+1)
            fn_dict['t'] = ff
            arr.extend(aa)
    fn = f"{fn_name}::{fnd}"
    arr.append(fn)
    return fn_name,arr,wrap_lambda(arity,q[1],fn_dict)


def gen_tests():
    random.seed(0)
    seen = set()
    for arity in range(4):
        for _ in range(10000):
            try:
                fn_name, arr, fn = gen_fn(arity,create_symbols(), max_level=100)
            except RecursionError as e:
                continue
            klong = KlongInterpreter()
            stmt = "; ".join(arr)
            if stmt in seen or len(arr) > 10: # filter out redundant and too complex (slow) calcs
                continue
            seen.add(stmt)
            klong(stmt)
            if arity == 0:
                fr = fn()
                kr = f"{fn_name}()"
            else:
                args = [1,2,3][:arity]
                fr = fn(*args)
                kr = f"{fn_name}({';'.join([str(x) for x in args])})"
            print(stmt)
            print(f't(\"{stmt}; {kr}\"; {kr}; {fr})')


if __name__ == '__main__':
    print("""
:"DO NOT MODIFY: GENERATED BY gen_test_fn.py"

err::0
wl::{.w(x);.p("")}
fail::{err::1;.d("failed: ");.p(x);.d("expected: ");wl(z);.d("got: ");wl(y);[]}
t::{:[~y~z;fail(x;y;z);[]]}

""")
    gen_tests()

    print("""
:[err;[];.p("ok!")]
""")
