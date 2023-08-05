import numpy as np
import os

from klongpy import KlongInterpreter
from klongpy.core import kg_equal, is_list

def die(m=None):
    raise RuntimeError(m)


def eval_cmp(expr_str, expected_str, klong=None):
    """
    Parse and execute both sides of a test.
    """
    klong = klong or KlongInterpreter()
    expr = klong.prog(expr_str)[1][0]
    expected = klong.prog(expected_str)[1][0]
    a = klong.call(expr)
    b = klong.call(expected)
    return kg_equal(a,b)


def eval_test(a, klong=None):
    """
    To get the system going we need a way to test the t(x;y;z) methods before we can parse them.
    This test bootstraps the testing process via parsing the t() format.
    """
    klong = klong or create_test_klong()
    s = a[2:-1]
    i,p = klong.prog(s)
    if i != len(s):
        return False
    a = klong.call(p[1])
    b = klong.call(p[2])
    if np.isarray(a) and np.isarray(b):
        c = a == b
        return not c[np.where(c == False)].any() if np.isarray(c) else c
    else:
        return kg_equal(a,b)


def create_test_klong():
    """
    Create a Klong instance that is similar to the Klong test suite,
    but modified to fail with die()
    """
    klong = KlongInterpreter()
    klong('err::0;')
    def fail(x,y,z):
        print(x,y,z)
        die()
    klong['fail'] = fail
    klong('t::{:[~y~z;fail(x;y;z);[]]}')
    klong('rnd::{:[x<0;-1;1]*_0.5+#x}')
    klong('rndn::{rnd(x*10^y)%10^y}')
    return klong


def run_file(x, klong=None):
    with open(x, "r") as f:
        klong = klong or KlongInterpreter()
        return klong, klong(f.read())


def rec_fn2(a,b,f):
    return np.asarray([rec_fn2(x, y, f) for x,y in zip(a,b)], dtype=object) if is_list(a) and is_list(b) else f(a,b)


