import numpy as np
import os

from klongpy import KlongInterpreter
from klongpy.core import is_list, is_number

def die(m=None):
    raise RuntimeError(m)


def array_equal(a,b):
    """
    Recursively determine if two values or arrays are equal.

    NumPy ops (e.g. array_equal) are not sufficiently general purpose for this, so we need our own.
    """
    if is_list(a):
        if is_list(b):
            if len(a) != len(b):
                return False
        else:
            return False
    else:
        if is_list(b):
            return False
        else:
            return np.isclose(a,b) if is_number(a) and is_number(b) else a == b

    r = np.asarray([array_equal(x,y) for x,y in zip(a,b)])
    return not r[np.where(r == False)].any()


def eval_cmp(expr_str, expected_str, klong=None):
    """
    Parse and execute both sides of a test.
    """
    klong = klong or KlongInterpreter()
    expr = klong.prog(expr_str)[1][0]
    expected = klong.prog(expected_str)[1][0]
    a = klong.call(expr)
    b = klong.call(expected)
    return array_equal(a,b)


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
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        c = a == b
        return not c[np.where(c == False)].any() if isinstance(c,np.ndarray) else c
    else:
        return array_equal(a,b)


def create_test_klong():
    """
    Create a Klong instance that is similar to the Klong test suite,
    but modified to fail with die()
    """
    klong = KlongInterpreter()
    klong.exec('err::0;')
    def fail(x,y,z):
        print(x,y,z)
        die()
    klong['fail'] = fail
    klong.exec('t::{:[~y~z;fail(x;y;z);[]]}')
    klong.exec('rnd::{:[x<0;-1;1]*_0.5+#x}')
    klong.exec('rndn::{rnd(x*10^y)%10^y}')
    return klong


def load_lib_file(x):
    cwd = os.getcwd()
    os.chdir(os.path.join(cwd, 'examples'))
    try:
        try:
            with open(x) as f:
                klong = KlongInterpreter()
                return klong, klong(f.read())
        except Exception as e:
            import traceback
            traceback.print_stack(e)
            print(f"Failed {x}")
            raise e
    finally:
        os.chdir(cwd)
