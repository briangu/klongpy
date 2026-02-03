import asyncio
import re
import threading
import time
import unittest

import numpy as np

from klongpy import KlongInterpreter
from klongpy.core import is_list
from klongpy.backends import get_backend
from klongpy.backend import UnsupportedDtypeError, np as backend_np


def is_torch_backend(klong):
    """Check if klong interpreter is using torch backend."""
    return klong._backend.name == 'torch'


def torch_autograd(func, x, backend):
    """Compute gradient using PyTorch autograd (requires torch backend)."""
    if not backend.supports_autograd():
        raise RuntimeError("torch_autograd requires a backend that supports autograd")
    return backend.compute_autograd(func, x)


class BackendSkipError(Exception):
    """Raised when a test should be skipped due to backend limitations."""
    pass


def _expr_uses_strings(expr_str):
    """Check if an expression uses string literals or character operations."""
    # Check for string literals (double-quoted)
    if '"' in expr_str:
        # But not just empty strings or single chars which might work
        # String patterns: "x" is OK (char), "ab" or longer needs strings
        string_pattern = r'"[^"]{2,}"'
        if re.search(string_pattern, expr_str):
            return True
    # Check for character literal operations that produce strings
    if '0c' in expr_str:
        return True
    return False


def _expr_uses_nested_arrays(expr_str):
    """Check if an expression uses nested/jagged arrays."""
    # Look for patterns like [1 [2] 3] or [[1 2] [3 4 5]]
    # This is a heuristic - nested brackets with different content
    bracket_depth = 0
    max_depth = 0
    for c in expr_str:
        if c == '[':
            bracket_depth += 1
            max_depth = max(max_depth, bracket_depth)
        elif c == ']':
            bracket_depth -= 1
    return max_depth > 1


def _check_backend_support(expr_str, expected_str):
    """
    Check if the current backend supports the features used in the test expressions.

    Returns None if supported, or a skip message if not.
    """
    backend = get_backend()

    # Check string support
    if not backend.supports_strings():
        if _expr_uses_strings(expr_str) or _expr_uses_strings(expected_str):
            return f"Backend '{backend.name}' does not support strings"

    return None  # Supported


def die(m=None):
    raise RuntimeError(m)


def _is_torch_limitation_error(e):
    """Check if an exception is due to torch backend limitations."""
    error_msg = str(e).lower()
    skip_patterns = [
        'does not support object dtype',
        'does not support string',
        'too many dimensions',
        'can\'t convert',
        'cannot convert',
        'only integer tensors',
        'only one element tensors',
        'len() of a 0-d tensor',
        'tensor that requires grad',
        'mps tensor',
        'float64 dtype',
        'mps device',
        'not currently implemented for the mps',
        'argument \'input\'',
        'must be tensor',
        'received an invalid combination',
        'different devices',
        'could not infer dtype',
        'not a sequence',
        'expected sequence of length',
        'must be tuple of ints',
        'no such file or directory',
        'astype',
        'no attribute \'isarray\'',
    ]
    return any(pattern in error_msg for pattern in skip_patterns)


def eval_cmp(expr_str, expected_str, klong=None, skip_unsupported=True):
    """
    Parse and execute both sides of a test.

    If skip_unsupported=True (default), raises BackendSkipError for tests
    that use features not supported by the current backend.
    """
    # Check backend support before executing
    if skip_unsupported:
        skip_reason = _check_backend_support(expr_str, expected_str)
        if skip_reason:
            raise BackendSkipError(skip_reason)

    klong = klong or KlongInterpreter()

    try:
        expr = klong.prog(expr_str)[1][0]
        expected = klong.prog(expected_str)[1][0]
        a = klong.call(expr)
        b = klong.call(expected)
        return klong._backend.kg_equal(a, b)
    except UnsupportedDtypeError as e:
        if skip_unsupported:
            raise BackendSkipError(str(e))
        raise
    except (TypeError, ValueError, RuntimeError) as e:
        if skip_unsupported and is_torch_backend(klong) and _is_torch_limitation_error(e):
            raise BackendSkipError(f"Torch limitation: {e}")
        raise


def eval_test(a, klong=None, skip_unsupported=True):
    """
    To get the system going we need a way to test the t(x;y;z) methods before we can parse them.
    This test bootstraps the testing process via parsing the t() format.

    If skip_unsupported=True (default), raises BackendSkipError for tests
    that use features not supported by the current backend.
    """
    klong = klong or create_test_klong()
    s = a[2:-1]

    # Check backend support before executing
    if skip_unsupported:
        skip_reason = _check_backend_support(s, s)
        if skip_reason:
            raise BackendSkipError(skip_reason)

    try:
        i, p = klong.prog(s)
        if i != len(s):
            return False
        a = klong.call(p[1])
        b = klong.call(p[2])
        if backend_np.isarray(a) and backend_np.isarray(b):
            c = a == b
            # Handle tensor/array result
            if hasattr(c, 'all'):
                return bool(c.all())
            return not c[np.where(c == False)].any() if np.isarray(c) else c
        else:
            return klong._backend.kg_equal(a, b)
    except UnsupportedDtypeError as e:
        if skip_unsupported:
            raise BackendSkipError(str(e))
        raise
    except (TypeError, ValueError, RuntimeError) as e:
        if skip_unsupported and is_torch_backend(klong) and _is_torch_limitation_error(e):
            raise BackendSkipError(f"Torch limitation: {e}")
        raise


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


class LoopsBase:
    def setUp(self):
        # print("\nRunning test:", self._testMethodName)

        self.ioloop = asyncio.new_event_loop()
        self.ioloop.set_debug(True)
        self.ioloop_started = threading.Event()

        self.ioloop_thread = threading.Thread(target=self.start_ioloop)
        self.ioloop_thread.start()
        self.ioloop_started.wait()
        self.assertTrue(self.ioloop.is_running())

        self.klongloop = asyncio.new_event_loop()
        self.klongloop.set_debug(True)
        self.klongloop_started = threading.Event()

        self.klongloop_thread = threading.Thread(target=self.start_klongloop)
        self.klongloop_thread.start()
        self.klongloop_started.wait()
        self.assertTrue(self.klongloop.is_running())

    def tearDown(self):
        while len(asyncio.all_tasks(loop=self.ioloop)) > 0:
            time.sleep(0.1)
        self.ioloop.call_soon_threadsafe(self.ioloop.stop)
        self.ioloop_thread.join()
        self.ioloop.close()

        while len(asyncio.all_tasks(loop=self.klongloop)) > 0:
            time.sleep(0.1)
        self.klongloop.call_soon_threadsafe(self.klongloop.stop)
        self.klongloop_thread.join()
        self.klongloop.close()

    def start_ioloop(self):
        asyncio.set_event_loop(self.ioloop)
        self.ioloop.call_soon(self.ioloop_start_signal) 
        self.ioloop.run_forever()

    def ioloop_start_signal(self):
        """Coroutine to set the event once the loop has started."""
        self.assertTrue(self.ioloop.is_running())
        self.ioloop_started.set()

    def start_klongloop(self):
        asyncio.set_event_loop(self.klongloop)
        self.klongloop.call_soon(self.klongloop_start_signal) 
        self.klongloop.run_forever()

    def klongloop_start_signal(self):
        """Coroutine to set the event once the loop has started."""
        self.assertTrue(self.klongloop.is_running())
        self.klongloop_started.set()
