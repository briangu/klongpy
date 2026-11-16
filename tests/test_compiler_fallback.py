"""
Ticket 017: the compiled fast path must not swallow run-time errors or re-run
an expression on the interpreter, must compile a function body once, and must
keep its caches bounded.

These tests use the numpy backend (it has a compiler), so they run without torch.
"""
import contextlib
import io
import signal
import unittest
import warnings

from klongpy import KlongInterpreter
from klongpy.compiler import CACHE_LIMIT, LRUCache, compile_guarded


def _compiled_entry(klong, source):
    return klong._compiled_cache[(source, klong._module)]


class CountingBackendSpy:
    """Counts backend.compile_expr_ir calls on one interpreter."""

    def __init__(self, klong):
        self.count = 0
        backend = klong._backend
        original = backend.compile_expr_ir

        def spy(ir, var_syms):
            self.count += 1
            return original(ir, var_syms)

        self._restore = (backend, original)
        backend.compile_expr_ir = spy

    def close(self):
        backend, original = self._restore
        backend.compile_expr_ir = original


class TestCompileOnce(unittest.TestCase):
    def test_compile_once_per_function(self):
        k = KlongInterpreter()
        spy = CountingBackendSpy(k)
        self.addCleanup(spy.close)
        k("a::5")
        k("g::{a+a}")
        r = k("g'!1000")
        self.assertEqual(list(r), [10] * 1000)
        self.assertEqual(spy.count, 1)

    def test_compile_once_reduce_body(self):
        k = KlongInterpreter()
        spy = CountingBackendSpy(k)
        self.addCleanup(spy.close)
        k("v::!100")
        k("w::{+/v}")
        r = k("w'!200")
        self.assertEqual(list(r), [4950] * 200)
        self.assertEqual(spy.count, 1)

    def test_uncompilable_body_not_retried(self):
        k = KlongInterpreter()
        calls = []
        import klongpy.interpreter as interp
        original = interp.compile_guarded
        interp.compile_guarded = lambda *a: calls.append(1) or original(*a)
        self.addCleanup(setattr, interp, 'compile_guarded', original)
        k("h::{x+1}")
        k("h'!100")
        # one attempt per distinct node (the body, the call h'!100 itself, the
        # statement h::...), not one per call of h
        self.assertLess(len(calls), 10)


class TestErrorsSurface(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_compile_failure_not_run_twice(self):
        k = KlongInterpreter()
        k("a::[1 2 3]")
        k("b::[1 2]")
        # compile and cache, then count calls of the compiled function
        with self.assertRaises(ValueError):
            k("a+b")
        entry = _compiled_entry(k, "a+b")
        self.assertTrue(entry)
        calls = []
        real = entry.fn
        entry.fn = lambda *a: calls.append(1) or real(*a)

        interp_calls = []
        real_add = k._vd['+']
        k._vd['+'] = lambda x, y: interp_calls.append(1) or real_add(x, y)

        with self.assertRaises(ValueError):
            k("a+b")
        self.assertEqual(len(calls), 1)
        self.assertEqual(interp_calls, [])

    def test_runtime_error_from_compiled_fn_propagates(self):
        class Boom(Exception):
            pass

        k = KlongInterpreter()
        k("a::[1 2 3]")
        k("a+1")
        entry = _compiled_entry(k, "a+1")

        def boom(*args):
            raise Boom()

        entry.fn = boom
        with self.assertRaises(Boom):
            k("a+1")

    def test_side_effects_in_function_run_once_when_it_raises(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            k = KlongInterpreter()  # binds the Klong stdout at creation
            k("a::[1 2 3]")
            k("b::[1 2]")
            k("n::0")
            k('f::{n::n+1;.p("effect");a+b}')
            with self.assertRaises(ValueError):
                k("f(0)")
            with self.assertRaises(ValueError):
                k("f(0)")
        self.assertEqual(k("n"), 2)
        self.assertEqual(out.getvalue().count("effect"), 2)

    def test_side_effect_written_via_dot_d_runs_once(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            k = KlongInterpreter()  # binds the Klong stdout at creation
            k("a::[1 2 3]")
            k("b::[1 2]")
            k('f::{.d("x");a+b}')
            with self.assertRaises(ValueError):
                k("f(0)")
        self.assertEqual(out.getvalue(), "x")

    def test_signal_exception_not_swallowed(self):
        k = KlongInterpreter()
        k("a::[1 2 3]")
        k("a+1")
        entry = _compiled_entry(k, "a+1")
        real = entry.fn

        def handler(signum, frame):
            raise TimeoutError("alarm")

        def slow(*args):
            signal.raise_signal(signal.SIGALRM)
            return real(*args)

        entry.fn = slow
        old = signal.signal(signal.SIGALRM, handler)
        self.addCleanup(signal.signal, signal.SIGALRM, old)
        with self.assertRaises(TimeoutError):
            k("a+1")

    def test_division_by_zero_still_matches_interpreter(self):
        # Python raises ZeroDivisionError for scalar a%0; the interpreter
        # answers :undefined. This is the one run-time fallback.
        k = KlongInterpreter()
        k("a::4")
        k("b::0")
        fresh = KlongInterpreter()
        fresh("a::4")
        fresh("b::0")
        expected = fresh("a%b")
        self.assertEqual(str(k("a%b")), str(expected))
        self.assertEqual(str(k("a%b")), str(expected))  # cached path too

    def test_unsupported_ast_falls_back_at_compile_time(self):
        k = KlongInterpreter()
        k("a::[1 2 3]")

        def broken(ir, var_syms):
            raise TypeError("cannot compile")

        k._backend.compile_expr_ir = broken
        self.addCleanup(lambda: k._backend.__dict__.pop('compile_expr_ir', None))
        self.assertEqual(list(k("a+1")), [2, 3, 4])


class TestRebinding(unittest.TestCase):
    """Rebinding a name must give what a fresh interpreter gives."""

    CASES = [
        ("a::2", ['a::"ab"', 'a::0cx', 'a::[1 2 3]', 'a::[[1 2][3]]', 'a::1.5',
                  'a::{x+1}', 'a::[]']),
        ("a::[1 2 3]", ['a::"ab"', 'a::5', 'a::2.5', 'a::[1.5 2]', 'a::[[1 2][3 4]]']),
        ("a::1.5", ['a::2', 'a::"ab"', 'a::[1 2]']),
    ]
    EXPRS = ["a*3", "a+1", "a<3", "-a", "a%0", "a^2", "+/a", "|/a"]

    @staticmethod
    def _result_of(k, src):
        try:
            return ('ok', str(k(src)))
        except Exception as e:  # compare error class with the interpreter's
            return ('err', type(e).__name__)

    def test_type_change_matches_interpreter(self):
        warnings.simplefilter("ignore")
        for first, rebinds in self.CASES:
            for rebind in rebinds:
                for expr in self.EXPRS:
                    with self.subTest(first=first, rebind=rebind, expr=expr):
                        k = KlongInterpreter()
                        k(first)
                        self._result_of(k, expr)  # compile and cache
                        k(rebind)
                        got = self._result_of(k, expr)
                        fresh = KlongInterpreter()
                        fresh(rebind)
                        self.assertEqual(got, self._result_of(fresh, expr))

    def test_type_change_inside_function_body(self):
        warnings.simplefilter("ignore")
        for rebind in ['a::"ab"', 'a::[1 2]', 'a::2.5', 'a::{x}']:
            with self.subTest(rebind=rebind):
                k = KlongInterpreter()
                k("a::3")
                k("f::{a+x}")
                self._result_of(k, "f(1)")
                self._result_of(k, "f(1)")
                k(rebind)
                got = self._result_of(k, "f(1)")
                fresh = KlongInterpreter()
                fresh("f::{a+x}")
                fresh(rebind)
                self.assertEqual(got, self._result_of(fresh, "f(1)"))

    def test_klong_rebind_does_not_use_stale_compiled_function(self):
        # Rebinding with ::  clears the top-level cache but not the compiled
        # function kept on a function body. The operand guard is what keeps
        # that function from running on a value of another kind.
        k = KlongInterpreter()
        calls = []
        backend = k._backend
        original = backend.compile_expr_ir

        def counting(ir, var_syms):
            fn, syms = original(ir, var_syms)
            return (lambda *a: calls.append(1) or fn(*a)), syms

        backend.compile_expr_ir = counting
        self.addCleanup(backend.__dict__.pop, 'compile_expr_ir', None)
        k("a::2")
        k("f::{a*3}")
        self.assertEqual(k("f(0)"), 6)
        self.assertEqual(len(calls), 1)
        k('a::"ab"')
        with self.assertRaises(Exception):
            k("f(0)")
        self.assertEqual(len(calls), 1)  # compiled function not used
        k("a::5")
        self.assertEqual(k("f(0)"), 15)
        self.assertEqual(len(calls), 2)


class TestCachesBounded(unittest.TestCase):
    def test_cache_bounded(self):
        k = KlongInterpreter()
        k("a::1")
        for i in range(CACHE_LIMIT * 3):
            self.assertEqual(k(f"a+{i}"), 1 + i)
        self.assertLessEqual(len(k._parse_cache), CACHE_LIMIT)
        self.assertLessEqual(len(k._compiled_cache), CACHE_LIMIT)
        self.assertLess(CACHE_LIMIT, 10000)

    def test_lru_keeps_recent_entries(self):
        c = LRUCache(3)
        c['a'] = 1
        c['b'] = 2
        c['c'] = 3
        self.assertEqual(c.get('a'), 1)  # a is now most recent
        c['d'] = 4  # evicts b
        self.assertIsNone(c.get('b'))
        self.assertEqual(sorted(c), ['a', 'c', 'd'])
        c.clear()
        self.assertEqual(len(c), 0)


if __name__ == '__main__':
    unittest.main()
