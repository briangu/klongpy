import asyncio
import threading
import unittest
from unittest.mock import patch

from utils import LoopsBase

from klongpy import KlongInterpreter
from klongpy.core import KGCall, KGSym, KlongException
from klongpy.sys_fn import *
from klongpy.sys_fn_timer import eval_sys_fn_timer, eval_sys_fn_cancel_timer


class TestSysFnTimer(LoopsBase, unittest.TestCase):

    def test_timer_return_0(self):
        klong = KlongInterpreter()
        klong['.system'] = {'ioloop': self.ioloop, 'klongloop': self.klongloop}

        async def _test():
            klong("result::0")
            klong('cb::{result::3;0}')
            klong('th::.timer("test";0;cb)')

        async def _test_result():
            r = klong("result")
            while r != 3:
                await asyncio.sleep(0)
                r = klong("result")
                self.assertEqual(r,3)

        task = self.klongloop.call_soon_threadsafe(asyncio.create_task, _test())
        asyncio.run_coroutine_threadsafe(_test_result(), self.klongloop).result()

        task.cancel()


    def test_timer_self_terminate(self):
        klong = KlongInterpreter()
        klong['.system'] = {'ioloop': self.ioloop, 'klongloop': self.klongloop}

        async def _test():
            klong("result::0")
            klong('cb::{result::result+1;result<2}')
            klong('th::.timer("test";0;cb)')

        async def _test_result():
            r = klong("result")
            self.assertTrue(r >= 0 and r < 3)
            while r != 2:
                await asyncio.sleep(0)
                r = klong("result")
            r = klong(".timerc(th)")
            self.assertEqual(r,0)

        task = self.klongloop.call_soon_threadsafe(asyncio.create_task, _test())
        asyncio.run_coroutine_threadsafe(_test_result(), self.klongloop).result()
        task.cancel()

    def test_timer_return_1_cancel(self):
        klong = KlongInterpreter()
        klong['.system'] = {'ioloop': self.ioloop, 'klongloop': self.klongloop}

        async def _test():
            klong("result::0")
            klong('cb::{result::result+1;1}')
            klong('th::.timer("test";0;cb)')

        async def _test_result():
            r = klong("result")
            self.assertTrue(r >= 0 and r < 3)
            while r != 2:
                await asyncio.sleep(0)
                r = klong("result")
            r = klong(".timerc(th)")
            self.assertEqual(r,1)
            r = klong(".timerc(th)")
            self.assertEqual(r,0)

        task = self.klongloop.call_soon_threadsafe(asyncio.create_task, _test())
        asyncio.run_coroutine_threadsafe(_test_result(), self.klongloop).result()
        task.cancel()

    def test_timer_1_sec_self_terminate(self):
        klong = KlongInterpreter()
        klong['.system'] = {'ioloop': self.ioloop, 'klongloop': self.klongloop}

        start_t = self.klongloop.time()

        async def _test():
            klong("result::0")
            klong('cb::{result::result+1;result<2}')
            klong('th::.timer("test";1;cb)')

        async def _test_result():
            r = klong("result")
            self.assertTrue(r >= 0 and r < 3)
            while r != 2:
                await asyncio.sleep(0)
                r = klong("result")
            delta_t = (self.klongloop.time() - start_t)
            self.assertTrue(delta_t >= 2)
            r = klong(".timerc(th)")
            self.assertEqual(r,0)

        task = self.klongloop.call_soon_threadsafe(asyncio.create_task, _test())
        asyncio.run_coroutine_threadsafe(_test_result(), self.klongloop).result()
        task.cancel()

    def test_timer_1_sec_cancel(self):
        klong = KlongInterpreter()
        klong['.system'] = {'ioloop': self.ioloop, 'klongloop': self.klongloop}

        start_t = self.klongloop.time()

        async def _test():
            klong("result::0")
            klong('cb::{result::result+1;1}')
            klong('th::.timer("test";1;cb)')

        async def _test_result():
            r = klong("result")
            self.assertTrue(r >= 0 and r < 3)
            while r != 2:
                await asyncio.sleep(0)
                r = klong("result")
            delta_t = (self.klongloop.time() - start_t)
            self.assertTrue(delta_t >= 2)
            r = klong(".timerc(th)")
            self.assertEqual(r,1)
            r = klong(".timerc(th)")
            self.assertEqual(r,0)

        task = self.klongloop.call_soon_threadsafe(asyncio.create_task, _test())
        asyncio.run_coroutine_threadsafe(_test_result(), self.klongloop).result()
        task.cancel()

    def test_timer_resolves_latest_callback(self):
        klong = KlongInterpreter()
        klong['.system'] = {'ioloop': self.ioloop, 'klongloop': self.klongloop}

        async def _test():
            klong("result::0")
            klong('cb::{result::1;1}')
            klong('th::.timer("test";0;cb)')

        async def _test_result():
            for _ in range(1000):
                if klong("result") == 1:
                    break
                await asyncio.sleep(0)
            self.assertEqual(klong("result"), 1)
            klong('cb::{result::2;0}')
            for _ in range(1000):
                if klong("result") == 2:
                    break
                await asyncio.sleep(0)
            self.assertEqual(klong("result"), 2)
            klong(".timerc(th)")

        task = self.klongloop.call_soon_threadsafe(asyncio.create_task, _test())
        asyncio.run_coroutine_threadsafe(_test_result(), self.klongloop).result()
        task.cancel()

    def test_timer_rejects_function_call(self):
        klong = KlongInterpreter()
        with self.assertRaisesRegex(KlongException, "not a function call"):
            eval_sys_fn_timer(klong, "test", 0, KGCall(KGSym("cb"), [], 1))

    def test_timer_rejects_non_callable(self):
        klong = KlongInterpreter()
        with self.assertRaisesRegex(KlongException, "z must be a function"):
            eval_sys_fn_timer(klong, "test", 0, 42)

    def test_timer_accepts_python_callable(self):
        klong = KlongInterpreter()
        klong['.system'] = {'klongloop': object()}
        callback = lambda: 1

        with patch('klongpy.sys_fn_timer._call_periodic', return_value="ok") as mock_call:
            result = eval_sys_fn_timer(klong, "test", 0, callback)

        self.assertEqual(result, "ok")
        self.assertIs(mock_call.call_args.args[3], callback)


class TestTimerArguments(LoopsBase, unittest.TestCase):

    def _klong(self):
        klong = KlongInterpreter()
        klong['.system'] = {'ioloop': self.ioloop, 'klongloop': self.klongloop}
        return klong

    def _call(self, klong, y, z, wait=0.0):
        """Invoke .timer on the klong loop thread; return the result (or raise)."""
        import concurrent.futures
        fut = concurrent.futures.Future()

        def go():
            try:
                fut.set_result(eval_sys_fn_timer(klong, "t", y, z))
            except BaseException as e:
                fut.set_exception(e)
        self.klongloop.call_soon_threadsafe(go)
        r = fut.result(timeout=5)
        if wait:
            import time
            time.sleep(wait)
        return r

    def test_timer_fractional_interval(self):
        klong = self._klong()
        count = [0]

        def cb():
            count[0] += 1
            return 1
        h = self._call(klong, 0.05, cb, wait=0.5)
        self.klongloop.call_soon_threadsafe(h.cancel)
        self.assertGreaterEqual(count[0], 5)
        self.assertLessEqual(count[0], 15)

    def test_timer_negative_interval_raises(self):
        from klongpy.core import KlongException
        with self.assertRaises(KlongException):
            self._call(self._klong(), -1, lambda: 1)

    def test_timer_non_numeric_interval_raises(self):
        from klongpy.core import KlongException
        with self.assertRaises(KlongException):
            self._call(self._klong(), "a", lambda: 1)

    def test_timer_non_function_raises(self):
        from klongpy.core import KlongException
        with self.assertRaises(KlongException):
            self._call(self._klong(), 1, 5)

    def test_timer_callback_exception_cancels(self):
        klong = self._klong()
        count = [0]

        def bad():
            count[0] += 1
            raise RuntimeError("boom")
        with self.assertLogs(level="ERROR"):
            h = self._call(klong, 0.05, bad, wait=0.3)
        self.assertEqual(count[0], 1)
        self.assertEqual(eval_sys_fn_cancel_timer(h), 0)
