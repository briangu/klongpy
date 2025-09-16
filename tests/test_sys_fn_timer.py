import asyncio
import threading
import unittest

from utils import kg_equal, LoopsBase

from klongpy import KlongInterpreter
from klongpy.sys_fn import *


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

    def test_timer_fractional_interval(self):
        klong = KlongInterpreter()
        klong['.system'] = {'ioloop': self.ioloop, 'klongloop': self.klongloop}

        start_t = self.klongloop.time()

        async def _test():
            klong("result::0")
            klong('cb::{result::result+1;result<2}')
            klong('th::.timer("test";0.1;cb)')

        async def _test_result():
            r = klong("result")
            self.assertTrue(r >= 0 and r < 3)
            while r != 2:
                await asyncio.sleep(0)
                r = klong("result")
            delta_t = (self.klongloop.time() - start_t)
            self.assertTrue(delta_t >= 0.2)
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

    def test_timer_dynamic_lookup(self):
        klong = KlongInterpreter()
        klong['.system'] = {'ioloop': self.ioloop, 'klongloop': self.klongloop}

        async def _test():
            klong('result::""')
            klong('cb::{result::"h1";1}')
            klong('th::.timer("test";0;cb)')
            while klong('result') != 'h1':
                await asyncio.sleep(0)
            klong('cb::{result::"h2";0}')

        async def _test_result():
            r = klong('result')
            while r != 'h2':
                await asyncio.sleep(0)
                r = klong('result')
            r = klong('.timerc(th)')
            self.assertEqual(r,0)

        task = self.klongloop.call_soon_threadsafe(asyncio.create_task, _test())
        asyncio.run_coroutine_threadsafe(_test_result(), self.klongloop).result()
        task.cancel()

    def test_timer_redefinition_non_function_uses_previous_callable(self):
        klong = KlongInterpreter()
        klong['.system'] = {'ioloop': self.ioloop, 'klongloop': self.klongloop}

        async def _test():
            klong('result::0')
            klong('cb::{result::result+1;1}')
            klong('th::.timer("test";0;cb)')
            while klong('result') < 3:
                await asyncio.sleep(0)
            klong('cb::0')
            await asyncio.sleep(0)
            klong('cb::{result::42;0}')

        async def _test_result():
            async def wait_for_result():
                while klong('result') != 42:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_for_result(), timeout=1)
            await asyncio.sleep(0)
            self.assertEqual(klong('result'), 42)
            self.assertEqual(klong('.timerc(th)'), 0)

        task = self.klongloop.call_soon_threadsafe(asyncio.create_task, _test())
        asyncio.run_coroutine_threadsafe(_test_result(), self.klongloop).result()
        task.cancel()
