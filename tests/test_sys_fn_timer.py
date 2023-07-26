import asyncio
import unittest

from utils import kg_equal

from klongpy import KlongInterpreter
from klongpy.sys_fn import *


class TestSysFnTimer(unittest.TestCase):

    # add test for N counts
    #   return 0
    #   rely on cancel
    # add test for interval 0 and 1

    def test_timer_return_0(self):
        klong = KlongInterpreter()

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

        loop = asyncio.get_event_loop()
        loop.run_until_complete(_test())
        loop.run_until_complete(_test_result())


    def test_timer_self_terminate(self):
        klong = KlongInterpreter()

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

        loop = asyncio.get_event_loop()
        loop.run_until_complete(_test())
        loop.run_until_complete(_test_result())


    def test_timer_return_1_cancel(self):
        klong = KlongInterpreter()

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

        loop = asyncio.get_event_loop()
        loop.run_until_complete(_test())
        loop.run_until_complete(_test_result())

    def test_timer_1_sec_self_terminate(self):
        klong = KlongInterpreter()

        loop = asyncio.get_event_loop()
        start_t = loop.time()

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
            delta_t = (loop.time() - start_t)
            self.assertTrue(delta_t >= 2)
            r = klong(".timerc(th)")
            self.assertEqual(r,0)

        loop.run_until_complete(_test())
        loop.run_until_complete(_test_result())

    def test_timer_1_sec_cancel(self):
        klong = KlongInterpreter()

        loop = asyncio.get_event_loop()
        start_t = loop.time()

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
            delta_t = (loop.time() - start_t)
            self.assertTrue(delta_t >= 2)
            r = klong(".timerc(th)")
            self.assertEqual(r,1)
            r = klong(".timerc(th)")
            self.assertEqual(r,0)

        loop.run_until_complete(_test())
        loop.run_until_complete(_test_result())
