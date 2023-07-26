import asyncio
import unittest

from utils import kg_equal

from klongpy import KlongInterpreter
from klongpy.sys_fn import *


class TestSysFnIPC(unittest.TestCase):

    def test_async_fn(self):
        klong = KlongInterpreter()

        async def _test():
            klong("fn::{x+1}")
            klong("result::0")
            klong("cb::{result::x}")
            klong("afn::.async(fn;cb)")
            r = klong("afn(2)")
            self.assertEqual(r,1)

        async def _test_result():
            r = klong("result")
            self.assertEqual(r,3)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(_test())
        loop.run_until_complete(_test_result())


    def test_async_python_lambda_fn(self):
        klong = KlongInterpreter()

        async def _test():
            klong["fn"] = lambda x: x+1
            klong("result::0")
            klong("cb::{result::x}")
            klong("afn::.async(fn;cb)")
            r = klong("afn(2)")
            self.assertEqual(r,1)

        async def _test_result():
            r = klong("result")
            self.assertEqual(r,3)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(_test())
        loop.run_until_complete(_test_result())

