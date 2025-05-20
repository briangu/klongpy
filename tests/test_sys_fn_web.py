import asyncio
import socket
import unittest

import aiohttp

from klongpy.repl import create_repl, cleanup_repl


class TestSysFnWeb(unittest.TestCase):
    def setUp(self):
        self.klong, self.loops = create_repl()
        (self.ioloop, self.ioloop_thread, self.io_stop,
         self.klongloop, self.klongloop_thread, self.klong_stop) = self.loops
        self.handle = None

    def tearDown(self):
        if self.handle is not None and self.handle.task is not None:
            asyncio.run_coroutine_threadsafe(self.handle.shutdown(), self.ioloop).result()
        cleanup_repl(self.loops)

    def _free_port(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def test_web_server_start_and_stop(self):
        klong = self.klong
        port = self._free_port()

        klong('.py("klongpy.web")')
        klong('index::{x;"hello"}')
        klong('get:::{}')
        klong('get,"/",index')
        klong('post:::{}')
        handle = klong(f'h::.web({port};get;post)')
        self.handle = handle

        async def fetch():
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{port}/") as resp:
                    return await resp.text()

        response = asyncio.run_coroutine_threadsafe(fetch(), self.ioloop).result()
        self.assertEqual(response, "hello")

        asyncio.run_coroutine_threadsafe(handle.shutdown(), self.ioloop).result()

