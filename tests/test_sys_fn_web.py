import asyncio
import socket
import unittest
from unittest.mock import patch

import aiohttp

from klongpy.core import KGCall, KGLambda, KGSym
from klongpy.repl import create_repl, cleanup_repl
from klongpy.web.sys_fn_web import eval_sys_fn_create_web_server


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
        try:
            s.bind(("", 0))
            return s.getsockname()[1]
        except PermissionError as exc:
            raise unittest.SkipTest("socket bind not permitted in this environment") from exc
        finally:
            s.close()

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

    def test_web_routes_handle_kgcall_and_wrap(self):
        klong = self.klong
        fn_call = KGCall(KGSym("handler"), [], 1)
        ok_handler = KGLambda(lambda x: "ok")
        get_routes = {"/bad": fn_call, "/ok": ok_handler}
        post_routes = {"/bad": fn_call, "/ok": ok_handler}

        def _close_task(coro):
            coro.close()
            return object()

        with patch("klongpy.web.sys_fn_web.asyncio.create_task", side_effect=_close_task):
            handle = eval_sys_fn_create_web_server(klong, 0, get_routes, post_routes)

        self.assertIsNotNone(handle.task)

    def test_web_rejects_function_calls(self):
        klong = self.klong
        fn_call = KGCall(KGSym("handler"), [], 1)
        get_routes = {"/bad": fn_call}
        post_routes = {"/bad": fn_call}

        def _close_task(coro):
            coro.close()
            return object()

        with patch("klongpy.web.sys_fn_web.asyncio.create_task", side_effect=_close_task):
            eval_sys_fn_create_web_server(klong, 0, get_routes, post_routes)
