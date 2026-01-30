import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

from klongpy import KlongInterpreter
from klongpy.core import KGCall, KGLambda, KGSym
from klongpy.web.sys_fn_web import (
    WebServerHandle,
    create_system_functions_web,
    eval_sys_fn_shutdown_web_server,
)


class TestWebServerHandle(unittest.TestCase):
    def test_str_with_bind(self):
        handle = WebServerHandle("127.0.0.1", 8080, None, None)
        result = str(handle)
        self.assertEqual(result, "web[127.0.0.1:8080]")

    def test_str_without_bind(self):
        handle = WebServerHandle(None, 8080, None, None)
        result = str(handle)
        self.assertEqual(result, "web[0.0.0.0:8080]")

    def test_shutdown(self):
        mock_runner = AsyncMock()
        mock_task = MagicMock()  # cancel() is sync, not async
        handle = WebServerHandle("localhost", 8080, mock_runner, mock_task)
        asyncio.run(handle.shutdown())
        mock_task.cancel.assert_called_once()
        mock_runner.cleanup.assert_awaited_once()
        self.assertIsNone(handle.runner)
        self.assertIsNone(handle.task)


class TestShutdownWebServer(unittest.TestCase):
    def test_shutdown_non_kgcall(self):
        klong = KlongInterpreter()
        result = eval_sys_fn_shutdown_web_server(klong, "not_a_kgcall")
        self.assertEqual(result, 0)

    def test_shutdown_kgcall_non_lambda(self):
        klong = KlongInterpreter()
        call = KGCall(KGSym("test"), [], 1)
        call.a = "not_a_lambda"
        result = eval_sys_fn_shutdown_web_server(klong, call)
        self.assertEqual(result, 0)

    def test_shutdown_kgcall_lambda_non_webhandle(self):
        klong = KlongInterpreter()
        klong['.system'] = {'ioloop': asyncio.new_event_loop()}
        inner_lambda = KGLambda(lambda x: x)
        inner_lambda.fn = "not_a_webhandle"
        call = KGCall(inner_lambda, [], 1)
        call.a = inner_lambda
        result = eval_sys_fn_shutdown_web_server(klong, call)
        self.assertEqual(result, 0)


class TestCreateSystemFunctions(unittest.TestCase):
    def test_create_system_functions_web(self):
        registry = create_system_functions_web()
        self.assertIsInstance(registry, dict)
        self.assertIn(".web", registry)
        self.assertIn(".webc", registry)


if __name__ == '__main__':
    unittest.main()
