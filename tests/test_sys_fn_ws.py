import unittest
from unittest.mock import AsyncMock

from utils import LoopsBase

from klongpy import KlongInterpreter
from klongpy.ws.sys_fn_ws import NetworkClient, KlongWSConnectionFailureException


class DummyConnectionProvider:
    async def connect(self):
        return object()

    async def close(self):
        return None

    def is_open(self):
        return True

    def __str__(self):
        return "dummy"


class FailingNetworkClient(NetworkClient):
    async def _listen(self, on_message):
        raise KlongWSConnectionFailureException()


class TestNetworkClient(LoopsBase, unittest.TestCase):
    def test_run_client_uses_self_handlers(self):
        klong = KlongInterpreter()
        provider = DummyConnectionProvider()
        on_connect = AsyncMock()
        on_error = AsyncMock()

        client = FailingNetworkClient(
            self.ioloop,
            self.klongloop,
            klong,
            provider,
            on_connect=on_connect,
            on_error=on_error,
        )

        client.run_client()
        client._run_exit_event.wait(timeout=1)

        self.assertTrue(client._run_exit_event.is_set())
        on_connect.assert_awaited_once_with(client)
        on_error.assert_awaited_once()
        self.assertIs(on_error.await_args.args[0], client)
        self.assertIsInstance(on_error.await_args.args[1], KlongWSConnectionFailureException)

        client.close()
