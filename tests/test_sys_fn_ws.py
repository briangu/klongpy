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

    def test_run_client_callback_uses_parameter_not_closure(self):
        """
        Test that _on_connect and _on_error use the 'client' parameter,
        not a closure over 'self'. This verifies the fix for using
        client.on_connect instead of self.on_connect.
        """
        klong = KlongInterpreter()
        provider = DummyConnectionProvider()

        # Track which client instance the callback received
        received_clients = []

        async def on_connect(client):
            received_clients.append(('connect', client))

        async def on_error(client, e):
            received_clients.append(('error', client))

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

        # Verify callbacks received the correct client instance
        self.assertEqual(len(received_clients), 2)
        self.assertEqual(received_clients[0][0], 'connect')
        self.assertIs(received_clients[0][1], client)
        self.assertEqual(received_clients[1][0], 'error')
        self.assertIs(received_clients[1][1], client)

        # Verify the client passed to callbacks has the expected attributes
        self.assertIs(received_clients[0][1].on_connect, on_connect)
        self.assertIs(received_clients[1][1].on_error, on_error)

        client.close()
