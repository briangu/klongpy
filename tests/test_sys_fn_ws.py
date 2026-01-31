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

    def test_callback_receives_self_not_client_parameter(self):
        """
        Test the pattern: client.on_connect(self) - access callback via
        'client' param but pass 'self' to it.

        We simulate this by creating the inner function pattern directly
        and calling it with a fake client that's different from self.
        """
        import asyncio

        received = []

        # The "self" that should be passed to callbacks
        real_self = object()

        # The callback that records what it receives
        async def on_connect(instance):
            received.append(instance)

        async def on_error(instance, e):
            received.append(instance)

        # A fake "client" with the callbacks attached
        class FakeClient:
            pass
        fake_client = FakeClient()
        fake_client.on_connect = on_connect
        fake_client.on_error = on_error

        # This mirrors the pattern in run_client:
        # async def _on_connect(client, **kwargs):
        #     if client.on_connect is not None:
        #         await client.on_connect(self)  # <-- uses client. to access, self to pass
        async def _on_connect(client, **kwargs):
            if client.on_connect is not None:
                await client.on_connect(real_self)  # Pass real_self, not client

        async def _on_error(client, e):
            if client.on_error is not None:
                await client.on_error(real_self, e)  # Pass real_self, not client

        # Call with fake_client as the 'client' parameter
        asyncio.run(_on_connect(fake_client))
        asyncio.run(_on_error(fake_client, Exception("test")))

        # Verify callbacks received real_self, not fake_client
        self.assertEqual(len(received), 2)
        self.assertIs(received[0], real_self)
        self.assertIs(received[1], real_self)
        self.assertIsNot(received[0], fake_client)
        self.assertIsNot(received[1], fake_client)
