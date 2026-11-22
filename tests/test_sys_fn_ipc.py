import asyncio
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from ipc_helpers import FakeKdbServer, bounded, wait_until
from utils import LoopsBase

from klongpy import KlongInterpreter
from klongpy.core import KlongException
from klongpy.ipc import Session
from klongpy.ipc.endpoint import Waiter
from klongpy.sys_fn_ipc import (ConnectionProvider, HostPortConnectionProvider,
                                KGAsyncCall, KlongIPCConnectionClosedException,
                                KlongIPCCreateConnectionException,
                                KlongIPCHandshakeException, NetworkClient,
                                ReaderWriterConnectionProvider)
from klongpy.utils import CallbackEvent


def run_coroutine_threadsafe(coro, loop):
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


class TestAsync(LoopsBase, unittest.TestCase):

    def test_async_fn(self):
        klong = KlongInterpreter()

        klong['.system'] = {'ioloop': self.ioloop, 'klongloop': self.klongloop}

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

        run_coroutine_threadsafe(_test(), self.klongloop)
        run_coroutine_threadsafe(_test_result(), self.klongloop)

    def test_async_python_lambda_fn(self):
        klong = KlongInterpreter()
        klong['.system'] = {'ioloop': self.ioloop, 'klongloop': self.klongloop}

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

        run_coroutine_threadsafe(_test(), self.klongloop)
        run_coroutine_threadsafe(_test_result(), self.klongloop)

    def test_async_python_remote_fn(self):
        """.async on a kdb+ connection sends without blocking; the callback runs on the answer."""
        klong = KlongInterpreter()
        klong['.system'] = {'ioloop': self.ioloop, 'klongloop': self.klongloop}

        # a fake remote network client that mimics the remote execution of lambda x: x+1
        nc = NetworkClient(self.ioloop, self.klongloop, klong, None)

        def fake_call_async(msg, on_done):
            self.assertEqual(msg, 2)
            self.klongloop.call_soon_threadsafe(on_done, 3, None)

        nc.call_async = MagicMock(side_effect=fake_call_async)
        nc.call = MagicMock(side_effect=AssertionError("a remote .async must not block in call"))

        async def _test():
            klong["fn"] = nc
            klong("result::0")
            klong("cb::{result::x}")
            klong("afn::.async(fn;cb)")
            r = klong("afn(2)")
            self.assertEqual(r,1)

        async def _test_result():
            r = klong("result")
            self.assertEqual(r,3)

        run_coroutine_threadsafe(_test(), self.klongloop)
        run_coroutine_threadsafe(_test_result(), self.klongloop)

        # ensure that the nc was called with the correct arguments
        nc.call_async.assert_called_once()
        nc.call.assert_not_called()

    def test_async_callback_resolves_latest(self):
        """Test that async wrapper picks up callback redefinitions"""
        klong = KlongInterpreter()
        klong['.system'] = {'ioloop': self.ioloop, 'klongloop': self.klongloop}

        async def _test():
            klong("fn::{x+1}")
            klong("result::0")
            klong("cb::{result::x}")  # First callback
            klong("afn::.async(fn;cb)")
            r = klong("afn(2)")
            self.assertEqual(r, 1)

        async def _test_result1():
            # Wait for first callback to execute
            await asyncio.sleep(0.1)
            r = klong("result")
            self.assertEqual(r, 3)  # fn(2) = 3, cb stores it

        async def _test_redefine():
            # Redefine callback
            klong("cb::{result::x*10}")
            r = klong("afn(3)")  # fn(3) = 4
            self.assertEqual(r, 1)

        async def _test_result2():
            # Wait for second callback to execute
            await asyncio.sleep(0.1)
            r = klong("result")
            self.assertEqual(r, 40)  # fn(3) = 4, new cb does 4*10 = 40

        run_coroutine_threadsafe(_test(), self.klongloop)
        run_coroutine_threadsafe(_test_result1(), self.klongloop)
        run_coroutine_threadsafe(_test_redefine(), self.klongloop)
        run_coroutine_threadsafe(_test_result2(), self.klongloop)


class CallableWithArity:
    arity = 1

    def __call__(self, x):
        return x + 1


class TestAsyncCall(unittest.IsolatedAsyncioTestCase):
    async def test_async_call_python_callable(self):
        klong = KlongInterpreter()
        fn = CallableWithArity()
        cb = MagicMock()

        async_call = KGAsyncCall(None, fn, cb, klong)
        await async_call.acall(klong, [2])

        cb.assert_called_once_with(3)


class TestConnectionProvider(unittest.TestCase):

    def test_connection_provider_raises_exception(self):
        conn = ConnectionProvider()
        with self.assertRaises(KlongIPCCreateConnectionException):
            asyncio.run(conn.connect())


class TestHostPortConnectionProvider(LoopsBase, unittest.TestCase):

    @patch('klongpy.sys_fn_ipc.asyncio.open_connection')
    def test_successful_connection(self, mock_open_connection):
        mock_reader = MagicMock()
        mock_writer = MagicMock()
        mock_open_connection.return_value = mock_reader, mock_writer

        provider = HostPortConnectionProvider("localhost", 8080)
        reader, writer = run_coroutine_threadsafe(provider.connect(), self.ioloop)

        self.assertEqual(reader, mock_reader)
        self.assertEqual(writer, mock_writer)

    @patch('klongpy.sys_fn_ipc.asyncio.open_connection', new_callable=AsyncMock)
    def test_retry_logic(self, mock_open_connection):
        mock_open_connection.side_effect = [OSError(), OSError(), (MagicMock(), MagicMock())]  # Fail twice, then succeed

        provider = HostPortConnectionProvider("localhost", 8080, max_retries=5, retry_delay=0)
        run_coroutine_threadsafe(provider.connect(), self.ioloop)

        self.assertEqual(mock_open_connection.call_count, 3)
        self.assertIsNotNone(provider.reader)
        self.assertIsNotNone(provider.writer)

    def test_is_open(self):
        provider = HostPortConnectionProvider("localhost", 8080)
        self.assertFalse(provider.is_open())

        provider.writer = MagicMock()
        provider.writer.is_closing.return_value = False
        self.assertTrue(provider.is_open())

        provider.writer.is_closing.return_value = True
        self.assertFalse(provider.is_open())

    def test_close(self):
        provider = HostPortConnectionProvider("localhost", 8080)
        reader = MagicMock()
        writer = MagicMock()
        provider.reader = reader
        provider.writer = writer
        provider._thread_ident = self.ioloop_thread.ident
        provider.writer.wait_closed = AsyncMock()
        provider.writer.is_closing.return_value = False

        self.assertTrue(provider.is_open())

        # Run the close method in the ioloop thread without waiting for its completion
        run_coroutine_threadsafe(provider.close(), self.ioloop)

        self.assertIsNone(provider.reader)
        self.assertIsNone(provider.writer)
        writer.close.assert_called_once()
        writer.wait_closed.assert_called_once()

        self.assertFalse(provider.is_open())

    def test_str(self):
        host = "localhost"
        port = 1234
        conn = HostPortConnectionProvider(host, port)
        self.assertEqual(str(conn), f"remote[{host}:{port}]")


class TestReaderWriterConnectionProvider(unittest.TestCase):

    def test_readerwriter_connection_provider_at_eof(self):
        mock_reader = MagicMock(at_eof=MagicMock(return_value=True))
        mock_writer = MagicMock()
        conn = ReaderWriterConnectionProvider(mock_reader, mock_writer, "localhost", 1234)
        with self.assertRaises(KlongIPCCreateConnectionException):
            asyncio.run(conn.connect())

    def test_readerwriter_connection_provider_successful_connection(self):
        mock_reader = MagicMock()
        mock_writer = MagicMock(is_closing=MagicMock(return_value=False))
        conn = ReaderWriterConnectionProvider(mock_reader, mock_writer, "localhost", 1234)
        reader, writer = asyncio.run(conn.connect())
        self.assertEqual(reader, mock_reader)
        self.assertEqual(writer, mock_writer)

    def test_str(self):
        host = "localhost"
        port = 1234
        conn = ReaderWriterConnectionProvider(None, None, host, port)
        self.assertEqual(str(conn), f"remote[{host}:{port}]")


class TestNetworkClient(LoopsBase, unittest.TestCase):

    def test_initialization(self):
        klong = KlongInterpreter()
        conn_provider = HostPortConnectionProvider("localhost", 1234)
        client = NetworkClient(self.ioloop, self.klongloop, klong, conn_provider)
        self.assertEqual(client.ioloop, self.ioloop)
        self.assertEqual(client.klongloop, self.klongloop)
        self.assertEqual(client.klong, klong)
        self.assertEqual(client.conn_provider, conn_provider)
        self.assertEqual(client.reader, None)
        self.assertEqual(client.writer, None)
        self.assertEqual(client.pending_responses, {})
        self.assertEqual(client.running, False)

    def test_initialization_with_callbacks(self):
        klong = KlongInterpreter()
        conn_provider = HostPortConnectionProvider("localhost", 1234)
        on_connect = lambda x: x
        on_error = lambda x: x
        on_close = lambda x: x

        client = NetworkClient(self.ioloop, self.klongloop, klong, conn_provider, on_connect=on_connect, on_error=on_error, on_close=on_close)
        self.assertEqual(client.ioloop, self.ioloop)
        self.assertEqual(client.klongloop, self.klongloop)
        self.assertEqual(client.klong, klong)
        self.assertEqual(client.conn_provider, conn_provider)
        self.assertEqual(client.reader, None)
        self.assertEqual(client.writer, None)
        self.assertEqual(client.pending_responses, {})
        self.assertEqual(client.running, False)
        self.assertEqual(client.on_connect, on_connect)
        self.assertEqual(client.on_error, on_error)
        self.assertEqual(client.on_close, on_close)

    def test_str(self):
        klong = KlongInterpreter()
        conn_provider = HostPortConnectionProvider("localhost", 1234)
        client = NetworkClient(self.ioloop, self.klongloop, klong, conn_provider)
        self.assertEqual(str(client), f"{str(conn_provider)}:fn")

    # -- helpers -----------------------------------------------------------

    def _peer(self, **kw):
        peer = FakeKdbServer(**kw)
        self.addCleanup(peer.stop)
        return peer

    def _provider(self, peer):
        return HostPortConnectionProvider("127.0.0.1", peer.port, max_retries=1, retry_delay=0)

    def _open_stream_provider(self, peer):
        """A ReaderWriterConnectionProvider over a real connection to the fake peer."""
        reader, writer = asyncio.run_coroutine_threadsafe(
            asyncio.open_connection("127.0.0.1", peer.port), self.ioloop).result(5)
        return ReaderWriterConnectionProvider(reader, writer, "127.0.0.1", peer.port)

    def _ready_session(self, role):
        """A session past its handshake, plus the matching peer session."""
        cli = Session.client("u", "p")
        srv = Session.server()
        srv.feed(cli.hello())
        reply = srv.accept()
        cli.feed(reply)
        return (cli, srv) if role == "client" else (srv, cli)

    def test_connect(self):
        klong = KlongInterpreter()
        peer = self._peer(handler=lambda v: v)
        seen = {}

        async def _test_on_connect(client):
            seen["thread"] = threading.current_thread().ident
            seen["reader"], seen["writer"] = client.reader, client.writer
            client.running = False

        conn_provider = self._open_stream_provider(peer)
        reader, writer = conn_provider.reader, conn_provider.writer
        client = NetworkClient(self.ioloop, self.klongloop, klong, conn_provider, on_connect=_test_on_connect)
        client.run_client()

        # after client.running is set to False, the _run method will exit
        # block until the _run method has finished
        self.assertTrue(client._run_exit_event.wait(5))

        self.assertEqual(seen["thread"], self.ioloop_thread.ident)
        self.assertIs(seen["reader"], reader)
        self.assertIs(seen["writer"], writer)
        self.assertEqual(peer.greetings, [("", "", 3)])      # the handshake was sent: ":" + capability 3
        self.assertFalse(client.running)
        self.assertEqual(client.reader, None)
        self.assertEqual(client.writer, None)

    def test_handshake_sends_credentials(self):
        peer = self._peer(handler=lambda v: v)
        klong = KlongInterpreter()
        client = NetworkClient(self.ioloop, self.klongloop, klong, self._provider(peer), user="bob", password="s:cret")
        client.run_client()
        try:
            self.assertEqual(peer.greetings, [("bob", "s:cret", 3)])
            self.assertEqual(client.call("1+1"), "1+1")           # the peer echoes
        finally:
            bounded(client.close)       # LoopsBase.tearDown waits for the client's loop tasks

    def test_close(self):
        klong = KlongInterpreter()
        peer = self._peer(handler=lambda v: v)
        connected = threading.Event()

        async def _test_on_connect(client):
            self.assertEqual(threading.current_thread().ident, self.ioloop_thread.ident)
            self.assertIsNotNone(client.reader)
            self.assertIsNotNone(client.writer)
            connected.set()

        client = NetworkClient(self.ioloop, self.klongloop, klong, self._provider(peer), on_connect=_test_on_connect)
        client.run_client()
        self.assertTrue(connected.wait(5))

        bounded(client.close)

        self.assertFalse(client.running)
        self.assertEqual(client.reader, None)
        self.assertEqual(client.writer, None)
        self.assertFalse(client.is_open())

    def test_on_error(self):
        klong = MagicMock()
        klong._context = MagicMock()

        on_error_called = False
        async def _test_on_error(client, e):
            nonlocal on_error_called
            on_error_called = True
            client.running = False

        conn_provider = MagicMock()
        conn_provider.connect.side_effect = KlongIPCCreateConnectionException()
        client = NetworkClient(self.ioloop, self.klongloop, klong, conn_provider, on_error=_test_on_error)
        client.run_client()

        # after client.running is set to False, the _run method will exit
        # block until the _run method has finished
        client._run_exit_event.wait()

        self.assertTrue(on_error_called)
        self.assertFalse(client.running)
        self.assertEqual(client.reader, None)
        self.assertEqual(client.writer, None)

    def test_on_close(self):
        peer = self._peer(handler=lambda v: v)

        async def _test_on_connect(client):
            client.running = False

        on_close_called = False
        async def _test_on_close(client):
            nonlocal on_close_called
            on_close_called = True

        conn_provider = self._open_stream_provider(peer)
        client = NetworkClient(self.ioloop, self.klongloop, "klong", conn_provider, on_connect=_test_on_connect, on_close=_test_on_close)
        client.run_client()

        # after client.running is set to False, the _run method will exit
        # block until the _run method has finished
        client._run_exit_event.wait()

        self.assertTrue(on_close_called)
        self.assertFalse(client.running)
        self.assertEqual(client.reader, None)
        self.assertEqual(client.writer, None)

    def test_on_close_not_called_when_never_connected(self):
        """.srv.c must not fire for a connection whose handshake failed (.srv.o never fired)."""
        peer = self._peer(reject=True)
        closed = []

        async def _on_close(client):
            closed.append(client)

        client = NetworkClient(self.ioloop, self.klongloop, "klong", self._provider(peer), on_close=_on_close)
        with self.assertRaises(KlongException):
            bounded(client.run_client)
        self.assertEqual(closed, [])

    @patch("klongpy.sys_fn_ipc.asyncio.open_connection", new_callable=AsyncMock)
    def test_max_retries(self, mock_open_connection):
        mock_open_connection.side_effect = ConnectionResetError()  # Simulate a connection error

        called_after_connect = False
        error_event = threading.Event()

        async def _test_on_connect(*args, **kwargs):
            nonlocal called_after_connect
            called_after_connect = True

        async def _test_on_error(client, e):
            nonlocal error_event
            error_event.set()
            client.running = False

        conn_provider = HostPortConnectionProvider("localhost", 1234, max_retries=3, retry_delay=0)
        client = NetworkClient(self.ioloop, self.klongloop, "klong", conn_provider, on_connect=_test_on_connect, on_error=_test_on_error)
        client.run_client()

        client._run_exit_event.wait()
        error_event.wait()

        self.assertFalse(called_after_connect)  # after_connect should never be called
        self.assertEqual(client.reader, None)
        self.assertEqual(client.writer, None)
        self.assertEqual(mock_open_connection.call_count, 3)  # Check if the retries were called 3 times

    def test_connect_after_failure(self):
        peer = self._peer(handler=lambda v: v)
        real_open = asyncio.open_connection
        calls = 0

        async def flaky_open(host, port):
            nonlocal calls
            calls += 1
            if calls <= 2:
                raise OSError()
            return await real_open("127.0.0.1", peer.port)

        called_after_error = False
        called_after_connect = False

        async def _test_on_connect(client, **kwargs):
            nonlocal called_after_connect
            called_after_connect = True
            client.running = False

        async def _test_on_error(client, e):
            nonlocal called_after_error
            called_after_error = True

        conn_provider = HostPortConnectionProvider("localhost", 1234, max_retries=3, retry_delay=0)
        client = NetworkClient(self.ioloop, self.klongloop, "klong", conn_provider, on_connect=_test_on_connect, on_error=_test_on_error)
        with patch("klongpy.sys_fn_ipc.asyncio.open_connection", flaky_open):
            client.run_client()
            client._run_exit_event.wait()

        self.assertTrue(called_after_connect)
        self.assertFalse(called_after_error)
        self.assertEqual(client.reader, None)
        self.assertEqual(client.writer, None)
        self.assertEqual(calls, 3)

    def test_listen_pending_response(self):
        """A response read from the stream completes the oldest pending request (no message ids)."""
        klong = MagicMock()
        cli, srv = self._ready_session("client")
        data, pend = cli.send_sync("1+1")
        msg = srv.feed(data)[0]
        svc = srv.begin_service(msg)
        response = srv.send_response(svc, 2)
        srv.end_service(svc)

        client = NetworkClient(self.ioloop, self.klongloop, klong, None)
        client.session = cli
        client.reader = AsyncMock()
        client.reader.read = AsyncMock(return_value=response)
        waiter = Waiter()
        client.pending_responses[pend] = waiter

        asyncio.run_coroutine_threadsafe(client._listen(), self.ioloop).result(5)

        self.assertEqual(waiter.inbox.get(timeout=1), ("response", 2))
        self.assertEqual(client.pending_responses, {})

    def test_listen_klong_request(self):
        """A request read from the stream is evaluated on the klong loop and answered on the stream."""
        klong = KlongInterpreter()
        srv, cli = self._ready_session("server")
        request, _ = cli.send_sync("1+1")

        client = NetworkClient(self.ioloop, self.klongloop, klong, None)
        client.session = srv
        client.reader = AsyncMock()
        client.reader.read = AsyncMock(return_value=request)
        written = []
        sent = threading.Event()
        client.writer = MagicMock()
        client.writer.is_closing.return_value = False
        client.writer.write.side_effect = lambda d: (written.append(d), sent.set())

        asyncio.run_coroutine_threadsafe(client._listen(), self.ioloop).result(5)

        self.assertTrue(sent.wait(5))
        # the whole response message: header, long atom 2 (tests/ipc_vectors "response-long-2")
        self.assertEqual(written, [bytes.fromhex("0102000011000000f90200000000000000")])

    def test_listen_eof_is_a_clean_close(self):
        cli, _ = self._ready_session("client")
        client = NetworkClient(self.ioloop, self.klongloop, MagicMock(), None)
        client.session = cli
        client.reader = AsyncMock()
        client.reader.read = AsyncMock(return_value=b"")
        with self.assertRaises(KlongIPCConnectionClosedException):
            asyncio.run_coroutine_threadsafe(client._listen(), self.ioloop).result(5)

    @patch.object(HostPortConnectionProvider, "connect", new_callable=AsyncMock)
    def test_call_disconnected(self, mock_connect):

        # Mock the connect method to raise a KlongConnectionException
        mock_connect.side_effect = KlongIPCCreateConnectionException("Unable to connect")

        # Create a mock Klong object
        klong = MagicMock()

        conn_provider = HostPortConnectionProvider("localhost", 1234)
        client = NetworkClient(self.ioloop, self.klongloop, klong, conn_provider)

        with self.assertRaises(KlongException) as context:
            client.call("some_message")

        self.assertTrue("connection not established" in str(context.exception))
        client.close()

    def test_call(self):
        klong = KlongInterpreter()
        peer = self._peer(handler=lambda v: v)
        client = NetworkClient(self.ioloop, self.klongloop, klong, self._provider(peer))
        client.run_client()

        response = bounded(lambda: client.call("test_message"))

        self.assertEqual(response, "test_message")
        self.assertEqual(peer.requests, [("sync", "test_message")])
        bounded(client.close)

    def test_call_server_exception(self):
        """An error object from the peer is raised by the call, carrying the peer's text."""
        from klongpy.ipc.endpoint import KlongRemoteException

        def boom(v):
            raise ValueError("type: no good")

        peer = self._peer(handler=boom)
        client = NetworkClient(self.ioloop, self.klongloop, KlongInterpreter(), self._provider(peer))
        client.run_client()
        try:
            with self.assertRaises(KlongRemoteException) as cm:
                bounded(lambda: client.call("x"))
            self.assertEqual(cm.exception.text, "type: no good")
            self.assertEqual(str(cm.exception), "no good")
            self.assertEqual(cm.exception.wire_name, "type")
            self.assertTrue(client.is_open())               # an error answer does not close the connection
        finally:
            bounded(client.close)

    def test_call_server_fail(self):
        """When the server closes the connection in the middle of a call, the call fails."""
        peer = self._peer(disconnect_on_request=True)
        client = NetworkClient(self.ioloop, self.klongloop, KlongInterpreter(), self._provider(peer))
        client.run_client()
        try:
            with self.assertRaisesRegex(KlongIPCConnectionClosedException, "connection closed"):
                bounded(lambda: client.call("x"), 5)
            self.assertEqual(peer.requests, [("sync", "x")])
        finally:
            bounded(client.close)

    def test_handshake_refused_raises_from_run_client(self):
        peer = self._peer(reject=True)
        client = NetworkClient(self.ioloop, self.klongloop, KlongInterpreter(), self._provider(peer))
        with self.assertRaises(KlongException) as cm:
            bounded(client.run_client)
        self.assertIn("handshake", str(cm.exception))

    def test_handshake_times_out_on_a_silent_peer(self):
        peer = self._peer(reply_handshake=False)
        client = NetworkClient(self.ioloop, self.klongloop, KlongInterpreter(), self._provider(peer))
        with patch("klongpy.ipc.endpoint.HANDSHAKE_TIMEOUT", 0.3):
            with self.assertRaises(KlongException) as cm:
                bounded(client.run_client, 5)
        self.assertIn("timed out", str(cm.exception))

    # test wrapping the NetworkClient in a NetworkClientDictHandle
    def test_network_client_dict_handler(self):
        pass

    # test client reconnect on connection failure
    def test_client_reconnect(self):
        pass

    # test client reconnect on connection failure with pending requests
    def test_client_reconnect_with_pending_requests(self):
        pass

# class TestServerNetworkClient(unittest):

#     def setUp(self):
#         self.ioloop = asyncio.new_event_loop()
#         self.ioloop_thread = threading.Thread(target=self.start_ioloop)
#         self.ioloop_thread.start()

#         self.klongloop = asyncio.new_event_loop()
#         self.klongloop_thread = threading.Thread(target=self.start_klongloop)
#         self.klongloop_thread.start()

#     def tearDown(self):
#         self.ioloop.call_soon_threadsafe(self.ioloop.stop)
#         self.ioloop_thread.join()

#         self.klongloop.call_soon_threadsafe(self.klongloop.stop)
#         self.klongloop_thread.join()

#     def start_ioloop(self):
#         asyncio.set_event_loop(self.ioloop)
#         self.ioloop.run_forever()

#     def start_klongloop(self):
#         asyncio.set_event_loop(self.klongloop)
#         self.klongloop.run_forever()

#     def test_initialization(self):


if __name__ == '__main__':
    unittest.main()
