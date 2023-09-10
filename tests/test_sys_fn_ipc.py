import asyncio
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from utils import LoopsBase

from klongpy import KlongInterpreter
from klongpy.sys_fn_ipc import *
from klongpy.utils import CallbackEvent


def run_coroutine_threadsafe(coro, loop):
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


class TestEncodeDecode(unittest.TestCase):
    def test_encode_decode(self):
        msg_id = uuid.uuid4()
        msg = {'key': 'value'}
        encoded_message = encode_message(msg_id, msg)
        decoded_msglen = decode_message_len(encoded_message[16:20])
        decoded_msg_id, decoded_message = decode_message(encoded_message[:16], encoded_message[20:])
        
        self.assertEqual(decoded_msglen, len(pickle.dumps(msg)))
        self.assertEqual(decoded_msg_id, msg_id)
        self.assertEqual(decoded_message, msg)


class TestStreamSendRecv(unittest.IsolatedAsyncioTestCase):
    async def test_stream_send_recv(self):
        msg_id = uuid.uuid4()
        msg = {'key': 'value'}

        # Mock writer
        writer = AsyncMock()
        writer.write = MagicMock()
        writer.close = MagicMock()

        # Mock reader
        reader = AsyncMock()
        reader.readexactly = AsyncMock(side_effect=[
            msg_id.bytes, # raw_msg_id
            struct.pack("!I", len(pickle.dumps(msg))), # raw_msglen
            pickle.dumps(msg) # data
        ])

        # Simulate sending the message
        await stream_send_msg(writer, msg_id, msg)
        writer.write.assert_called_once()
        writer.drain.assert_awaited_once()

        # Simulate receiving the message
        received_msg_id, received_message = await stream_recv_msg(reader)

        self.assertEqual(received_msg_id, msg_id)
        self.assertEqual(received_message, msg)


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

    def test_connect(self):
        klong = KlongInterpreter()
        reader = AsyncMock()
        writer = AsyncMock()
        writer.write = MagicMock()
        writer.close = MagicMock()
        writer.is_closing = MagicMock(return_value=False)

        async def _test_on_connect(client):
            self.assertEqual(threading.current_thread().ident, self.ioloop_thread.ident)
            self.assertEqual(client.reader, reader)
            self.assertEqual(client.writer, writer)
            client.running = False

        conn_provider = ReaderWriterConnectionProvider(reader, writer, "localhost", 1234)
        client = NetworkClient(self.ioloop, self.klongloop, klong, conn_provider, on_connect=_test_on_connect)
        client.run_client()

        # after client.running is set to False, the _run method will exit
        # block until the _run method has finished
        client._run_exit_event.wait()

        self.assertFalse(client.running)
        self.assertEqual(client.reader, None)
        self.assertEqual(client.writer, None)

    @patch("klongpy.sys_fn_ipc.uuid.uuid4", new_callable=MagicMock)
    @patch("klongpy.sys_fn_ipc.stream_send_msg")
    @patch("klongpy.sys_fn_ipc.stream_recv_msg")
    def test_close(self, mock_stream_recv_msg, mock_stream_send_msg: MagicMock, mock_uuid):
        klong = MagicMock()

        reader = AsyncMock()
        writer = AsyncMock()
        writer.write = MagicMock()
        writer.close = MagicMock()
        writer.is_closing = MagicMock(return_value=False)

        msg_id = uuid.uuid4()
        # msg = KGRemoteCloseConnection()
        msg = "hello"

        mock_uuid.return_value = msg_id
        mock_stream_recv_msg.return_value = (msg_id, msg)

        async def _test_on_connect(client):
            self.assertEqual(threading.current_thread().ident, self.ioloop_thread.ident)
            self.assertEqual(client.reader, reader)
            self.assertEqual(client.writer, writer)

        conn_provider = ReaderWriterConnectionProvider(reader, writer, "localhost", 1234)
        client = NetworkClient(self.ioloop, self.klongloop, klong, conn_provider, on_connect=_test_on_connect)
        client.run_client()

        client.close()

        # mock_stream_send_msg.assert_called_with(writer, msg_id, msg)

        self.assertFalse(client.running)
        self.assertEqual(client.reader, None)
        self.assertEqual(client.writer, None)

    @patch("klongpy.sys_fn_ipc.stream_recv_msg")
    def test_on_error(self, mock_stream_recv_msg):
        klong = MagicMock()
        klong._context = MagicMock()

        # reader = AsyncMock()
        writer = AsyncMock()
        writer.write = MagicMock()
        writer.close = MagicMock()
        writer.is_closing = MagicMock(return_value=False)

        # mock_stream_recv_msg.return_value = (uuid.uuid4(), "test_message")

        # async def _test_on_connect(client):
        #     raise KlongIPCCreateConnectionException()

        on_error_called = False
        async def _test_on_error(client, e):
            nonlocal on_error_called
            on_error_called = True
            client.running = False

        # conn_provider = ReaderWriterConnectionProvider(reader, writer, "localhost", 1234)
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
        reader = AsyncMock()
        writer = AsyncMock()
        writer.close = MagicMock()
        writer.write = MagicMock()
        writer.is_closing = MagicMock(return_value=False)

        async def _test_on_connect(client):
            client.running = False

        on_close_called = False
        async def _test_on_close(client):
            nonlocal on_close_called
            on_close_called = True

        conn_provider = ReaderWriterConnectionProvider(reader, writer, "localhost", 1234)
        client = NetworkClient(self.ioloop, self.klongloop, "klong", conn_provider, on_connect=_test_on_connect, on_close=_test_on_close)
        client.run_client()

        # after client.running is set to False, the _run method will exit
        # block until the _run method has finished
        client._run_exit_event.wait()

        self.assertTrue(on_close_called)
        self.assertFalse(client.running)
        self.assertEqual(client.reader, None)
        self.assertEqual(client.writer, None)

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

    @patch("klongpy.sys_fn_ipc.asyncio.open_connection", new_callable=AsyncMock)
    def test_connect_after_failure(self, mock_open_connection):
        mock_open_connection.side_effect = [OSError(), OSError(), (MagicMock(), MagicMock())]

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
        client.run_client()

        client._run_exit_event.wait()

        self.assertTrue(called_after_connect)
        self.assertFalse(called_after_error)
        self.assertEqual(client.reader, None)
        self.assertEqual(client.writer, None)
        self.assertEqual(mock_open_connection.call_count, 3)

    @patch("klongpy.sys_fn_ipc.stream_send_msg")
    @patch("klongpy.sys_fn_ipc.stream_recv_msg")
    def test_listen_pending_response(self, mock_stream_recv_msg, mock_stream_send_msg):
        klong = MagicMock()

        # Define the message and response
        msg_id = uuid.uuid4()
        msg = "test_message"

        mock_stream_recv_msg.return_value = (msg_id, msg)

        client = NetworkClient(self.ioloop, self.klongloop, klong, None)
        future = self.ioloop.create_future()
        client.pending_responses[msg_id] = future

        run_coroutine_threadsafe(client._listen(), self.ioloop)

        self.assertEqual(future.result(), msg)

    @patch("klongpy.sys_fn_ipc.stream_send_msg")
    @patch("klongpy.sys_fn_ipc.stream_recv_msg")
    def test_listen_klong_request(self, mock_stream_recv_msg, mock_stream_send_msg):
        # Define the message and response
        msg_id = uuid.uuid4()
        msg = "test_message"

        klong = MagicMock()
        klong.return_value = msg

        mock_stream_recv_msg.return_value = (msg_id, msg)

        client = NetworkClient(self.ioloop, self.klongloop, klong, None)

        run_coroutine_threadsafe(client._listen(), self.ioloop)

        klong.assert_called_once_with(msg)
        mock_stream_send_msg.assert_called_once_with(client.writer, msg_id, msg)

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

    @patch("klongpy.sys_fn_ipc.uuid.uuid4", new_callable=MagicMock)
    @patch("klongpy.sys_fn_ipc.stream_send_msg")
    @patch("klongpy.sys_fn_ipc.stream_recv_msg")
    def test_call(self, mock_stream_recv_msg, mock_stream_send_msg, mock_uuid):

        klong = MagicMock()

        writer = AsyncMock()
        writer.is_closing = MagicMock(return_value=False)
        writer.close = MagicMock()
        reader = AsyncMock()

        msg_id = uuid.uuid4()
        msg = "test_message"

        mock_uuid.return_value = msg_id
        mock_stream_recv_msg.return_value = (msg_id, msg)

        conn_provider = ReaderWriterConnectionProvider(reader, writer, "localhost", 1234)
        client = NetworkClient(self.ioloop, self.klongloop, klong, conn_provider)
        client.run_client()

        response = client.call(msg)

        self.assertEqual(response, msg)

        client.close()

    # test that when the server closes the connection in the middle of a call, the client raises an exception        
    def test_call_server_fail(self):
        pass

    # test that when the server has a KlongException, the client raises an exception
    def test_call_server_exception(self):
        pass

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
