import asyncio
import threading
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from klongpy import KlongInterpreter
from klongpy.sys_fn_ipc import *


# This utility will run the async test functions
def async_test(func):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(func(*args, **kwargs))
        loop.close()
    return wrapper


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


class TestAsync(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()


    def test_async_fn(self):
        klong = KlongInterpreter()

        loop = asyncio.get_event_loop()
        klong['.system'] = {'ioloop': None, 'klongloop': loop}

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

        loop.run_until_complete(_test())
        loop.run_until_complete(_test_result())

    def test_async_python_lambda_fn(self):
        klong = KlongInterpreter()
        loop = asyncio.get_event_loop()
        klong['.system'] = {'ioloop': None, 'klongloop': loop}

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


class TestConnectionProvider(unittest.TestCase):

    def test_connection_provider_raises_exception(self):
        conn = ConnectionProvider()
        with self.assertRaises(KlongConnectionException):
            asyncio.run(conn.connect())


class TestHostPortConnectionProvider(unittest.TestCase):

    def setUp(self):
        self.ioloop = asyncio.new_event_loop()
        self.ioloop_thread = threading.Thread(target=self.start_ioloop)
        self.ioloop_thread.start()
        self.finished_event = threading.Event()

    def tearDown(self):
        self.ioloop.call_soon_threadsafe(self.ioloop.stop)
        self.ioloop_thread.join()

    def start_ioloop(self):
        asyncio.set_event_loop(self.ioloop)
        self.ioloop.run_forever()

    def run_coroutine_threadsafe(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self.ioloop)
        return future.result()

    def run_coroutine_in_ioloop(self, coro):
        async def wrapper():
            await coro
            self.finished_event.set()  # Signal that the coroutine has finished
        
        self.ioloop.call_soon_threadsafe(asyncio.create_task, wrapper())

    @patch('klongpy.sys_fn_ipc.asyncio.open_connection')
    def test_successful_connection(self, mock_open_connection):
        mock_reader = MagicMock()
        mock_writer = MagicMock()
        mock_open_connection.return_value = mock_reader, mock_writer

        provider = HostPortConnectionProvider(None, "localhost", 8080)
        reader, writer = self.run_coroutine_threadsafe(provider.connect())

        self.assertEqual(reader, mock_reader)
        self.assertEqual(writer, mock_writer)

    @patch('klongpy.sys_fn_ipc.asyncio.open_connection', new_callable=AsyncMock)
    def test_retry_logic(self, mock_open_connection):
        mock_open_connection.side_effect = [OSError(), OSError(), (MagicMock(), MagicMock())]  # Fail twice, then succeed

        provider = HostPortConnectionProvider(None, "localhost", 8080, max_retries=5, retry_delay=0)
        self.run_coroutine_in_ioloop(provider.connect())
        self.finished_event.wait()

        self.assertEqual(mock_open_connection.call_count, 3)
        self.assertIsNotNone(provider.reader)
        self.assertIsNotNone(provider.writer)

    def test_is_open(self):
        provider = HostPortConnectionProvider(None, "localhost", 8080)
        self.assertFalse(provider.is_open())

        provider.writer = MagicMock()
        provider.writer.is_closing.return_value = False
        self.assertTrue(provider.is_open())

        provider.writer.is_closing.return_value = True
        self.assertFalse(provider.is_open())

    def test_close(self):
        provider = HostPortConnectionProvider(None, "localhost", 8080)
        reader = MagicMock()
        writer = MagicMock()
        provider.reader = reader
        provider.writer = writer
        provider._thread_ident = self.ioloop_thread.ident
        provider.writer.wait_closed = AsyncMock()
        provider.writer.is_closing.return_value = False

        self.assertTrue(provider.is_open())

        # Run the close method in the ioloop thread without waiting for its completion
        self.run_coroutine_in_ioloop(provider.close())
        self.finished_event.wait()

        self.assertIsNone(provider.reader)
        self.assertIsNone(provider.writer)
        writer.close.assert_called_once()
        writer.wait_closed.assert_called_once()

        self.assertFalse(provider.is_open())


class TestReaderWriterConnectionProvider(unittest.TestCase):

    def test_readerwriter_connection_provider_at_eof(self):
        mock_reader = MagicMock(at_eof=MagicMock(return_value=True))
        mock_writer = MagicMock()
        conn = ReaderWriterConnectionProvider(mock_reader, mock_writer)
        with self.assertRaises(KlongConnectionException):
            asyncio.run(conn.connect())

    def test_readerwriter_connection_provider_successful_connection(self):
        mock_reader = MagicMock(at_eof=MagicMock(return_value=False))
        mock_writer = MagicMock()
        conn = ReaderWriterConnectionProvider(mock_reader, mock_writer)
        reader, writer = asyncio.run(conn.connect())
        self.assertEqual(reader, mock_reader)
        self.assertEqual(writer, mock_writer)


class TestNetworkClient(unittest.TestCase):

    def setUp(self):
        self.ioloop = asyncio.new_event_loop()
        self.ioloop_thread = threading.Thread(target=self.start_ioloop)
        self.ioloop_thread.start()

        self.klongloop = asyncio.new_event_loop()
        self.klongloop_thread = threading.Thread(target=self.start_klongloop)
        self.klongloop_thread.start()

    def tearDown(self):
        self.ioloop.call_soon_threadsafe(self.ioloop.stop)
        self.ioloop_thread.join()

        self.klongloop.call_soon_threadsafe(self.klongloop.stop)
        self.klongloop_thread.join()

    def start_ioloop(self):
        asyncio.set_event_loop(self.ioloop)
        self.ioloop.run_forever()

    def start_klongloop(self):
        asyncio.set_event_loop(self.klongloop)
        self.klongloop.run_forever()

    def test_initialization(self):
        klong = KlongInterpreter()
        conn_provider = HostPortConnectionProvider(self.ioloop, "localhost", 1234)
        client = NetworkClient(self.ioloop, self.klongloop, klong, conn_provider)
        self.assertEqual(client.ioloop, self.ioloop)
        self.assertEqual(client.klongloop, self.klongloop)
        self.assertEqual(client.klong, klong)
        self.assertEqual(client.conn_provider, conn_provider)
        self.assertTrue(client.running)

    @patch("klongpy.sys_fn_ipc.asyncio.open_connection", new_callable=AsyncMock)
    def test_connect(self, mock_open_connection):
        reader = AsyncMock()
        writer = AsyncMock()
        writer.close = MagicMock()
        mock_open_connection.return_value = (reader, writer)

        connected_event = threading.Event()

        async def _test_on_connect(client):
            self.assertEqual(threading.current_thread().ident, self.ioloop_thread.ident)
            self.assertEqual(client.reader, reader)
            self.assertEqual(client.writer, writer)
            client.running = False
            connected_event.set()

        conn_provider = HostPortConnectionProvider(self.ioloop, "localhost", 1234)
        client = NetworkClient(self.ioloop, self.klongloop, "klong", conn_provider, on_connect=_test_on_connect)

        connected_event.wait()

        self.assertTrue(client.reader is None)
        self.assertTrue(client.writer is None)
        client.close()


    @patch("klongpy.sys_fn_ipc.asyncio.open_connection", new_callable=AsyncMock)
    def test_max_retries(self, mock_open_connection):
        mock_open_connection.side_effect = ConnectionResetError()  # Simulate a connection error

        self.called_after_connect = False
        error_event = threading.Event()

        async def _test_on_connect(*args, **kwargs):
            self.called_after_connect = True

        async def _test_on_error(*args, **kwargs):
            error_event.set()

        conn_provider = HostPortConnectionProvider(self.ioloop, "localhost", 1234, max_retries=3, retry_delay=0)
        client = NetworkClient(self.ioloop, self.klongloop, "klong", conn_provider, on_connect=_test_on_connect, on_error=_test_on_error)

        error_event.wait()

        self.assertFalse(self.called_after_connect)  # after_connect should never be called
        self.assertEqual(client.reader, None)
        self.assertEqual(client.writer, None)
        self.assertEqual(mock_open_connection.call_count, 3)  # Check if the retries were called 3 times


    @patch.object(HostPortConnectionProvider, "connect", new_callable=AsyncMock)
    def test_call_no_writer(self, mock_connect):

        # Mock the connect method to raise a KlongConnectionException
        mock_connect.side_effect = KlongConnectionException("Unable to connect")

        # Create a mock Klong object
        klong = MagicMock()

        conn_provider = HostPortConnectionProvider(self.ioloop, "localhost", 1234)
        client = NetworkClient(self.ioloop, self.klongloop, klong, conn_provider)

        with self.assertRaises(KlongException) as context:
            client.call("some_message")

        self.assertTrue("connection not established" in str(context.exception))
        client.close()


    # @patch("klongpy.sys_fn_ipc.asyncio.open_connection", new_callable=AsyncMock)
    @patch("klongpy.sys_fn_ipc.uuid.uuid4", new_callable=MagicMock)
    @patch("klongpy.sys_fn_ipc.stream_send_msg")
    @patch("klongpy.sys_fn_ipc.stream_recv_msg")
    def test_call(self, mock_stream_recv_msg, mock_stream_send_msg, mock_uuid):

        # Create a mock Klong object
        klong = MagicMock()

        writer = AsyncMock()
        reader = AsyncMock()
        # mock_open_connection.return_value = (reader, writer)

        # Define the message and response
        msg_id = uuid.uuid4()
        msg = "test_message"

        mock_uuid.return_value = msg_id
        
        conn_provider = ReaderWriterConnectionProvider(reader, writer)
        client = NetworkClient(self.ioloop, self.klongloop, klong, conn_provider)

        # Make mock_stream_recv_msg return a value only once
        mock_stream_recv_msg.return_value = (msg_id, msg)

        # Wait for the client to establish a connection
        timeout = 10
        start_time = time.time()
        while client.writer is None and time.time() - start_time < timeout:
            time.sleep(0.1)

        response = client.call(msg)

        self.assertEqual(response, msg)

        client.close()


    async def mock_stream_recv_msg(reader):
        return None, "2+2"
    
    async def mock_stream_send_msg(writer, msg_id, msg):
        writer._client.running = False
        writer.write(msg)
    
    @patch("klongpy.sys_fn_ipc.stream_recv_msg", mock_stream_recv_msg)
    @patch("klongpy.sys_fn_ipc.stream_send_msg", mock_stream_send_msg)
    @patch("klongpy.sys_fn_ipc.asyncio.open_connection", new_callable=AsyncMock)
    def test_listen(self, mock_open_connection):
        writer = AsyncMock()
        writer.write = MagicMock()
        writer.close = MagicMock()
        reader = AsyncMock()

        mock_open_connection.return_value = (reader, writer)

        klong = KlongInterpreter()
        conn_provider = HostPortConnectionProvider(self.ioloop, "127.0.0.1", 8888)
        client = NetworkClient(self.ioloop, self.klongloop, klong, conn_provider)

        writer._client = client

        while(client.writer is None):
            time.sleep(0)

        while client.running:
            time.sleep(0)

        client.close()

        client.writer.write.assert_called_with(4)


if __name__ == '__main__':
    unittest.main()
