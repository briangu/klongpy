import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from klongpy import KlongInterpreter
from klongpy.sys_fn_ipc import *


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
        client = NetworkClient(self.ioloop, self.klongloop, klong, "localhost", 1234)
        self.assertEqual(client.ioloop, self.ioloop)
        self.assertEqual(client.klongloop, self.klongloop)
        self.assertEqual(client.klong, klong)
        self.assertEqual(client.host, "localhost")
        self.assertEqual(client.port, 1234)
        self.assertEqual(client.max_retries, 5)
        self.assertEqual(client.retry_delay, 5.0)
        self.assertTrue(client.running)

    @patch("klongpy.sys_fn_ipc.asyncio.open_connection", new_callable=AsyncMock)
    def test_connect(self, mock_open_connection):
        reader = AsyncMock()
        writer = AsyncMock()
        writer.close = MagicMock()
        mock_open_connection.return_value = (reader, writer)

        self.called_after_connect = False
        async def mock_after_connect():
            self.called_after_connect = True

        client = NetworkClient(self.ioloop, self.klongloop, "klong", "localhost", 1234, after_connect=mock_after_connect)
        while client.writer is None:
            time.sleep(0)

        self.assertTrue(self.called_after_connect)
        self.assertEqual(client.reader, reader)
        self.assertEqual(client.writer, writer)
        client.close()

    def test_max_retries(self):
        self.mock_open_connect_called = False
        async def mock_open_connection(*args, **kwargs):
            self.mock_open_connect_called = True
            raise ConnectionResetError()
        
        with patch('klongpy.sys_fn_ipc.asyncio.open_connection', side_effect=mock_open_connection):
            self.called_after_connect = False
            async def mock_after_connect():
                self.called_after_connect = True
            client = NetworkClient(self.ioloop, self.klongloop, "klong", "localhost", 1234, retry_delay=0, after_connect=mock_after_connect)
            asyncio.run(client.connect(after_connect=mock_after_connect))
            self.assertFalse(self.called_after_connect)
            self.assertEqual(client.reader, None)
            self.assertEqual(client.writer, None)
            client.close()

        self.assertTrue(self.mock_open_connect_called)

    @patch("klongpy.sys_fn_ipc.asyncio.open_connection", new_callable=AsyncMock)
    def test_call_no_writer(self, mock_open_connection):
 
        # Create a mock Klong object
        klong = MagicMock()

        writer = None
        reader = None

        mock_open_connection.return_value = (reader, writer)

        client = NetworkClient(self.ioloop, self.klongloop, klong, '127.0.0.1', 8888)

        with self.assertRaises(KlongException) as context:
            client.call("some_message")

        self.assertTrue("connection not established" in str(context.exception))

        client.close()

    @patch("klongpy.sys_fn_ipc.asyncio.open_connection", new_callable=AsyncMock)
    @patch("klongpy.sys_fn_ipc.uuid.uuid4", new_callable=MagicMock)
    @patch("klongpy.sys_fn_ipc.stream_send_msg")
    @patch("klongpy.sys_fn_ipc.stream_recv_msg")
    def test_call(self, mock_stream_recv_msg, mock_stream_send_msg, mock_uuid, mock_open_connection):
 
        # Create a mock Klong object
        klong = MagicMock()

        writer = AsyncMock()
        writer.write = MagicMock()
        writer.close = MagicMock()
        reader = AsyncMock()

        mock_open_connection.return_value = (reader, writer)

        # Define the message and response
        msg_id = uuid.uuid4()
        msg = "test_message"

        mock_uuid.return_value = msg_id
        mock_stream_recv_msg.return_value = (msg_id, msg)

        client = NetworkClient(self.ioloop, self.klongloop, klong, '127.0.0.1', 8888)

        while(client.writer is None):
            pass

        response = client.call(msg)

        self.assertEqual(response, msg)

        client.close()

        # Ensure the stream_send_msg was called with the correct parameters
        # mock_stream_send_msg.assert_called_once_with(client.writer, msg_id, msg)

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
        client = NetworkClient(self.ioloop, self.klongloop, klong, '127.0.0.1', 8888)

        writer._client = client

        while(client.writer is None):
            time.sleep(0)

        while client.running:
            time.sleep(0)

        client.close()

        client.writer.write.assert_called_with(4)


if __name__ == '__main__':
    unittest.main()
