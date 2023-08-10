import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

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
        writer.write = AsyncMock()

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
        # Close the event loop
        self.klongloop.call_soon_threadsafe(self.klongloop.stop)
        self.klongloop_thread.join()

        # Stop the ioloop and join the thread
        self.ioloop.call_soon_threadsafe(self.ioloop.stop)
        self.ioloop_thread.join()

    def start_ioloop(self):
        asyncio.set_event_loop(self.ioloop)
        self.ioloop.run_forever()
        print('exiting ioloop')

    def start_klongloop(self):
        asyncio.set_event_loop(self.klongloop)
        self.klongloop.run_forever()
        print('exiting klongloop')

    def test_initialization(self):
        ioloop = asyncio.new_event_loop()
        klongloop = asyncio.new_event_loop()
        try:
            klong = KlongInterpreter()
            client = NetworkClient(ioloop, klongloop, klong, "localhost", 1234)
            self.assertEqual(client.ioloop, ioloop)
            self.assertEqual(client.klongloop, klongloop)
            self.assertEqual(client.klong, klong)
            self.assertEqual(client.host, "localhost")
            self.assertEqual(client.port, 1234)
            self.assertEqual(client.max_retries, 5)
            self.assertEqual(client.retry_delay, 5.0)
            self.assertTrue(client.running)
        finally:
            ioloop.close()
            klongloop.close()

    def test_connect(self):
        mock_reader = AsyncMock()
        mock_writer = AsyncMock()

        async def mock_open_connection(*args, **kwargs):
            return mock_reader, mock_writer

        ioloop = asyncio.new_event_loop()
        klongloop = asyncio.new_event_loop()

        try:
            with patch('klongpy.sys_fn_ipc.asyncio.open_connection', side_effect=mock_open_connection):
                client = NetworkClient(ioloop, klongloop, "klong", "localhost", 1234,)
                async def mock_after_connect():
                    client.running = False
                asyncio.run(client.connect(after_connect=mock_after_connect))
                self.assertEqual(client.reader, mock_reader)
                self.assertEqual(client.writer, mock_writer)
        finally:
            ioloop.close()
            klongloop.close()

    def test_max_retries(self):
        async def mock_open_connection(*args, **kwargs):
            raise ConnectionResetError()
        
        ioloop = asyncio.new_event_loop()
        klongloop = asyncio.new_event_loop()
        try:
            with patch('klongpy.sys_fn_ipc.asyncio.open_connection', side_effect=mock_open_connection):
                client = NetworkClient(ioloop, klongloop, "klong", "localhost", 1234, retry_delay=0)
                called_after_connect = False
                async def mock_after_connect():
                    called_after_connect = True
                asyncio.run(client.connect(after_connect=mock_after_connect))
                self.assertFalse(called_after_connect)
                self.assertEqual(client.reader, None)
                self.assertEqual(client.writer, None)
        finally:
            ioloop.close()
            klongloop.close()


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
        mock_stream_send_msg.assert_called_once_with(client.writer, msg_id, msg)

    async def mock_stream_recv_msg(reader):
        reader._client.running = False
        return None, "2+2"
        
    async def mock_execute_server_command(klong, msg_id, msg, writer):
        assert msg == "2+2"
        writer.write("4")

    @patch("klongpy.sys_fn_ipc.stream_recv_msg", mock_stream_recv_msg)
    @patch("klongpy.sys_fn_ipc.execute_server_command", mock_execute_server_command)
    @patch("klongpy.sys_fn_ipc.asyncio.open_connection", new_callable=AsyncMock)
    def test_listen(self, mock_open_connection):
        writer = AsyncMock()
        writer.write = AsyncMock()

        reader = AsyncMock()

        mock_open_connection.return_value = (reader, writer)

        klong = KlongInterpreter()
        client = NetworkClient(self.ioloop, self.klongloop, klong, '127.0.0.1', 8888)

        fut = asyncio.run_coroutine_threadsafe(client.listen(), self.ioloop)
        fut.result()

        client.close()

        client.writer.write.assert_called_with("4")



if __name__ == '__main__':
    unittest.main()
