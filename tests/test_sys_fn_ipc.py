import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

from klongpy import KlongInterpreter
from klongpy.sys_fn_ipc import *


class TestEncodeDecode(unittest.TestCase):
    def test_encode_decode(self):
        msg_id = uuid.uuid4()
        msg = {'key': 'value'}
        encoded_message = encode_message(msg_id.bytes, msg)
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
        await stream_send_msg(writer, msg_id.bytes, msg)
        writer.write.assert_called_once()
        writer.drain.assert_awaited_once()

        # Simulate receiving the message
        received_msg_id, received_message = await stream_recv_msg(reader)

        self.assertEqual(received_msg_id, msg_id)
        self.assertEqual(received_message, msg)


class TestAsync(unittest.TestCase):

    def test_async_fn(self):
        klong = KlongInterpreter()

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

        loop = asyncio.get_event_loop()
        loop.run_until_complete(_test())
        loop.run_until_complete(_test_result())

    def test_async_python_lambda_fn(self):
        klong = KlongInterpreter()

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

    @patch('klongpy.sys_fn_ipc._main_loop')
    def test_initialization(self, mock_loop):
        client = NetworkClient(mock_loop, "klong", "localhost", 1234)
        self.assertEqual(client.ioloop, mock_loop)
        self.assertEqual(client.klong, "klong")
        self.assertEqual(client.host, "localhost")
        self.assertEqual(client.port, 1234)
        self.assertEqual(client.max_retries, 5)
        self.assertEqual(client.retry_delay, 5.0)
        self.assertTrue(client.running)
        self.assertEqual(client.ioloop, mock_loop)

    @patch('klongpy.sys_fn_ipc._main_loop')
    def test_connect(self, mock_loop):
        mock_reader = AsyncMock()
        mock_writer = AsyncMock()

        async def mock_open_connection(*args, **kwargs):
            return mock_reader, mock_writer

        with patch('klongpy.sys_fn_ipc.asyncio.open_connection', side_effect=mock_open_connection):
            client = NetworkClient(mock_loop, "klong", "localhost", 1234,)
            async def mock_after_connect():
                client.running = False
            asyncio.run(client.connect(after_connect=mock_after_connect))
            self.assertEqual(client.reader, mock_reader)
            self.assertEqual(client.writer, mock_writer)

    @patch('klongpy.sys_fn_ipc._main_loop')
    def test_max_retries(self, mock_loop):
        async def mock_open_connection(*args, **kwargs):
            raise ConnectionResetError()

        with patch('klongpy.sys_fn_ipc.asyncio.open_connection', side_effect=mock_open_connection):
            client = NetworkClient(mock_loop, "klong", "localhost", 1234, retry_delay=0)
            called_after_connect = False
            async def mock_after_connect():
                called_after_connect = True
            asyncio.run(client.connect(after_connect=mock_after_connect))
            self.assertFalse(called_after_connect)
            self.assertEqual(client.reader, None)
            self.assertEqual(client.writer, None)

    @patch('klongpy.sys_fn_ipc._main_loop'  )
    @patch('klongpy.sys_fn_ipc.uuid.uuid4')
    def test_call(self, mock_uuid, mock_loop):
        # Arrange
        msg_id = b"test_id"
        mock_uuid.return_value.bytes = msg_id
        mock_writer = AsyncMock()

        fake_future = asyncio.Future()
        fake_future.set_result("response")
        mock_loop.create_future = MagicMock(return_value=fake_future)

        client = NetworkClient(mock_loop, "klong", "localhost", 1234)
        client.writer = mock_writer  # Simulate an open connection

        # Act
        response = client.call("command")

        # Assert
        self.assertEqual(response, "response")
        self.assertIn(msg_id, client.pending_responses)

    async def mock_stream_recv_msg(reader):
        reader._client.running = False
        return None, "2+2"
        
    async def mock_execute_server_command(klong, msg_id, msg, writer):
        assert msg == "2+2"
        writer.write("4")

    @patch("klongpy.sys_fn_ipc.stream_recv_msg", mock_stream_recv_msg)
    @patch("klongpy.sys_fn_ipc.execute_server_command", mock_execute_server_command)
    def test_listen(self):
        loop = asyncio.get_event_loop()
        klong = None # Replace with the actual klong object if needed
        client = NetworkClient(loop, klong, "localhost", 1234)

        # Mock writer
        client.reader = AsyncMock()
        client.reader._client = client
        client.writer = AsyncMock()
        client.writer.write = AsyncMock()

        # Run listen method
        loop.run_until_complete(client.listen())

        # Check the result
        client.writer.write.assert_called_with("4")

if __name__ == '__main__':
    unittest.main()
