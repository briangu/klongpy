import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from utils import LoopsBase

from klongpy import KlongInterpreter
from klongpy.ws.sys_fn_ws import (
    NumpyEncoder,
    encode_message,
    decode_message,
    ConnectionProvider,
    ClientConnectionProvider,
    ExistingConnectionProvider,
    NetworkClient,
    KlongWSCreateConnectionException,
    KlongWSConnectionFailureException,
    create_system_functions_websockets,
    create_system_var_websockets,
    eval_sys_fn_shutdown_client,
)


class TestNumpyEncoder(unittest.TestCase):
    def test_encode_numpy_array(self):
        arr = np.array([1, 2, 3])
        result = json.dumps(arr, cls=NumpyEncoder)
        self.assertEqual(result, "[1, 2, 3]")

    def test_encode_nested_numpy(self):
        data = {"arr": np.array([1, 2]), "val": 42}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        self.assertEqual(parsed["arr"], [1, 2])
        self.assertEqual(parsed["val"], 42)


class TestEncodeDecode(unittest.TestCase):
    def test_encode_decode_message(self):
        msg = {"key": "value", "num": 42}
        encoded = encode_message(msg)
        decoded = decode_message(encoded)
        self.assertEqual(decoded, msg)

    def test_encode_decode_list(self):
        msg = [1, 2, 3, "test"]
        encoded = encode_message(msg)
        decoded = decode_message(encoded)
        self.assertEqual(decoded, msg)


class TestConnectionProvider(unittest.TestCase):
    def test_base_provider_raises_exception(self):
        provider = ConnectionProvider()
        with self.assertRaises(KlongWSCreateConnectionException):
            asyncio.run(provider.connect())


class TestClientConnectionProvider(unittest.TestCase):
    def test_str(self):
        provider = ClientConnectionProvider("ws://localhost:8080")
        self.assertEqual(str(provider), "remote[ws://localhost:8080]")

    def test_is_open_no_websocket(self):
        provider = ClientConnectionProvider("ws://localhost:8080")
        self.assertFalse(provider.is_open())

    def test_is_open_with_websocket(self):
        provider = ClientConnectionProvider("ws://localhost:8080")
        mock_ws = MagicMock()
        mock_ws.closed = False
        provider.websocket = mock_ws
        self.assertTrue(provider.is_open())

    def test_is_open_with_closed_websocket(self):
        provider = ClientConnectionProvider("ws://localhost:8080")
        mock_ws = MagicMock()
        mock_ws.closed = True
        provider.websocket = mock_ws
        self.assertFalse(provider.is_open())

    @patch("klongpy.ws.sys_fn_ws.websockets.connect")
    def test_connect_failure(self, mock_connect):
        mock_connect.side_effect = Exception("Connection failed")
        provider = ClientConnectionProvider("ws://localhost:8080")
        with self.assertRaises(KlongWSCreateConnectionException):
            asyncio.run(provider.connect())

    def test_close_with_no_websocket(self):
        provider = ClientConnectionProvider("ws://localhost:8080")
        asyncio.run(provider.close())
        self.assertIsNone(provider.websocket)


class TestExistingConnectionProvider(unittest.TestCase):
    def test_str(self):
        provider = ExistingConnectionProvider(None, "ws://test:1234")
        self.assertEqual(str(provider), "remote[ws://test:1234]")

    def test_connect_no_websocket_raises(self):
        provider = ExistingConnectionProvider(None, "ws://test:1234")
        with self.assertRaises(KlongWSCreateConnectionException):
            asyncio.run(provider.connect())

    def test_connect_with_websocket(self):
        mock_ws = MagicMock()
        provider = ExistingConnectionProvider(mock_ws, "ws://test:1234")
        result = asyncio.run(provider.connect())
        self.assertEqual(result, mock_ws)

    def test_close(self):
        mock_ws = AsyncMock()
        provider = ExistingConnectionProvider(mock_ws, "ws://test:1234")
        asyncio.run(provider.close())
        self.assertIsNone(provider.websocket)


class TestNetworkClientBasics(LoopsBase, unittest.TestCase):
    def test_initialization(self):
        klong = KlongInterpreter()
        provider = MagicMock()
        client = NetworkClient(self.ioloop, self.klongloop, klong, provider)
        self.assertEqual(client.ioloop, self.ioloop)
        self.assertEqual(client.klongloop, self.klongloop)
        self.assertEqual(client.klong, klong)
        self.assertFalse(client.running)

    def test_str(self):
        klong = KlongInterpreter()
        provider = MagicMock()
        provider.__str__ = MagicMock(return_value="test_provider")
        client = NetworkClient(self.ioloop, self.klongloop, klong, provider)
        self.assertEqual(str(client), "test_provider:fn")

    def test_get_arity(self):
        klong = KlongInterpreter()
        provider = MagicMock()
        client = NetworkClient(self.ioloop, self.klongloop, klong, provider)
        self.assertEqual(client.get_arity(), 1)

    def test_is_open(self):
        klong = KlongInterpreter()
        provider = MagicMock()
        provider.is_open.return_value = True
        client = NetworkClient(self.ioloop, self.klongloop, klong, provider)
        self.assertTrue(client.is_open())

    def test_create_from_uri(self):
        klong = KlongInterpreter()
        client = NetworkClient.create_from_uri(
            self.ioloop, self.klongloop, klong, "ws://localhost:8080"
        )
        self.assertIsInstance(client, NetworkClient)
        self.assertIsInstance(client.conn_provider, ClientConnectionProvider)


class TestShutdownClient(unittest.TestCase):
    def test_shutdown_closed_client(self):
        mock_client = MagicMock()
        mock_client.is_open.return_value = False
        result = eval_sys_fn_shutdown_client(mock_client)
        self.assertEqual(result, 0)


class TestSystemFunctions(unittest.TestCase):
    def test_create_system_functions_websockets(self):
        registry = create_system_functions_websockets()
        self.assertIsInstance(registry, dict)
        self.assertIn(".ws", registry)
        self.assertIn(".wsc", registry)

    def test_create_system_var_websockets(self):
        registry = create_system_var_websockets()
        self.assertIsInstance(registry, dict)
        self.assertIn(".srv.o", registry)
        self.assertIn(".srv.c", registry)
        self.assertIn(".srv.e", registry)


if __name__ == '__main__':
    unittest.main()
