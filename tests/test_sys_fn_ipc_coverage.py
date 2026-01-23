import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from utils import LoopsBase

from klongpy import KlongInterpreter
from klongpy.core import KGSym, KGCall
from klongpy.sys_fn_ipc import (
    KGRemoteFnRef,
    KGRemoteFnCall,
    KGRemoteDictSetCall,
    KGRemoteDictGetCall,
    KGRemoteFnProxy,
    NetworkClientDictHandle,
    TcpServerHandler,
    create_system_functions_ipc,
    create_system_var_ipc,
    eval_sys_fn_create_client,
    eval_sys_fn_create_dict_client,
    eval_sys_fn_shutdown_client,
)


class TestKGRemoteFnRef(unittest.TestCase):
    def test_str_monad(self):
        ref = KGRemoteFnRef(1)
        result = str(ref)
        self.assertIn("monad", result.lower())

    def test_str_dyad(self):
        ref = KGRemoteFnRef(2)
        result = str(ref)
        self.assertIn("dyad", result.lower())


class TestKGRemoteFnCall(unittest.TestCase):
    def test_initialization(self):
        sym = KGSym("test")
        call = KGRemoteFnCall(sym, [1, 2, 3])
        self.assertEqual(call.sym, sym)
        self.assertEqual(call.params, [1, 2, 3])


class TestKGRemoteDictSetCall(unittest.TestCase):
    def test_initialization(self):
        call = KGRemoteDictSetCall("key", "value")
        self.assertEqual(call.key, "key")
        self.assertEqual(call.value, "value")


class TestKGRemoteDictGetCall(unittest.TestCase):
    def test_initialization(self):
        call = KGRemoteDictGetCall("key")
        self.assertEqual(call.key, "key")


class TestKGRemoteFnProxy(LoopsBase, unittest.TestCase):
    def test_str(self):
        klong = KlongInterpreter()
        mock_nc = MagicMock()
        mock_nc.__str__ = MagicMock(return_value="mock_nc")
        sym = KGSym("testfn")
        proxy = KGRemoteFnProxy(mock_nc, sym, 1)
        result = str(proxy)
        self.assertIn("testfn", result)
        self.assertIn("mock_nc", result)


class TestNetworkClientDictHandle(LoopsBase, unittest.TestCase):
    def test_getitem(self):
        mock_nc = MagicMock()
        mock_nc.call.return_value = "value"
        handle = NetworkClientDictHandle(mock_nc)
        result = handle["key"]
        self.assertEqual(result, "value")

    def test_setitem(self):
        mock_nc = MagicMock()
        handle = NetworkClientDictHandle(mock_nc)
        handle["key"] = "value"
        mock_nc.call.assert_called_once()

    def test_contains_raises(self):
        mock_nc = MagicMock()
        handle = NetworkClientDictHandle(mock_nc)
        with self.assertRaises(NotImplementedError):
            _ = "key" in handle

    def test_close(self):
        mock_nc = MagicMock()
        handle = NetworkClientDictHandle(mock_nc)
        handle.close()
        mock_nc.close.assert_called_once()

    def test_is_open(self):
        mock_nc = MagicMock()
        mock_nc.is_open.return_value = True
        handle = NetworkClientDictHandle(mock_nc)
        self.assertTrue(handle.is_open())

    def test_str(self):
        mock_nc = MagicMock()
        mock_nc.conn_provider.__str__ = MagicMock(return_value="test_provider")
        handle = NetworkClientDictHandle(mock_nc)
        result = str(handle)
        self.assertIn("dict", result)


class TestTcpServerHandler(unittest.TestCase):
    def test_shutdown_server_not_started(self):
        handler = TcpServerHandler()
        result = handler.shutdown_server()
        self.assertEqual(result, 0)


class TestSystemFunctions(unittest.TestCase):
    def test_create_system_functions_ipc(self):
        registry = create_system_functions_ipc()
        self.assertIsInstance(registry, dict)
        self.assertIn(".cli", registry)
        self.assertIn(".clid", registry)
        self.assertIn(".clic", registry)
        self.assertIn(".srv", registry)
        self.assertIn(".async", registry)

    def test_create_system_var_ipc(self):
        registry = create_system_var_ipc()
        self.assertIsInstance(registry, dict)
        self.assertIn(".srv.o", registry)
        self.assertIn(".srv.c", registry)
        self.assertIn(".srv.e", registry)


class TestEvalShutdownClient(unittest.TestCase):
    def test_shutdown_with_kgcall(self):
        mock_nc = MagicMock()
        mock_nc.is_open.return_value = False
        call = KGCall(mock_nc, [], 1)
        result = eval_sys_fn_shutdown_client(call)
        self.assertEqual(result, 0)

    def test_shutdown_closed_client(self):
        mock_nc = MagicMock()
        mock_nc.is_open.return_value = False
        result = eval_sys_fn_shutdown_client(mock_nc)
        self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()
