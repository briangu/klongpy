import unittest
from types import SimpleNamespace

from klongpy import KlongInterpreter
from klongpy.core import KGSym
import klongpy.sys_fn_kdb as kdb


class DummyQConnection:
    def __init__(self, host='localhost', port=5000, username=None, password=None):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.connected = False
        self.queries = []

    def open(self):
        self.connected = True

    def __call__(self, qexp):
        self.queries.append(qexp)
        return f"EXEC:{qexp}"

    def close(self):
        self.connected = False

    def is_connected(self):
        return self.connected


class TestKdbIPC(unittest.TestCase):
    def setUp(self):
        self.patcher = unittest.mock.patch(
            'klongpy.sys_fn_kdb.qconnection',
            SimpleNamespace(QConnection=DummyQConnection, QFunction=type('QFunction', (), {}))
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_qcli_and_qclic(self):
        klong = KlongInterpreter()
        klong('c::.qcli(1234)')
        proxy = klong('c')
        self.assertTrue(proxy.connection.is_open())
        r = klong('c("1+1")')
        self.assertEqual(r, 'EXEC: 1+1')
        self.assertEqual(proxy.connection.conn.queries[-1], ' 1+1')
        klong('.qclic(c)')
        self.assertFalse(proxy.connection.is_open())

    def test_qclid_set_get(self):
        klong = KlongInterpreter()
        klong('d::.qclid(4321)')
        d = klong('d')
        self.assertTrue(d.is_open())
        d.set(KGSym('a'), 42)
        self.assertEqual(d.connection.conn.queries[-1], 'a::42')
        r = d.get('x+y')
        self.assertEqual(r, 'EXEC:x+y')
        self.assertEqual(d.connection.conn.queries[-1], 'x+y')
        klong('.qclic(d)')
        self.assertFalse(d.is_open())

