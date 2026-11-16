"""Tables over the kdb+ IPC protocol (type 98)."""
import unittest

import numpy as np

from klongpy.ipc import codec
from klongpy.ipc.codec import KdbTable, decode, encode

try:
    import pandas  # noqa: F401
    import duckdb  # noqa: F401
    HAVE_DB = True
except ImportError:
    HAVE_DB = False

# ([] a:1 2 3; b:1.5 2.5 3.5) as q sends it, without the 8-byte header
Q_TABLE = bytes.fromhex(
    "62" "00" "63"
    "0b00" "02000000" "6100" "6200"
    "0000" "02000000"
    "0700" "03000000" "0100000000000000" "0200000000000000" "0300000000000000"
    "0900" "03000000" "000000000000f83f" "0000000000000440" "0000000000000c40")


class TestTableCodec(unittest.TestCase):
    def setUp(self):
        self.previous = codec._table_factory
        codec.set_table_factory(None)
        self.addCleanup(codec.set_table_factory, self.previous)

    def test_decode_q_table_without_klongpy_db(self):
        t = decode(Q_TABLE)
        self.assertIsInstance(t, KdbTable)
        self.assertEqual([str(k) for k in t], ["a", "b"])
        cols = list(t.values())
        np.testing.assert_array_equal(cols[0], [1, 2, 3])
        np.testing.assert_array_equal(cols[1], [1.5, 2.5, 3.5])

    def test_table_bytes_round_trip_exactly(self):
        self.assertEqual(bytes(encode(decode(Q_TABLE))), Q_TABLE)

    def test_ragged_columns_are_refused(self):
        t = KdbTable(a=np.array([1, 2]), b=np.array([1.0]))
        with self.assertRaises(Exception) as cm:
            encode(t)
        self.assertIn("differ in length", str(cm.exception))


@unittest.skipUnless(HAVE_DB, "klongpy.db needs pandas and duckdb")
class TestKlongTable(unittest.TestCase):
    def setUp(self):
        from klongpy.db import sys_fn_db
        self.db = sys_fn_db
        self.previous = codec._table_factory
        codec.set_table_factory(sys_fn_db._table_from_wire)
        self.addCleanup(codec.set_table_factory, self.previous)

    def test_table_encodes_as_q_table(self):
        import pandas as pd
        t = self.db.Table(pd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]}))
        self.assertEqual(bytes(encode(t)), Q_TABLE)

    def test_q_table_arrives_as_table(self):
        t = decode(Q_TABLE)
        self.assertIsInstance(t, self.db.Table)
        self.assertEqual(t.columns, ["a", "b"])
        np.testing.assert_array_equal(t.get("a"), [1, 2, 3])

    def test_string_column_and_pending_inserts(self):
        import pandas as pd
        t = self.db.Table(pd.DataFrame({"s": ["x", "yy"], "n": [1, 2]}))
        t.insert(np.array(["zzz", 3], dtype=object))
        r = decode(encode(t))
        self.assertEqual(len(r), 3)
        self.assertEqual(list(r.get("s")), ["x", "yy", "zzz"])

    def test_empty_table(self):
        import pandas as pd
        r = decode(encode(self.db.Table(pd.DataFrame({"a": np.array([], dtype="int64")}))))
        self.assertEqual(r.columns, ["a"])
        self.assertEqual(len(r), 0)



@unittest.skipUnless(HAVE_DB, "klongpy.db needs pandas and duckdb")
class TestTableBetweenInterpreters(unittest.TestCase):
    """The examples/db/dfs.kg flow: write a table to a server and read it back."""

    def test_write_and_read(self):
        from tests.test_ipc_endpoint import IPCCase

        class Case(IPCCase):
            def runTest(self):
                pass

        case = Case()
        try:
            server, client = case.pair()
            case.kg(server, '.py("klongpy.db");tbs:::{};read::{tbs?x};write::{tbs,x,,y;1}')
            case.kg(client, '.py("klongpy.db");T::.table(,"a",,[1 2 3])')
            self.assertEqual(case.kg(client, 'c(:write,"T",T)'), 1)
            self.assertEqual(case.kg(server, '#tbs?"T"'), 3)
            r = case.kg(client, 'R::c(:read,,"T")')
            self.assertEqual(type(r).__name__, "Table")
            np.testing.assert_array_equal(r.get("a"), [1, 2, 3])
        finally:
            case.doCleanups()
