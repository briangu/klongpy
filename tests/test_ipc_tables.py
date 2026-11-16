"""Tables over the kdb+ IPC protocol (type 98)."""
import struct
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

    def test_datetime_column_is_a_timestamp_column(self):
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2, 3], "t": pd.to_datetime(["2000-01-01", "2024-01-15 10:30:00", None], format="ISO8601")})
        want = (b"\x62\x00\x63" + b"\x0b\x00" + struct.pack("<i", 2) + b"a\x00t\x00"
                + b"\x00\x00" + struct.pack("<i", 2)
                + b"\x07\x00" + struct.pack("<i", 3) + struct.pack("<qqq", 1, 2, 3)
                + b"\x0c\x00" + struct.pack("<i", 3) + struct.pack("<qqq", 0, 758629800000000000, -2 ** 63))
        self.assertEqual(bytes(encode(self.db.Table(df))), want)

    def test_every_pandas_datetime_unit_and_timedelta(self):
        import pandas as pd
        for unit in ("s", "ms", "us", "ns"):
            with self.subTest(unit=unit):
                t = pd.to_datetime(["2000-01-02", "NaT"]).astype("datetime64[%s]" % unit)
                df = pd.DataFrame({"t": t, "d": pd.to_timedelta([1, 2], unit="s").astype("timedelta64[%s]" % unit)})
                r = decode(encode(self.db.Table(df)))
                self.assertEqual(r.get("t").tolist(), [86400 * 10 ** 9, -2 ** 63])
                self.assertEqual(r.get("d").tolist(), [10 ** 9, 2 * 10 ** 9])

    def test_timestamp_column_round_trip(self):
        """Out as a timestamp column; back as the kdb+ nanosecond integers (design section 4: temporals decode
        to the underlying integer), which are exactly what was sent."""
        import pandas as pd
        sent = pd.to_datetime(["2000-01-01", "2024-01-15 10:30:00.123456789", "1999-12-31 23:59:59"], format="ISO8601")
        back = decode(encode(self.db.Table(pd.DataFrame({"t": sent})))).get("t")
        self.assertEqual(np.asarray(back).dtype, np.int64)
        recovered = pd.to_datetime(np.asarray(back)) + (pd.Timestamp("2000-01-01") - pd.Timestamp(0))
        self.assertTrue((recovered == sent).all())

    def test_timezone_aware_column_is_sent_as_utc(self):
        import pandas as pd
        t = pd.to_datetime(["2000-01-01 01:00:00"]).tz_localize("Europe/Paris")        # 00:00:00 UTC
        r = decode(encode(self.db.Table(pd.DataFrame({"t": t}))))
        self.assertEqual(np.asarray(r.get("t")).tolist(), [0])

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
