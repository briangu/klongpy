from importlib import import_module
from importlib.util import find_spec
import struct

import numpy as np
import pytest

from klongpy import KlongInterpreter
from klongpy.ipc import encode
from klongpy.core import KGSym
from .peer import connect, server
from .runtime import PROGRAMS, load, required


pandas = import_module("pandas") if find_spec("pandas") else None
arrow = import_module("pyarrow") if find_spec("pyarrow") else None


def test_dataframe_to_klong_and_back_to_numpy():
    if pandas is None:
        required("Install the pandas database extra for the DataFrame contract")
    kg = KlongInterpreter(backend="numpy")
    kg["frame"] = pandas.DataFrame({"volume": [2**53 + 3, 2**53 + 1]})
    kg('t::.table(frame);d::.db(:{;"ticks";t})')
    result = np.asarray(kg('d("SELECT volume FROM ticks ORDER BY volume")'))
    assert result.dtype.kind in "iu"
    np.testing.assert_array_equal(result, [2**53 + 1, 2**53 + 3])


def test_dataframe_round_trip_preserves_nanoseconds_and_nulls():
    if pandas is None:
        required("Install pandas to test the DataFrame round trip")
    kg = KlongInterpreter(backend="numpy")
    frame = pandas.DataFrame({"ts": pandas.to_datetime([946684800000000001, None], unit="ns"),
                              "px": pandas.array([2**53 + 1, None], dtype="Int64")})
    kg["frame"] = frame
    kg("t::.table(frame)")
    pandas.testing.assert_frame_equal(kg["t"].get_dataframe(), frame)


def test_arrow_column_interchange():
    if arrow is None:
        required("Install pyarrow for the Arrow table interchange test")
    kg = KlongInterpreter(backend="numpy")
    source = arrow.table({"ts": arrow.array([1, None], type=arrow.timestamp("ns")),
                          "px": arrow.array([2**53 + 1, None], type=arrow.int64())})
    kg["source"] = source
    kg("t::.table(source)")
    assert arrow.table(kg["t"]).equals(source)


# Little-endian q object bodies from -8! applied to t and 0#0!t in the live q test below.
# Holding the bytes outside our codec detects loss of type information in either host.
@pytest.mark.parametrize("body_hex", [
    "636200630b000100000073796d000000010000000b0002000000414100424200"
    "6200630b00020000007473007078000000020000000c00020000000100000000000000"
    "000000000000008007000200000001000000000020000000000000000080",
    "6200630b000300000073796d007473007078000000030000000b00000000000c0000000000070000000000",
], ids=["keyed-temporal-with-null", "typed-empty-table"])
@pytest.mark.parametrize("kind", ["kg", "kgpy"])
def test_wire_table_identity_without_reference_process(kind, request, tmp_path, body_hex):
    body = bytes.fromhex(body_hex)
    with server(request.getfixturevalue(kind), kind, tmp_path) as (port, _), connect(port) as peer:
        assert peer.request("bounce::{x};1") == 1
        # Synchronous call with two items: the function symbol and an already serialized table.
        request_body = b"\x00\x00\x02\x00\x00\x00" + encode(KGSym("bounce")) + body
        peer.sock.sendall(peer.session.frame(1, request_body))
        with peer.sock.makefile("rb") as response:
            header = response.read(8)
            assert len(header) == 8
            assert header[:3] == bytes([1, 2, 0]), header.hex()
            size, = struct.unpack_from("<I", header, 4)
            assert 8 < size < 4096
            assert response.read(size - 8) == body


@pytest.mark.v2_reference
def test_market_oracle_in_q(q, tmp_path):
    with server(q, "q", tmp_path) as (port, _), connect(port) as peer:
        for statement in (PROGRAMS / "market.q").read_text().splitlines():
            if statement and not statement.startswith("/"):
                peer.request(statement + ";1")
        for fn, expected in [("lastpx", [103, 201, 300]), ("minutes", [[0, 7], [1, 10]]),
                             ("bids", [98, -1, 101, -1, 101, 198]), ("unmatched", 2),
                             ("bars", [[200, 201, 100, 201], [102, 300, 102, 300]]),
                             ("bbvwap", 200.4), ("rollingqty", [2, 3, 6, 1, 9, 5])]:
            np.testing.assert_array_equal(peer.request(f"{fn}[]"), expected)


@pytest.mark.parametrize("sender,receiver", [("kg", "kgpy"), ("kgpy", "kg")])
def test_table_upsert_across_processes(request, sender, receiver, tmp_path):
    with server(request.getfixturevalue(receiver), receiver, tmp_path / receiver) as (port, _):
        with connect(port) as peer:
            peer.request('change(t)::{.insert(t;[2 7 99]);t};1')
        with server(request.getfixturevalue(sender), sender, tmp_path / sender) as (origin, _):
            with connect(origin) as client:
                client.request(f"{load('tables.kg')};1")
                client.request(f'.index(ledger;["account" "lot"]);c::.cli("127.0.0.1:{port}");1')
                client.request('back::c(:change,,ledger);1')
                assert client.request('.schema(back)~["account" "lot" "qty"]') == 1
                client.request('.insert(back;[1 7 55]);d::.db(:{;"t";back});1')
                rows = client.request('d("SELECT * FROM t ORDER BY account")')
                np.testing.assert_array_equal(rows, [[1, 7, 55], [2, 7, 99]])


@pytest.mark.parametrize("sender,receiver", [
    pytest.param("kg", "kg", id="kg-to-kg"),
    pytest.param("kg", "kgpy", id="kg-to-kgpy"),
    pytest.param("kgpy", "kg", id="kgpy-to-kg"),
    pytest.param("kgpy", "kgpy", id="kgpy-to-kgpy"),
])
def test_remote_error_preserves_connection_and_server_state(request, sender, receiver, tmp_path):
    with server(request.getfixturevalue(receiver), receiver, tmp_path / "receiver") as (port, _):
        with connect(port) as setup:
            assert setup.request(
                "requests::[];opened::0;.srv.o::{opened::opened+1;x};"
                "record::{:[x<0;missingAuditHandler();requests::requests,,x];+/requests};1"
            ) == 1

        with server(request.getfixturevalue(sender), sender, tmp_path / "sender") as (origin, _):
            with connect(origin) as client:
                assert client.request(f'c::.cli("127.0.0.1:{port}");1') == 1
                assert client.request("c(:record,,11)") == 11
                with pytest.raises(AssertionError, match="missingAuditHandler"):
                    client.request("c(:record,,-1)")
                assert client.request("c(:record,,7)") == 18
                # Count opens to check that recovery used the original connection.
                assert client.request('c("opened")') == 1

                with connect(port) as observer:
                    assert observer.request("opened") == 2
                    np.testing.assert_array_equal(observer.request("requests"), [11, 7])


@pytest.mark.parametrize("kind", ["kg", "kgpy"])
@pytest.mark.v2_reference
@pytest.mark.parametrize("qvalue", [
    pytest.param("t", id="keyed-nanoseconds-and-long"),
    pytest.param("0#t", id="empty-keyed"),
    pytest.param("0!t", id="unkeyed"),
    pytest.param("0#0!t", id="empty-unkeyed"),
    pytest.param("([sym:`AA`AA;lot:1 2] qty:0N 7)", id="composite-key"),
    pytest.param("([]b:01b;h:1 0Nh;i:1 0Ni;e:1 0Ne)", id="numeric-widths"),
    pytest.param("([]d:2000.01.01 0Nd;n:0D00:00:00.000000001 0Nn)", id="date-and-timespan"),
    pytest.param("([]s:`AA`;txt:(\"aa\";\"\"))", id="symbols-and-strings"),
])
def test_q_typed_table_round_trip(q, kind, request, tmp_path, qvalue):
    # Ask q itself to compare keyed values, including the types of empty columns.
    # The Python IPC codec never decodes the table under test.
    with server(request.getfixturevalue(kind), kind, tmp_path / kind) as (port, _):
        with connect(port) as klong:
            klong.request("bounce::{x};1")
        with server(q, "q", tmp_path / "q") as (qport, _):
            with connect(qport) as peer:
                peer.request(f'h:hopen `:127.0.0.1:{port};1')
                peer.request('t:([sym:`AA`BB] ts:2000.01.01D00:00:00.000000001 0Np;px:9007199254740993 0N);1')
                assert peer.request(f"{{(-8!x)~-8!h(`bounce;x)}} {qvalue}") == 1, qvalue


@pytest.mark.parametrize("case", ["empty-schema", "multiple-chunks", "null-versus-nan", "sliced-offset"])
def test_arrow_layouts(case):
    if arrow is None:
        required("Install pyarrow for Arrow table coverage")
    if case == "empty-schema":
        source = arrow.table({"px": arrow.array([], type=arrow.float64()),
                              "ts": arrow.array([], type=arrow.timestamp("ns"))})
    elif case == "multiple-chunks":
        source = arrow.table({"q": arrow.chunked_array([[7, None], [], [11, 2**53 + 7]], type=arrow.int64())})
    elif case == "sliced-offset":
        source = arrow.table({"s": ["ignore", None, "BB", "CC"], "q": [13, 17, 19, 23]}).slice(1, 2)
    else:
        source = arrow.table({"q": arrow.array([None, float("nan"), -1.5], type=arrow.float64())})
    kg = KlongInterpreter(backend="numpy")
    kg["inputtable"] = source
    kg("t::.table(inputtable)")
    result = arrow.table(kg["t"])
    assert result.schema == source.schema
    if case == "null-versus-nan":
        col = result.column("q")
        assert col.is_null().to_pylist() == [True, False, False]
        assert np.isnan(col[1].as_py()) and col[2].as_py() == -1.5
    else:
        assert result.to_pylist() == source.to_pylist()
