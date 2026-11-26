"""Crash the live and historical servers on both sides of an end-of-day handoff."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import json
import socket
import subprocess
import struct
import sys
import threading
import time
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

from examples.stocks.tickerplant.native_bench import NativeClients, native_session
from examples.stocks.tickerplant.profile_report import profile, subscriber_delivery
from examples.stocks.tickerplant.sweep_head import profiled_head, profiled_subscriber, render
from examples.stocks.tickerplant.journal import records
from tests.ipc_helpers import free_port
from .peer import connect
from .runtime import ROOT, PROGRAMS


EXAMPLE = ROOT / "examples/stocks/tickerplant"


@contextmanager
def journal_server(port, directory, *args):
    command = [sys.executable, "-m", "examples.stocks.tickerplant.journal", "serve",
               "--port", str(port), "--directory", str(directory), *args]
    with (directory.parent / "journal.log").open("ab") as log:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 10
            while proc.poll() is None and time.monotonic() < deadline:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.1):
                        break
                except OSError:
                    time.sleep(.025)
            else:
                pytest.fail((directory.parent / "journal.log").read_text())
            yield proc
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)


def test_native_ticker_wal(kg, tmp_path):
    subprocess.run([*kg, str(PROGRAMS / "tickerplant_native_wal.kg")], cwd=tmp_path, check=True, timeout=15)


def test_tickerplant_duckdb_worker_budget(kg, tmp_path):
    expression = (
        f'.l("{(EXAMPLE / "head.kg").as_posix()}");startHead("");'
        f'.l("{(EXAMPLE / "hdb.kg").as_posix()}");'
        f'.l("{(PROGRAMS / "tickerplant_workers.kg").as_posix()}")'
    )
    subprocess.run([*kg, "-e", expression], cwd=tmp_path, check=True, timeout=20)


@contextmanager
def running(kg, script, port, directory, *args):
    address = f"127.0.0.1:{port}"
    with (directory / (script + ".log")).open("ab") as log:
        proc = subprocess.Popen([*kg, str(EXAMPLE / script), address, *args], cwd=directory,
                                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 10
            while proc.poll() is None and time.monotonic() < deadline:
                try:
                    with connect(port) as client:
                        assert client.request("1+1") == 2
                    break
                except ConnectionRefusedError:
                    time.sleep(0.025)
            else:
                pytest.fail((directory / (script + ".log")).read_text())
            yield proc
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)
            proc.stdin.close()


def test_tickerplant_ack_restart_and_roll(kg, tmp_path):
    (tmp_path / "transfer").mkdir()
    head, hdb = free_port(), free_port()
    while head == hdb:
        hdb = free_port()
    peer = f"127.0.0.1:{hdb}"
    with running(kg, "head.kg", head, tmp_path, peer), connect(head) as writer:
        writer.request(f'.l("{PROGRAMS / "tickerplant.kg"}");1')
        with pytest.raises(AssertionError):
            writer.call("roll", 0)
    with ExitStack() as hosts:
        hosts.enter_context(running(kg, "head.kg", head, tmp_path, peer))
        with connect(head) as writer:
            assert writer.request("[;dayNo;closing;nextSeq]").tolist() == [0, 1, 4]
            with pytest.raises(AssertionError, match="pending rollover"):
                writer.call("publish", 4, np.array([104]), np.array([4]), np.array([998]), np.array([1]))
            # Recreate a loss of the reply after the HDB commit and kill the HDB.
            with running(kg, "hdb.kg", hdb, tmp_path), connect(hdb) as hist:
                assert hist.call("seal", 0, 0, 4) == 4
            hosts.enter_context(running(kg, "hdb.kg", hdb, tmp_path))
            assert writer.call("roll", 0) == writer.call("roll", 0) == 4
            np.testing.assert_array_equal(writer.request("liveStats()"), [0, 0, 0, 0])
            assert writer.call("publish", 4, np.array([104]), np.array([4]), np.array([998]), np.array([1])) == 5
            assert writer.call("roll", 1) == 5
        with connect(hdb) as hist:
            np.testing.assert_array_equal(hist.call("historyStats", 0), [4, 6, 3586, 7])
            np.testing.assert_array_equal(hist.call("historyStats", 1), [1, 4, 998, 1])
            np.testing.assert_array_equal(hist.call("historyQuote", 0, 4), [2, 995])
            np.testing.assert_array_equal(hist.request('history("SELECT * FROM trades ORDER BY seq")'),
                [[0, 0, 100, 4, 990, 2], [0, 1, 102, 7, 800, 1], [0, 2, 101, 4, 995, 3],
                 [0, 3, 103, 7, 801, 1], [1, 4, 104, 4, 998, 1]])


def test_tickerplant_subscriber_and_replay(kg, tmp_path):
    (tmp_path / "transfer").mkdir()
    ports = set()
    while len(ports) < 3:
        ports.add(free_port())
    head, hdb, sub = ports
    feed = f"127.0.0.1:{head}"
    with running(kg, "hdb.kg", hdb, tmp_path), running(kg, "head.kg", head, tmp_path, f"127.0.0.1:{hdb}"), connect(head) as writer:
        with running(kg, "subscriber.kg", sub, tmp_path, feed), connect(sub) as reader:
            assert writer.call("publish", 0, [10, 11], [1, 2], [101, 202], [3, 4]) == 2
            np.testing.assert_array_equal(reader.request("subStats()"), [2, 2, 7])
        assert writer.call("publish", 2, [12], [2], [204], [5]) == 3
        np.testing.assert_array_equal(writer.call("latest", 1), [0, 101])
        np.testing.assert_array_equal(writer.call("ticksFrom", 1, 8), [[1, 11, 2, 202, 4], [2, 12, 2, 204, 5]])
        assert writer.call("roll", 0) == 3
        with connect(hdb) as hist:
            np.testing.assert_array_equal(hist.call("historyFrom", 0, 1, 8), [[1, 11, 2, 202, 4], [2, 12, 2, 204, 5]])


def test_tickerplant_variable_batches_and_atomic_chunks(kg, tmp_path):
    port = free_port()
    with running(kg, "head.kg", port, tmp_path, "127.0.0.1:1"), connect(port) as writer:
        assert writer.request(f'.l("{PROGRAMS / "tickerplant_batches.kg"}");1') == 1


def test_tickerplant_two_publishers_claim_the_same_sequence(kg, tmp_path):
    port = free_port()
    ready = threading.Barrier(2)

    def submit(price):
        with connect(port) as writer:
            ready.wait(timeout=5)
            try:
                return writer.call("publish", 0, [10], [7], [price], [2])
            except AssertionError as error:
                assert "unexpected first sequence" in str(error)
                return 0

    with running(kg, "head.kg", port, tmp_path, "127.0.0.1:1"), ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(submit, [100, 101])) == [0, 1]
    with running(kg, "head.kg", port, tmp_path, "127.0.0.1:1"), connect(port) as writer:
        assert writer.request("nextSeq") == 1
        row = writer.call("ticksFrom", 0, 8).tolist()
        assert row in ([0, 10, 7, 100, 2], [0, 10, 7, 101, 2])


@pytest.mark.parametrize("rate,seconds", [(10000, .15), (0, 1.05)])
def test_tickerplant_pipelined_publisher(kg, tmp_path, rate, seconds):
    (tmp_path / "transfer").mkdir()
    ports = set()
    while len(ports) < 3:
        ports.add(free_port())
    head, hdb, subscriber = [f"127.0.0.1:{p}" for p in ports]
    with ExitStack() as hosts:
        for script, addr, peers in [("hdb.kg", hdb, []), ("head.kg", head, [hdb]),
                                    ("subscriber.kg", subscriber, [head])]:
            hosts.enter_context(running(kg, script, int(addr.split(":")[1]), tmp_path, *peers))
        result = subprocess.run([sys.executable, "-m", "examples.stocks.tickerplant.bench",
                                 "--head", head, "--hdb", hdb, "--subscriber", subscriber,
                                 "--in-flight", "4", "--batch-cycle", "64", "128", "--rate", str(rate),
                                 "--seconds", str(seconds), "--sessions", "2"], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        warm, measured = map(json.loads, result.stdout.splitlines())
        assert measured["first"] == warm["end"]
        assert measured["verified_rows"] == measured["end"] - measured["first"] >= max(64, rate * seconds)
        assert measured["in_flight"] == 4 and measured["subscribers"] == 1
        assert measured["live"]["round_trip"]["samples"] and measured["history"]["round_trip"]["samples"]
        checkpoints = measured["acknowledged_each_second"]
        assert len(checkpoints) == int(seconds)
        assert all(0 < n <= measured["acknowledged_by_deadline"] for n in checkpoints)


def test_tickerplant_wide_memory_fixture(kg, tmp_path):
    source = tmp_path / "sweep-head.kg"
    source.write_text(render((EXAMPLE / "head.kg").read_text(), 131072, True))
    expression = f'.l("{source.as_posix()}");startHead("");.l("{(PROGRAMS / "tickerplant_sweep.kg").as_posix()}")'
    subprocess.run([*kg, "-e", expression], cwd=tmp_path, check=True, timeout=20)


def test_tickerplant_sweep_journal_replay(kg, tmp_path):
    (tmp_path / "transfer").mkdir()
    ports = set()
    while len(ports) < 3:
        ports.add(free_port())
    head, hdb, journal = ports
    archive = f"127.0.0.1:{hdb}"
    directory = tmp_path / "log"
    password = tmp_path / "password"
    password.write_text("ReplayTest")
    source = tmp_path / "head-sweep.kg"
    source.write_text(render((EXAMPLE / "head.kg").read_text(), 65536, True, str(journal), password.read_text()))
    payload = np.arange(17)
    with running(kg, "hdb.kg", hdb, tmp_path):
        for restart in (False, True):
            with journal_server(journal, directory, "--password-file", str(password)):
                with running(kg, str(source), head, tmp_path, archive), connect(head) as client:
                    if not restart:
                        assert client.call("publish", 0, *([payload] * 4)) == 17
                    else:
                        assert client.request("nextSeq") == 17
                        result = client.call("ticksFrom", 0, 17)
                        np.testing.assert_array_equal(result, np.column_stack([payload] * 5))
                        assert client.call("publish", 17, *([payload] * 4)) == client.call("roll", 0) == 34
    path = directory / "day-0.wal"
    assert [seq for seq, _ in records(path, 0)] == [0, 17]
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="damaged"):
        list(records(path, 0))


def test_replay_after_wal_sync_before_publish_reply(kg, tmp_path):
    port = free_port()
    fault = tmp_path / "crash-head.kg"
    source = (EXAMPLE / "head.kg").read_text()
    commit = 'live("COMMIT");1}'
    assert source.count(commit) == 1
    fault.write_text(source.replace(commit, '.x(1);' + commit))
    values = np.array([11, 22, 33])
    with running(kg, str(fault), port, tmp_path, "127.0.0.1:1") as proc, connect(port) as writer:
        with pytest.raises((AssertionError, ConnectionError)):
            writer.call("publish", 0, *([values] * 4))
        assert proc.wait(timeout=5) != 0
    path = tmp_path / "log/day-0.wal"
    committed = path.read_bytes()
    for _ in range(2):
        with running(kg, "head.kg", port, tmp_path, "127.0.0.1:1"), connect(port) as reader:
            assert reader.request("nextSeq") == 3
            np.testing.assert_array_equal(reader.call("ticksFrom", 0, 8),
                                          np.column_stack([np.arange(3)] + [values] * 4))
            assert path.read_bytes() == committed
    assert not (tmp_path / "rdb.duckdb").exists()


def test_replay_open_day_after_roll_with_torn_tail(kg, tmp_path, unused_tcp_port_factory):
    (tmp_path / "transfer").mkdir()
    head, hdb = (unused_tcp_port_factory() for _ in range(2))
    log = tmp_path / "log"
    addresses = [f"127.0.0.1:{hdb}", str(log)]
    with running(kg, "hdb.kg", hdb, tmp_path), \
            running(kg, "head.kg", head, tmp_path, *addresses), connect(head) as writer:
        assert writer.call("publish", 0, [10], [1], [101], [2]) == writer.call("roll", 0) == 1
        assert writer.call("publish", 1, [11, 12], [2, 3], [102, 103], [3, 4]) == 3
        with pytest.raises(AssertionError):
            writer.call("publish", 3, [13], [7], [104.5], [5])
        assert writer.call("publish", 3, [13], [7], [104], [5]) == 4
    active = log / "day-1.wal"
    committed = active.read_bytes()
    with active.open("ab") as stream:
        stream.write(committed[:9])
    with running(kg, "head.kg", head, tmp_path, *addresses), connect(head) as reader:
        assert reader.request("[;dayNo;dayFirst;nextSeq;closing]").tolist() == [1, 1, 4, 0]
        np.testing.assert_array_equal(reader.call("ticksFrom", 0, 8),
                                      [[1, 11, 2, 102, 3], [2, 12, 3, 103, 4], [3, 13, 7, 104, 5]])
        assert active.read_bytes() == committed
        assert reader.call("publish", 4, [14], [2], [105], [1]) == 5
    assert [seq for seq, _ in records(active, 1)] == [1, 3, 4]


@pytest.mark.parametrize(("offset", "error"), [(-1, "damaged journal record"), (20, "damaged journal header")])
def test_corrupt_wal_prevents_serving(kg, tmp_path, offset, error):
    port = free_port()
    with running(kg, "head.kg", port, tmp_path, "127.0.0.1:1"), connect(port) as writer:
        for i in range(2):
            assert writer.call("publish", i, [10], [1], [101], [2]) == i + 1
    path = tmp_path / "log/day-0.wal"
    damaged = bytearray(path.read_bytes())
    damaged[offset] ^= 1
    path.write_bytes(damaged)
    proc = subprocess.run([*kg, str(EXAMPLE / "head.kg"), f"127.0.0.1:{port}", "127.0.0.1:1"],
                          cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert proc.returncode != 0 and error in proc.stderr
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.1", port), timeout=.2)
    assert path.read_bytes() == damaged


def test_legacy_sweep_wal_resumes_at_hdb_checkpoint(kg, tmp_path):
    directory = tmp_path / "log"
    directory.mkdir()
    cols = np.array([[12, 13], [2, 2], [111, 115], [6, 7]], dtype="<i8")
    payload = cols.tobytes()
    legacy = struct.pack("<4sQQII", b"KGT1", 7, 200, 2, zlib.crc32(payload)) + payload
    path = directory / "day-7.wal"
    path.write_bytes(legacy)
    port = free_port()
    with running(kg, "head.kg", port, tmp_path, "127.0.0.1:1", str(directory), "7", "200"), connect(port) as client:
        assert client.request("[;dayNo;dayFirst;nextSeq]").tolist() == [7, 200, 202]
        np.testing.assert_array_equal(client.call("ticksFrom", 0, 8), np.column_stack([[200, 201], *cols]))
        assert client.call("publish", 202, [14], [5], [120], [2]) == 203
    assert path.read_bytes().startswith(legacy)
    assert [seq for seq, _ in records(path, 7)] == [200, 202]


def test_wal_outage_blocks_publish_until_recovery(kg, tmp_path, unused_tcp_port_factory):
    log = tmp_path / "log"
    journal, head = (unused_tcp_port_factory() for _ in range(2))
    addresses = ["127.0.0.1:1", f"127.0.0.1:{journal}"]
    with journal_server(journal, log) as proc, running(kg, "head.kg", head, tmp_path, *addresses), connect(head) as writer:
        assert writer.call("publish", 0, [10], [1], [100], [2]) == 1
        proc.kill()
        proc.wait(timeout=5)
        with pytest.raises(AssertionError):
            writer.call("publish", 1, [11], [1], [101], [3])
        with pytest.raises(AssertionError, match="restart"):
            writer.call("ticksFrom", 0, 8)
    with journal_server(journal, log), running(kg, "head.kg", head, tmp_path, *addresses), connect(head) as writer:
        assert writer.request("nextSeq") == 1
        np.testing.assert_array_equal(writer.call("ticksFrom", 0, 8), [0, 10, 1, 100, 2])
        assert writer.call("publish", 1, [11], [1], [101], [3]) == 2


def test_restart_after_hdb_commit_and_wal_advance(kg, tmp_path):
    (tmp_path / "transfer").mkdir()
    head, hdb = free_port(), free_port()
    while head == hdb:
        hdb = free_port()
    addresses = [f"127.0.0.1:{hdb}"]
    fault = tmp_path / "crash-roll.kg"
    source = (EXAMPLE / "head.kg").read_text()
    clear = 'live("DELETE FROM rdb.ticks")'
    assert source.count(clear) == 1
    fault.write_text(source.replace(clear, ".x(1);" + clear))
    with running(kg, "hdb.kg", hdb, tmp_path), connect(hdb) as hist:
        with running(kg, str(fault), head, tmp_path, *addresses) as proc, connect(head) as writer:
            assert writer.call("publish", 0, [10], [1], [101], [2]) == 1
            with pytest.raises((AssertionError, ConnectionError)):
                writer.call("roll", 0)
            assert proc.wait(timeout=5) != 0
        with running(kg, "head.kg", head, tmp_path, *addresses), connect(head) as writer:
            assert writer.request("[;dayNo;dayFirst;nextSeq;closing]").tolist() == [1, 1, 1, 0]
            assert writer.call("roll", 0) == 1
            np.testing.assert_array_equal(writer.request("liveStats()"), [0, 0, 0, 0])
            np.testing.assert_array_equal(hist.call("historyFrom", 0, 0, 8), [0, 10, 1, 101, 2])


def test_native_head_takes_over_external_journal(kg, tmp_path, unused_tcp_port_factory):
    head, hdb, journal, second = (unused_tcp_port_factory() for _ in range(4))
    directory = tmp_path / "log"
    archive = f"127.0.0.1:{hdb}"
    (tmp_path / "transfer").mkdir()
    with journal_server(journal, directory), running(kg, "head.kg", head, tmp_path, archive, f"127.0.0.1:{journal}"), connect(head) as writer:
        assert writer.call("publish", 0, [101, 105], [2, 1], [250, 195], [3, 4]) == 2
        with pytest.raises(AssertionError):
            writer.call("roll", 0)
        contender = subprocess.run([*kg, str(EXAMPLE / "head.kg"), f"127.0.0.1:{second}", archive],
                                   cwd=tmp_path, capture_output=True, text=True, timeout=10)
        assert contender.returncode != 0 and "locked" in contender.stderr
    previous = (directory / "day-0.wal").read_bytes()
    with running(kg, "head.kg", head, tmp_path, archive), connect(head) as writer:
        assert writer.request("[;dayNo;closing;nextSeq]").tolist() == [0, 1, 2]
        np.testing.assert_array_equal(writer.call("ticksFrom", 0, 8), [[0, 101, 2, 250, 3], [1, 105, 1, 195, 4]])
        with running(kg, "hdb.kg", hdb, tmp_path):
            assert writer.call("roll", 0) == 2
        assert writer.call("publish", 2, [110], [4], [251], [1]) == 3
    assert (directory / "day-0.wal").read_bytes() == previous
    [(first, cols)] = records(directory / "day-1.wal", 1)
    assert first == 2
    np.testing.assert_array_equal(cols, [[110], [4], [251], [1]])


@pytest.mark.skipif(sys.platform == "win32", reason="uses a Unix process file-size limit")
def test_native_partial_wal_write_recovers_only_committed_batch(kg, tmp_path):
    head = free_port()
    # Limit just the child and let write() return EFBIG on the remainder of the second batch.
    limited = ["/bin/sh", "-c", 'ulimit -f 128 || exit 1; trap "" XFSZ; exec "$@"', "head", *kg]
    path = tmp_path / "log/day-0.wal"
    with running(limited, "head.kg", head, tmp_path, "127.0.0.1:1"), connect(head) as client:
        assert client.call("publish", 0, [10], [1], [100], [2]) == 1
        first_record = path.read_bytes()
        batch = np.arange(10000)
        with pytest.raises(AssertionError):
            client.call("publish", 1, *([batch] * 4))
        assert path.stat().st_size > len(first_record)
        with pytest.raises(AssertionError, match="restart"):
            client.request("liveStats()")
    with running(kg, "head.kg", head, tmp_path, "127.0.0.1:1"), connect(head) as client:
        assert path.read_bytes() == first_record
        assert client.request("nextSeq") == 1
        np.testing.assert_array_equal(client.call("ticksFrom", 0, 8), [0, 10, 1, 100, 2])
        assert client.call("publish", 1, *([batch] * 4)) == 10001


def test_one_native_writer_per_directory(kg, tmp_path, unused_tcp_port_factory):
    head, other = (unused_tcp_port_factory() for _ in range(2))
    with running(kg, "head.kg", head, tmp_path, "127.0.0.1:1"), connect(head) as client:
        # A second open in this interpreter must preserve the existing OS lock.
        assert client.request('*.try({.tickerwal("log")};[])') == 0
        contender = subprocess.run([*kg, str(EXAMPLE / "head.kg"), f"127.0.0.1:{other}", "127.0.0.1:1"],
                                   cwd=tmp_path, capture_output=True, text=True, timeout=10)
        assert contender.returncode != 0 and "locked" in contender.stderr
        assert client.call("publish", 0, [110], [4], [251], [1]) == 1


def test_native_traffic_clients_and_rollover(kg, tmp_path, unused_tcp_port_factory):
    (tmp_path / "transfer").mkdir()
    head, hdb, sub1, sub2 = (unused_tcp_port_factory() for _ in range(4))
    addr = lambda p: f"127.0.0.1:{p}"
    args = SimpleNamespace(head=addr(head), hdb=addr(hdb), subscriber=[addr(sub1), addr(sub2)], query_hz=20)
    clients = NativeClients(kg[0], tmp_path, None)
    with ExitStack() as hosts:
        for script, port, peers in [("hdb.kg", hdb, []), ("head.kg", head, [addr(hdb)]),
                                    ("subscriber.kg", sub1, [addr(head)]), ("subscriber.kg", sub2, [addr(head)])]:
            hosts.enter_context(running(kg, script, port, tmp_path, *peers))
        seed = native_session(args, clients, 4000, .25, 64, 1, None)
        measured = native_session(args, clients, 50000, 1, 128, 4, (0, 0, seed["end"]))
        cmd = [sys.executable, "-m", "examples.stocks.tickerplant.native_bench", "--mode", "repeat",
               "--kg", kg[0], "--head", args.head, "--hdb", args.hdb,
               "--subscriber", args.subscriber[0], "--subscriber", args.subscriber[1],
               "--rate", "8192", "--batch", "128", "--depth", "2", "--workers", "1",
               "--confirm-seconds", "1", "--confirm-runs", "2", "--archive-probes", "2"]
        output = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True, timeout=25)
        repeats = [json.loads(line) for line in output.stdout.splitlines()]
        assert repeats[0]["result"]["archive_probes"][0]["hits"]["samples"] == 2
        assert [(row["phase"], row["result"]["day"], row["result"]["verified_rows"])
                for row in repeats] == [("repeat", 2, 8192), ("repeat", 3, 8192)]
    assert measured["verified_rows"] == 391 * 128
    assert measured["subscribers"] == 2
    assert measured["ack"]["samples"] == 391
    assert all(measured[k]["round_trip"]["samples"] for k in ("live", "history"))


def test_historical_quote_index_and_legacy_days(kg, tmp_path):
    (tmp_path / "transfer").mkdir()
    expression = f'.l("{EXAMPLE / "hdb.kg"}");.l("{PROGRAMS / "tickerplant_history.kg"}");1'
    subprocess.run([*kg, "-e", expression], cwd=tmp_path, check=True, capture_output=True)


@pytest.mark.parametrize("age", [0, 4])
def test_native_small_message_feed(kg, tmp_path, unused_tcp_port_factory, age):
    (tmp_path / "transfer").mkdir()
    head, hdb, sub = (unused_tcp_port_factory() for _ in range(3))
    addr = lambda p: f"127.0.0.1:{p}"
    args = SimpleNamespace(head=addr(head), hdb=addr(hdb), subscriber=[addr(sub)], query_hz=20,
                           feed_message_rows=8, burst_messages=4, max_age_ms=age)
    c = NativeClients(kg[0], tmp_path, None)
    with ExitStack() as servers:
        for script, port, peers in [("hdb.kg", hdb, []), ("head.kg", head, [addr(hdb)]),
                                    ("subscriber.kg", sub, [addr(head)])]:
            servers.enter_context(running(kg, script, port, tmp_path, *peers))
        session = native_session(args, c, 4096, .51, 256, 1, None)
    assert session["verified_rows"] == 2096
    assert session["feed"]["first_arrival_to_ack"]["samples"] == session["ack"]["samples"]
    assert session["feed"]["messages"] == 262
    assert session["feed"]["max_batch"] <= 256
    if age:
        assert session["feed"]["batches"] > 9
    else:
        assert session["feed"]["batches"] == 9


def test_latest_quote_tail_restart_and_roll(kg, tmp_path):
    (tmp_path / "transfer").mkdir()
    hdb, head = free_port(), free_port()
    peer = f"127.0.0.1:{hdb}"
    with running(kg, "head.kg", head, tmp_path, peer), connect(head) as writer:
        assert writer.request(f'.l("{PROGRAMS / "tickerplant_tail.kg"}");1') == 1
    with running(kg, "hdb.kg", hdb, tmp_path), running(kg, "head.kg", head, tmp_path, peer), connect(head) as writer:
        np.testing.assert_array_equal(writer.call("latest", 9), [4097, 222])
        assert writer.call("roll", 0) == 4098
        assert len(writer.call("latest", 9)) == 0

def test_native_ticker_stage_and_delivery_trace(kg, tmp_path, unused_tcp_port_factory, monkeypatch):
    (tmp_path / "transfer").mkdir()
    head, hdb, sub = (unused_tcp_port_factory() for _ in range(3))
    addr = lambda port: f"127.0.0.1:{port}"
    args = SimpleNamespace(head=addr(head), hdb=addr(hdb), subscriber=[addr(sub)], query_hz=20,
                           trace_dir=tmp_path / "traces", feed_message_rows=8, burst_messages=4, max_age_ms=4)
    monkeypatch.setenv("KG_TICKER_WAL_PROFILE", "1")
    head_script = tmp_path / "profile-head.kg"
    head_script.write_text(profiled_head((EXAMPLE / "head.kg").read_text()))
    sub_script = tmp_path / "profile-sub.kg"
    sub_script.write_text(profiled_subscriber((EXAMPLE / "subscriber.kg").read_text()))
    clients = NativeClients(kg[0], tmp_path, None)
    with ExitStack() as servers:
        for script, port, peers in [("hdb.kg", hdb, []), (str(head_script), head, [addr(hdb)]),
                                    (str(sub_script), sub, [addr(head)])]:
            servers.enter_context(running(kg, script, port, tmp_path, *peers))
        result = native_session(args, clients, 4096, .5, 256, 1, None)
    trace = json.loads((args.trace_dir / "day-0.json").read_text())
    head_events = [json.loads(s) for s in head_script.with_suffix(".kg.log").read_text().splitlines()
                   if s.startswith(('["stage"', '["wal_us"', '["parquet"'))]
    applied = [json.loads(s) for s in sub_script.with_suffix(".kg.log").read_text().splitlines() if s.startswith('["applied"')]
    report = profile(trace, head_events)
    assert report["wal"]["fdatasync"]["samples"] == result["feed"]["batches"]
    assert report["wal_bytes"] == result["verified_rows"] * 32 + 32 * result["feed"]["batches"]
    assert report["parquet"][1] == 2
    assert report["quote"]["handler"]["samples"] > 0
    [delivery] = subscriber_delivery(trace, [applied])
    assert delivery["first_arrival_to_applied"]["samples"] == result["feed"]["batches"]
