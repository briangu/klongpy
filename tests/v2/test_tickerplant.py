"""Crash the live and historical servers on both sides of an end-of-day handoff."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import json
import os
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from examples.stocks.tickerplant.sweep_head import render
from examples.stocks.tickerplant.sweep_journal import records, replay
from tests.ipc_helpers import free_port
from .peer import connect
from .runtime import ROOT, PROGRAMS


EXAMPLE = ROOT / "examples/stocks/tickerplant"


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


@pytest.mark.skipif(os.name != "posix", reason="the journal fixture uses POSIX file and directory sync")
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
            proc = subprocess.Popen([sys.executable, "-m", "examples.stocks.tickerplant.sweep_journal", "serve",
                                     "--port", str(journal), "--directory", str(directory),
                                     "--password-file", str(password)], cwd=ROOT)
            try:
                deadline = time.monotonic() + 10
                while not directory.is_dir() and time.monotonic() < deadline:
                    time.sleep(.02)
                # A TCP connection checks the listener without sending a journal request.
                while time.monotonic() < deadline:
                    try:
                        with socket.create_connection(("127.0.0.1", journal), timeout=.1):
                            break
                    except OSError:
                        time.sleep(.02)
                with running(kg, str(source), head, tmp_path, archive), connect(head) as client:
                    if not restart:
                        assert client.call("publish", 0, *([payload] * 4)) == 17
                    else:
                        assert replay(directory, f"127.0.0.1:{head}", archive) == (0, 17)
                        result = client.call("ticksFrom", 0, 17)
                        np.testing.assert_array_equal(result, np.column_stack([payload] * 5))
                        assert client.call("publish", 17, *([payload] * 4)) == client.call("roll", 0) == 34
            finally:
                proc.kill()
                proc.wait(timeout=5)
    path = directory / "day-0.wal"
    assert [seq for seq, _ in records(path, 0)] == [0, 17]
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="damaged"):
        list(records(path, 0))
