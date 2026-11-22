"""Crash the live and historical servers on both sides of an end-of-day handoff."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import subprocess
import threading
import time

import numpy as np
import pytest

from tests.ipc_helpers import free_port
from .peer import connect
from .runtime import ROOT, PROGRAMS


EXAMPLE = ROOT / "examples/stocks/tickerplant"


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
