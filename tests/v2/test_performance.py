"""Measure both servers with identical data, clients, query schedules and one compute thread."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.util import find_spec
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from klongpy.core import KGSym
from klongpy import KlongInterpreter
from .peer import connect, server
from .runtime import PROGRAMS, evaluate, load, required


NAMES = [KGSym(f"S{i:02}") for i in range(16)]
BATCH = 256
CAPACITY_BATCHES = 4096
PRELOAD_BATCHES = 128
LIVE_BATCHES = 256
INTERVAL = 0.01              # 25,600 ticks/sec offered independently of service time
QUOTE_STEP = 1024
QUERY_COUNT = 200            # per query type
psutil = import_module("psutil") if sys.platform == "win32" and find_spec("psutil") else None


def test_native_vector_time_budget(kg, tmp_path, record_property):
    # Includes process startup, vector generation, reduction and output.
    times = []
    for _ in range(5):
        start = time.perf_counter()
        assert evaluate(kg, "v::!1000000;+/v", tmp_path, timeout=5) == "499999500000"
        times.append(time.perf_counter() - start)
    record_property("native_vector_median_seconds", statistics.median(times))
    assert statistics.median(times) < 1.0


def test_running_sum_temporary_heap_budget(kg, tmp_path):
    # One million int64 values occupy 8 MB. The process budget permits one such vector.
    assert evaluate(kg, load("scan_memory.kg") + ";1", tmp_path,
                    options=("--heap-limit", str(10 * 1024 * 1024))) == "1"


def test_sql_numeric_column_heap_budget(kg, tmp_path):
    result = evaluate(kg, load("sql_column_memory.kg") + ";1", tmp_path,
                      options=("--heap-limit", str(20 * 1024 * 1024)))
    assert result == "1"


def test_dense_integer_grade_heap_budget(kg, tmp_path):
    assert evaluate(kg, load("dense_grade_memory.kg") + ";1", tmp_path,
                    options=("--heap-limit", str(5 * 1024 * 1024))) == "1"


def test_group_ten_thousand_station_labels(kg, tmp_path, record_property):
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        assert evaluate(kg, load("group_labels.kg") + ";1", tmp_path) == "1"
        samples.append(time.perf_counter() - start)
    elapsed = statistics.median(samples)
    record_property("distinct_and_group_station_labels_seconds", elapsed)
    assert elapsed < 1.0


def test_composite_key_update_scaling(host, tmp_path, record_property):
    # Each call starts with a fresh table and times only the same 256-row batch.
    for rows in (2048, 8192, 32768):
        expr = f"""{load("keyed_updates.kg")};.l("nstat");nstat.M(1_{{keyedUpdateTime({rows})}}'!6)"""
        elapsed = float(evaluate(host, expr, tmp_path))
        assert math.isfinite(elapsed) and elapsed >= 0
        record_property(f"upsert_256_into_{rows}_median_seconds", elapsed)


@pytest.mark.parametrize("placement", ["same_interpreter", "other_interpreter"])
def test_hosted_scalar_with_resident_symbols(placement, record_property):
    worker = KlongInterpreter(backend="numpy")
    resident = worker if placement == "same_interpreter" else KlongInterpreter(backend="numpy")
    resident(load("resident_symbols.kg"))

    def time_scalar():
        samples = []
        for _ in range(7):
            start = time.thread_time_ns()
            for _ in range(20):
                assert worker("1+1") == 2
            samples.append((time.thread_time_ns() - start) / 20)
        return statistics.median(samples)

    assert worker("1+1") == 2
    baseline = time_scalar()
    assert resident("keepSymbols(524288)") == 524288
    with_table = time_scalar()
    assert resident('#resident@"sym"') == 524288
    resident("resident::0")
    record_property("scalar_baseline_ns", baseline)
    record_property("scalar_with_table_ns", with_table)
    # Allow 100 us of fixed overhead and a fivefold margin across build profiles.
    assert with_table <= 5 * baseline + 100_000


def test_group_resident_integer_ids(kg, tmp_path, record_property):
    # The same 64 instrument IDs recur in a non-sorted order. Time processing
    # resident data and returning the small report, with setup outside the timer.
    first_seen = [(17 * i) % 64 - 23 for i in range(64)]
    with server(kg, "kg", tmp_path) as (port, _), connect(port) as client:
        client.request(f'{load("group_ids.kg")};ids::(((17*!1000000)!64)-23);1')
        for name, expected in (("distinctIds", first_seen), ("groupCounts", [15625] * 64)):
            query = name + "(ids)"
            np.testing.assert_array_equal(client.request(query), expected)
            samples = []
            for _ in range(7):
                start = time.perf_counter()
                result = client.request(query)
                samples.append(time.perf_counter() - start)
                np.testing.assert_array_equal(result, expected)
            median = statistics.median(samples)
            record_property(name + "_median_seconds", median)
            assert median < 1.0


def test_one_python_call_per_query(record_property):
    kg = KlongInterpreter(backend="numpy")
    kg(load("market.kg"))
    query = kg["lastpx"]
    np.testing.assert_array_equal(query(), [103, 201, 300])
    samples = []
    for _ in range(100):
        start = time.perf_counter()
        result = query()
        samples.append((time.perf_counter() - start) * 1000)
        np.testing.assert_array_equal(result, [103, 201, 300])
    p99 = percentile(samples, 0.99)
    record_property("python_call_p50_ms", percentile(samples, 0.50))
    record_property("python_call_p99_ms", p99)
    assert p99 < 100


@pytest.mark.parametrize("kind", ["kg", "kgpy"])
def test_selective_history_after_reopen(kind, request, tmp_path, record_property):
    command = request.getfixturevalue(kind)
    assert evaluate(command, f'{load("history.kg")};history("SELECT count(*) FROM trades")',
                    tmp_path, timeout=30) == "1000000"
    # The writer has exited. Filesystem pages may still be in the OS cache.
    expected = sum(1000 + i % 991 for i in range(500000, 750000) if i % 64 == 17)
    with server(command, kind, tmp_path) as (port, _), connect(port) as client:
        client.request('h::.db("history.duckdb");h("SET threads=1");1')
        samples = []
        for _ in range(6):
            start = time.perf_counter()
            result = client.request('h("SELECT sum(px)::BIGINT FROM trades WHERE dayno=2 AND sym=17")')
            samples.append(time.perf_counter() - start)
            assert result == expected
    record_property("first_query_seconds", samples[0])
    record_property("warm_query_median_seconds", statistics.median(samples[1:]))
    assert max(samples) < 1, samples


def vector_energy_time(kind, command, size, directory):
    with server(command, kind, directory) as (port, _), connect(port) as client:
        if kind == "q":
            client.request(f"energy:{{sum x*x}};v:(til {size}) mod 16;1")
            query = "energy v"
        else:
            client.request(f'{load("arrays.kg")};v::(!{size})!16;1')
            query = "energy(v)"
        expected = (size // 16) * sum(i * i for i in range(16))
        assert client.request(query) == expected
        samples = []
        for _ in range(7):
            start = time.perf_counter()
            assert client.request(query) == expected
            samples.append(time.perf_counter() - start)
    return statistics.median(samples)


@pytest.mark.parametrize("size", [4096, 1000000])
def test_resident_vector_energy(kg, size, tmp_path, record_property):
    seconds = vector_energy_time("kg", kg, size, tmp_path)
    record_property("resident_energy_median_seconds", seconds)
    assert seconds < 0.1


@pytest.mark.v2_reference
@pytest.mark.parametrize("size", [4096, 1000000])
def test_compare_resident_arrays(q, kg, kgpy, size, tmp_path, record_property):
    for kind, command in (("q", q), ("kg", kg), ("kgpy", kgpy)):
        seconds = vector_energy_time(kind, command, size, tmp_path / kind)
        record_property(f"{kind}_resident_energy_median_seconds", seconds)


def batch(number):
    ts = np.arange(number * BATCH, (number + 1) * BATCH, dtype=np.int64)
    return NAMES * (BATCH // len(NAMES)), ts, ts + 1000


def expected_query(name, last_batch):
    last = (last_batch + 1) * BATCH - len(NAMES) + np.arange(len(NAMES))
    if name == "latest":
        return last + 1000
    return 2000 + last // QUOTE_STEP * QUOTE_STEP + np.arange(len(NAMES))


def percentile(samples, p):
    return sorted(samples)[math.ceil(len(samples) * p) - 1]


@dataclass
class Metrics:
    rows_per_second: float
    p50_ms: dict
    p99_ms: dict
    p99_client_delay_ms: dict
    p99_round_trip_ms: dict
    max_ingest_lag_ms: float
    observed_rss_kib: int


@dataclass
class QuerySamples:
    elapsed_ms: list
    client_delay_ms: list
    round_trip_ms: list


def rss_kib(pid):
    if sys.platform.startswith("linux"):
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
        return int(next(line for line in lines if line.startswith("VmHWM:")).split()[1])
    if sys.platform == "win32":
        if psutil is None:
            required("Install psutil for Windows process memory measurements")
        return psutil.Process(pid).memory_info().peak_wset // 1024
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True, timeout=3))


def initialize(client, kind):
    if kind == "q":
        for line in (PROGRAMS / "stream.q").read_text().splitlines():
            if line and not line.startswith("/"):
                client.request(line + ";1")
    else:
        client.request(f'{load("stream.kg")};1')
    quote_times = np.arange(0, (PRELOAD_BATCHES + LIVE_BATCHES) * BATCH, QUOTE_STEP)
    symbols = [s for s in NAMES for _ in quote_times]
    times = np.tile(quote_times, len(NAMES))
    bids = 2000 + times + np.repeat(np.arange(len(NAMES)), len(quote_times))
    assert client.call("init", NAMES, symbols, times, bids) == 1


@pytest.mark.parametrize("kind", ["kg", "kgpy"])
def test_stream_before_and_after_batch(kind, request, tmp_path):
    with server(request.getfixturevalue(kind), kind, tmp_path) as (port, _), connect(port) as client:
        initialize(client, kind)
        np.testing.assert_array_equal(client.request("asofbid()"), [-1] * len(NAMES))
        for n in range(5):  # crosses the first change of quote
            assert client.call("ingest", *batch(n)) == 1
            for name in ("latest", "asofbid"):
                np.testing.assert_array_equal(client.request(name + "()"), expected_query(name, n))
        assert client.request("nrows()") == len(NAMES) + BATCH * 5


def measure(kind, command, directory):
    suffix = "[]" if kind == "q" else "()"
    inputs = [batch(i) for i in range(CAPACITY_BATCHES + 1)]
    with server(command, kind, directory) as (port, process), ExitStack() as pool:
        writer, last, asof = (pool.enter_context(connect(port)) for _ in range(3))
        initialize(writer, kind)
        # Warm parsing, the IPC connection and allocation. Exclude that batch from timing.
        assert writer.call("ingest", *inputs[0]) == 1
        for name in ("latest", "asofbid"):
            np.testing.assert_array_equal(writer.request(name + suffix), expected_query(name, 0))
        start = time.perf_counter()
        for columns in inputs[1:]:
            assert writer.call("ingest", *columns) == 1
            assert time.perf_counter() - start < 20, f"{kind} took over 20s to ingest the capacity sample"
        capacity = CAPACITY_BATCHES * BATCH / (time.perf_counter() - start)
        assert writer.request("nrows" + suffix) == len(NAMES) + len(inputs) * BATCH
        initialize(writer, kind)
        inputs = inputs[:PRELOAD_BATCHES + LIVE_BATCHES]
        for columns in inputs[:PRELOAD_BATCHES]:
            assert writer.call("ingest", *columns) == 1
        first_live = PRELOAD_BATCHES
        base = time.perf_counter() + 0.1
        watermark = threading.Lock()
        submitted = acknowledged = first_live - 1

        def ingest():
            nonlocal submitted, acknowledged
            lag = []
            for i in range(LIVE_BATCHES):
                scheduled = base + i * INTERVAL
                time.sleep(max(0, scheduled - time.perf_counter()))
                with watermark:
                    submitted = first_live + i
                assert writer.call("ingest", *inputs[first_live + i]) == 1
                with watermark:
                    acknowledged = first_live + i
                lag.append(time.perf_counter() - scheduled)
                assert lag[-1] < 2, f"{kind} ingest fell more than 2 seconds behind"
            return max(lag) * 1000

        def query(client, name):
            samples = QuerySamples([], [], [])
            for i in range(QUERY_COUNT):
                scheduled = base + (i + 0.5) * INTERVAL * LIVE_BATCHES / QUERY_COUNT
                time.sleep(max(0, scheduled - time.perf_counter()))
                with watermark:
                    lower = acknowledged
                sent = time.perf_counter()
                values = client.request(name + suffix)
                received = time.perf_counter()
                elapsed = received - scheduled
                with watermark:
                    upper = submitted
                assert elapsed < 2, f"{kind} {name} fell more than 2 seconds behind"
                samples.elapsed_ms.append(elapsed * 1000)
                samples.client_delay_ms.append((sent - scheduled) * 1000)
                samples.round_trip_ms.append((received - sent) * 1000)
                # Every snapshot must correspond to a complete, committed prefix of the feed.
                first = int(values[0])
                assert expected_query(name, lower)[0] <= first <= expected_query(name, upper)[0]
                if name == "latest":
                    n = (first - 1000) // BATCH
                    assert first_live - 1 <= n < len(inputs)
                else:
                    quote_time = first - 2000
                    assert quote_time >= 0 and quote_time % QUOTE_STEP == 0
                    n = quote_time // BATCH
                    assert n < len(inputs)
                np.testing.assert_array_equal(values, expected_query(name, n))
            return samples

        with ThreadPoolExecutor(max_workers=3) as executor:
            reads = {name: executor.submit(query, client, name) for name, client in
                     [("latest", last), ("asofbid", asof)]}
            writes = executor.submit(ingest)
            rss = rss_kib(process.pid)
            while not all(f.done() for f in [writes, *reads.values()]):
                time.sleep(0.05)
                rss = max(rss, rss_kib(process.pid))
            lag = writes.result()
            samples = {name: task.result() for name, task in reads.items()}

        assert writer.request("nrows" + suffix) == len(NAMES) + len(inputs) * BATCH
        for name in samples:
            np.testing.assert_array_equal(writer.request(name + suffix), expected_query(name, len(inputs) - 1))
        return Metrics(
            rows_per_second=capacity,
            p50_ms={k: percentile(v.elapsed_ms, 0.50) for k, v in samples.items()},
            p99_ms={k: percentile(v.elapsed_ms, 0.99) for k, v in samples.items()},
            p99_client_delay_ms={k: percentile(v.client_delay_ms, 0.99) for k, v in samples.items()},
            p99_round_trip_ms={k: percentile(v.round_trip_ms, 0.99) for k, v in samples.items()},
            max_ingest_lag_ms=lag,
            observed_rss_kib=rss,
        )


@pytest.mark.parametrize("kind", ["kg", "kgpy"])
def test_tick_service_under_load(kind, request, tmp_path, record_property):
    actual = measure(kind, request.getfixturevalue(kind), tmp_path)
    record_property(kind, json.dumps(asdict(actual), sort_keys=True))
    assert actual.max_ingest_lag_ms < 100, asdict(actual)
    assert max(actual.p99_ms.values()) < 100, asdict(actual)


@pytest.mark.v2_reference
def test_standalone_performance_against_q(q, kg, tmp_path, record_property):
    commands = {"q": q, "kg": kg}
    throughput_ratios = []
    latency_ratios = {"latest": [], "asofbid": []}
    for trial in range(3):
        order = ("q", "kg") if trial % 2 == 0 else ("kg", "q")
        runs = {}
        for kind in order:
            stats = measure(kind, commands[kind], tmp_path / str(trial) / kind)
            record_property(f"{kind}_trial_{trial + 1}", json.dumps(asdict(stats), sort_keys=True))
            assert stats.max_ingest_lag_ms < 100, stats
            runs[kind] = stats
        actual, reference = runs["kg"], runs["q"]
        throughput_ratios.append(actual.rows_per_second / reference.rows_per_second)
        for name in latency_ratios:
            latency_ratios[name].append(actual.p99_ms[name] / reference.p99_ms[name])
    assert statistics.median(throughput_ratios) >= 0.80, throughput_ratios
    for name, ratios in latency_ratios.items():
        assert statistics.median(ratios) <= 1.25, (name, ratios)
