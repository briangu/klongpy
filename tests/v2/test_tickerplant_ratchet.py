"""The ticker plant rate search must back off and confirm its claimed result."""
import json
from pathlib import Path

import pytest

from examples.stocks.tickerplant.bench import endpoint
from tests.ipc_helpers import FakeKdbServer
from examples.stocks.tickerplant.ratchet import find_rate, on_time, select_shape


def test_bracket_and_confirm():
    trials = []

    def trial(rate, phase):
        trials.append((rate, phase))
        return rate <= (10 if phase == "probe" else 9)

    assert find_rate(4, 16, 1, 2, trial) == (9, 10)
    assert trials[:3] == [(4, "probe"), (8, "probe"), (16, "probe")]
    assert trials[-2:] == [(9, "confirm"), (9, "confirm")]


def test_second_confirmation_can_fail():
    attempts = []

    def trial(rate, phase):
        attempts.append((rate, phase))
        return phase == "probe" or rate < 8 or attempts.count((rate, phase)) == 1

    assert find_rate(4, 8, 2, 2, trial) == (6, 8)


def test_cap_and_lower_bound():
    assert find_rate(8, 12, 2, 2, lambda *_: True) == (12, None)
    assert find_rate(8, 12, 2, 2, lambda *_: False) == (0, 2)


def test_deadline_does_not_credit_drained_rows():
    result = dict(batch=100, target_rows_s=1000, window_s=1.05,
                  acknowledged_by_deadline=1000, backlog_at_deadline=100, drain_s=.02)
    assert on_time(result)
    assert not on_time(dict(result, acknowledged_by_deadline=900))
    assert not on_time(dict(result, backlog_at_deadline=200))
    assert not on_time(dict(result, drain_s=.11))


def test_pipeline_tolerance_rejects_growing_or_transient_backlog():
    result = dict(batch=100, target_rows_s=1000, window_s=4,
                  acknowledged_by_deadline=3900, backlog_at_deadline=100, drain_s=.06,
                  acknowledged_each_second=[900, 1900, 2900, 3900])
    assert not on_time(result)
    assert on_time(result, backlog_batches=2)
    assert not on_time(dict(result, acknowledged_each_second=[900, 1900, 2400, 3900]), backlog_batches=2)
    assert not on_time(dict(result, acknowledged_by_deadline=3800, backlog_at_deadline=200,
                            acknowledged_each_second=[1000, 2000, 2900, 3800]), backlog_batches=2)


def test_sweep_selects_measured_batch_workers_and_pipeline():
    tried = []

    def measure(batch, workers, depth):
        tried.append((batch, workers, depth))
        return min(batch, 262144) - max(0, batch - 262144) + workers * 100 + min(depth, 4) * 20 - abs(depth - 4)

    assert select_shape(measure) == (262144, 8, 4)
    assert {b for b, w, d in tried if (w, d) == (4, 1)} == {65536, 131072, 262144, 524288}
    assert (262144, 8, 8) in tried


@pytest.mark.parametrize(("filename", "profiles"), [
    ("2026-11-25-100gbe.jsonl", {"peakram", "steadyram", "peakjournal"}),
    ("2026-11-26-native-100gbe.jsonl", {"nativewal", "peaknowal", "strictnowal", "strictwal"}),
])
def test_published_rate_searches_replay(filename, profiles):
    receipt = Path(__file__).resolve().parents[2] / "examples/stocks/tickerplant/results" / filename
    rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    selections = [row for row in rows if row["kind"] == "selection"]
    assert {row["profile"] for row in selections} == profiles
    for config in selections:
        results = [row for row in rows if row["kind"] == "rate_search" and row["profile"] == config["profile"]]
        attempts = iter(row for row in results if row.get("phase") in {"probe", "confirm"})

        def trial(rate, phase):
            row = next(attempts)
            result = row["result"]
            assert (rate, phase) == (result["target_rows_s"], row["phase"])
            assert result["verified_rows"] == result["end"] - result["first"]
            assert result["subscribers"] == 2
            assert result["live"]["round_trip"]["samples"] and result["history"]["round_trip"]["samples"]
            assert on_time(result, config["backlog_batches"]) == row["on_time"]
            return row["on_time"]

        answer = find_rate(config["start_rate"], config["max_rate"], config["step"], config["confirm_runs"], trial)
        assert answer == (results[-1]["confirmed_rate"], results[-1]["lowest_missed_rate"])
        assert next(attempts, None) is None


def test_benchmark_reads_ipc_password_from_file(tmp_path, monkeypatch):
    password = tmp_path / "password"
    password.write_text("local-test-only\n")
    monkeypatch.setenv("KG_TICKER_BENCH_PASSWORD_FILE", str(password))
    peer = FakeKdbServer(lambda x: x)
    try:
        with endpoint(f"127.0.0.1:{peer.port}") as client:
            assert client.request(3) == 3
        assert peer.greetings[0][:2] == ("bench", "local-test-only")
    finally:
        peer.stop()


def test_search_progresses_when_ceiling_is_between_grid_points():
    probes = []
    def trial(rate, phase):
        if phase == "probe":
            assert rate not in probes
            probes.append(rate)
        return rate <= 8
    assert find_rate(8, 15, 4, 2, trial) == (8, 12)
