import pytest

from examples.stocks.tickerplant.benchmark_metrics import quote_budget
from examples.stocks.tickerplant.rate_search import find_grid_rate


def test_scheduled_slots_include_skipped_and_late_responses():
    measured = quote_budget([(0, .030), (.05, .105), (.15, .153), (.20, .202)], .2, 20, 50)
    assert measured["planned"] == 4
    assert (measured["answered"], measured["skipped"], measured["late"]) == (3, 1, 1)
    assert measured["within_target"] == 2 and not measured["passed"]
    assert not quote_budget([], 60, 20, 50)["passed"]


def test_quote_budget_uses_exact_window_and_rejects_duplicate_slots():
    times = [(i / 20, i / 20 + .005) for i in range(1188)]
    assert quote_budget(times, 60, 20, 50)["passed"]
    assert not quote_budget(times[:-1], 60, 20, 50)["passed"]
    with pytest.raises(ValueError, match="duplicate"):
        quote_budget(times + times[:1], 60, 20, 50)


def test_offset_schedule_has_no_extra_last_slot():
    result = quote_budget([(.025, .03), (.075, .08)], .1, 20, 50, phase=.025)
    assert result["planned"] == result["within_target"] == 2


def test_grid_search_checks_above_a_lower_miss_and_confirms_twice():
    calls = []
    def trial(rate, phase):
        calls.append((rate, phase))
        return rate in (20, 40) and not (rate == 40 and calls.count((40, "confirm")) == 2)
    assert find_grid_rate(10, 50, 10, 2, trial) == (20, 30)
    assert calls.count((40, "confirm")) == calls.count((20, "confirm")) == 2
