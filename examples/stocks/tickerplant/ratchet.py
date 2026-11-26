"""Find a sustainable offered trade rate using the ticker plant benchmark."""
import argparse
import json
import math
import time

from .bench import endpoint, run_session


def on_time(result, backlog_batches=0):
    batch = result["batch"]
    rate = result["target_rows_s"]
    window = result["window_s"]
    if backlog_batches:
        counts = result["acknowledged_each_second"]
        mid = len(counts) // 2
        at_mid = counts[mid - 1] if mid else 0
        budget = backlog_batches * batch
        second_half = math.floor(rate * (window - mid) / batch) * batch
        return (result["acknowledged_by_deadline"] >= batch
                and result["backlog_at_deadline"] <= budget and result["drain_s"] <= budget / rate
                and all(math.ceil(t * rate / batch) * batch - count <= budget
                        for t, count in enumerate(counts, 1))
                and result["acknowledged_by_deadline"] - at_mid >= second_half - batch)
    whole_batches = math.floor(rate * window / batch)
    return (result["acknowledged_by_deadline"] >= max(1, whole_batches) * batch
            and result["backlog_at_deadline"] <= batch
            and result["drain_s"] <= batch / rate)


def find_rate(start, ceiling, step, confirmations, trial):
    """Return the highest confirmed grid point and the lowest observed miss above it."""
    low, high = 0, None
    candidate = start
    while trial(candidate, "probe"):
        low = candidate
        if candidate == ceiling:
            break
        candidate = min(ceiling, 2 * candidate)
    else:
        high = candidate

    while high is not None and high - low > step:
        candidate = low + ((high - low) // (2 * step)) * step
        if trial(candidate, "probe"):
            low = candidate
        else:
            high = candidate

    def confirmed(rate):
        return all(trial(rate, "confirm") for _ in range(confirmations))

    if low == 0 or confirmed(low):
        return low, high

    # A short-window pass can fail over a minute. Back off, then refine using
    # full confirmations so a noisy short probe cannot become the final answer.
    high = low
    distance = step
    candidate = max(0, high - distance)
    while candidate and not confirmed(candidate):
        high = candidate
        distance *= 2
        candidate = max(0, high - distance)
    low = candidate
    while high - low > step:
        candidate = low + ((high - low) // (2 * step)) * step
        if confirmed(candidate):
            low = candidate
        else:
            high = candidate
    return low, high


def select_shape(measure):
    """Repeat the two fastest single-batch shapes, then try overlapping requests."""
    sizes = (65536, 131072, 262144, 524288)
    baseline = {batch: measure(batch, 4, 1) for batch in sizes}
    fastest = sorted(sizes, key=baseline.get, reverse=True)[:2]
    candidates = []
    for workers in (8, 4):
        for batch in fastest if workers == 8 else reversed(fastest):
            rate = measure(batch, workers, 1)
            if workers == 4:
                rate = (rate + baseline[batch]) / 2
            candidates.append((rate, batch, workers))
    rate, batch, workers = max(candidates)
    pipelines = [(rate, batch, 1)]
    for size in sorted(set(fastest) | {sizes[0]}):
        for depth in (2, 4):
            pipelines.append((measure(size, workers, depth), size, depth))
    _, batch, depth = max(pipelines)
    if depth > 1:
        pipelines.append((measure(batch, workers, 8), batch, 8))
        _, batch, depth = max(pipelines)
    return batch, workers, depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", default="127.0.0.1:5510")
    parser.add_argument("--hdb", default="127.0.0.1:5511")
    parser.add_argument("--subscriber", action="append", default=[])
    parser.add_argument("--in-flight", type=int, default=1)
    parser.add_argument("--batch", type=int, default=65536)
    parser.add_argument("--query-hz", type=float, default=20)
    parser.add_argument("--start-rate", type=int, default=1000000)
    parser.add_argument("--max-rate", type=int, default=8000000)
    parser.add_argument("--step", type=int, default=100000)
    parser.add_argument("--probe-seconds", type=int, default=8)
    parser.add_argument("--confirm-seconds", type=int, default=60)
    parser.add_argument("--confirm-runs", type=int, default=2)
    parser.add_argument("--backlog-batches", type=int, default=0,
                        help="allow a bounded pipeline backlog and check steady-state progress")
    parser.add_argument("--sweep", action="store_true", help="select batch size, database workers and pipeline depth")
    parser.add_argument("--sweep-seconds", type=int, default=5)
    args = parser.parse_args()
    if min(args.in_flight, args.step, args.batch, args.probe_seconds, args.confirm_seconds,
           args.confirm_runs, args.query_hz, args.sweep_seconds) <= 0:
        parser.error("step, batch, windows, confirmations and query frequency must be positive")
    if not (0 < args.start_rate <= args.max_rate) or any(
            r % args.step for r in (args.start_rate, args.max_rate)):
        parser.error("start and maximum rates must be positive multiples of the step, with start <= maximum")
    if args.step * min(args.probe_seconds, args.confirm_seconds) < args.batch:
        parser.error("even the lowest rate must offer a full batch per window")
    if args.backlog_batches < 0:
        parser.error("the backlog allowance must be nonnegative")
    args.batch_cycle = None
    with endpoint(args.head) as head, endpoint(args.hdb) as hist:
        day = int(head.request("dayNo"))
        previous = None
        if day:
            count, first, end = map(int, hist.call("archived", day - 1))
            if count:
                previous = (day - 1, first, end + 1)

        def session(rate, seconds, phase, workers=None):
            nonlocal previous
            args.rate = rate
            args.seconds = seconds
            started = time.time()
            result = run_session(args, head, hist, previous)
            if phase != "seed" and not all(result[k]["round_trip"]["samples"] for k in ("live", "history")):
                raise RuntimeError("a query client produced no samples")
            previous = (result["day"], result["first"], result["end"])
            passed = on_time(result, args.backlog_batches) if rate else None
            print(json.dumps(dict(phase=phase, started=started, on_time=passed,
                                  backlog_batches=args.backlog_batches, workers=workers, result=result)), flush=True)
            return result, passed

        if args.sweep:
            hist.request('history("SET threads=8");1')
            head.request('live("SET threads=4");1')
            args.batch, args.in_flight = 8192, 1
        if previous is None:
            session(250000 if args.sweep else args.batch, 1, "seed")
        if args.sweep:
            def measure(batch, workers, depth):
                args.batch, args.in_flight = batch, depth
                head.request(f'live("SET threads={workers}");1')
                result, _ = session(0, args.sweep_seconds, "sweep", workers)
                return result["rows_s"]

            args.batch, workers, args.in_flight = select_shape(measure)
            head.request(f'live("SET threads={workers}");1')
            if args.step * min(args.probe_seconds, args.confirm_seconds) < args.batch:
                parser.error("the lowest rate must offer a full selected batch per window")
            print(json.dumps(dict(phase="selected", batch=args.batch, workers=workers,
                                  hdb_workers=8, in_flight=args.in_flight)), flush=True)

        def run(rate, phase):
            seconds = args.probe_seconds if phase == "probe" else args.confirm_seconds
            _, passed = session(rate, seconds, phase)
            return passed

        best, miss = find_rate(args.start_rate, args.max_rate, args.step, args.confirm_runs, run)
        print(json.dumps(dict(confirmed_rate=best, lowest_missed_rate=miss, step=args.step,
                              confirm_seconds=args.confirm_seconds, confirm_runs=args.confirm_runs,
                              backlog_batches=args.backlog_batches)), flush=True)


if __name__ == "__main__":
    main()
