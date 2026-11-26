"""Find a sustainable offered trade rate using the ticker plant benchmark."""
import argparse
import json
import time

from .bench import endpoint, run_session
from .rate_search import find_rate, on_time, select_shape


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
