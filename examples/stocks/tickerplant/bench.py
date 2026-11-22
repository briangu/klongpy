"""Measure a native kg ticker plant over TCP. Run as a module from the source distribution."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
import json
import math
import statistics
import threading
import time

import numpy as np

from tests.v2.peer import connect


EPOCH = 1700000000000000000
SYMBOLS = 1024


@dataclass
class Watermarks:
    acknowledged: int
    submitted: int
    lock: threading.Lock = field(default_factory=threading.Lock)


def endpoint(address):
    host, port = address.rsplit(":", 1)
    return connect(int(port), host, timeout=180)


def columns(first, n):
    seq = np.arange(first, first + n, dtype=np.int64)
    return (EPOCH + seq * 1000 + (seq % 523) * 20, seq % SYMBOLS,
            10000 + (seq * 17) % 100003, 1 + seq % 97)


def quantile(values, q):
    return sorted(values)[math.ceil(q * len(values)) - 1] if values else None


def timings(samples):
    return dict(samples=len(samples), p50_ms=statistics.median(samples) if samples else None,
                p99_ms=quantile(samples, .99), max_ms=max(samples, default=None))


def last_symbol_seq(end, symbol):
    return end - 1 - ((end - 1 - symbol) % SYMBOLS)


def read_quotes(client, stop, hz, first, bounds, previous=None):
    elapsed, scheduled = [], []
    missed = 0
    next_read = time.perf_counter()
    i = 0
    while not stop.wait(max(0, next_read - time.perf_counter())):
        symbol = (i * 113) % SYMBOLS
        sent = time.perf_counter()
        if previous is None:
            with bounds.lock:
                lower = last_symbol_seq(bounds.acknowledged, symbol)
            quote = client.call("latest", symbol)
            with bounds.lock:
                upper = last_symbol_seq(bounds.submitted, symbol)
            if (len(quote) and not max(first, lower) <= int(quote[0]) <= upper) or (not len(quote) and lower >= first):
                raise ValueError(f"live quote outside acknowledged/submitted bounds: {quote}, {lower}, {upper}")
        else:
            day, start, end = previous
            quote = client.call("historyQuote", day, symbol)
            last = last_symbol_seq(end, symbol)
            if (len(quote) > 0) != (last >= start) or (len(quote) and quote[0] != last):
                raise ValueError(f"wrong archived quote for {day=} {symbol=}: {quote}")
        received = time.perf_counter()
        if len(quote):
            seq, price = map(int, quote)
            if seq % SYMBOLS != symbol or price != 10000 + seq * 17 % 100003:
                raise ValueError(f"wrong quote {symbol=}: {quote}")
            if previous is None and seq < first:
                raise ValueError(f"stale live quote {seq} before {first}")
        elapsed.append((received - sent) * 1000)
        scheduled.append((received - next_read) * 1000)
        next_read += 1 / hz
        if next_read < received:
            skip = math.ceil((received - next_read) * hz)
            next_read += skip / hz
            missed += skip
        i += 1
    return dict(round_trip=timings(elapsed), scheduled=timings(scheduled), missed=missed)


def verify(hist, day, first, end):
    np.testing.assert_array_equal(hist.call("archived", day), [end - first, first, end - 1])
    # Bound the distinct-count hash table even when a session has tens of millions of rows.
    for low in range(first, end, 1000000):
        high = min(low + 1000000, end)
        sql = (f"SELECT count(*),count(DISTINCT seq),count(*) FILTER (WHERE "
               f"ts IS DISTINCT FROM {EPOCH}+seq*1000+(seq%523)*20 OR sym IS DISTINCT FROM seq%{SYMBOLS} "
               "OR px IS DISTINCT FROM 10000+(seq*17)%100003 OR qty IS DISTINCT FROM 1+seq%97) "
               f"FROM trades WHERE dayno={day} AND seq>={low} AND seq<{high}")
        actual = hist.request(f'history("{sql}")')
        np.testing.assert_array_equal(actual, [high - low, high - low, 0], err_msg=f"day {day}, seq {low}..{high-1}")


def volume_before(end):
    cycles, rest = divmod(end, 97)
    return cycles * (97 * 98 // 2) + rest * (rest + 1) // 2


def roll_with_probes(args, head, day, end, previous):
    """Keep queries in flight while the writer exports and the HDB imports."""
    stop = threading.Event()
    gate = threading.Event()

    def probe(client, historical):
        samples = []
        gate.wait()
        while not stop.wait(.01 if not samples else 1 / args.query_hz):
            sent = time.perf_counter()
            if historical:
                old_day, low, high = previous
                quote = client.call("historyQuote", old_day, 0)
                want = last_symbol_seq(high, 0)
                if (len(quote) > 0) != (want >= low):
                    raise ValueError("historical quote disappeared during rollover")
            else:
                quote = client.call("latest", 0)
                want = last_symbol_seq(end, 0)
            samples.append((time.perf_counter() - sent) * 1000)
            if len(quote) and list(map(int, quote)) != [want, 10000 + want * 17 % 100003]:
                raise ValueError(f"incorrect rollover quote: {quote}")
        return timings(samples)

    with endpoint(args.head) as live, endpoint(args.hdb) as archive, ThreadPoolExecutor(max_workers=2) as pool:
        live_probe = pool.submit(probe, live, False)
        hist_probe = pool.submit(probe, archive, True) if previous else None
        start = time.perf_counter()
        gate.set()
        try:
            if head.call("roll", day) != end:
                raise ValueError("incorrect end-of-day acknowledgment")
            elapsed = time.perf_counter() - start
        finally:
            stop.set()
        return elapsed, live_probe.result(), hist_probe.result() if hist_probe else None


def run_session(args, head, hist, previous):
    day, first, day_first, closing = map(int, head.request("[;dayNo;nextSeq;dayFirst;closing]"))
    if first != day_first or closing:
        raise ValueError("start the benchmark with an empty, open trading session")
    stop = threading.Event()
    next_seq = first
    acknowledged_in_window = 0
    acks, lag = [], []
    with ExitStack() as stack:
        readers = stack.enter_context(ThreadPoolExecutor(max_workers=2))
        live = stack.enter_context(endpoint(args.head))
        arch = stack.enter_context(endpoint(args.hdb))
        subscribers = [stack.enter_context(endpoint(addr)) for addr in args.subscriber]
        initial = [list(map(int, sub.request("subStats()"))) for sub in subscribers]
        if any(cursor != first for cursor, _, _ in initial):
            raise ValueError("subscriber must be caught up before the session starts")
        bounds = Watermarks(first, first)
        live_reads = readers.submit(read_quotes, live, stop, args.query_hz, first, bounds)
        history_reads = readers.submit(read_quotes, arch, stop, args.query_hz, first, bounds, previous) if previous else None
        try:
            start = time.perf_counter()
            finish = start + args.seconds
            while True:
                due = start + (next_seq - first) / args.rate if args.rate else 0
                if acks and (due or time.perf_counter()) >= finish:
                    break
                if due:
                    time.sleep(max(0, due - time.perf_counter()))
                batch = args.batch_cycle[len(acks) % len(args.batch_cycle)] if args.batch_cycle else args.batch
                payload = columns(next_seq, batch)
                with bounds.lock:
                    bounds.submitted = next_seq + batch
                sent = time.perf_counter()
                ack = head.call("publish", next_seq, *payload)
                now = time.perf_counter()
                if ack != next_seq + batch:
                    raise ValueError(f"unexpected watermark {ack} after {next_seq}")
                next_seq = ack
                if now <= finish:
                    acknowledged_in_window = next_seq - first
                with bounds.lock:
                    bounds.acknowledged = next_seq
                acks.append((now - sent) * 1000)
                if due:
                    lag.append((now - due) * 1000)
            ingest_seconds = time.perf_counter() - start
        finally:
            stop.set()
            # Both workers must finish before their sockets are closed.
            live_metrics = live_reads.result()
            history_metrics = history_reads.result() if history_reads else None
        for sub, (_, count, volume) in zip(subscribers, initial, strict=True):
            np.testing.assert_array_equal(sub.request("subStats()"),
                [next_seq, count + next_seq - first, volume + volume_before(next_seq) - volume_before(first)])
    roll_seconds, roll_live, roll_hist = roll_with_probes(args, head, day, next_seq, previous)
    verify(hist, day, first, next_seq)
    n = next_seq - first
    return dict(day=day, first=first, end=next_seq, verified_rows=n, batch=args.batch_cycle or args.batch,
                target_rows_s=args.rate, window_s=args.seconds, acknowledged_by_deadline=acknowledged_in_window,
                backlog_at_deadline=n-acknowledged_in_window, drain_s=max(0.0, ingest_seconds-args.seconds),
                query_hz=args.query_hz, subscribers=len(subscribers), ingest_s=ingest_seconds,
                rows_s=n / ingest_seconds, roll_s=roll_seconds, ingest_and_roll_rows_s=n/(ingest_seconds+roll_seconds),
                ack=timings(acks), ack_from_schedule=timings(lag), live=live_metrics, history=history_metrics,
                roll_live=roll_live, roll_history=roll_hist)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", default="127.0.0.1:5510")
    parser.add_argument("--hdb", default="127.0.0.1:5511")
    parser.add_argument("--subscriber", action="append", default=[], help="check this subscriber's receipt, repeat for fan-out")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--batch-cycle", type=int, nargs="+", help="rotate through these batch lengths")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--rate", type=float, default=0, help="offered trades/sec; zero saturates a synchronous publisher")
    parser.add_argument("--query-hz", type=float, default=20)
    args = parser.parse_args()
    if min(args.batch, args.seconds, args.sessions, args.query_hz, *(args.batch_cycle or [])) <= 0 or args.rate < 0:
        parser.error("batch, seconds, sessions and query-hz must be positive; rate must be nonnegative")
    with endpoint(args.head) as head, endpoint(args.hdb) as hist:
        day = int(head.request("dayNo"))
        previous = None
        if day:
            n, low, high = map(int, hist.call("archived", day - 1))
            if n:
                previous = (day - 1, low, high + 1)
        for _ in range(args.sessions):
            result = run_session(args, head, hist, previous)
            print(json.dumps(result), flush=True)
            previous = (result["day"], result["first"], result["end"])


if __name__ == "__main__":
    main()
