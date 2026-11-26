"""Measure a native kg ticker plant over TCP. Run as a module from the source distribution."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
import json
from itertools import accumulate
import math
import os
from pathlib import Path
import select
import threading
import time

import numpy as np

from klongpy.core import KGSym
from klongpy.ipc import Message
from tests.v2.peer import connect

from .benchmark_metrics import EPOCH, SYMBOLS, archive_checks, last_symbol_seq, timings, volume_before


@dataclass
class Watermarks:
    acknowledged: int
    submitted: int
    lock: threading.Lock = field(default_factory=threading.Lock)


def endpoint(address):
    host, port = address.rsplit(":", 1)
    path = os.environ.get("KG_TICKER_BENCH_PASSWORD_FILE")
    if path:
        return connect(int(port), host, timeout=180, user="bench", password=Path(path).read_text().rstrip("\r\n"))
    return connect(int(port), host, timeout=180)


def columns(first, n):
    seq = np.arange(first, first + n, dtype=np.int64)
    return (EPOCH + seq * 1000 + (seq % 523) * 20, seq % SYMBOLS,
            10000 + (seq * 17) % 100003, 1 + seq % 97)


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
    for count, sql in archive_checks(day, first, end):
        actual = hist.request(f'history("{sql}")')
        np.testing.assert_array_equal(actual, [count, count, 0], err_msg=f"day {day}")


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


def publish_batches(args, head, first, start, bounds):
    """Keep a bounded number of ordered sync calls on one publisher connection."""
    finish = start + args.seconds
    seq, count = first, 0
    pending = deque()
    while True:
        now = time.perf_counter()
        due = start + (seq - first) / args.rate if args.rate else 0
        more = count == 0 or (due or now) < finish
        can_send = more and len(pending) < args.in_flight
        if pending:
            wait = max(0, pending[0][1] + head.timeout - now)
            if can_send:
                wait = min(wait, max(0, due - now))
            if select.select([head.sock], [], [], wait)[0]:
                data = head.sock.recv(65536)
                received = time.perf_counter()
                if not data:
                    raise ConnectionError("publisher lost its connection waiting for a batch acknowledgment")
                for event in head.session.feed(data):
                    expected, sent, scheduled, request = pending.popleft()
                    if (not isinstance(event, Message) or event.pending is not request or event.error
                            or event.value != expected):
                        raise ValueError(f"expected batch watermark {expected}, received {event}")
                    yield expected, sent, scheduled, received
                continue
            if time.perf_counter() >= pending[0][1] + head.timeout:
                raise TimeoutError("batch acknowledgment deadline expired")
            if not can_send:
                continue
        if not more:
            return
        if due:
            time.sleep(max(0, due - time.perf_counter()))
        batch = args.batch_cycle[count % len(args.batch_cycle)] if args.batch_cycle else args.batch
        payload = columns(seq, batch)
        end = seq + batch
        with bounds.lock:
            bounds.submitted = end
        sent = time.perf_counter()
        if args.in_flight == 1:
            ack = head.call("publish", seq, *payload)
            if ack != end:
                raise ValueError(f"unexpected watermark {ack} after {seq}")
            yield end, sent, due, time.perf_counter()
        else:
            packet, request = head.session.send_sync([KGSym("publish"), seq, *payload], call=True, blocking=False)
            head.sock.sendall(packet)
            pending.append((end, sent, due, request))
        seq, count = end, count + 1


def run_session(args, head, hist, previous):
    day, first, day_first, closing = map(int, head.request("[;dayNo;nextSeq;dayFirst;closing]"))
    if first != day_first or closing:
        raise ValueError("start the benchmark with an empty, open trading session")
    stop = threading.Event()
    next_seq = first
    acknowledged_in_window = 0
    acknowledged_each_second = [0] * math.floor(args.seconds)
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
            for next_seq, sent, due, now in publish_batches(args, head, first, start, bounds):
                if now <= finish:
                    acknowledged_in_window = next_seq - first
                    second = max(1, math.ceil(now - start))
                    if second <= len(acknowledged_each_second):
                        acknowledged_each_second[second - 1] = acknowledged_in_window
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
                acknowledged_each_second=list(accumulate(acknowledged_each_second, max)),
                backlog_at_deadline=n-acknowledged_in_window, drain_s=max(0.0, ingest_seconds-args.seconds),
                query_hz=args.query_hz, in_flight=args.in_flight, subscribers=len(subscribers), ingest_s=ingest_seconds,
                rows_s=n / ingest_seconds, roll_s=roll_seconds, ingest_and_roll_rows_s=n/(ingest_seconds+roll_seconds),
                ack=timings(acks), ack_from_schedule=timings(lag), live=live_metrics, history=history_metrics,
                roll_live=roll_live, roll_history=roll_hist)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", default="127.0.0.1:5510")
    parser.add_argument("--hdb", default="127.0.0.1:5511")
    parser.add_argument("--subscriber", action="append", default=[], help="check this subscriber's receipt, repeat for fan-out")
    parser.add_argument("--in-flight", type=int, default=1, help="maximum ordered batches awaiting reply")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--batch-cycle", type=int, nargs="+", help="rotate through these batch lengths")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--rate", type=float, default=0, help="offered trades/sec; zero saturates a synchronous publisher")
    parser.add_argument("--query-hz", type=float, default=20)
    args = parser.parse_args()
    if min(args.in_flight, args.batch, args.seconds, args.sessions, args.query_hz, *(args.batch_cycle or [])) <= 0:
        parser.error("batch sizes, in-flight limit, duration, sessions and query frequency must be positive")
    if args.rate < 0:
        parser.error("the offered rate must be nonnegative")
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
