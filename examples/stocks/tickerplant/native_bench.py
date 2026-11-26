"""Run the ticker-plant suite with native kg publishers, query clients and validators."""
import argparse
from bisect import bisect_right
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

from .benchmark_metrics import SYMBOLS, archive_checks, last_symbol_seq, quote_budget, timings, volume_before
from .rate_search import find_grid_rate, find_rate, on_time, select_shape


CLIENTS = Path(__file__).resolve().with_name("native_clients.kg")


def literal(value):
    return '"' + str(value).replace('"', '""') + '"'


def emit(row):
    print(json.dumps(row, separators=(",", ":")), flush=True)


class NativeClients:
    def __init__(self, kg, directory, password_file):
        self.kg, self.directory = kg, directory
        self.password_file = password_file or ""

    def script(self, name, source, *, library=CLIENTS):
        path = self.directory / (name + ".kg")
        path.write_text(f'.l({literal(library)});passFile::{literal(self.password_file)}\n' + source)
        return [self.kg, str(path)]

    def control(self, address, expressions):
        # One native process and connection for the whole sequence, including archive validation.
        source = [f'peer::openPeer({literal(address)})']
        for expr in expressions:
            source.append(f'emit(peer({literal(expr)}))')
        source.append(".clic(peer)")
        run = subprocess.run(self.script("control", "\n".join(source)), capture_output=True,
                             text=True, timeout=1800)
        if run.returncode or run.stderr:
            raise RuntimeError(run.stderr[-1200:] or f"kg control exited {run.returncode}")
        return [json.loads(line) for line in run.stdout.splitlines()]

    def clock_bounds(self, address):
        # Bounds on server wall time minus client wall time, from same-host round trips.
        source = (f'peer::openPeer({literal(address)});'
                  'sample(i)::{[a b c];a::.now();b::peer(".now()");c::.now();emit([;b-c;b-a])};'
                  "sample'!16;.clic(peer)")
        run = subprocess.run(self.script("clock", source), capture_output=True, text=True, check=True, timeout=30)
        return min(map(json.loads, run.stdout.splitlines()), key=lambda pair: pair[1]-pair[0])

    def launch(self, stack, name, source, *, library=CLIENTS):
        log = self.directory / (name + ".jsonl")
        err = self.directory / (name + ".err")
        out = stack.enter_context(log.open("w"))
        errors = stack.enter_context(err.open("w"))
        p = subprocess.Popen(self.script(name, source, library=library), stdout=out, stderr=errors)
        def reap():
            if p.poll() is None:
                p.kill()
            p.wait()
        stack.callback(reap)
        return p, log, err


def events(task, final, timeout):
    p, log, err = task
    code = p.wait(timeout=timeout)
    problems = err.read_text()
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    if code or problems or not rows or rows[-1][0] != final:
        raise RuntimeError(f"{log.stem} exited {code}, {problems[-1000:]}, last event {rows[-1:]}")
    return rows


def summaries(rows, sent, ack, first, end, previous, ingest, roll_start, roll_end, hz, watermarks, phase):
    elapsed, scheduled, rollover = [], [], []
    missed = 0
    for _, sym, t0, t1, due, seq, price in rows[:-1]:
        if previous:
            day, low, high = previous
            want = last_symbol_seq(high, sym)
            if seq != (want if want >= low else -1):
                raise ValueError(f"wrong historical quote {seq=} for {day=} {sym=}")
        else:
            committed = bisect_right(ack, t0)
            submitted = bisect_right(sent, t1)
            low = last_symbol_seq(first + watermarks[committed], sym)
            high = last_symbol_seq(first + watermarks[submitted], sym)
            if seq == -1:
                if low >= first and t1 < roll_start:
                    raise ValueError("missing committed live quote")
            elif not max(first, low) <= seq <= high:
                raise ValueError(f"live quote outside published range: {sym=} {seq=} {low=} {high=}")
        if seq != -1 and (seq % SYMBOLS != sym or price != 10000 + seq * 17 % 100003):
            raise ValueError("incorrect quote payload")
        if 0 <= t0 < ingest:
            elapsed.append((t1-t0)*1000)
            scheduled.append((t1-due)*1000)
            missed += max(0, math.floor((min(t1, ingest)-phase)*hz) - round((due-phase)*hz))
        if t0 < roll_end and t1 >= roll_start:
            rollover.append((t1-t0)*1000)
    return dict(round_trip=timings(elapsed), scheduled=timings(scheduled), missed=missed), timings(rollover)


def summarize_feed(log, submitted, acknowledgments, rows, capacity, args):
    """Summarize formation, backpressure and receipt separately for variable-size batches."""
    batches = [r for r in log if r[0] == "feedBatch"]
    if ([r[1] for r in batches] != submitted or sum(r[2] for r in batches) != rows
            or any(r[2] > capacity or r[2] <= 0 for r in batches)):
        raise ValueError("incomplete feed batches")
    return dict(message_rows=args.feed_message_rows, burst_messages=args.burst_messages, max_age_ms=args.max_age_ms,
                messages=math.ceil(rows/args.feed_message_rows), min_batch=min(r[2] for r in batches),
                max_batch=max(r[2] for r in batches), batches=len(batches),
                first_arrival_to_ack=timings([(a-r[4])*1000 for a, r in zip(acknowledgments, batches, strict=True)]),
                first_due_to_ack=timings([(a-r[3])*1000 for a, r in zip(acknowledgments, batches, strict=True)]),
                first_arrival_to_seal=timings([(r[6]-r[4])*1000 for r in batches]),
                seal_to_send=timings([(r[5]-r[6])*1000 for r in batches]),
                first_arrival_to_send=timings([(r[5]-r[4])*1000 for r in batches]),
                first_due_to_arrival=timings([(r[4]-r[3])*1000 for r in batches]))


def native_session(args, clients, rate, seconds, batch, depth, previous, phase=0, use_feed=True):
    [[day, first, day_first, closing]] = clients.control(args.head, ["[;dayNo;nextSeq;dayFirst;closing]"])
    if first != day_first or closing:
        raise ValueError("a new trading session must be empty and open")
    initial = [clients.control(sub, ["subStats()"])[0] for sub in args.subscriber]
    if any(row[0] != first for row in initial):
        raise ValueError("subscriber is behind before ingest")
    stop = clients.directory / "stop"
    stop.unlink(missing_ok=True)
    trace = getattr(args, "trace_dir", None)
    if trace:
        trace.mkdir(parents=True, exist_ok=True)
        clock_before = clients.clock_bounds(args.head)
    start = time.time() + .5
    queries = [(args.head, -1)] + ([(args.hdb, previous[0])] if previous else [])
    message_rows = getattr(args, "feed_message_rows", None) if use_feed else None
    if message_rows:
        total = math.ceil(rate * seconds / message_rows) * message_rows
        library = CLIENTS.with_name("native_feed.kg")
        publisher = (f'feedPublisher({literal(args.head)};{first};{message_rows};{batch};{args.burst_messages};'
                     f'{args.max_age_ms/1000};{rate};{total};{start})')
    else:
        library = CLIENTS
        publisher = f"publisher({literal(args.head)};{first};{batch};{depth};{rate};{seconds};{start})"
    with ExitStack() as stack:
        probes = [clients.launch(stack, f"quotes{i}", f"quotePhase::{phase};quotes({literal(addr)};{d};{args.query_hz};{start};{literal(stop)})")
                  for i, (addr, d) in enumerate(queries)]
        pub = clients.launch(stack, "publisher", publisher, library=library)
        log = events(pub, "done", max(180, seconds*4))
        sent = [row[2] for row in log if row[0] == "send"]
        ack = [row[2] for row in log if row[0] == "ack"]
        watermarks = [0] + [row[1] for row in log if row[0] == "ack"]
        submitted = [row[1] for row in log if row[0] == "send"]
        n, ingest = log[-1][1:]
        expected = total if message_rows else math.ceil(rate*seconds/batch)*batch if rate else n
        if (not ack or submitted != watermarks[1:] or n != watermarks[-1] or n != expected
                or any(a >= b for a, b in zip(watermarks, watermarks[1:]))
                or (not message_rows and n != batch * len(ack))):
            raise ValueError("incomplete publisher session")
        end = first+n
        # Keep the entire paced quote window open when the last batch finishes early.
        if rate:
            time.sleep(max(0, start + seconds - time.time()))
        # Measure the remote call inside kg, excluding process startup and control traffic.
        roll = clients.script("roll", f'peer::openPeer({literal(args.head)});t::.pc();wall::.now();'
                              f'r::peer([;:roll;{day}]);emit([;r;.pc()-t;wall-{start}]);.clic(peer)')
        r = subprocess.run(roll, capture_output=True, text=True, timeout=600, check=True)
        actual, roll_s, roll_start = json.loads(r.stdout)
        if actual != end:
            raise ValueError("bad rollover watermark")
        stop.touch()
        samples = [events(p, "stopped", 200) for p in probes]
    if trace:
        with (trace / f"day-{day}.json").open("x") as output:
            json.dump(dict(day=day, start=start, end=end, first=first, window_s=seconds, batch=batch,
                           clock_before=clock_before, clock_after=clients.clock_bounds(args.head),
                           publisher=log, live=samples[0]), output)
    for sub, (_, received, volume) in zip(args.subscriber, initial, strict=True):
        if clients.control(sub, ["subStats()"])[0] != [end, received+n, volume+volume_before(end)-volume_before(first)]:
            raise ValueError("incorrect subscriber cursor or volume")
    checks = list(archive_checks(day, first, end))
    expected = [[n, first, end-1], *([count, count, 0] for count, _ in checks)]
    actual = clients.control(args.hdb, [f"archived({day})", *('history('+literal(sql)+')' for _, sql in checks)])
    if actual != expected:
        raise ValueError(f"archive for day {day} failed independent sequence/payload verification")
    live, roll_live = summaries(samples[0], sent, ack, first, end, None, ingest, roll_start, roll_start+roll_s, args.query_hz, watermarks, phase)
    hist, roll_hist = (summaries(samples[1], sent, ack, first, end, previous, ingest, roll_start, roll_start+roll_s, args.query_hz, watermarks, phase)
                       if previous else (None, None))
    if not getattr(args, "quote_slo_ms", None) and (not live["round_trip"]["samples"] or (previous and not hist["round_trip"]["samples"])):
        raise ValueError("missing concurrent quote samples")
    for reader, rows in zip((live, hist), samples):
        reader["offer_window"] = quote_budget(((row[4], row[3]) for row in rows[:-1]),
                                            seconds, args.query_hz, getattr(args, "quote_slo_ms", None) or 50, phase)
    by_deadline = watermarks[bisect_right(ack, seconds)]
    result = dict(day=day, first=first, end=end, verified_rows=n, batch=batch, in_flight=depth,
                subscribers=len(initial), target_rows_s=rate, window_s=seconds, ingest_start_epoch=start, ingest_s=ingest, rows_s=n/ingest,
                roll_s=roll_s, ingest_and_roll_rows_s=n/(ingest+roll_s), offer_start_to_close_s=roll_start+roll_s,
                offer_start_to_close_rows_s=n/(roll_start+roll_s), acknowledged_by_deadline=by_deadline,
                acknowledged_each_second=[watermarks[bisect_right(ack, t)] for t in range(1, int(seconds)+1)],
                backlog_at_deadline=n-by_deadline, drain_s=max(0, ingest-seconds), query_hz=args.query_hz, quote_phase_ms=1000*phase,
                ack=timings([(a-s)*1000 for a, s in zip(ack, sent, strict=True)]),
                live=live, history=hist, roll_live=roll_live, roll_history=roll_hist)
    if message_rows:
        result["feed"] = summarize_feed(log, submitted, ack, n, batch, args)
    return result


def archive_probes(clients, address, newest, repeats):
    """Time exact latest-price IPC queries on old, middle and recent sealed days."""
    days = sorted({0, newest // 2, newest})
    bounds = clients.control(address, [f"archived({day})" for day in days])
    by_day = {day: bound for day, bound in zip(days, bounds, strict=True) if bound[0]}
    days = list(by_day)
    src = [f'peer::openPeer({literal(address)})',
           'sample(d;s)::{[a v b];a::.pc();v::peer([;:historyQuote;d;s]);b::.pc();emit([;d;s;b-a;v])}',
           "probe(i)::{[s];s::(113*i)!1024;{sample(x;s);sample(x;1024)}'days}",
           'days::[' + ' '.join(map(str, days)) + ']', f"probe'!{repeats};.clic(peer)"]
    run = subprocess.run(clients.script("history-probe", "\n".join(src)), capture_output=True,
                         text=True, check=True, timeout=600)
    samples = {(day, hit): [] for day in days for hit in (True, False)}
    for day, sym, elapsed, quote in map(json.loads, run.stdout.splitlines()):
        _, low, high = by_day[day]
        seq = last_symbol_seq(high + 1, sym)
        expected = [seq, 10000 + seq * 17 % 100003] if sym < SYMBOLS and seq >= low else []
        if quote != expected:
            raise ValueError(f"invalid historical probe on day {day} symbol {sym}")
        samples[day, bool(expected)].append(elapsed * 1000)
    if sum(map(len, samples.values())) != 2*len(days)*repeats:
        raise ValueError("incomplete archive probes")
    return [dict(day=day, rows=by_day[day][0], hits=timings(samples[day, True]),
                 absent=timings(samples[day, False])) for day in days]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-probes", type=int, default=0, help="timed historical hit/miss pairs per sampled archive day after each close")
    parser.add_argument("--trace-dir", type=Path, help="save client events and head clock bounds per session")
    parser.add_argument("--kg", default="kg")
    parser.add_argument("--head", required=True)
    parser.add_argument("--hdb", required=True)
    parser.add_argument("--subscriber", action="append", required=True)
    parser.add_argument("--password-file", default=os.environ.get("KG_TICKER_BENCH_PASSWORD_FILE"))
    parser.add_argument("--mode", choices=["fixed", "sweep", "search", "grid", "repeat"], required=True)
    parser.add_argument("--rate", type=int, help="offered trades/s for repeat mode (0 for unpaced)")
    parser.add_argument("--batch", type=int, default=262144)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--feed-message-rows", type=int, help="replay small synthetic messages, one WAL batch in flight")
    parser.add_argument("--burst-messages", type=int, default=1)
    parser.add_argument("--max-age-ms", type=float, default=0, help="flush oldest admitted feed message on timer; zero is size only")
    parser.add_argument("--query-hz", type=int, default=20)
    parser.add_argument("--confirmation-phase-ms", type=float, default=0, help="offset alternating confirmation readers to check sensitivity to batch cadence")
    parser.add_argument("--quote-slo-ms", type=float,
                        help="also require 99%% of all scheduled slots per reader to finish within this latency")
    parser.add_argument("--backlog-batches", type=int, default=4)
    parser.add_argument("--start-rate", type=int, default=1048576)
    parser.add_argument("--max-rate", type=int, default=8388608)
    parser.add_argument("--step", type=int, default=131072)
    parser.add_argument("--sweep-seconds", type=int, default=5)
    parser.add_argument("--probe-seconds", type=int, default=8)
    parser.add_argument("--confirm-seconds", type=int, default=60)
    parser.add_argument("--confirm-runs", type=int, default=2)
    args = parser.parse_args()
    if args.query_hz <= 0 or not 0 <= args.confirmation_phase_ms < 1000/args.query_hz:
        parser.error("confirmation phase must lie in one quote period")
    if args.mode == "grid" and (args.step <= 0 or (args.max_rate-args.start_rate) % args.step):
        parser.error("grid endpoints must align with a positive step")
    if args.archive_probes < 0:
        parser.error("archive probe count must be nonnegative")
    if args.feed_message_rows is not None:
        if (args.mode != "repeat" or not args.rate or args.depth != 1 or args.feed_message_rows <= 0
                or args.batch % args.feed_message_rows or args.burst_messages < 1 or not 0 <= args.max_age_ms < math.inf):
            parser.error("small-message replay requires paced repeat, depth one and batch divisible by positive message size")
    if args.quote_slo_ms is not None and not 0 < args.quote_slo_ms < math.inf:
        parser.error("quote target must be finite and positive")
    if args.mode == "repeat":
        if args.rate is None or args.rate < 0:
            parser.error("repeat mode requires a nonnegative --rate")
    elif args.rate is not None:
        parser.error("--rate applies to repeat mode")
    if min(args.batch, args.workers, args.depth, args.query_hz, args.step, args.sweep_seconds, args.probe_seconds,
           args.confirm_seconds, args.confirm_runs) <= 0 or not 0 < args.start_rate <= args.max_rate or args.backlog_batches < 0:
        parser.error("positive rates, session dimensions and nonnegative backlog required")
    with tempfile.TemporaryDirectory() as work:
        c = NativeClients(args.kg, Path(work), args.password_file)
        [day] = c.control(args.head, ["dayNo"])
        previous = None
        if day:
            [[count, low, high]] = c.control(args.hdb, [f"archived({day-1})"])
            if count:
                previous = day-1, low, high+1

        confirmation_number = 0

        def session(phase, rate, seconds, batch, workers, depth):
            nonlocal previous, confirmation_number
            offset = 0
            if phase in ("confirm", "repeat"):
                offset = (confirmation_number % 2) * args.confirmation_phase_ms / 1000
                confirmation_number += 1
            elif phase == "probe":
                confirmation_number = 0
            c.control(args.head, [f'live("SET threads={workers}");1'])
            started = time.time()
            result = native_session(args, c, rate, seconds, batch, depth, previous, offset, use_feed=phase != "seed")
            if args.archive_probes:
                result["archive_probes"] = archive_probes(c, args.hdb, result["day"], args.archive_probes)
            previous = result["day"], result["first"], result["end"]
            passed = on_time(result, args.backlog_batches) if rate else None
            qualified = passed
            if args.quote_slo_ms and phase != "seed":
                qualified = passed and all(result[k] and result[k]["offer_window"]["passed"] for k in ("live", "history"))
            emit(dict(phase=phase, started=started, workers=workers, backlog_batches=args.backlog_batches,
                      on_time=passed, qualified=qualified, result=result))
            return result["rows_s"], qualified

        hdb_workers = 4 if args.mode == "fixed" else 8
        c.control(args.hdb, [f'history("SET threads={hdb_workers}");1'])
        if previous is None:
            session("seed", 250000, 1, 8192, 4, 1)
        if args.mode == "fixed":
            session("paced", 750000, 60, 32768, 4, 1)
            session("unpaced", 0, 60, 65536, 4, 1)
            return
        batch, workers, depth = args.batch, args.workers, args.depth
        if args.mode == "repeat":
            for _ in range(args.confirm_runs):
                session("repeat", args.rate, args.confirm_seconds, batch, workers, depth)
            return
        if args.mode == "sweep":
            batch, workers, depth = select_shape(lambda b, w, d: session("sweep", 0, args.sweep_seconds, b, w, d)[0])
        emit(dict(phase="selected", batch=batch, workers=workers, hdb_workers=hdb_workers, in_flight=depth))

        def trial(rate, phase):
            seconds = args.probe_seconds if phase == "probe" else args.confirm_seconds
            return session(phase, rate, seconds, batch, workers, depth)[1]
        search = find_grid_rate if args.mode == "grid" else find_rate
        best, miss = search(args.start_rate, args.max_rate, args.step, args.confirm_runs, trial)
        emit(dict(confirmed_rate=best, lowest_missed_rate=miss, step=args.step, confirm_seconds=args.confirm_seconds,
                  confirm_runs=args.confirm_runs, backlog_batches=args.backlog_batches))


if __name__ == "__main__":
    main()
