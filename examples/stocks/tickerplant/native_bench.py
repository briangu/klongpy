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

from .benchmark_metrics import SYMBOLS, archive_checks, last_symbol_seq, timings, volume_before
from .rate_search import find_rate, on_time, select_shape


CLIENTS = Path(__file__).resolve().with_name("native_clients.kg")


def literal(value):
    return '"' + str(value).replace('"', '""') + '"'


def emit(row):
    print(json.dumps(row, separators=(",", ":")), flush=True)


class NativeClients:
    def __init__(self, kg, directory, password_file):
        self.kg, self.directory = kg, directory
        self.prelude = f'.l({literal(CLIENTS)});passFile::{literal(password_file or "")}\n'

    def script(self, name, source):
        path = self.directory / (name + ".kg")
        path.write_text(self.prelude + source)
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

    def launch(self, stack, name, source):
        log = self.directory / (name + ".jsonl")
        err = self.directory / (name + ".err")
        out = stack.enter_context(log.open("w"))
        errors = stack.enter_context(err.open("w"))
        p = subprocess.Popen(self.script(name, source), stdout=out, stderr=errors)
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


def summaries(rows, sent, ack, first, end, previous, ingest, roll_start, roll_end, hz):
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
            batch = (end - first) // len(ack)
            low = last_symbol_seq(first + committed * batch, sym)
            high = last_symbol_seq(first + submitted * batch, sym)
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
            missed += max(0, math.floor(min(t1, ingest)*hz) - round(due*hz))
        if t0 < roll_end and t1 >= roll_start:
            rollover.append((t1-t0)*1000)
    return dict(round_trip=timings(elapsed), scheduled=timings(scheduled), missed=missed), timings(rollover)


def native_session(args, clients, rate, seconds, batch, depth, previous):
    [[day, first, day_first, closing]] = clients.control(args.head, ["[;dayNo;nextSeq;dayFirst;closing]"])
    if first != day_first or closing:
        raise ValueError("a new trading session must be empty and open")
    initial = [clients.control(sub, ["subStats()"])[0] for sub in args.subscriber]
    if any(row[0] != first for row in initial):
        raise ValueError("subscriber is behind before ingest")
    stop = clients.directory / "stop"
    stop.unlink(missing_ok=True)
    start = time.time() + .5
    queries = [(args.head, -1)] + ([(args.hdb, previous[0])] if previous else [])
    with ExitStack() as stack:
        probes = [clients.launch(stack, f"quotes{i}", f"quotes({literal(addr)};{d};{args.query_hz};{start};{literal(stop)})")
                  for i, (addr, d) in enumerate(queries)]
        pub = clients.launch(stack, "publisher",
                             f"publisher({literal(args.head)};{first};{batch};{depth};{rate};{seconds};{start})")
        log = events(pub, "done", max(180, seconds*4))
        sent = [t for kind, _, t in log if kind == "send"]
        ack = [t for kind, _, t in log if kind == "ack"]
        n, ingest = log[-1][1:]
        if not ack or len(sent) != len(ack) or n != len(ack)*batch or (rate and n != math.ceil(rate*seconds/batch)*batch):
            raise ValueError("incomplete publisher session")
        end = first+n
        # Measure the remote call inside kg, excluding process startup and control traffic.
        roll = clients.script("roll", f'peer::openPeer({literal(args.head)});t::.pc();wall::.now();'
                              f'r::peer([;:roll;{day}]);emit([;r;.pc()-t;wall-{start}]);.clic(peer)')
        r = subprocess.run(roll, capture_output=True, text=True, timeout=600, check=True)
        actual, roll_s, roll_start = json.loads(r.stdout)
        if actual != end:
            raise ValueError("bad rollover watermark")
        stop.touch()
        samples = [events(p, "stopped", 200) for p in probes]
    for sub, (_, received, volume) in zip(args.subscriber, initial, strict=True):
        if clients.control(sub, ["subStats()"])[0] != [end, received+n, volume+volume_before(end)-volume_before(first)]:
            raise ValueError("incorrect subscriber cursor or volume")
    checks = list(archive_checks(day, first, end))
    expected = [[n, first, end-1], *([count, count, 0] for count, _ in checks)]
    actual = clients.control(args.hdb, [f"archived({day})", *('history('+literal(sql)+')' for _, sql in checks)])
    if actual != expected:
        raise ValueError(f"archive for day {day} failed independent sequence/payload verification")
    live, roll_live = summaries(samples[0], sent, ack, first, end, None, ingest, roll_start, roll_start+roll_s, args.query_hz)
    hist, roll_hist = (summaries(samples[1], sent, ack, first, end, previous, ingest, roll_start, roll_start+roll_s, args.query_hz)
                       if previous else (None, None))
    if not live["round_trip"]["samples"] or (previous and not hist["round_trip"]["samples"]):
        raise ValueError("missing concurrent quote samples")
    by_deadline = bisect_right(ack, seconds)*batch
    return dict(day=day, first=first, end=end, verified_rows=n, batch=batch, in_flight=depth,
                subscribers=len(initial), target_rows_s=rate, window_s=seconds, ingest_start_epoch=start, ingest_s=ingest, rows_s=n/ingest,
                roll_s=roll_s, ingest_and_roll_rows_s=n/(ingest+roll_s), acknowledged_by_deadline=by_deadline,
                acknowledged_each_second=[bisect_right(ack, t)*batch for t in range(1, int(seconds)+1)],
                backlog_at_deadline=n-by_deadline, drain_s=max(0, ingest-seconds), query_hz=args.query_hz,
                ack=timings([(a-s)*1000 for a, s in zip(ack, sent, strict=True)]),
                live=live, history=hist, roll_live=roll_live, roll_history=roll_hist)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kg", default="kg")
    parser.add_argument("--head", required=True)
    parser.add_argument("--hdb", required=True)
    parser.add_argument("--subscriber", action="append", required=True)
    parser.add_argument("--password-file", default=os.environ.get("KG_TICKER_BENCH_PASSWORD_FILE"))
    parser.add_argument("--mode", choices=["fixed", "sweep", "search"], required=True)
    parser.add_argument("--batch", type=int, default=262144)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--query-hz", type=int, default=20)
    parser.add_argument("--backlog-batches", type=int, default=4)
    parser.add_argument("--start-rate", type=int, default=1048576)
    parser.add_argument("--max-rate", type=int, default=8388608)
    parser.add_argument("--step", type=int, default=131072)
    parser.add_argument("--sweep-seconds", type=int, default=5)
    parser.add_argument("--probe-seconds", type=int, default=8)
    parser.add_argument("--confirm-seconds", type=int, default=60)
    parser.add_argument("--confirm-runs", type=int, default=2)
    args = parser.parse_args()
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

        def session(phase, rate, seconds, batch, workers, depth):
            nonlocal previous
            c.control(args.head, [f'live("SET threads={workers}");1'])
            started = time.time()
            result = native_session(args, c, rate, seconds, batch, depth, previous)
            previous = result["day"], result["first"], result["end"]
            passed = on_time(result, args.backlog_batches) if rate else None
            emit(dict(phase=phase, started=started, workers=workers, backlog_batches=args.backlog_batches,
                      on_time=passed, result=result))
            return result["rows_s"], passed

        hdb_workers = 4 if args.mode == "fixed" else 8
        c.control(args.hdb, [f'history("SET threads={hdb_workers}");1'])
        if previous is None:
            session("seed", 250000, 1, 8192, 4, 1)
        if args.mode == "fixed":
            session("paced", 750000, 60, 32768, 4, 1)
            session("unpaced", 0, 60, 65536, 4, 1)
            return
        batch, workers, depth = args.batch, args.workers, args.depth
        if args.mode == "sweep":
            batch, workers, depth = select_shape(lambda b, w, d: session("sweep", 0, args.sweep_seconds, b, w, d)[0])
        emit(dict(phase="selected", batch=batch, workers=workers, hdb_workers=hdb_workers, in_flight=depth))

        def trial(rate, phase):
            seconds = args.probe_seconds if phase == "probe" else args.confirm_seconds
            return session(phase, rate, seconds, batch, workers, depth)[1]
        best, miss = find_rate(args.start_rate, args.max_rate, args.step, args.confirm_runs, trial)
        emit(dict(confirmed_rate=best, lowest_missed_rate=miss, step=args.step, confirm_seconds=args.confirm_seconds,
                  confirm_runs=args.confirm_runs, backlog_batches=args.backlog_batches))


if __name__ == "__main__":
    main()
