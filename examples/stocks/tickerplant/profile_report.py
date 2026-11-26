"""Join optional native client and head traces from one session and summarize same-request timings."""
import argparse
from bisect import bisect_left
from collections import defaultdict
import json
from pathlib import Path

from .benchmark_metrics import timings


def distribution(values):
    return dict(**timings(values), total_ms=sum(values))


def profile(client, lines):
    start = client["start"]
    low = min(client["clock_before"][0], client["clock_after"][0])
    high = max(client["clock_before"][1], client["clock_after"][1])
    offset = (low + high) / 2
    starts, spans = {}, defaultdict(list)
    wal, sizes = [], {}
    for row in lines:
        if row[0] == "wal_us":
            if client["first"] <= row[2] < client["end"]:
                wal.append(row[3:])
        elif row[0] == "parquet":
            sizes[row[1]] = row[2]
        elif row[0] == "stage":
            _, stage, end, mono, wall = row
            if end:
                before, entered = starts.pop(stage)
                spans[stage].append((entered, wall, (mono-before)*1000))
            else:
                if stage in starts:
                    raise ValueError(f"overlapping {stage} markers")
                starts[stage] = mono, wall

    receipts = [r for r in client["publisher"] if r[0] == "ack"]
    end = start + receipts[-1][2] + high
    begin = start + low
    active = {k: [ms for t, _, ms in v if begin <= t <= end] for k, v in spans.items()}
    active = {k:v for k,v in active.items() if v}
    if len(active.get("publish", [])) != len(receipts) or len(wal) != len(receipts):
        raise ValueError("incomplete publish or WAL profile")

    publishes = [(a,b) for a,b,_ in spans["publish"] if begin <= a <= end]
    sends = [r[2] for r in client["publisher"] if r[0] == "send"]
    producer = defaultdict(list)
    for (a,b), sent, receipt in zip(publishes, sends, receipts, strict=True):
        producer["send_to_handler"].append((a-start-offset-sent)*1000)
        producer["handler_exit_to_ack"].append((start+offset+receipt[2]-b)*1000)
    subscribers = defaultdict(list)
    deliveries = spans["subscriber"]
    delivery_starts = [entry for entry, _, _ in deliveries]
    for a, b in publishes:
        window = deliveries[bisect_left(delivery_starts, a):bisect_left(delivery_starts, b)]
        durations = [ms for _, leave, ms in window if leave <= b]
        for ordinal, duration in enumerate(durations, 1):
            subscribers[f"subscriber_{ordinal}_in_delivery_order"].append(duration)

    queries = spans["quote"]
    entered = [q[0] for q in queries]
    components = defaultdict(list)
    previous = -1
    publish_ends = [finish for _, finish in publishes]
    publish_starts = [entry for entry, _ in publishes]
    for _, _, sent, received, due, _, _ in client["live"][:-1]:
        if due >= client["window_s"]:
            continue
        # One outstanding quote per connection. Identify the matching handler by order and both clock bounds.
        i = max(previous + 1, bisect_left(entered, start + sent + low - .0005))
        if i == len(entered) or queries[i][1] > start + received + high + .0005:
            raise ValueError("client request has no matching server handler in clock bounds")
        previous = i
        entry, _, service = queries[i]
        components["scheduled_to_dispatch"].append((sent-due)*1000)
        components["dispatch_to_handler"].append((entry-start-sent-offset)*1000)
        components["handler"].append(service)
        components["round_trip_minus_handler"].append((received-sent)*1000-service)
        sent_head_time = start + sent + offset
        # Overlapping publish handlers explain some of the wait. Server work before decoding and
        # network transit remain included in dispatch_to_handler. Wall-clock matching is approximate.
        overlapping = publishes[bisect_left(publish_ends, sent_head_time):bisect_left(publish_starts, entry)]
        overlap = sum(max(0, min(entry, b)-max(sent_head_time, a)) for a, b in overlapping)
        components["overlap_with_publish"].append(overlap*1000)

    wfields = ["encode", "checksum", "write", "fdatasync"]
    last = spans["export"]
    closes = [a for a,_,_ in last if a >= begin]
    closing = {}
    if closes:
        close_start = closes[0]
        for k in ("export","seal","reset"):
            closing[k] = next(ms for a,_,ms in spans[k] if a >= close_start)
    return dict(publisher={k:distribution(v) for k,v in producer.items()},
                subscribers={k:distribution(v) for k,v in subscribers.items()}, parquet=sizes.get(client.get("day")), stages={k:distribution(v) for k,v in active.items()}, roll_ms=closing,
                wal_bytes=sum(r[0] for r in wal),
                wal={k:distribution([r[i+1]/1000 for r in wal]) for i,k in enumerate(wfields)},
                quote={k:distribution(v) for k,v in components.items()},
                clock_offset_ms=[1000*low,1000*high])


def subscriber_delivery(client, applied):
    """Join batch watermarks on one load host. Both subscribers must apply every batch once."""
    batches = [r for r in client["publisher"] if r[0] == "feedBatch"]
    arrivals = {client["first"]+row[1]: row[4] for row in batches}
    due = {client["first"]+row[1]: row[3] for row in batches}
    expected = list(arrivals)
    if not expected:
        raise ValueError("requires a small-message publisher trace")
    replies = []
    for rows in applied:
        receipts = [(cursor, wall-client["start"]) for kind,cursor,wall in rows
                    if kind == "applied" and client["first"] < cursor <= client["end"]]
        if [cursor for cursor,_ in receipts] != expected:
            raise ValueError("incomplete or out-of-order subscriber trace")
        replies.append(dict(first_arrival_to_applied=timings([(t-arrivals[n])*1000 for n,t in receipts]),
                            first_due_to_applied=timings([(t-due[n])*1000 for n,t in receipts])))
    return replies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("client", type=Path, help="per-day JSON produced by --trace-dir")
    parser.add_argument("head", nargs="?", type=Path, help="profiled head log")
    parser.add_argument("--subscriber", action="append", type=Path, help="profiled subscriber log from the feed host")
    args = parser.parse_args()
    client = json.loads(args.client.read_text())
    if args.subscriber:
        traces = [[json.loads(line) for line in p.read_text().splitlines() if line.startswith('["applied"')]
                  for p in args.subscriber]
        print(json.dumps(subscriber_delivery(client, traces)))
        return
    if not args.head:
        parser.error("supply the head log or subscriber logs")
    with args.head.open() as stream:
        rows = (json.loads(line) for line in stream if line.startswith(('["stage"','["wal_us"','["parquet"')))
        print(json.dumps(profile(client, rows)))


if __name__ == "__main__":
    main()
