# Ticker plant optimization on Milan and Tinkara

The [quote-qualified sweep](#durable-rate-with-two-quote-readers) confirmed **2,097,152 offered trades/s** with both readers meeting their 50 ms budget. The post-RC1 baseline confirmed 3,801,088 offered trades/s with the WAL and two subscribers. One batch drained 13–36 ms after the minute. Live quote p99 was 125–126 ms, and 509–530 of the 1,200 scheduled live reads were missed. Historical reads missed 601–603 slots. Milan received 1.02 Gbit/s and transmitted 2.04 Gbit/s on its 100 GbE interface. Rollover added 40–41 seconds after a minute of ingest. [Baseline measurements](results/2026-11-26-native-100gbe.jsonl)

## What the benchmark counts

The durable rate is the offered number of synthetic trade rows per second. The publisher receives a batch watermark after the head syncs its WAL, commits the RAM RDB and receives each connected subscriber’s acknowledgment. Both subscriber cursors, their total quantities, and every archived sequence and payload are checked. KX’s reference implementation separates ticker plant and RDB processes, supports timer-based publication, and saves the day from the RDB. These implementations have different acknowledgment boundaries. [KX ticker plant](https://github.com/KxSystems/kdb-tick/blob/master/tick.q) · [KX RDB](https://github.com/KxSystems/kdb-tick/blob/master/tick/r.q)

The engineering target is 99% of **all scheduled quote requests** answered in 50 ms at each of two independent 20 Hz readers. A one-minute run has 1,200 slots per reader and allows 12 late or skipped combined. Timing starts when the request was due. The load generator records when it actually sent the request, so delayed clients and skipped requests remain visible. The trade condition also bounds backlog at every second, limits end drain, and checks the second half kept pace. The search requires two consecutive 60-second passes at a rate. These targets are specific to this example. The scheduled-time measurement follows the approach described by wrk2’s author. [wrk2 latency methodology](https://github.com/giltene/wrk2)

STAC-M1 follows data from exchange feed to consuming applications. The small-message experiment therefore also times the oldest message in a batch through native subscriber application and the durable publisher reply. Nasdaq’s MoldUDP64 permits several sequenced messages in a packet. The synthetic replay groups messages into deterministic bursts on the load host and records how late they arrive relative to their schedule. The measurement starts in the native feed handler, downstream of exchange decoding and multicast packet recovery. The archive experiment records row count, archive depth, lookup hit/miss latency and trading-day rollover. STAC-M3 covers a much broader historical analytics workload. These are project benchmarks. [STAC-M1](https://docs.stacresearch.com/m1) · [Nasdaq MoldUDP64 specification](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/moldudp64.pdf) · [STAC-M3](https://docs.stacresearch.com/m3)

## Original experiment list

The [five-experiment qualification](#five-experiment-qualification) follows up on rate qualification, latency attribution, the head’s quote path, small-message batching and archive growth. The original list below also includes future work on concurrent fan-out, WAL group commit and scaling across heads.

| Order | Experiment | What decides it |
|---|---|---|
| 1 (complete) | Hold the offer at 3,145,728 trades/s. Compare the throughput configuration (524,288 trades per batch, eight requests in flight) with 262,144 trades and two in flight. Run each with the WAL enabled and disabled. | All six configurations passed the four-batch trade budget. Smaller batches gave the largest reduction in live-quote latency. [Measurements below](#batch-size-and-pipeline-depth) |
| 2 | Add a quote-latency condition to the rate search. Start with 20 reads/s per reader, scheduled p99 ≤50 ms and at most 12 missed slots per reader in a minute. These are proposed interactive targets. Include smaller feed messages and a maximum batch age. | Report the highest offer that passes **both** the trade and quote checks on two consecutive one-minute trials. A zero-sample reader fails. Measure first-trade-arrival to durable acknowledgment, including batch fill. Check uniform, hot-symbol and infrequently traded instruments. |
| 3 | Time the path of one batch. On the head, separate IPC decode, table staging, DuckDB insert/commit, WAL encode/checksum/write/sync and each subscriber acknowledgment. On the load host, separate trade generation from time awaiting a reply. Measure how long an incoming quote waits before its handler starts. | Run paired trials with timing on and off. Record calls, bytes, total time and p99 per stage. Pick the largest contributor to missed quotes at the fixed offer before changing code. |
| 4 | Reduce time a live query waits behind a publish. Test serving a pending quote between committed batches. If query execution itself dominates, test a last-quote-by-symbol index updated on commit and rebuilt on WAL replay. | Lower missed slots and scheduled p99 at the same durable rate. Check late timestamps, sparse symbols, failed writes, restart and the first quote after rollover. Readers must only see committed data. |
| 5 | Reduce fan-out cost. Measure 0, 1, 2 and 4 subscribers. Try encoding an immutable batch once and sending it to subscribers concurrently. Keep the publisher acknowledgment behind the required subscriber receipts. | Measure CPU per million trades, bytes sent, individual subscriber lag and the slowest acknowledgment. Exercise slow, disconnected and reconnecting subscribers and check ordering and watermarks. |
| 6 | Reduce WAL and RDB work separately. If encode/copy dominates, try fewer copies of packed columns and reusable sequence/staging buffers. If sync dominates, try bounded group commit. Account for lost or delayed replies at each flush boundary. | Compare WAL on/off at identical batch size, depth and worker count. Count syncs and bytes per million trades. Kill and restart the head at batch boundaries, verify the acknowledged prefix, then resume and check the archived fields. |
| 7 | Cut historical quote and rollover time. Measure Parquet export, transfer over the separate NFS network, HDB import/commit and head reset separately. Test a per-day last-quote summary, then compare transfer paths if the handoff dominates. | Report historical scheduled p99, missed slots, import time, time until the next day accepts a trade and throughput including rollover. Check absent symbols, old archives, duplicate closes and recovery during the handoff. |
| 8 | Measure scaling and compare with q. Sweep worker counts with CPU affinity controlled. Compare one publisher with additional ordered publishers or disjoint shards as separate workloads. Check NIC counters, retransmits and NUMA placement as CPU or link use rises. Repeat the winning setup against KX q when a matching runner is available. | Publish rows/s and CPU-seconds per million rows alongside durability, subscriber acknowledgment policy, p99 and misses. Qualify horizontal scaling separately from a single ordered head. [Kdb+ comparison context](../../../docs/performance.md#how-this-compares-with-kdbtick) |

## Batch size and pipeline depth

For a durable feed at **3,145,728 trades/s**, use **262,144 trades per batch and at most two in flight** as the starting point for latency tuning. It acknowledged all 188,743,680 trades by the deadline in each measured minute. Scheduled live-quote p99 fell from 130 ms to 70–73 ms. At depth eight, halving the batch alone brought p99 down to 76–77 ms. At the larger batch size, reducing depth alone brought it to about 128 ms.

| WAL | Trades/batch | In flight | Live scheduled p99 | Missed live slots out of 1,200 | Ingest + rollover, M trades/s |
|---|---:|---:|---:|---:|---:|
| Native | 524,288 | 8 | 130.2–130.3 ms | 462–493 | 2.02–2.14 |
| Native | 524,288 | 2 | 127.7–128.1 ms | 447–471 | 2.01–2.02 |
| Native | 262,144 | 8 | 75.7–76.6 ms | 244–249 | 2.01–2.04 |
| Native | 262,144 | 2 | 69.8–72.7 ms | 124–239 | 2.03–2.04 |
| Disabled | 524,288 | 8 | 93.5–98.1 ms | 191–204 | 2.04–2.06 |
| Disabled | 262,144 | 2 | 54.6–56.7 ms | 18–28 | 2.07–2.12 |

These November 26 UTC runs used the same native Linux executable as the original sweep. Each configuration started with an empty HDB, a short seed and one full warm-up day. The table shows the two subsequent one-minute runs. Each reader queried a rotating set of symbols at 20 Hz. All 12 measured sessions verified both subscribers and every archived row. In one large-batch, depth-two WAL run, one batch finished after the deadline. The other 11 sessions acknowledged every trade inside the minute.

Each durable minute had 360 or 720 publish batches, respectively, with a WAL data sync per batch. The smaller configuration still missed 124–239 live quote slots with the WAL and 18–28 without it. Historical quotes missed zero slots across the measured sessions. The original rate search accumulated many more trading days, so its historical p99 describes a different archive. Milan averaged about 0.84 Gbit/s received and 1.68–1.69 Gbit/s transmitted. Those measurements motivated the subsequent fixed-depth grid and head profiling against the 50 ms quote target. [All trial records, deadline counts, scheduled latency and counters](results/2026-11-26-quote-batches-100gbe.jsonl)

## How to compare changes

Keep Milan as head and RAM-only RDB, Tinkara as disk-backed HDB and load host, and two independent native `kg` subscribers. Keep the native publisher and quote readers, trade columns, storage placement, pod limits and query frequency fixed. Change one factor per comparison when attributing a gain. A comparison that changes batch size and pipeline depth together measures the configuration as a whole.

The existing publisher generates whole batches. At a uniform 3,145,728 trades/s, accumulating 524,288 incoming trades would take about 167 ms and accumulating 262,144 would take 83 ms. A small-message replay with timestamps at the feed handler will expose this delay as well as the processing time at the head.

Run one full warm-up day per configuration, followed by two 60-second sessions against equal-sized prior days. Reverse configuration order on a repeat when results are close. Check every archived row, both subscriber cursors and total volume. Use WAL-disabled trials to isolate cost. For durable results, keep sync-before-ack and WAL replay on restart. Report the offered rate, deadline acknowledgments, drain, query round-trip and scheduled p99, missed slots, rollover, host NIC counters and actual CPU use. A percentile over completed reads alone conceals skipped work.

For experiment 1, start the [wide head](README.md#reproduce-the-in-memory-sweep) with a maximum batch of 524,288, a fresh HDB and two subscribers. Set the four endpoint variables to their IPC addresses. Use the same head for both batch sizes, and add `--no-wal` to the head generator only for the WAL-disabled controls.

```sh
python -m examples.stocks.tickerplant.native_bench --kg kg --mode repeat \
  --head "$HEAD_ADDR" --hdb "$HDB_ADDR" --subscriber "$SUB1" --subscriber "$SUB2" \
  --workers 4 --batch 262144 --depth 2 --rate 3145728 \
  --query-hz 20 --confirm-seconds 60 --confirm-runs 3 > trial.jsonl
```

## Five-experiment qualification

Milan and Tinkara each had an AMD EPYC 7773X, 64 cores and 128 hardware threads. The benchmark interface on each host reported 100,000 Mbit/s, MTU 1500. The head used local NVMe for the WAL. Tinkara used local NVMe for the historical DuckDB. A separate NFS mount carried the closing Parquet file. The head, HDB and load pods had CPU limits of 24, 16 and 16 respectively. DuckDB used four workers on the head and eight on the HDB. Two native subscribers acknowledged every batch. Each configuration had fresh database and WAL directories on the same role pods. The timing A/B started with a short seed and then ran two measured sessions. Grid probes accumulated archived days before the long confirmations. Host caches and CPU placement followed the operating system defaults. The query target applies during ingest. Rollover is timed separately because it blocks new head requests.

### Durable rate with two quote readers

The highest offer confirmed on the tested grid was **2,097,152 trades/s**, in 98,304-trade batches with two requests in flight. It passed two one-minute grid confirmations and then passed two more on the final Linux binary in an independent fixture. All four runs verified every archived row and both subscribers. The table gives the independent final pair.

| Final 60-second session | First | Second |
|---|---:|---:|
| Quote schedule offset | 0 ms | 17 ms |
| Trades verified | 125,829,120 | 125,829,120 |
| Trades acknowledged by 60 s | 125,730,816 | 125,829,120 |
| Post-deadline drain | 17.3 ms | 0 ms |
| Live quotes within 50 ms | 1,192 / 1,200 | 1,192 / 1,200 |
| Historical quotes within 50 ms | 1,200 / 1,200 | 1,200 / 1,200 |
| Live / historical scheduled p99 among answered slots | 46.41 / 2.63 ms | 44.92 / 2.33 ms |
| Rollover RPC | 5.86 s | 5.08 s |
| Verified trades/s from offer start through completed rollover | 1,908,440 | 1,932,218 |

In each final session the live reader skipped four slots and answered four more after their deadline. Milan averaged 0.56 Gbit/s received and 1.12 Gbit/s sent, with zero observed NIC drops or errors. The head pod averaged 0.74–0.76 CPU cores in these windows. That average includes time waiting for storage and replies; the topology uses a single ordered head, DuckDB worker threads and separate native client and subscriber processes.

The sweep screened batches of 32,768, 65,536, 98,304, 131,072 and 196,608 rows at depth two, using 262,144-trade/s steps. Offers reached 4,194,304/s for the three largest shapes. A 131,072-row batch passed one minute at 3,145,728/s before the second minute failed the trade rule. A 196,608-row batch passed a 10-second probe at 3,932,160/s; the long run missed both the trade and quote conditions. Higher short-run rates remain in the receipt alongside the failed confirmations. The bound applies to this workload and grid.

Across 21 configurations, 123 sessions verified 4,564,762,624 synthetic rows, including seeds and failed rate attempts. A separate SIGKILL check recovered all 491 acknowledged open-day rows from the local WAL. The restarted head served the restored quotes, accepted 59 more rows, delivered the new rows to both reconnected subscribers, and archived all 550 with their fields intact. [All trial records, hashes, per-stage timings and counters](results/2026-11-26-qualified-100gbe.jsonl)

### Where quote latency goes

At 1,572,864 offered trades/s, 65,536 trades per batch and two requests in flight, the original head spent a median 27.5–28.4 ms inside a publish handler. Median WAL time was 12.3 ms, including roughly 0.48 ms encoding, 0.18 ms checksumming, 3.17 ms writing and 8.3 ms syncing. DuckDB insertion took about 4 ms and commit about 2 ms. Each serial subscriber call took about 3.5–3.6 ms. These are per-stage medians from two 20-second sessions.

| Live quote measurement at that offer | Original | With committed-quote tail |
|---|---:|---:|
| Handler p99, tracing enabled | 4.08–4.33 ms | 0.06–0.07 ms |
| Client dispatch to handler entry, p99 | 30.4–31.8 ms | 29.6–30.7 ms |
| Time in that wait overlapping a publish, p99 | 30.3–31.7 ms | 29.5–30.6 ms |
| Scheduled completion p99, tracing **off** | 35.5–36.7 ms | 32.8–35.0 ms |

The head retains symbols and prices from at most the last 2,048 rows of the last committed durable batch. A hit returns the most recent sequence and price. A miss uses the existing RDB query. Replay rebuilds the tail and rollover clears it. The benchmark rotates across 1,024 symbols, so this particular feed gives the tail a high hit rate. The Klong regression covers sparse and absent symbols, reverse timestamps, invalid publishes, one-row batches, restart and day rollover.

The client used a fixed schedule. Before and after each traced session it made 16 native clock probes to the head and retained the tightest round-trip bounds. The trace reader matches the ordered quote and publish requests across hosts, and uses the head's monotonic clock for handler duration. Dispatch-to-handler includes transport, decode and waiting. The overlapping publish interval is measured separately. Both tracing settings used the same profiling-capable executable and were checked against the same trade and subscriber data.

### Small feed messages and maximum batch age

A native load process generated eight-trade messages in bursts of four, admitted them on an independent schedule, and accumulated at most 1,024 rows per batch. One durable publish could be outstanding. A sealed batch could wait for that receipt before transmission. Each 15-second high-rate session admitted all 61,440 messages, published 491,520 trades, reconciled both subscribers and validated the archive. The publisher recorded when the oldest message was due and when it actually entered each batch. Each subscriber timestamped the applied watermark before replying. All three processes used the same Tinkara host clock for the delivery comparison.

| Offered trades/s | Flush policy | Oldest admission to seal p99 | Oldest admission to application at either subscriber, p99 range | Oldest admission to durable publisher ACK p99 | Oldest scheduled arrival to ACK p99 |
|---:|---|---:|---:|---:|---:|
| 32,768 | 1,024 rows | 31.07–31.08 ms | 40.99–47.29 ms | 43.91–47.43 ms | 44.63–79.23 ms |
| 32,768 | 1,024 rows or 2 ms | 3.00–3.02 ms | 27.94–30.42 ms | 28.56–31.79 ms | 42.93–54.48 ms |
| 4,096 | 1,024 rows or 2 ms | 3.03–3.05 ms | 22.68–22.98 ms | 23.02–23.15 ms | 30.99–31.01 ms |

Ranges cover the two runs, and for subscriber application both native receivers. The 2 ms setting is a flush threshold sampled by a 1 ms timer. At the higher rate, an already sealed batch waited another 14–16 ms at p99 before it could send. Scheduled-to-ACK timing includes generator stalls and backpressure before admission. The size-only run with 79.23 ms scheduled p99 shows how much of that delay can disappear when timing starts at admission. These tests use an in-process synthetic feed handler on Tinkara and real cross-host IPC to Milan.

### Historical queries and rollover

Each configuration imported five 18,874,368-row sessions after a 253,952-row seed. Post-close probes issued 64 present-symbol and 64 absent-symbol lookups per sampled day. At that sample size the nearest-rank p99 is the slowest lookup in each group. With 94,625,792 trades in the archive, the previous implementation's hit p99 was 5.84 ms on day 2 and 5.54 ms on day 5. A per-day closing-quote table brought those times to 0.73 ms and 0.76 ms. Absent-symbol p99 on those days was 1.34/1.25 ms before and 1.21/1.15 ms after. The absent symbol was outside the traded range of 0–1,023. The oldest sampled day was the short seed. These are native IPC round trips on Tinkara after import and validation, with warm filesystem caches.

| Same synthetic trading day | Original handoff and archive | Parquet v2 plus closing-quote table |
|---|---:|---:|
| Head rollover RPC, range across five days | 2.57–2.87 s | 1.45–1.55 s |
| Transfer file per full day | 251.0 MB | 6.45 MB |
| Day 5 Parquet export | 1.36 s | 0.83 s |
| Day 5 HDB seal RPC | 0.85 s | 0.40 s |
| Day 5 head table reset | 0.068 s | 0.068 s |

The new handoff selects DuckDB’s Parquet v2 encoding. Monotonic sequences and patterned integer columns compress particularly well on this generated feed. The HDB inserts raw trades, each symbol’s last quote by sequence, and the closing watermark in one transaction. A retry returns the sealed watermark. Older archives keep their original query path and can be opened directly. The injected index-write failure test checks rollback and retry. The times above compare the two changes together. [DuckDB’s Parquet writer and encodings](https://duckdb.org/2025/01/22/parquet-encodings)

### Reproduce the measurements

The qualification record lists SHA-256 hashes for the seven final harness files. Use the measured revision to reproduce those hashes. [Benchmark source at a17775c](https://github.com/briangu/klongpy/tree/a17775c87ad9b89d807c3a65512c77966141e6e9/examples/stocks/tickerplant)

Use fresh working directories per configuration and preserve them through its consecutive sessions. Start two native subscribers using the [example instructions](README.md). Substitute your four IPC addresses and choose the 524,288-row staging limit in the head renderer. The cumulative sequence watermark, every archived row and both subscribers are checked before advancing the rate.

```sh
python -m examples.stocks.tickerplant.native_bench --kg kg --mode grid \
  --head "$HEAD_ADDR" --hdb "$HDB_ADDR" --subscriber "$SUB1" --subscriber "$SUB2" \
  --workers 4 --batch 65536 --depth 2 --query-hz 20 --quote-slo-ms 50 \
  --start-rate 1310720 --max-rate 3145728 --step 262144 --probe-seconds 10 \
  --confirm-seconds 60 --confirm-runs 2 --confirmation-phase-ms 17
```

The descending grid screens higher offers even when the regular publisher and quote clocks align at some rates. It backs off after a failed short screen or either failed long confirmation. Confirmation readers start at offsets of 0 and 17 ms respectively. The second confirmation also follows a full archived day; differences between the two can include storage and scheduling variability. In repeat mode, use the same four endpoints with the flags below.

| Experiment | Repeat-mode flags |
|---|---|
| Head timing, each variant | `--rate 1572864 --batch 65536 --depth 2 --workers 4 --confirm-seconds 20 --confirm-runs 2 --quote-slo-ms 50 --trace-dir CLIENT_TRACES` |
| Small-message feed | `--rate 32768 --batch 1024 --depth 1 --workers 4 --feed-message-rows 8 --burst-messages 4 --max-age-ms 2 --confirm-seconds 15 --confirm-runs 2 --trace-dir CLIENT_TRACES`. Repeat with age zero for size-only batching. |
| Archive growth | `--rate 1572864 --batch 65536 --depth 2 --workers 4 --confirm-seconds 12 --confirm-runs 5 --archive-probes 64` |

For stage timings, render the head with `python -m examples.stocks.tickerplant.sweep_head --max-batch 524288 --profile`, and place `journal.kg` beside the generated head. The profiled head enables journal timing. The archived Rust implementation used `KG_TICKER_WAL_PROFILE=1`. Save the head’s combined stdout and stderr. Render subscribers with `python -m examples.stocks.tickerplant.sweep_head --profile-subscriber examples/stocks/tickerplant/subscriber.kg` to collect application timestamps. Run the feed and both subscribers on the same load host for that measurement. `python -m examples.stocks.tickerplant.profile_report CLIENT_TRACES/day-1.json head.log` joins the head stages. Supply `--subscriber sub1.log --subscriber sub2.log` in place of the head log for delivery latencies.

In the [saved native traces](results/2026-11-26-stage-traces.tar.gz), `stageon` and `quoteon` contain the original and optimized head runs. `histold` and `archv2` contain the corresponding archive comparisons. The three feed directories include both subscribers’ applied-watermark logs. From the checkout, extract them into an empty directory and run the same report against measured events.

```sh
mkdir -p /tmp/kg-ticker-traces
tar xzf examples/stocks/tickerplant/results/2026-11-26-stage-traces.tar.gz -C /tmp/kg-ticker-traces
python -m examples.stocks.tickerplant.profile_report \
  /tmp/kg-ticker-traces/stageon/day-1.json /tmp/kg-ticker-traces/stageon/head.jsonl
```

For the original scripts, use `git show dfefd68:examples/stocks/tickerplant/head.kg` and the corresponding `hdb.kg`. The head-path A/B retained the original historical server and Parquet format in both arms. The archive A/B retained the committed-quote tail in both arms and changed the historical server and the Parquet writer together.

### Reproduce the Klong journal comparison

The November 27 comparison used the same head, HDB and load placement, CPU limits, NVMe volumes and 100 GbE route as the qualification above. Each service configuration started in a fresh data directory. The head renderer used `--max-batch 524288` with tracing disabled. The three builds used the same Rust 1.92 release toolchain. [Rust journal source](https://github.com/briangu/klongpy/tree/4184b5d) · [Original Klong journal](https://github.com/briangu/klongpy/tree/9644fa6) · [Optimized Klong journal](https://github.com/briangu/klongpy/tree/c80ae65)

For the fixed-rate comparison, use the native harness's `--mode repeat` with `--rate 2097152 --batch 98304 --depth 2 --workers 4 --query-hz 20 --quote-slo-ms 50 --confirmation-phase-ms 17 --backlog-batches 4`. Run once with `--confirm-seconds 12 --confirm-runs 1` and again against the same processes with `--confirm-seconds 60 --confirm-runs 2`. The harness adds a seed day when the archive is empty. Both measured minutes independently check both subscriber cursors and quantities, distinct sequences and all archived payload fields. The tail latency is measured from the scheduled client query. Every skipped slot counts against the target.

For the burst feed, start another fresh configuration. Use `--rate 32768 --batch 1024 --depth 1 --feed-message-rows 8 --burst-messages 4 --max-age-ms 2 --confirm-seconds 15 --confirm-runs 2` with the same quote settings. Provide a fresh `--trace-dir` on the load host. This workload timestamps the oldest scheduled and admitted message in each durable batch.

The direct test takes the module, a fresh directory, rows per batch and number of batches. For example, from the repository root on a host with a local NVMe volume:

```sh
kg native/scripts/bench_ticker_journal.kg \
  "$PWD/examples/stocks/tickerplant/journal.kg" "$FRESH_WAL_DIR" 98304 64
```

Pass `builtin` as the first argument when running the old Rust binary. Each trial reuses four precomputed columns, times each synchronous append, rewinds, checks every replayed column against the original, and reaches EOF. The runs used 128 batches at 8 and 1,024 rows and 64 batches at 98,304 rows. At each size the first series ran Rust, original Klong, original Klong, Rust. A second series ran optimized Klong, original Klong, original Klong, optimized Klong. The daily files from all three writers matched byte for byte at every size.
