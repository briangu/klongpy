# A ticker plant with live and historical queries

The head accepts ordered trade batches over kdb+ IPC, syncs each batch to a write-ahead log (WAL), commits the in-memory real-time database (RDB) transaction, and sends the batch to each subscriber. A separate historical database (HDB) serves completed trading sessions. This example co-locates the ticker plant and RDB as one native `kg` process. KX's reference distribution runs its ticker plant and RDB as separate processes. [KX tick scripts](https://github.com/KxSystems/kdb-tick)

## Run locally

Use the checkout's desktop `kg` with the `duckdb`, `ipc` and `ticker-wal` capabilities (`kg --capabilities`). The native head owns its WAL writer. Set `DATA` to an absolute data directory and `EXAMPLE` to the absolute path of this example.

```sh
mkdir -p "$DATA/transfer"
```

Run each native server in its own terminal with `$DATA` as its working directory.

```sh
kg "$EXAMPLE/hdb.kg" 127.0.0.1:5511
kg "$EXAMPLE/head.kg" 127.0.0.1:5510 127.0.0.1:5511
kg "$EXAMPLE/subscriber.kg" 127.0.0.1:5512 127.0.0.1:5510
```

The servers accept Klong expressions over IPC. Use loopback locally and a private network between trusted clients on a cluster. On separate hosts, mount the same transfer directory on the head and HDB. By default the head creates `log/` in its working directory. Pass a directory as its third argument to place the WAL on a different volume. Keep the journal directory (daily WAL files and the small SQLite session checkpoint) and the HDB's `history.duckdb` on persistent storage. The RDB uses DuckDB in RAM with disk spill disabled.

### Restart and rollover

Restart the head with the same WAL directory. When taking over a directory previously owned by the Python benchmark writer, stop that writer first. Both implementations take the same exclusive checkpoint lock. Before listening for clients the head rebuilds the open session by replaying its WAL in sequence. Replay needs no HDB connection, sends no subscriber callbacks and leaves complete records in place. After reconnecting, a publisher reads `nextSeq` before resending a batch whose reply it lost. Consumers can request missed rows with `ticksFrom` or `historyFrom`.

If a write was interrupted, replay can discard an incomplete final record of the open day. A checksum mismatch or sequence gap stops startup. The journal records the closing watermark before the head exports the day and advances the checkpoint only after the HDB confirms the import. If the head restarts with `closing=1`, bring up the HDB and retry `roll(dayNo)`. The HDB recognizes a day it already committed. Sealed WAL files remain in the journal directory.

**Upgrading an RC1 data directory.** Stop the feed, finish any open day using the old head, and record `dayNo` and `dayFirst` from the now-empty head. Keep the old files. Start the new head in a fresh directory with the same transfer mount and HDB. On the first head launch, append `WAL-DIRECTORY DAY FIRST-SEQUENCE` after the HDB address to initialize the new journal at those coordinates. For a legacy benchmark journal without a session checkpoint, use the day and first sequence following the HDB's last archived trade (`coalesce(max(dayno)+1,0), coalesce(max(seq)+1,0)` from `trades`). The journal can replay complete records from the earlier binary format.

## Run the benchmark

For services configured with an IPC authentication hook for the `bench` user, set `KG_TICKER_BENCH_PASSWORD_FILE` on the load generator to a private file containing that user's password. Both benchmark commands use it for the head, HDB and subscriber monitoring connections. Configure the native head-to-HDB and subscriber-to-head connections with the same services' credentials.

Run the benchmark from a source checkout with the development Python dependencies installed. It writes synthetic trades and closes each measured session, so give it its own data directory.

```sh
python -m examples.stocks.tickerplant.bench --subscriber 127.0.0.1:5512 \
  --batch 8192 --rate 250000 --seconds 20 --sessions 2
```

`--rate 0` saturates one synchronous publisher. `--batch-cycle 3000 8192 6000 4096` alternates feed sizes. Repeat `--subscriber` for each subscriber whose cursor and volume should be verified. One independent IPC connection queries the latest trade for a rotating instrument. Another queries the previous session on the HDB as soon as one exists. The default rate is 20 queries per second on each connection.

### Run a native publisher and quote clients

The [100 GbE qualification](optimization.md#five-experiment-qualification) includes the durable rate with a 50 ms quote target, a small-message replay and matched historical archive results. The [reproduction commands](optimization.md#reproduce-the-measurements) run the same rate grid, timing probes and archive checks.

The standard head, HDB and each subscriber are separate native `kg` processes. The native benchmark also runs the publisher, live quote reader and historical quote reader as independent `kg` processes. The publisher generates the trade columns in Klong and pipelines ordered IPC requests through `.async`. Python's standard library schedules the trials and checks the receipts. All IPC, including the checks against the HDB, goes through `kg`.

With two subscribers running and an empty open day, run the fixed-resource suite from the load host. Use `--head` and `--hdb` for the servers' benchmark interfaces and `--kg` for the candidate binary.

```sh
python -m examples.stocks.tickerplant.native_bench --kg kg --mode fixed \
  --head 127.0.0.1:5510 --hdb 127.0.0.1:5511 \
  --subscriber 127.0.0.1:5512 --subscriber 127.0.0.1:5513
```

This seeds a historical day, offers 750,000 trades/s for one minute, then runs an unpaced minute. Each trading day rolls to the HDB. Post-roll checks cover every trade's sequence, timestamp, instrument, price and quantity, plus each subscriber's cursor, trade count and volume.

For the rate search, use a fresh data directory and generate the wide head as described below. Change `--mode fixed` to `--mode sweep` to test batch sizes, worker counts and pipeline depths, then confirm an offered rate over two one-minute sessions. The native sweep allows a backlog of four scheduled batches by default. Use `--backlog-batches 0` for the strict deadline. `--mode search --batch N --workers W --depth D` runs the same rate search with a fixed configuration. The client clocks ingestion and acknowledgments inside the publisher. Quote timings include IPC round trips, scheduled delays and missed slots. Each JSON line reports the verified row count and the timing for one trial, or summarizes the configuration or search result.

The [native 100 GbE measurements](results/2026-11-26-native-100gbe.jsonl) record the executable and script hashes, hardware and pod limits, per-trial network counters, both deadline policies and WAL crash recovery. The [performance notes](../../../docs/performance.md#native-wal-and-native-clients-on-100-gbe) compare ingest, end-of-day rollover and quote latency.

### Find a sustained rate

To compare two configurations at the same offered rate, use `--mode repeat` with the same server and subscriber arguments as the native benchmark above. For example, add `--rate 3145728 --batch 262144 --depth 2 --workers 4 --confirm-seconds 60 --confirm-runs 3`. Each run closes and verifies a trading day even when its acknowledgments miss the deadline. Start each configuration with fresh data and use the first full day to warm the HDB before comparing the remaining two. The repeat mode keeps the offered rate fixed for every run.

Start the servers above, then run the rate search against a fresh benchmark data directory.

```sh
python -m examples.stocks.tickerplant.ratchet --subscriber 127.0.0.1:5512 \
  --batch 65536 --start-rate 1048576 --max-rate 8388608 --step 131072 \
  --probe-seconds 8 --confirm-seconds 60 --confirm-runs 2
```

The driver doubles the offered rate until it misses or reaches the cap, then narrows the interval to one rate step. Each pass requires acknowledgments for every whole batch that fits within the offer window. A final partial interval can drain for at most one batch interval. Choose rates and durations that offer an exact number of batches to require every trade by the deadline. At the winning rate, consecutive full-length sessions must pass. A missed confirmation triggers another search at lower rates.

The Python publisher in `bench` and `ratchet` accepts `--in-flight N`. The native publisher takes `--depth N` in `--mode search`. These options let one connection send up to N ordered batches while earlier replies are pending. Every batch still needs its own acknowledgment after subscriber delivery. The reported acknowledgment latency includes waiting behind earlier batches.

Use `--backlog-batches N` on the ratchet to measure a pipeline that may carry work across the deadline. At each one-second checkpoint, at most N scheduled batches may be outstanding, including work still waiting to be sent. The same limit applies at the deadline, and the final drain must finish within N batch intervals. In the second half of the window, acknowledged trades must keep up with the whole batches offered in that period, within one batch. Each trial records the cumulative acknowledgments at every second so you can inspect how the backlog changed. The default zero uses the original deadline test.

Every trial rolls a trading day, verifies the archived sequence and payload of every trade, and checks the configured subscribers' cursors and volume. The driver seeds a prior day when needed so both live and historical queries run during each trial. Data or protocol errors abort the search. Size memory for the largest open day and disk for all the archived trials. The JSON lines include latency and missed query slots for each trial. The final line reports the confirmed rate and the lowest higher rate seen to miss. A null miss means the configured cap passed.

### Reproduce the in-memory sweep

From the repo root, generate an experimental head in a fresh data directory. Give the HDB its own empty data directory and make the same `transfer/` directory available to both servers. Use the HDB and subscriber launch commands above and start two subscribers on different ports. Start the generated head in place of `head.kg`.

```sh
python -m examples.stocks.tickerplant.sweep_head --max-batch 524288 > "$DATA/head-sweep.kg"
(cd "$DATA" && kg head-sweep.kg 127.0.0.1:5510 127.0.0.1:5511)
```

This head stages up to 524,288 trades in one insert. The standard head stages 65,536. Both keep the RDB in RAM and use the WAL by default. For the benchmark control, generate the head with `--no-wal` and use a fresh data directory and HDB. That control loses its open day and sequence state on restart.

```sh
python -m examples.stocks.tickerplant.ratchet --sweep \
  --subscriber 127.0.0.1:5512 --subscriber 127.0.0.1:5513 \
  --start-rate 1048576 --max-rate 8388608 --step 131072 \
  --sweep-seconds 5 --probe-seconds 8 --confirm-seconds 60 --confirm-runs 2 \
  --backlog-batches 4 > sweep.jsonl
```

`--sweep` starts with single-request batches of 65,536, 131,072, 262,144 and 524,288 trades. It retests the two fastest sizes with eight and four RDB workers. It then tests two and four outstanding batches, and tries eight at the best pipelined size. The HDB uses eight workers throughout. The driver prints the selected settings and then searches for the confirmed rate. The run history includes failed rates and timing for both quote clients.

The [November 25 measurements](results/2026-11-25-100gbe.jsonl) include the hardware, fixed-resource comparison, configuration probes, every rate-search trial and the recovery check. Each `kind=rate_search` trial reports its offered rate, trades acknowledged by the deadline, backlog, drain time, query latency and verified rows. The final record for each search gives the highest confirmed rate and its next observed miss. The test suite replays the decisions from these records. To reproduce the exact November 25 write and recovery paths, use the [archived benchmark source](https://github.com/briangu/klongpy/tree/ef5fd3d97925842cde6f1bc73087b9e7495f7ffa/examples/stocks/tickerplant).

For the two-host run, place the head and its journal volume on the first host. On the second host, run the HDB in one pod and the publisher and two subscribers in another. Mount the same transfer directory on the head and HDB. Bind the servers to the benchmark NICs and pass their addresses to `--head` and `--hdb`. Leave both subscriber monitoring addresses on publisher loopback and point both subscriber feed connections at the head. On Linux, `ip route get PEER_IP` identifies the outgoing interface, `/sys/class/net/INTERFACE/speed` reports its speed in Mb/s, and the deltas of `statistics/rx_bytes` and `statistics/tx_bytes` under that interface measure host traffic during ingest. [Hardware, resource limits and measured rates](../../../docs/performance.md#finding-the-sustained-rate)

The November 25 journaled benchmark used a separate Python writer on the head host. For that comparison, start `python -m examples.stocks.tickerplant.journal serve --directory "$DATA/log" --port 5520` and add `--journal-port 5520` to the renderer. To enable authentication for this fixture, pass `--password-file` to the writer and `--journal-password-file` to the renderer. Use a private file with an alphanumeric test password and keep the generated script private.

## IPC contract

- `publish(first;ts;sym;px;qty)` takes four equal-length integer vectors containing nanosecond timestamps, instrument IDs, prices in ticks and quantities. The head assigns consecutive sequence numbers starting at `first`. A successful reply gives the next expected sequence. A gap or overlap fails before any row is written. After losing a reply, read `nextSeq` and resume at that offset. Multiple publishers must coordinate their offsets.
- `latest(sym)` returns the last sequence and price in arrival order. The durable head checks the most recent 2,048 rows of its last committed batch in RAM and looks up other symbols in the RDB. A late exchange timestamp keeps its original value. `ticksFrom(first;n)` returns up to 65,536 ordered rows for a live-session replay. Both endpoints return an empty list for no match. A single SQL row returns its fields directly.
- A subscriber calls `subscribe(0)` on its feed connection and receives the first sequence it should expect. The head calls `onTicks(first;ts;sym;px;qty)` back on that connection. Each subscriber returns the new cursor. Fan-out is synchronous and serial. A failed, timed-out or incorrect subscriber acknowledgment removes that subscriber and increments `lostSubs`. A subscriber has two seconds to respond. A reconnecting consumer can request missed rows using the live and historical replay calls.
- `roll(dayNo)` closes a nonempty session and returns its exclusive ending sequence. The head records that it is closing, writes the current session to Parquet in the transfer directory, and asks the HDB to import it. The HDB checks the file's count and sequence bounds, commits the import, and acknowledges. Only then does the head clear the RDB and advance the day. A repeat of the last close returns the same watermark. After a crash in the handoff, retry the close before publishing. Queries and writes on the head can wait during rollover. The HDB also serializes requests while importing a partition.
- `historyQuote(day;sym)`, `historyStats(day)`, and `historyFrom(day;first;n)` query the HDB. The HDB commits archived rows to its own DuckDB file.

The head keeps one database connection throughout the session. The head and HDB each let DuckDB use up to four workers within its detected CPU budget. Reusable Klong tables stage power-of-two batch widths. A partial batch inserts only its valid rows. Inputs above 65,536 rows use multiple blocks inside one transaction. The WAL covers the whole input batch. Subscribers see it after both the WAL sync and the RAM transaction commit.

## Reading a benchmark result

Each JSON line covers one trading session. `rows_s` includes input generation, IPC, the configured commit path and subscriber delivery. `ingest_and_roll_rows_s` includes the handoff to the HDB. The paced run continues until every batch due during the offer window is acknowledged. It records the backlog at the deadline and how long it took to drain. The batch is scheduled as a unit, so the timing excludes a real feed handler's wait to fill that batch.

The driver checks live reads against publisher watermarks and historical reads against the prior session. It reports query round-trip and scheduled latency, including missed query slots. Separate query probes record the longest observed wait during the handoff. After rollover a separate SQL check verifies the count, unique sequence coverage and all four payload fields for every archived row. This check runs outside the throughput timer. [Cluster measurements](../../../docs/performance.md#ticker-plant-throughput-and-rollover)
