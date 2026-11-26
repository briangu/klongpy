# A ticker plant with live and historical queries

The head accepts ordered trade batches over kdb+ IPC, commits them to the real-time database (RDB), and sends the committed batch to each subscribed client. A separate historical database (HDB) serves completed trading sessions. This example co-locates the ticker plant and RDB as one native `kg` process. KX's reference distribution runs its ticker plant and RDB as separate processes. [KX tick scripts](https://github.com/KxSystems/kdb-tick)

## Run locally

Use the desktop `kg` with DuckDB and IPC. Create a data directory containing a `transfer/` subdirectory. From the data directory, run these commands in separate terminals in the order shown. Set `EXAMPLE` to the absolute path of this example in each terminal.

```sh
kg "$EXAMPLE/hdb.kg" 127.0.0.1:5511
kg "$EXAMPLE/head.kg" 127.0.0.1:5510 127.0.0.1:5511
kg "$EXAMPLE/subscriber.kg" 127.0.0.1:5512 127.0.0.1:5510
```

The servers accept Klong expressions over IPC. Use loopback locally and a private network between trusted clients on a cluster. On separate hosts, mount the same transfer directory on the head and HDB. Each process can keep its own DuckDB file on local storage. Keep both files on persistent volumes when they must survive node or pod replacement.

For services configured with an IPC authentication hook for the `bench` user, set `KG_TICKER_BENCH_PASSWORD_FILE` on the load generator to a private file containing that user's password. Both benchmark commands use it for the head, HDB and subscriber monitoring connections. Configure the native head-to-HDB and subscriber-to-head connections with the same services' credentials.

Run the benchmark from a source checkout with the development Python dependencies installed. It writes synthetic trades and closes each measured session, so give it its own data directory.

```sh
python -m examples.stocks.tickerplant.bench --subscriber 127.0.0.1:5512 \
  --batch 8192 --rate 250000 --seconds 20 --sessions 2
```

`--rate 0` saturates one synchronous publisher. `--batch-cycle 3000 8192 6000 4096` alternates feed sizes. Repeat `--subscriber` for each subscriber whose cursor and volume should be verified. One independent IPC connection queries the latest trade for a rotating instrument. Another queries the previous session on the HDB as soon as one exists. The default rate is 20 queries per second on each connection.

### Find a sustained rate

Start the servers above, then run the rate search against a fresh benchmark data directory.

```sh
python -m examples.stocks.tickerplant.ratchet --subscriber 127.0.0.1:5512 \
  --batch 65536 --start-rate 1048576 --max-rate 8388608 --step 131072 \
  --probe-seconds 8 --confirm-seconds 60 --confirm-runs 2
```

The driver doubles the offered rate until it misses or reaches the cap, then narrows the interval to one rate step. Each pass requires acknowledgments for every whole batch that fits within the offer window. A final partial interval can drain for at most one batch interval. Choose rates and durations that offer an exact number of batches to require every trade by the deadline. At the winning rate, consecutive full-length sessions must pass. A missed confirmation triggers another search at lower rates.

Both benchmark commands accept `--in-flight N`. The default publisher waits for each acknowledgment. Increasing this value lets one connection send up to N ordered batches while earlier replies are pending. Every batch still needs its own acknowledgment after subscriber delivery. The reported acknowledgment latency includes waiting behind earlier batches.

Use `--backlog-batches N` on the ratchet to measure a pipeline that may carry work across the deadline. At each one-second checkpoint, at most N scheduled batches may be outstanding, including work still waiting to be sent. The same limit applies at the deadline, and the final drain must finish within N batch intervals. In the second half of the window, acknowledged trades must keep up with the whole batches offered in that period, within one batch. Each trial records the cumulative acknowledgments at every second so you can inspect how the backlog changed. The default zero uses the original deadline test.

Every trial rolls a trading day, verifies the archived sequence and payload of every trade, and checks the configured subscribers' cursors and volume. The driver seeds a prior day when needed so both live and historical queries run during each trial. Data or protocol errors abort the search. Size memory for the largest open day and disk for all the archived trials. The JSON lines include latency and missed query slots for each trial. The final line reports the confirmed rate and the lowest higher rate seen to miss. A null miss means the configured cap passed.

### Reproduce the in-memory sweep

From the repo root, generate an experimental head in a fresh data directory. Give the HDB its own empty data directory and make the same `transfer/` directory available to both servers. Use the HDB and subscriber launch commands above and start two subscribers on different ports. Start the generated head in place of `head.kg`.

```sh
python -m examples.stocks.tickerplant.sweep_head --memory --max-batch 524288 > "$DATA/head-sweep.kg"
(cd "$DATA" && kg head-sweep.kg 127.0.0.1:5510 127.0.0.1:5511)
```

This head keeps the open trading day in DuckDB memory, disables disk spill, and can stage 524,288 trades in one insert. The disk-backed example uses 65,536. A RAM-only run loses the open day if the head restarts.

```sh
python -m examples.stocks.tickerplant.ratchet --sweep \
  --subscriber 127.0.0.1:5512 --subscriber 127.0.0.1:5513 \
  --start-rate 1048576 --max-rate 8388608 --step 131072 \
  --sweep-seconds 5 --probe-seconds 8 --confirm-seconds 60 --confirm-runs 2 \
  --backlog-batches 4 > sweep.jsonl
```

`--sweep` starts with single-request batches of 65,536, 131,072, 262,144 and 524,288 trades. It retests the two fastest sizes with eight and four RDB workers. It then tests two and four outstanding batches, and tries eight at the best pipelined size. The HDB uses eight workers throughout. The driver prints the selected settings and then searches for the confirmed rate. The run history includes failed rates and timing for both quote clients.

The [November 25 measurements](results/2026-11-25-100gbe.jsonl) include the hardware, fixed-resource comparison, configuration probes, every rate-search trial and the recovery check. Each `kind=rate_search` trial reports its offered rate, trades acknowledged by the deadline, backlog, drain time, query latency and verified rows. The final record for each search gives the highest confirmed rate and its next observed miss. The test suite replays the decisions from these records.

For the two-host run, place the head and optional journal on the first host. On the second host, run the HDB in one pod and the publisher and two subscribers in another. Mount the same transfer directory on the head and HDB. Bind the servers to the benchmark NICs and pass their addresses to `--head` and `--hdb`. Leave both subscriber monitoring addresses on publisher loopback and point both subscriber feed connections at the head. On Linux, `ip route get PEER_IP` identifies the outgoing interface, `/sys/class/net/INTERFACE/speed` reports its speed in Mb/s, and the deltas of `statistics/rx_bytes` and `statistics/tx_bytes` under that interface measure host traffic during ingest. [Hardware, resource limits and measured rates](../../../docs/performance.md#finding-the-sustained-rate)

For a head with a synchronous journal, start the journal on the head host and add `--journal-port 5520` when rendering the head. `serve` listens on loopback and uses POSIX file sync. Each record covers one batch and the head waits for its disk acknowledgment before sending the trades to the subscribers. On a shared host, pass the same private `--password-file` to the writer and `--journal-password-file` to the renderer. Use a file containing an alphanumeric test password and keep the generated head private.

```sh
python -m examples.stocks.tickerplant.sweep_journal serve --directory "$DATA/log" --port 5520
```

To replay an open day, restart the journal and the generated head with the same log directory and HDB. Before connecting the feed or subscribers, run the command below. It reads the HDB's last archived watermark, verifies the open-day records and restores their original sequences to the RDB. Stop the fixture if it reports a journal or replay error. The process-restart test retains the local disk throughout.

```sh
python -m examples.stocks.tickerplant.sweep_journal replay --directory "$DATA/log"
```

## IPC contract

- `publish(first;ts;sym;px;qty)` takes four equal-length integer vectors containing nanosecond timestamps, instrument IDs, prices in ticks and quantities. The head assigns consecutive sequence numbers starting at `first`. A successful reply gives the next expected sequence. A gap or overlap fails before any row is written. After losing a reply, read `nextSeq` and resume at that offset. Multiple publishers must coordinate their offsets.
- `latest(sym)` returns the last sequence and price in arrival order. A late exchange timestamp keeps its original value. `ticksFrom(first;n)` returns up to 65,536 ordered rows for a live-session replay. Both endpoints return an empty list for no match. A single SQL row returns its fields directly.
- A subscriber calls `subscribe(0)` on its feed connection and receives the first sequence it should expect. The head calls `onTicks(first;ts;sym;px;qty)` back on that connection. Each subscriber returns the new cursor. Fan-out is synchronous and serial. A failed, timed-out or incorrect subscriber acknowledgment removes that subscriber and increments `lostSubs`. A subscriber has two seconds to respond. A reconnecting consumer can request missed rows using the live and historical replay calls.
- `roll(dayNo)` closes a nonempty session and returns its exclusive ending sequence. The head records that it is closing, writes the current session to Parquet in the transfer directory, and asks the HDB to import it. The HDB checks the file's count and sequence bounds, commits the import, and acknowledges. Only then does the head clear the RDB and advance the day. A repeat of the last close returns the same watermark. After a crash in the handoff, retry the close before publishing. Queries and writes on the head can wait during rollover. The HDB also serializes requests while importing a partition.
- `historyQuote(day;sym)`, `historyStats(day)`, and `historyFrom(day;first;n)` query the HDB. The RDB and HDB use DuckDB's normal commit durability. Replication and automatic failover are outside this example.

The head keeps one database connection throughout the session. The head and HDB each let DuckDB use up to four workers within its detected CPU budget. Reusable Klong tables stage power-of-two batch widths. A partial batch inserts only its valid rows. Inputs above 65,536 rows use multiple blocks inside one transaction. The committed RDB is the recovery log. A subscriber acknowledgment or the loss of one never removes a committed trade.

## Reading a benchmark result

Each JSON line covers one trading session. `rows_s` includes input generation, IPC, durable commit and subscriber delivery. `ingest_and_roll_rows_s` includes the handoff to the HDB. The paced run continues until every batch due during the offer window is acknowledged. It records the backlog at the deadline and how long it took to drain. The batch is scheduled as a unit, so the timing excludes a real feed handler's wait to fill that batch.

The driver checks live reads against publisher watermarks and historical reads against the prior session. It reports query round-trip and scheduled latency, including missed query slots. Separate query probes record the longest observed wait during the handoff. After rollover a separate SQL check verifies the count, unique sequence coverage and all four payload fields for every archived row. This check runs outside the throughput timer. [Cluster measurements](../../../docs/performance.md#ticker-plant-throughput-and-rollover)
