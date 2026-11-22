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

Run the benchmark from a source checkout with the development Python dependencies installed. It writes synthetic trades and closes each measured session, so give it its own data directory.

```sh
python -m examples.stocks.tickerplant.bench --subscriber 127.0.0.1:5512 \
  --batch 8192 --rate 250000 --seconds 20 --sessions 2
```

`--rate 0` saturates one synchronous publisher. `--batch-cycle 3000 8192 6000 4096` alternates feed sizes. Repeat `--subscriber` for each subscriber whose cursor and volume should be verified. One independent IPC connection queries the latest trade for a rotating instrument. Another queries the previous session on the HDB as soon as one exists. The default rate is 20 queries per second on each connection.

## IPC contract

- `publish(first;ts;sym;px;qty)` takes four equal-length integer vectors containing nanosecond timestamps, instrument IDs, prices in ticks and quantities. The head assigns consecutive sequence numbers starting at `first`. A successful reply gives the next expected sequence. A gap or overlap fails before any row is written. After losing a reply, read `nextSeq` and resume at that offset. Multiple publishers must coordinate their offsets.
- `latest(sym)` returns the last sequence and price in arrival order. A late exchange timestamp keeps its original value. `ticksFrom(first;n)` returns up to 65,536 ordered rows for a live-session replay. Both endpoints return an empty list for no match. A single SQL row returns its fields directly.
- A subscriber calls `subscribe(0)` on its feed connection and receives the first sequence it should expect. The head calls `onTicks(first;ts;sym;px;qty)` back on that connection. Each subscriber returns the new cursor. Fan-out is synchronous and serial. A failed, timed-out or incorrect subscriber acknowledgment removes that subscriber and increments `lostSubs`. A subscriber has two seconds to respond. A reconnecting consumer can request missed rows using the live and historical replay calls.
- `roll(dayNo)` closes a nonempty session and returns its exclusive ending sequence. The head records that it is closing, writes the current session to Parquet in the transfer directory, and asks the HDB to import it. The HDB checks the file's count and sequence bounds, commits the import, and acknowledges. Only then does the head clear the RDB and advance the day. A repeat of the last close returns the same watermark. After a crash in the handoff, retry the close before publishing. Queries and writes on the head can wait during rollover. The HDB also serializes requests while importing a partition.
- `historyQuote(day;sym)`, `historyStats(day)`, and `historyFrom(day;first;n)` query the HDB. The RDB and HDB use DuckDB's normal commit durability. Replication and automatic failover are outside this example.

The head keeps one database connection throughout the session. Reusable Klong tables stage power-of-two batch widths. A partial batch inserts only its valid rows. Inputs above 65,536 rows use multiple blocks inside one transaction. The committed RDB is the recovery log. A subscriber acknowledgment or the loss of one never removes a committed trade.

## Reading a benchmark result

Each JSON line covers one trading session. `rows_s` includes input generation, IPC, durable commit and subscriber delivery. `ingest_and_roll_rows_s` includes the handoff to the HDB. The paced run continues until every batch due during the offer window is acknowledged. It records the backlog at the deadline and how long it took to drain. The batch is scheduled as a unit, so the timing excludes a real feed handler's wait to fill that batch.

The driver checks live reads against publisher watermarks and historical reads against the prior session. It reports query round-trip and scheduled latency, including missed query slots. Separate query probes record the longest observed wait during the handoff. After rollover a separate SQL check verifies the count, unique sequence coverage and all four payload fields for every archived row. This check runs outside the throughput timer. [Cluster measurements](../../../docs/performance.md#ticker-plant-throughput-and-rollover)
