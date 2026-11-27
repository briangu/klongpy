# Performance

The release comparisons below describe their workload, baseline, candidate and machine. The older array microbenchmarks that follow describe a separate development build.

## Native kg and NumPy on resident CPU arrays

The November 21 development build ran 16 array workloads on an Apple M2 Ultra with 192 GiB, using standalone `kg` and NumPy 2.5.5 with Apple Accelerate. Each workload uses matching contiguous 64-bit inputs. The harness checks every output value, shape and inferred numeric dtype against NumPy before timing. Both engines warm up and calibrate a batch of at least 5 ms. The table reports medians of seven batches for 4,096 and one million elements and five batches for eight million. Timing covers the operation on resident data, including result allocation. Input construction, interpreter startup and JSON transfer happen outside the timer. These workloads cover NumPy operations also exercised by the upstream benchmark suite. [NumPy reductions](https://github.com/numpy/numpy/blob/main/benchmarks/benchmarks/bench_reduce.py) · [NumPy sorting workloads](https://github.com/numpy/numpy/blob/main/benchmarks/benchmarks/bench_function_base.py)

The ratio divides NumPy time by `kg` time. A value above one favors `kg`. Matrices have 64 columns. The wide keys form a deterministic strided permutation at the reported sizes. Sorted runs contain 64 copies of each key. Dot uses `numpy.dot`, including its BLAS implementation, and `+/a*b` in Klong.

| Operation | kg at 1M (µs) | NumPy at 1M (µs) | NumPy/kg at 4,096 | At 1M | At 8M |
|---|---:|---:|---:|---:|---:|
| float64 add | 282.9 | 259.2 | 1.19× | 0.92× | 1.13× |
| float64 multiply | 265.4 | 264.5 | 1.11× | 1.00× | 0.97× |
| float64 sum | 201.8 | 177.8 | 1.44× | 0.88× | 1.24× |
| float64 min | 158.8 | 92.3 | 1.24× | 0.58× | 0.97× |
| float64 dot | 450.9 | 110.0 | 0.52× | 0.24× | 0.63× |
| int64 cumulative sum | 375.1 | 2,032.6 | 4.81× | 5.42× | 6.58× |
| float64 first differences | 477.8 | 182.8 | 1.08× | 0.38× | 0.57× |
| float64 gather | 1,596.8 | 920.8 | 0.53× | 0.58× | 1.35× |
| int64 positive indices | 716.0 | 787.5 | 1.48× | 1.10× | 1.00× |
| int64 stable argsort, 257 keys | 1,372.0 | 78,441.0 | 21.74× | 57.17× | 79.08× |
| int64 stable argsort, wide keys | 27,157.0 | 70,399.2 | 2.12× | 2.59× | 3.21× |
| int64 stable argsort, presorted | 299.9 | 851.3 | 3.25× | 2.84× | 2.71× |
| int64 stable argsort, sorted runs | 285.0 | 849.1 | 3.48× | 2.98× | 3.12× |
| int64 stable argsort, uniform | 290.1 | 836.5 | 3.31× | 2.88× | 2.86× |
| int64 column sums | 169.3 | 481.7 | 3.78× | 2.85× | 2.75× |
| int64 row sums | 100.3 | 173.8 | 3.31× | 1.73× | 1.12× |

The new stable grade counts keys when the integer span is small relative to the vector, and returns row positions directly for ascending inputs. On one million signed integers cycling through 257 keys, `kg` went from 28.32 ms to 1.37 ms. The equivalent stable NumPy argsort took 78.44 ms. With the order of the engine timing runs reversed, the ratio was again 57×.

Key width matters. A separate run timed `np.argsort(keys.astype(np.int16), kind="stable")`, including the conversion from int64. On the same patterned keys, NumPy took 5.07 ms and `kg` took 1.49 ms, a 3.41× advantage for `kg`. On one million randomly drawn keys in the same range, NumPy took 5.34 ms with conversion and 98.33 ms on int64. The native grade took 1.50 ms. All three implementations returned the same million indices for each distribution.

Reductions over packed integer matrices now visit the rows directly. At one million elements, column sums went from 2.12 ms to 0.169 ms and row sums from 1.18 ms to 0.100 ms. NumPy retained the lead on the float64 dot product and first differences at both large sizes.

To reproduce from the checkout, build the release CLI and run the following commands. The thread settings must be in the environment before starting Python.

```sh
cargo build --locked --release --manifest-path native/Cargo.toml -p klong-cli
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python native/scripts/bench_numpy.py --json numpy-comparison.json
.venv/bin/python native/scripts/bench_numpy.py --sizes 8000000 --samples 5 --json numpy-large.json
```

## NYC taxi and One Billion Row Challenge

The post-RC1 development build ran the four NYC taxi reports on **1,128,213,169 real trips** on an Apple M2 Ultra. The input was 96 local TLC Parquet files totalling 21.51 GB. It included yellow cabs from January 2009 through June 2015 and green cabs from January 2014 through June 2015. The five published 2013 green links returned HTTP 403. This monthly TLC snapshot differs from the original benchmark author's historical export. [NYC TLC files](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) · [Original benchmark](https://tech.marksblogg.com/duckdb-1b-taxi-rides.html)

The original four queries scan the source for each report. The example reads the required columns together and aggregates trips by cab type, passenger count, pickup year and rounded distance. It builds the four reports from those group counts and fare totals. A separate PyArrow and NumPy scan verified all group keys and counts, row order and the fare averages to floating-point tolerance. The distance report returned 6,806 groups.

With eight DuckDB threads, the original four queries took **19.51 s** and the optimized example took **11.24 s**, a **1.74× speedup**. Times are medians of three paired warm-cache runs in alternating order, including process startup, reading Parquet, aggregation and writing all four JSON results. Median peak resident memory rose from 2.09 GiB to 2.18 GiB.

The weather example ran on **1,000,000,000 measurements** generated by the upstream Java program. With the default thread settings, the median was **4.78 s** for standalone `kg` and **4.89 s** for `kgpy` over three fresh processes each. Both outputs matched the upstream Java reference byte for byte across all 413 stations. The example parses and groups in DuckDB and computes the mean using exact integer tenths. The previous Python worker pipeline lost results across workers, so it provides no valid timing baseline. [One Billion Row Challenge](https://github.com/gunnarmorling/1brc)

Two measurements isolate work in the Klong runtime. Reading 2,964,624 January 2024 yellow-cab fares into one Klong integer vector and summing the cents fell from **134.3 ms to 58.6 ms** across nine paired process runs. Median peak resident memory fell from **140 MiB to 95 MiB**. Matching numeric SQL columns now copy directly into Klong's packed vectors until a null requires a mixed list. Separately, grouping 200,000 resident labels drawn from 413 station strings fell from **161 ms to 4.7 ms**. That operator test exercises the dictionary index used by Klong's `=` on labels. The weather example uses SQL for its grouping.

### Comparisons on the same M2 Ultra

On the same 96 taxi files, `kg` took **11.31 s** and DuckDB 2.0.0 through Python took **11.41 s** to execute the same SQL and print all four reports. In a second paired series, `kg` took **11.25 s** and Polars 2.1.0 took **7.79 s**. Both used eight threads and aggregated once before producing the reports. The Polars streaming plan selected the year directly from historical ISO date strings. The Klong example cast those fields to timestamps. All timings are medians of three fresh-process runs after a warm-up. The independent checker verified the results of every run.

On the same billion weather measurements, the comparison run took **5.22 s** with `kg`, **5.67 s** with DuckDB through Python, and **1.05 s** with the winning Java team's specialized parser running on OpenJDK 27. All three used the machine's default 24 logical CPUs and produced identical output. The official competition measured a GraalVM native binary on eight AMD EPYC cores. [Winning Java source](https://github.com/gunnarmorling/1brc/blob/db064194be375edc02d6dbcd21268ad40f7e2869/src/main/java/dev/morling/onebrc/CalculateAverage_thomaswue.java) · [Competition hardware and timing rules](https://github.com/gunnarmorling/1brc#evaluating-results)

| Runtime operation | `kg` | Comparison |
|---|---:|---:|
| Group 200,000 strings across 413 station names | 4.73 ms | 185.52 ms in PeachQ 0.92 |
| Group 200,000 symbols across 265 zone names | 2.13 ms | 2.21 ms in PeachQ 0.92 |
| Read and sum 2.96 million fares, including startup | 57.4 ms | 133.3 ms in DuckDB and NumPy |
| Read and sum the same fares in an open session | 13.7 ms | 15.1 ms in DuckDB and NumPy |

The grouping timer starts after constructing the input and stops before checking every group's row positions. The fresh-process fare timings include imports and connection setup. The open-session figures are medians of 20 scans and host-side reductions per implementation, with an untimed warm-up in each process.

## Ticker plant throughput and rollover

On November 21, 2026, the native ticker plant sustained **750,150–750,235 synthetic trades/s** across three one-minute runs. Including end-of-session rollover, the three rates ranged from **662,072 to 664,659 trades/s**. In each run, the publisher sent batches of 32,768 trades over kdb+ IPC. The head committed each batch to the real-time database (RDB) and waited for both subscribers to acknowledge it. A separate historical database (HDB) served the previous session while the head ingested the current one.

The benchmark ran on three Linux x86-64 Kubernetes nodes, with one pod per node. Kubernetes limited **each pod to four CPUs and 8 GiB of RAM**.

| Pod role | Host processor | Physical cores / hardware threads on host |
|---|---|---:|
| Ticker plant head and RDB | AMD EPYC 7773X | 64 / 128 |
| HDB | Intel Core i9-11900K | 8 / 16 |
| Publisher, two subscribers and both query clients | Intel Core i7-11700K | 8 / 16 |

Both databases used node-local NVMe storage with ext4. The head transferred closed sessions to the HDB as Parquet files over NFS. The servers ran the native `kg` executable from the November 19 RC1 Linux wheel (SHA-256 prefix `c8855a024400`) with the new Klong example programs.

The feed had 1,024 instrument IDs, integer prices, quantities and nanosecond timestamps. One ordered publisher waited for each batch acknowledgment. Both subscribers checked sequence continuity, returned their processed watermark and had their total volume checked. Independent IPC clients requested the latest live and historical prices at 20 queries per second each. After each rollover the driver checked every stored payload field, distinct sequence coverage and row count.

| Two-subscriber workload | Batch size | Ingest rate | Rate including rollover |
|---|---:|---:|---:|
| Alternating feed sizes, two six-second runs | 3,000 / 8,192 / 6,000 / 4,096 | 446k–456k trades/s | 393k–406k trades/s |
| Saturated publisher, two eight-second runs | 65,536 | 1.42M–1.57M trades/s | 1.12M–1.20M trades/s |
| Paced at 750k/s for one minute | 32,768 | 750k trades/s | 662k–665k trades/s |
| Offered 1.6M/s for 15 seconds, including drain | 65,536 | 1.54M trades/s | 1.21M trades/s |

The variable-size feed initially managed 74k–79k trades/s because the head reopened its database when the batch width changed. Reusing power-of-two staging buffers on one connection made the median rate 5.9× higher. A partial batch writes only valid rows. Larger messages can span multiple buffers inside a single transaction.

Each one-minute run verified 45,023,232 trades. Ingest time includes generation, IPC, database commits and subscriber acknowledgments through the final batch. Dividing the verified count by ingest time plus rollover time gives the last column. The independent check of archived rows ran outside the timer.

| One-minute session | Ingest (s) | Rollover (s) | Ingest (trades/s) | Including rollover (trades/s) |
|---|---:|---:|---:|---:|
| 33 | 60.019 | 7.896 | 750,150 | 662,931 |
| 39 | 60.014 | 7.725 | 750,216 | 664,659 |
| 41 (queries also measured during rollover) | 60.012 | 7.991 | 750,235 | 662,072 |

In session 41, the final batch drained 12 ms after the offer window. Batch acknowledgment p99 was 108.8 ms. During ingestion, live-query round-trip p99 was 97.4 ms and historical-query p99 was 20.4 ms. The live client missed 101 scheduled query slots. The historical client missed zero. During rollover, a live query waited 7.98 s and a historical query waited 3.06 s.

After the load runs, the publisher left 1,048,576 acknowledged trades in an open session. Both database processes were killed. The recovered head returned the same watermark and rolled the open session. After another forced restart, the verifier checked every payload field and sequence for **352,108,704 trades across 43 sessions**. This tests process death with the same storage available.

At an offered 1.6M trades/s, 1,179,648 trades were still awaiting acknowledgment at the end of the 15-second window. Draining the queue took another 0.59 s and the HDB handoff took 4.28 s. The measured arrival schedule begins when a batch is due. A live feed must also budget for the time it takes to fill the batch. [Example and commands](https://github.com/briangu/klongpy/tree/main/examples/stocks/tickerplant)

### Milan and Tinkara over 100 GbE

On November 25, the same Linux RC1 executable and workload ran on two AMD EPYC 7773X hosts. Milan ran the head. Tinkara ran the HDB and load generator in separate pods, with both subscribers and query clients on the load generator. Each pod retained its four-CPU and 8 GiB limits. The publisher and both subscriber streams used the hosts' 100 GbE interfaces. Historical queries stayed on Tinkara. Both databases used local NVMe, and the Parquet handoff used the NFS server on the 10 GbE network. The control runs changed the IPC addresses to the same hosts' 10 GbE interfaces.

| Two-subscriber workload | 100 GbE ingest | 10 GbE ingest | 100 GbE including rollover | 10 GbE including rollover |
|---|---:|---:|---:|---:|
| 750k trades/s offered for 60 s, batches of 32,768 | 750k/s | 750k/s | 660k–666k/s | 666k/s |
| Unpaced for 8 s, batches of 65,536 (median of three runs per link) | 1.412M/s | 1.359M/s | 1.111M/s | 1.075M/s |
| Unpaced for 60 s, batches of 65,536 (one run per link) | 1.372M/s | 1.441M/s | 1.096M/s | 1.131M/s |

The unpaced 100 GbE minute verified 82,313,216 trades. Live-query round-trip p99 was 110.2 ms, with 206 missed query slots. Historical-query p99 was 10.2 ms with none missed. During 54 interior one-second samples, Milan's 100 GbE interface averaged 0.37 Gbit/s received and 0.73 Gbit/s transmitted. Those are host interface counters. The faster link won the short-run median, and the 10 GbE control won the one-minute pair.

### Tuning durable commits and rollover

On the same 100 GbE route, a 12-second profile split 247 batches into explicit transactions. The head spent 7.16 s in DuckDB's `COMMIT`, 0.88 s executing the inserts and 0.12 s in `BEGIN`. A separate 12-second run timed 1.59 s delivering batches to the two subscribers. This led to a change in the ticker plant example: both databases can now use up to four DuckDB workers within the engine's detected CPU budget. The RC1 example used one worker on the head and two on the HDB. The comparison used standard disk sync, the default 16 MiB checkpoint threshold and the same RC1 executable for both configurations.

At **750,000 trades/s offered**, each configuration verified 45,023,232 rows in one minute. With the higher worker limits, batch acknowledgment p99 fell from **103.3 to 63.7 ms**, live-query p99 from **83.0 to 53.3 ms**, and missed live-query slots from 53 to 14. Rollover fell from **7.99 to 5.88 s**. Throughput including rollover rose from **662,103 to 683,376 trades/s**. This fixed-work comparison used one run per configuration and 32,768 trades per batch.

The unpaced comparison used three paired one-minute trials with 65,536 trades per batch and alternated which configuration ran first. The HDB accumulated sessions throughout. The figures below are medians across runs, including medians of the individual run p99s.

| Unpaced workload | RC1 worker limits | Up to four workers per database |
|---|---:|---:|
| Ingest rate | 1.712M trades/s | 1.633M trades/s |
| Rate including rollover | 1.264M trades/s | 1.318M trades/s |
| Batch acknowledgment p99 | 116.0 ms | 88.9 ms |
| Live-query round-trip p99 | 106.8 ms | 75.5 ms |
| Historical-query round-trip p99 | 34.7 ms | 35.9 ms |

### In-memory intraday trades on 100 GbE

On November 25, Milan hosted the ticker plant and Tinkara hosted two subscribers, the publisher, both quote clients and a separate disk-backed HDB. The head and HDB each used four DuckDB workers. All three pods had four CPUs. The head had 16 GiB of memory and each Tinkara pod had 8 GiB. Each configuration started with a fresh trading history, seeded a day, ran a minute at 750,000 trades/s with 32,768 trades per batch, then ran an unpaced minute with 65,536 trades per batch. The two RAM configurations attached DuckDB's `:memory:` database and disabled temporary-file spill. The disk control used local NVMe. All three used the same RC1 Linux executable, synthetic trades, subscriber checks, concurrent quote queries and Parquet handoff to the HDB.

| One unpaced minute, one run per configuration | Disk RDB | RAM RDB + synchronous journal | RAM RDB, WAL disabled |
|---|---:|---:|---:|
| Verified trades | 87,162,880 | 104,726,528 | 182,517,760 |
| Ingest, million trades/s | 1.452 | 1.745 | 3.042 |
| Including rollover, million trades/s | 1.189 | 1.388 | 2.024 |
| Rollover | 13.24 s | 15.43 s | 30.17 s |
| Batch acknowledgment p99 | 89.5 ms | 54.4 ms | 33.7 ms |
| Live quote p99 | 77.8 ms | 39.3 ms | 25.6 ms |
| Historical quote p99 | 7.5 ms | 7.9 ms | 9.4 ms |

At the fixed 750,000/s offer, each run verified 45,023,232 trades. Live-quote p99 was 65.0 ms on disk, 31.3 ms with the journal, and 14.6 ms with the WAL disabled. Rollover took 6.27, 6.36 and 5.61 seconds, respectively. The original 8 GiB head could hold the paced session. In its unpaced run with the WAL disabled, DuckDB reached its 6.3 GiB allocation limit after 155,648,000 acknowledgments and rejected the next batch. The table uses the reruns with equal headroom across storage configurations.

The November 25 benchmark ran its Python journal writer as a separate process on Milan. For every batch it appended the sequence watermark and four integer columns to a checksummed daily file, called `fdatasync` on local NVMe, and replied before the head forwarded the batch to the subscribers. In a separate recovery check, both head and journal processes were killed after acknowledging three batches. Replaying the journal into an empty RDB restored all 491 trades and their fields. The restarted head accepted 59 further trades and the HDB verified the complete 550-row session. This checks process recovery on the same disk. With the WAL disabled, the head must stay up through rollover to retain the open day. The tagged RC1 example used the disk RDB. In the current source example, the native head writes the WAL in-process and replays it into a RAM RDB before serving. The November 25 rates below describe the earlier writer. [Startup and recovery](https://github.com/briangu/klongpy/tree/main/examples/stocks/tickerplant#restart-and-rollover)

#### Finding the sustained rate

The rate search kept the same two hosts, one native head, two synchronous subscribers and both query clients. It raised the head pod limits to 24 CPUs and 40 GiB, the HDB to 16 CPUs and 24 GiB, and the load pod to 16 CPUs and 16 GiB. The experimental head could stage a single batch of up to 524,288 trades. Five-second probes varied batch size, DuckDB worker count and the number of ordered batches awaiting acknowledgment on one publisher connection. Every trial closed its session and verified its HDB data. [Load driver and rate-search commands](https://github.com/briangu/klongpy/tree/main/examples/stocks/tickerplant)

In trials with the WAL disabled, 524,288 trades per batch, eight DuckDB workers on each database and up to four batches in flight, the eight-second probe passed at 6,553,600 offered trades/s with a four-batch backlog allowance. The full minute at the same rate fell 160 batches behind and took another 18.76 seconds to drain. A separate strict-deadline search confirmed 3,538,944 trades/s in two minutes run as separate trading days. All 212,336,640 trades per day were acknowledged before the deadline. Live quotes missed 313 and 227 scheduled slots during ingest, and rollover took 33.05 and 35.32 seconds.

With at most four outstanding scheduled batches allowed at each one-second checkpoint, the search with the WAL disabled confirmed **5,242,880 offered trades/s** twice. Each run verified **314,572,800 trades**. The first had one batch outstanding at the deadline and drained it in 49 ms. The second acknowledged every trade inside the minute. At the next step, **5,373,952/s**, the head caught up by the deadline but reached six outstanding batches during ingest. Across the two passing runs, Milan's 100 GbE host interface averaged about **1.40 Gbit/s received and 2.81 Gbit/s transmitted** during ingest.

The journaled sweep selected 262,144 trades per batch, 8 RDB workers and up to 4 batches in flight. It confirmed **3,014,656 offered trades/s** twice. The next rate seen to miss was **3,145,728/s**. The four-batch budget and second-half progress check applied to both configurations. Each quote client targeted 20 reads/s. The HDB accumulated trading days throughout each search.

| Two one-minute confirmations | RAM RDB + synchronous journal | RAM RDB, WAL disabled |
|---|---:|---:|
| Offered trades/s | 3,014,656 | 5,242,880 |
| Verified trades per minute | 180,879,360 | 314,572,800 |
| Trades acknowledged by the deadline | 180,879,360 | 314,048,512–314,572,800 |
| Drain after the deadline | 0.0 ms | 0.0–49.2 ms |
| Rate including rollover | 1.986–2.050M trades/s | 2.640–2.731M trades/s |
| Rollover | 28.23–31.07 s | 55.21–59.09 s |
| Batch acknowledgment p99 | 114.8–118.4 ms | 159.7–161.7 ms |
| Live quote p99 | 75.0–77.4 ms | 63.1–74.3 ms |
| Historical quote p99 | 65.9–71.4 ms | 171.7–177.1 ms |
| Missed live quote slots | 244–253 | 86–174 |
| Missed historical quote slots | 273–364 | 828–838 |

A passing rate qualifies the trade stream under the stated backlog budget. The quote clients skipped slots when a preceding request was still in flight. Every completed quote and all archived trade fields were checked. The [machine-readable measurements](https://github.com/briangu/klongpy/blob/main/examples/stocks/tickerplant/results/2026-11-25-100gbe.jsonl) include the failed attempts, one-second progress in the bounded-backlog searches and each run’s query timings.

### Native WAL and native clients on 100 GbE

On November 26 UTC (November 25 in California), Milan ran one native `kg` head with a RAM-only DuckDB RDB. Its in-process WAL synced each batch on local NVMe before subscriber delivery. Tinkara ran the publisher, two subscribers, two quote readers and the disk-backed HDB as six separate native `kg` processes. Both subscribers checked the incoming sequence and returned their cursors before the publisher received its acknowledgment. The Python benchmark harness launched the clients, collected timings and verified their output. Every IPC request and reply used native `kg`. This run used a Linux build from the source checkout, SHA-256 `e172c476fcc9…`.

The fixed-resource tests used the same four-CPU pod limits and memory budgets as the previous in-memory comparison. Each workload ran for one minute with four DuckDB workers per database and one publisher batch in flight. Both quote readers targeted 20 queries/s. The HDB verified every archived trade after rollover.

| Workload | WAL | Verified trades | Ingest trades/s | Including rollover | Live / historical query p99 |
|---|---|---:|---:|---:|---:|
| 750,000/s offered | Native | 45,023,232 | 750,309 | 686,966/s | 25.8 / 4.1 ms |
| Unpaced | Native | 106,954,752 | 1,781,461 | 1,409,253/s | 40.0 / 5.6 ms |
| 750,000/s offered | Disabled | 45,023,232 | 750,364 | 682,691/s | 13.2 / 4.1 ms |
| Unpaced | Disabled | 199,688,192 | 3,327,752 | 2,151,627/s | 21.5 / 5.9 ms |

The expanded native-WAL sweep selected 524,288 trades per batch, four RDB workers, eight HDB workers and up to eight batches in flight. It confirmed **3,801,088 offered trades/s** in two one-minute runs, 26% above the earlier 3,014,656/s result on the same hosts. The WAL implementation, publisher, quote readers and selected batch settings all changed between runs. Each native-WAL run verified 228,065,280 trades. At the deadline, 227,540,992 trades had been acknowledged. The last batch drained in 13–36 ms. Milan’s 100 GbE interface averaged 1.02 Gbit/s received and 2.04 Gbit/s transmitted during ingest, with zero observed drops or errors.

Each expanded search selected its own batch size, worker count and pipeline depth. The four-batch budget checks the backlog at every one-second checkpoint and bounds the final drain. The end-of-minute rule requires every whole batch due by 60 seconds to have been acknowledged. All winning rates below scheduled an exact number of batches. The HDB started fresh in each search and accumulated the earlier trials.

| WAL | Deadline rule | Confirmed offered trades/s | Next observed miss | Including rollover, M trades/s | Live quote p99 | Historical quote p99 |
|---|---|---:|---:|---:|---:|---:|
| Native | Four-batch budget | 3,801,088 | 3,932,160 | 2.26–2.27 | 124.7–126.2 ms | 89.8–99.5 ms |
| Disabled | Four-batch budget | 4,980,736 | 5,111,808 | 2.58–2.60 | 83.0–84.4 ms | 174.4–189.7 ms |
| Native | End-of-minute | 3,145,728 | 3,276,800 | 1.98–2.03 | 126.6–132.5 ms | 82.8–85.1 ms |
| Disabled | End-of-minute | 3,407,872 | 3,538,944 | 2.17–2.23 | 49.9–59.4 ms | 158.7–159.1 ms |

Both WAL-disabled confirmations under the four-batch rule acknowledged all 298,844,160 trades by the deadline. In the corresponding native-WAL confirmations, the live reader missed 509 and 530 scheduled slots and the historical reader missed 601 and 603. With the WAL disabled, the live reader missed 286 and 341 slots and the historical reader missed 821 and 868. Each reader targeted 1,200 slots per minute. Query p99 covers completed calls.

A separate recovery trial killed the native head with `SIGKILL` after 491 acknowledged trades. On restart it replayed all 491 from its WAL into an empty RDB. Two subscribers connected at the restored cursor, the publisher added 59 trades, and the HDB verified all 550 after rollover. This exercised process loss with the local WAL disk intact. Across the fixed workloads and all four searches, the HDB verified 13,652,017,152 synthetic trades in 149 sessions, including seeds and failed rate attempts. The separate recovery check verified another 550 rows. [Trial records, hardware, executable hashes and counters](https://github.com/briangu/klongpy/blob/main/examples/stocks/tickerplant/results/2026-11-26-native-100gbe.jsonl)

#### Durable ingest with an interactive quote budget

These measurements used the Rust journal at the [benchmark revision](https://github.com/briangu/klongpy/tree/a17775c/examples/stocks/tickerplant). The [Klong journal comparison](#klong-journal-on-100-gbe) repeats the qualified offer with the same on-disk format.

A post-RC1 source-build sweep on Milan and Tinkara required at least 99% of each reader’s 20 scheduled quotes/s to return within 50 ms. It counted skipped slots and measured latency from the scheduled time. A qualifying rate had to pass that condition on both live and historical reads, keep trade backlog within the four-batch budget, and pass twice for 60 seconds. Both native subscribers acknowledged the durable stream. The head used four DuckDB workers and the HDB used eight.

**2,097,152 offered trades/s** was the highest rate confirmed on the tested grid. A final independent pair used 98,304-row batches with two requests in flight and verified 125,829,120 trades in each minute. In each run, 1,192 of 1,200 live quotes and all 1,200 historical quotes finished within 50 ms. Live scheduled p99 was 44.92–46.41 ms and historical p99 was 2.33–2.63 ms. One batch drained 17.3 ms after the first deadline. All trades were acknowledged within the second deadline. Through completed rollover, the rates were 1.91 and 1.93 million trades/s. Milan averaged 0.56 Gbit/s received and 1.12 Gbit/s sent. [Qualification records and hardware](https://github.com/briangu/klongpy/blob/main/examples/stocks/tickerplant/results/2026-11-26-qualified-100gbe.jsonl)

In a separate comparison at a fixed 1,572,864 trades/s, the committed-quote tail reduced live handler p99 from 4.08–4.33 ms to 0.06–0.07 ms. With tracing off, scheduled client p99 fell from 35.5–36.7 ms to 32.8–35.0 ms. The head still spent a median 8.3 ms in the NVMe data sync per 65,536-trade batch. At a fixed 32,768 synthetic trades/s in eight-trade messages, adding a 2 ms batch-age threshold cut oldest admitted message to durable acknowledgment p99 from 43.91–47.43 ms to 28.56–31.79 ms. At an archive size of 94.6 million synthetic trades, historical latest-price p99 on the most recent day fell from 5.54 ms to 0.76 ms. Each full 18.9-million-row day closed in 1.45–1.55 s with Parquet v2 and a transactional closing-quote table, compared with 2.57–2.87 s before. [Methods and reproduction](https://github.com/briangu/klongpy/blob/main/examples/stocks/tickerplant/optimization.md#five-experiment-qualification)

#### Klong journal on 100 GbE

On November 27, 2026, each of three journal builds sustained **2,097,152 offered synthetic trades/s** on the same two AMD EPYC 7773X hosts. Milan synced every batch to local NVMe and updated the in-memory DuckDB. Two native subscribers on Tinkara acknowledged delivery. The publisher kept at most two 98,304-trade batches outstanding. Live and historical readers each scheduled 20 queries/s, and each had to finish at least 99% of its slots within 50 ms. All six one-minute runs met the quote and four-batch backlog budgets. Each verified 125,829,120 archived rows and every trade field after rollover. The comparison used a single offered rate.

| Journal | Live slots within 50 ms, of 1,200 per run | Live scheduled p99 | Historical scheduled p99 | Offer through completed rollover |
|---|---:|---:|---:|---:|
| Rust control | 1,200 / 1,198 | 44.1–45.5 ms | 2.2–2.7 ms | 1.935–1.936 M trades/s |
| Original Klong | 1,200 / 1,198 | 42.6–44.0 ms | 2.2–2.6 ms | 1.937–1.939 M trades/s |
| Optimized Klong | 1,198 / 1,198 | 44.5–45.1 ms | 2.2–2.4 ms | 1.932–1.938 M trades/s |

The two runs started their quote schedules at offsets of 0 and 17 ms. Scheduled p99 covers answered slots. Skipped slots count against the 99% budget. The optimized build answered all 2,400 historical slots on time. Its publisher acknowledgment p99 was 60.2–64.8 ms, measured from batch submission. Both 60-second runs drained within 11.5 ms of the deadline, and rollover took 4.88–5.08 s. The head’s 100 GbE interface averaged 0.56 Gbit/s received and 1.12 Gbit/s transmitted, with zero recorded drops or errors.

**Append and replay.** A separate test on Milan excluded IPC, DuckDB and subscribers. It timed a synchronous append of four precomputed integer columns, then read every batch back from the warm file cache and compared every field. At 98,304 rows per batch, the daily file held 6,291,456 rows in 64 records. All three writers produced identical bytes.

| Journal | Median durable append across trials | Verified replay |
|---|---:|---:|
| Rust control | 14.65–14.81 ms | 47.0–57.2 M rows/s |
| Original Klong | 17.83–18.16 ms | 19.5–20.3 M rows/s |
| Optimized Klong | 14.08–14.30 ms | 77.9–128.5 M rows/s |

The Klong journal now reads all four columns in one operation. On little-endian hosts, native-width binary reads can fill the integer vector directly, and an exact reshape of an unshared temporary can reuse its storage. On append it gets the row count from the original timestamp column, saving one row copy. A regression test reads and reshapes one million 64-bit integers under a constrained heap budget. At eight rows per batch, optimized Klong replayed 96–97 thousand rows/s versus 134–139 thousand for the Rust control.

**Burst feed.** On fresh fixtures, four eight-trade messages arrived per scheduled burst at 32,768 trades/s. The publisher used a 1,024-row cap, a 2 ms age threshold and one outstanding batch. Each 15-second trial verified 491,520 trades. All 300 scheduled live reads and all 300 historical reads met the 50 ms deadline in each optimized trial. Latencies below describe the oldest message in each batch, including its wait to form a batch and for earlier writes to clear.

| Journal | Scheduled arrival to durable publisher reply, p99 | Admitted arrival to durable publisher reply, p99 |
|---|---:|---:|
| Rust control | 33.2–33.4 ms | 24.0–24.1 ms |
| Optimized Klong | 33.8–33.9 ms | 24.7 ms |

**Process restart.** A separate head opened a copy of the same 25,165,824-row probe WAL, appended one sentinel trade and acknowledged it. Two successive trials killed the head with SIGKILL and restarted it against its exclusive checkpoint. Timing ran from the kill to a client receiving the correct next sequence and sentinel quote. Each restart restored all 25,165,825 rows. The OS page cache remained warm throughout this process-crash test.

- Rust control returned the verified quote in 2.19 and 2.18 seconds.
- Original Klong returned the verified quote in 2.25 and 2.32 seconds.
- Optimized Klong returned the verified quote in 2.14 and 2.18 seconds.

The measurement file includes every seed, probe and measured session, the direct append samples, replay timings, binary hashes, CPU limits and host NIC counters. [Measurement records](https://github.com/briangu/klongpy/blob/main/examples/stocks/tickerplant/results/2026-11-27-klong-journal-100gbe.jsonl) · [Commands and methodology](https://github.com/briangu/klongpy/blob/main/examples/stocks/tickerplant/optimization.md#reproduce-the-klong-journal-comparison)

#### How this compares with kdb+tick

The reference kdb+tick ticker plant implements the journal policy in q. With logging enabled, `tick.q` opens a daily file and writes each `upd` call and its arguments through a binary file handle. The RDB replays the log with the runtime's `-11!` primitive. In real-time mode the ticker plant publishes before logging. The Klong example implements framing, checksums, sequence validation, replay and rollover in `journal.kg`, using runtime primitives for binary I/O, CRC32 and SQLite. Its publisher acknowledgment follows a disk sync, the RDB commit and both subscribers' replies in this benchmark. [KX ticker plant](https://github.com/KxSystems/kdb-tick/blob/85c08ff192b0a103b323246c7300a37919be6159/tick.q) · [KX RDB replay](https://github.com/KxSystems/kdb-tick/blob/85c08ff192b0a103b323246c7300a37919be6159/tick/r.q)

KX gives a sizing figure of up to 4.5 million streaming records/s per core and up to 10 million records/s with batching. Its reference ticker plant publishes updates asynchronously. Our 3.80M trades/s measurement includes a local WAL sync and acknowledgment by two subscribers, with four DuckDB workers configured for the RDB. We have yet to measure KX q on Milan and Tinkara with the same durability, subscriber, query and rollover requirements. Our direct comparison with PeachQ below uses a separate single-threaded workload. [KX sizing guidance](https://code.kx.com/q/wp/disaster-recovery/) · [KX subscriber implementation](https://github.com/KxSystems/kdb-tick/blob/master/tick/u.q)

### FPGA analytics on committed trades

The 100 GbE journal benchmark above ran on CPUs. In a separate experiment on November 27, 2026, an Alveo U250 replayed a price history and returned a summary for each trading strategy. All 20 physical-board regression tests passed. The journal benchmark measures how quickly a publisher can get a durable acknowledgment. The U250 experiment measures what an analytics consumer can compute from a batch it already has. [Board tests and timing records](https://github.com/briangu/klongpy/blob/0c46f01/examples/hardware/results/2026-11-27-u250.json)

| Workload on the physical U250 | FPGA timing | CPU timing on the card host |
|---|---|---|
| Replay 10 million ticks for 1,024 crossover strategies | 1.46–1.49 s with history in card memory. 3.42–3.65 s from process launch through exit. | 11.19–11.30 s for the same emitted C across 32 processes |
| Filter 100,000 blocks of 256 samples with a 16-tap FIR | 106 ms in the kernel. 2.23 s for the full call over SSH. | 0.95 s in NumPy |

The strategy sweep uploaded 40 MB once, ran 32 groups on 32 physical lanes and returned 24 KiB. All 3,072 results for profit, maximum drawdown and trade count matched the C reference bit for bit. The FIR filter returned 241 samples per block, or 96.4 MB in total. Moving those results back across the Gen3 ×1 PCIe link and the SSH connection made the remote call slower than NumPy. [Reproduce the strategy sweep](hardware.md#reusing-one-history-across-1024-strategies) · [FIR transfer timings](hardware.md#streaming-fir-on-the-u250)

**A trade bar is a candidate for subscriber offload.** The head already sequences and journals each batch, and the publisher waits for the subscribers’ replies. A subscriber could queue committed trades for a separate analytics worker and return its cursor before the FPGA finishes. The worker would need its own last-processed sequence so it could rebuild unfinished bars from the journal after a restart. If its queue filled, backpressure could still delay the publisher. Measuring commit-to-bar freshness alongside publisher acknowledgment latency would show the cost of that integration.

The prototype takes ordered price and quantity pairs for one symbol and interval. It returns open, high, low, close, total quantity, the sum of price times quantity, and trade count. Vitis C simulation matched an independent integer reference on all seven values across 200,003 trades. The core scheduled one update per clock, but each trade needs two 64-bit input words and the port reads one word per clock. At the 250 MHz target that gives an estimated input ceiling of 125 million trades/s before transfer overhead. Physical throughput for this kernel and its effect on live ingest remain unmeasured. A separate 64-lane strategy design also reached C synthesis. The strategy timings above come from the 32-lane board image. [Trade-bar experiment](hardware.md#a-trade-bar-in-c-simulation-and-synthesis) · [64-lane synthesis results](hardware.md#known-limits)

## Streaming tables and queries

On 2026-11-20 UTC, standalone `kg` from the installed **2.0.0rc1** Linux x86-64 wheel ingested **0.959–0.977 million rows/s** over six controlled trials on an AMD EPYC 7773X. A loopback client loaded 1,048,576 rows into memory in 256-row batches. In a separate phase, a writer offered 25,600 rows/s against 32,768 preloaded rows while two clients queried last price and as-of bid. Each trial checked the query results and final row counts. Query latency runs from the scheduled time to the reply.

The reference was PeachQ 0.92. Both servers used one query-engine thread. Each series contains three paired `kg`/PeachQ trials, alternating which server ran first. In the controlled series, each server ran on CPU 8 and the clients used CPUs 9–12 on the same socket. All processes shared a six-CPU container quota. The source archive SHA-256 starts with `547dd7b9e927`, the Linux wheel's with `9b6fcffb0e85`, the standalone executable's with `c8855a024400`, and the PeachQ executable's with `69a0befb1abc`.

| `kg` / PeachQ 0.92 | Controlled series 1 | Controlled series 2 | Acceptance bound |
|---|---:|---:|---:|
| Ingestion throughput | 1.644× | 1.672× | ≥0.80× |
| Last-price p99 | 0.378× | 0.391× | ≤1.25× |
| As-of bid p99 | 0.416× | 0.366× | ≤1.25× |

Each table entry is the median of the three paired ratios. Across the six controlled trials, PeachQ ingested 0.572–1.198 million rows/s. Individual throughput ratios ranged from 0.816× to 1.679×.

### PeachQ comparison at default CPU placement

| Unrestricted series | Ingestion throughput ratio | Last-price p99 ratio | As-of bid p99 ratio |
|---|---:|---:|---:|
| First series | 1.706× | 0.697× | 1.043× |
| First repeat | 1.689× | 0.776× | 1.069× |
| Second repeat | 1.737× | 0.712× | 1.243× |

Each current series met the median bounds. In the second repeat, one pair exceeded the 1.25× as-of bound at 1.353×. Across all nine pairs at default placement, the highest absolute query p99 on either engine was 17.09 ms, and the slowest batch acknowledgement was 22.87 ms.

Before the idle SQL change, four unrestricted series using standalone executable `6306836e445e` recorded median as-of p99 ratios of 1.471×, 1.433×, 1.074× and 1.342×. A separate instrumented run of that executable recorded 1.455×. The first last-price query occupied 8.89–11.45 ms of the 9.40–12.05 ms that the first as-of query waited between send and SQL entry. Linux charged the query thread 18.44–20.85 ms of CPU time across the two SQL calls, at roughly 1 ms resolution. Those calls include both publishing newly appended rows for SQL and executing the queries.

From a source checkout with the candidate installed and the PeachQ executable available, run the reference selection with this command.

```sh
KLONG_V2_RELEASE=1 KLONG_NATIVE=/path/to/kg KLONG_TEST_PEACHQ=/path/to/q \
  python -m pytest tests/v2 -m v2_reference -o junit_family=xunit1 --junitxml=reference.xml
```

### Hosted and standalone feed at default CPU placement

With default CPU placement, the installed RC1 wheel passed the 25,600-row/s qualification feed on both commands while two clients queried last price and as-of bid. The regression budget is 100 ms for each query p99 and for the slowest batch acknowledgement, measured from its scheduled time. Ingestion capacity comes from the separate phase without readers.

| Command | Ingestion capacity | Worst ingest lag | Last-price p99 | As-of bid p99 |
|---|---:|---:|---:|---:|
| `kg` | 948,377 rows/s | 15.76 ms | 9.35 ms | 14.49 ms |
| `kgpy` | 148,809 rows/s | 76.24 ms | 23.22 ms | 19.60 ms |

In a historical set of 12 runs before the idle SQL change, we tested each of three hosted wheels and PeachQ once per round over three rounds, changing the order between rounds. Those measurements used default CPU placement and the same feed.

| Build | Median ingestion capacity | Worst ingest lag across runs | Last-price p99 range | As-of bid p99 range | Runs meeting the 100 ms budget |
|---|---:|---:|---:|---:|---:|
| Development `kgpy`, before shared-value ownership | 189,381 rows/s | 27.82–34.16 ms | 16.29–22.98 ms | 16.36–22.99 ms | 3/3 |
| First RC1 candidate `kgpy` | 146,725 rows/s | 37.05–108.10 ms | 17.47–18.59 ms | 23.42–25.27 ms | 2/3 |
| Later RC1 candidate `kgpy` | 149,502 rows/s | 46.61–63.49 ms | 15.28–17.76 ms | 24.15–26.70 ms | 3/3 |
| PeachQ 0.92 | 531,142 rows/s | 11.82–14.67 ms | 12.66–12.97 ms | 12.70–12.88 ms | 3/3 |

The later RC1 candidate ingested about 21% fewer rows per second than the development wheel at the median paired ratio. Relative to the first RC1 candidate, it had lower last-price p99 and higher as-of p99 in every round. The first RC1 candidate missed the ingest budget once in this series. In an earlier three-pair comparison it passed in all three runs, with 146,000 rows/s median and worst ingest lags of 25.61–90.59 ms. The development wheel reached 191,000 rows/s and 16.94–25.33 ms in that comparison. The SHA-256 prefixes for the development, first RC1 and later RC1 wheels are `263fbeabd43e`, `ac24a06efc25` and `b4b3bba8600c`.

An earlier v2 build reached 228 ms worst hosted ingest lag on this workload. The first three runs after the live SQL scan change recorded 24.28, 23.95 and 33.62 ms.

```sh
KLONG_NATIVE=/path/to/kg KLONG_V2_KGPY=/path/to/kgpy \
  python -m pytest tests/v2/test_performance.py::test_tick_service_under_load \
  -o junit_family=xunit1 --junitxml=tick-feed.xml
```

## Distinct IDs and group counts

A before-and-after comparison on Linux x86-64 used standalone release builds from the same v2 development line, immediately before and after the dense-integer optimization. On an AMD EPYC 7773X, the loopback client requested distinct IDs and group counts from one million resident int64 rows with 64 densely spaced IDs. Input construction ran before timing. Each operation had a warm-up, then seven timed requests whose results were checked.

| Operation | Before, median | After, median | Median paired speedup | Range across five pairs |
|---|---:|---:|---:|---:|
| Distinct IDs (`?ids`) | 29.732 ms | 2.800 ms | 10.65× | 10.37–10.97× |
| Group counts (`#'=ids`) | 30.394 ms | 11.988 ms | 2.54× | 2.52–2.85× |

The build order alternated over five paired trials. The time columns are the medians of their trial medians. A control with 64 widely spaced IDs took 29.709 ms before and 29.824 ms after for distinct IDs. The installed RC1 Linux wheel ran the same 64-dense-ID case in **2.43 ms for distinct IDs and 11.86 ms for group counts** (medians of seven calls).

Run the resident 64-ID workload against each binary separately:

```sh
KLONG_NATIVE=/path/to/kg python -m pytest \
  tests/v2/test_performance.py::test_group_resident_integer_ids \
  -o junit_family=xunit1 --junitxml=group-ids.xml
```

## Peak heap for an integer running sum

On an Apple M2 Ultra, standalone `kg` used **8.02 MB of peak heap** to generate one million consecutive int64 values, compute their running sum and read its last value. The preceding implementation used **16.02 MB**. The new scan reuses the temporary input buffer. A vector held in a variable or through an alias retains its original contents.

Both builds returned `499999500000`. The new build also completed with a 10 MiB interpreter heap limit. The earlier build reported an out-of-memory error at that limit. These figures come from `kg --stats`, which measures the interpreter heap. In nine alternating local trial pairs, the median runtime was about 0.49 ms on both builds.

```sh
kg --stats --heap-limit 10485760 -e '(+\!1000000)@999999'
```

## Python-hosted runtime compared with v1

On 2026-11-20 UTC, we evaluated the same expressions through `KlongInterpreter` on pure-Python 1.0.0 and the installed 2.0.0rc1 Linux x86-64 wheel. The benchmark used CPU 8 of an AMD EPYC 7773X. Both builds used Python 3.11.2, NumPy 2.4.6 and one numeric library thread. Three paired trials alternated the build order, using a fresh process for each side. Each case had three warm-up calls and nine measured calls. Timing covered the interpreter call and the creation of its result. Setup, validation and process startup ran outside the clock. Every result was checked.

| Workload | v1 median | RC1 median | Paired speedup | Range across three pairs |
|---|---:|---:|---:|---:|
| Recursive `fib(20)` | 2106.115 ms | 3.274 ms | 643.302× | 627.105–649.182× |
| While loop, 100,000 steps | 5720.051 ms | 13.635 ms | 419.515× | 405.980–423.309× |
| Lambda reduction, 100,000 integers | 848.544 ms | 8.217 ms | 103.377× | 101.647–104.239× |
| Element-wise arithmetic, 1,000,000 integers | 2.172 ms | 1.631 ms | 1.331× | 1.327–1.437× |
| Sum, 1,000,000 integers | 0.406 ms | 0.279 ms | 1.467× | 1.457–1.477× |
| Running sum (scan), 1,000,000 integers | 0.753 ms | 1.019 ms | 0.740× | 0.734–0.832× |
| Sort, 100,000 seeded floats | 16.483 ms | 2.658 ms | 6.201× | 6.185–6.357× |
| Group counts, 1,000,000 Python-supplied integers with 64 IDs | 35.394 ms | 48.632 ms | 0.728× | 0.725–0.744× |

Time columns give the median of three trial medians for each build. Speedup is the median paired v1/RC1 elapsed-time ratio. The running sum and group counts on IDs supplied as a NumPy array were slower on RC1 in all three pairs. The standalone group-count benchmark operates on resident Klong vectors. The v1 commit begins `9315318cbb03`, the RC1 source archive SHA-256 begins `547dd7b9e927`, the wheel's begins `9b6fcffb0e85`, and its native Python extension's begins `bb2649021452`.

The running sum returns a writable million-element NumPy array to Python. An installed v2 wheel from before the temporary-buffer change measured 3.686 ms on this workload. A later RC1 candidate (`b4b3bba8600c`) measured 0.662 ms. The current wheel measured 1.019 ms. The native heap test above checks buffer reuse independently of the hosted result timing.

From the checkout, run the same workload with the v2 wheel installed in the active Python environment. The output includes all timing samples.

```sh
python scripts/bench_versions.py --v1-source /path/to/v1-source \
  --v2-source "$(python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')" --out v1-v2.json
```

## First launch under Rosetta

On Apple Silicon running macOS 26.6.2, the original Intel Mac RC1 wheel took 10.904 seconds to print `kg --help` after a fresh installation. That exceeded the existing ten-second timeout for automated clients. The release wheel limits C++ inlining in DuckDB and folds identical functions at link time. The package shrank from 37.4 MB to 33.0 MB.

We compared the executables on fresh file copies, alternating the original, release, release and original builds for each workload. Help took 10.55–10.75 seconds on the original and 8.80–8.82 seconds on the release wheel. A Klong script that created a three-row table and queried its sum took 10.41–10.60 seconds on the original and 8.78–8.84 seconds on the release wheel. Three subsequent help calls on the same release executable took 66–70 ms.

We also timed one-thread DuckDB queries on two million resident rows. Across three alternating pairs, the release wheel's median time ratios to the original were 0.990× for an integer sum, 0.975× for counts across 64 groups and 0.977× for a join. Each process took nine checked measurements after warm-up. An alternative build optimized for size started in 5.35 seconds, but those same warm queries took 19–35% longer. These measurements compare Intel binaries under Rosetta. The tested release wheel's SHA-256 starts with `f84de715b399`.

## Earlier CPU microbenchmarks

These figures are for the dated development build in the conditions table. Whole-array NumPy arithmetic, reduction and sorting often approached direct NumPy on 100,000 to a million elements. Dispatch cost was more visible in loops and per-element lambdas. Use the workloads and sample sizes below when comparing your own measurements.

## Conditions

| Item | Value |
|---|---|
| Date | 2026-11-18 |
| Machine | AMD Ryzen 9 7950X3D, 16 cores (32 threads), 62 GB of memory |
| Operating system | Ubuntu 20.04.6 LTS, Linux 5.4 |
| Python | 3.12.16 |
| NumPy | 2.5.5 (OpenBLAS) |
| PyTorch | 2.15.0, CPU build; 16 threads by default |
| Native runtime | the `--release` build of `native/` (rustc 1.100.0), run as `kg --bench 1 -e PROGRAM` |
| GPU | none measured |
| Load | idle (load average below 0.1) before the run |

Method: each row runs 3 untimed warm-up runs, then 9 timed runs. The tables show the best run and the median, as `best / median`.

* **Single thread.** The process is pinned to one core and sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` to 1, and torch to one thread. These are the main tables.
* **Default threads.** The same rows with no pinning and the default thread count of NumPy and torch. The native runtime has one thread and is not repeated.

Repeatability: the whole benchmark was run twice, and the tables show the first run. In the first table the best times of the two runs agree within 8%, except one CPython row (17%). In the single-thread NumPy, torch and baseline rows of the array tables, 91 of 96 rows agree within 10%. The default-thread rows and the native runtime's array rows vary more (up to a factor of 1.7 for the native runtime), so ratios that use them are good to about 1.5 times.

In the vector, reduce and sort tables, rows of 100,000 elements or fewer are timed in batches of about 20 ms so that the timer resolution does not matter. The script is `scripts/bench_performance.py`; the last section shows how to rerun it.

## Programs the interpreter runs

These programs spend their time in the interpreter, not in an array library. The thread count does not change them, so the single-thread figures are shown. The CPython column is the same work written in plain Python: `fib` recursion, a `while` loop, `functools.reduce` with a lambda, `itertools.accumulate`, a list comprehension, a dictionary and `str.join`.

| Program | KlongPy, NumPy backend | KlongPy, torch backend | Native runtime | CPython |
|---|---|---|---|---|
| recursive fib(24) | 1.28 s / 1.32 s | 2.67 s / 2.69 s | 20.4 ms / 20.6 ms | 3.06 ms / 3.09 ms |
| while loop to 100000 | 577 ms / 578 ms | 1.24 s / 1.26 s | 21.1 ms / 21.4 ms | 1.41 ms / 1.42 ms |
| `{x+y}/!100000` | 319 ms / 319 ms | 526 ms / 529 ms | 11.4 ms / 11.4 ms | 3.17 ms / 3.19 ms |
| `{x+y}\!10000` | 32.7 ms / 32.8 ms | 55.7 ms / 56.5 ms | 1.21 ms / 1.24 ms | 413 us / 423 us |
| `#{x*x}'!100000` | 309 ms / 311 ms | 558 ms / 566 ms | 11.8 ms / 12 ms | 2.05 ms / 2.05 ms |
| `+/{x*x}'!1000000` | 3.17 s / 3.19 s | 5.51 s / 5.54 s | 117 ms / 119 ms | 28 ms / 28.1 ms |
| append in a loop, 2000 | 11.6 ms / 11.6 ms | 18 ms / 18.1 ms | 557 us / 572 us | 1.71 ms / 1.73 ms |
| 50000 dictionary inserts and lookups | 473 ms / 475 ms | 962 ms / 987 ms | 18.7 ms / 19.4 ms | 2.97 ms / 2.98 ms |
| 20000 string keys | 225 ms / 226 ms | 392 ms / 398 ms | 9.19 ms / 9.23 ms | 1.76 ms / 1.79 ms |
| format and join 20000 numbers | 19.3 ms / 19.4 ms | 48.4 ms / 48.9 ms | 1.88 ms / 1.97 ms | 806 us / 832 us |
| find in a 60000 character text | 13.7 ms / 13.9 ms | 16.2 ms / 16.3 ms | 551 us / 585 us | 81.9 us / 83.1 us |
| index a non-ASCII string, 2000 | 7.03 ms / 7.06 ms | 8.8 ms / 9.01 ms | 253 us / 272 us | 35.6 us / 35.6 us |

What the table shows:

Each ratio compares the best times of two columns.

* KlongPy with the NumPy backend against CPython: it takes 7 to 420 times as long over all twelve rows. Recursion and the `while` loop take 410 to 420 times as long, lambdas, dictionaries, finding text and indexing a non-ASCII string 79 to 200 times, formatting numbers 24 times and appending 6.8 times.
* The native runtime against KlongPy with the NumPy backend: it is 24 to 63 times faster on recursion, loops, lambdas, dictionaries, finding text and indexing a non-ASCII string, 21 times faster on appending and 10 times faster on formatting numbers.
* The native runtime against CPython: it takes 2.3 to 15 times as long on every row except appending, where it is 3.1 times faster.
* KlongPy with the torch backend against the NumPy backend: it takes 1.2 to 2.5 times as long on all twelve rows.

## Vector arithmetic

The expression is `2*1+!N`: a range, an addition and a multiplication, each over the whole array.

Single thread:

| N | NumPy | KlongPy, NumPy backend | KlongPy, torch backend | Native runtime |
|---|---|---|---|---|
| 1,000 | 1.56 us / 1.56 us | 4.85 us / 4.86 us | 7.76 us / 7.77 us | 22.6 us / 24.9 us |
| 10,000 | 4.94 us / 4.95 us | 8.24 us / 8.25 us | 15 us / 15 us | 106 us / 108 us |
| 100,000 | 37.2 us / 37.2 us | 47.8 us / 47.9 us | 90.5 us / 92.7 us | 1.01 ms / 1.04 ms |
| 1,000,000 | 357 us / 362 us | 425 us / 431 us | 794 us / 809 us | 11.1 ms / 11.2 ms |
| 10,000,000 | 9.85 ms / 10.1 ms | 20 ms / 21 ms | 11.2 ms / 11.2 ms | 107 ms / 108 ms |

Default threads:

| N | NumPy | KlongPy, NumPy backend | KlongPy, torch backend |
|---|---|---|---|
| 1,000 | 1.63 us / 1.63 us | 4.82 us / 4.86 us | 8.49 us / 8.52 us |
| 10,000 | 5.1 us / 5.16 us | 8.42 us / 8.53 us | 15.8 us / 15.8 us |
| 100,000 | 38.4 us / 38.5 us | 49.9 us / 50.2 us | 47.9 us / 53.5 us |
| 1,000,000 | 369 us / 373 us | 443 us / 449 us | 127 us / 128 us |
| 10,000,000 | 9.82 ms / 10.3 ms | 17.5 ms / 18.1 ms | 9.07 ms / 9.42 ms |

* KlongPy with the NumPy backend against NumPy, single thread: 3.1 times as long at 1,000 elements, because the interpreter costs a few microseconds per expression. It takes 1.2 to 1.3 times as long at 100,000 and 1,000,000 elements and 2.0 times as long at 10,000,000.
* KlongPy with the torch backend against NumPy, single thread: it takes 1.1 to 5.0 times as long at every size.
* The same with default threads (16 for torch): the torch backend is 2.9 times faster than NumPy at 1,000,000 elements and 1.1 times faster at 10,000,000. Against the NumPy backend of KlongPy it is 3.5 and 1.9 times faster.
* The native runtime against the NumPy backend, single thread: it takes 4.7 to 26 times as long on this expression. Over all the single-thread array tables on this page the range is 1.8 to 26.

## Reduce, scan and power

Single thread:

| Expression | N | NumPy | KlongPy, NumPy backend | KlongPy, torch backend | Native runtime |
|---|---|---|---|---|---|
| `+/!N` | 100,000 | 24 us / 24.1 us | 27.5 us / 28.3 us | 35 us / 35.2 us | 98.5 us / 160 us |
| `+/!N` | 1,000,000 | 237 us / 242 us | 239 us / 240 us | 285 us / 291 us | 870 us / 1.5 ms |
| `+/!N` | 10,000,000 | 6.84 ms / 6.87 ms | 7.11 ms / 7.19 ms | 3.96 ms / 3.98 ms | 15.1 ms / 17.1 ms |
| `+\!N` (scan) | 100,000 | 45.4 us / 45.6 us | 48.2 us / 48.3 us | 54.5 us / 55.4 us | 207 us / 312 us |
| `+\!N` (scan) | 1,000,000 | 942 us / 944 us | 439 us / 445 us | 463 us / 468 us | 3.01 ms / 3.32 ms |
| `+/(!N)*(!N)` (dot product) | 100,000 | 74.1 us / 74.3 us | 65.3 us / 65.7 us | 99.8 us / 100 us | 530 us / 715 us |
| `+/(!N)*(!N)` (dot product) | 1,000,000 | 1.21 ms / 1.21 ms | 1.4 ms / 1.41 ms | 899 us / 909 us | 6.29 ms / 7.41 ms |
| `(1+!N)^2` (power) | 100,000 | 36.5 us / 36.6 us | 82.3 us / 82.7 us | 197 us / 198 us | 776 us / 870 us |
| `(1+!N)^2` (power) | 1,000,000 | 353 us / 357 us | 733 us / 735 us | 1.8 ms / 1.81 ms | 8.97 ms / 9.28 ms |
| `+100 100^!10000` (transpose) | 10,000 | 4.43 us / 4.44 us | 2.52 us / 2.53 us | 3.9 us / 3.96 us | 7.12 us / 13.1 us |

Default threads:

| Expression | N | NumPy | KlongPy, NumPy backend | KlongPy, torch backend |
|---|---|---|---|---|
| `+/!N` | 100,000 | 25 us / 25.1 us | 28.9 us / 28.9 us | 22.7 us / 24.4 us |
| `+/!N` | 1,000,000 | 245 us / 248 us | 248 us / 253 us | 55.8 us / 57 us |
| `+/!N` | 10,000,000 | 6.95 ms / 7.38 ms | 6.89 ms / 7.39 ms | 1.14 ms / 2.22 ms |
| `+\!N` (scan) | 100,000 | 46.8 us / 47 us | 49.6 us / 50 us | 59.8 us / 59.9 us |
| `+\!N` (scan) | 1,000,000 | 991 us / 1.02 ms | 463 us / 467 us | 449 us / 463 us |
| `+/(!N)*(!N)` (dot product) | 100,000 | 76.4 us / 76.8 us | 68.9 us / 69.2 us | 60.5 us / 63.8 us |
| `+/(!N)*(!N)` (dot product) | 1,000,000 | 1.25 ms / 1.29 ms | 1.47 ms / 1.49 ms | 151 us / 154 us |
| `(1+!N)^2` (power) | 100,000 | 37.6 us / 37.8 us | 83.9 us / 84.6 us | 90.7 us / 94.6 us |
| `(1+!N)^2` (power) | 1,000,000 | 364 us / 457 us | 755 us / 761 us | 273 us / 275 us |
| `+100 100^!10000` (transpose) | 10,000 | 4.61 us / 4.7 us | 2.29 us / 2.29 us | 3.8 us / 3.82 us |

* KlongPy with the NumPy backend against NumPy: a reduce (`+/`) and a dot product take 0.9 to 1.2 times as long, with one thread or many. The scan `+\` at a million elements takes half as long as `np.cumsum` (2.1 times faster).
* The power `(1+!N)^2` on the NumPy backend takes 2.1 to 2.3 times as long as NumPy.
* With 16 threads the torch backend is faster than the NumPy backend of KlongPy at a million elements: 4.4 times for `+/!N`, 9.7 times for the dot product and 2.8 times for the power. At 100,000 elements the two are within 1.3 times of each other.
* On one thread the torch backend takes longer than the NumPy backend in 8 of the 10 rows. It is faster for `+/!N` at 10,000,000 elements (1.8 times) and for the dot product at 1,000,000 (1.6 times).

## Grade and sort

`<x` (grade) returns the sort order of `x`. `x@<x` sorts by indexing with it. Single thread:

| Expression | N | NumPy | KlongPy, NumPy backend | KlongPy, torch backend | Native runtime |
|---|---|---|---|---|---|
| `<!N` (grade of a sorted range) | 10,000 | 16.9 us / 17 us | 19.4 us / 19.4 us | 54.4 us / 54.5 us | 35.6 us / 79.5 us |
| `<!N` (grade of a sorted range) | 100,000 | 162 us / 167 us | 164 us / 164 us | 990 us / 992 us | 490 us / 505 us |
| `<!N` (grade of a sorted range) | 1,000,000 | 2.08 ms / 2.35 ms | 1.59 ms / 1.61 ms | 13.7 ms / 13.8 ms | 4.36 ms / 5.37 ms |
| `<r` (grade, random floats) | 10,000 | 229 us / 229 us | 233 us / 233 us | 404 us / 416 us | n/a |
| `<r` (grade, random floats) | 100,000 | 3.53 ms / 3.53 ms | 3.76 ms / 3.83 ms | 5.79 ms / 5.88 ms | n/a |
| `<r` (grade, random floats) | 1,000,000 | 44.6 ms / 45.2 ms | 44.5 ms / 45.6 ms | 70.4 ms / 70.9 ms | n/a |
| `r@<r` (sort, random floats) | 10,000 | 235 us / 235 us | 243 us / 243 us | 446 us / 456 us | n/a |
| `r@<r` (sort, random floats) | 100,000 | 3.61 ms / 3.62 ms | 3.67 ms / 3.78 ms | 6.22 ms / 6.31 ms | n/a |
| `r@<r` (sort, random floats) | 1,000,000 | 46.3 ms / 47.9 ms | 45.6 ms / 47 ms | 76.7 ms / 78.2 ms | n/a |

Default threads:

| Expression | N | NumPy | KlongPy, NumPy backend | KlongPy, torch backend |
|---|---|---|---|---|
| `<!N` (grade of a sorted range) | 10,000 | 17.5 us / 17.5 us | 19.8 us / 19.8 us | 56.2 us / 56.9 us |
| `<!N` (grade of a sorted range) | 100,000 | 166 us / 166 us | 169 us / 170 us | 582 us / 587 us |
| `<!N` (grade of a sorted range) | 1,000,000 | 2.14 ms / 2.15 ms | 1.64 ms / 1.65 ms | 3.94 ms / 4.3 ms |
| `<r` (grade, random floats) | 10,000 | 236 us / 236 us | 238 us / 238 us | 421 us / 430 us |
| `<r` (grade, random floats) | 100,000 | 3.62 ms / 3.63 ms | 3.59 ms / 3.59 ms | 6.55 ms / 6.59 ms |
| `<r` (grade, random floats) | 1,000,000 | 45.8 ms / 46.6 ms | 45.6 ms / 46 ms | 73 ms / 74.7 ms |
| `r@<r` (sort, random floats) | 10,000 | 242 us / 242 us | 248 us / 248 us | 480 us / 485 us |
| `r@<r` (sort, random floats) | 100,000 | 3.71 ms / 3.71 ms | 3.67 ms / 3.68 ms | 6.7 ms / 6.76 ms |
| `r@<r` (sort, random floats) | 1,000,000 | 47.5 ms / 48.4 ms | 46.7 ms / 51.6 ms | 76.5 ms / 86.6 ms |

* KlongPy with the NumPy backend against NumPy: grade takes 0.8 to 1.1 times as long as `np.argsort`, and `x@<x` on random floats takes 1.0 times as long as `r[np.argsort(r)]` (within 3% at every size).
* KlongPy with the torch backend against the NumPy backend, on random floats (grade and sort): it takes 1.5 to 1.9 times as long at every size, with one thread or many.
* The same for the grade of a sorted range: 2.4 to 3.4 times as long with 16 threads, and 2.8 to 8.6 times as long on one thread (2.8 at 10,000 elements, 6.0 at 100,000, 8.6 at 1,000,000).

## Matrix multiply through `.pyf`

`.pyf` imports a Python function, so these rows time a library call: the work is done by NumPy or torch, and KlongPy adds no measurable time. Single thread:

| N (N x N, float32) | NumPy | `.pyf` to NumPy | torch | `.pyf` to torch |
|---|---|---|---|---|
| 1000 | 12.1 ms / 12.3 ms | 12 ms / 12 ms | 12.2 ms / 13 ms | 12.3 ms / 12.7 ms |
| 2000 | 96.1 ms / 96.5 ms | 95.8 ms / 96.2 ms | 101 ms / 102 ms | 99.8 ms / 100 ms |
| 4000 | 765 ms / 766 ms | 763 ms / 764 ms | 818 ms / 823 ms | 821 ms / 834 ms |

Default threads:

| N (N x N, float32) | NumPy | `.pyf` to NumPy | torch | `.pyf` to torch |
|---|---|---|---|---|
| 1000 | 1.46 ms / 1.51 ms | 1.4 ms / 1.41 ms | 2.32 ms / 2.41 ms | 2.44 ms / 2.6 ms |
| 2000 | 7.86 ms / 7.98 ms | 7.95 ms / 7.99 ms | 10.7 ms / 11.7 ms | 10.9 ms / 11.2 ms |
| 4000 | 62.6 ms / 66.4 ms | 62.6 ms / 63 ms | 100 ms / 108 ms | 94.3 ms / 102 ms |

On the CPU, NumPy is as fast as torch or faster for this float32 product. Comparing the NumPy and torch columns: torch takes 1.4 to 1.6 times as long with 16 threads and 1.0 to 1.1 times as long on one thread. Choose torch for a matrix multiply to run it on a GPU, not for the CPU.

## The native engine, measured in process on one machine

This section is a second set of measurements, from `native/scripts/bench_engine.py`. It times the engine inside one process, after a warm-up run, with no process launch per timing, and runs the same workloads in CPython and NumPy. It was made to decide what to optimise in the engine. The "before" columns are the engine as it was before any speed work (the commit of 2026-11-18, "Packaging: one abi3 wheel with the extension, kgpy and kg"), measured on 2026-11-18. The "after" columns are the engine after three rounds of speed work, measured on 2026-11-19, and the CPython and NumPy columns are from the 2026-11-19 runs. The tables above come from another machine with another method and are not repeated here.

| Item | Value |
|---|---|
| Date | "before": 2026-11-18. "after": 2026-11-19 |
| Machine | Apple M2 Ultra, 24 cores, 192 GB of memory, macOS 26.6 |
| Load | not idle: the machine was shared. "Before": `uptime` showed load averages of 9 to 14 during the runs (13.8, 12.9, 11.0 and 9.1 for the one-minute average at the start of the four runs). "After": the one-minute average was 15.98 at the start of the first run and 15.95 at the start of the second; the runs of the older extension build below started at 10.50 |
| Python, NumPy | 3.12.9, 2.5.5 |
| Rust | 1.90.0-nightly, `--release`, one code generation unit |
| Engine "after" | the commit "Engine speed, third round: append in place, direct calls to named functions, leaf arithmetic" |
| Engine "before rounds two and three" | the extension built from the commit before the second round, "Migration guide lists every break from 0.7; scope wording corrected; arity error for system functions". It has the first round of speed work and not the second and third. Measured on 2026-11-19, in the extension only |
| kg | `native/target/release/kg`, timed with the monotonic elapsed-time clock `.pc()` around calls to the workload |
| Extension | `klong_native` through `KlongInterpreter` from `klongpy` (NumPy backend), timed with the same Klong function (`.pc()`), so a result is never converted to Python |

Method: a workload runs once to warm up, and that run sizes a batch of at least 3 ms; the figure is the best of 3 to 30 batches (as many as fit in half a second), per run. A run of 2 s or more is timed once. Each cell is the better of two rounds, and the two rounds were run one after the other. "Before" and "after" were measured in different sessions, under the loads stated above, so the rows that the work did not change show the noise between sessions: in the 39 array rows that no change was aimed at (add, multiply, the two chains, dot, sum, max of integers, the grade of integers, take and drop, at the three sizes), "before" divided by "after" is between 0.90 and 1.11 in 37 rows and 0.75 and 0.76 in two (`add float` and `multiply float` at 100,000 elements, discussed below). A ratio of two cells from different sessions is good to about 10%. A workload is the body of `wk::{expr;0}`, so what is timed is making the result and dropping it. The CPython column uses lists, loops and the builtins that fit (`sum`, `min`, `sorted`, `itertools.accumulate`); the NumPy column is the vectorised form. The arrays are `ia`, ints 0 to n-1, `ib`, `3+ia`, `fa`, `0.5*ia`, `fb`, `1.5+fa`, `pi`, a permutation of 0 to n-1 (`(7919*ia)!n`) and `pf`, `0.5*pi`. The two append rows were added to the script after the first measurement, so they have no "before" figure.

The historical float chain rows below used `(fa*fb+fa)-fb` in Klong. Right-to-left evaluation makes the inner expression `fa*(fb+fa)`, while the Python runners used `(fa*fb)+fa`. The harness now parenthesizes the Klong product explicitly. Those three historical rows timed different arithmetic expressions.

### Programs the interpreter runs

| Program | CPython | kg before | kg after | kg speed-up | extension before | extension, rounds 1 only | extension after | kg after / CPython |
|---|---|---|---|---|---|---|---|---|
| fib(24) recursive | 4.17 ms | 28.5 ms | 14.9 ms | 1.92x | 27.7 ms | 21.5 ms | 14.8 ms | 3.56 |
| loop by while, 1e5 | 1.54 ms | 28.3 ms | 8.85 ms | 3.20x | 28.3 ms | 12 ms | 8.97 ms | 5.75 |
| loop by converge, 1e5 | 1.54 ms | 18.9 ms | 6.19 ms | 3.05x | 18.6 ms | 9.08 ms | 6.26 ms | 4.02 |
| reduce with a lambda, 1e5 | 4.06 ms | 15.3 ms | 5.3 ms | 2.89x | 15.1 ms | 6.92 ms | 5.31 ms | 1.31 |
| each with a lambda, 1e5 | 4.19 ms | 15.2 ms | 5.25 ms | 2.90x | 14.4 ms | 6.94 ms | 5.3 ms | 1.25 |
| dictionary updates, 5e4 | 1.76 ms | 18.8 ms | 11.3 ms | 1.66x | 17.2 ms | 13.4 ms | 11.1 ms | 6.45 |
| string building, 2e4 | 713 us | 27.1 ms | 19.4 ms | 1.40x | 27.3 ms | 19.2 ms | 19.1 ms | 27.2 |
| string append, results dropped, 2e4 | 703 us | - | 2 ms | - | - | 9.06 ms | 1.92 ms | 2.84 |
| vector append, results dropped, 2e4 | 288 us | - | 1.98 ms | - | - | 88 ms | 1.91 ms | 6.88 |
| named-parameter calls, 1e5 | 5.91 ms | 25.3 ms | 12.8 ms | 1.98x | 24.9 ms | 16.8 ms | 12.1 ms | 2.16 |
| classic calls, 1e5 | 5.89 ms | 23.4 ms | 10.6 ms | 2.20x | 22.8 ms | 14.3 ms | 10.5 ms | 1.81 |
| qualified calls (dictionary), 1e5 | 6.73 ms | 26.5 ms | 14.4 ms | 1.84x | 26 ms | 17.4 ms | 14.6 ms | 2.15 |
| module-qualified calls, 1e5 | 6.84 ms | 29 ms | 16.2 ms | 1.79x | 28.8 ms | 19.7 ms | 16.8 ms | 2.37 |

* `kg` and the extension run the same engine: over all the rows they differ by 1.6% at the median and by at most 25% (the comparison of integers at 100,000 elements).
* On the eleven program rows that have a "before", the engine is 1.4 to 3.2 times faster than before: 1.4 for string building, 1.7 to 2.2 for dictionary updates and the four kinds of call, 1.9 for recursion and 2.9 to 3.2 for the loops and the lambdas under adverbs.
* It is still 1.3 to 27 times slower than CPython on these programs: 1.3 for a lambda under an adverb (reduce and each), 1.8 to 2.4 for calls, 2.8 for the string append, 3.6 for recursion, 4.0 to 5.8 for loops, 6.5 for dictionary updates, 6.9 for the vector append and 27 for building a string by joining while keeping the result.
* The two append rows drop the result of each step, so the string or vector is held by one name only. Before rounds two and three, the extension took 9.06 ms and 88 ms on them; now it takes 1.92 ms and 1.91 ms.

### Arrays

The tables have one row per operation. "kg before" is the engine before any speed work, "kg after" the engine now, in the same units.

**1,000 elements**

| Operation | NumPy | CPython | kg before | kg after | kg speed-up | extension, round 1 only | extension after | kg after / NumPy |
|---|---|---|---|---|---|---|---|---|
| `add int` | 0.484 us | 23.2 us | 0.455 us | 0.479 us | 0.95x | 0.448 us | 0.466 us | 0.99 |
| `multiply int` | 0.798 us | 23.7 us | 0.683 us | 0.726 us | 0.94x | 0.721 us | 0.741 us | 0.91 |
| `add float` | 0.498 us | 24.4 us | 0.444 us | 0.471 us | 0.94x | 0.429 us | 0.473 us | 0.95 |
| `multiply float` | 0.487 us | 24.7 us | 0.468 us | 0.467 us | 1.00x | 0.443 us | 0.479 us | 0.96 |
| `add mixed int/float` | 1.02 us | 30.2 us | 3.36 us | 0.482 us | 6.98x | 0.442 us | 0.476 us | 0.47 |
| `chain (a+b)*b int` | 1.19 us | 33.4 us | 0.884 us | 0.836 us | 1.06x | 0.789 us | 0.816 us | 0.70 |
| `chain (a*b+a)-b float` | 1.37 us | 29.2 us | 0.859 us | 0.773 us | 1.11x | 0.72 us | 0.769 us | 0.57 |
| `dot +/a*b float` | 1.02 us | 27 us | 0.659 us | 0.674 us | 0.98x | 0.661 us | 0.671 us | 0.66 |
| `compare int <` | 0.419 us | 18.7 us | 4.11 us | 0.468 us | 8.79x | 0.456 us | 0.465 us | 1.12 |
| `sum int` | 0.49 us | 2.32 us | 0.311 us | 0.342 us | 0.91x | 0.308 us | 0.327 us | 0.70 |
| `sum float` | 0.562 us | 2.09 us | 0.442 us | 0.467 us | 0.95x | 0.439 us | 0.458 us | 0.83 |
| `min float` | 0.467 us | 4.59 us | 17.5 us | 0.471 us | 37.2x | 0.437 us | 0.455 us | 1.01 |
| `max int` | 0.486 us | 6.39 us | 0.398 us | 0.424 us | 0.94x | 0.394 us | 0.403 us | 0.87 |
| `scan +\ int` | 2.45 us | 15.7 us | 0.747 us | 0.744 us | 1.00x | 0.709 us | 0.729 us | 0.30 |
| `grade up int` | 23.4 us | 56.3 us | 11.7 us | 12 us | 0.97x | 12 us | 12.3 us | 0.51 |
| `grade up float` | 25.8 us | 56.4 us | 11.5 us | 9.93 us | 1.16x | 9.99 us | 10 us | 0.39 |
| `sort float (x@<x)` | 26.8 us | 34.1 us | 22.9 us | 11.2 us | 2.04x | 21 us | 11.3 us | 0.42 |
| `take h#a` | 0.315 us | 0.637 us | 0.349 us | 0.383 us | 0.91x | 0.346 us | 0.359 us | 1.21 |
| `drop h_a` | 0.315 us | 0.593 us | 0.356 us | 0.379 us | 0.94x | 0.354 us | 0.353 us | 1.20 |
| `reshape [2 h]:^a` | 0.484 us | 1.25 us | 7 us | 0.502 us | 13.9x | 0.499 us | 0.495 us | 1.04 |
| `find a?(n-1)` | 1.04 us | 26.9 us | 5.97 us | 0.685 us | 8.72x | 0.65 us | 0.66 us | 0.66 |
| `where &(p>h)` | 1.53 us | 23 us | 9.51 us | 1.07 us | 8.91x | 2.91 us | 1.02 us | 0.70 |
| `each-pair a{x+y}'b` | 0.501 us | 36.8 us | 161 us | 58.1 us | 2.77x | 71.9 us | 59 us | 116 |

**100,000 elements**

| Operation | NumPy | CPython | kg before | kg after | kg speed-up | extension, round 1 only | extension after | kg after / NumPy |
|---|---|---|---|---|---|---|---|---|
| `add int` | 18.9 us | 2.4 ms | 19.2 us | 18.5 us | 1.04x | 18.6 us | 18.7 us | 0.98 |
| `multiply int` | 35.9 us | 2.46 ms | 37.6 us | 36.5 us | 1.03x | 37.8 us | 37.5 us | 1.02 |
| `add float` | 24.2 us | 2.3 ms | 18.2 us | 24 us | 0.76x | 23.8 us | 24.2 us | 0.99 |
| `multiply float` | 19 us | 2.29 ms | 18.3 us | 24.3 us | 0.75x | 24.5 us | 24.3 us | 1.28 |
| `add mixed int/float` | 54.7 us | 2.87 ms | 301 us | 19.6 us | 15.4x | 18.5 us | 24.2 us | 0.36 |
| `chain (a+b)*b int` | 55.7 us | 3.2 ms | 59.4 us | 53.6 us | 1.11x | 53.9 us | 58.4 us | 0.96 |
| `chain (a*b+a)-b float` | 63.4 us | 2.78 ms | 58 us | 60.4 us | 0.96x | 65.5 us | 66.4 us | 0.95 |
| `dot +/a*b float` | 37.4 us | 2.76 ms | 41.5 us | 39.8 us | 1.04x | 43.8 us | 40.4 us | 1.07 |
| `compare int <` | 18.5 us | 1.61 ms | 380 us | 19.7 us | 19.3x | 24 us | 24.6 us | 1.07 |
| `sum int` | 9.19 us | 248 us | 9.02 us | 9.17 us | 0.98x | 9.02 us | 9.27 us | 1.00 |
| `sum float` | 18.2 us | 230 us | 20.7 us | 20.5 us | 1.01x | 20.4 us | 20.5 us | 1.13 |
| `min float` | 9.23 us | 451 us | 1.77 ms | 16.2 us | 109x | 15.9 us | 16.2 us | 1.75 |
| `max int` | 9.31 us | 605 us | 16 us | 16.2 us | 0.99x | 16.1 us | 16.2 us | 1.74 |
| `scan +\ int` | 199 us | 1.51 ms | 109 us | 37.9 us | 2.88x | 41.3 us | 38.2 us | 0.19 |
| `grade up int` | 6.04 ms | 7.23 ms | 2.26 ms | 2.26 ms | 1.00x | 2.32 ms | 2.26 ms | 0.37 |
| `grade up float` | 4.04 ms | 7.67 ms | 7.45 ms | 1.88 ms | 3.96x | 1.84 ms | 1.86 ms | 0.47 |
| `sort float (x@<x)` | 4.12 ms | 4.35 ms | 8.64 ms | 2.01 ms | 4.30x | 3.04 ms | 2 ms | 0.49 |
| `take h#a` | 5.52 us | 100 us | 6.39 us | 6.87 us | 0.93x | 5.59 us | 6.66 us | 1.25 |
| `drop h_a` | 6.84 us | 81.2 us | 6.72 us | 6.91 us | 0.97x | 6.82 us | 6.73 us | 1.01 |
| `reshape [2 h]:^a` | 13.2 us | 193 us | 652 us | 13.1 us | 49.9x | 13.2 us | 13 us | 0.99 |
| `find a?(n-1)` | 27 us | 2.86 ms | 551 us | 35.6 us | 15.5x | 35.4 us | 35.9 us | 1.32 |
| `where &(p>h)` | 76.2 us | 2.47 ms | 885 us | 69 us | 12.8x | 251 us | 68.5 us | 0.90 |
| `each-pair a{x+y}'b` | 19 us | 3.62 ms | 15.7 ms | 5.81 ms | 2.70x | 7.28 ms | 6.06 ms | 306 |

**10,000,000 elements**

| Operation | NumPy | CPython | kg before | kg after | kg speed-up | extension, round 1 only | extension after | kg after / NumPy |
|---|---|---|---|---|---|---|---|---|
| `add int` | 2.64 ms | 302 ms | 2.6 ms | 2.88 ms | 0.90x | 2.64 ms | 2.83 ms | 1.09 |
| `multiply int` | 3.62 ms | 290 ms | 3.7 ms | 3.75 ms | 0.99x | 3.72 ms | 3.75 ms | 1.03 |
| `add float` | 2.64 ms | 305 ms | 2.59 ms | 2.81 ms | 0.92x | 2.65 ms | 2.64 ms | 1.06 |
| `multiply float` | 2.68 ms | 310 ms | 2.59 ms | 2.83 ms | 0.91x | 2.64 ms | 2.68 ms | 1.06 |
| `add mixed int/float` | 6 ms | 364 ms | 30.4 ms | 2.7 ms | 11.2x | 2.61 ms | 2.75 ms | 0.45 |
| `chain (a+b)*b int` | 6.78 ms | 400 ms | 6.27 ms | 6.95 ms | 0.90x | 6.47 ms | 6.56 ms | 1.03 |
| `chain (a*b+a)-b float` | 7.86 ms | 357 ms | 7.88 ms | 7.85 ms | 1.00x | 7.83 ms | 7.83 ms | 1.00 |
| `dot +/a*b float` | 4.5 ms | 281 ms | 4.75 ms | 4.91 ms | 0.97x | 4.77 ms | 4.74 ms | 1.09 |
| `compare int <` | 2.61 ms | 174 ms | 38.3 ms | 2.64 ms | 14.5x | 2.65 ms | 2.6 ms | 1.01 |
| `sum int` | 1.38 ms | 25.7 ms | 1.35 ms | 1.38 ms | 0.98x | 1.37 ms | 1.37 ms | 1.00 |
| `sum float` | 1.88 ms | 23.7 ms | 2.13 ms | 2.12 ms | 1.00x | 2.1 ms | 2.13 ms | 1.13 |
| `min float` | 1.41 ms | 46.4 ms | 176 ms | 1.67 ms | 105x | 1.68 ms | 1.66 ms | 1.19 |
| `max int` | 1.41 ms | 60.8 ms | 1.59 ms | 1.67 ms | 0.95x | 1.68 ms | 1.66 ms | 1.18 |
| `scan +\ int` | 20.3 ms | 248 ms | 4.83 ms | 3.75 ms | 1.29x | 4.49 ms | 3.77 ms | 0.18 |
| `grade up int` | 907 ms | 818 ms | 355 ms | 370 ms | 0.96x | 374 ms | 365 ms | 0.41 |
| `grade up float` | 425 ms | 805 ms | 894 ms | 250 ms | 3.58x | 246 ms | 248 ms | 0.59 |
| `sort float (x@<x)` | 455 ms | 377 ms | 1.03 s | 265 ms | 3.88x | 379 ms | 280 ms | 0.58 |
| `take h#a` | 697 us | 9.59 ms | 675 us | 679 us | 0.99x | 685 us | 707 us | 0.97 |
| `drop h_a` | 703 us | 9.66 ms | 672 us | 683 us | 0.98x | 685 us | 711 us | 0.97 |
| `reshape [2 h]:^a` | 1.43 ms | 19 ms | 67.9 ms | 1.4 ms | 48.4x | 1.41 ms | 1.43 ms | 0.98 |
| `find a?(n-1)` | 3.1 ms | 294 ms | 55.8 ms | 3.52 ms | 15.8x | 3.5 ms | 3.57 ms | 1.14 |
| `where &(p>h)` | 8.47 ms | 258 ms | 91.6 ms | 9.45 ms | 9.70x | 21.5 ms | 9.62 ms | 1.12 |
| `each-pair a{x+y}'b` | 2.62 ms | 432 ms | 1.62 s | 604 ms | 2.68x | 756 ms | 615 ms | 231 |

* Element-wise arithmetic, sums and `take` and `drop` run at the speed of NumPy (0.70 to 1.28 times its time) at every size, and did before. At 10,000,000 elements they are bound by memory: the same number of bytes are read and written as NumPy's.
* What changed in the first round is the operations that went through one boxed value per element: comparisons (8.8 to 19 times faster than before), mixed integer and float arithmetic (7.0 to 15 times), the minimum of floats (37 to 109 times), reshape (14 to 50 times), find (8.7 to 16 times), the grade of floats (3.6 to 4.0 times at 100,000 and 10,000,000 elements). After the work, `kg` takes 0.36 to 1.3 times as long as NumPy on all of these except the ones listed next.
* What changed in rounds two and three, between the extension built before them and now, in the same session: `where` 3.7 times faster at 100,000 elements (251 us to 68.5 us), 2.2 times at 10,000,000 and 2.9 times at 1,000; the sort of floats (`x@<x`) 1.9 times at 1,000 elements, 1.5 times at 100,000 and 1.35 times at 10,000,000; each-pair with a lambda 1.2 times (7.28 ms to 6.06 ms at 100,000 elements). Over the three sizes, `where` is now 8.9 to 12.8 times faster than before, the sort of floats 2.0 to 4.3 times.
* Still slower than NumPy: `&/` and `|/` at 1.2 to 1.8 times at 100,000 and 10,000,000 elements (eight lanes, not NumPy's SIMD minimum), `find` at 1.1 to 1.3 times, `take` and `drop` at 1.2 times at 1,000 elements, and the each-pair row, which calls a Klong lambda per pair (58 to 60 ns per pair here; NumPy does not call a function). `where` is at 0.70 to 1.12 times NumPy's time.
* The grade of integers takes 0.37 to 0.51 times as long as NumPy's stable argsort, the grade of floats 0.39 to 0.59 times and the sort of floats 0.42 to 0.58 times.
* Rows that got slower during the work, measured:
  * `scan +\ int` at 100,000 elements took 108 us after rounds two and three, against 41 to 44 us for the extension built before them. The loop itself had not been edited: the compiler produced slower code for it once the kernels for the other scans were added next to it. The loop is now a function of its own and the row is 37.9 us (3.75 ms at 10,000,000 elements). The two scan rows in the tables are from that later build, measured on its own in the same session.
  * `add float` and `multiply float` at 100,000 elements take 24 us, against 18.2 us and 18.3 us before (0.76 and 0.75 times the speed). The extension built before rounds two and three took 23.8 to 24.5 us on the same rows in this session, so rounds two and three did not cause it. The rows were 19.6 us in the first table of this section, so this is a difference between sessions or in code layout from the first round; its cause was not found. NumPy's `add float` took 24.2 us in this session and 23.8 us in the first.

### What the profiles said

Profiles of the first round were taken with `sample` (macOS, 1 ms interval, the time at the top of the stack, symbolised from a release build with line tables). They covered the rows with the worst ratios and decided the order of the work:

* **Calls driven by an adverb** (`{x*x}'a`, `{x+y}/a`, `:~`): about a third of the time was in the allocator (`malloc` and `free`), from the argument vector, the list of layers of a projection and a frame that did not come from the frame pool, made for every call by `apply`. A function literal called with exactly the arguments it takes now skips those and uses the pool: 1.3 to 1.9 times faster.
* **Recursion** (fib): no single place. `eval_node` 20% of the time itself, `eval_dyad` 11%, name resolution (`resolve` and `lookup`) 13%, copying and dropping values 12%, pushing and popping frames 7%, and the stack check and the deferred-failure check at the head of every evaluation 6%. Two numbers in a dyad now skip the pairing routine (1.13 times) and the two checks are inlined (1.12 times).
* **String building** (`s::s,"ab"`): 60% in `memmove` (the whole string is copied by every join, as the value semantics of strings say), 20% counting the code points of the result again, 17% in the allocator. The count is now the sum of the two counts (1.38 times). The copy remains when the result is kept.
* **Dictionary updates**: the allocator, dropping values, and the hash probe; each `d,x,,x*2` builds two small lists. Only the calls part was cut in the first round (1.3 times).
* **Comparison, mixed arithmetic, float minimum, reshape, find, where**: the loop itself. They went through the generic element loop (a division per element to find the operand, a boxed `Value` per element, dispatch per element), not through the vectorised loops that addition and multiplication have. They now have loops the compiler vectorises, or a path that copies the numbers instead of boxing them.
* **Grade and sort of floats**: 85% in the sort's comparison, which goes through the index to the number (a cache miss per comparison). Sorting (key, index) pairs, the key an integer that orders as the float does, makes it 3.7 to 4.0 times faster.

Rounds two and three were also taken from profiles (their commit messages say so); the profile figures of those rounds were not kept for this page. What the code and the commit messages give as the reason for each:

* **Scalar path** (round two): the operands that are a number or a name are read in place instead of being evaluated as a node, and the call of a function literal is cheaper. The commit message gives 15 to 28 percent faster.
* **Unique and group** (round two): `?x` of a flat vector of integers or floats searched what it had kept so far, so it took time as the square of the length when most values were different. It now sorts (key, position) pairs and reads the runs. `=x` of integers is done the same way.
* **Gather** (round two): `a@b` for a flat vector `a` and an in-range integer vector `b` picks the elements without making a value of each.
* **Cycling take** (round two): `k#b` with `k` larger than the length of a flat vector of numbers repeats the vector without a value per element.
* **Integer divide and remainder, scan kernels** (round two): the running product, minimum and maximum of integers and the division and remainder of integers have kernels.
* **Appending in a loop** (round three): `v::v,y` copied the whole value on every append, so building a string or a vector in a loop took time as the square of its length. When nothing else holds the value, the append is now done in place. The commit message says so; the two append rows above are the measurement.
* **Calls to a named function, arithmetic on leaves, dictionary updates** (round three): a direct call to a function held by a name, arithmetic whose operands are leaves, and a direct join for `d,x,,x*2`. The commit message gives fib 14.5 percent faster and dictionary updates 14.8 percent.

What limits the engine now: the tree walk (dispatch in `eval_node` and `eval_dyad`, `Value` clones and drops, the frame push and pop with its table of shadowed names), the value semantics of strings and of dictionary entries (a new object per join and per entry when the result is kept), the lambda call under an adverb (52 ns per element in the each row, 58 to 60 ns per pair in the each-pair rows), and, at 10,000,000 elements, memory. This is from the profiles of the first round and the per-call times above; the engine was not profiled again after round three.

### What each change did

The first part of the table was measured alone, on `kg`, the same machine and load, against the engine without that change. Every change but the last two of that part is behind the cargo feature `speed` of `klong-core`, which `kg` and the extension enable and the board images do not. The second part could not be measured alone, because there is no build with one of rounds two and three and not the other; it compares the extension built before round two with the extension now, in the same session, with the load given above (10.50 at the start of the older build's runs, 15.95 at the start of the current build's second run). The commits of rounds two and three are also behind the feature `speed`.

| Change | Effect |
|---|---|
| Vectorised loops for the comparisons, mixed integer and float arithmetic, the minimum and maximum of floats, and the float reduce `&/ |/ */` | comparison 16 times at 100,000 elements, mixed 13 times, `&/` of floats 111 times (the first version, which fell back to the old fold when the minimum was a zero, gave 4.3 times) |
| Reshape, find and where without boxing; the grade of floats by (key, index) pairs; the scan `+\` of floats | reshape 63 times, find 16 times, where 3.6 times, grade of floats 4.1 times, sort of floats 2.9 times (all at 100,000 elements); the scan of 1,000,000 floats 22 times (27.9 ms to 1.26 ms, not in the set) |
| `apply_fast`: a function literal under an adverb | `{x+y}/` 1.74, each 1.77, while 1.87, converge 1.50, named-parameter calls 1.40, classic calls 1.44, calls through a dictionary 1.38 |
| Two numbers in a dyad skip the pairing routine | fib(24) 1.13 times |
| `#[inline(always)]` on the stack check and the deferred-failure check | fib(24) 1.12 times |
| Joined text takes its code point count from its parts | string building 1.38 times |
| A temporary array is the buffer of the result (`(a+b)*b`) | 1.05 to 1.14 times at 100,000 elements, 0.98 to 1.01 at 10,000,000; the peak heap of a chain of four operations on a million elements fell from 40 MB to 32 MB. Kept for the second figure and the first |
| `codegen-units = 1` in the release profile | the programs above 4 to 11% faster, arrays between 0.78 and 1.27 at 100,000 elements (add of floats got slower by 22% at that size, a layout effect that is not there at 1,000 or 10,000,000), the build 10% longer |
| Fat LTO with `codegen-units = 1` | rejected: the geometric mean over the 55 rows measured with it is 1.027 against 1.017 without it, for 50% more build time |
| Inlining the stack check and the deferred-failure check everywhere | rejected: kept only under `speed`, so that the board image does not grow |
| Round two: cheaper leaf operands and calls, and the per-chain adverb body (the body of the verb of an adverb chain is found once) | with round three, on the extension, before rounds two and three to now: fib(24) 1.45 times, loop by while 1.34, converge 1.45, reduce 1.30, each 1.31, named-parameter calls 1.39, classic calls 1.36, qualified calls 1.19, module-qualified calls 1.17. The commit message of round two gives 15 to 28 percent faster for the scalar path |
| Round two: sort-based unique and group, gather, cycling take, integer divide and remainder, scan kernels | `where` 3.7 times at 100,000 elements, the sort of floats 1.5 times at 100,000 and 1.35 times at 10,000,000, each-pair 1.2 times; not attributed to one change. Unique, group, cycling take and integer divide have no row in the set and no measurement here. The commit message says unique of a flat vector is no longer quadratic |
| Round three: append in place | string append with the result dropped 4.7 times (9.06 ms to 1.92 ms), vector append 46 times (88 ms to 1.91 ms), 20,000 appends. Building a string while keeping the result (`s::s,"ab"`) is unchanged: 19.2 ms to 19.1 ms |
| Round three: direct calls to named functions, leaf arithmetic, direct joins for dictionary updates | with round two, as in the row above. The commit message of round three gives fib(24) 14.5 percent faster and dictionary updates 14.8 percent; dictionary updates measured 1.21 times (13.4 ms to 11.1 ms) over both rounds |

Not tried: explicit SIMD kernels (the vectorised loops are at NumPy's speed where memory allows), and a name-lookup cache (`resolve` and `lookup` are 13% of recursion, so the most it could give is a few percent).

### Float32 scalars

`Klong.set_float_width(32)` makes the engine of the extension compute floats as float32 does: every float value `eval` returns is the float32 nearest to it (a literal as well), and floats print as the shortest float32 text. `0.1+0.2` gives `0.3`, as torch gives in float32. With the width at 64, the default, nothing is rounded: the check costs nothing that could be measured (0.95 to 1.03 times the time of an engine without the check, on the call-heavy programs and two vector expressions, the best of three runs each). With the width at 32 the call-heavy programs take 1.0 times (integers) to 1.3 times as long (recursion with floats 1.31, a loop of float additions 1.15, each with a float lambda 1.16) and an element-wise float expression on 100,000 elements 1.6 to 1.8 times (one more pass over the result).

## What to write

* Write whole-array expressions. `+/x*x` does the work in the array library; `+/{x*x}'x` calls a lambda per element and takes 113 times as long as the same loop in CPython (NumPy backend).
* Avoid per-element lambdas (`f'x`) on large arrays when an operator does the job.
* Avoid appending to a list in a loop; build the result with one expression.
* Use the grade `<x` when you need the order, and `x@<x` to sort.
* Use the NumPy backend for arrays under a million elements. Use the torch backend on the CPU for reductions, dot products and element-wise work on arrays of a million elements or more, with several threads, and for gradients (see [PyTorch Backend and Autograd](torch_backend.md)). It is slower for sorting and for scalar programs.
* For a loop that must be fast, use the [native runtime](native/index.md) when the program fits, or compile a [static-profile](static-profile.md) function to C.

## GPU

This page has no GPU numbers. To time your own, create the interpreter with `device="cuda"` (or `"mps"`) and wait for the device before you stop the clock, because GPU calls return before the work finishes:

```python
import time, torch
from klongpy import KlongInterpreter

klong = KlongInterpreter(backend="torch", device="cuda")
klong("a::2*1+!10000000")
klong("a+a")                       # warm up
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(9):
    klong("a+a")
torch.cuda.synchronize()
print((time.perf_counter() - t) / 9, "seconds per run")
```

`tests/perf_vector.py` times a vector expression and a matrix multiply on every backend and device it finds.

## Rerun the measurements

`scripts/bench_performance.py` measures one configuration per process and writes JSON:

```text
python scripts/bench_performance.py --help
python scripts/bench_performance.py --backend numpy --out numpy.json
python scripts/bench_performance.py --backend torch --out torch.json
python scripts/bench_performance.py --native native/target/release/kg --out native.json
python scripts/bench_performance.py --baseline --out baseline.json
```

`native/scripts/bench_engine.py` is the in-process benchmark set of the section on the native engine above (`--kg PATH`, `--ext DIR` for the directory that holds `klong_native.abi3.so`, `--sizes`, `--only`, `--json`):

```text
python native/scripts/bench_engine.py --ext DIR_OF_THE_EXTENSION
```

For single-thread rows, set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, pass `--torch-threads 1` and pin the process to one core.
