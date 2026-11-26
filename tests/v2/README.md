# KlongPy v2 acceptance

The [RC2 dogfooding notes](rc2-dogfooding.md) track friction contributors encounter while developing and testing with native `kg`.

The suite defines what a v2 candidate must do to ship. Standalone `kg` and Python-hosted `kgpy` must run the same Klong database programs. Desktop `kg` must also run numeric array operations on a physical GPU. The Python host must expose the engine to arrays, callables, DataFrames, Arrow tables and device tensors.

## Run the tests

For development, run `python -m pytest tests/v2 -q -rs`. Database assertions execute even while the implementation is missing. Tests needing an external reference, an optional Python package or an explicit GPU report what is missing. Build `kg` with `cargo build --release -p klong-cli` inside `native/`, and install the Python package from the same source. The programs `test_language.kg`, `test_market.kg`, `test_table_values.kg` and `test_tables.kg` in `programs/` can also be run directly with either command. An assertion failure prints the case name, expected value and actual result and exits nonzero.

For release acceptance, run all four selections below against the same candidate. Each must exit zero with zero skipped tests. `KLONG_V2_RELEASE=1` turns a missing prerequisite into an error. Use `KLONG_NATIVE` and `KLONG_V2_KGPY` to point to the candidate executables when testing an installation. Use the candidate's Python environment for pytest as well. Its database interop tests need pandas and pyarrow. The JUnit reports record the checkout revision, dirty status, platform and measurements.

```sh
export KLONG_V2_RELEASE=1
python -m pytest tests/v2 -m 'not v2_gpu and not v2_reference and not v2_build' -o junit_family=xunit1 --junitxml=desktop.xml
KLONG_TEST_PEACHQ=/path/to/peachq/q python -m pytest tests/v2 -m v2_reference -o junit_family=xunit1 --junitxml=reference.xml
KLONG_V2_GPU_DEVICE=mps python -m pytest tests/v2 -m v2_gpu -o junit_family=xunit1 --junitxml=gpu.xml
KLONG_V2_BUILD=1 KLONG_V2_CARGO_TOOLCHAIN=stable python -m pytest tests/v2 -m v2_build -o junit_family=xunit1 --junitxml=build.xml
```

The source-build job uses Cargo, a POSIX C/C++ toolchain and Rust targets `thumbv6m-none-eabi` and `riscv32imac-unknown-none-elf`. Select a rustup toolchain with `KLONG_V2_CARGO_TOOLCHAIN`. The test installs into a temporary directory. The reference, source-build and GPU jobs can run on different machines. On a single machine with all prerequisites set the same variables and run `python -m pytest tests/v2 -o junit_family=xunit1 --junitxml=v2.xml`. Run the full GPU selection on a platform that supports both commands. The macOS candidate provides native Metal and hosted MPS. For each additional accelerator supported by the Python host, run its selected cases on that hardware. On an NVIDIA worker with the assigned GPU visible in the container, use:

```sh
KLONG_V2_RELEASE=1 KLONG_V2_GPU_DEVICE=cuda:0 python -m pytest tests/v2 -m v2_hosted_gpu -o junit_family=xunit1 --junitxml=cuda.xml
```

A missing device, CPU fallback or unsupported command line fails the corresponding selection. Windows runners use psutil for process memory measurements.

From the matching checkout, check the completed reports together:

```sh
python -m tests.v2.verify_reports desktop.xml reference.xml gpu.xml build.xml
```

The verifier collects the current test IDs and checks every reported result. A missing case, an unexpected case, a skip or a failure gives a nonzero exit status. Extra accelerator reports may repeat successful cases. Keep artifact hashes, hardware details and earlier failed runs with the reports used for sign-off.

When distributing an identified candidate, set `KLONG_V2_SOURCE_ARCHIVE` on each worker to the source archive used for its build. The suite records the file's SHA-256. Pass `--source path/to/klongpy-VERSION.tar.gz` to the verifier to check that each supplied report identifies that same archive. The source distribution includes these tests, the conformance programs and parser snapshots.

## What a pass establishes

| Area | Evidence |
|---|---|
| Klong programs | Both commands run the table and SQL cases. The tables-only build also checks that a failed re-index preserves the old composite key through an alias, re-keying changes which row an upsert selects, and a mixed batch preserves the order of existing and newly appended rows. SQL cases check rollback after a duplicate key and scan adjacent all-null and all-valid pages in two numeric columns, including an integer above 2⁵³. |
| Market queries | A six-trade fixture has hand-calculated last prices, minute volume, as-of bids, open/high/low/close bars, a volume-weighted price, and rolling two-trade volume by symbol. Cases include a late trade, equal quote timestamps, a trade before the first quote, and a symbol with no quote. The q reference runs independent queries against the same observations. |
| Persistence | Each host reopens a database migrated by the other. A committed column rename, column default and primary key survive reopening. A table rename and row update left uncommitted disappear when the writer closes. An independent DuckDB client checks the resulting schema and exact rows. The saved market dataset also gives the same answers after restart and a rolled-back delete. |
| IPC | All four client/server combinations can issue a successful call on the same connection after a remote Klong error. A second client verifies the server state. A remote composite-key update still replaces the correct row after a round trip. Recorded wire examples and a live q peer check type fidelity for populated and empty tables, keys, nanosecond timestamps, typed nulls and symbols. |
| Python | A strided NumPy array and a Python callback compose with a Klong function. DataFrame, Arrow and device-tensor tests exercise actual values through the public interpreter, including integers beyond double precision. An escaped Klong function can read its database after the original interpreter variable goes out of scope. When the final Python handle is dropped and collected, another `kg` process must be able to open the file. |
| GPU | A reduction and its gradient agree with explicit CPU answers. Device telemetry verifies the chain ran on the selected accelerator. A database result feeds a second device calculation. |

The new public forms pinned by these tests are built-in `.table` and `.db` on both desktop hosts, `.db("path")` for a persistent DuckDB database, and native `kg --device DEVICE`. Indexed mutations and the shape of existing `.db` query results use the current Klong APIs. A Python table exposes `get_dataframe()` and supports Arrow interchange.

For native device tests, `KLONG_GPU_TRACE` names a JSON file written by an instrumented candidate or backend profiler. It records the actual `device`, `kernel_count`, `device_operations` (Klong verbs executed there, including fused verbs), and cumulative `host_readback_bytes` for the invocation. The standalone energy program requires both multiplication and sum on the device and at most eight bytes returned to host memory. Timing includes startup, transfers, execution and retrieval of the scalar.

## Coverage by release requirement

| Requirement | Established coverage | Additional v2 acceptance |
|---|---|---|
| Language and compiler | `conformance/cases/` checks language semantics. `tests/golden/` checks parser trees. `tests/static/` checks the compiler, generated C and HLS against executable references. | `test_klong_cases.py` runs each named case in `programs/table_cases.kg` and `programs/sql_cases.kg` independently on both desktop commands. Wrong shapes, partial batches, precision, SQL NULL, query shape, isolation, rollback, constraints and recovery each have their own assertion. |
| Stateful tables and storage | `test_sys_fn_db.py` and `test_table_dict_and_size.py` exercise the Python adapter. | `test_storage.py` compares 36 successive overlapping batches with a Python dictionary after every update. A separate DuckDB client checks both file directions, and opens the file after a writer is killed with an uncommitted transaction. |
| Wire compatibility and Python interop | The existing codec, session and endpoint suites cover low-level framing and request lifecycles. | A real q peer compares serialized tables after an echo through each host, including composite keys, numeric widths, timestamps, dates and timespans. Arrow cases cover sliced and chunked arrays, empty schemas, and the distinction between NULL and NaN. |
| Accelerators | The static compiler suite runs generated HLS against reference implementations. Existing backend tests exercise Python tensor operations. | The GPU suite checks fractional values, a partial final block, matrix broadcasting and gradients on a selected device. `test_build_targets.py` links the board C ABI to a generated HLS kernel and verifies the result and actual submission count with a simulated attachment. |
| Headless command use | Existing CLI tests check usage and source diagnostics. | `test_cli_automation.py` runs with closed stdin. It checks JSON capability discovery, verbatim script arguments including flags, and the output channels and exit codes. On both hosts it inspects a script with `lint` and `ast`, then runs the script to verify that only execution produces its imported file's side effect. |
| Distribution and target composition | Existing wheel jobs install outside the checkout. Board scripts link firmware and report flash and available RAM. | A temporary source install must run the REPL, a script and an expression with ordinary desktop defaults. The matrix builds a core-only desktop, a tables-only desktop, and Arm/RISC-V board archives. Dependency checks exclude desktop engines from board builds. |

The test named `caseReplayIdempotent`, for example, can also run directly with `kg -e '.l("tests/v2/programs/table_cases.kg");caseReplayIdempotent()'`. For a physical Pico plus FPGA, record the board, bitstream, link, end-to-end latency and flash/RAM use. The simulated attachment exercises the software boundary. Physical transport, timing and electrical behavior require the real board.

## Performance budgets

The native ticker-plant cases check an in-process native WAL and RAM-only RDB, takeover of the external benchmark writer's files, a short write at the OS file-size limit, automatic WAL replay before serving, a lost publish reply after WAL sync, torn tails, checksum failures, competing publishers, subscriber delivery, and a head or HDB crash during rollover. Variable-width batches and multi-block rollback run through the same IPC entry point. The [ticker-plant example](../../examples/stocks/tickerplant/README.md) includes the cross-node load driver. When staging installed-wheel tests outside the checkout, copy the example directory from the candidate sdist alongside the tests.

The dense integer grade test orders 262,144 signed keys under a 5 MiB Klong heap limit. It checks the length and endpoints of all three groups. The numerical regression cases also cover stable ties, descending grades, 64-bit integer limits, reductions over each matrix axis and wrapping sums. Run the separate [NumPy comparison](../../docs/performance.md#native-kg-and-numpy-on-resident-cpu-arrays) to measure the kernels against NumPy on the same inputs.

To profile keyed updates, run `python -m pytest tests/v2/test_performance.py -k composite_key_update_scaling -o junit_family=xunit1 --junitxml=upserts.xml`. Both hosts record the median time for a 256-row batch on tables of 2,048, 8,192 and 32,768 rows. The batch corrects 128 existing composite keys, adds 64 keys and revises 64 of those existing and new keys again. Each sample checks every final key, quantity and row position. These timings establish a scaling baseline. For one sample, run `kg -e '.l("tests/v2/programs/keyed_updates.kg");keyedUpdateTime(32768)'` from the repo root.

Standalone `kg` must find distinct keys and group 80,000 observations over 10,000 station names in under one second median across three fresh processes. The check verifies every key and all eight row positions for each station.

The temporary running-sum case gives standalone `kg` a 10 MiB heap budget. It computes the last prefix sum of one million consecutive int64 values and verifies that scanning a shared vector preserves both aliases. Run `kg --heap-limit 10485760 tests/v2/programs/scan_memory.kg` to exercise it directly.

The SQL column case gives standalone `kg` a 20 MiB Klong heap budget. It reads one million `BIGINT` rows as a vector and checks the length and exact sum. Run `kg --heap-limit 20971520 tests/v2/programs/sql_column_memory.kg` to exercise it directly. DuckDB manages its own memory outside this budget.

The hosted scalar test times `1+1` before and after retaining a 524,288-row symbol table, once in the calling interpreter and once in another interpreter. It measures CPU time on the calling thread. The median cost with the table must stay within five times the baseline plus 100 microseconds. This catches ownership tracking that walks the entire table on each call.

The resident ID benchmark measures distinct IDs and counts per group for one million observations across 64 integer IDs. It checks the returned first-seen order and counts against an independent answer on every sample. Run `python -m pytest tests/v2/test_performance.py::test_group_resident_integer_ids -o junit_family=xunit1 --junitxml=ids.xml` to record both medians and the command digest.

The streaming benchmark runs separate processes on the same machine with one query-engine thread, an in-memory database and the same localhost IPC client. It sends identical 256-row symbol, time and price batches. It times 1,048,576 rows from a single producer after warm-up. It then resets the tables, preloads 32,768 rows and offers 25,600 ticks/s for 256 batches while two independent clients each issue 200 scheduled queries. Latency is measured from the scheduled request time and therefore includes queuing. Every query is checked against the feed updates that could have committed when it ran. Final counts and values detect missing or duplicated rows.

| Measurement | Initial acceptance bound |
|---|---|
| Standalone ingest throughput | At least 80% of the selected q reference on the same machine |
| Standalone p99 for last price and as-of bid | Each at most 1.25 times the corresponding q p99 |
| Offered-load test on both hosts | Slowest batch acknowledgement under 100 ms from schedule, and both query p99 values under 100 ms |
| Python call boundary on the small market fixture | p99 under 100 ms across 100 direct calls, checking each returned vector |
| Standalone CPU smoke test | Correct sum of a million integers in under 1 s median across five fresh processes |
| Resident vector, single CPU thread | Correct standalone sum of squares at 4,096 and 1,000,000 elements in under 100 ms median, including local IPC and excluding input construction. The reference job also records the same measurement for q, `kg` and `kgpy`. |
| Selective historical query | On one million persisted rows, each of six calls returns the exact price sum for one day and symbol in under 1 s |

Whitney's TAQ manual describes how the requested dates, symbols and columns affect disk reads. The history test therefore records the first selective query after reopening the database separately from the median of five repeats. The OS can still cache the file across a process restart. The Shakti project also distinguishes large-vector arithmetic from interpreter overhead. The resident-vector test measures both sizes after a warm-up and returns only a scalar over IPC. The numerical bounds here are project targets established for these fixtures. [Whitney's TAQ manual, section 8](https://github.com/KxSystems/kdb/blob/master/d/a/taq.htm) · [Shakti project notes](https://github.com/fshakti/shakti/blob/main/README.md)

Use an otherwise idle Linux or macOS reference host. Choose PeachQ with `KLONG_TEST_PEACHQ` or KX q with `KLONG_V2_KDB`. The latter takes precedence if both are set. The suite identifies the actual reference engine and records its version in the report. A comparison uses three paired trials, alternates which engine runs first and takes the median of the three ratios for each bound. Reports also include p50 latency and server resident memory (Linux high-water RSS or sampled RSS on macOS). The initial bounds apply to the named engine, feed size, concurrency and durability setting. A performance claim about KX kdb+ specifically requires measurements against KX q.

## Release decision

The latency gate measures from the scheduled request until the reply arrives. Each run also records the client's p99 delay before starting the request and its p99 round-trip time. Use these fields to investigate a slow run on a shared machine. The two percentiles can come from different requests.

For the slowest ingest acknowledgement, the report records the delay before sending and the request round-trip time for that same batch. A late send can include time spent waiting for the preceding acknowledgement. Each query result is checked against the feed watermarks captured around its request after the timed traffic finishes.

On Linux, record CPU affinity for the servers and client alongside the CPU quota. To diagnose a placement-sensitive result, repeat with executable wrappers for `KLONG_NATIVE`, `KLONG_V2_KGPY` and the q reference that each invoke `exec taskset -c "$SERVER_CPU" /absolute/path/to/command "$@"`. Choose an allowed CPU and export `SERVER_CPU`. This pins each server to the same CPU during its trial. Keep the client placement, workload and bounds the same, and retain both results. [Linux measurements](rc2-dogfooding.md#running-the-python-hosted-feed-on-linux)

For a comparison with separate server and client CPUs, inspect the allowed topology with `lscpu -e=CPU,CORE,SOCKET,NODE,CACHE`. Use the wrappers above for both servers and prefix the pytest command with `taskset -c "$CLIENT_CPUS"`. Choose different physical cores on the same socket for the client. Record both affinities, the quota and the default-placement report. Each ratio still compares the original scheduled workload and thresholds. [Measured CPU placements](../../docs/performance.md#streaming-tables-and-queries)

Require the four selections above, the existing Rust workspace tests, both language conformance runs and golden trees, the Python backend matrix, and the candidate installation and documentation checks to pass on the same source revision. Keep the release's existing hardware checks for the board and compiler targets it ships. Report the older suite's expected failures, platform skips and the cases in `conformance/pending/` separately. The four v2 selections require every case to pass. Record the supported platforms, tested GPU models and installed package versions alongside the XML files. A development run with skipped reference or hardware tests leaves that part of release acceptance open.
