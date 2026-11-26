# Using kg to develop kg — RC2 notes

Record friction encountered while doing real development work in `kg`. Include the command or small program, the observed result, any workaround, and a check that would demonstrate an improvement. Update an existing entry when it recurs or gets fixed. The open entries are candidates for RC2.


## Replacing a live SQL staging table during durable ingestion

The first ticker-plant prototype opened a new in-memory database and attached the same RDB file whenever the batch length changed. A three-row batch followed by a single-row batch was visible in SQL before the head was killed. Reopening the file recovered zero rows on one run. Explicitly detaching the file before replacing the connection passed the crash test, but alternating batch sizes repeatedly reopened the database. The example now keeps one connection and stages incoming columns in reusable tables of different widths. The restart test kills the head during a failed handoff and kills the HDB after its commit. Another test injects a null into the second chunk of a large batch and verifies that the first chunk rolls back. [Regression tests](test_tickerplant.py)

## Open

### Passing a benchmark operation through a helper

`call(f)::{f()};forward(f)::{call(f)};forward({[1 2]+[3 4]})` tried to call the already computed list. The diagnostic identified the list and suggested indexing it. The NumPy benchmark passes a one-argument function instead. With `call(f)::{f(0)};forward(f)::{call(f)};forward({x;[1 2]+[3 4]})` the same path returns `[4 6]`. An example explaining when a reference to a zero-argument function evaluates would help when writing timing and callback helpers.

### Passing file arguments to both desktop commands

While porting the weather example, `kg args.kg sample.txt` put the arguments in `.a`, and `kgpy args.kg -- sample.txt` put them in `.os.argv`. With a file containing only `.p(.a)`, hosted execution printed the unbound symbol `.a` in default mode and reported a name error at strict level 2. Comparing the value to `:undefined` therefore missed the absent binding. The weather example tests whether a guarded read of `.a` returned a list and uses `.os.argv` on the hosted path. The RC2 check is whether a single file-arguments variable works in a script on both hosts, including under strict mode.

### Checking a numeric threshold in a generated program

While measuring batched SQL publication after the RC1 source freeze, an agent wrote `full(n)::{n>=2048}`. `kg lint` returned zero errors and zero warnings. Calling `full(2048)` failed with `=: group needs a list`. The AST showed the actual interpretation as a greater-than comparison whose right operand groups the integer literal. In Klong, `full(n)::{~n<2048}` expresses this threshold. Checking 2047, 2048 and 2049 returns 0, 1 and 1. A targeted lint diagnostic for grouping an integer literal would catch this before the generated program reaches a benchmark runner.

### Identifying a locally rebuilt executable

On 2026-11-19 two macOS arm64 desktop executables reported the same `kg --version` and identical `kg --capabilities` JSON. Both said `2.0.0-rc.1`. One failed the rollback regression below and the other passed. When running workers from the same dirty checkout, the repository revision alone could not tell us which binary a report used.

Acceptance JUnit reports now record the SHA-256 of the selected `kg` command and the extension loaded by the test interpreter. The same CLI tests produced different command digests for two builds that both reported `2.0.0-rc.1`, and the same digest for their shared Python extension. When using a wrapper or an external `kgpy` environment, also record the binary and extension it actually launches.

We also saw a Python-hosted test start in the checkout and launch a child from a temporary directory. The child imported an older installed copy of `klongpy` and emitted a different diagnostic. Running `python -c 'import klongpy; print(klongpy.__file__)'` from both directories exposed the mismatch. An editable install or an explicit checkout `PYTHONPATH` gave the child the intended Python code.

On the same Mac, a worker tried `cargo build --release --manifest-path native/klong-py/Cargo.toml` from the repo root to rebuild the comparison host. Linking failed with unresolved Python symbols. Running `(cd native/klong-py && cargo build --release)` picked up the crate-local Cargo configuration and succeeded. Keep the working directory in copyable build commands and check them from a fresh shell.

### Building both desktop commands on a fresh worker

On a six-CPU Linux worker, we built `klong-cli` and then `native/klong-py` with the same absolute `CARGO_TARGET_DIR`. Each Cargo workspace built 305 DuckDB C++ objects in its own output directory. This made a small test-runner correction expensive to verify. On the replacement worker we set `CXX="ccache g++"` and used the same writable `CCACHE_DIR` for both builds. The CLI build took 7 minutes 16 seconds. The extension built in 42 seconds and `ccache -s` confirmed it reused all 305 objects. For RC2, document a tested cold-build command for the pair and how agents can reuse the C++ compilation between them.

### Finding a standard library function by what it does

While inspecting benchmark medians on 2026-11-19, `kg -e '.l("nstat");nstat.med([1 2 3])'` reported a missing member. Searching the library source turned up `nstat.M`. It returned `2`. I also tried `kg -e '.l("help");.l("nstat");help("median")'`; it printed only `[]`.

The RC2 check is whether someone starting with the word “median” can discover the qualified spelling and a working example from the installed command, without opening the library source.

### Running a file of named regression cases

On 2026-11-19, `kg tests/v2/programs/sql_cases.kg` exited successfully with empty stdout. The file defines functions. The Python harness discovers and invokes each one. For a direct red/green cycle I had to find the case name and invoke it explicitly from the repo root.

```sh
kg -e '.l("tests/v2/programs/sql_cases.kg");caseRollbackWithUnreadableView()'
```

The old binary stopped at `d("ROLLBACK")` with a type error. The fixed binary printed `1`. For RC2, give contributors an obvious command to run the whole case file with native `kg`, identify the cases that ran, and return a failing exit status when an assertion fails.

### Reading the heap limit after an allocation is refused

While checking that a rejected batch leaves its table intact on 2026-11-19, we ran this with both the desktop and table-only binaries.

```sh
kg --heap-limit 72K --stats -e 't::.table(:{;"sym";,:start;"n";,7});b::1024#,[;:next;17];r::.try({.insert(t;b)};[]);[;r@0;#t;+/t@"n"]'
```

Both printed `[0 1 7]`, confirming failure with the original row intact. The heap summary on stderr included `limit 0, refused 1`. We had to retain the command to know the configured budget. The RC2 check is that the diagnostic identifies the 72 KiB user budget alongside the refusal count.

### Inspecting invisible characters in a returned string

On 2026-11-19 we needed to distinguish embedded NUL from truncation in a SQL result. In `kg -e 'd::.db(:{});d("SELECT concat(chr(97),chr(0),chr(98))")'`, the displayed string contains a literal zero byte between `a` and `b`. A terminal makes that byte easy to miss. Piping the output through `od -An -tx1` showed `22 61 00 62 22 0a`. In a Klong script, `.p(.json(value))` makes the NUL visible as `\u0000`. For RC2, document how to inspect such values from the CLI and verify the example with a three-byte string.

We hit a similar display ambiguity when debugging IPC. A vector containing `1 2 3` still showed the expected numbers after a 32-bit vector had been widened to 64 bits. Asking the receiving q process for its type exposed the change from type 6 to type 7. Use the live peer check when the wire representation matters. [Vector round-trip regression](../test_ipc_peachq.py)

An empty q table exposed a related trap in the Python test harness. The table ``([]a:`long$();b:`symbol$())`` has zero rows and two columns. The default Python mapping reports two from `len()`; the optional pandas-backed table reports zero. The same test therefore gave different answers depending on which adapter was loaded. Check the ordered column names and the number of rows separately when verifying a table round-trip.

### Multiple agents testing on one machine

On 2026-11-19 we ran the legacy NumPy, Torch CPU, and custom-provider suites concurrently on one Mac. The simulated FPGA build tests reported another vendor build in progress, and a management test reported its TCP address was already in use. All seven build-flow cases and the management fan-out passed in serial replay with their original assertions. The FPGA fixture now supplies the process list of its simulated build machine. A separate regression exercises refusal when that list contains a vendor tool. Four concurrent copies of the entire 18-test module passed after the fixture waited for the actual lock and the status command read the finish marker after checking the PID. The latter fixes a real race where a successful build could briefly report `stopped`.

Sharing the machine also complicated the live feed benchmark. On one run the as-of client started its request 169 ms late at p99. A runtime query profile cannot account for time before the request starts. The benchmark now records the pre-request delay and round-trip p99 alongside the end-to-end latency used for acceptance. Use a quiet host for RC2 performance evidence. [Performance test](test_performance.py)

We reran the unchanged test in a six-CPU Linux pod on an AMD EPYC 7773X with PeachQ 0.92. All 20 reference cases passed. A second run of the performance case, scheduled before the first run, also passed. Across the three paired trials in each run, the median `kg`/PeachQ ingestion ratios were 1.665 and 1.664. The corresponding end-to-end p99 ratios were 0.923 and 0.869 for latest price, and 0.890 and 0.946 for as-of bid. Native `kg` computed the medians from the individual trial reports with `nstat.M(kgSamples%qSamples)`. The Linux executable's SHA-256 was `f22de863361f8993253b7e8b08f4940da9c7c27585e01c17180f3a7d6541d166`.

### Running the Python-hosted feed on Linux

In the same pod, `pytest tests/v2/test_performance.py::test_tick_service_under_load` measured 14 ms worst ingest lag for `kg` and 228 ms for `kgpy` with default CPU placement. The acceptance limit is 100 ms. Additional hosted runs reproduced the backlog. The late requests accumulated over the feed. Per-request timings and cgroup counters showed no CPU quota throttling during the three instrumented runs.

On the earlier build we prefixed the server commands with `taskset -c 8` and left the client free to run on other CPUs. The original two workload tests and the three-pair reference comparison passed with that placement. Worst ingest lag was 6.7 ms for `kg` and 8.5 ms for `kgpy`. A separate hosted run on CPU 16 gave 9.1 ms.

The later build with dictionary-backed pages for repeated text in live SQL scans passed three consecutive runs at default CPU placement. Worst ingest lag was 20.23, 17.28 and 10.71 ms for `kg`, and 24.28, 23.95 and 33.62 ms for `kgpy`. The highest query p99 across both commands was 27.30 ms. The same build passed all 20 live PeachQ reference tests. We verified the new scan directly with `kg -e '.l("tests/v2/programs/sql_cases.kg");caseLiveTextPages()'`, which returned `1`, and recorded both binaries' SHA-256 hashes alongside the reports.

For the relative PeachQ comparison, we timed both query bodies with `.now()` in a private copy of `stream.kg`. In three unrestricted trials the first last-price query ran for 10.24–11.51 ms before the as-of body could start. The as-of body took another 9.90–10.69 ms. Recording only aggregate client percentiles had obscured this ordering. The original scheduled p99 criterion and unrestricted result remain in the [performance measurements](../../docs/performance.md#streaming-tables-and-queries).

An earlier Linux rebuild had extracted changed Rust files from a tar archive with source modification times older than the cached Cargo build. Cargo reused an old artifact. We refreshed the changed files' timestamps before rebuilding the CLI and extension, and checked that both SHA-256 hashes differed from their baselines before collecting results. Use a fresh Cargo target directory for an independent release build when practical. When testing an overlay with a reused target directory, check the source timestamps, run the new regression against the built command and record the artifact hashes.

### Emitting one JSON record from `kg`

While checking benchmark maxima with RC2, `kg -e '.p(.json(:{;"max";|/[1 2 3]}))'` emitted two lines. The first was the JSON record printed by `.p` and the second was the REPL representation of the expression's return value. I put the same expression in a file and ran `kg result.kg`. That produced only `{"max":3}` and an independent JSON parser read it successfully. Use script mode for machine-readable stdout.

### Building the simulator for the executable docs

An RC2 build of `klong-sim` on its own changed the expected source spans in the console transcripts and silently accepted the fourth argument to `.param("n";"int";1;2)`. The diagnostic feature restored the spans. The wide-argument feature let `.param` see all four arguments and return the required `arity` error. The workspace build enables both through its desktop targets. The equivalent simulator-only command is `cargo build --locked --release --manifest-path native/Cargo.toml -p klong-sim --features klong-console/diag,klong-core/wide-args`. [Serial console build instructions](../../docs/native/console.md)

### A named parameter with the spelling `x`

While writing a Linux latency probe, `kg -e 'foo(x)::{x};foo(1)'` reported an undefined function at the call. Running the same expression with `--strict 1` located the definition and explained the reserved parameter names. Both `foo(a)::{a};foo(1)` and `foo::{x};foo(1)` return 1. Use strict mode when checking generated function definitions. [Named parameter rules](../../docs/language-additions.md#named-parameters)

### Checking the Pico version after a flash

On November 20 the bench supervisor verified the RC1 flash on node 2, but `klong_bench.py status` still showed its earlier `0.0.1` greeting. The firmware answered `2.0.0-rc.1` when asked directly. After a flash, use `python3 native/tools/klong_bench.py --host HOST eval --node 2 '.ver()?"version"'` and keep the supervisor's readback receipt to identify the exact image. The bridge caches the greeting between polls.

## Fixed in the working tree

### Interpreting the benchmark clock

While timing table corrections, the engine benchmark described `.pc()` as CPU time. Starting the clock before reading from a named pipe and delaying its writer by 250 ms produced 253 ms in `kg` and 252 ms in `kgpy` on the M2 Ultra. Both include time spent waiting. The benchmark instructions and performance guide now identify the desktop clock as monotonic elapsed time.

- **Importing variadic Python functions.** A `.kg` script run with `kgpy` on CPython 3.14 got `0.0` from both `.pyc("hypot";[3 4];:{})` and `hypot(3;4)` after `.pyf("math";"hypot")`. Python now exposes the function's variadic parameter as `*coordinates`. The importer checks the parameter kind, and both calls return `5.0`. [Variadic import regression](../test_webc_kvs_db_pyc.py)
- **Reopening a database after a hosted session ends.** On 2026-11-19 six discarded Python sessions survived garbage collection. A separate `kg` process could reopen a DuckDB writer’s file only after its Python process exited or explicitly cleared the last database binding. On the fixed bridge, an escaped Klong function still reads the table after its interpreter variable is dropped. After the final handle is released, a native child opens the file and reads `42`. The shared-value tests also cover callbacks installed after transfer, Python finalizers, and an unrelated interpreter blocked in a callback. [Database reproducer](test_storage.py) · [Shared-value tests](test_host_lifetime.py)
- **Parking shared host values and reading metadata.** Three groups of 96 short-lived consumers imported a dictionary while its source waited in a Python callback. The consumer weak references cleared, and parking released their pending Python references, but 288 numeric group entries survived after the source was dropped. The bridge now removes those entries once it has assigned pending references to the correct group for publication. On Apple Arm64, symbol membership and warning retrieval also rebuilt unchanged roots. Giving those two calls a scope that preserves active-interpreter tracking without marking roots dirty lowered CPU time for a predecoded 256-row ingest batch by about 7% in the local probe. [Ownership tests](../../native/klong-py/src/graph.rs) · [Feed workload](test_performance.py)
- **Loading sibling modules on Windows.** An installed `kg` and `kgpy` tested under Wine each failed a small `.kg` script whose module loaded `./child`; the diagnostic searched the working directory. Drive-qualified entries with spaces in `KLONGPATH` also failed. The loader now uses the canonical importing directory and Windows path-list separators. The rebuilt Windows `kg` passed the sibling reproducer and all 340 module cases, and macOS passed both complete conformance runs. [Load-order regression](../test_load_order.py)
- **Incomplete parser snapshots in an installed-wheel check.** Copying only `tests/golden` and running `conformance/run.py --trees --cmd kg` reported 139 matches. The language and library source files were missing, while their expected trees were present. The runner now reports which snapshots lack source files and exits nonzero. Staging `tests/kgtests/language` and `klongpy/lib` alongside the fixtures let the installed Linux wheel check all 169 trees. [Runner regression](../test_conformance.py)
- **Checking the commands installed by a wheel.** A fresh wheel environment still imported the checkout's `klongpy` when run from the checkout root. The conformance runner also adds its own parent directory to the import path. Run the installed `kg` and `kgpy` from a temporary directory with the native override unset, and copy the conformance directory there before testing the Python host. The [wheel workflow](../../.github/workflows/wheels.yml) now does both and verifies the package import is inside the fresh environment. We verified live SQL and both conformance runs from an installed macOS wheel.
- **Finding the evaluator from the agent instructions.** On 2026-11-19 the architecture map pointed to `klongpy/parser.py` and three Python verb and adverb files that no longer exist. Following the public interpreter import led to `rust_engine.py`. The [agent instructions](../../AGENTS.md) now name the shared Rust parser and evaluator and the Python provider hooks.
- **Arguments after the script name.** On 2026-11-19 passing `--stats` to a script enabled interpreter statistics and removed it from `.a`. Options following the filename now reach the script verbatim. The regression also covers an empty argument and a filename containing spaces. [CLI regression tests](test_cli_automation.py)
- **Discovering compile-time features.** To choose a compatible job, an agent previously needed to know the Klong expression for reading capabilities as JSON. `kg --help` now lists `--capabilities`, which works with stdin closed and returns the same data as `.ver()`. [CLI regression tests](test_cli_automation.py)

### Building Pico examples with the checkout's kg

On 2026-11-19, `native/boards/pico-bench/build.sh` built the Rust archive and then stopped in `gen_c_examples.py` with `ModuleNotFoundError: No module named 'numpy'`. The generator used native `kg compile c` for the example sources, then used NumPy and the Python extension to compute expected values. Selecting the project Python environment with `KLONG_PYTHON` and the matching extension with `KLONGPY_NATIVE_PATH` produced the UF2 and passed the memory layout check.

The generator now evaluates the examples and all 18 pinned results through standalone `kg`. It uses a private copy of the board module for PID's ADC and PWM stand-ins. We ran the generator with `python3 -S`, verified that all nine generated files matched the previous hosted output byte for byte, and linked the Pico firmware. Changing any one pin made generation fail. [Pico example generator](../../native/boards/pico-bench/tools/gen_c_examples.py)

### Grouping resident station and zone labels

The weather and taxi work prompted a standalone `kg` probe of `?` and `=`. On 200,000 resident rows cycling through 413 station strings, median time on the local Mac fell from 160 to 4.2 ms for distinct values and from 161 to 4.7 ms for group positions. Cycling through 265 symbols gave 50 to 1.6 ms and 51 to 2.2 ms respectively. These are synthetic operator timings. The probe builds its labels before the timer. The new path uses a dictionary index for string and symbol lists and preserves first-seen order. [Grouping cases](programs/group_keys.kg)

The first probe used positional script arguments. In `kg`, the first argument is `.a@0`. Passing the same argument list after the filename to `kgpy` gave an unrecognized-arguments error. For both commands we could load the probe with `.l` in `-e` and pass values to a Klong function.

### Closing the last hosted IPC client from a timer

Fixed during RC1 qualification. A timer callback that printed `"closing"` and called `.clic(c)` occasionally took five seconds to exit and printed an IPC timeout. The file runner saw the last connection disappear while the I/O loop was still closing it. Calling `close()` on that I/O loop could also block the loop that needed to finish the close. The regression tests hold the transport open until the test releases it and check that the file runner stays alive until the close completes. The original subprocess example then exited with only `closing` on all 25 runs. [IPC lifecycle tests](../test_ipc_lifecycle.py)

### Receiving a reply as an IPC call times out

Fixed during RC1 qualification. Two Python threads issued blocking calls through the same Klong connection after setting `.clit(c;0.5)`. If both callers resumed after their deadlines, each could raise a distinct timeout. A caller could also discard a reply or disconnect that the I/O thread had already delivered. The regression checks synchronize both calls after registration and delay a caller until a terminal result is available. Both callers now receive the same terminal error on timeout, and a delivered reply survives a scheduling delay. [Blocking-call regressions](../test_ipc_endpoint.py)

### Sending a journal operation over native IPC

While testing ticker plant rollover, the request `wal([;:closing;dayNo;nextSeq])` reached the journal with the integer `0` as its operation. The head already bound `closing` to its local session flag. Using the unbound operation symbol `:beginClose` preserved the name on the wire. The pending-rollover test restarts the head with the HDB offline and checks that the journal retained the close intent. [Ticker plant regression](test_tickerplant.py)

A direct native probe initially failed with `undefined function: .tickerwal`. The desktop build now registers this optional writer. The probe opens a local directory, appends a batch, rewinds, and checks every replayed field in Klong. [Native WAL probe](programs/tickerplant_native_wal.kg)

On the first launch of one freshly linked arm64 macOS build, three checks exceeded their startup deadlines. A simultaneous standalone WAL probe sampled entirely at `_dyld_start`, with only the executable and loader mapped. Once that probe finished, the same binary passed all 28 ticker-plant and rate-search checks in 6.45 seconds. The startup timeouts stayed in place.

The first native benchmark publisher wrote `>=` in a timer predicate. The runtime read two verbs, reported a group operation on a scalar, cancelled the timer, and the CLI then exited successfully. The publisher now spells the condition with Klong's negated less-than. The harness requires the terminal acknowledgment event as well as a successful exit, and rejects timer or asynchronous-call diagnostics. [Native benchmark clients](../../examples/stocks/tickerplant/native_clients.kg)

### Restarting successive ticker plant profiles on the same port

On the 100 GbE hosts, stopping one benchmark profile and immediately starting a fresh HDB twice produced an `address in use` error from `.srv`. A TCP connection check alone briefly succeeded during one handoff, but the new server had exited by the time the native client queried it. For sequential runs, check that the new process stays alive and answers an authenticated application request before starting the timer. Fresh ports on the same benchmark interfaces let the strict-deadline profile start immediately.

### Timing small native feed messages

The load generator and both subscriber processes ran on Tinkara. Each process had its own origin for `.pc()`. Subtracting those values across processes would give the wrong delivery latency. The benchmark uses a common host epoch to join subscriber receipts by watermark, and monotonic elapsed time for batch formation and acknowledgment within the publisher. Across Milan and Tinkara it records a short round-trip clock calibration before and after the sample. See the [native feed](../../examples/stocks/tickerplant/native_feed.kg) and [trace reader](../../examples/stocks/tickerplant/profile_report.py).

A 1 ms native timer can observe a 2 ms flush threshold late. One outstanding durable write can also delay sending an already sealed batch. The feed records sealing and sending separately, alongside scheduled arrival. This makes backpressure visible when a timer-based batching policy is used.
