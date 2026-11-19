# Performance

This page says where KlongPy is fast, where it is slow and what to write to stay on the fast side. Read it before you choose a backend or move a loop into a program.

KlongPy is fast when one operator works on a whole array: for element-wise arithmetic, reductions and sorting on 100,000 to a million elements the NumPy backend takes at most 1.3 times as long as NumPy, and array power takes 2.1 to 2.3 times as long. KlongPy is slow when the interpreter does the work itself, in loops, recursion, per-element lambdas and dictionaries: the NumPy backend takes 7 to 420 times as long as CPython on the programs below. The [native runtime](native/index.md) is 10 to 63 times faster than the NumPy backend on those programs and 1.8 to 26 times slower than it on array work. The torch backend on the CPU pays off from about a million elements, and mostly with many threads.

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
| kg | `native/target/release/kg`, timed with `.pc()`, the process CPU time, around a call of the workload |
| Extension | `klong_native` through `KlongInterpreter` from `klongpy` (NumPy backend), timed with the same Klong function (`.pc()`), so a result is never converted to Python |

Method: a workload runs once to warm up, and that run sizes a batch of at least 3 ms; the figure is the best of 3 to 30 batches (as many as fit in half a second), per run. A run of 2 s or more is timed once. Each cell is the better of two rounds, and the two rounds were run one after the other. "Before" and "after" were measured in different sessions, under the loads stated above, so the rows that the work did not change show the noise between sessions: in the 39 array rows that no change was aimed at (add, multiply, the two chains, dot, sum, max of integers, the grade of integers, take and drop, at the three sizes), "before" divided by "after" is between 0.90 and 1.11 in 37 rows and 0.75 and 0.76 in two (`add float` and `multiply float` at 100,000 elements, discussed below). A ratio of two cells from different sessions is good to about 10%. A workload is the body of `wk::{expr;0}`, so what is timed is making the result and dropping it. The CPython column uses lists, loops and the builtins that fit (`sum`, `min`, `sorted`, `itertools.accumulate`); the NumPy column is the vectorised form. The arrays are `ia`, ints 0 to n-1, `ib`, `3+ia`, `fa`, `0.5*ia`, `fb`, `1.5+fa`, `pi`, a permutation of 0 to n-1 (`(7919*ia)!n`) and `pf`, `0.5*pi`. The two append rows were added to the script after the first measurement, so they have no "before" figure.

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
