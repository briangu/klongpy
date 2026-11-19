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

This section is a second set of measurements, from `native/scripts/bench_engine.py`. It times the engine inside one process, after a warm-up run, with no process launch per timing, and runs the same workloads in CPython and NumPy. It was made to decide what to optimise in the engine, and the "before" columns are the engine as it was before that work (the commit of 2026-11-18, "Packaging: one abi3 wheel with the extension, kgpy and kg"). The tables above come from another machine with another method and are not repeated here.

| Item | Value |
|---|---|
| Date | 2026-11-18 |
| Machine | Apple M2 Ultra, 24 cores, 192 GB of memory, macOS 26.6 |
| Load | not idle: the machine was shared, and `uptime` showed load averages of 9 to 14 during the runs (13.8, 12.9, 11.0 and 9.1 for the one-minute average at the start of the four runs) |
| Python, NumPy | 3.12.9, 2.5.5 |
| Rust | 1.90.0-nightly, `--release`, one code generation unit |
| kg | `native/target/release/kg`, timed with `.pc()`, the process CPU time, around a call of the workload |
| Extension | `klong_native` through `KlongInterpreter` from `klongpy` (NumPy backend), timed with the same Klong function (`.pc()`), so a result is never converted to Python |

Method: a workload runs once to warm up, and that run sizes a batch of at least 3 ms; the figure is the best of 3 to 30 batches (as many as fit in half a second), per run. A run of 2 s or more is timed once. Each cell is the better of two rounds, and the two rounds were run one after the other, "before" and "after" alternating. The machine was loaded by other work, so a ratio of two cells is good to about 10%: in the array rows that the work did not change, "before" divided by "after" is between 0.91 and 1.16. A workload is the body of `wk::{expr;0}`, so what is timed is making the result and dropping it. The CPython column uses lists, loops and the builtins that fit (`sum`, `min`, `sorted`, `itertools.accumulate`); the NumPy column is the vectorised form. The arrays are `ia`, ints 0 to n-1, `ib`, `3+ia`, `fa`, `0.5*ia`, `fb`, `1.5+fa`, `pi`, a permutation of 0 to n-1 (`(7919*ia)!n`) and `pf`, `0.5*pi`.

### Programs the interpreter runs

| Program | CPython | kg before | kg after | kg speed-up | extension before | extension after | kg after / CPython |
|---|---|---|---|---|---|---|---|
| fib(24) recursive | 4.16 ms | 28.5 ms | 20.8 ms | 1.37x | 27.7 ms | 21.1 ms | 5.01 |
| loop by while, 1e5 | 1.52 ms | 28.3 ms | 12.3 ms | 2.31x | 28.3 ms | 11.8 ms | 8.05 |
| loop by converge, 1e5 | 1.53 ms | 18.9 ms | 9.11 ms | 2.07x | 18.6 ms | 9.14 ms | 5.97 |
| reduce with a lambda, 1e5 | 4.07 ms | 15.3 ms | 7.09 ms | 2.15x | 15.1 ms | 6.86 ms | 1.74 |
| each with a lambda, 1e5 | 4.12 ms | 15.2 ms | 6.97 ms | 2.18x | 14.4 ms | 6.84 ms | 1.69 |
| dictionary updates, 5e4 | 1.74 ms | 18.8 ms | 14 ms | 1.34x | 17.2 ms | 13.4 ms | 8.05 |
| string building, 2e4 | 702 us | 27.1 ms | 19.5 ms | 1.39x | 27.3 ms | 18.9 ms | 27.8 |
| named-parameter calls, 1e5 | 5.88 ms | 25.3 ms | 16.5 ms | 1.54x | 24.9 ms | 16.3 ms | 2.80 |
| classic calls, 1e5 | 5.89 ms | 23.4 ms | 14.3 ms | 1.63x | 22.8 ms | 14.2 ms | 2.43 |
| qualified calls (dictionary), 1e5 | 6.77 ms | 26.5 ms | 17.2 ms | 1.54x | 26 ms | 17 ms | 2.54 |
| module-qualified calls, 1e5 | 6.84 ms | 29 ms | 19.9 ms | 1.46x | 28.8 ms | 19.6 ms | 2.91 |

* `kg` and the extension run the same engine: over all the rows they differ by 2% at the median and by at most 15% (the sort of floats at 10,000,000 elements).
* The engine is 1.3 to 2.3 times faster than before, and still 1.7 to 28 times slower than CPython on these programs: 1.7 for a lambda under an adverb (reduce and each), 2.4 to 2.9 for calls, 5 to 8 for recursion, loops and dictionary updates and 28 for building a string by joining.

### Arrays

The tables have one row per operation. "kg before" is the engine before the work, "kg after" the engine now, in the same units.

**1,000 elements**

| Operation | NumPy | CPython | kg before | kg after | kg speed-up | extension after | kg after / NumPy |
|---|---|---|---|---|---|---|---|
| `add int` | 0.477 us | 23.2 us | 0.455 us | 0.449 us | 1.01x | 0.425 us | 0.94 |
| `multiply int` | 0.771 us | 23.7 us | 0.683 us | 0.715 us | 0.96x | 0.702 us | 0.93 |
| `add float` | 0.467 us | 24.3 us | 0.444 us | 0.455 us | 0.98x | 0.426 us | 0.97 |
| `multiply float` | 0.473 us | 24.9 us | 0.468 us | 0.455 us | 1.03x | 0.426 us | 0.96 |
| `add mixed int/float` | 0.981 us | 30.3 us | 3.36 us | 0.45 us | 7.45x | 0.426 us | 0.46 |
| `chain (a+b)*b int` | 1.17 us | 32.8 us | 0.884 us | 0.802 us | 1.10x | 0.774 us | 0.69 |
| `chain (a*b+a)-b float` | 1.34 us | 29.4 us | 0.859 us | 0.739 us | 1.16x | 0.712 us | 0.55 |
| `dot +/a*b float` | 1.01 us | 26.9 us | 0.659 us | 0.668 us | 0.99x | 0.656 us | 0.66 |
| `compare int <` | 0.412 us | 18.5 us | 4.11 us | 0.463 us | 8.87x | 0.466 us | 1.12 |
| `sum int` | 0.483 us | 2.35 us | 0.311 us | 0.315 us | 0.99x | 0.308 us | 0.65 |
| `sum float` | 0.559 us | 2.04 us | 0.442 us | 0.451 us | 0.98x | 0.447 us | 0.81 |
| `min float` | 0.464 us | 4.63 us | 17.5 us | 0.443 us | 39.6x | 0.429 us | 0.95 |
| `max int` | 0.485 us | 6.42 us | 0.398 us | 0.407 us | 0.98x | 0.386 us | 0.84 |
| `scan +\ int` | 2.22 us | 15.4 us | 0.747 us | 0.728 us | 1.03x | 0.701 us | 0.33 |
| `grade up int` | 21.8 us | 55.1 us | 11.7 us | 12 us | 0.97x | 11.8 us | 0.55 |
| `grade up float` | 25.6 us | 52.4 us | 11.5 us | 9.94 us | 1.16x | 9.78 us | 0.39 |
| `sort float (x@<x)` | 26.2 us | 33.2 us | 22.9 us | 20.4 us | 1.12x | 20.8 us | 0.78 |
| `take h#a` | 0.306 us | 0.628 us | 0.349 us | 0.355 us | 0.98x | 0.348 us | 1.16 |
| `drop h_a` | 0.313 us | 0.588 us | 0.356 us | 0.353 us | 1.01x | 0.343 us | 1.13 |
| `reshape [2 h]:^a` | 0.449 us | 1.2 us | 7 us | 0.516 us | 13.6x | 0.484 us | 1.15 |
| `find a?(n-1)` | 1.03 us | 26.7 us | 5.97 us | 0.655 us | 9.12x | 0.655 us | 0.64 |
| `where &(p>h)` | 1.5 us | 22.9 us | 9.51 us | 2.65 us | 3.59x | 2.86 us | 1.76 |
| `each-pair a{x+y}'b` | 0.485 us | 37.2 us | 161 us | 72.5 us | 2.22x | 72.8 us | 150 |

**100,000 elements**

| Operation | NumPy | CPython | kg before | kg after | kg speed-up | extension after | kg after / NumPy |
|---|---|---|---|---|---|---|---|
| `add int` | 18.3 us | 2.48 ms | 19.2 us | 18.3 us | 1.05x | 20.8 us | 1.00 |
| `multiply int` | 35.2 us | 2.45 ms | 37.6 us | 37.5 us | 1.00x | 37.5 us | 1.07 |
| `add float` | 23.8 us | 2.32 ms | 18.2 us | 19.6 us | 0.93x | 18.3 us | 0.83 |
| `multiply float` | 23.8 us | 2.31 ms | 18.3 us | 19.6 us | 0.93x | 18.4 us | 0.82 |
| `add mixed int/float` | 55.8 us | 2.91 ms | 301 us | 18.7 us | 16.1x | 19.1 us | 0.33 |
| `chain (a+b)*b int` | 59.4 us | 3.13 ms | 59.4 us | 53.7 us | 1.11x | 56.3 us | 0.90 |
| `chain (a*b+a)-b float` | 60.4 us | 2.74 ms | 58 us | 63.3 us | 0.92x | 65.5 us | 1.05 |
| `dot +/a*b float` | 36.6 us | 2.72 ms | 41.5 us | 41.1 us | 1.01x | 39.4 us | 1.12 |
| `compare int <` | 18.3 us | 1.6 ms | 380 us | 19.7 us | 19.3x | 18.5 us | 1.08 |
| `sum int` | 9.2 us | 247 us | 9.02 us | 9.16 us | 0.99x | 8.98 us | 1.00 |
| `sum float` | 18.1 us | 231 us | 20.7 us | 20.4 us | 1.02x | 20.2 us | 1.13 |
| `min float` | 9.26 us | 458 us | 1.77 ms | 15.9 us | 111x | 15.9 us | 1.72 |
| `max int` | 9.27 us | 605 us | 16 us | 16 us | 1.00x | 15.9 us | 1.72 |
| `scan +\ int` | 182 us | 1.52 ms | 109 us | 42.8 us | 2.56x | 39.8 us | 0.23 |
| `grade up int` | 5.59 ms | 7.43 ms | 2.26 ms | 2.29 ms | 0.99x | 2.23 ms | 0.41 |
| `grade up float` | 3.97 ms | 7.54 ms | 7.45 ms | 1.87 ms | 3.99x | 1.83 ms | 0.47 |
| `sort float (x@<x)` | 4.11 ms | 4.29 ms | 8.64 ms | 3.06 ms | 2.82x | 3.04 ms | 0.74 |
| `take h#a` | 5.53 us | 96.6 us | 6.39 us | 5.64 us | 1.13x | 5.46 us | 1.02 |
| `drop h_a` | 5.53 us | 79.6 us | 6.72 us | 5.81 us | 1.16x | 5.44 us | 1.05 |
| `reshape [2 h]:^a` | 10.5 us | 195 us | 652 us | 11.1 us | 58.8x | 12 us | 1.05 |
| `find a?(n-1)` | 26.9 us | 2.89 ms | 551 us | 35.4 us | 15.6x | 35.4 us | 1.32 |
| `where &(p>h)` | 75 us | 2.46 ms | 885 us | 251 us | 3.52x | 232 us | 3.35 |
| `each-pair a{x+y}'b` | 19.1 us | 3.52 ms | 15.7 ms | 7.43 ms | 2.12x | 7.21 ms | 389 |

**10,000,000 elements**

| Operation | NumPy | CPython | kg before | kg after | kg speed-up | extension after | kg after / NumPy |
|---|---|---|---|---|---|---|---|
| `add int` | 2.61 ms | 300 ms | 2.6 ms | 2.7 ms | 0.96x | 2.65 ms | 1.03 |
| `multiply int` | 3.53 ms | 309 ms | 3.7 ms | 3.91 ms | 0.95x | 3.65 ms | 1.11 |
| `add float` | 2.61 ms | 294 ms | 2.59 ms | 2.71 ms | 0.96x | 2.6 ms | 1.04 |
| `multiply float` | 2.75 ms | 311 ms | 2.59 ms | 2.79 ms | 0.93x | 2.61 ms | 1.01 |
| `add mixed int/float` | 6 ms | 373 ms | 30.4 ms | 2.78 ms | 11.0x | 2.62 ms | 0.46 |
| `chain (a+b)*b int` | 6.72 ms | 383 ms | 6.27 ms | 6.87 ms | 0.91x | 6.47 ms | 1.02 |
| `chain (a*b+a)-b float` | 7.97 ms | 359 ms | 7.88 ms | 7.91 ms | 1.00x | 7.76 ms | 0.99 |
| `dot +/a*b float` | 4.46 ms | 282 ms | 4.75 ms | 4.8 ms | 0.99x | 4.7 ms | 1.07 |
| `compare int <` | 2.58 ms | 173 ms | 38.3 ms | 2.65 ms | 14.4x | 2.6 ms | 1.03 |
| `sum int` | 1.37 ms | 25.7 ms | 1.35 ms | 1.38 ms | 0.98x | 1.37 ms | 1.01 |
| `sum float` | 1.86 ms | 23.6 ms | 2.13 ms | 2.16 ms | 0.99x | 2.1 ms | 1.16 |
| `min float` | 1.38 ms | 46.6 ms | 176 ms | 1.68 ms | 105x | 1.64 ms | 1.21 |
| `max int` | 1.4 ms | 62 ms | 1.59 ms | 1.68 ms | 0.95x | 1.66 ms | 1.20 |
| `scan +\ int` | 18.7 ms | 261 ms | 4.83 ms | 5.21 ms | 0.93x | 4.56 ms | 0.28 |
| `grade up int` | 797 ms | 773 ms | 355 ms | 362 ms | 0.98x | 356 ms | 0.45 |
| `grade up float` | 409 ms | 767 ms | 894 ms | 242 ms | 3.69x | 243 ms | 0.59 |
| `sort float (x@<x)` | 436 ms | 346 ms | 1.03 s | 437 ms | 2.36x | 378 ms | 1.00 |
| `take h#a` | 680 us | 9.55 ms | 675 us | 709 us | 0.95x | 687 us | 1.04 |
| `drop h_a` | 680 us | 8.79 ms | 672 us | 693 us | 0.97x | 687 us | 1.02 |
| `reshape [2 h]:^a` | 1.42 ms | 20.3 ms | 67.9 ms | 1.44 ms | 47.2x | 1.42 ms | 1.01 |
| `find a?(n-1)` | 3.24 ms | 295 ms | 55.8 ms | 3.57 ms | 15.6x | 3.53 ms | 1.10 |
| `where &(p>h)` | 8.65 ms | 254 ms | 91.6 ms | 20.9 ms | 4.39x | 21.1 ms | 2.41 |
| `each-pair a{x+y}'b` | 2.67 ms | 409 ms | 1.62 s | 750 ms | 2.16x | 752 ms | 281 |

* Element-wise arithmetic, sums and `take` and `drop` run at the speed of NumPy (0.8 to 1.2 times its time) at every size, and did before. At 10,000,000 elements they are bound by memory: the same number of bytes are read and written as NumPy's.
* What changed is the operations that went through one boxed value per element: comparisons (8.9 to 19 times faster), mixed integer and float arithmetic (7 to 16 times), the minimum and maximum of floats (40 to 111 times), reshape (14 to 59 times), find (9 to 16 times), where (3.5 to 4.4 times), the grade and sort of floats (2.4 to 4.0 times at 100,000 and 10,000,000 elements). After the work, `kg` takes 0.3 to 1.3 times as long as NumPy on all of these except the ones listed next.
* Still slower than NumPy: `&/` and `|/` at 1.2 to 1.7 times (eight lanes, not NumPy's SIMD minimum), `where` at 1.8 to 3.4 times, and the each-pair row, which calls a Klong lambda per pair (73 ns per pair here; NumPy does not call a function).
* The grade of integers takes 0.41 to 0.55 times as long as NumPy's stable argsort, the grade of floats 0.39 to 0.59 times and the sort of floats 0.74 to 1.0 times.

### What the profiles said

Profiles were taken with `sample` (macOS, 1 ms interval, the time at the top of the stack, symbolised from a release build with line tables). They covered the rows with the worst ratios and decided the order of the work:

* **Calls driven by an adverb** (`{x*x}'a`, `{x+y}/a`, `:~`): about a third of the time was in the allocator (`malloc` and `free`), from the argument vector, the list of layers of a projection and a frame that did not come from the frame pool, made for every call by `apply`. A function literal called with exactly the arguments it takes now skips those and uses the pool: 1.3 to 1.9 times faster.
* **Recursion** (fib): no single place. `eval_node` 20% of the time itself, `eval_dyad` 11%, name resolution (`resolve` and `lookup`) 13%, copying and dropping values 12%, pushing and popping frames 7%, and the stack check and the deferred-failure check at the head of every evaluation 6%. Two numbers in a dyad now skip the pairing routine (1.13 times) and the two checks are inlined (1.12 times).
* **String building** (`s::s,"ab"`): 60% in `memmove` (the whole string is copied by every join, as the value semantics of strings say), 20% counting the code points of the result again, 17% in the allocator. The count is now the sum of the two counts (1.38 times). The copy remains.
* **Dictionary updates**: the allocator, dropping values, and the hash probe; each `d,x,,x*2` builds two small lists. Only the calls part was cut (1.3 times).
* **Comparison, mixed arithmetic, float minimum, reshape, find, where**: the loop itself. They went through the generic element loop (a division per element to find the operand, a boxed `Value` per element, dispatch per element), not through the vectorised loops that addition and multiplication have. They now have loops the compiler vectorises, or a path that copies the numbers instead of boxing them.
* **Grade and sort of floats**: 85% in the sort's comparison, which goes through the index to the number (a cache miss per comparison). Sorting (key, index) pairs, the key an integer that orders as the float does, makes it 3.7 to 4.0 times faster.

What limits the engine now, from the same profiles: the tree walk (dispatch in `eval_node` and `eval_dyad`, `Value` clones and drops, the frame push and pop with its table of shadowed names), the value semantics of strings and of dictionary entries (a new object per join and per entry), the lambda call under an adverb (70 to 80 ns), and, at 10,000,000 elements, memory.

### What each change did

Measured alone, on `kg`, the same machine and load, against the engine without that change. Every change but the last two is behind the cargo feature `speed` of `klong-core`, which `kg` and the extension enable and the board images do not.

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
