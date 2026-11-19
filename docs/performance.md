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

For single-thread rows, set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, pass `--torch-threads 1` and pin the process to one core.
