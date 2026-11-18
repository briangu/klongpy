# Performance

This page says where KlongPy is fast, where it is slow and what to write to stay on the fast side. Read it before you choose a backend or move a loop into a program.

KlongPy is fast when one operator works on a whole array: from 100,000 to a million elements it takes about as long as NumPy (at most 1.4 times as long). It is slow when the interpreter does the work itself, in loops, recursion, per-element lambdas and dictionaries, where it takes 60 to 440 times as long as CPython. The [native runtime](native/index.md) is about 24 to 66 times faster than KlongPy on that interpreter work and 2 to 22 times slower on array work. The torch backend on the CPU pays off from about a million elements, and only with many threads.

## Conditions

| Item | Value |
|---|---|
| Date | 2026-11-17 |
| Machine | AMD Ryzen 9 7950X3D, 16 cores (32 threads), 62 GB of memory |
| Operating system | Ubuntu 20.04.6 LTS, Linux 5.4 |
| Python | 3.12.16 |
| NumPy | 2.5.5 (OpenBLAS) |
| PyTorch | 2.15.0, CPU build; 16 threads by default |
| Native runtime | the `--release` build of `native/` (rustc 1.100.0), run as `klong --bench 1 -e PROGRAM` |
| GPU | none measured |

Method: each row runs 3 untimed warm-up runs, then 9 timed runs. The tables show the best run and the median, as `best / median`.

* **Single thread.** The process is pinned to one core and sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` to 1, and torch to one thread. These are the main tables.
* **Default threads.** The same rows with no pinning and the default thread count of NumPy and torch. The native runtime has one thread and is not repeated.

In the vector, reduce and sort tables, rows of 100,000 elements or fewer are timed in batches of about 20 ms so that the timer resolution does not matter. The script is `scripts/bench_performance.py`; the last section shows how to rerun it.

## Programs the interpreter runs

These programs spend their time in the interpreter, not in an array library. The thread count does not change them, so the single-thread figures are shown. The CPython column is the same work written in plain Python: `fib` recursion, a `while` loop, `functools.reduce` with a lambda, `itertools.accumulate`, a list comprehension, a dictionary and `str.join`.

| Program | KlongPy, NumPy backend | KlongPy, torch backend | Native runtime | CPython |
|---|---|---|---|---|
| recursive fib(24) | 1.37 s / 1.38 s | 2.65 s / 2.66 s | 20.8 ms / 21.1 ms | 3.14 ms / 3.17 ms |
| while loop to 100000 | 598 ms / 601 ms | 1.26 s / 1.27 s | 21.1 ms / 21.5 ms | 1.45 ms / 1.47 ms |
| `{x+y}/!100000` | 338 ms / 339 ms | 532 ms / 541 ms | 11.5 ms / 11.6 ms | 3.24 ms / 3.25 ms |
| `{x+y}\!10000` | 33.9 ms / 34.1 ms | 55.9 ms / 56.7 ms | 1.23 ms / 1.33 ms | 421 us / 429 us |
| `#{x*x}'!100000` | 329 ms / 330 ms | 549 ms / 563 ms | 12.4 ms / 12.8 ms | 2.01 ms / 2.03 ms |
| `+/{x*x}'!1000000` | 3.31 s / 3.32 s | 5.62 s / 5.66 s | 120 ms / 129 ms | 29.6 ms / 29.7 ms |
| append in a loop, 2000 | 103 ms / 103 ms | 1.12 s / 1.13 s | 544 us / 569 us | 1.74 ms / 1.79 ms |
| 50000 dictionary inserts and lookups | 482 ms / 484 ms | 1 s / 1.01 s | 19.3 ms / 19.5 ms | 3.03 ms / 3.07 ms |
| 20000 string keys | 220 ms / 221 ms | 370 ms / 374 ms | 9.35 ms / 9.82 ms | 1.81 ms / 1.83 ms |
| format and join 20000 numbers | 19.2 ms / 19.8 ms | 48.5 ms / 49.1 ms | 1.77 ms / 1.98 ms | 818 us / 841 us |
| find in a 60000 character text | 13.8 ms / 13.9 ms | 16.1 ms / 16.3 ms | 539 us / 596 us | 78.4 us / 83.1 us |
| index a non-ASCII string, 2000 | 376 ms / 380 ms | 476 ms / 487 ms | 264 us / 284 us | 29.7 us / 29.7 us |

What the table shows:

* On the NumPy backend, KlongPy takes 60 to 440 times as long as CPython on recursion, loops, lambdas, appends and dictionaries. Finding text takes 176 times as long and formatting numbers 24 times.
* The native runtime is 24 to 66 times faster than KlongPy on recursion, loops, lambdas and dictionaries, and 11 times faster on formatting numbers. It is 189 times faster on appending in a loop and 1,400 times faster on indexing a non-ASCII string.
* The torch backend takes 1.2 to 2.5 times as long as the NumPy backend on these programs, and 10.9 times as long for appending.
* Known limit: indexing a long non-ASCII string element by element is slow, and the cost grows with the square of the length (the last row).

## Vector arithmetic

The expression is `2*1+!N`: a range, an addition and a multiplication, each over the whole array.

Single thread:

| N | NumPy | KlongPy, NumPy backend | KlongPy, torch backend | Native runtime |
|---|---|---|---|---|
| 1,000 | 1.56 us / 1.57 us | 5.05 us / 5.11 us | 7.74 us / 7.86 us | 21.4 us / 23.7 us |
| 10,000 | 4.95 us / 4.96 us | 8.17 us / 8.24 us | 15 us / 15 us | 83.1 us / 106 us |
| 100,000 | 37.3 us / 37.4 us | 47.2 us / 47.3 us | 89.2 us / 90.2 us | 895 us / 984 us |
| 1,000,000 | 360 us / 363 us | 425 us / 429 us | 798 us / 843 us | 9.33 ms / 10.1 ms |
| 10,000,000 | 10.1 ms / 10.3 ms | 17.7 ms / 18 ms | 11.4 ms / 11.5 ms | 107 ms / 108 ms |

Default threads:

| N | NumPy | KlongPy, NumPy backend | KlongPy, torch backend |
|---|---|---|---|
| 1,000 | 1.64 us / 1.67 us | 5.02 us / 5.07 us | 8.5 us / 8.51 us |
| 10,000 | 5.19 us / 5.2 us | 8.46 us / 8.46 us | 15.8 us / 15.8 us |
| 100,000 | 42.1 us / 42.1 us | 49.1 us / 49.2 us | 49.6 us / 52 us |
| 1,000,000 | 341 us / 429 us | 444 us / 450 us | 123 us / 128 us |
| 10,000,000 | 12.6 ms / 13.3 ms | 17.2 ms / 17.3 ms | 8.8 ms / 9.07 ms |

* The NumPy backend is 3.2 times slower than NumPy at 1,000 elements, because the interpreter costs a few microseconds per expression. At 100,000 and 1,000,000 elements it is 1.2 to 1.3 times slower, and at 10,000,000 it is 1.8 times slower on one thread.
* The torch backend is slower than NumPy on one thread up to a million elements. With 16 threads it is 3.6 times faster at 1,000,000 elements and 2 times faster at 10,000,000.
* The native runtime is 4 to 22 times slower than the NumPy backend on this work.

## Reduce, scan and power

Single thread:

| Expression | N | NumPy | KlongPy, NumPy backend | KlongPy, torch backend | Native runtime |
|---|---|---|---|---|---|
| `+/!N` | 100,000 | 24.3 us / 24.5 us | 27.4 us / 27.5 us | 35.1 us / 35.3 us | 102 us / 164 us |
| `+/!N` | 1,000,000 | 239 us / 242 us | 240 us / 241 us | 285 us / 290 us | 886 us / 1.49 ms |
| `+/!N` | 10,000,000 | 7.08 ms / 7.18 ms | 7.31 ms / 7.36 ms | 3.95 ms / 3.98 ms | 16.6 ms / 17.1 ms |
| `+\!N` (scan) | 100,000 | 45.5 us / 45.6 us | 49.1 us / 49.8 us | 52.7 us / 53.4 us | 210 us / 311 us |
| `+\!N` (scan) | 1,000,000 | 944 us / 958 us | 456 us / 457 us | 476 us / 481 us | 2.75 ms / 3.17 ms |
| `+/(!N)*(!N)` (dot product) | 100,000 | 74.4 us / 74.4 us | 66 us / 66.4 us | 102 us / 109 us | 741 us / 750 us |
| `+/(!N)*(!N)` (dot product) | 1,000,000 | 1.21 ms / 1.25 ms | 1.68 ms / 1.7 ms | 918 us / 921 us | 6.44 ms / 7.44 ms |
| `(1+!N)^2` (power) | 100,000 | 36.6 us / 36.7 us | 108 us / 111 us | 255 us / 260 us | 794 us / 870 us |
| `(1+!N)^2` (power) | 1,000,000 | 355 us / 360 us | 1.71 ms / 1.71 ms | 2.4 ms / 2.41 ms | 8.96 ms / 9.29 ms |
| `+100 100^!10000` (transpose) | 10,000 | 4.46 us / 4.47 us | 2.55 us / 2.56 us | 3.77 us / 3.78 us | 8.31 us / 13.1 us |

Default threads:

| Expression | N | NumPy | KlongPy, NumPy backend | KlongPy, torch backend |
|---|---|---|---|---|
| `+/!N` | 100,000 | 26 us / 26.1 us | 29 us / 29 us | 23.2 us / 24.2 us |
| `+/!N` | 1,000,000 | 239 us / 240 us | 249 us / 252 us | 55.8 us / 57 us |
| `+/!N` | 10,000,000 | 7.75 ms / 7.93 ms | 6.86 ms / 7.03 ms | 747 us / 2.11 ms |
| `+\!N` (scan) | 100,000 | 47.9 us / 48 us | 49.7 us / 49.7 us | 60.8 us / 61.1 us |
| `+\!N` (scan) | 1,000,000 | 1.76 ms / 1.83 ms | 456 us / 463 us | 474 us / 481 us |
| `+/(!N)*(!N)` (dot product) | 100,000 | 78 us / 78.4 us | 69.1 us / 69.2 us | 61 us / 62.1 us |
| `+/(!N)*(!N)` (dot product) | 1,000,000 | 1.45 ms / 2.11 ms | 1.71 ms / 1.71 ms | 113 us / 118 us |
| `(1+!N)^2` (power) | 100,000 | 40.6 us / 40.7 us | 109 us / 111 us | 108 us / 120 us |
| `(1+!N)^2` (power) | 1,000,000 | 342 us / 346 us | 1.76 ms / 1.77 ms | 232 us / 237 us |
| `+100 100^!10000` (transpose) | 10,000 | 4.6 us / 4.6 us | 2.36 us / 2.37 us | 3.84 us / 3.85 us |

* A reduce (`+/`) and a dot product on the NumPy backend take 0.9 to 1.4 times as long as NumPy. The scan `+\` at a million elements is 2 times faster than `np.cumsum`.
* `(1+!N)^2` takes 3 to 5 times as long as NumPy.
* With 16 threads the torch backend is 4.5 times faster than the NumPy backend for `+/!N` at a million elements, 15 times faster for the dot product and 7.6 times faster for the power. At 100,000 elements the two are within 1.25 times of each other, and on one thread torch is slower in most rows.

## Grade and sort

`<x` (grade) returns the sort order of `x`. `x@<x` sorts by indexing with it. Single thread:

| Expression | N | NumPy | KlongPy, NumPy backend | KlongPy, torch backend | Native runtime |
|---|---|---|---|---|---|
| `<!N` (grade of a sorted range) | 10,000 | 17 us / 17 us | 19.3 us / 19.3 us | 54.1 us / 54.1 us | 33.2 us / 59.4 us |
| `<!N` (grade of a sorted range) | 100,000 | 161 us / 161 us | 164 us / 164 us | 978 us / 1 ms | 313 us / 471 us |
| `<!N` (grade of a sorted range) | 1,000,000 | 2.08 ms / 2.08 ms | 1.6 ms / 1.6 ms | 13.1 ms / 13.9 ms | 4.33 ms / 4.54 ms |
| `<r` (grade, random floats) | 10,000 | 230 us / 231 us | 233 us / 233 us | 400 us / 408 us | n/a |
| `<r` (grade, random floats) | 100,000 | 3.54 ms / 3.55 ms | 3.55 ms / 3.56 ms | 5.81 ms / 5.82 ms | n/a |
| `<r` (grade, random floats) | 1,000,000 | 44.7 ms / 46 ms | 44.6 ms / 44.8 ms | 70.6 ms / 71.7 ms | n/a |
| `r@<r` (sort, random floats) | 10,000 | 236 us / 236 us | 6.12 ms / 6.16 ms | 13.9 ms / 14.2 ms | n/a |
| `r@<r` (sort, random floats) | 100,000 | 3.62 ms / 3.63 ms | 63.2 ms / 64.3 ms | 186 ms / 192 ms | n/a |
| `r@<r` (sort, random floats) | 1,000,000 | 46.4 ms / 47.6 ms | 662 ms / 671 ms | 2.02 s / 2.1 s | n/a |

Default threads:

| Expression | N | NumPy | KlongPy, NumPy backend | KlongPy, torch backend |
|---|---|---|---|---|
| `<!N` (grade of a sorted range) | 10,000 | 17.5 us / 17.5 us | 19.9 us / 19.9 us | 56.2 us / 56.9 us |
| `<!N` (grade of a sorted range) | 100,000 | 169 us / 170 us | 170 us / 174 us | 594 us / 597 us |
| `<!N` (grade of a sorted range) | 1,000,000 | 3.02 ms / 3.15 ms | 1.67 ms / 1.68 ms | 2.74 ms / 3.25 ms |
| `<r` (grade, random floats) | 10,000 | 236 us / 236 us | 238 us / 238 us | 421 us / 428 us |
| `<r` (grade, random floats) | 100,000 | 3.71 ms / 3.72 ms | 3.63 ms / 3.63 ms | 6.59 ms / 6.64 ms |
| `<r` (grade, random floats) | 1,000,000 | 58.5 ms / 59 ms | 45.7 ms / 46.2 ms | 73.1 ms / 73.4 ms |
| `r@<r` (sort, random floats) | 10,000 | 241 us / 241 us | 5.98 ms / 6 ms | 16.5 ms / 17.1 ms |
| `r@<r` (sort, random floats) | 100,000 | 3.82 ms / 3.84 ms | 62.2 ms / 63.4 ms | 214 ms / 220 ms |
| `r@<r` (sort, random floats) | 1,000,000 | 64.2 ms / 66.4 ms | 647 ms / 656 ms | 2.26 s / 2.64 s |

* Grade takes as long as `np.argsort`. The torch backend takes 1.6 to 3.5 times longer at every size, with one or many threads, and 8 times longer on one thread for a sorted range of a million.
* `r@<r` takes 14 to 26 times as long as `r[np.argsort(r)]`. Known limit: indexing by a vector is slow. A library call sorts the same data much faster: after `.pyf("numpy";"sort")`, `sort(r)` took 49 ms for a million random floats in a trial run.

## Matrix multiply through `.pyf`

`.pyf` imports a Python function, so these rows time a library call: the work is done by NumPy or torch, and KlongPy adds no measurable time. Single thread:

| N (N x N, float32) | NumPy | `.pyf` to NumPy | torch | `.pyf` to torch |
|---|---|---|---|---|
| 1000 | 12 ms / 12 ms | 12 ms / 12 ms | 12.6 ms / 12.9 ms | 12.3 ms / 12.5 ms |
| 2000 | 96.1 ms / 96.4 ms | 95.9 ms / 96.5 ms | 101 ms / 102 ms | 100 ms / 101 ms |
| 4000 | 764 ms / 766 ms | 765 ms / 766 ms | 822 ms / 824 ms | 821 ms / 824 ms |

Default threads:

| N (N x N, float32) | NumPy | `.pyf` to NumPy | torch | `.pyf` to torch |
|---|---|---|---|---|
| 1000 | 1.57 ms / 1.67 ms | 1.43 ms / 1.5 ms | 2.21 ms / 2.43 ms | 2.37 ms / 2.55 ms |
| 2000 | 7.8 ms / 7.85 ms | 7.91 ms / 8 ms | 10.7 ms / 16.6 ms | 15.1 ms / 16.2 ms |
| 4000 | 62.9 ms / 63.2 ms | 62.5 ms / 62.9 ms | 102 ms / 104 ms | 99.4 ms / 106 ms |

On the CPU, NumPy is as fast as torch or faster for this float32 product: torch takes 1.6 to 1.9 times as long with 16 threads. Choose torch for a matrix multiply to run it on a GPU, not for the CPU.

## What to write

* Write whole-array expressions. `+/x*x` does the work in the array library; `+/{x*x}'x` calls a lambda per element and takes 112 times as long as the same loop in CPython.
* Avoid per-element lambdas (`f'x`) on large arrays when an operator does the job.
* Avoid appending to a list in a loop; build the result with one expression.
* Use the grade `<x` when you need the order. `x@<x` is slow (known limit: indexing by a vector).
* Avoid indexing a long non-ASCII string one element at a time.
* Use the NumPy backend for arrays under a million elements. Use the torch backend on the CPU for reductions, dot products and element-wise work on arrays of a million elements or more, with several threads, and for gradients (see [PyTorch backend](torch_backend.md)). It is slower for sorting and for scalar programs.
* For a loop that must be fast, use the [native runtime](native/index.md) when the program fits, or compile a [static-profile](static-profile.md) function to C.

## GPU

No GPU was measured in this run, so this page gives no GPU numbers. To time your own, create the interpreter with `device="cuda"` (or `"mps"`) and wait for the device before you stop the clock, because GPU calls return before the work finishes:

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
python scripts/bench_performance.py --native native/target/release/klong --out native.json
python scripts/bench_performance.py --baseline --out baseline.json
```

For single-thread rows, set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, pass `--torch-threads 1` and pin the process to one core.
