# Performance

The Klong language is simple, so the overhead is low. The bulk of the compute time will likely be spent in the array backend doing actual work.

GPU acceleration helps compute-bound operations such as matrix multiply. The element-wise results below depend on the backend and the data type, and on whether the timer waits for the GPU (see the notes under the table).

Use `kgpy --backend torch` to enable the PyTorch backend.

## Benchmark

The benchmark script is `tests/perf_vector.py`. It tests two workload types:

1. **Vector ops** (element-wise, memory-bound): `2*1+!10000000`
2. **Matrix multiply** (compute-bound): 4000×4000 matmul

## Results

### Apple Silicon (MPS), measured 2026-11-16

Machine: Apple M2 Ultra, macOS 26.6, Python 3.12.9, NumPy 2.5.5, PyTorch 2.15.0. The machine was not idle (load average about 9 from other work), so treat differences of 10 to 20 percent as noise. Two consecutive runs of `python tests/perf_vector.py` are shown as run 1 and run 2 (seconds per iteration, and the ratio to NumPy):

| Workload | NumPy | KlongPy numpy | KlongPy torch, cpu | KlongPy torch, mps |
|---|---|---|---|---|
| Vector ops, run 1 | 0.007051 | 0.006949 (1.01x) | 0.002335 (3.02x) | 0.000684 (10.31x) |
| Vector ops, run 2 | 0.007259 | 0.007020 (1.03x) | 0.002201 (3.30x) | 0.000682 (10.64x) |
| Matrix multiply, run 1 | 0.030635 | 0.032075 (0.96x) | 0.030842 (0.99x) | 0.006343 (4.83x) |
| Matrix multiply, run 2 | 0.033059 | 0.029536 (1.12x) | 0.028752 (1.15x) | 0.006377 (5.18x) |

```bash
$ python tests/perf_vector.py
```

Notes on reading the table:

* The matrix multiply timing waits for the GPU (`torch.mps.synchronize()`). The vector timing does not, so the vector figure for MPS measures how fast work is queued, not how fast it finishes.
* Both backends build an `int64` result for the vector expression `2*1+!10000000`. The torch CPU row is faster than the numpy row; the cause was not investigated.
* The numpy backend row is the cost of going through the KlongPy interpreter and compiler instead of calling NumPy directly. It is close to NumPy for these two workloads.
* The measured MPS matrix multiply speedup is about 5x. The earlier table below showed about 3x on a different machine.

### CUDA GPU, measured February 2026 (not re-measured)

These figures come from the version of this page that was written in February 2026, before the changes in version 2. The machine and the library versions were not recorded, and no CUDA device was available to measure again. They are kept as an example of the shape of the result, not as a current number.

```bash
$ python3 tests/perf_vector.py
============================================================
VECTOR OPS (element-wise, memory-bound)
  Size: 10,000,000 elements, Iterations: 100
============================================================
NumPy (baseline)                    0.021854s
KlongPy (numpy)                     0.001413s  (15.46x vs NumPy)
KlongPy (torch, cpu)                0.000029s  (761.22x vs NumPy)
KlongPy (torch, cuda)               0.000028s  (784.04x vs NumPy)

============================================================
MATRIX MULTIPLY (compute-bound, GPU advantage)
  Size: 4000x4000, Iterations: 5
============================================================
NumPy (baseline)                    0.078615s
KlongPy (numpy)                     0.075400s  (1.04x vs NumPy)
KlongPy (torch, cpu)                0.077350s  (1.02x vs NumPy)
KlongPy (torch, cuda)               0.002339s  (33.62x vs NumPy)
```

The vector figures of 0.000029 s for CPU and 0.000028 s for CUDA are far below what it takes to write 10 million 8-byte integers (80 MB), and the 2026-11-16 CPU run took 0.0022 s. They probably do not measure the full work, and the cause was not investigated. Do not rely on the 761x and 784x ratios.

### Apple Silicon (MPS), measured February 2026 (superseded)

The table above replaces this one. It was measured on an unrecorded Apple Silicon machine.

```bash
$ python3 tests/perf_vector.py
============================================================
VECTOR OPS (element-wise, memory-bound)
  Size: 10,000,000 elements, Iterations: 100
============================================================
NumPy (baseline)                    0.007547s
KlongPy (numpy)                     0.007378s  (1.02x vs NumPy)
KlongPy (torch, cpu)                0.007360s  (1.03x vs NumPy)
KlongPy (torch, mps)                0.007320s  (1.03x vs NumPy)

============================================================
MATRIX MULTIPLY (compute-bound, GPU advantage)
  Size: 4000x4000, Iterations: 5
============================================================
NumPy (baseline)                    0.034870s
KlongPy (numpy)                     0.036095s  (0.97x vs NumPy)
KlongPy (torch, cpu)                0.035907s  (0.97x vs NumPy)
KlongPy (torch, mps)                0.011203s  (3.11x vs NumPy)
```

**Observations:**

* Matrix multiply is the workload where a GPU helps: about 5x on MPS in the 2026-11-16 run, and 34x on CUDA in the February run.
* Results vary by hardware and workload.

See [PyTorch Backend](torch_backend.md) for the backend itself, with a second table of expression timings (also measured 2026-11-16), and the [Roadmap](roadmap.md) for known limits.
