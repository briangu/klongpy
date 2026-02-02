# Performance

The Klong language is simple, so the overhead is low. The bulk of the compute time will likely be spent in the array backend doing actual work.

**Key insight**: GPU acceleration shines for compute-bound operations (matrix multiply), not memory-bound operations (element-wise ops).

Use `USE_TORCH=1` or `KLONGPY_BACKEND=torch` to enable the PyTorch backend.

## Benchmark

The benchmark (`tests/perf_vector.py`) tests two workload types:

1. **Vector ops** (element-wise, memory-bound): `2*1+!10000000`
2. **Matrix multiply** (compute-bound): 4000×4000 matmul

## Results

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
- **Vector ops**: All backends perform similarly (memory bandwidth limited)
- **Matrix multiply**: GPU (MPS) is ~3x faster than CPU (compute bound, parallelizable)
- CUDA GPUs typically show even larger speedups for matrix operations

See [torch_backend.md](torch_backend.md) for more details on the PyTorch backend and performance characteristics.
