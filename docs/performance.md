# Performance

The Klong language is simple, so the overhead is low. The bulk of the compute time will likely be spent in the array backend doing actual work.

**Key insight**: GPU acceleration shines for compute-bound operations (matrix multiply), not memory-bound operations (element-wise ops).

Use `USE_TORCH=1` or `KLONGPY_BACKEND=torch` to enable the PyTorch backend.

## Benchmark

The benchmark (`tests/perf_vector.py`) tests two workload types:

1. **Vector ops** (element-wise, memory-bound): `2*1+!10000000`
2. **Matrix multiply** (compute-bound): 4000×4000 matmul

## Results

### CUDA GPU

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

### Apple Silicon (MPS)

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
- **CUDA**: Massive speedups for both vector ops (784x) and matrix multiply (34x)
- **MPS**: ~3x speedup for matrix multiply; vector ops similar across backends
- Results vary by hardware and workload characteristics

See [torch_backend.md](torch_backend.md) for more details on the PyTorch backend and performance characteristics.
