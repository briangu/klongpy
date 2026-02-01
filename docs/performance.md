# Performance

The Klong language is simple, so the overhead is low. The bulk of the compute time will likely be spent in the array backend doing actual work.

Here's a rough benchmark showing magnitude differences between Python, KlongPy (CPU + GPU) and NumPy (CPU).

**Spoiler**: GPU-backed KlongPy with PyTorch is significantly faster than naive Python and CPU-bound KlongPy.

Results will vary by machine. Use `USE_TORCH=1` or `KLONG_BACKEND=torch` to enable the PyTorch backend.

## Python

```python
def python_vec(number=100):
    r = timeit.timeit(lambda: [2 * (1 + x) for x in range(10000000)], number=number)
    return r/number
```

## KlongPy

```python
# NumPy or PyTorch (PyTorch is enabled via USE_TORCH=1 environment variable)
def klong_vec(number=100):
    klong = KlongInterpreter()
    r = timeit.timeit(lambda: klong("2*1+!10000000"), number=number)
    return r/number
```

## NumPy (explicit usage)

```python
def numpy_vec(number=100):
    r = timeit.timeit(lambda: np.multiply(np.add(np.arange(10000000), 1), 2), number=number)
    return r/number
```

## Results

### CPU (NumPy backend)

```bash
$ python3 tests/perf_vector.py
Python: 0.369111s
KlongPy: 0.017946s
Numpy: 0.017896s
Python / KlongPy => 20.57x
```

### GPU (PyTorch backend with CUDA)

With PyTorch and CUDA available:

```bash
$ USE_TORCH=1 python3 tests/perf_vector.py
Backend: torch, Device: cuda:0
KlongPy: 0.000461s
Python / KlongPy => 790x
NumPy / KlongPy => 37x
```

### Apple Silicon (PyTorch backend with MPS)

With PyTorch on Apple Silicon:

```bash
$ USE_TORCH=1 python3 tests/perf_vector.py
Backend: torch, Device: mps:0
```

See [torch_backend.md](torch_backend.md) for more details on the PyTorch backend and performance characteristics.
