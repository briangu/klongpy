# Performance

The Klong language is simple, so the overhead is low.  The bulk of the compute time will likely be spent in NumPy doing actual work.

Here's a contrived rough benchmark to show the magnitude differences between Python, KlongPy (CPU + GPU) and Numpy (CPU).

**Spoiler**: GPU-backed KlongPy is about 790x faster than naive Python and 36x faster than NumPy-backed KlongPy.

### Python

```python
def python_vec(number=100):
    r = timeit.timeit(lambda: [2 * (1 + x) for x in range(10000000)], number=number)
    return r/number
```

### KlongPy

```python
# NumPy and CuPy (CuPy is enabled via USE_GPU=1 environment variable
def klong_vec(number=100):
    klong = KlongInterpreter()
    r = timeit.timeit(lambda: klong("2*1+!10000000"), number=number)
    return r/number
```

### NumPy (explicit usage)

```python
def NumPy_vec(number=100):
    r = timeit.timeit(lambda: np.multiply(np.add(np.arange(10000000), 1), 2), number=number)
    return r/number
```

## Results

### CPU (AMD Ryzen 9 7950x)

    $ python3 tests/perf_vector.py
    Python: 0.369111s
    KlongPy USE_GPU=None: 0.017946s
    Numpy: 0.017896s
    Python / KlongPy => 20.568334
    Numpy / KlongPy => 0.997245

### GPU (Same CPU with NVIDIA GeForce RTX 3090)

    $ USE_GPU=1 python3 tests/perf_vector.py
    Python: 0.364893s
    KlongPy USE_GPU=1: 0.000461s
    NumPy: 0.017053s
    Python / KlongPy => 790.678069
    Numpy / KlongPy => 36.951443

