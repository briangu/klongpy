"""
Performance benchmark comparing backends across different workloads.

- Vector ops: Simple element-wise operations (memory-bound)
- Matrix ops: Matrix multiplication (compute-bound, shows GPU advantage)

Usage:
    python tests/perf_vector.py
"""
import timeit

import importlib
import importlib.util
import numpy as np

from klongpy import KlongInterpreter

_TORCH_SPEC = importlib.util.find_spec("torch")
torch = importlib.import_module("torch") if _TORCH_SPEC else None
TORCH_AVAILABLE = torch is not None


def get_torch_devices():
    """Get list of available devices for torch backend."""
    if not TORCH_AVAILABLE:
        return []
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def has_torch():
    return TORCH_AVAILABLE


def vector_benchmark(backend=None, device=None, size=10_000_000, number=100):
    """
    Element-wise vector operations (memory-bound).
    GPU won't show much speedup here due to memory transfer overhead.
    """
    klong = KlongInterpreter(backend=backend, device=device)
    expr = f"2*1+!{size}"
    r = timeit.timeit(lambda: klong(expr), number=number)
    return r / number, klong._backend


def matrix_benchmark(backend=None, device=None, size=1000, number=10):
    """
    Matrix multiplication (compute-bound).
    GPU shines here: O(n³) compute vs O(n²) memory transfer.
    """
    klong = KlongInterpreter(backend=backend, device=device)
    # Create random matrices and convert to backend format
    a_np = np.random.rand(size, size).astype(np.float32)
    b_np = np.random.rand(size, size).astype(np.float32)
    klong['a'] = klong._backend.kg_asarray(a_np)
    klong['b'] = klong._backend.kg_asarray(b_np)
    # Import matmul from backend's underlying library
    if backend == 'torch':
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch backend requested but torch is not available")
        klong('.pyf("torch";"matmul")')
        # Warmup for GPU (compile kernels, etc.)
        for _ in range(5):
            klong("matmul(a;b)")
        sync = (torch.mps.synchronize if device == 'mps' else
                torch.cuda.synchronize if device == 'cuda' else lambda: None)
        sync()

        def timed_matmul():
            klong("matmul(a;b)")
            sync()
        r = timeit.timeit(timed_matmul, number=number)
    else:
        klong('.pyf("numpy";"matmul")')
        r = timeit.timeit(lambda: klong("matmul(a;b)"), number=number)
    return r / number, klong._backend


def numpy_vector(size=10_000_000, number=100):
    """Baseline NumPy vector ops."""
    r = timeit.timeit(
        lambda: np.multiply(np.add(np.arange(size), 1), 2),
        number=number
    )
    return r / number


def numpy_matrix(size=1000, number=10):
    """Baseline NumPy matrix multiply."""
    a = np.random.rand(size, size).astype(np.float32)
    b = np.random.rand(size, size).astype(np.float32)
    r = timeit.timeit(lambda: np.matmul(a, b), number=number)
    return r / number


def run_benchmarks():
    """Run all benchmarks and display results."""
    vector_size = 10_000_000
    vector_iters = 100
    matrix_size = 4000  # Larger matrices show GPU advantage
    matrix_iters = 5

    print("=" * 60)
    print("VECTOR OPS (element-wise, memory-bound)")
    print(f"  Size: {vector_size:,} elements, Iterations: {vector_iters}")
    print("=" * 60)

    # NumPy baseline
    np_vec = numpy_vector(size=vector_size, number=vector_iters)
    print(f"{'NumPy (baseline)':<35} {np_vec:.6f}s")

    # KlongPy with numpy backend
    klong_vec, backend = vector_benchmark(
        backend='numpy', size=vector_size, number=vector_iters
    )
    speedup = np_vec / klong_vec
    print(f"{'KlongPy (numpy)':<35} {klong_vec:.6f}s  ({speedup:.2f}x vs NumPy)")

    # Torch backends
    if has_torch():
        for device in get_torch_devices():
            klong_vec, backend = vector_benchmark(
                backend='torch', device=device,
                size=vector_size, number=vector_iters
            )
            speedup = np_vec / klong_vec
            print(f"{'KlongPy (torch, ' + device + ')':<35} {klong_vec:.6f}s  ({speedup:.2f}x vs NumPy)")

    print()
    print("=" * 60)
    print("MATRIX MULTIPLY (compute-bound, GPU advantage)")
    print(f"  Size: {matrix_size}x{matrix_size}, Iterations: {matrix_iters}")
    print("=" * 60)

    # NumPy baseline
    np_mat = numpy_matrix(size=matrix_size, number=matrix_iters)
    print(f"{'NumPy (baseline)':<35} {np_mat:.6f}s")

    # KlongPy with numpy backend
    klong_mat, backend = matrix_benchmark(
        backend='numpy', size=matrix_size, number=matrix_iters
    )
    speedup = np_mat / klong_mat
    print(f"{'KlongPy (numpy)':<35} {klong_mat:.6f}s  ({speedup:.2f}x vs NumPy)")

    # Torch backends
    if has_torch():
        for device in get_torch_devices():
            klong_mat, backend = matrix_benchmark(
                backend='torch', device=device,
                size=matrix_size, number=matrix_iters
            )
            speedup = np_mat / klong_mat
            print(f"{'KlongPy (torch, ' + device + ')':<35} {klong_mat:.6f}s  ({speedup:.2f}x vs NumPy)")

    print()


if __name__ == "__main__":
    run_benchmarks()
