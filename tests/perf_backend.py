#!/usr/bin/env python
"""
Performance benchmarks comparing NumPy and Torch backends for KlongPy.

Usage:
    # Run with NumPy backend (default)
    python tests/perf_backend.py

    # Run with Torch backend
    USE_TORCH=1 python tests/perf_backend.py

    # Run comparison (both backends)
    python tests/perf_backend.py --compare
"""
import argparse
import os
import subprocess
import sys
import time


def get_backend_name():
    """Get the current backend name."""
    return "torch" if os.environ.get("USE_TORCH") else "numpy"


def benchmark(name, klong, expr, warmup=3, iterations=20):
    """Run a benchmark and return average time in ms."""
    # Warmup
    for _ in range(warmup):
        try:
            klong(expr)
        except Exception:
            return None, "warmup failed"

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        klong(expr)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    avg = sum(times) / len(times)
    return avg, None


def get_benchmarks():
    """Return list of (name, expression) tuples for benchmarking."""
    return [
        # Basic arithmetic on arrays - small
        ("vector_add_100K", "1+!100000"),
        ("vector_mult_100K", "2*!100000"),
        ("compound_expr_100K", "2*1+!100000"),

        # Basic arithmetic - large (torch should win here)
        ("vector_add_1M", "1+!1000000"),
        ("compound_expr_1M", "2*1+!1000000"),

        # Reductions
        ("sum_100K", "+/!100000"),
        ("sum_1M", "+/!1000000"),
        ("product_100", "*/1+!100"),

        # Mathematical functions
        ("power_10K", "(1+!10000)^2"),

        # Array operations
        ("transpose_100x100", "+100 100^!10000"),

        # Sorting/grading
        ("grade_up_10K", "<!10000"),
        ("grade_up_100K", "<!100000"),

        # Vector operations
        ("dot_product_10K", "+/(!10000)*(!10000)"),
        ("dot_product_100K", "+/(!100000)*(!100000)"),
        ("enumerate_100K", "!100000"),
        ("enumerate_1M", "!1000000"),
    ]


def run_benchmarks(iterations=20):
    """Run all benchmarks and return results dict."""
    from klongpy import KlongInterpreter

    klong = KlongInterpreter()
    backend = get_backend_name()

    results = {}
    for name, expr in get_benchmarks():
        avg_ms, err = benchmark(name, klong, expr, iterations=iterations)
        if avg_ms is not None:
            results[name] = avg_ms
        else:
            results[name] = None

    return backend, results


def print_results(backend, results):
    """Print benchmark results in a formatted table."""
    print(f"\n{'=' * 55}")
    print(f"Backend: {backend.upper()}")
    print(f"{'=' * 55}")
    print(f"{'Benchmark':<30} {'Time (ms)':>12} {'Ops/sec':>12}")
    print(f"{'-' * 55}")

    for name, avg_ms in results.items():
        if avg_ms is not None:
            ops_per_sec = 1000 / avg_ms if avg_ms > 0 else float('inf')
            print(f"{name:<30} {avg_ms:>12.3f} {ops_per_sec:>12.1f}")
        else:
            print(f"{name:<30} {'SKIP':>12} {'-':>12}")


def run_comparison():
    """Run benchmarks for both backends and compare."""
    print("\nRunning NumPy backend benchmarks...")
    result = subprocess.run(
        [sys.executable, __file__, "--json"],
        capture_output=True,
        text=True,
        env={**os.environ, "USE_TORCH": ""}
    )
    numpy_output = result.stdout.strip()

    print("Running Torch backend benchmarks...")
    result = subprocess.run(
        [sys.executable, __file__, "--json"],
        capture_output=True,
        text=True,
        env={**os.environ, "USE_TORCH": "1"}
    )
    torch_output = result.stdout.strip()

    # Parse JSON results
    import json
    try:
        numpy_results = json.loads(numpy_output)
        torch_results = json.loads(torch_output)
    except json.JSONDecodeError as e:
        print(f"Error parsing results: {e}")
        print(f"NumPy output: {numpy_output[:200]}")
        print(f"Torch output: {torch_output[:200]}")
        return

    # Print comparison table
    print(f"\n{'=' * 70}")
    print("BACKEND COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Benchmark':<25} {'NumPy (ms)':>12} {'Torch (ms)':>12} {'Speedup':>12}")
    print(f"{'-' * 70}")

    for name in numpy_results:
        np_time = numpy_results.get(name)
        torch_time = torch_results.get(name)

        np_str = f"{np_time:.3f}" if np_time else "SKIP"
        torch_str = f"{torch_time:.3f}" if torch_time else "SKIP"

        if np_time and torch_time:
            speedup = np_time / torch_time
            speedup_str = f"{speedup:.2f}x"
            if speedup > 1:
                speedup_str += " (torch)"
            elif speedup < 1:
                speedup_str = f"{1/speedup:.2f}x (numpy)"
            else:
                speedup_str = "1.00x (equal)"
        else:
            speedup_str = "-"

        print(f"{name:<25} {np_str:>12} {torch_str:>12} {speedup_str:>12}")


def main():
    parser = argparse.ArgumentParser(description="KlongPy backend performance benchmarks")
    parser.add_argument("--compare", action="store_true", help="Compare both backends")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--iterations", type=int, default=20, help="Number of iterations per benchmark")
    args = parser.parse_args()

    if args.compare:
        run_comparison()
        return

    backend, results = run_benchmarks(iterations=args.iterations)

    if args.json:
        import json
        print(json.dumps(results))
    else:
        print_results(backend, results)


if __name__ == "__main__":
    main()
