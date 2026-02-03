#!/usr/bin/env python
"""
Performance benchmarks comparing NumPy, Torch, and Torch+JIT backends for KlongPy.

Usage:
    # Run with NumPy backend (default)
    python tests/perf_backend.py

    # Run with Torch backend
    python tests/perf_backend.py --backend torch

    # Run with Torch + JIT compilation (requires C++ compiler)
    python tests/perf_backend.py --backend torch --jit

    # Run with Torch + JIT using eager backend (no C++ compiler needed)
    python tests/perf_backend.py --backend torch --jit --eager

    # Run comparison (all backends)
    python tests/perf_backend.py --compare

    # Run JIT-specific benchmarks only
    python tests/perf_backend.py --backend torch --jit-only

    # Run JIT benchmarks with eager backend (no C++ compiler)
    python tests/perf_backend.py --backend torch --jit-only --eager

    # Run JIT benchmarks with different modes
    python tests/perf_backend.py --backend torch --jit-only --jit-mode max-autotune
"""
import argparse
import subprocess
import sys
import time


# Global to store the backend name set via command line
_current_backend = "numpy"


def set_backend(name):
    """Set the current backend name."""
    global _current_backend
    _current_backend = name


def get_backend_name():
    """Get the current backend name."""
    return _current_backend


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


def benchmark_fn(name, fn, arg, warmup=3, iterations=20):
    """Run a benchmark on a compiled function."""
    import torch

    # Ensure arg is a tensor
    if not isinstance(arg, torch.Tensor):
        arg = torch.tensor(arg, dtype=torch.float32)

    # Warmup
    for _ in range(warmup):
        try:
            fn(arg)
        except Exception as e:
            return None, f"warmup failed: {e}"

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn(arg)
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


def get_jit_benchmarks():
    """Return list of (name, fn_def, example_input, test_input) for JIT benchmarks.

    These are function-based benchmarks suitable for torch.compile.
    Note: Functions using reduction operators (+/) don't compile well due to
    Klong interpreter callbacks during tracing. Focus on element-wise operations.
    """
    return [
        # Simple element-wise functions (work well with compilation)
        ("jit_square", "{x^2}", 3.0, 1000000),
        ("jit_cube", "{x^3}", 2.0, 1000000),
        ("jit_quadratic", "{(x^2)+(2*x)+1}", 3.0, 1000000),
        ("jit_poly4", "{(x^4)-(3*x^3)+(2*x^2)-x+1}", 2.0, 100000),

        # More complex element-wise
        ("jit_sincos", "{(.sin(x))^2+(.cos(x))^2}", 1.0, 100000),
        ("jit_exp_decay", "{2.718^(-x*x)}", 1.0, 100000),
        ("jit_tanh_approx", "{(2.718^x-2.718^(-x))%(2.718^x+2.718^(-x))}", 0.5, 100000),

        # Arithmetic chains
        ("jit_arithmetic", "{((x+1)*2-3)%4}", 5.0, 1000000),
        ("jit_power_chain", "{(x^2)^2}", 2.0, 1000000),
    ]


def run_benchmarks(iterations=20):
    """Run all benchmarks and return results dict."""
    from klongpy import KlongInterpreter

    backend = get_backend_name()
    klong = KlongInterpreter(backend=backend)

    results = {}
    for name, expr in get_benchmarks():
        avg_ms, err = benchmark(name, klong, expr, iterations=iterations)
        if avg_ms is not None:
            results[name] = avg_ms
        else:
            results[name] = None

    return backend, results


def run_jit_benchmarks(iterations=20, mode="default", use_eager=False):
    """Run JIT-compiled function benchmarks."""
    import torch
    from klongpy import KlongInterpreter

    backend = get_backend_name()
    if backend != "torch":
        print("JIT benchmarks require torch backend (--backend torch)")
        return backend, {}, {}
    klong = KlongInterpreter(backend=backend)

    results = {}
    results_nojit = {}

    # Use eager backend if requested (no C++ compiler needed)
    jit_backend = "eager" if use_eager else "inductor"

    for name, fn_def, example_input, test_input in get_jit_benchmarks():
        # Define the function
        klong(f"f::{fn_def}")

        # Convert test input to tensor
        if isinstance(test_input, list):
            tensor_input = torch.tensor(test_input, dtype=torch.float32)
        else:
            # For scalar functions with large test_input, create array of that size
            if isinstance(test_input, (int, float)) and test_input > 100:
                tensor_input = torch.randn(int(test_input), dtype=torch.float32)
            else:
                tensor_input = torch.tensor([test_input], dtype=torch.float32)

        # Benchmark without JIT first
        try:
            # Initialize variable in Klong, then set from Python
            klong('inp::0')
            klong['inp'] = tensor_input

            def nojit_fn(x):
                klong['inp'] = x
                return klong('f(inp)')

            avg_ms, err = benchmark_fn(f"{name}_nojit", nojit_fn, tensor_input, iterations=iterations)
            results_nojit[name] = avg_ms
        except Exception as e:
            results_nojit[name] = None

        # Try to compile with JIT
        try:
            # Use compilex with specified backend (mode only matters for inductor)
            if use_eager:
                compiled_fn = klong(f'.compilex(f;{example_input};:{{["backend" "eager"]}})')
            else:
                compiled_fn = klong(f'.compilex(f;{example_input};:{{["mode" "{mode}"]}})')

            avg_ms, err = benchmark_fn(name, compiled_fn, tensor_input, iterations=iterations)
            if avg_ms is not None:
                results[name] = avg_ms
            else:
                results[name] = None
        except Exception as e:
            # JIT compilation failed (likely no C++ compiler)
            results[name] = None

    return backend, results, results_nojit


def run_jit_only_benchmarks(iterations=20, mode="default", use_eager=False):
    """Run and display JIT-specific benchmarks."""
    backend, jit_results, nojit_results = run_jit_benchmarks(iterations=iterations, mode=mode, use_eager=use_eager)

    if not jit_results and not nojit_results:
        return

    print(f"\n{'=' * 75}")
    print(f"JIT COMPILATION BENCHMARKS (mode: {mode})")
    print(f"{'=' * 75}")
    print(f"{'Benchmark':<25} {'No JIT (ms)':>12} {'JIT (ms)':>12} {'Speedup':>15}")
    print(f"{'-' * 75}")

    for name in nojit_results:
        nojit_time = nojit_results.get(name)
        jit_time = jit_results.get(name)

        nojit_str = f"{nojit_time:.3f}" if nojit_time else "SKIP"
        jit_str = f"{jit_time:.3f}" if jit_time else "SKIP"

        if nojit_time and jit_time:
            speedup = nojit_time / jit_time
            if speedup >= 1:
                speedup_str = f"{speedup:.2f}x (jit)"
            else:
                speedup_str = f"{1/speedup:.2f}x (no-jit)"
        else:
            speedup_str = "-"

        print(f"{name:<25} {nojit_str:>12} {jit_str:>12} {speedup_str:>15}")

    print(f"\nNote: JIT compilation has startup overhead but faster subsequent calls.")
    print(f"      Best for functions called many times in loops.")


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
    """Run benchmarks for all backends and compare."""
    print("\nRunning NumPy backend benchmarks...")
    result = subprocess.run(
        [sys.executable, __file__, "--json", "--backend", "numpy"],
        capture_output=True,
        text=True,
    )
    numpy_output = result.stdout.strip()

    print("Running Torch backend benchmarks...")
    result = subprocess.run(
        [sys.executable, __file__, "--json", "--backend", "torch"],
        capture_output=True,
        text=True,
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


def run_full_comparison():
    """Run comprehensive comparison including JIT."""
    import json

    print("\n" + "=" * 80)
    print("COMPREHENSIVE BACKEND COMPARISON")
    print("=" * 80)

    # Run standard benchmarks
    run_comparison()

    # Run JIT benchmarks if torch available
    print("\n\nRunning JIT compilation benchmarks...")
    print("(Using eager backend - add --no-eager for inductor which requires C++ compiler)")

    try:
        result = subprocess.run(
            [sys.executable, __file__, "--jit-only", "--json", "--eager", "--backend", "torch"],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0 and result.stdout.strip():
            try:
                jit_data = json.loads(result.stdout.strip())
                if jit_data.get("jit") or jit_data.get("nojit"):
                    print(f"\n{'=' * 75}")
                    print("JIT COMPILATION RESULTS (eager backend)")
                    print(f"{'=' * 75}")
                    print(f"{'Benchmark':<25} {'Torch (ms)':>12} {'Torch+JIT (ms)':>14} {'Speedup':>15}")
                    print(f"{'-' * 75}")

                    nojit = jit_data.get("nojit", {})
                    jit = jit_data.get("jit", {})

                    for name in nojit:
                        nojit_time = nojit.get(name)
                        jit_time = jit.get(name)

                        nojit_str = f"{nojit_time:.3f}" if nojit_time else "SKIP"
                        jit_str = f"{jit_time:.3f}" if jit_time else "N/A"

                        if nojit_time and jit_time:
                            speedup = nojit_time / jit_time
                            if speedup >= 1:
                                speedup_str = f"{speedup:.2f}x (jit)"
                            else:
                                speedup_str = f"{1/speedup:.2f}x (no-jit)"
                        else:
                            speedup_str = "-"

                        print(f"{name:<25} {nojit_str:>12} {jit_str:>14} {speedup_str:>15}")

                    print(f"\nNote: Eager backend shows baseline - use inductor for real optimization.")
                    print(f"      Run with: python tests/perf_backend.py --jit-only (requires C++ compiler)")
            except json.JSONDecodeError:
                print("\nJIT benchmarks output (non-JSON):")
                print(result.stdout[:500] if result.stdout else "(no output)")
        else:
            print("\nJIT benchmarks skipped or failed.")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print("\nJIT benchmarks timed out.")
    except Exception as e:
        print(f"\nJIT benchmarks failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="KlongPy backend performance benchmarks")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy",
                        help="Backend to use: numpy (default) or torch")
    parser.add_argument("--compare", action="store_true", help="Compare all backends")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--jit", action="store_true", help="Include JIT benchmarks (torch only)")
    parser.add_argument("--jit-only", action="store_true", help="Run only JIT benchmarks")
    parser.add_argument("--jit-mode", choices=["default", "reduce-overhead", "max-autotune"],
                        default="default", help="JIT compilation mode")
    parser.add_argument("--eager", action="store_true",
                        help="Use eager backend for JIT (no C++ compiler needed)")
    parser.add_argument("--iterations", type=int, default=20, help="Number of iterations per benchmark")
    args = parser.parse_args()

    # Set the backend from command line
    set_backend(args.backend)

    if args.compare:
        run_full_comparison()
        return

    if args.jit_only:
        if args.json:
            import json
            backend, jit_results, nojit_results = run_jit_benchmarks(
                iterations=args.iterations, mode=args.jit_mode, use_eager=args.eager
            )
            print(json.dumps({"jit": jit_results, "nojit": nojit_results}))
        else:
            run_jit_only_benchmarks(iterations=args.iterations, mode=args.jit_mode, use_eager=args.eager)
        return

    backend, results = run_benchmarks(iterations=args.iterations)

    if args.jit and backend == "torch":
        # Also run JIT benchmarks
        _, jit_results, nojit_results = run_jit_benchmarks(
            iterations=args.iterations, mode=args.jit_mode, use_eager=args.eager
        )
        results["_jit"] = jit_results
        results["_nojit"] = nojit_results

    if args.json:
        import json
        print(json.dumps(results))
    else:
        print_results(backend, results)

        if args.jit and backend == "torch" and "_jit" in results:
            print(f"\n{'=' * 75}")
            print(f"JIT COMPILATION BENCHMARKS (mode: {args.jit_mode})")
            print(f"{'=' * 75}")
            print(f"{'Benchmark':<25} {'No JIT (ms)':>12} {'JIT (ms)':>12} {'Speedup':>15}")
            print(f"{'-' * 75}")

            nojit_results = results.get("_nojit", {})
            jit_results = results.get("_jit", {})

            for name in nojit_results:
                nojit_time = nojit_results.get(name)
                jit_time = jit_results.get(name)

                nojit_str = f"{nojit_time:.3f}" if nojit_time else "SKIP"
                jit_str = f"{jit_time:.3f}" if jit_time else "N/A"

                if nojit_time and jit_time:
                    speedup = nojit_time / jit_time
                    if speedup >= 1:
                        speedup_str = f"{speedup:.2f}x (jit)"
                    else:
                        speedup_str = f"{1/speedup:.2f}x (no-jit)"
                else:
                    speedup_str = "-"

                print(f"{name:<25} {nojit_str:>12} {jit_str:>12} {speedup_str:>15}")


if __name__ == "__main__":
    main()
