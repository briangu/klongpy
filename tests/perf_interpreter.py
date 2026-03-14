#!/usr/bin/env python
"""
KlongPy interpreter performance benchmarks.

Tests real workloads that stress the interpreter, not just cached results.
Modeled after kdb+ benchmark patterns: VWAP, running sums, sort/rank,
boolean filtering, group-by, each/over/scan adverbs, and recursive functions.

Usage:
    python tests/perf_interpreter.py
    python tests/perf_interpreter.py --iterations 50
    python tests/perf_interpreter.py --category vector
"""
import argparse
import time
import numpy as np
from klongpy import KlongInterpreter


def benchmark(name, klong, expr, warmup=3, iterations=20, setup=None):
    """Run a benchmark, clearing result cache each iteration to force recomputation."""
    # Warmup
    for _ in range(warmup):
        if setup:
            setup(klong)
        klong._result_cache.clear()
        try:
            klong(expr)
        except Exception as e:
            return None, f"warmup failed: {e}"

    # Timed runs
    times = []
    for _ in range(iterations):
        if setup:
            setup(klong)
        klong._result_cache.clear()
        start = time.perf_counter()
        klong(expr)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    times.sort()
    # Use median of middle 60% to reduce variance
    n = len(times)
    lo, hi = n // 5, n - n // 5
    trimmed = times[lo:hi]
    avg = sum(trimmed) / len(trimmed)
    return avg, None


def setup_random_data(klong, n=1_000_000, seed=42):
    """Set up random float arrays for benchmarks."""
    rng = np.random.default_rng(seed)
    klong['a'] = rng.random(n)
    klong['b'] = rng.random(n)


def setup_trade_data(klong, n=100_000, n_syms=100, seed=42):
    """Set up simulated trade data for VWAP/analytics benchmarks."""
    rng = np.random.default_rng(seed)
    klong['price'] = 50.0 + rng.standard_normal(n) * 10.0
    klong['size'] = rng.integers(100, 10000, size=n).astype(float)
    klong['sym'] = rng.integers(0, n_syms, size=n)


def setup_int_data(klong, n=100_000, seed=42):
    """Set up integer arrays for sort/rank/group benchmarks."""
    rng = np.random.default_rng(seed)
    klong['iv'] = rng.integers(0, n, size=n)
    klong['ig'] = rng.integers(0, 1000, size=n)  # 1000 groups


def setup_ewma_data(klong, n=10_000, seed=42):
    """Set up data for EWMA scan benchmark."""
    rng = np.random.default_rng(seed)
    klong['ret'] = rng.standard_normal(n) * 0.01  # daily returns
    klong['alpha'] = 0.06  # ~30-day half-life


def get_benchmarks():
    """Return categorized benchmarks: (category, name, expr, setup_fn)."""
    return [
        # ── Vector analytics (NumPy-backed, tests dispatch + computation) ──
        ("vector", "running_sum_1M",    "+\\a",             lambda k: setup_random_data(k, 1_000_000)),
        ("vector", "cumul_product_100K", "*\\100000#a",     lambda k: setup_random_data(k, 1_000_000)),
        ("vector", "deltas_1M",         "-:'a",             lambda k: setup_random_data(k, 1_000_000)),
        ("vector", "vec_arith_chain_1M","(a*2+b)%a+1",     lambda k: setup_random_data(k, 1_000_000)),
        ("vector", "dot_product_1M",    "+/a*b",            lambda k: setup_random_data(k, 1_000_000)),

        # ── Reduction (fold) patterns ──
        ("reduce", "sum_1M",    "+/a",  lambda k: setup_random_data(k, 1_000_000)),
        ("reduce", "max_1M",    "|/a",  lambda k: setup_random_data(k, 1_000_000)),
        ("reduce", "min_1M",    "&/a",  lambda k: setup_random_data(k, 1_000_000)),

        # ── Sort / rank / group ──
        ("sort", "grade_up_100K",   "<iv",      lambda k: setup_int_data(k, 100_000)),
        ("sort", "sort_100K",       "iv@<iv",   lambda k: setup_int_data(k, 100_000)),
        ("sort", "rank_100K",       "<<iv",     lambda k: setup_int_data(k, 100_000)),
        ("sort", "grade_up_1M",     "<iv",      lambda k: setup_int_data(k, 1_000_000)),
        ("sort", "unique_100K",     "?ig",      lambda k: setup_int_data(k, 100_000)),

        # ── Boolean indexing / where ──
        ("filter", "where_50pct_1M",  "&(a>0.5)",     lambda k: setup_random_data(k, 1_000_000)),
        ("filter", "filter_index_1M", "a@&(a>0.5)",   lambda k: setup_random_data(k, 1_000_000)),
        ("filter", "count_filter_1M", "#&(a>0.5)",     lambda k: setup_random_data(k, 1_000_000)),

        # ── Financial analytics ──
        ("finance", "vwap_100K",         "(+/price*size)%+/size",   lambda k: setup_trade_data(k, 100_000)),
        ("finance", "trade_value_100K",  "+/price*size",            lambda k: setup_trade_data(k, 100_000)),
        ("finance", "price_range_100K",  "(|/price)-&/price",       lambda k: setup_trade_data(k, 100_000)),
        ("finance", "returns_100K",      "-:'price",                lambda k: setup_trade_data(k, 100_000)),

        # ── Each / adverb patterns ──
        ("adverb", "each_square_10K",   "{x*x}'!10000",            None),
        ("adverb", "each_fn_10K",       "{x*x+1}'!10000",          None),
        ("adverb", "each2_add_10K",     "(!10000){x+y}'(!10000)",  None),
        ("adverb", "over_custom_1K",    "{x+y*y}/!1000",           None),

        # ── Scan / fold with custom functions ──
        ("scan", "ewma_10K",            "{(x*alpha)+(y*(1-alpha))}\\ret",  lambda k: setup_ewma_data(k, 10_000)),
        ("scan", "scan_add_10K",        "{x+y}\\!10000",                   None),
        ("scan", "scan_compound_10K",   "{x+y*2}\\!10000",                 None),
        ("scan", "over_sum_10K",        "{x+y}/!10000",                    None),

        # ── Recursive / interpreter-heavy ──
        ("interp", "fib_20",    "fib(20)",  None),
        ("interp", "fib_25",    "fib(25)",  None),
        ("interp", "nested_fn", "h(1000)",  None),
        ("interp", "loop_1K",   "1000{x+1}:*0", None),
    ]


def create_interpreter():
    """Create interpreter with benchmark helper functions defined."""
    klong = KlongInterpreter()
    # Recursive fibonacci
    klong('fib::{:[x<2;x;fib(x-1)+fib(x-2)]}')
    # Nested function calls
    klong('f1::{x+1}')
    klong('f2::{x*2}')
    klong('f3::{x-3}')
    klong('h::{f3(f2(f1(x)))}')
    return klong


def print_results(results, category_filter=None):
    """Print benchmark results."""
    print(f"\n{'=' * 65}")
    print(f"KlongPy Interpreter Benchmarks (cold eval, warm parse)")
    print(f"{'=' * 65}")

    current_cat = None
    for cat, name, avg_ms, err in results:
        if category_filter and cat != category_filter:
            continue
        if cat != current_cat:
            current_cat = cat
            print(f"\n  [{cat}]")
            print(f"  {'Benchmark':<25} {'Time (ms)':>12} {'Ops/sec':>12}")
            print(f"  {'-' * 50}")

        if avg_ms is not None:
            ops = 1000 / avg_ms if avg_ms > 0 else float('inf')
            print(f"  {name:<25} {avg_ms:>12.3f} {ops:>12.1f}")
        else:
            print(f"  {name:<25} {'FAIL':>12} {err}")

    # Print aggregate
    valid = [(cat, name, ms, err) for cat, name, ms, err in results
             if ms is not None and (not category_filter or cat == category_filter)]
    if valid:
        total = sum(ms for _, _, ms, _ in valid)
        print(f"\n  {'AGGREGATE':25} {total:>12.3f} ms ({len(valid)} benchmarks)")


def main():
    parser = argparse.ArgumentParser(description="KlongPy interpreter benchmarks")
    parser.add_argument("--iterations", type=int, default=30,
                        help="Number of timed iterations per benchmark")
    parser.add_argument("--category", type=str, default=None,
                        help="Run only benchmarks in this category")
    args = parser.parse_args()

    klong = create_interpreter()

    results = []
    for cat, name, expr, setup_fn in get_benchmarks():
        if args.category and cat != args.category:
            continue
        if setup_fn:
            setup_fn(klong)
        avg_ms, err = benchmark(name, klong, expr,
                                iterations=args.iterations, setup=setup_fn)
        results.append((cat, name, avg_ms, err))

    print_results(results, args.category)


if __name__ == "__main__":
    main()
