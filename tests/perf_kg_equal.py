#!/usr/bin/env python3
"""Benchmark kg_equal performance across backends and input shapes.

Usage:
    python tests/perf_kg_equal.py                 # numpy backend
    python tests/perf_kg_equal.py --backend torch # torch backend
    python tests/perf_kg_equal.py --backend torch --device mps

Notes:
    - This benchmark focuses on BackendProvider.kg_equal.
    - It keeps data creation outside timed sections.
"""
import argparse
import csv
import time
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

from klongpy.backends import get_backend

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None


def _sync_if_needed(backend) -> None:
    if backend.name != "torch" or torch is None:
        return
    dev = str(backend.device)
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    elif dev.startswith("mps") and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _bench_case(label: str, fn: Callable[[], None], backend, warmup: int, iters: int) -> Tuple[str, float]:
    for _ in range(warmup):
        fn()
    _sync_if_needed(backend)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync_if_needed(backend)
    elapsed = time.perf_counter() - start
    return label, (elapsed / iters) * 1e6  # us/op


def _baseline_kg_equal(a, b, backend) -> bool:
    if a is b:
        return True

    is_numpy_a = isinstance(a, np.ndarray)
    is_numpy_b = isinstance(b, np.ndarray)
    is_backend_a = backend.is_backend_array(a)
    is_backend_b = backend.is_backend_array(b)

    if is_backend_a:
        a = backend.to_numpy(a)
        is_backend_a = False
        is_numpy_a = isinstance(a, np.ndarray)
    if is_backend_b:
        b = backend.to_numpy(b)
        is_backend_b = False
        is_numpy_b = isinstance(b, np.ndarray)

    na, nb = is_numpy_a or is_backend_a, is_numpy_b or is_backend_b

    if na and nb:
        a_dtype = backend.get_dtype_kind(a)
        b_dtype = backend.get_dtype_kind(b)
        if a_dtype == b_dtype and a_dtype != 'O':
            return bool(np.array_equal(a, b))

    na, nb = na or isinstance(a, list), nb or isinstance(b, list)

    if na != nb:
        if is_numpy_a and a.ndim == 0 and not nb:
            return _baseline_kg_equal(a.item(), b, backend)
        if is_numpy_b and b.ndim == 0 and not na:
            return _baseline_kg_equal(a, b.item(), backend)
        if is_backend_a and hasattr(a, 'ndim') and a.ndim == 0 and not nb:
            return _baseline_kg_equal(backend.scalar_to_python(a), b, backend)
        if is_backend_b and hasattr(b, 'ndim') and b.ndim == 0 and not na:
            return _baseline_kg_equal(a, backend.scalar_to_python(b), backend)
        return False

    if na:
        a_is_0d = hasattr(a, 'ndim') and a.ndim == 0
        b_is_0d = hasattr(b, 'ndim') and b.ndim == 0
        if a_is_0d or b_is_0d:
            a_val = backend.scalar_to_python(a) if a_is_0d else a
            b_val = backend.scalar_to_python(b) if b_is_0d else b
            return _baseline_kg_equal(a_val, b_val, backend)
        return len(a) == len(b) and all(_baseline_kg_equal(x, y, backend) for x, y in zip(a, b))

    if backend.is_number(a) and backend.is_number(b):
        if backend.is_backend_array(a):
            a = backend.scalar_to_python(a)
        if backend.is_backend_array(b):
            b = backend.scalar_to_python(b)
        result = np.isclose(a, b)
        if hasattr(result, 'item'):
            return bool(result.item())
        return bool(result)

    result = a == b
    if hasattr(result, 'all'):
        return bool(result.all())
    if hasattr(result, 'item'):
        return bool(result.item())
    return bool(result)


def _build_cases(backend, n: int, n2: int) -> List[Tuple[str, Callable[[], None], Callable[[], None]]]:
    cases: List[Tuple[str, Callable[[], None], Callable[[], None]]] = []

    def _add_case(label: str, a, b) -> None:
        cases.append((
            label,
            lambda a=a, b=b: backend.kg_equal(a, b),
            lambda a=a, b=b: _baseline_kg_equal(a, b, backend),
        ))

    # Scalars
    _add_case("scalar_int_eq", 1, 1)
    _add_case("scalar_int_ne", 1, 2)
    _add_case("scalar_float_eq", 1.5, 1.5)
    _add_case("scalar_float_ne", 1.5, 1.6)

    # 0-d arrays
    np_0d_a = np.array(1)
    np_0d_b = np.array(1)
    np_0d_c = np.array(2)
    _add_case("np_0d_eq", np_0d_a, np_0d_b)
    _add_case("np_0d_ne", np_0d_a, np_0d_c)

    # Numpy arrays
    np_a = np.arange(n, dtype=np.float32)
    np_b = np_a.copy()
    np_c = np_a.copy()
    np_c[-1] += 1
    _add_case(f"np_1d_eq_{n}", np_a, np_b)
    _add_case(f"np_1d_ne_{n}", np_a, np_c)

    np_2d_a = np.arange(n2 * 100, dtype=np.float32).reshape(n2, 100)
    np_2d_b = np_2d_a.copy()
    np_2d_c = np_2d_a.copy()
    np_2d_c[-1, -1] += 1
    _add_case(f"np_2d_eq_{n2}x100", np_2d_a, np_2d_b)
    _add_case(f"np_2d_ne_{n2}x100", np_2d_a, np_2d_c)

    # Lists
    list_a = list(range(10))
    list_b = list(range(10))
    list_c = list(range(9)) + [99]
    _add_case("list_10_eq", list_a, list_b)
    _add_case("list_10_ne", list_a, list_c)

    list_big_a = list(range(n2))
    list_big_b = list(range(n2))
    list_big_c = list(range(n2))
    list_big_c[-1] = -1
    _add_case(f"list_{n2}_eq", list_big_a, list_big_b)
    _add_case(f"list_{n2}_ne", list_big_a, list_big_c)

    # Backend arrays (torch) and mixed comparisons
    if backend.name == "torch":
        t_a = backend.kg_asarray(np_a)
        t_b = backend.kg_asarray(np_b)
        t_c = backend.kg_asarray(np_c)
        _add_case(f"torch_1d_eq_{n}", t_a, t_b)
        _add_case(f"torch_1d_ne_{n}", t_a, t_c)
        # Mixed (torch vs numpy)
        _add_case(f"torch_vs_numpy_eq_{n}", t_a, np_b)
        _add_case(f"numpy_vs_torch_eq_{n}", np_b, t_a)

    # Object arrays (numpy-only)
    if backend.name == "numpy":
        obj_a = np.asarray([1, "a", 3], dtype=object)
        obj_b = np.asarray([1, "a", 3], dtype=object)
        obj_c = np.asarray([1, "b", 3], dtype=object)
        _add_case("obj_array_eq", obj_a, obj_b)
        _add_case("obj_array_ne", obj_a, obj_c)

    return cases


def _write_csv(rows: List[Tuple[str, str, str, str, float]], csv_path: Path, append: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or not append
    mode = "a" if append else "w"
    with csv_path.open(mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["label", "backend", "device", "case", "us_per_op"])
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark kg_equal performance")
    parser.add_argument("--backend", default="numpy", choices=["numpy", "torch"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--n", type=int, default=100000, help="1d array size")
    parser.add_argument("--n2", type=int, default=1000, help="list size / 2d rows")
    parser.add_argument("--csv", default=None, help="Write results to CSV file")
    parser.add_argument("--label", default="run", help="Label for CSV rows")
    parser.add_argument("--append", action="store_true", help="Append to CSV instead of overwrite")
    parser.add_argument("--compare-baseline", action="store_true", help="Benchmark baseline algorithm too")
    args = parser.parse_args()

    backend = get_backend(args.backend, device=args.device)
    cases = _build_cases(backend, args.n, args.n2)

    device = str(getattr(backend, "device", "cpu"))
    print(f"backend={backend.name} device={device} iters={args.iters} warmup={args.warmup}")
    if args.compare_baseline:
        print("case,baseline_us,current_us,ratio")
    else:
        print("label,us_per_op")
    csv_rows: List[Tuple[str, str, str, str, float]] = []
    for label, fn, baseline_fn in cases:
        if args.compare_baseline:
            _, us_base = _bench_case(label, baseline_fn, backend, args.warmup, args.iters)
            _, us_curr = _bench_case(label, fn, backend, args.warmup, args.iters)
            ratio = (us_curr / us_base) if us_base else float("inf")
            print(f"{label},{us_base:.3f},{us_curr:.3f},{ratio:.3f}")
            csv_rows.append((f"{args.label}_baseline", backend.name, device, label, round(us_base, 6)))
            csv_rows.append((f"{args.label}_current", backend.name, device, label, round(us_curr, 6)))
        else:
            _, us = _bench_case(label, fn, backend, args.warmup, args.iters)
            print(f"{label},{us:.3f}")
            csv_rows.append((args.label, backend.name, device, label, round(us, 6)))

    if args.csv:
        _write_csv(csv_rows, Path(args.csv), args.append)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
