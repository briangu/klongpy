"""Compare resident native kg arrays with NumPy. See docs/performance.md."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
THREAD_VARS = ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")


def workloads(n):
    positions = np.arange(n, dtype=np.int64)
    order = 7919 * positions % n
    labels = ((104729 * positions % 1000003) % 257) - 128
    runs = positions // 64
    uniform = np.full(n, 7, dtype=np.int64)
    a = 0.25 * labels
    b = 0.5 * ((order % 113) - 56)
    matrix = labels.reshape(-1, 64)
    # Every callable allocates or returns its usual public API result.
    return {
        "float64 add": ("a+b", lambda: np.add(a, b)),
        "float64 multiply": ("a*b", lambda: np.multiply(a, b)),
        "float64 sum": ("+/a", lambda: np.sum(a)),
        "float64 min": ("&/a", lambda: np.min(a)),
        "float64 dot": ("+/a*b", lambda: np.dot(a, b)),
        "int64 cumulative sum": ("+\\labels", lambda: np.cumsum(labels)),
        "float64 first differences": ("(1_a)-(-1)_a", lambda: np.diff(a)),
        "float64 gather": ("a@order", lambda: a[order]),
        "int64 positive indices": ("&labels>0", lambda: np.flatnonzero(labels > 0)),
        "int64 stable argsort, 257 keys": ("<labels", lambda: np.argsort(labels, kind="stable")),
        "int64 stable argsort, wide keys": ("<order", lambda: np.argsort(order, kind="stable")),
        "int64 stable argsort, presorted": ("<positions", lambda: np.argsort(positions, kind="stable")),
        "int64 stable argsort, sorted runs": ("<runs", lambda: np.argsort(runs, kind="stable")),
        "int64 stable argsort, uniform": ("<uniform", lambda: np.argsort(uniform, kind="stable")),
        "int64 column sums": ("+/matrix", lambda: np.sum(matrix, axis=0)),
        "int64 row sums": ("+/'matrix", lambda: np.sum(matrix, axis=1)),
    }


def measure_numpy(fn, samples):
    def batch(count):
        start = time.perf_counter()
        for _ in range(count):
            fn()
        return time.perf_counter() - start

    reps = 1
    while batch(reps) < 0.005:
        reps *= 2
    return [reps, [batch(reps) / reps for _ in range(samples)]]


def run_kg(kg, n, directory, expressions):
    # Put large JSON validation records on disk, outside both timing loops.
    program = directory / "run.kg"
    setup = (HERE / "bench_numpy.kg").read_text()
    program.write_text(f"n::{n}\n" + setup + "\n".join(f".p(.json({expr}))" for expr in expressions) + "\n")
    output = directory / "results.jsonl"
    with output.open("wb") as dest:
        subprocess.run([kg, program], stdout=dest, check=True)
    return output


def run_size(kg, n, samples, directory, numpy_first):
    cases = workloads(n)
    expressions = [expr for expr, _ in cases.values()]
    output = run_kg(kg, n, directory, expressions)
    with output.open() as records:
        for (name, (_, reference)), line in zip(cases.items(), records, strict=True):
            # The float inputs are binary fractions, so sums and dot are exact at these sizes.
            np.testing.assert_array_equal(np.asarray(json.loads(line)), np.asarray(reference()),
                                          strict=True, err_msg=name)

    def time_kg():
        queries = [f"measure({{{expr};x}};{samples})" for expr in expressions]
        return [json.loads(line) for line in run_kg(kg, n, directory, queries).read_text().splitlines()]

    def time_numpy():
        return [measure_numpy(fn, samples) for _, fn in cases.values()]

    if numpy_first:
        py, native = time_numpy(), time_kg()
    else:
        native, py = time_kg(), time_numpy()
    results = []
    for name, (kr, kt), (pr, pt) in zip(cases, native, py, strict=True):
        if len(kt) != samples or len(pt) != samples or min(kt + pt) <= 0:
            raise ValueError(f"invalid timing for {name}: {kt}, {pt}")
        km, pm = statistics.median(kt), statistics.median(pt)
        print(f"{n:>9}  {name:34} {km*1e6:>10.2f} {pm*1e6:>10.2f} {pm/km:>8.2f}", flush=True)
        results.append(dict(n=n, workload=name, kg_repetitions=kr, numpy_repetitions=pr,
                            kg_seconds=kt, numpy_seconds=pt, kg_median=km, numpy_median=pm))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kg", type=Path, default=ROOT / "native/target/release/kg")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 1000000])
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.samples < 1 or any(n < 64 or n % 64 for n in args.sizes):
        parser.error("use positive samples and sizes that are positive multiples of 64")
    kg = args.kg.resolve(strict=True)
    print(" elements  workload                                kg µs   NumPy µs  NumPy/kg", flush=True)
    results = []
    with tempfile.TemporaryDirectory() as temp:
        for i, n in enumerate(args.sizes):
            results.extend(run_size(kg, n, args.samples, Path(temp), numpy_first=bool(i % 2)))
    if args.json:
        with kg.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        blas = np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {})
        args.json.write_text(json.dumps(dict(kg_sha256=digest, numpy_version=np.__version__, blas=blas.get("name"),
                                             platform=platform.platform(), python=platform.python_version(),
                                             threads={k: os.environ.get(k) for k in THREAD_VARS},
                                             results=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
