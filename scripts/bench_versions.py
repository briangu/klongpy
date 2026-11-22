"""Compare two KlongPy source trees through their public NumPy-hosted interpreters."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tomllib

import numpy as np
import klongpy
from klongpy import KlongInterpreter


CASES = (
    ("fib(20)", "fib::{:[x<2;x;fib(x-1)+fib(x-2)]}", "fib(20)", lambda: 6765),
    ("while to 100000", "", "{x<100000}{x+1}:~0", lambda: 100000),
    ("lambda reduce, 100000", "", "{x+y}/!100000", lambda: sum(range(100000))),
    ("element-wise arithmetic, 1000000", "", "2*1+!1000000", lambda: 2 * (1 + np.arange(1000000))),
    ("sum, 1000000", "", "+/!1000000", lambda: sum(range(1000000))),
    ("scan, 1000000", "", "+\\!1000000", lambda: np.cumsum(np.arange(1000000))),
    ("sort seeded floats, 100000", "r", "r@<r", lambda: np.sort(np.random.RandomState(1).rand(100000))),
    ("group 64 IDs, 1000000", "g", "#' =g", lambda: np.full(64, 15625)),
)


def worker():
    rows = {}
    for name, setup, program, expectation in CASES:
        expected = expectation()
        klong = KlongInterpreter(backend="numpy")
        if setup == "r":
            klong["r"] = np.random.RandomState(1).rand(100000)
        elif setup == "g":
            klong["g"] = np.arange(1000000) % 64
        elif setup:
            klong(setup)
        for _ in range(3):
            np.testing.assert_array_equal(klong(program), expected)
        samples = []
        for _ in range(9):
            wall_start = time.perf_counter_ns()
            cpu_start = time.thread_time_ns()
            answer = klong(program)
            cpu_ns = time.thread_time_ns() - cpu_start
            wall_ns = time.perf_counter_ns() - wall_start
            np.testing.assert_array_equal(answer, expected)
            samples.append({"wall_ns": wall_ns, "cpu_ns": cpu_ns})
        rows[name] = samples
        print(f"checked {name}", file=sys.stderr, flush=True)
    manifest = Path(klongpy.__file__).resolve().parents[1] / "pyproject.toml"
    version = getattr(klongpy, "__version__", None)
    if version is None and manifest.is_file():
        with manifest.open("rb") as stream:
            version = tomllib.load(stream).get("project", {}).get("version")
    result = {
        "module": klongpy.__file__,
        "version": version,
        "python": sys.version,
        "numpy": np.__version__,
        "rows": rows,
    }
    extension = sys.modules.get("klong_native") or sys.modules.get("klongpy._native")
    if extension is not None:
        with Path(extension.__file__).open("rb") as stream:
            result["extension_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def summarize(trials):
    for name, *_ in CASES:
        medians = {
            version: [statistics.median(sample["wall_ns"] for sample in trial[version]["rows"][name])
                      for trial in trials]
            for version in ("v1", "v2")
        }
        ratios = [old / new for old, new in zip(medians["v1"], medians["v2"])]
        old_ms = statistics.median(medians["v1"]) / 1e6
        new_ms = statistics.median(medians["v2"]) / 1e6
        print(f"{name}: {old_ms:.3f} ms -> {new_ms:.3f} ms, "
              f"{statistics.median(ratios):.2f}x ({min(ratios):.2f}–{max(ratios):.2f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--v1-source", type=Path, help="directory containing the v1 klongpy package")
    parser.add_argument("--v2-source", type=Path, default=Path.cwd(), help="v2 source root (default: current directory)")
    parser.add_argument("--v2-native", type=Path, help="directory containing the matching extension, e.g. klong_native.abi3.so")
    parser.add_argument("--pairs", type=int, default=3, help="number of paired trials (default: 3)")
    parser.add_argument("--out", type=Path, help="write every individual timing sample to this JSON file")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker()))
        return
    if args.v1_source is None or args.out is None:
        parser.error("provide --v1-source and --out")
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    paths = {"v1": args.v1_source.resolve(), "v2": args.v2_source.resolve()}
    for path in paths.values():
        if not (path / "klongpy" / "__init__.py").is_file():
            parser.error(f"{path} does not contain a klongpy package")
    if paths["v1"] == paths["v2"]:
        parser.error("--v1-source and --v2-source must differ")
    trials = []
    for index in range(args.pairs):
        pair = {"trial": index + 1}
        order = ("v1", "v2") if index % 2 == 0 else ("v2", "v1")
        for version in order:
            print(f"pair {index + 1}: {version}", file=sys.stderr, flush=True)
            path = paths[version]
            env = os.environ.copy()
            env.update(PYTHONPATH=str(path), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
            if version == "v1":
                env.pop("KLONGPY_NATIVE_PATH", None)
            elif args.v2_native:
                env["KLONGPY_NATIVE_PATH"] = str(args.v2_native.resolve())
            run = json.loads(subprocess.check_output(
                [sys.executable, str(Path(__file__).resolve()), "--worker"], cwd=path, env=env, text=True,
            ))
            if Path(run["module"]).resolve().parent.parent != path:
                raise RuntimeError(f"{version} loaded KlongPy from {run['module']}, expected {path}")
            pair[version] = run
        trials.append(pair)
        args.out.write_text(json.dumps(trials, indent=2) + "\n")
    summarize(trials)


if __name__ == "__main__":
    main()
