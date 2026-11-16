#!/usr/bin/env python3
"""
Micro-benchmarks of the native runtime against KlongPy (ticket 102), and peak heap.

    .venv/bin/python native/scripts/bench.py [path to klong]

Both are timed in process, parse and evaluation of the source text, best of 5 (KlongPy with
`time.perf_counter` around `klong(src)`, the native binary with `klong --bench 5 -e src`).
The last column but one is the native binary as a whole process (start, parse, run, print, exit). Peak heap is what `klong --stats` reports:
bytes requested from the allocator by the program, above the interpreter's own start-up state.
"""
import os
import re
import subprocess
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KLONG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "native", "target", "release", "klong")

PROGRAMS = [
    ("start-up", "1"),
    ("recursive fib(24)", "fib::{:[x<2;x;fib(x-1)+fib(x-2)]};fib(24)"),
    ("+/ over 100000 integers", "+/!100000"),
    ("{x+y}/ over 100000 integers", "{x+y}/!100000"),
    ("each with a lambda, 100000", "#{x*x}'!100000"),
    ("sum of squares, 1e6", "+/{x*x}'!1000000"),
    ("build a list by appending, 2000", "a::[];{a::a,x;x}'!2000;#a"),
]


def best(f, n=5):
    times = []
    for _ in range(n):
        t = time.perf_counter()
        r = f()
        times.append(time.perf_counter() - t)
    return min(times), r


def main():
    sys.path.insert(0, ROOT)
    warnings.simplefilter("ignore")
    from klongpy import KlongInterpreter

    sys.setrecursionlimit(100000)
    print(f"{'program':34} {'KlongPy ms':>11} {'native ms':>10} {'ratio':>8} {'process ms':>11} {'peak heap':>10}")
    for name, src in PROGRAMS:
        klong = KlongInterpreter()
        t_py, r_py = best(lambda: klong(src))
        t_proc, out = best(lambda: subprocess.run([KLONG, "--stats", "-e", src], capture_output=True, text=True))
        b = subprocess.run([KLONG, "--bench", "5", "-e", src], capture_output=True, text=True)
        t_n = int(re.search(r"bench: (\d+) ns", b.stderr).group(1)) / 1e9
        got = out.stdout.strip()
        if str(r_py) != got:
            got += f"  (KlongPy: {r_py})"
        m = re.search(r"peak (\d+) bytes", out.stderr)
        peak = int(m.group(1)) if m else -1
        print(f"{name:34} {t_py * 1000:11.3f} {t_n * 1000:10.3f} {t_py / t_n:7.1f}x {t_proc * 1000:11.2f} {peak:10d}   = {got}")


if __name__ == "__main__":
    main()
