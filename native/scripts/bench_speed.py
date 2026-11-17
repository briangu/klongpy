#!/usr/bin/env python3
"""A small speed baseline for the native Klong runtime: the same four programs on the host binary, in the
simulator (virtual time: what the model of the Pico says, from its fitted per-call cost) and, with a bridge,
on a real node. Prints a table; `--json FILE` keeps the numbers for a comparison.

    native/scripts/bench_speed.py                                   # the host binary
    native/scripts/bench_speed.py --host-bin /path/to/other/klong   # another build, to compare
    native/scripts/bench_speed.py --sim                             # also the simulator's model of the board
    KLONG_BENCH_HOST=bench.local native/scripts/bench_speed.py --node 3   # also node 3 through the bridge

The programs (sizes differ per target: a Pico takes about 95 us for a call of a Klong function):
    fib       fib::{:[x<2;x;fib(x-1)+fib(x-2)]}; fib(N)       function calls, integer compare and add
    vsum      +/!N                                            a vector made and summed (one allocation, no calls)
    strbuild  s::"";{{s::s,"ab"}'!x;#s}(N)                    a string grown by joining (N allocations)
    dict      d:::{};{{d,x,,x*x}'!x;#d}(N)                    a dictionary updated N times
Each prints its own time with `.pc()`; the results are checked against each other.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
BENCHES = [
    # name, setup, expression, per-target N, unit and count of units for the per-unit column
    ("fib", "fib::{:[x<2;x;fib(x-1)+fib(x-2)]}", "fib({n})", {"host": 24, "sim": 16, "node": 16}, "calls", lambda n: 2 * round(1.618033988749895 ** (n + 1) / 2.2360679775) - 1 if n > 1 else 1),
    ("vsum", "", "+/!{n}", {"host": 2_000_000, "sim": 20_000, "node": 5_000}, "elements", lambda n: n),
    ("strbuild", 's::"";g::{{s::s,"ab"}\'!x;#s}', "g({n})", {"host": 20_000, "sim": 500, "node": 300}, "joins", lambda n: n),
    ("dict", "d:::{};g::{{d,x,,x*x}\'!x;#d}", "g({n})", {"host": 20_000, "sim": 500, "node": 300}, "updates", lambda n: n),
]


def program(setup, expr, n):
    lines = []
    if setup:
        lines.append(setup)
    lines.append("t::.pc();r::%s;.p((.pc()-t),r)" % expr.replace("{n}", str(n)))
    return lines


def parse(text):
    ms = re.findall(r"\[([0-9.eE+-]+) ([^\]\s]+)\]", text)
    if not ms:
        raise ValueError("no timing in %r" % text[-200:])
    return float(ms[-1][0]), ms[-1][1]


def run_host(binary, name, setup, expr, n):
    src = "\n".join(program(setup, expr, n)) + "\n"
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".kg", delete=False) as f:
        f.write(src)
    try:
        out = subprocess.run([binary, f.name], capture_output=True, text=True, timeout=600).stdout
    finally:
        os.unlink(f.name)
    return parse(out)


def run_sim(sim, name, setup, expr, n):
    lines = ["0> " + l for l in program(setup, expr, n)]
    out = subprocess.run([sim, "1"], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=600).stdout
    return parse(out)


def run_node(cli, host, node, name, setup, expr, n):
    out = ""
    for l in program(setup, expr, n):
        r = subprocess.run([sys.executable, cli, "--host", host, "eval", "--node", str(node), l], capture_output=True, text=True, timeout=600)
        out = r.stdout
        if r.returncode:
            raise RuntimeError(r.stderr.strip() or out)
        time.sleep(0.3)
    return parse(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host-bin", default=os.path.join(NATIVE, "target", "release", "klong"))
    ap.add_argument("--sim", action="store_true", help="also the simulator's model of a board (virtual time)")
    ap.add_argument("--node", type=int, help="also this node, through the bridge named by KLONG_BENCH_HOST")
    ap.add_argument("--json", help="write the numbers to this file")
    a = ap.parse_args(argv)
    cli = os.path.join(NATIVE, "tools", "klong_bench.py")
    sim = os.path.join(NATIVE, "target", "release", "klong-bench")
    targets = [("host", lambda *x: run_host(a.host_bin, *x))]
    if a.sim:
        targets.append(("sim", lambda *x: run_sim(sim, *x)))
    if a.node is not None:
        h = os.environ.get("KLONG_BENCH_HOST")
        if not h:
            ap.error("--node needs KLONG_BENCH_HOST")
        targets.append(("node", lambda *x: run_node(cli, h, a.node, *x)))
    rows, table = [], {}
    for name, setup, expr, sizes, unit, count in BENCHES:
        for tname, run in targets:
            n = sizes[tname]
            secs, result = run(name, setup, expr, n)
            per = secs / count(n) * 1e6
            rows.append({"bench": name, "target": tname, "n": n, "seconds": secs, "result": result, "us_per_" + unit: per})
            table[(name, tname)] = (n, secs, per, unit, result)
    print("%-9s %-5s %10s %12s %14s  %s" % ("bench", "where", "N", "seconds", "us per unit", "result"))
    for name, setup, expr, sizes, unit, count in BENCHES:
        for tname, _ in targets:
            n, secs, per, unit, result = table[(name, tname)]
            print("%-9s %-5s %10d %12.6f %10.2f %-6s  %s" % (name, tname, n, secs, per, unit[:5], result[:24]))
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
