#!/usr/bin/env python3
"""
In-process timings of the engine: scalar and call-heavy programs, and array operations at 1e3, 1e5 and 1e7 elements, with the same
workloads written for CPython and for NumPy as references. Nothing here launches a process per timing: each runner is one process that
builds its data once, runs every workload once to warm up, and keeps the best of the timed runs that follow.

    .venv/bin/python native/scripts/bench_engine.py [--kg PATH] [--ext DIR] [--sizes 1000,100000] [--only SUBSTR] [--json FILE]

  kg    the standalone binary running a Klong script (`--kg`, default native/target/release/kg): time is `.pc()`, the CPU time of the
        process, around a call of the workload, so (a loaded machine enters it less than wall time)
  ext   the extension (`--ext DIR` holds klong_native.abi3.so; it is put on KLONGPATH-style `KLONGPY_NATIVE_PATH`): `KlongInterpreter`
        from klongpy, `time.perf_counter` around the evaluation of `wk(0)` (the workload returns 0, so no result is converted)
  cpy   CPython: lists and loops (the builtins where one does the job: sum, min, sorted, itertools.accumulate)
  np    NumPy: the vectorised form of the array workloads (not defined for the call-heavy ones)

A workload runs once to warm up. When that took 2 s or more it is the figure (one run); otherwise the best of 3 to 30 further runs,
as many as fit in half a second. Each workload is the body of `wk::{expr;0}`: the result is dropped, so what is timed is making it
(and, for the extension, nothing is converted). Sizes of the arrays: ia ints 0..n-1, ib ints 3+ia, fa floats 0.5*ia, fb floats 1.5+fa,
pi a permutation of 0..n-1 (7919*ia mod n), pf floats 0.5*pi.
"""
import argparse
import itertools
import functools
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.abspath(__file__)

# -- the workloads ------------------------------------------------------------------------------------------------------------
# (name, setup Klong, body of wk, CPython callable factory, NumPy callable factory)
SCALAR_N = 100000


def scalar_workloads():
    """Call-heavy programs on scalars. Each: name, Klong definitions, the body of wk, a CPython function (or None)."""
    N = SCALAR_N

    def fib(x):
        return x if x < 2 else fib(x - 1) + fib(x - 2)

    def loop():
        i = 0
        while i < N:
            i += 1
        return i

    def dicts():
        d = {}
        for i in range(50000):
            d[i] = i * 2
        return d

    def strings():
        s = ""
        for _ in range(20000):
            s = s + "ab"
        return s

    def vectors():
        v = []
        for i in range(20000):
            v.append(i)
        return v

    def sq(a):
        return a * a

    ns = type("ns", (), {"f": staticmethod(lambda x: x * x)})

    return [
        ("fib(24) recursive", "fib::{:[x<2;x;fib(x-1)+fib(x-2)]}", "fib(24)", lambda: fib(24)),
        ("loop by while, 1e5", "", "{x<%d}{x+1}:~0" % N, loop),
        ("loop by converge, 1e5", "", "{:[x<%d;x+1;x]}:~0" % N, loop),
        ("reduce with a lambda, 1e5", "", "{x+y}/!%d" % N, lambda: functools.reduce(lambda x, y: x + y, range(N))),
        ("each with a lambda, 1e5", "", "{x*x}'!%d" % N, lambda: list(map(lambda x: x * x, range(N)))),
        ("dictionary updates, 5e4", "", "d:::{};{d,x,,x*2}'!50000", dicts),
        ("string building, 2e4", "", 's::"";{s::s,"ab"}\'!20000', strings),
        ("string append, results dropped, 2e4", "", 's::"";{s::s,"ab";x}\'!20000', strings),
        ("vector append, results dropped, 2e4", "", "v::!0;{v::v,x;x}'!20000", vectors),
        ("named-parameter calls, 1e5", "sq(a)::{a*a}", "{sq(x)}'!%d" % N, lambda: list(map(lambda x: sq(x), range(N)))),
        ("classic calls, 1e5", "sqc::{x*x}", "{sqc(x)}'!%d" % N, lambda: list(map(lambda x: sq(x), range(N)))),
        ("qualified calls (dictionary), 1e5", "m:::{;\"f\";{x*x}}", "{m.f(x)}'!%d" % N, lambda: list(map(lambda x: ns.f(x), range(N)))),
        ("module-qualified calls, 1e5", "@MODULE@", "{mm.f(x)}'!%d" % N, lambda: list(map(lambda x: ns.f(x), range(N)))),
    ]


def array_workloads():
    """Array operations: name, Klong expression, CPython (lists, given a namespace of data) or None, NumPy (given the data)."""
    A = {}

    def w(name, kl, cpy, npy):
        A[name] = (kl, cpy, npy)

    w("add int", "ia+ib", lambda d: [x + y for x, y in zip(d.li, d.lb)], lambda d: d.ia + d.ib)
    w("multiply int", "ia*ib", lambda d: [x * y for x, y in zip(d.li, d.lb)], lambda d: d.ia * d.ib)
    w("add float", "fa+fb", lambda d: [x + y for x, y in zip(d.lf, d.lg)], lambda d: d.fa + d.fb)
    w("multiply float", "fa*fb", lambda d: [x * y for x, y in zip(d.lf, d.lg)], lambda d: d.fa * d.fb)
    w("add mixed int/float", "ia+fb", lambda d: [x + y for x, y in zip(d.li, d.lg)], lambda d: d.ia + d.fb)
    w("chain (a+b)*b int", "(ia+ib)*ib", lambda d: [(x + y) * y for x, y in zip(d.li, d.lb)], lambda d: (d.ia + d.ib) * d.ib)
    w("chain (a*b+a)-b float", "(fa*fb+fa)-fb", lambda d: [(x * y + x) - y for x, y in zip(d.lf, d.lg)], lambda d: (d.fa * d.fb + d.fa) - d.fb)
    w("dot +/a*b float", "+/fa*fb", lambda d: sum(x * y for x, y in zip(d.lf, d.lg)), lambda d: (d.fa * d.fb).sum())
    w("compare int <", "pi<ib", lambda d: [x < y for x, y in zip(d.lp, d.lb)], lambda d: d.pi < d.ib)
    w("sum int", "+/ia", lambda d: sum(d.li), lambda d: d.ia.sum())
    w("sum float", "+/fa", lambda d: sum(d.lf), lambda d: d.fa.sum())
    w("min float", "&/pf", lambda d: min(d.lq), lambda d: d.pf.min())
    w("max int", "|/pi", lambda d: max(d.lp), lambda d: d.pi.max())
    w("scan +\\ int", "+\\ia", lambda d: list(itertools.accumulate(d.li)), lambda d: d.ia.cumsum())
    w("grade up int", "<pi", lambda d: sorted(range(len(d.lp)), key=d.lp.__getitem__), lambda d: d.pi.argsort(kind="stable"))
    w("grade up float", "<pf", lambda d: sorted(range(len(d.lq)), key=d.lq.__getitem__), lambda d: d.pf.argsort(kind="stable"))
    w("sort float (x@<x)", "pf@<pf", lambda d: sorted(d.lq), lambda d: d.pf[d.pf.argsort(kind="stable")])
    w("take h#a", "h#ia", lambda d: d.li[:d.h], lambda d: d.ia[:d.h].copy())
    w("drop h_a", "h_ia", lambda d: d.li[d.h:], lambda d: d.ia[d.h:].copy())
    w("reshape [2 h]:^a", "(2,h):^ia", lambda d: [d.li[:d.h], d.li[d.h:]], lambda d: d.ia.reshape(2, d.h).copy())
    w("find a?(n-1)", "ia?(n-1)", lambda d: [i for i, x in enumerate(d.li) if x == d.n - 1], lambda d: (d.ia == d.n - 1).nonzero()[0])
    w("where &(p>h)", "&pi>h", lambda d: [i for i, x in enumerate(d.lp) if x > d.h], lambda d: (d.pi > d.h).nonzero()[0])
    w("each-pair a{x+y}'b", "ia{x+y}'ib", lambda d: list(map(lambda x, y: x + y, d.li, d.lb)), lambda d: d.ia + d.ib)
    return A


KL_SETUP = ("n::%d;h::n:%%2;ia::!n;ib::3+ia;fa::0.5*ia;fb::1.5+fa;pi::(7919*ia)!n;pf::0.5*pi;")

# (the names of these two functions are odd on purpose: a classic function sees its caller's names, and the workloads use n, h, i, a ...)
KL_BT = ("tt::{[qf qr q0];qf::x;qr::y;q0::.pc();{qf(0)}'!qr;.pc()-q0}"
         "\nbt::{[qf qr qt qn qb qi];qf::x;qr::1;qt::tt(qf;1);{qt<0.003}{qr::qr*4;qt::tt(qf;qr)}:~0;qb::qt%qr;qn:::[qt>2;0;3|30&_0.5%qt];qi::0;"
         "{qi<qn}{qt::tt(qf;qr);qb::qb&qt%qr;qi::qi+1}:~0;qb,qr}")

MODULE_SRC = "f(a)::{a*a}\n.export(:{;\"f\";f})\n"


# -- the runners --------------------------------------------------------------------------------------------------------------

def want(name, only):
    return only is None or only in name


def kg_program(sizes, only, modpath):
    lines = [KL_BT]
    for title, defs, body, _py in scalar_workloads():
        if not want(title, only):
            continue
        defs = defs.replace("@MODULE@", 'mm::.import("%s")' % modpath)
        if defs:
            lines.append(defs)
        lines.append("wk::{%s;0}" % body)
        lines.append('qres::bt(wk);.p("R|%s|",($qres@0),"|",$qres@1)' % title)
    for n in sizes:
        lines.append(KL_SETUP % n)
        for name, (kl, _c, _n) in array_workloads().items():
            if not want(name, only):
                continue
            lines.append("wk::{%s;0}" % kl)
            lines.append('qres::bt(wk);.p("R|%s %d|",($qres@0),"|",$qres@1)' % (name, n))
    return "\n".join(lines) + "\n"


def run_kg(kg, sizes, only):
    out = {}
    with tempfile.TemporaryDirectory() as d:
        mod = os.path.join(d, "benchmod.kg")
        with open(mod, "w") as f:
            f.write(MODULE_SRC)
        prog = os.path.join(d, "bench.kg")
        with open(prog, "w") as f:
            f.write(kg_program(sizes, only, mod[:-3]))
        r = subprocess.run([kg, prog], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("kg failed: " + r.stdout[-500:] + r.stderr[-500:])
    for line in r.stdout.splitlines():
        if line.startswith("R|"):
            _r, name, t, runs = line.split("|")
            out[name] = float(t)
    return out


def best_of(fn):
    """The same method as `bt` in Klong: the first run is the warm-up and calibrates a batch of at least 3 ms (a clock that can resolve it);
    then the best of 3 to 30 batches that fit in half a second, per run; a run of 2 s or more is the figure."""
    t = time.perf_counter()
    fn()
    t = time.perf_counter() - t
    r = 1
    while t < 0.003:
        r = r * 4
        t0 = time.perf_counter()
        for _ in range(r):
            fn()
        t = time.perf_counter() - t0
    best = t / r
    if t > 2:
        return best
    for _ in range(max(3, min(30, int(0.5 / t)))):
        t0 = time.perf_counter()
        for _ in range(r):
            fn()
        best = min(best, (time.perf_counter() - t0) / r)
    return best


def run_ext(ext, sizes, only):
    code = r'''
import json, os, sys, tempfile, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, %r)
sys.setrecursionlimit(100000)
import bench_engine as B
print(json.dumps(B.worker_ext(%r, %r)))
''' % (os.path.dirname(HERE), sizes, only)
    env = dict(os.environ, KLONGPY_NATIVE_PATH=ext, PYTHONPATH=ROOT)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.exit("extension runner failed: " + r.stderr[-800:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def worker_ext(sizes, only):
    from klongpy import KlongInterpreter
    out = {}
    with tempfile.TemporaryDirectory() as d:
        mod = os.path.join(d, "benchmod.kg")
        with open(mod, "w") as f:
            f.write(MODULE_SRC)
        k = KlongInterpreter()
        k(KL_BT)
        for title, defs, body, _py in scalar_workloads():
            if not want(title, only):
                continue
            defs = defs.replace("@MODULE@", 'mm::.import("%s")' % mod[:-3])
            if defs:
                k(defs)
            k("wk::{%s;0}" % body)
            out[title] = float(k("bt(wk)")[0])
        for n in sizes:
            k(KL_SETUP % n)
            for name, (kl, _c, _n) in array_workloads().items():
                if not want(name, only):
                    continue
                k("wk::{%s;0}" % kl)
                out["%s %d" % (name, n)] = float(k("bt(wk)")[0])
    return out


class Data:
    pass


def run_py(sizes, only, numpy_side):
    out = {}
    if not numpy_side:
        sys.setrecursionlimit(100000)
        for title, _defs, _body, fn in scalar_workloads():
            if want(title, only) and fn is not None:
                out[title] = best_of(fn)
    import numpy as np
    for n in sizes:
        d = Data()
        d.n = n
        d.h = n // 2
        d.ia = np.arange(n, dtype=np.int64)
        d.ib = d.ia + 3
        d.fa = d.ia * 0.5
        d.fb = d.fa + 1.5
        d.pi = (d.ia * 7919) % n
        d.pf = d.pi * 0.5
        if not numpy_side:
            d.li, d.lb, d.lf, d.lg, d.lp, d.lq = (x.tolist() for x in (d.ia, d.ib, d.fa, d.fb, d.pi, d.pf))
        for name, (_kl, cpy, npy) in array_workloads().items():
            if want(name, only):
                out["%s %d" % (name, n)] = best_of((lambda: npy(d)) if numpy_side else (lambda: cpy(d)))
        del d
    return out


# -- the table ------------------------------------------------------------------------------------------------------------------

def fmt(t):
    if t is None:
        return "-"
    if t >= 1:
        return "%.2f s" % t
    if t >= 1e-3:
        return "%.2f ms" % (t * 1e3)
    return "%.1f us" % (t * 1e6)


def ratio(a, b):
    return "-" if a is None or b is None else "%.1f" % (a / b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg", default=os.path.join(ROOT, "native", "target", "release", "kg"))
    ap.add_argument("--ext", help="directory with klong_native.abi3.so")
    ap.add_argument("--sizes", default="1000,100000,10000000")
    ap.add_argument("--only", help="run the workloads whose name contains this")
    ap.add_argument("--json", help="keep the numbers here")
    ap.add_argument("--no-ref", action="store_true", help="skip CPython and NumPy")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    res = {"kg": run_kg(a.kg, sizes, a.only)}
    if a.ext:
        res["ext"] = run_ext(a.ext, sizes, a.only)
    if not a.no_ref:
        res["cpy"] = run_py(sizes, a.only, False)
        res["np"] = run_py(sizes, a.only, True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    names = list(res["kg"])
    print("%-40s %10s %10s %10s %10s %8s %8s %8s" % ("workload", "kg", "ext", "CPython", "NumPy", "kg/cpy", "kg/np", "ext/kg"))
    for nm in names:
        g = res["kg"].get(nm)
        e = res.get("ext", {}).get(nm)
        c = res.get("cpy", {}).get(nm)
        p = res.get("np", {}).get(nm)
        print("%-40s %10s %10s %10s %10s %8s %8s %8s" % (nm, fmt(g), fmt(e), fmt(c), fmt(p), ratio(g, c), ratio(g, p), ratio(e, g)))


if __name__ == "__main__":
    main()
