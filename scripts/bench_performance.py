#!/usr/bin/env python3
"""
Performance measurements for docs/performance.md. One process measures one configuration.

    python scripts/bench_performance.py --backend numpy  --out numpy.json
    python scripts/bench_performance.py --backend torch  --out torch.json     # torch CPU
    python scripts/bench_performance.py --native native/target/release/klong --out native.json
    python scripts/bench_performance.py --baseline --out baseline.json         # plain Python and NumPy

Thread settings are taken from the environment (set OMP_NUM_THREADS=1 and MKL/OPENBLAS likewise
for a single-threaded run); the torch thread count is set with --torch-threads.
Each row is run REPS times after WARMUP untimed runs; best and median seconds per run are stored.
Rows through `.pyf` are library calls (the work is done by NumPy or torch, not by the interpreter).
"""
import argparse, json, os, platform, re, statistics, subprocess, sys, time, warnings

REPS, WARMUP = 9, 3
SIZES_VEC = [1000, 10000, 100000, 1000000, 10000000]


def timeit_rows(fn, reps=REPS, warmup=WARMUP, min_time=0.0):
    """Seconds per call. Calls faster than min_time are batched so the timer resolution does not matter."""
    for _ in range(warmup):
        fn()
    t = time.perf_counter(); fn(); one = time.perf_counter() - t
    batch = max(1, int(min_time / one)) if one < min_time else 1
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(batch):
            fn()
        ts.append((time.perf_counter() - t) / batch)
    return {"best": min(ts), "median": statistics.median(ts), "reps": reps, "batch": batch}


def rec(rows, topic, name, size, d, **kw):
    d = dict(d, topic=topic, name=name, size=size, **kw)
    rows.append(d)
    print("%-10s %-34s %10s best %.6g median %.6g" % (topic, name, size, d["best"], d["median"]), flush=True)


def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)


# (topic, name, size, setup program, program). N is substituted.
def programs():
    P = []
    P.append(("scalar", "recursive fib(24)", 24, "fib::{:[x<2;x;fib(x-1)+fib(x-2)]}", "fib(24)"))
    P.append(("scalar", "while loop to 100000", 100000, "", "{x<100000}{x+1}:~0"))
    P.append(("lambda", "{x+y}/ over !100000", 100000, "", "{x+y}/!100000"))
    P.append(("lambda", "{x+y}\\ over !10000 (scan)", 10000, "", "{x+y}\\!10000"))
    P.append(("lambda", "#{x*x}'!100000 (each)", 100000, "", "#{x*x}'!100000"))
    P.append(("lambda", "+/{x*x}'!1000000", 1000000, "", "+/{x*x}'!1000000"))
    P.append(("list", "append in a loop, 2000", 2000, "", "a::[];{a::a,x;x}'!2000;#a"))
    for n in SIZES_VEC:
        P.append(("vector", "2*1+!N", n, "", "2*1+!%d" % n))
    for n in (100000, 1000000, 10000000):
        P.append(("reduce", "+/!N", n, "", "+/!%d" % n))
    for n in (100000, 1000000):
        P.append(("reduce", "+\\!N (scan)", n, "", "+\\!%d" % n))
        P.append(("reduce", "+/(!N)*(!N) (dot)", n, "", "+/(!%d)*(!%d)" % (n, n)))
        P.append(("reduce", "(1+!N)^2", n, "", "(1+!%d)^2" % n))
    P.append(("reduce", "+100 100^!10000 (transpose)", 10000, "", "+100 100^!10000"))
    for n in (10000, 100000, 1000000):
        P.append(("sort", "<!N (grade of a sorted range)", n, "", "<!%d" % n))
    for n in (10000, 100000, 1000000):
        P.append(("sort", "<r (grade, random floats)", n, "r::random", "<r"))
        P.append(("sort", "r@<r (sort, random floats)", n, "r::random", "r@<r"))
    P.append(("string", "format and join 20000 numbers", 20000, "", "#,/$'!20000"))
    P.append(("string", "find in a 60000 char text", 60000, "", 's::,/20000#,"ab ";(#s?"b"),#s?0ca'))
    P.append(("string", "index a non-ASCII string, 2000", 2000, "", 's::2000#"aé";#{s@x}\'!2000'))
    P.append(("dict", "50000 inserts and lookups", 50000, "", "d:::{};{d,x,,x*2}'!50000;+/{d?x}'!50000"))
    P.append(("dict", "20000 string keys", 20000, "", "d:::{};k::$'!20000;{d,x,,#x}'k;+/{d?x}'k"))
    return P


def run_klongpy(args):
    sys.setrecursionlimit(100000)
    warnings.simplefilter("ignore")
    import numpy as np
    from klongpy import KlongInterpreter
    rows = []
    be = args.backend
    kw = dict(backend=be, device="cpu") if be == "torch" else dict(backend=be)
    if be == "torch":
        import torch
        torch.set_num_threads(args.torch_threads)
    for topic, name, size, setup, prog in programs():
        # random floats are generated outside the interpreter so both backends sort the same data
        k = KlongInterpreter(**kw)
        if setup.startswith("r::"):
            np.random.seed(1)
            k["r"] = k._backend.kg_asarray(np.random.rand(size))
        elif setup:
            k(setup)
        min_time = 0.02 if topic in ("vector", "reduce", "sort") and size <= 100000 else 0.0
        try:
            d = timeit_rows(lambda: k(prog), min_time=min_time)
        except Exception as e:
            print("FAILED", name, size, repr(e)[:100]); continue
        rec(rows, topic, name, size, d, backend=be, program=prog)
    # library calls through .pyf: matmul float32
    for n in (1000, 2000, 4000):
        k = KlongInterpreter(**kw)
        a = np.random.rand(n, n).astype(np.float32); b = np.random.rand(n, n).astype(np.float32)
        k["a"] = k._backend.kg_asarray(a); k["b"] = k._backend.kg_asarray(b)
        k('.pyf("torch";"matmul")' if be == "torch" else '.pyf("numpy";"matmul")')
        d = timeit_rows(lambda: k("matmul(a;b)"), reps=REPS, warmup=3)
        rec(rows, "matmul", "matmul via .pyf (library call), float32", n, d, backend=be, program="matmul(a;b)")
    return rows


def run_baseline(args):
    import numpy as np
    sys.setrecursionlimit(100000)
    rows = []
    def add(topic, name, size, fn, mt=0.0):
        rec(rows, topic, name, size, timeit_rows(fn, min_time=mt), backend="python/numpy")
    add("scalar", "recursive fib(24) [python]", 24, lambda: fib(24))
    def wl():
        i = 0
        while i < 100000: i += 1
    add("scalar", "while loop to 100000 [python]", 100000, wl)
    def red():
        from functools import reduce
        return reduce(lambda x, y: x + y, range(100000))
    add("lambda", "reduce(lambda) over range(100000) [python]", 100000, red)
    def scan():
        from itertools import accumulate
        return list(accumulate(range(10000), lambda x, y: x + y))
    add("lambda", "accumulate(lambda) 10000 [python]", 10000, scan)
    add("lambda", "[x*x for x in range(100000)] [python]", 100000, lambda: len([x * x for x in range(100000)]))
    add("lambda", "sum(x*x for 1e6) [python]", 1000000, lambda: sum(x * x for x in range(1000000)))
    def app():
        a = []
        for x in range(2000): a = a + [x]
    add("list", "append in a loop, 2000 [python]", 2000, app)
    for n in SIZES_VEC:
        add("vector", "2*(1+np.arange(N)) [numpy]", n, lambda n=n: 2 * (1 + np.arange(n)), 0.02 if n <= 100000 else 0)
    for n in (100000, 1000000, 10000000):
        add("reduce", "np.arange(N).sum() [numpy]", n, lambda n=n: np.arange(n).sum(), 0.02 if n <= 100000 else 0)
    for n in (100000, 1000000):
        add("reduce", "np.cumsum(np.arange(N)) [numpy]", n, lambda n=n: np.cumsum(np.arange(n)), 0.02)
        add("reduce", "np.dot(arange,arange) [numpy]", n, lambda n=n: np.dot(np.arange(n), np.arange(n)), 0.02)
        add("reduce", "(1+np.arange(N))**2 [numpy]", n, lambda n=n: (1 + np.arange(n)) ** 2, 0.02)
    add("reduce", "transpose of 100x100 [numpy]", 10000, lambda: np.arange(10000).reshape(100, 100).T.copy(), 0.02)
    for n in (10000, 100000, 1000000):
        add("sort", "np.argsort(np.arange(N)) [numpy]", n, lambda n=n: np.argsort(np.arange(n)), 0.02 if n <= 100000 else 0)
    for n in (10000, 100000, 1000000):
        np.random.seed(1); r = np.random.rand(n); rl = r.tolist()
        add("sort", "np.argsort(random floats) [numpy]", n, lambda r=r: np.argsort(r), 0.02 if n <= 100000 else 0)
        add("sort", "r[np.argsort(r)] [numpy]", n, lambda r=r: r[np.argsort(r)], 0.02 if n <= 100000 else 0)
        add("sort", "np.sort(random floats) [numpy]", n, lambda r=r: np.sort(r), 0.02 if n <= 100000 else 0)
        add("sort", "sorted(list of random floats) [python]", n, lambda rl=rl: sorted(rl))
        add("sort", "sorted(range(N), key=r.__getitem__) [python]", n, lambda rl=rl, n=n: sorted(range(n), key=rl.__getitem__))
    add("string", "len(''.join(map(str, range(20000)))) [python]", 20000, lambda: len("".join(map(str, range(20000)))))
    def find():
        s = "".join(["ab "] * 20000)
        return s.find("b"), len(s)
    add("string", "build and find in a 60000 char text [python]", 60000, find)
    s = "aé" * 1000
    add("string", "index a non-ASCII string, 2000 [python]", 2000, lambda: [s[i] for i in range(2000)])
    def dd():
        d = {}
        for x in range(50000): d[x] = x * 2
        return sum(d[x] for x in range(50000))
    add("dict", "50000 inserts and lookups [python]", 50000, dd)
    def ds():
        k = [str(i) for i in range(20000)]
        d = {}
        for x in k: d[x] = len(x)
        return sum(d[x] for x in k)
    add("dict", "20000 string keys [python]", 20000, ds)
    for n in (1000, 2000, 4000):
        a = np.random.rand(n, n).astype(np.float32); b = np.random.rand(n, n).astype(np.float32)
        add("matmul", "np.matmul float32 [numpy]", n, lambda a=a, b=b: np.matmul(a, b))
    try:
        import torch
        torch.set_num_threads(args.torch_threads)
        for n in (1000, 2000, 4000):
            a = torch.rand(n, n); b = torch.rand(n, n)
            add("matmul", "torch.matmul float32 [torch]", n, lambda a=a, b=b: torch.matmul(a, b))
        for n in (100000, 1000000, 10000000):
            add("vector", "2*(1+torch.arange(N)) [torch]", n, lambda n=n: 2 * (1 + torch.arange(n)), 0.02 if n <= 100000 else 0)
    except ImportError:
        pass
    return rows


def run_native(args):
    rows = []
    NWARM, NREP = 2, REPS
    for topic, name, size, setup, prog in programs():
        if setup.startswith("r::"):
            continue  # random data is injected from Python; the native runtime rows use _rand below
        src = (setup + ";" if setup else "") + prog
        ts = []
        try:
            for i in range(NWARM + NREP):
                p = subprocess.run([args.native, "--bench", "1", "-e", src], capture_output=True, text=True, timeout=300)
                m = re.search(r"bench: (\d+) ns", p.stderr)
                if not m:
                    raise RuntimeError((p.stdout + p.stderr)[:100])
                if i >= NWARM:
                    ts.append(int(m.group(1)) / 1e9)
        except Exception as e:
            print("FAILED", name, size, repr(e)[:100]); continue
        rec(rows, topic, name, size, {"best": min(ts), "median": statistics.median(ts), "reps": NREP, "batch": 1},
            backend="native", program=src, result=p.stdout.strip()[:40])
    return rows


def main():
    ap = argparse.ArgumentParser(description="Measure one configuration and write the rows as JSON (see the module docstring).")
    ap.add_argument("--backend", choices=["numpy", "torch"], help="time KlongPy on this backend (torch runs on the CPU)")
    ap.add_argument("--baseline", action="store_true", help="time the CPython and NumPy equivalents instead of KlongPy")
    ap.add_argument("--native", metavar="BINARY", help="time the native runtime binary (native/target/release/klong)")
    ap.add_argument("--torch-threads", type=int, default=0, help="torch thread count; 0 keeps the torch default")
    ap.add_argument("--out", required=True, help="JSON file to write")
    args = ap.parse_args()
    if args.torch_threads == 0:
        try:
            import torch; args.torch_threads = torch.get_num_threads()
        except ImportError:
            pass
    load0 = os.getloadavg()
    rows = run_native(args) if args.native else run_baseline(args) if args.baseline else run_klongpy(args)
    meta = {"load_before": load0, "load_after": os.getloadavg(), "argv": sys.argv, "python": platform.python_version(),
            "env": {k: v for k, v in os.environ.items() if k.endswith("_NUM_THREADS")},
            "torch_threads": args.torch_threads, "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
