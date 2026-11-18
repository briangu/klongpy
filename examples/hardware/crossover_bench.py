"""
The measurement of examples/hardware/crossover.kg: a tick history, the parameter sets of the lanes, the same strategy in
numpy, and the C harness that runs the emitted C on a CPU. Python 3.8 and numpy only, so it runs on the machine with
the card as well as on the workstation. See docs/hardware.md, section Strategies.

    python3 crossover_bench.py gen HISTORY.bin PARAMS.bin [--ticks N] [--lanes L] [--seed S]
    python3 crossover_bench.py numpy HISTORY.bin PARAMS.bin [--ticks N] [--out SUMMARY.bin]
    python3 crossover_bench.py cfile DIR       writes DIR/cpu_bench.c, the harness of the emitted C
    python3 crossover_bench.py cpu DIR HISTORY.bin PARAMS.bin [--procs 1,16,32] [--cflags "-O3"] [--json OUT.json]
                                               compile DIR/crossover.c (kgpy compile c) with the harness and time it
    python3 crossover_bench.py interp HISTORY.bin PARAMS.bin --ticks N --out SUMMARY.bin [--procs P]
                                               the interpreter stand-in (needs klongpy), one process per lane at a time

HISTORY.bin is the price of every tick as little-endian int32, fixed<32,16> raw (65536 is 1.0). PARAMS.bin is, per
lane, int32 fast, int32 slow, int32 stop (fixed<32,16> raw). A summary file is, per lane, three int64 (fixed<64,32>
raw): profit, maximum drawdown, trades.
"""
import argparse
import sys
import time

import numpy as np

ONE = 1 << 16


def param_grid(lanes=64):
    """The parameter sets of the lanes: fast, slow (ticks) and stop (price units) as three arrays of length ``lanes``."""
    fast, slow, stop = [2, 4, 8, 12], [16, 20, 26, 32], [0.0625, 0.125, 0.25, 1.0]
    grid = [(f, s, p) for f in fast for s in slow for p in stop]
    # up to 64 lanes: evenly spaced sets of the 4 x 4 x 4 grid (every window and stop distance appears); more: the grid again
    out = [grid[(i * len(grid) // lanes) % len(grid)] if lanes <= len(grid) else grid[i % len(grid)] for i in range(lanes)]
    return (np.array([g[0] for g in out], dtype=np.int32), np.array([g[1] for g in out], dtype=np.int32),
            np.array([int(round(g[2] * ONE)) for g in out], dtype=np.int32))


def history(n, seed=113):
    """
    A synthetic price: a random walk with a rare jump, folded (reflected) into the band 5 to 15, as fixed<32,16>
    raw integers. The sums of 32 prices and their products with a window length stay far inside 32 bits.
    """
    rng = np.random.default_rng(seed)
    steps = rng.integers(-2048, 2049, n, dtype=np.int64)
    jump = rng.random(n) < 0.002
    steps = steps + np.where(jump, rng.integers(-3 * ONE, 3 * ONE, n, dtype=np.int64), 0)
    y = np.cumsum(steps) + 10 * ONE
    lo, w = 5 * ONE, 10 * ONE
    m = np.mod(y - lo, 2 * w)
    return (lo + np.where(m < w, m, 2 * w - m)).astype("<i4")


def wrap32(v):
    return ((v + (1 << 31)) & ((1 << 32) - 1)) - (1 << 31)


def numpy_run(ticks, fast, slow, stop, chunk=1 << 16):
    """
    The strategy of crossover.kg in numpy, integers only. The two sums and the signal of every tick are computed for
    all lanes and a block of ticks at once (prefix sums); the position, which depends on its own past, is stepped one
    tick at a time over all lanes. Returns an int64 array [lanes, 3]: profit, maximum drawdown, trades (fixed<64,32> raw).
    """
    ticks = np.asarray(ticks, dtype=np.int64)
    L = len(fast)
    fw = np.clip(np.asarray(fast, dtype=np.int64), 1, 32)
    sw = np.clip(np.asarray(slow, dtype=np.int64), 1, 32)
    stop = np.asarray(stop, dtype=np.int64)
    pos = np.zeros(L, dtype=np.int64)
    blk = np.zeros(L, dtype=np.int64)
    ent = np.zeros(L, dtype=np.int64)
    prev = 0
    trades = np.zeros(L, dtype=np.int64)
    profit = np.zeros(L, dtype=np.int64)
    peak = np.zeros(L, dtype=np.int64)
    maxdd = np.zeros(L, dtype=np.int64)
    n = len(ticks)
    cum = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(ticks)])
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        t = np.arange(c0, c1)
        sf = cum[t + 1][None, :] - cum[np.maximum(0, t[None, :] + 1 - fw[:, None])]
        ss = cum[t + 1][None, :] - cum[np.maximum(0, t[None, :] + 1 - sw[:, None])]
        d = wrap32(wrap32(sf * sw[:, None]) - wrap32(ss * fw[:, None]))
        ready = np.minimum(t + 1, 64)[None, :] > (sw[:, None] - 1)
        g = ready * ((d > 0).astype(np.int64) - (d < 0).astype(np.int64))
        for i in range(c1 - c0):
            x = int(ticks[c0 + i])
            gi = g[:, i]
            pnl = np.where(pos > 0, x - prev, np.where(pos < 0, prev - x, 0))
            profit = profit + pnl
            hit = np.where(pos > 0, ent > wrap32(x + stop), np.where(pos < 0, wrap32(x - stop) > ent, False))
            al = np.where(gi == blk, blk, 0)
            ta = np.where(gi == al, 0, gi)
            tb = np.where(gi == pos, 0, gi)
            ea = np.where(ta == pos, ent, x)
            eb = np.where(tb == pos, ent, x)
            trades = trades + np.where(hit, (tb != pos) & (tb != 0), (ta != pos) & (ta != 0))
            ent = np.where(hit, eb, ea)
            blk = np.where(hit, pos, al)
            pos = np.where(hit, tb, ta)
            prev = x
            peak = np.maximum(peak, profit)
            maxdd = np.maximum(maxdd, peak - profit)
    out = np.empty((L, 3), dtype=np.int64)
    out[:, 0] = profit * ONE          # fixed<32,16> to fixed<64,32>: 16 more fraction bits
    out[:, 1] = maxdd * ONE
    out[:, 2] = trades * (1 << 32)
    return out


C_HARNESS = r'''/* cpu_bench.c - the emitted C of crossover.kg on one core, a share of the lanes over the whole history.
 *   cpu_bench HISTORY.bin PARAMS.bin FIRST_LANE LAST_LANE(excl.) OUT.bin [NTICKS]
 * OUT.bin: per lane in the share, three int64 (profit, maximum drawdown, trades) and a double (seconds for the lane). */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "crossover.h"

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static void *slurp(const char *path, size_t *bytes, size_t limit) {
    FILE *f = fopen(path, "rb");
    void *p;
    if (!f) { perror(path); exit(2); }
    fseek(f, 0, SEEK_END);
    *bytes = (size_t)ftell(f);
    if (limit && limit < *bytes) { *bytes = limit; }
    fseek(f, 0, SEEK_SET);
    p = malloc(*bytes);
    if (fread(p, 1, *bytes, f) != *bytes) { perror("read"); exit(2); }
    fclose(f);
    return p;
}

int main(int argc, char **argv) {
    size_t hb, pb;
    int32_t *ticks, *par;
    long n, l0, l1, l;
    FILE *out;
    double t_all;
    if (argc < 6) { fprintf(stderr, "usage: cpu_bench HISTORY PARAMS FIRST LAST OUT [NTICKS]\n"); return 2; }
    ticks = (int32_t *)slurp(argv[1], &hb, argc > 6 ? 4 * (size_t)atol(argv[6]) : 0);       /* only the ticks that are run are read */
    par = (int32_t *)slurp(argv[2], &pb, 0);
    n = (long)(hb / 4);
    l0 = atol(argv[3]);
    l1 = atol(argv[4]);
    out = fopen(argv[5], "wb");
    t_all = now();
    for (l = l0; l < l1; l++) {
        crossover_t p;
        int64_t s[3];
        double t0, t1;
        long i;
        crossover_init(&p);
        crossover_set_fast(&p, par[3 * l]);
        crossover_set_slow(&p, par[3 * l + 1]);
        crossover_set_stop(&p, par[3 * l + 2]);
        t0 = now();
        for (i = 0; i < n; i++) {
            crossover_step(&p, ticks[i]);
        }
        t1 = now();
        crossover_summary(&p, s);
        fwrite(s, sizeof s, 1, out);
        t1 -= t0;
        fwrite(&t1, sizeof t1, 1, out);
    }
    t_all = now() - t_all;
    fclose(out);
    printf("lanes %ld..%ld ticks %ld seconds %.6f\n", l0, l1, n, t_all);
    return 0;
}
'''


def cpu_run(d, history, params, procs=(1,), cflags=("-O3",), cc="gcc", ticks=0):
    """
    Compile the emitted C of crossover.kg in directory d with the harness and run it over all lanes with each number of
    processes in ``procs`` (one process per share of the lanes, all started together). Returns a list of dicts with the
    wall seconds, the mean seconds a lane took inside its process, and the summary of every lane (int64 [lanes, 3]).
    """
    import os
    import subprocess
    with open(os.path.join(d, "cpu_bench.c"), "w") as f:
        f.write(C_HARNESS)
    exe = os.path.join(d, "cpu_bench")
    subprocess.run([cc, "-std=c99"] + list(cflags) + ["-I" + d, os.path.join(d, "crossover.c"), os.path.join(d, "cpu_bench.c"),
                    "-o", exe, "-lm"], check=True)
    lanes = len(np.fromfile(params, dtype="<i4")) // 3
    n = ticks or len(np.fromfile(history, dtype="<i4"))
    results = []
    for P in procs:
        shares = [(i * lanes // P, (i + 1) * lanes // P) for i in range(P)]
        shares = [s for s in shares if s[1] > s[0]]
        t0 = time.perf_counter()
        ps = [subprocess.Popen([exe, history, params, str(a), str(b), os.path.join(d, "out_%d_%d.bin" % (P, a)), str(n)],
                               stdout=subprocess.PIPE) for a, b in shares]
        for p in ps:
            p.communicate()
        wall = time.perf_counter() - t0
        recs = []
        lane_s = []
        for a, b in shares:
            raw = np.fromfile(os.path.join(d, "out_%d_%d.bin" % (P, a)), dtype=np.uint8).reshape(b - a, 32)
            recs.append(raw[:, :24].copy().view("<i8").reshape(b - a, 3))
            lane_s.extend(raw[:, 24:].copy().view("<f8").reshape(-1).tolist())
        results.append({"processes": len(shares), "wall_s": wall, "mean_lane_s": float(np.mean(lane_s)), "ticks": n, "lanes": lanes,
                        "summary": np.concatenate(recs)})
    return results


def _interp_lane(args):
    path, ticks, fast, slow, stop = args
    import numpy as np
    from klongpy.static.compile import compile_program
    from klongpy.static.eval_ir import IrEvaluator
    import os
    prog = compile_program(os.path.join(os.path.dirname(os.path.abspath(__file__)), "crossover.kg")).program
    ev = IrEvaluator(prog)
    ev.state["fast"], ev.state["slow"], ev.state["stop"] = np.array(fast), np.array(slow), np.array(stop)
    step = next(e for e in prog.entries if e.decl.name == "step")
    summ = next(e for e in prog.entries if e.decl.name == "summary")
    for x in ticks:
        ev.run(step, [np.array(int(x))])
    return np.asarray(ev.run(summ, []), dtype=np.int64)


def interp_run(ticks, fast, slow, stop, procs=8):
    """The interpreter stand-in (klongpy.static.eval_ir on the typed IR): one lane per task, ``procs`` at a time."""
    import multiprocessing
    jobs = [(None, ticks, int(f), int(s), int(p)) for f, s, p in zip(fast, slow, stop)]
    with multiprocessing.Pool(procs) as pool:
        return np.array(pool.map(_interp_lane, jobs, chunksize=1))


def main(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("history")
    g.add_argument("params")
    g.add_argument("--ticks", type=int, default=10_000_000)
    g.add_argument("--lanes", type=int, default=32)
    g.add_argument("--seed", type=int, default=113)
    r = sub.add_parser("numpy")
    r.add_argument("history")
    r.add_argument("params")
    r.add_argument("--ticks", type=int, default=0)
    r.add_argument("--out")
    c = sub.add_parser("cfile")
    c.add_argument("dir")
    u = sub.add_parser("cpu")
    u.add_argument("dir")
    u.add_argument("history")
    u.add_argument("params")
    u.add_argument("--procs", default="1")
    u.add_argument("--cflags", default="-O3")
    u.add_argument("--ticks", type=int, default=0)
    u.add_argument("--json")
    u.add_argument("--out")
    i = sub.add_parser("interp")
    i.add_argument("history")
    i.add_argument("params")
    i.add_argument("--ticks", type=int, default=100000)
    i.add_argument("--out", required=True)
    i.add_argument("--procs", type=int, default=8)
    a = ap.parse_args(argv)
    if a.cmd == "gen":
        h = history(a.ticks, a.seed)
        f, s, p = param_grid(a.lanes)
        h.tofile(a.history)
        np.stack([f, s, p], axis=1).astype("<i4").tofile(a.params)
        print("wrote %d ticks (%.2f to %.2f) and %d parameter sets" % (len(h), h.min() / ONE, h.max() / ONE, a.lanes))
    elif a.cmd == "numpy":
        h = np.fromfile(a.history, dtype="<i4")
        if a.ticks:
            h = h[:a.ticks]
        par = np.fromfile(a.params, dtype="<i4").reshape(-1, 3)
        t0 = time.perf_counter()
        out = numpy_run(h, par[:, 0], par[:, 1], par[:, 2])
        dt = time.perf_counter() - t0
        if a.out:
            out.tofile(a.out)
        print("numpy: %d ticks x %d lanes in %.3f s" % (len(h), len(par), dt))
    elif a.cmd == "cpu":
        import json
        res = cpu_run(a.dir, a.history, a.params, [int(x) for x in a.procs.split(",")], a.cflags.split(), ticks=a.ticks)
        first = res[0]["summary"]
        for r in res:
            r["same_as_first"] = bool((r["summary"] == first).all())
            r["ticks_per_s_per_lane"] = r["ticks"] / r["mean_lane_s"]
            r["lane_ticks_per_s_total"] = r["ticks"] * r["lanes"] / r["wall_s"]
        if a.out:
            first.astype("<i8").tofile(a.out)
        slim = [{k: v for k, v in r.items() if k != "summary"} for r in res]
        slim = [dict(r, cflags=a.cflags) for r in slim]
        text = json.dumps(slim, indent=1)
        if a.json:
            with open(a.json, "w") as f:
                f.write(text)
        print(text)
    elif a.cmd == "interp":
        h = np.fromfile(a.history, dtype="<i4")[:a.ticks]
        par = np.fromfile(a.params, dtype="<i4").reshape(-1, 3)
        t0 = time.perf_counter()
        out = interp_run(h, par[:, 0], par[:, 1], par[:, 2], a.procs)
        out.astype("<i8").tofile(a.out)
        print("interpreter: %d ticks x %d lanes in %.1f s on %d processes" % (len(h), len(par), time.perf_counter() - t0, a.procs))
    else:
        with open(a.dir + "/cpu_bench.c", "w") as f:
            f.write(C_HARNESS)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
