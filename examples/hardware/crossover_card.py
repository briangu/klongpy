"""
Replay a tick history on the card with the 32-lane image of crossover.kg and time it (docs/hardware.md, Strategies).

    KLONG_HW_HOST=user@host python examples/hardware/crossover_card.py HISTORY.bin PARAMS.bin [--json OUT.json]
                                                [--expect SUMMARY.bin] [--prefix N --expect-prefix SUMMARY.bin] [--runs R]

HISTORY.bin and PARAMS.bin are the files of crossover_bench.py (the history as fixed<32,16> raw int32, one row of
fast, slow, stop per lane). The image must have been built:  kgpy --hw-build examples/hardware/crossover.kg --lanes 32
(with the clock you chose); this script binds it the way Klong does (``.hw.lanes``), sets the parameters of the lanes,
uploads the history once and runs it. ``--expect`` is a summary file (int64 [lanes, 3]) that the full replay must equal
(for example the output of the emitted C); ``--expect-prefix`` the same for the first ``--prefix`` ticks. The exit
status is 1 if one differs. The times are the server's clock on the machine with the card ("kernel", from the
submission to the completion of the run) and this process's clock ("wall", which includes the ssh round trips).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROGRAM = os.path.join(HERE, "crossover.kg")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("history")
    ap.add_argument("params")
    ap.add_argument("--json")
    ap.add_argument("--expect")
    ap.add_argument("--prefix", type=int, default=100000)
    ap.add_argument("--expect-prefix")
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--lanes", type=int, default=32)
    a = ap.parse_args(argv)
    from klongpy.hw import binding
    ticks = np.fromfile(a.history, dtype="<i4")
    par = np.fromfile(a.params, dtype="<i4").reshape(-1, 3)[:a.lanes]
    result = {"ticks": int(len(ticks)), "lanes": int(a.lanes)}
    t0 = time.perf_counter()
    b = binding.bind([PROGRAM], lanes=a.lanes)
    result["bind_s"] = time.perf_counter() - t0
    result["image"] = os.path.basename(b.path)
    result["mode"] = b.mode
    result["build"] = {k: b.build.get(k) for k in ("key", "clock_mhz_requested", "clock_mhz_image", "wall_seconds", "timing", "utilisation")}
    b.set_register("fast", par[:, 0].astype(np.int64))
    b.set_register("slow", par[:, 1].astype(np.int64))
    b.set_register("stop", par[:, 2].astype(np.float64) / 65536.0)
    step = b.by_name["step"][0]
    k = b.session.kernel(step.top)
    # upload the history once: it stays in card memory
    t0 = time.perf_counter()
    h, ph = b.session.put(ticks)
    result["upload"] = {"bytes": int(ticks.nbytes), "server_s": ph["seconds"], "wall_s": time.perf_counter() - t0}
    status = 0

    def replay(n):
        t = time.perf_counter()
        out = k.run(h, reset=True, n=n)
        wall = time.perf_counter() - t
        return np.asarray(out), dict(k.last, wall_s=wall)

    # the prefix and the whole history
    out_prefix, last = replay(a.prefix)
    result["prefix"] = {"ticks": a.prefix, "kernel_s": last["kernel_s"], "wall_s": last["wall_s"]}
    if a.expect_prefix:
        want = np.fromfile(a.expect_prefix, dtype="<i8").reshape(-1, 3)[:a.lanes]
        result["prefix"]["equal"] = bool(np.array_equal(out_prefix, want))
        status |= 0 if result["prefix"]["equal"] else 1
    runs = []
    out = None
    for i in range(a.runs):
        out, last = replay(len(ticks))
        runs.append({"kernel_s": last["kernel_s"], "wall_s": last["wall_s"], "get_s": last["get_s"]})
    result["runs"] = runs
    best = min(r["kernel_s"] for r in runs)
    result["best_kernel_s"] = best
    result["best_wall_s"] = min(r["wall_s"] for r in runs)
    result["ticks_per_s_per_lane_kernel"] = len(ticks) / best
    result["ticks_per_s_total_kernel"] = len(ticks) * a.lanes / best
    result["ticks_per_s_per_lane_wall"] = len(ticks) / result["best_wall_s"]
    result["ticks_per_s_total_wall"] = len(ticks) * a.lanes / result["best_wall_s"]
    result["end_to_end_with_upload_s"] = result["upload"]["wall_s"] + result["best_wall_s"]
    result["status_word"] = int(last["status"])
    result["summary"] = (np.asarray(out) / 2.0 ** 32).tolist()
    if a.expect:
        want = np.fromfile(a.expect, dtype="<i8").reshape(-1, 3)[:a.lanes]
        result["full_equal"] = bool(np.array_equal(out, want))
        status |= 0 if result["full_equal"] else 1
    np.asarray(out, dtype="<i8").tofile((a.json or "card").rsplit(".", 1)[0] + ".summary.bin")
    text = json.dumps(result, indent=1, default=str)
    if a.json:
        with open(a.json, "w") as f:
            f.write(text)
    print(text)
    binding.close_all()
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
