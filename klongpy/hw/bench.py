"""
``python -m klongpy.hw.bench [--host user@host] [--sizes 1,1000,100000]``: end-to-end time of the ``fir`` kernel on
the card against numpy on this machine and on the card host.

For each number of blocks per call it prints, for the card:
    put    host memory to card memory (the server's clock, around the DMA), seconds
    kernel from submit to done (the server's clock), seconds
    get    card memory to host memory, seconds
    wall   the whole call as the client sees it, which includes the ssh pipe between the two machines
and for numpy: the best time of the same computation (the gather and weighted sum that ``fir.kg`` states), here and
on the card host (``--numpy-host``, run there with the numpy of the server's virtual environment).
All of it is measured, not estimated. Needs a built, passing image of examples/embedded/fir.kg.
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

import numpy as np

from .client import HwSession, host_target

# the computation of examples/embedded/fir.kg: run::{+/taps*x@idx} over blocks of 256 samples
NUMPY_FIR = '''
import numpy as np, time, sys
raw = (1.0 + np.minimum(np.arange(16), 15 - np.arange(16))).astype(np.float32)
taps = (raw / raw.sum()).astype(np.float32)
idx = (np.arange(241)[None, :] + np.arange(16)[:, None])          # 16 x 241 sample positions
def fir(x):                                                        # x: (n, 256) -> (n, 241)
    return (taps[None, :, None] * x[:, idx]).sum(axis=1, dtype=np.float32)
out = []
for n in SIZES:
    x = np.random.default_rng(5).random((n, 256), dtype=np.float32)
    best = None
    for _ in range(REPEAT if n < 10000 else 3):
        t0 = time.perf_counter(); fir(x); dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    out.append((n, best))
print(out)
'''


def numpy_times(sizes, repeat=5):
    ns = {"SIZES": sizes, "REPEAT": repeat}
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(NUMPY_FIR, ns)
    return eval(buf.getvalue())


def numpy_times_on(host, sizes, repeat=5):
    code = "SIZES=%r\nREPEAT=%d\n%s" % (sizes, repeat, NUMPY_FIR)
    from .remote import resolve_root
    py = shlex.quote(resolve_root(host) + "/venv/bin/python") + " -"
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", host, py], input=code.encode(),
                       capture_output=True, timeout=1200)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="replace")[-500:])
    return eval(p.stdout.decode().strip().splitlines()[-1])


def card_times(host, image, sizes, repeat=3):
    rows = []
    with HwSession(host, timeout=600) as hw:
        hw.load(image)
        k = hw.kernel("fir_run_kernel")
        for n in sizes:
            x = np.random.default_rng(5).random((n, 256), dtype=np.float32)
            best = None
            for _ in range(repeat if n < 10000 else 2):
                t0 = time.perf_counter()
                k.run(x)
                wall = time.perf_counter() - t0
                r = dict(k.last, wall_s=wall)
                if best is None or r["wall_s"] < best["wall_s"]:
                    best = r
            rows.append((n, best))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m klongpy.hw.bench")
    ap.add_argument("--host", default=host_target())
    ap.add_argument("--sizes", default="1,1000,100000")
    ap.add_argument("--clock-mhz", type=int, default=300)
    ap.add_argument("--numpy-host", action="store_true", help="also run numpy on the card host")
    a = ap.parse_args(argv)
    sizes = [int(s) for s in a.sizes.split(",")]
    from ..static.compile import emit_hls_program  # noqa: F401  (the key needs the programs)
    from .build import find_image
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, "..", ".."))
    files = [os.path.join(root, "examples", "embedded", "mavg.kg"), os.path.join(root, "examples", "embedded", "fir.kg"),
             os.path.join(root, "tests", "static", "hw", "trap.kg")]
    image, meta = find_image(files, host=a.host, clock_mhz=a.clock_mhz)
    print("image %s (WNS %s ns at %s MHz)" % (image, meta["timing"]["wns_ns"], meta.get("clock_mhz_image")))
    here_np = dict(numpy_times(sizes))
    host_np = dict(numpy_times_on(a.host, sizes)) if a.numpy_host and a.host else {}
    print("%8s | %10s %10s %10s %10s | %10s | %10s %10s" % ("blocks", "put s", "kernel s", "get s", "wall s", "card/block", "numpy here", "numpy host"))
    for n, r in card_times(a.host, image, sizes):
        print("%8d | %10.6f %10.6f %10.6f %10.6f | %8.2f us | %10.6f %10s" % (
            n, r["put_s"], r["kernel_s"], r["get_s"], r["wall_s"], r["kernel_s"] / n * 1e6, here_np[n],
            ("%10.6f" % host_np[n]) if n in host_np else "-"))


if __name__ == "__main__":
    sys.exit(main())
