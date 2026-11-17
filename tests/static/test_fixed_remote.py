"""
Fixed-point numbers through Vitis HLS C simulation: the same vectors as test_fixed.py, run
through the generated kernels on the build machine, must give the scaled integers of the reference semantics
(klongpy/static/fixed.py, which test_fixed.py holds equal to the interpreter stand-in and to the C emitter).
Skipped unless KLONG_HW_BUILD names a reachable machine. This is C simulation, not co-simulation and not hardware.
"""
import unittest

import numpy as np

from klongpy.hw.synth import synthesize
from klongpy.static import fixed as fx
from klongpy.static.compile import emit_hls_program

from .test_emit_hls import np_type, pack
from .test_hls_remote import REMOTE

remote_only = unittest.skipUnless(REMOTE, "set KLONG_HW_BUILD=user@host to run Vitis HLS on the build machine")

OPS = ("add", "sub", "mul", "max", "min", "div")
CMP = ("lt", "gt", "eq")


def reference(spec, a, b, division):
    ops = {"add": fx.add(a, b, spec), "sub": fx.sub(a, b, spec), "mul": fx.mul(a, b, spec), "max": fx.fmax(a, b), "min": fx.fmin(a, b)}
    rows = [ops[k] for k in OPS[:5]]
    if division:
        rows.append(fx.div(a, np.where(b == 0, 1, b), spec))
    return np.stack(rows, axis=1)


def program(t, division):
    items = "x+y;x-y;x*y;x|y;x&y" + (";x%y" if division else "")
    n = 6 if division else 5
    return (f'arith::{{[;{items}]}}\ncmp::{{[;x<y;x>y;x=y]}}\n'
            f'.entry("arith";"{t} {t} -> {t}[{n}]")\n.entry("cmp";"{t} {t} -> int[3]")\n')


def run(wrapped, w, i, a, b, division):
    t = f"fixed<{w},{i}>"
    res = emit_hls_program("fxops.kg", None, text=program(t, division), prefix="fxops", wrapped=wrapped)
    out = {}
    for e in res.entries:
        r = synthesize(res, REMOTE, timeout=1800, csim_data={"in.bin": pack(e, 32, [[x, y] for x, y in zip(a, b)])},
                       synth=False, only=e["c_name"])[e["c_name"]]
        assert r["result"].returncode == 0, r["log"][-3000:] + r["stdout"][-2000:]
        dt = np.dtype("<i4") if w <= 32 else np.dtype("<i8")
        out[e["c_name"]] = np.frombuffer(r["out"], dtype=dt if e["c_name"] == "arith" else np.dtype("<i4")) \
            .reshape(len(a), -1).astype(np.int64), int.from_bytes(r["status"], "little")
    return out


@remote_only
class TestFixedInHls(unittest.TestCase):
    def check(self, wrapped, w, i, a, b):
        spec = fx.Spec(w, i)
        division = w <= 32
        out = run(wrapped, w, i, a, b, division)
        arith, status = out["arith"]
        self.assertEqual(status, 0)
        bb = np.where(b == 0, 1, b) if division else b
        want = reference(spec, a, bb, division)
        self.assertTrue(np.array_equal(arith, want))
        cmp_, status = out["cmp"]
        self.assertEqual(status, 0)
        self.assertTrue(np.array_equal(cmp_, np.stack([fx.compare(k, a, bb) for k in CMP], axis=1)))

    def test_fixed_8_4_every_pair(self):
        v = np.arange(-128, 128, dtype=np.int64)
        a, b = (x.reshape(-1) for x in np.meshgrid(v, v, indexing="ij"))
        b = np.where(b == 0, 1, b)
        self.check(False, 8, 4, a, b)
        self.check(True, 8, 4, a, b)            # the wrapped kernel (stage A) too

    def test_fixed_32_16_random(self):
        rng = np.random.default_rng(1)
        a = rng.integers(-(1 << 31), 1 << 31, 3000, dtype=np.int64)
        b = rng.integers(-(1 << 31), 1 << 31, 3000, dtype=np.int64)
        a[:3], b[:3] = [-(1 << 31), (1 << 31) - 1, 0], [-1, -1, 7]
        self.check(False, 32, 16, a, np.where(b == 0, 1, b))

    def test_fixed_64_32_random(self):
        rng = np.random.default_rng(2)
        a = rng.integers(-(1 << 62), 1 << 62, 2000, dtype=np.int64) * 2
        b = rng.integers(-(1 << 62), 1 << 62, 2000, dtype=np.int64) * 2
        a[:3], b[:3] = [-(1 << 63), (1 << 63) - 1, 0], [-1, -1, 7]
        self.check(False, 64, 32, a, b)


if __name__ == "__main__":
    unittest.main()
