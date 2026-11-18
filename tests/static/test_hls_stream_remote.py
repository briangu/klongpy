"""
Stage B on the build machine: C simulation of the streamable kernels in Vitis HLS against stage A and
the interpreter, and C synthesis estimates. Skipped unless KLONG_FPGA_BUILD names a reachable machine:

    KLONG_FPGA_BUILD=user@fpga-host python -m pytest tests/static/test_hls_stream_remote.py

The numbers these tests look at are C simulation results (bit equality) and C synthesis estimates (the initiation interval of
the pipelined loops, the area). They are not co-simulation and not hardware. No Vivado or v++ run is started.
"""
import os
import unittest

import numpy as np

from klongpy.static import call_entry, fixed as fx, load_program
from klongpy.static.compile import emit_hls_program
from klongpy.hw.reports import stream_summary
from klongpy.hw.synth import synthesize

from .test_emit_hls import HERE, pack, program, special_floats
from .test_hls_remote import REMOTE
from .test_hls_stream import LANES_EMA, regs_bytes, unpack_raw

remote_only = unittest.skipUnless(REMOTE, "set KLONG_FPGA_BUILD=user@host to run Vitis HLS on the build machine")


def csim(res, entry, calls, regs=None, synth=False, chunk=0):
    data = {"in.bin": pack(entry, 32, calls)}
    if regs is not None:
        data["regs.bin"] = regs
    r = synthesize(res, REMOTE, timeout=1800, csim_data=data, synth=synth, chunk=chunk, only=entry["c_name"])[entry["c_name"]]
    assert r["result"].returncode == 0, r["log"][-3000:] + r["stdout"][-2000:]
    return r


@remote_only
class TestStreamableKernelsInVitisHls(unittest.TestCase):
    runs = {}

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(113)
        cls.mavg = [[float(v)] for v in special_floats(rng, (1000,))]
        cls.fir = [[blk] for blk in special_floats(rng, (20, 256), rate=0.002)]
        cls.fir_clean = [[(rng.random(256) * 20 - 10).astype(np.float32)] for _ in range(20)]     # no inf or nan: a tree may differ there
        fixed = os.path.join(HERE, "hw", "fir_fixed.kg")
        cls.fixed_path = fixed
        cls.fixed_calls = [[rng.integers(-2000000, 2000000, 256)] for _ in range(20)]
        for key, path, calls, kw in (("mavg", program("mavg"), cls.mavg, {}), ("fir", program("fir"), cls.fir, {}),
                                     ("fir_reassoc", program("fir"), cls.fir_clean, {"reassociate": True}),
                                     ("fir_fixed", fixed, cls.fixed_calls, {})):
            res = emit_hls_program(path, None, **kw)
            cls.runs[key] = (res, res.entries[0], csim(res, res.entries[0], calls, synth=True))

    def stage_a(self, path, calls):
        res = emit_hls_program(path, None, wrapped=True)
        e = res.entries[0]
        return e, csim(res, e, calls)["out"]

    def test_mavg_streamable_csim_equals_stage_a_bit_for_bit(self):
        res, e, r = self.runs["mavg"]
        self.assertEqual(int.from_bytes(r["status"], "little"), 0)
        self.assertEqual(r["out"], self.stage_a(program("mavg"), self.mavg)[1])

    def test_fir_streamable_csim_equals_stage_a_bit_for_bit(self):
        res, e, r = self.runs["fir"]
        self.assertEqual(r["out"], self.stage_a(program("fir"), self.fir)[1])

    def test_fir_with_reassociation_stays_within_the_tolerance(self):
        res, e, r = self.runs["fir_reassoc"]
        want = np.frombuffer(self.stage_a(program("fir"), self.fir_clean)[1], "<f4")
        got = np.frombuffer(r["out"], "<f4")
        self.assertTrue(np.allclose(want, got, rtol=1e-5, atol=1e-5))

    def test_fixed_fir_csim_equals_the_interpreters_scaled_integers(self):
        res, e, r = self.runs["fir_fixed"]
        k = load_program(self.fixed_path, strict=2, backend="numpy")
        got = unpack_raw(e, r["out"])
        for rec, g in zip(self.fixed_calls, got):
            want = call_entry(k, "run", fx.FixedArray(rec[0], fx.spec_of("fixed<32,16>")), raw=True)
            self.assertTrue(np.array_equal(want, g))

    def test_mavg_reaches_one_record_per_cycle(self):
        res, e, r = self.runs["mavg"]
        sm = stream_summary(r["modules"], e["in_words"], e["out_words"])
        self.assertEqual(sm["core_ii"], 1)
        self.assertEqual(sm["interval"], 1)

    def test_fir_takes_one_input_word_per_cycle_in_every_variant(self):
        for key in ("fir", "fir_reassoc", "fir_fixed"):
            res, e, r = self.runs[key]
            with self.subTest(variant=key):
                sm = stream_summary(r["modules"], e["in_words"], e["out_words"])
                self.assertEqual(sm["core_ii"], 1)
                self.assertEqual(sm["interval"], 256)       # a record is 256 input words
        self.assertLess(self.runs["fir_reassoc"][2]["modules"]["core_Pipeline_VITIS_LOOP_52_1"]["loops"][0]["depth"],
                        self.runs["fir"][2]["modules"]["core_Pipeline_VITIS_LOOP_52_1"]["loops"][0]["depth"])

    def test_a_float_state_through_a_float_add_does_not_reach_one_per_cycle(self):
        text = "integ::0.0\nstep::{integ::integ+(x*0.5);integ}\n.entry(\"step\";\"float -> float\")\n"
        res = emit_hls_program("integ.kg", None, text=text)
        e = res.entries[0]
        r = csim(res, e, [[float(i)] for i in range(50)], synth=True)
        sm = stream_summary(r["modules"], e["in_words"], e["out_words"])
        self.assertGreater(sm["core_ii"], 1)
        from klongpy.hw.reports import ii_warnings
        self.assertTrue(ii_warnings(r["log"]), "the tool says why")

    def test_lanes_csim_equals_the_interpreter_per_lane(self):
        for lanes in (1, 2, 16):
            with self.subTest(lanes=lanes):
                res = emit_hls_program("ema.kg", None, text=LANES_EMA, lanes=lanes, prefix="ema")
                e = res.entries[0]
                rng = np.random.default_rng(lanes)
                xs = rng.integers(-3000000, 3000000, 100)
                params = {"alpha": [0.5 / (1 + i) for i in range(lanes)], "depth": [1 + i % 3 for i in range(lanes)]}
                r = csim(res, e, [[int(x)] for x in xs], regs=regs_bytes(res.manifest, params))
                got = np.frombuffer(r["out"], "<i4").reshape(len(xs), lanes)
                for l in range(lanes):
                    k = load_program("t.kg", text=LANES_EMA, strict=2, backend="numpy")
                    from klongpy.static import set_param
                    for name, vals in params.items():
                        set_param(k, name, vals[l])
                    want = [int(call_entry(k, "step", fx.FixedArray(int(x), fx.spec_of("fixed<32,16>")), raw=True)) for x in xs]
                    self.assertEqual(want, got[:, l].tolist(), f"lane {l} of {lanes}")

    def test_sixteen_lanes_synthesize_and_keep_one_record_per_cycle_in_the_core(self):
        res = emit_hls_program("ema.kg", None, text=LANES_EMA, lanes=16, prefix="ema")
        e = res.entries[0]
        r = synthesize(res, REMOTE, timeout=1800, only=e["c_name"])[e["c_name"]]
        self.assertEqual(r["result"].returncode, 0, r["log"][-2000:])
        sm = stream_summary(r["modules"], e["in_words"], e["out_words"])
        self.assertEqual(sm["core_ii"], 1)
        self.assertEqual(sm["store_ii"], 1)
        self.assertEqual(sm["interval"], 16)        # 16 result words per record leave through one word per cycle


if __name__ == "__main__":
    unittest.main()
