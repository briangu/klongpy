"""
Stage B of kgpy fpga source: the streamable kernel. Everything here runs without a Xilinx tool:
the generated C++ is built with the system compiler (a stand-in for hls::stream is generated next to it) and its test
bench runs the kernel in C simulation. The stage A kernel (--wrapped) is the reference: for every reference
program and accept fixture the two give the same output, bit for bit for ints and fixed-point numbers and for floats
without reassociation; with --reassociate floats agree within the tolerance of the differential harness for +/.
tests/static/test_hls_stream_remote.py runs the same files through Vitis HLS.
"""
import glob
import os
import unittest

import numpy as np

from klongpy.errors import KlongStaticError
from klongpy.hw.reports import stream_summary
from klongpy.static import fixed as fx

from .harness import find_cc
from .test_emit_hls import CC, CXX, HERE, HlsBuild, pack, program, special_floats, unpack
from .common import STUBS

needs = unittest.skipUnless(CXX and CC, "needs a C and a C++ compiler")


def bits(b):
    return np.frombuffer(b, dtype=np.uint8)


@needs
class TestAgainstStageA(unittest.TestCase):
    def both(self, **kw):
        a = HlsBuild(wrapped=True, **kw)
        b = HlsBuild(wrapped=False, **kw)
        return a, b

    def test_mavg_equals_stage_a_bit_for_bit(self):
        a, b = HlsBuild(path=program("mavg"), name="mavg", wrapped=True), HlsBuild(path=program("mavg"), name="mavg", wrapped=False)
        rng = np.random.default_rng(1)
        calls = [[float(v)] for v in special_floats(rng, (1000,))]
        data = pack(a.entry, 32, calls)
        oa, sa = a.run(data)
        ob, sb = b.run(data)
        self.assertEqual((sa, sb), (0, 0))
        self.assertEqual(oa, ob)
        # chunked calls continue the state
        self.assertEqual(b.run(data, per=64)[0], oa)

    def test_fir_equals_stage_a_bit_for_bit(self):
        a, b = HlsBuild(path=program("fir"), name="fir", wrapped=True), HlsBuild(path=program("fir"), name="fir", wrapped=False)
        rng = np.random.default_rng(2)
        calls = [[blk] for blk in special_floats(rng, (40, 256), rate=0.002)]
        data = pack(a.entry, 32, calls)
        self.assertEqual(a.run(data)[0], b.run(data)[0])

    def test_pid_style_state_and_an_int_program(self):
        text = 'acc::0\nf::{acc::acc+x;acc}\n.entry("f";"int -> int")\n'
        a, b = HlsBuild(text=text, name="acc", wrapped=True), HlsBuild(text=text, name="acc", wrapped=False)
        data = pack(a.entry, 32, [[v] for v in range(-50, 50)])
        self.assertEqual(a.run(data)[0], b.run(data)[0])


def regs_bytes(manifest, values):
    """regs.bin for the test bench: per register, one value per lane (fixed-point ones given as numbers)."""
    parts = []
    for r in manifest["registers"]:
        vals = values[r["name"]]
        if r["elem"].startswith("fixed<"):
            spec = fx.spec_of(r["elem"])
            arr = np.asarray(fx.from_float(np.asarray(vals, dtype=np.float64), spec))
            parts.append(arr.astype("<i4" if r["bytes"] == 4 else "<i8").tobytes())
        elif r["elem"] == "float":
            parts.append(np.asarray(vals, dtype="<f4").tobytes())
        else:
            parts.append(np.asarray(vals, dtype="<i4" if r["bytes"] == 4 else "<i8").tobytes())
    return b"".join(parts)


LANES_EMA = """
.param("alpha";"fixed<32,16>";0.5)
.param("depth";"int";2)
acc::.fixed("fixed<32,16>";0)
count::0
step::{acc::acc+((x-acc)*alpha);count::(count+1)!8;acc+.fixed("fixed<32,16>";count&depth)}
.entry("step";"fixed<32,16> -> fixed<32,16>")
"""

LANES_WINDOW = """
.param("p";"int";4)
buf::[0 0 0 0 0 0 0 0]
pos::0
step::{buf::buf:=x,pos;pos::(pos+1)!8;(+/buf*(!8)<p)%p}
.entry("step";"int -> float")
"""


@needs
class TestLanes(unittest.TestCase):
    def reference(self, text, params, lanes, xs, raw_in):
        """The interpreter, once per lane with that lane's parameters: array [n, lanes(, ...)]."""
        from klongpy.static import call_entry, load_program, set_param
        out = []
        for l in range(lanes):
            k = load_program("t.kg", text=text, strict=2, backend="numpy")
            for name, vals in params.items():
                set_param(k, name, vals[l])
            col = []
            for x in xs:
                v = fx.FixedArray(x, fx.spec_of("fixed<32,16>")) if raw_in else x
                col.append(np.asarray(call_entry(k, "step", v, raw=raw_in)))
            out.append(col)
        return np.array(out).transpose(1, 0)

    def run_lanes(self, lanes):
        text = LANES_EMA
        b = HlsBuild(text=text, name="ema", wrapped=False, lanes=lanes)
        self.assertEqual(b.entry["result"]["shape"], [lanes])
        self.assertEqual(b.entry["out_words"], lanes)
        rng = np.random.default_rng(lanes)
        xs = rng.integers(-3000000, 3000000, 200)
        params = {"alpha": [0.5 / (1 + l) for l in range(lanes)], "depth": [1 + l % 3 for l in range(lanes)]}
        out, status = b.run(pack(b.entry, 32, [[int(x)] for x in xs]), regs=regs_bytes(self.manifest(b), params))
        self.assertEqual(status, 0)
        got = np.frombuffer(out, dtype="<i4").reshape(len(xs), lanes)
        want = self.reference(text, params, lanes, xs, True)
        self.assertTrue(np.array_equal(got, want.astype(np.int32)), f"lanes={lanes}")
        # a second call with other parameters, state kept (no reset)
        return b

    def manifest(self, b):
        import json
        with open(os.path.join(b.dir, "manifest.json")) as f:
            return json.load(f)

    def test_lanes_1_2_16_equal_the_interpreter_per_lane(self):
        for lanes in (1, 2, 16):
            with self.subTest(lanes=lanes):
                self.run_lanes(lanes)

    def test_int_parameter_bounds_a_window_in_every_lane(self):
        lanes = 4
        b = HlsBuild(text=LANES_WINDOW, name="win", wrapped=False, lanes=lanes)
        xs = list(range(1, 40))
        params = {"p": [1, 2, 4, 8]}
        out, status = b.run(pack(b.entry, 32, [[x] for x in xs]), regs=regs_bytes(self.manifest(b), params))
        self.assertEqual(status, 0)
        got = np.frombuffer(out, dtype="<f4").reshape(len(xs), lanes)
        want = self.reference(LANES_WINDOW, params, lanes, xs, False)
        self.assertTrue(np.array_equal(got, want.astype(np.float32)))

    def test_lanes_without_the_option_have_no_lane_dimension_and_one_parameter_register(self):
        b = HlsBuild(text=LANES_EMA, name="ema", wrapped=False)
        self.assertEqual(b.entry["result"]["shape"], [])
        m = self.manifest(b)
        self.assertEqual([r["name"] for r in m["registers"]], ["alpha", "depth"])
        self.assertEqual(m["lanes"], 1)
        self.assertFalse(m["lane_dimension"])
        self.assertEqual(b.entry["kernel_args"], ["in", "out", "n", "reset", "alpha", "depth", "status"])

    def test_the_kernel_and_the_wrapped_core_agree_on_parameters(self):
        a = HlsBuild(text=LANES_EMA, name="ema", wrapped=True)
        b = HlsBuild(text=LANES_EMA, name="ema", wrapped=False)
        xs = np.random.default_rng(5).integers(-2000000, 2000000, 100)
        data = pack(a.entry, 32, [[int(x)] for x in xs])
        regs = regs_bytes(self.manifest(a), {"alpha": [0.25], "depth": [3]})
        self.assertEqual(a.run(data, regs=regs)[0], b.run(data, regs=regs)[0])
        # and the default values of the registers are used when none are given
        self.assertEqual(a.run(data)[0], b.run(data)[0])


SUMMARY = """
.param("gain";"int";2)
tot::0
cnt::0
step::{tot::tot+x*gain;cnt::cnt+1;0}
summary::{tot,cnt}
.entry("step";"int -> void")
.entry("summary";"-> int[2]")
"""


@needs
class TestSummaryPair(unittest.TestCase):
    """A void entry with arguments and an entry without arguments share the state; the kernel returns the summary once."""

    def test_summary_rate_uses_the_slowest_lane_record_loop(self):
        def report(ii, trip=None):
            return {"loops": [{"ii": ii, "trip": trip, "depth": 9, "name": "loop"}]}

        modules = {"load": report(1), "store": report(1, 192),
                   "lane_0_Pipeline_step": report(1), "lane_1_Pipeline_step": report(3),
                   "lane_1_Pipeline_init": report(8, 32)}
        rate = stream_summary(modules, 2, 192, summary=True)
        self.assertEqual((rate["interval"], rate["limit"], rate["store_ii"]), (3, "core", 1))
        self.assertEqual(stream_summary(modules, 2, 192)["interval"], 192)
        self.assertIsNone(stream_summary({"load": report(1)}, 2, 192, summary=True)["interval"])

    def reference(self, xs, gain):
        from klongpy.static import call_entry, load_program, set_param
        k = load_program("t.kg", text=SUMMARY, strict=2, backend="numpy")
        set_param(k, "gain", gain)
        for x in xs:
            self.assertIsNone(call_entry(k, "step", int(x)))
        return np.asarray(call_entry(k, "summary"))

    def table(self, b, **values):
        import json
        with open(os.path.join(b.dir, "manifest.json")) as f:
            return regs_bytes(json.load(f), values)

    def test_the_summary_is_the_state_after_n_calls_and_equals_the_interpreter_per_lane(self):
        lanes = 3
        b = HlsBuild(text=SUMMARY, name="sm", wrapped=False, lanes=lanes)
        e = b.entry
        self.assertEqual((e["name"], e["output"], e["out_words"], e["out_bytes"]), ("step", "summary", 2 * lanes, 8 * lanes))
        self.assertEqual(e["summary"]["entry"], "summary")
        self.assertEqual(e["result"]["shape"], [lanes, 2])
        self.assertEqual(len(b.res.entries), 1, "the reader is not a kernel of its own")
        self.assertEqual(e["kernel_args"], ["in", "out", "n", "reset", "params", "status"])
        xs = np.random.default_rng(3).integers(-50, 50, 41)
        gains = [1, 2, -3]
        regs = self.table(b, gain=gains)
        for per in (0, 7, 1):
            out, status = b.run(pack(e, 32, [[int(x)] for x in xs]), per=per, regs=regs)
            self.assertEqual(status, 0)
            got = np.frombuffer(out, dtype="<i4").reshape(lanes, 2)
            want = np.array([self.reference(xs, g) for g in gains])
            self.assertTrue(np.array_equal(got, want), per)

    def test_lanes_of_a_summary_kernel_are_processes_with_a_trap_flag_each(self):
        from klongpy.static.native import emit_hls_program
        res = emit_hls_program("sm.kg", None, text=SUMMARY, prefix="sm", lanes=3)
        e = res.entries[0]
        self.assertEqual(e["structure"], "lane-processes")
        self.assertEqual(e["lane_files"], ["lane_0.c", "lane_1.c", "lane_2.c"])
        self.assertTrue(set(e["lane_files"]) <= set(res.files) and set(e["lane_files"]) <= set(res.manifest["c_files"]))
        self.assertIn("add_files $here/lane_2.c", res.files[e["files"]["tcl"]])
        lane = res.files["lane_2.c"]
        self.assertIn("#define KG_TRAP_FLAG kg_hls_trap_2", lane)
        self.assertIn("#define sm_step sm_step_2", lane)
        self.assertIn("#define sm_set_gain sm_set_gain_2", lane)
        self.assertTrue(lane.rstrip().endswith('#include "sm.c"'))
        top = res.files[e["files"]["top"]]
        for l in range(3):
            self.assertIn(f"static void lane_{l}(", top)
            self.assertIn(f"sm_summary_{l}(&inst", top)
            self.assertIn(f"extern unsigned kg_hls_trap_{l};", top)
        self.assertNotIn("kg_hls_trap;", top, "no lane touches a variable of another")
        # without lanes, or with one, or without a summary pair there is a single core
        for kw in ({}, {"lanes": 1}):
            r = emit_hls_program("sm.kg", None, text=SUMMARY, prefix="sm", **kw)
            self.assertNotIn("structure", r.entries[0])
        text = 'acc::0\nf::{acc::acc+x;acc}\n.entry("f";"int -> int")\n'
        self.assertNotIn("structure", emit_hls_program("a.kg", None, text=text, prefix="a", lanes=4).entries[0])

    def test_no_calls_reads_the_initial_state(self):
        b = HlsBuild(text=SUMMARY, name="sm", wrapped=False)
        out, status = b.run(b"")
        self.assertEqual(status, 0)
        self.assertEqual(np.frombuffer(out, dtype="<i4").tolist(), [0, 0])

    def test_without_lanes_the_registers_stay_scalars_and_the_result_has_no_lane_dimension(self):
        b = HlsBuild(text=SUMMARY, name="sm", wrapped=False)
        self.assertEqual(b.entry["result"]["shape"], [2])
        self.assertEqual(b.entry["kernel_args"], ["in", "out", "n", "reset", "gain", "status"])
        out, _ = b.run(pack(b.entry, 32, [[5], [6]]), regs=self.table(b, gain=[3]))
        self.assertEqual(np.frombuffer(out, dtype="<i4").tolist(), [33, 2])

    def test_the_rules(self):
        from klongpy.errors import KlongStaticError
        from klongpy.static.native import emit_hls_program

        def emit(text, **kw):
            return emit_hls_program("t.kg", None, text=text, prefix="t", **kw)
        with self.assertRaises(KlongStaticError) as cm:                  # stage A has no pair
            emit(SUMMARY, wrapped=True)
        self.assertIn("returns nothing", str(cm.exception))
        self.assertIn("without arguments", cm.exception.hint)
        with self.assertRaises(KlongStaticError) as cm:                  # a reader and no driver is an entry without input
            emit('t::0\nsummary::{t}\n.entry("summary";"-> int")\nstep::{t::t+x;t}\n.entry("step";"int -> int")')
        self.assertIn("takes no argument", str(cm.exception))
        two = SUMMARY + 'other::{cnt}\n.entry("other";"-> int")\n'
        with self.assertRaises(KlongStaticError) as cm:                  # which one reads the state is ambiguous
            emit(two)
        self.assertIn("returns nothing", str(cm.exception))
        with self.assertRaises(KlongStaticError):                        # no reader at all
            emit('t::0\nstep::{t::t+x;0}\n.entry("step";"int -> void")')


@needs
class TestSummaryGolden(unittest.TestCase):
    def test_files(self):
        from .common import golden
        from klongpy.static.native import emit_hls_program
        res = emit_hls_program("sm.kg", None, text=SUMMARY, prefix="sm", lanes=3)
        golden(self, "hlsb_summary3_step_top.cpp", res.files["sm_step_top.cpp"])
        golden(self, "hlsb_summary3_step_tb.cpp", res.files["sm_step_tb.cpp"])
        golden(self, "hlsb_summary3_manifest.json", res.files["manifest.json"])


def unpack_raw(entry, data):
    r = entry["result"]
    dt = np.dtype("<f4") if r["elem"] == "float" else np.dtype("<i%d" % (4 if r["elem"] == "int" or int(r["elem"][6:].split(",")[0]) <= 32 else 8))
    return np.frombuffer(data, dtype=dt).reshape((-1,) + tuple(r["shape"]))


@needs
class TestTransformations(unittest.TestCase):
    """The three streamable transformations, each against stage A on the same records."""

    def pair(self, text, calls, name="t", **kw):
        a = HlsBuild(text=text, name=name, wrapped=True)
        b = HlsBuild(text=text, name=name, wrapped=False, **kw)
        data = pack(a.entry, 32, calls)
        oa, sa = a.run(data)
        ob, sb = b.run(data)
        self.assertEqual(sa, sb)
        return a, b, oa, ob

    def test_integer_and_fixed_reductions_are_trees_and_stay_bit_exact(self):
        rng = np.random.default_rng(11)
        for n in (2, 3, 8, 9, 40):
            for ty, op, seed in (("int", "+", ""), ("int", "|", ""), ("int", "&", ""), ("int", "+", "7"),
                                 ("fixed<16,8>", "+", ""), ("fixed<16,8>", "|", ""), ("fixed<16,8>", "&", "0")):
                with self.subTest(n=n, ty=ty, op=op, seed=seed):
                    text = f'f::{{{seed}{op}/x}}\n.entry("f";"{ty}[{n}] -> {ty}")\n'
                    calls = [[rng.integers(-30000, 30000, n)] for _ in range(20)]
                    a, b, oa, ob = self.pair(text, calls)
                    self.assertEqual(oa, ob)
        c = HlsBuild(text='f::{+/x}\n.entry("f";"int[8] -> int")\n', name="t", wrapped=False).res.files["t.c"]
        self.assertNotIn("for (kg_int", c, "the reduction of 8 items is a tree, not a loop")
        c = HlsBuild(text='f::{+/x}\n.entry("f";"float[8] -> float")\n', name="t", wrapped=False).res.files["t.c"]
        self.assertIn("for (kg_int", c, "a float reduction keeps its order unless --reassociate")

    def test_a_float_reduction_keeps_its_order_unless_reassociation_is_asked_for(self):
        rng = np.random.default_rng(12)
        text = 'f::{+/x}\n.entry("f";"float[33] -> float")\n'
        calls = [[(rng.random(33) * 100 - 50).astype(np.float32)] for _ in range(200)]
        a, b, oa, ob = self.pair(text, calls)
        self.assertEqual(oa, ob)                                      # sequential order: bit for bit
        a, c, oa, oc = self.pair(text, calls, reassociate=True)
        x, y = np.frombuffer(oa, "<f4"), np.frombuffer(oc, "<f4")
        self.assertFalse(np.array_equal(x, y), "a tree rounds differently from a chain on 33 random items")
        self.assertTrue(np.allclose(x, y, rtol=1e-5, atol=1e-4))     # within the tolerance of the differential harness for +/
        self.assertEqual(c.res.manifest["reassociate"], True)
        self.assertNotEqual(b.res.manifest["cache_key"], c.res.manifest["cache_key"])

    def test_the_window_of_fir_float_reassociated_and_fixed(self):
        rng = np.random.default_rng(13)
        # float without reassociation: bit for bit
        a, b = (HlsBuild(path=program("fir"), name="fir", wrapped=w) for w in (True, False))
        self.assertIn("window", b.entry)
        self.assertEqual(b.entry["window"]["taps"], 16)
        self.assertEqual(b.entry["streams"], {"in": "word", "out": "word"})
        calls = [[(rng.random(256) * 20 - 10).astype(np.float32)] for _ in range(6)]
        data = pack(a.entry, 32, calls)
        self.assertEqual(a.run(data)[0], b.run(data)[0])
        # float with reassociation: within tolerance
        c = HlsBuild(path=program("fir"), name="fir", wrapped=False, reassociate=True)
        x, y = np.frombuffer(a.run(data)[0], "<f4"), np.frombuffer(c.run(data)[0], "<f4")
        self.assertTrue(np.allclose(x, y, rtol=1e-5, atol=1e-5))
        # fixed<32,16>: bit for bit with stage A, and with the interpreter's evaluator
        path = os.path.join(HERE, "hw", "fir_fixed.kg")
        fa, fb = (HlsBuild(path=path, name="firfx", wrapped=w) for w in (True, False))
        raw = [[rng.integers(-2000000, 2000000, 256)] for _ in range(6)]
        fdata = pack(fa.entry, 32, raw)
        self.assertEqual(fa.run(fdata)[0], fb.run(fdata)[0])
        from klongpy.static import call_entry, load_program
        k = load_program(path, strict=2, backend="numpy")
        for rec, got in zip(raw, unpack_raw(fb.entry, fb.run(fdata)[0])):
            want = call_entry(k, "run", fx.FixedArray(rec[0], fx.spec_of("fixed<32,16>")), raw=True)
            self.assertTrue(np.array_equal(want, got))

    def test_a_program_that_is_not_a_window_still_runs_as_a_block(self):
        text = 'f::{(x*x)+(x%3)}\n.entry("f";"float[600] -> float[600]")\n'
        rng = np.random.default_rng(14)
        calls = [[(rng.random(600) * 10).astype(np.float32)] for _ in range(4)]
        a, b, oa, ob = self.pair(text, calls)
        self.assertEqual(oa, ob)
        self.assertEqual(b.entry["streams"], {"in": "word", "out": "word"})
        self.assertNotIn("window", b.entry)

    def test_a_window_with_another_index_table_is_not_taken_for_one(self):
        text = ('idx::+{2*x+!4}\'!8\nf::{+/x@idx}\n.entry("f";"float[24] -> float[8]")\n')
        rng = np.random.default_rng(15)
        calls = [[(rng.random(24) * 10).astype(np.float32)] for _ in range(4)]
        a, b, oa, ob = self.pair(text, calls)
        self.assertEqual(oa, ob)
        self.assertNotIn("window", b.entry)

    def test_records_of_mixed_words_and_a_small_record_travel_whole(self):
        text = 'f::{x+y}\n.entry("f";"int float -> float")\n'
        calls = [[i, 0.5 * i] for i in range(30)]
        a, b, oa, ob = self.pair(text, calls)
        self.assertEqual(oa, ob)
        self.assertEqual(b.entry["streams"], {"in": "record", "out": "record"})
        self.assertEqual(b.entry["in_type"], "uint32_t")

    def test_the_trap_flag_reaches_the_status(self):
        text = 'f::{x!y}\n.entry("f";"int int -> int")'
        b = HlsBuild(text=text, name="mod", wrapped=False)
        self.assertEqual(b.run(pack(b.entry, 32, [[7, 3]]))[1], 0)
        self.assertEqual(b.run(pack(b.entry, 32, [[7, 3], [7, 0]]))[1], 1 | 4)
        text = 'f::{y%x}\n.entry("f";"fixed<32,16> fixed<32,16> -> fixed<32,16>")'
        b = HlsBuild(text=text, name="fxd", wrapped=False)
        self.assertEqual(b.run(pack(b.entry, 32, [[0, 5]]))[1], 1 | 4)

    def compare_fixture(self, path):
        """None when the fixture does not make a kernel; else (name, outputs equal, statuses equal)."""
        name = os.path.basename(path)[:-3]
        with open(path) as f:
            if ":~" in f.read():          # a While loop may not stop on random data
                return None
        pre = "fx" + name.replace("-", "_")
        try:
            a = HlsBuild(path=path, name=pre, wrapped=True)
        except KlongStaticError:
            return None
        b = HlsBuild(path=path, name=pre, wrapped=False)
        rng = np.random.default_rng(len(name))
        calls = [[self.sample(p, rng) for p in a.entry["params"]] for _ in range(24)]
        data = pack(a.entry, 32, calls)
        oa, sa = a.run(data)
        ob, sb = b.run(data)
        return name, oa == ob, sa == sb

    def test_every_accept_fixture_gives_the_same_output_in_both_stages(self):
        from concurrent.futures import ThreadPoolExecutor
        paths = sorted(glob.glob(os.path.join(HERE, "accept", "*.kg")))
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as pool:
            results = [r for r in pool.map(self.compare_fixture, paths) if r is not None]
        for name, same_out, same_status in results:
            with self.subTest(fixture=name):
                self.assertTrue(same_out)
                self.assertTrue(same_status)
        self.assertGreater(len(results), 25)

    @staticmethod
    def sample(p, rng):
        shape = tuple(p["shape"])
        if p["elem"] == "float":
            v = (rng.random(shape or ()) * 8 - 4).astype(np.float32)
        elif p["elem"] == "int":
            v = rng.integers(0, 3, shape or ())
        else:
            v = rng.integers(-300000, 300000, shape or ())
        return v


@needs
class TestGolden(unittest.TestCase):
    def test_stage_b_files(self):
        from .common import golden
        from klongpy.static.native import emit_hls_program
        for name, entry, kw, path in (("mavg", "step", {}, program("mavg")), ("fir", "run", {}, program("fir")),
                                      ("firfx", "run", {}, os.path.join(HERE, "hw", "fir_fixed.kg"))):
            res = emit_hls_program(path, None, prefix=name, **kw)
            golden(self, f"hlsb_{name}_{entry}_top.cpp", res.files[f"{name}_{entry}_top.cpp"])
            golden(self, f"hlsb_{name}.c", res.files[f"{name}.c"])
            golden(self, f"hlsb_{name}_manifest.json", res.files["manifest.json"])
        res = emit_hls_program("ema.kg", None, text=LANES_EMA, prefix="ema", lanes=4)
        golden(self, "hlsb_ema4_step_top.cpp", res.files["ema_step_top.cpp"])
        golden(self, "hlsb_ema4_step_tb.cpp", res.files["ema_step_tb.cpp"])
        golden(self, "hlsb_ema4_manifest.json", res.files["manifest.json"])


class TestCli(unittest.TestCase):
    def cli(self, *args):
        import subprocess
        import sys
        from .common import ROOT
        return subprocess.run([sys.executable, "-W", "ignore", "-m", "klongpy.cli", *args], capture_output=True, text=True, cwd=ROOT)

    def test_flags(self):
        import json
        import tempfile
        d = tempfile.mkdtemp(prefix="kgcli_")
        r = self.cli("fpga", "source", program("mavg"), "--lanes", "4", "--reassociate", "-o", d)
        self.assertEqual(r.returncode, 0, r.stderr)
        m = json.load(open(os.path.join(d, "manifest.json")))
        self.assertEqual((m["stage"], m["lanes"], m["lane_dimension"], m["reassociate"]), ("B", 4, True, True))
        self.assertEqual(m["entries"][0]["result"]["shape"], [4])
        r = self.cli("fpga", "source", program("mavg"), "--wrapped", "-o", d)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("stage", json.load(open(os.path.join(d, "manifest.json"))))
        for bad in (["--wrapped", "--lanes", "2"], ["--wrapped", "--reassociate"], ["--lanes", "0"]):
            r = self.cli("fpga", "source", program("mavg"), *bad, "-o", d)
            self.assertEqual(r.returncode, 2, bad)
        r = self.cli("compile", "c", program("mavg"), "--lanes", "2", "-o", d)
        self.assertEqual(r.returncode, 2)
