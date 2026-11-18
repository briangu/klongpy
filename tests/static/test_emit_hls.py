"""
kgpy fpga source: the files that Vitis HLS turns into a kernel. Everything here runs without a
Xilinx tool: the generated C++ is compiled with the system compiler (the pragmas are ``#pragma HLS``, which other
compilers ignore, and the top function needs no Xilinx header), and the test bench built from it runs the
program, whose outputs must equal the interpreter's. tests/static/test_hls_remote.py runs the same files in
Vitis HLS on the build machine.

Golden files: tests/static/golden/hls_*  (KG_UPDATE_GOLDEN=1 rewrites them).
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from klongpy.errors import KlongStaticError
from klongpy.static.compile import emit_hls_program
from klongpy.static.emit_hls import PART, reachable_externs
from klongpy.static.signature import Ty

from .harness import Host, find_cc, same
from .test_ir import EMBEDDED, ROOT, STUBS, golden

HERE = os.path.dirname(os.path.abspath(__file__))
CXX = shutil.which(os.environ.get("KG_TEST_CXX", "")) if os.environ.get("KG_TEST_CXX") else (
    shutil.which("clang++") or shutil.which("g++") or shutil.which("c++"))
CC = find_cc()
_KEEP = []


def program(name):
    return os.path.join(EMBEDDED, name + ".kg")


def emit(name_or_path, **kw):
    path = name_or_path if os.sep in name_or_path or name_or_path.endswith(".kg") else program(name_or_path)
    kw.setdefault("wrapped", True)         # these tests are about stage A (kgpy fpga source --wrapped); stage B is in test_hls_stream.py
    return emit_hls_program(path, None, **kw)


class HlsBuild:
    """A program emitted for HLS, with the test bench built by the system compiler: run(records) -> (out, status)."""

    def __init__(self, text=None, path=None, name="t", int_bits=32, wrapped=True, **kw):
        self.dir = tempfile.mkdtemp(prefix="kghls_")
        _KEEP.append(self.dir)
        self.res = emit_hls_program(path or name + ".kg", self.dir, text=text, int_bits=int_bits, prefix=name, externs=STUBS,
                                    wrapped=wrapped, **kw)
        self.prefix = self.res.prefix
        self.entry = self.res.entries[0]
        self.regs = self.res.manifest.get("registers", [])
        flags = ["-DKG_TRAP_FLAG=kg_hls_trap", "-I" + self.dir] + (["-DKG_INT64"] if int_bits == 64 else [])
        P = self.prefix
        e = self.entry["files"]
        cmds = [
            [CC, "-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic", "-Wno-unknown-pragmas", "-O1", "-ffp-contract=off", "-c",
             os.path.join(self.dir, P + ".c"), "-o", os.path.join(self.dir, "core.o")] + flags,
            [CXX, "-std=c++14", "-Wall", "-Wextra", "-Werror", "-Wno-unknown-pragmas", "-O1", "-ffp-contract=off", "-c",
             os.path.join(self.dir, e["top"]), "-o", os.path.join(self.dir, "top.o")] + flags,
            [CXX, "-std=c++14", "-Wall", "-Wextra", "-Werror", "-O1", "-c", os.path.join(self.dir, e["tb"]), "-o",
             os.path.join(self.dir, "tb.o")] + flags,
            [CXX] + [os.path.join(self.dir, f) for f in ("core.o", "top.o", "tb.o")] + ["-lm", "-o", os.path.join(self.dir, "tb")],
        ]
        # a kernel with one process per lane has the program's C once more per lane (lane_<l>.c)
        lane_objs = []
        for lf in self.entry.get("lane_files", ()):
            o = os.path.join(self.dir, lf[:-2] + ".o")
            cmds.insert(0, [CC, "-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic", "-Wno-unknown-pragmas", "-O1",
                            "-ffp-contract=off", "-c", os.path.join(self.dir, lf), "-o", o] + flags)
            lane_objs.append(o)
        cmds[-1] = cmds[-1][:1] + lane_objs + cmds[-1][1:]
        for cmd in cmds:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise AssertionError(f"{' '.join(cmd)}\n{r.stderr}")

    def run(self, records, per=0, regs=None):
        """records: bytes of in.bin; regs: bytes of regs.bin (the parameters). Returns (out.bin bytes, status word)."""
        d = os.path.join(self.dir, "data")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "in.bin"), "wb") as f:
            f.write(records)
        if regs is not None:
            with open(os.path.join(d, "regs.bin"), "wb") as f:
                f.write(regs)
        r = subprocess.run([os.path.join(self.dir, "tb"), d, str(per)], capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError(f"tb failed: {r.stderr}")
        with open(os.path.join(d, "out.bin"), "rb") as f:
            out = f.read()
        with open(os.path.join(d, "status.bin"), "rb") as f:
            status = int.from_bytes(f.read(), "little")
        return out, status


def np_type(elem, int_bits=32):
    if elem.startswith("fixed<"):
        return np.dtype("<i4" if int(elem[6:].split(",")[0]) <= 32 else "<i8")
    return np.dtype("<f4") if elem == "float" else np.dtype("<i4" if int_bits == 32 else "<i8")


def pack(entry, int_bits, calls):
    """The bytes of in.bin for a list of calls, each a list of arguments (the record layout of the manifest)."""
    mixed = len({p["elem"] for p in entry["params"]}) > 1
    width = np_type(entry["params"][0]["elem"], int_bits).itemsize
    parts = []
    for args in calls:
        for p, a in zip(entry["params"], args):
            a = np.asarray(a).reshape(-1)
            if mixed:
                parts.append(a.astype(np_type(p["elem"], int_bits) if width == 8 else np_type(p["elem"])).tobytes())
            else:
                parts.append(a.astype(np_type(p["elem"], int_bits)).tobytes())
    return b"".join(parts)


def unpack(entry, int_bits, data):
    r = entry["result"]
    arr = np.frombuffer(data, dtype=np_type(r["elem"], int_bits))
    n = len(arr) // r["words"] if r["words"] else 0
    return [arr[i * r["words"]:(i + 1) * r["words"]].reshape(r["shape"] or ()) for i in range(n)]


def compare_with_interpreter(test, build, host, calls, name, int_bits=32, per=0, rtol=None):
    """Run the bench over the calls and the interpreter over the same ones; every result must match."""
    entry = build.entry
    out, status = build.run(pack(entry, int_bits, calls), per=per)
    test.assertEqual(status, 0)
    got = unpack(entry, int_bits, out)
    test.assertEqual(len(got), len(calls))
    r = entry["result"]
    ty = Ty(r["elem"], tuple(r["shape"]))
    for i, (args, g) in enumerate(zip(calls, got)):
        kind, want = host.call(name, *args)
        test.assertEqual(kind, "ok", f"call {i}: {want}")
        d = same(want, g if r["shape"] else g.item(), ty, 32, rtol=rtol)
        test.assertIsNone(d, f"call {i} of {name}: {d}")


def special_floats(rng, shape, scale=100.0, rate=0.02):
    x = ((rng.random(shape) * 2 - 1) * scale).astype(np.float32)
    flat = x.reshape(-1)
    k = max(1, int(flat.size * rate))
    idx = rng.choice(flat.size, size=k, replace=False)
    flat[idx] = rng.choice(np.array([np.nan, np.inf, -np.inf], dtype=np.float32), size=k)
    return x


class TestFiles(unittest.TestCase):
    def test_golden_files(self):
        for name, entry in (("mavg", "step"), ("fir", "run")):
            res = emit(name)
            golden(self, f"hls_{name}_{entry}_top.cpp", res.files[f"{name}_{entry}_top.cpp"])
            golden(self, f"hls_{name}_{entry}_tb.cpp", res.files[f"{name}_{entry}_tb.cpp"])
            golden(self, f"hls_{name}_{entry}.tcl", res.files[f"hls_{entry}.tcl"])
            golden(self, f"hls_{name}_manifest.json", res.files["manifest.json"])

    def test_the_files_of_an_emission(self):
        res = emit("fir")
        self.assertEqual(sorted(res.files), ["fir.c", "fir.h", "fir_run_tb.cpp", "fir_run_top.cpp", "hls_run.tcl",
                                             "klong_rt.h", "manifest.json"])
        with open(os.path.join(ROOT, "klongpy", "static", "klong_rt.h")) as f:
            self.assertEqual(res.files["klong_rt.h"], f.read())

    def test_the_c_core_is_the_unchanged_output_of_the_c_emitter(self):
        from klongpy.static.compile import emit_program
        c = emit_program(program("mavg"), None)
        res = emit("mavg")
        self.assertEqual(res.files["mavg.c"], c.c)
        self.assertEqual(res.files["mavg.h"], c.h)

    def test_manifest(self):
        m = json.loads(emit("mavg").files["manifest.json"])
        self.assertEqual((m["program"], m["int_bits"], m["float_bits"], m["lanes"], m["clock_mhz"]), ("mavg", 32, 32, 1, 300))
        self.assertEqual(m["tool"]["part"], PART)
        self.assertEqual(m["tool"]["flow_target"], "vitis")
        self.assertEqual(len(m["cache_key"]), 64)
        (e,) = m["entries"]
        self.assertEqual((e["name"], e["top"], e["in_words"], e["out_words"], e["in_bytes"], e["out_bytes"]),
                         ("step", "mavg_step_kernel", 1, 1, 4, 4))
        self.assertEqual((e["in_type"], e["out_type"]), ("kg_float", "kg_float"))
        m = json.loads(emit("fir").files["manifest.json"])
        (e,) = m["entries"]
        self.assertEqual((e["in_words"], e["out_words"], e["in_bytes"], e["out_bytes"]), (256, 241, 1024, 964))
        self.assertEqual(e["params"], [{"name": "x", "elem": "float", "shape": [256], "offset": 0, "words": 256}])
        self.assertEqual(e["result"], {"elem": "float", "shape": [241], "words": 241})

    def test_the_cache_key_follows_what_it_is_made_of(self):
        key = lambda **kw: json.loads(emit("mavg", **kw).files["manifest.json"])["cache_key"]  # noqa: E731
        self.assertEqual(key(), key())
        self.assertNotEqual(key(), key(clock_mhz=250))
        self.assertNotEqual(key(), key(int_bits=64))
        with open(program("mavg")) as f:
            text = f.read()
        self.assertNotEqual(key(), json.loads(emit_hls_program("mavg.kg", None, text=text.replace("0.0 0.0 0.0 0.0", "0.0 0.0 0.0 1.0")
                                                               ).files["manifest.json"])["cache_key"])

    def test_clock(self):
        res = emit("mavg", clock_mhz=250)
        self.assertIn("create_clock -period 4.000", res.files["hls_step.tcl"])
        self.assertIn("create_clock -period 3.333", emit("mavg").files["hls_step.tcl"])
        self.assertIn(f"set_part {{{PART}}}", res.files["hls_step.tcl"])
        self.assertIn("-flow_target vitis", res.files["hls_step.tcl"])
        self.assertIn("KG_EXPORT", res.files["hls_step.tcl"])

    def test_int64_adds_the_flag(self):
        res = emit("mavg", int_bits=64)
        self.assertIn("-DKG_INT64", res.files["hls_step.tcl"])
        self.assertEqual(json.loads(res.files["manifest.json"])["int_bits"], 64)

    def test_an_entry_that_calls_an_extern_is_rejected(self):
        with self.assertRaises(KlongStaticError) as cm:
            emit("pid")
        msg = str(cm.exception)
        self.assertIn("read.adc", msg)
        self.assertIn("write.pwm", msg)
        self.assertIn("no externs", msg)
        self.assertIn("[hls]", msg)

    def test_reachable_externs(self):
        from klongpy.static.compile import compile_program
        p = compile_program(program("pid"), externs=STUBS).program
        self.assertEqual(reachable_externs(p.entries[0]), ["read.adc", "write.pwm"])
        p = compile_program(program("fir")).program
        self.assertEqual(reachable_externs(p.entries[0]), [])

    def test_entries_that_have_no_record_are_rejected_with_a_reason(self):
        for text, why in (('f::{1}\n.entry("f";"-> int")', "no argument"),):
            with self.subTest(why=why):
                with self.assertRaises(KlongStaticError) as cm:
                    emit_hls_program("t.kg", None, text=text, wrapped=True)
                self.assertIn(why, str(cm.exception))
        text = 'f::{x+y}\n.entry("f";"int float -> float")'
        with self.assertRaises(KlongStaticError) as cm:
            emit_hls_program("t.kg", None, text=text, int_bits=64, wrapped=True)
        self.assertIn("--int64", str(cm.exception))

    def test_float64_is_rejected(self):
        from klongpy.static.emit_hls import emit_hls
        from klongpy.static.compile import emit_program
        e = emit_program(program("mavg"), None)
        with self.assertRaises(ValueError):
            emit_hls(e.compiled.program, e, float_bits=64)

    def test_every_accept_fixture_emits_or_is_rejected_for_a_stated_reason(self):
        emitted, rejected = 0, 0
        for path in sorted(glob.glob(os.path.join(HERE, "accept", "*.kg"))):
            with self.subTest(fixture=os.path.basename(path)):
                try:
                    res = emit_hls_program(path, None, externs=STUBS, wrapped=True)
                except KlongStaticError as e:
                    self.assertIn("[hls]", str(e))
                    self.assertTrue("no externs" in str(e) or "no argument" in str(e), str(e))
                    rejected += 1
                    continue
                emitted += 1
                self.assertIn("manifest.json", res.files)
                for ent in res.entries:
                    self.assertTrue(ent["in_words"] > 0 and ent["out_words"] > 0)
        self.assertGreater(emitted, 30)
        self.assertGreater(rejected, 3)


@unittest.skipUnless(CXX and CC, "needs a C and a C++ compiler")
class TestCompiledWithTheSystemCompiler(unittest.TestCase):
    def test_every_emitted_fixture_compiles(self):
        for path in sorted(glob.glob(os.path.join(HERE, "accept", "*.kg"))):
            try:
                res = emit_hls_program(path, None, externs=STUBS, wrapped=True)
            except KlongStaticError:
                continue
            with self.subTest(fixture=os.path.basename(path)):
                d = tempfile.mkdtemp(prefix="kghls_")
                _KEEP.append(d)
                res.write(d)
                for ent in res.entries:
                    for f in (ent["files"]["top"], ent["files"]["tb"]):
                        r = subprocess.run([CXX, "-std=c++14", "-fsyntax-only", "-Wall", "-Wextra", "-Wno-unknown-pragmas",
                                            "-DKG_TRAP_FLAG=kg_hls_trap", "-I" + d, os.path.join(d, f)] +
                                           (["-DKG_INT64"] if res.manifest["int_bits"] == 64 else []),
                                           capture_output=True, text=True)
                        self.assertEqual(r.returncode, 0, r.stderr)

    def test_the_top_needs_no_xilinx_header(self):
        res = emit("fir")
        self.assertNotIn("hls_stream", res.files["fir_run_top.cpp"])
        self.assertNotIn("ap_int", res.files["fir_run_top.cpp"])
        self.assertNotIn("#include <hls", res.files["fir_run_top.cpp"])

    def test_mavg_equals_the_interpreter_over_a_stream(self):
        path = program("mavg")
        b = HlsBuild(path=path, name="mavg")
        rng = np.random.default_rng(1)
        xs = special_floats(rng, (300,))
        calls = [[float(v)] for v in xs]
        with open(path) as f:
            text = f.read()
        compare_with_interpreter(self, b, Host(text, path=path), calls, "step")
        # the state is kept between calls of the kernel
        compare_with_interpreter(self, b, Host(text, path=path), calls, "step", per=7)
        compare_with_interpreter(self, b, Host(text, path=path), calls, "step", per=1)

    def test_fir_equals_the_interpreter_over_blocks(self):
        path = program("fir")
        b = HlsBuild(path=path, name="fir")
        rng = np.random.default_rng(2)
        blocks = special_floats(rng, (20, 256))
        with open(path) as f:
            text = f.read()
        compare_with_interpreter(self, b, Host(text, path=path), [[blk] for blk in blocks], "run")
        compare_with_interpreter(self, b, Host(text, path=path), [[blk] for blk in blocks], "run", per=3)

    def test_state_starts_from_init_and_a_reset_restarts_it(self):
        path = program("mavg")
        b = HlsBuild(path=path, name="mavg")
        ones = pack(b.entry, 32, [[1.0]] * 6)
        out1, _ = b.run(ones)
        out2, _ = b.run(ones)       # each run of the bench is a new process: reset on its first call
        self.assertEqual(out1, out2)
        self.assertEqual(unpack(b.entry, 32, out1)[0].item(), 0.25)

    def test_array_and_scalar_arguments(self):
        text = 'f::{(x@y)+z}\n.entry("f";"int[4] int int -> int")'
        b = HlsBuild(text=text, name="arr")
        self.assertEqual(b.entry["in_type"], "kg_int")
        calls = [[[10, 20, 30, 40], 2, 5], [[1, 2, 3, 4], -1, 100], [[7, 7, 7, 7], 0, 0]]
        compare_with_interpreter(self, b, Host(text), calls, "f")

    def test_mixed_int_and_float(self):
        text = 'f::{x+y}\n.entry("f";"int float -> float")'
        b = HlsBuild(text=text, name="mix2")
        self.assertEqual(b.entry["in_type"], "uint32_t")
        calls = [[3, 0.5], [-7, 2.25], [100, -0.125]]
        compare_with_interpreter(self, b, Host(text), calls, "f")

    def test_int64(self):
        text = 'f::{x*y}\n.entry("f";"int int -> int")'
        b = HlsBuild(text=text, name="i64", int_bits=64)
        calls = [[3000000000, 5], [-4, 7000000000]]
        compare_with_interpreter(self, b, Host(text), calls, "f", int_bits=64)

    def test_the_trap_flag_is_set_by_an_index_out_of_range(self):
        text = 'f::{x@y}\n.entry("f";"int[4] int -> int")'
        b = HlsBuild(text=text, name="idx")
        ok = pack(b.entry, 32, [[[1, 2, 3, 4], 3], [[1, 2, 3, 4], -1]])
        self.assertEqual(b.run(ok)[1], 0)
        bad = pack(b.entry, 32, [[[1, 2, 3, 4], 1], [[1, 2, 3, 4], 9], [[1, 2, 3, 4], 2]])
        out, status = b.run(bad)
        self.assertEqual(status, 1 | 2)
        self.assertEqual(len(out), 12)      # the kernel does not stop: every record has its result
        self.assertEqual(b.run(ok)[1], 0)   # a later run starts clean

    def test_the_trap_flag_is_set_by_an_integer_division_by_zero(self):
        text = 'f::{x!y}\n.entry("f";"int int -> int")'
        b = HlsBuild(text=text, name="mod")
        self.assertEqual(b.run(pack(b.entry, 32, [[7, 3]]))[1], 0)
        self.assertEqual(b.run(pack(b.entry, 32, [[7, 3], [7, 0]]))[1], 1 | 4)


class TestCli(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run([sys.executable, "-m", "klongpy.cli", *args], capture_output=True, text=True, cwd=ROOT)

    def test_emit_hls_writes_the_files(self):
        d = tempfile.mkdtemp(prefix="kghls_")
        _KEEP.append(d)
        r = self.cli("fpga", "source", program("fir"), "--wrapped", "-o", d, "--clock-mhz", "250")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("fir_run_kernel", r.stdout)
        self.assertEqual(sorted(os.listdir(d)), ["fir.c", "fir.h", "fir_run_tb.cpp", "fir_run_top.cpp", "hls_run.tcl",
                                                 "klong_rt.h", "manifest.json"])
        with open(os.path.join(d, "hls_run.tcl")) as f:
            self.assertIn("-period 4.000", f.read())

    def test_extern_is_rejected_with_a_diagnostic(self):
        d = tempfile.mkdtemp(prefix="kghls_")
        _KEEP.append(d)
        r = self.cli("fpga", "source", program("pid"), "-o", d)
        self.assertEqual(r.returncode, 1)
        self.assertIn("no externs", r.stderr)
        self.assertEqual(os.listdir(d), [])

    def test_float64_is_rejected(self):
        r = self.cli("fpga", "source", program("mavg"), "--float64")
        self.assertEqual(r.returncode, 2)
        self.assertIn("unrecognized arguments: --float64", r.stderr)

    def test_synth_belongs_to_compile_fpga(self):
        r = self.cli("compile", "c", program("mavg"), "--synth")
        self.assertEqual(r.returncode, 2)
        self.assertIn("unrecognized arguments: --synth", r.stderr)

    def test_help_names_the_options(self):
        r = self.cli("fpga", "source", "--help")
        self.assertIn("--synth", r.stdout)
        self.assertIn("--clock-mhz", r.stdout)


if __name__ == "__main__":
    unittest.main()
