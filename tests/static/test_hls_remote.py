"""
Vitis HLS on the build machine: C simulation of the generated kernels against the interpreter, C
synthesis, and the trap flag. Skipped unless KLONG_FPGA_BUILD names a reachable machine (user@host, key-based ssh):

    KLONG_FPGA_BUILD=user@fpga-host python -m pytest tests/static/test_hls_remote.py

What these runs show is C simulation (the generated C++ run by the tool's compiler) and C synthesis estimates. They
are not co-simulation and not hardware. No Vivado or v++ run is started.
"""
import os
import re
import shutil
import unittest
from unittest import mock

import numpy as np

from klongpy.hw.remote import Remote, RemoteError, build_target, hls_settings, resolve_root, root_setting
from klongpy.hw.reports import warning_classes
from klongpy.hw.synth import synthesize
from klongpy.static.compile import emit_hls_program
from klongpy.static.signature import Ty

from .harness import Host, same
from .test_emit_hls import pack, program, special_floats, unpack


def _remote():
    if not os.environ.get("KLONG_FPGA_BUILD"):
        return None
    try:
        r = Remote()
        r.check()
        return r
    except RemoteError:
        return None


REMOTE = _remote()
N = 1000


def interpreter_values(path, entry, calls):
    with open(path) as f:
        host = Host(f.read(), path=path)
    out = []
    for args in calls:
        kind, v = host.call(entry, *args)
        assert kind == "ok", v
        out.append(v)
    return out


def float_rules(test, want, got, shape):
    """The float rules of the C tests: float32 results within 1e-5, any nan equals any nan."""
    ty = Ty("float", tuple(shape))
    for i, (w, g) in enumerate(zip(want, got)):
        d = same(w, g if shape else g.item(), ty, 32)
        test.assertIsNone(d, f"record {i}: {d}")


@unittest.skipUnless(REMOTE, "set KLONG_FPGA_BUILD=user@host to run Vitis HLS on the build machine")
class TestCsimAndSynthesis(unittest.TestCase):
    runs = {}

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(110)
        cls.mavg_calls = [[float(v)] for v in special_floats(rng, (N,))]
        cls.fir_calls = [[blk] for blk in special_floats(rng, (N, 256), rate=0.002)]
        for name, entry, calls in (("mavg", "step", cls.mavg_calls), ("fir", "run", cls.fir_calls)):
            res = emit_hls_program(program(name), None, wrapped=True)
            e = res.entries[0]
            data = {"in.bin": pack(e, 32, calls)}
            cls.runs[name] = (res, e, synthesize(res, REMOTE, timeout=1800, csim_data=data))

    def check_csim(self, name, entry, calls):
        res, e, out = self.runs[name]
        r = out[e["c_name"]]
        self.assertEqual(r["result"].returncode, 0, r["log"][-3000:] + r["stdout"][-2000:])
        self.assertIsNotNone(r["out"], "the test bench wrote no out.bin")
        self.assertEqual(int.from_bytes(r["status"], "little"), 0)
        got = unpack(e, 32, r["out"])
        self.assertEqual(len(got), N)
        float_rules(self, interpreter_values(program(name), entry, calls), got, e["result"]["shape"])

    def test_mavg_csim_equals_the_interpreter_over_1000_records(self):
        self.check_csim("mavg", "step", self.mavg_calls)

    def test_fir_csim_equals_the_interpreter_over_1000_records(self):
        self.check_csim("fir", "run", self.fir_calls)

    def test_synthesis_completes_without_errors(self):
        for name in ("mavg", "fir"):
            res, e, out = self.runs[name]
            r = out[e["c_name"]]
            with self.subTest(program=name):
                self.assertEqual(r["result"].returncode, 0)
                self.assertIsNotNone(r["top"], "no csynth report")
                self.assertEqual(r["top"]["module"], e["top"])
                self.assertEqual(r["top"]["part"], "xcu250-figd2104-2L-e")
                self.assertFalse([c for c, (sev, _, _) in r["warnings"].items() if sev != "WARNING"], r["warnings"])
                self.assertNotRegex(r["log"], r"(?m)^ERROR:")
                self.assertGreater(r["top"]["lut"], 0)
                self.assertGreater(r["top"]["fmax_mhz"], 100)

    def test_chunked_calls_continue_the_state(self):
        res, e, out = self.runs["mavg"]
        data = {"in.bin": pack(e, 32, self.mavg_calls)}
        again = synthesize(res, REMOTE, timeout=900, csim_data=data, chunk=64, synth=False)[e["c_name"]]
        self.assertEqual(again["result"].returncode, 0, again["log"][-2000:])
        self.assertEqual(again["out"], out[e["c_name"]]["out"])


@unittest.skipUnless(REMOTE, "set KLONG_FPGA_BUILD=user@host to run Vitis HLS on the build machine")
class TestTrapFlag(unittest.TestCase):
    def run_csim(self, text, name, calls):
        res = emit_hls_program(name + ".kg", None, text=text, prefix=name, wrapped=True)
        e = res.entries[0]
        r = synthesize(res, REMOTE, timeout=900, csim_data={"in.bin": pack(e, 32, calls)}, synth=False)[e["c_name"]]
        self.assertEqual(r["result"].returncode, 0, r["log"][-2000:] + r["stdout"][-2000:])
        return r, int.from_bytes(r["status"], "little"), unpack(e, 32, r["out"])

    def test_an_index_out_of_range_sets_the_flag(self):
        text = 'f::{x@y}\n.entry("f";"int[4] int -> int")'
        _, status, got = self.run_csim(text, "idx", [[[1, 2, 3, 4], 3], [[1, 2, 3, 4], -1]])
        self.assertEqual(status, 0)
        self.assertEqual([int(v) for v in got], [4, 4])
        _, status, got = self.run_csim(text, "idx", [[[1, 2, 3, 4], 1], [[1, 2, 3, 4], 9], [[1, 2, 3, 4], 2]])
        self.assertEqual(status, 1 | 2)
        self.assertEqual(len(got), 3)

    def test_an_integer_division_by_zero_sets_the_flag(self):
        text = 'f::{x!y}\n.entry("f";"int int -> int")'
        _, status, _ = self.run_csim(text, "md", [[7, 3]])
        self.assertEqual(status, 0)
        _, status, got = self.run_csim(text, "md", [[7, 3], [7, 0]])
        self.assertEqual(status, 1 | 4)
        self.assertEqual(int(got[0]), 1)


class TestRemoteTool(unittest.TestCase):
    """The tool itself, on this machine (no ssh): paths, files, timeouts, the build rule."""

    def setUp(self):
        import tempfile
        self.root = tempfile.mkdtemp(prefix="kgremote_")
        self.r = Remote(target=None, root=self.root, settings=None)

    def test_runs_in_a_fresh_directory_with_the_files(self):
        res = self.r.run("cat a.txt sub/b.txt > c.txt; pwd", files={"a.txt": b"A", "sub/b.txt": "B"}, fetch=["c.txt"], keep=True)
        self.assertEqual(res.returncode, 0)
        self.assertEqual(res.read("c.txt"), "AB")
        self.assertTrue(res.workdir.startswith(self.root + "/work/"))
        self.assertTrue(os.path.isdir(res.workdir))
        self.assertEqual(res.stdout.strip(), res.workdir)
        self.r.remove_workdir(res.workdir)
        self.assertFalse(os.path.exists(res.workdir))

    def test_the_directory_is_removed_unless_kept(self):
        res = self.r.run("true")
        self.assertFalse(os.path.exists(res.workdir))

    def test_globs_are_fetched(self):
        res = self.r.run("mkdir -p p/q; echo 1 > p/q/x_csynth.xml; echo 2 > p/q/y_csynth.xml; echo 3 > p/q/z.rpt",
                         fetch=["p/q/*_csynth.xml", "missing/*"])
        self.assertEqual(sorted(os.listdir(res.path("p/q"))), ["x_csynth.xml", "y_csynth.xml"])

    def test_a_failing_command_returns_its_status_and_output(self):
        res = self.r.run("echo out; echo err >&2; exit 3")
        self.assertEqual((res.returncode, res.stdout.strip(), res.stderr.strip()), (3, "out", "err"))

    def test_timeout_stops_the_command(self):
        res = self.r.run("sleep 30; echo late", timeout=1)
        self.assertTrue(res.timed_out)
        self.assertNotIn("late", res.stdout)

    def test_paths_are_validated(self):
        for bad in ("/etc/passwd", "../x", "a/../../x", "-x", ""):
            with self.subTest(path=bad):
                with self.assertRaises(RemoteError):
                    self.r.run("true", files={bad: b"x"})
        with self.assertRaises(RemoteError):
            self.r.remove_workdir("/tmp")
        with self.assertRaises(RemoteError):
            self.r.remove_workdir(self.root + "/work/../..")
        with self.assertRaises(RemoteError):
            self.r.remove_workdir(self.root + "/work")
        with self.assertRaises(RemoteError):
            self.r.run("true", fetch=["../../etc/passwd"])

    def test_the_remote_root_is_fixed(self):
        with mock.patch.dict(os.environ, {"KLONG_FPGA_ROOT": "/srv/klong-hw"}):
            with self.assertRaises(RemoteError):
                Remote(target="someone@somewhere", root="/tmp/x")
            self.assertEqual(Remote(target="someone@somewhere").root, "/srv/klong-hw")
            self.assertEqual(Remote(target="someone@somewhere", root="/srv/klong-hw").root, "/srv/klong-hw")

    def test_the_default_root_is_under_the_home_of_the_machine(self):
        env = {k: v for k, v in os.environ.items() if k != "KLONG_FPGA_ROOT"}
        self.assertEqual(root_setting(env), "~/klong-hw")
        self.assertEqual(root_setting({"KLONG_FPGA_ROOT": "/data/hw/"}), "/data/hw")
        for bad in ("relative/dir", "/a/../b", "/with space", "~", "/", "~/../x"):
            with self.subTest(root=bad):
                with self.assertRaises(RemoteError):
                    root_setting({"KLONG_FPGA_ROOT": bad})
        self.assertEqual(resolve_root(None, "/data/hw"), "/data/hw")
        self.assertEqual(resolve_root(None, "~/klong-hw"), os.path.join(os.path.expanduser("~"), "klong-hw"))

        class Done:
            returncode, stdout, stderr = 0, b"/home/someone\n", b""

        with mock.patch.dict(os.environ, env, clear=True), mock.patch("subprocess.run", return_value=Done) as run:
            r = Remote(target="someone@fpga-host")          # no connection yet
            self.assertEqual(run.call_count, 0)
            self.assertEqual(r.root, "/home/someone/klong-hw")
            self.assertEqual(r.work, "/home/someone/klong-hw/work")
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args[0][0][-2:], ["someone@fpga-host", 'printf %s "$HOME"'])

    def test_the_hls_settings_come_from_the_environment(self):
        self.assertIsNone(hls_settings({}))
        self.assertEqual(hls_settings({"KLONG_FPGA_HLS_SETTINGS": "/opt/hls/settings64.sh"}), "/opt/hls/settings64.sh")
        with mock.patch.dict(os.environ, {"KLONG_FPGA_HLS_SETTINGS": "/opt/hls/settings64.sh"}):
            self.assertEqual(Remote(target=None).settings, "/opt/hls/settings64.sh")
            self.assertIsNone(Remote(target=None, settings=None).settings)

    def test_destinations_are_validated(self):
        for bad in ("a b", "-oProxyCommand=x", "a;b", "x@y z"):
            with self.subTest(target=bad):
                with self.assertRaises(RemoteError):
                    Remote(target=bad)
        self.assertEqual(build_target({"KLONG_FPGA_BUILD": ""}), None)
        self.assertEqual(build_target({"KLONG_FPGA_BUILD": "user@fpga-host"}), "user@fpga-host")

    @unittest.skipUnless(shutil.which("flock"), "needs flock")
    def test_a_vivado_build_holds_the_lock(self):
        res = self.r.run(f"flock -n 8 8>{self.root}/build.lock && echo free || echo held", vivado=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stdout.strip(), "held")

    @unittest.skipUnless(shutil.which("flock"), "needs flock")
    def test_a_second_build_is_refused_while_the_first_runs(self):
        import subprocess
        import time
        holder = subprocess.Popen(["bash", "-c", f"exec 9>{self.root}/build.lock; flock 9; sleep 20"])
        try:
            time.sleep(0.5)
            if not os.path.exists(self.root + "/build.lock"):
                self.skipTest("no flock")
            res = self.r.run("echo started", vivado=True)
            self.assertEqual(res.returncode, 75)
            self.assertNotIn("started", res.stdout)
            self.assertIn("refusing", res.stderr)
        finally:
            holder.terminate()      # the process this test started
            holder.wait()

    def test_hls_runs_do_not_need_the_lock(self):
        res = self.r.run("echo ok")
        self.assertEqual(res.stdout.strip(), "ok")

    def test_warning_classes(self):
        log = ("WARNING: [HLS 200-880] The II Violation in module 'a'\nINFO: [HLS 200-10] x\n"
               "WARNING: [HLS 200-880] The II Violation in module 'b'\nWARNING: [RTGEN 206-101] Register 'p' is power-on init\n")
        c = warning_classes(log)
        self.assertEqual(sorted(c), ["HLS 200-880", "RTGEN 206-101"])
        self.assertEqual(c["HLS 200-880"][1], 2)
        self.assertTrue(re.match(r"The II Violation", c["HLS 200-880"][2]))


if __name__ == "__main__":
    unittest.main()
