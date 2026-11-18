"""
kgpy fpga build: the timing gate, the cache, the detached build script and the status of a build.
Everything here runs without a Xilinx tool: the build script runs on this machine with stand-ins for ``vitis_hls``
and ``v++`` (shell scripts that write the files the real tools write), so what is tested is our flow: the guard,
the phases, the gate, build.json, the cache. The real tools are exercised by tests/static/test_hw_card.py.
"""
import io
import json
import os
import shutil
import stat
import sys
import tempfile
import time
import unittest

from klongpy.hw import gate
from klongpy.hw.build import Builder, Plan, hw_build_main, suggest_clock
from klongpy.hw.remote import Remote, RemoteError
from klongpy.static.compile import emit_hls_program

from .test_ir import EMBEDDED

HEADER = ("    WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns)  THS Failing Endpoints  "
          "THS Total Endpoints     WPWS(ns)     TPWS(ns)  TPWS Failing Endpoints  TPWS Total Endpoints  \n"
          "    -------      -------  ---------------------  -------------------      -------      -------  "
          "---------------------  -------------------     --------     --------  ----------------------  --------------------  \n")
PASS_RPT = ("Copyright 1986-2022 Xilinx, Inc.\n\n------\n| Design Timing Summary\n| ---------------------\n------\n\n" + HEADER +
            "      0.057        0.000                      0              1493154        0.009        0.000"
            "                      0              1489305        0.000        0.000                       0               "
            "505431  \n\n\nAll user specified timing constraints are met.\n\n\n| Clock Summary\n")
FAIL_RPT = PASS_RPT.replace("      0.057        0.000                      0", "     -1.597      -68.287                     55") \
                   .replace("All user specified timing constraints are met.", "Timing constraints are not met.")
UTIL = ("| CLB LUTs                   | 406298 | 56815 |       1792 |   1726208 | 23.54 |\n"
        "| CLB Registers              | 465990 | 99159 |          0 |   3456000 | 13.48 |\n"
        "| Block RAM Tile    | 382.5 |     0 |          0 |      2688 | 14.23 |\n"
        "| DSPs           |  340 |     4 |          0 |     12288 |  2.77 |\n")
INFO = "Scalable Clocks\n---------------\n   Name:      DATA_CLK\n   Index:     0\n   Type:      DATA\n   Frequency: 300 MHz\n"

STUB_HLS = r"""#!/usr/bin/env bash
if [ "$1" = "-version" ]; then echo "Vitis HLS stand-in 2022.2"; exit 0; fi
tcl="$2"
proj=$(sed -n 's/^open_project -reset \(proj_[A-Za-z0-9_]*\).*/\1/p' "$tcl")
mkdir -p "$proj/sol1/impl"
echo "export from $KG_EXPORT" > "$proj/sol1/impl/export.xo"
echo "HLS stand-in ran $tcl"
"""
STUB_VPP = r"""#!/usr/bin/env bash
if [ "$1" = "--version" ]; then echo "v++ stand-in 2022.2"; exit 0; fi
out=""; prev=""
for a in "$@"; do if [ "$prev" = "-o" ]; then out="$a"; fi; prev="$a"; done
while [ -f "$STUB_HOLD" ]; do sleep 0.2; done
mkdir -p _x/reports/link/imp
cp "$STUB_REPORT" _x/reports/link/imp/impl_1_hw_bb_locked_timing_summary_routed.rpt
cp "$STUB_UTIL" _x/reports/link/imp/impl_1_full_util_routed.rpt
echo "image of $*" > "$out"
echo "$@" > vpp-args.txt
exit "${STUB_EXIT:-0}"
"""


def write_exec(path, text):
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, 0o755)


class TimingGateTest(unittest.TestCase):
    def test_parse_pass_and_fail(self):
        p = gate.parse_timing_summary(PASS_RPT)
        self.assertTrue(p["met"])
        self.assertEqual((p["wns"], p["tns"], p["tns_failing"], p["whs"]), (0.057, 0.0, 0, 0.009))
        self.assertTrue(gate.passes(p))
        f = gate.parse_timing_summary(FAIL_RPT)
        self.assertFalse(f["met"])
        self.assertEqual((f["wns"], f["tns"], f["tns_failing"]), (-1.597, -68.287, 55))
        self.assertFalse(gate.passes(f))

    def test_both_conditions_are_needed(self):
        self.assertFalse(gate.passes(gate.parse_timing_summary(PASS_RPT.replace("      0.057 ", "     -0.001 "))))
        self.assertFalse(gate.passes(gate.parse_timing_summary(PASS_RPT.replace("All user specified timing constraints are met.", ""))))
        self.assertFalse(gate.passes(gate.parse_timing_summary("no summary at all")))
        self.assertFalse(gate.passes(None))

    def test_utilization_and_clock(self):
        u = gate.parse_utilization(UTIL)
        self.assertEqual(u["lut"]["used"], 406298)
        self.assertEqual(u["dsp"]["available"], 12288)
        self.assertEqual(gate.parse_clock(INFO), 300.0)

    def _link(self, rpt):
        d = tempfile.mkdtemp(prefix="kggate_")
        self.addCleanup(shutil.rmtree, d, True)
        os.makedirs(os.path.join(d, "_x/reports/link/imp"))
        with open(os.path.join(d, "_x/reports/link/imp/impl_1_hw_bb_locked_timing_summary_routed.rpt"), "w") as f:
            f.write(rpt)
        with open(os.path.join(d, "img.xclbin"), "wb") as f:
            f.write(b"bits")
        return d

    def test_failing_image_is_renamed_and_never_loadable(self):
        d = self._link(FAIL_RPT)
        ok, timing, name = gate.gate(d, "img.xclbin")
        self.assertFalse(ok)
        self.assertEqual(name, "img.timing-failed.xclbin")
        self.assertFalse(os.path.exists(os.path.join(d, "img.xclbin")))
        self.assertEqual(timing["wns_ns"], -1.597)
        good, why = gate.image_ok(os.path.join(d, name))
        self.assertFalse(good)
        self.assertIn("failed routed timing", why)

    def test_image_ok_rules(self):
        d = self._link(PASS_RPT)
        x = os.path.join(d, "img.xclbin")
        self.assertEqual(gate.image_ok(x)[0], False)            # no build.json yet
        self.assertEqual(gate.main([d, "--xclbin", "img.xclbin", "--key", "k", "--clock-mhz", "300"]), 0)
        ok, meta = gate.image_ok(x)
        self.assertTrue(ok, meta)
        self.assertTrue(meta["passed"] and meta["timing"]["wns_ns"] == 0.057)
        with open(x, "ab") as f:                                # the file changed after the gate
            f.write(b"x")
        ok, why = gate.image_ok(x)
        self.assertFalse(ok)
        self.assertIn("SHA-256", why)
        # a build.json that says the timing failed
        with open(os.path.join(d, "build.json")) as f:
            m = json.load(f)
        m["timing"]["wns_ns"] = -0.1
        with open(os.path.join(d, "build.json"), "w") as f:
            json.dump(m, f)
        self.assertFalse(gate.image_ok(x)[0])

    def test_gate_main_failure_exit_status_and_json(self):
        d = self._link(FAIL_RPT)
        self.assertEqual(gate.main([d, "--xclbin", "img.xclbin", "--key", "k", "--clock-mhz", "300"]), 4)
        with open(os.path.join(d, "build.json")) as f:
            m = json.load(f)
        self.assertFalse(m["passed"])
        self.assertEqual(m["xclbin"], "img.timing-failed.xclbin")
        self.assertFalse(gate.image_ok(os.path.join(d, m["xclbin"]))[0])

    def test_emulation_image_only_in_emulation(self):
        d = self._link(PASS_RPT)
        self.assertEqual(gate.main([d, "--xclbin", "img.xclbin", "--target", "hw_emu"]), 0)
        x = os.path.join(d, "img.xclbin")
        self.assertFalse(gate.image_ok(x)[0])
        os.environ["XCL_EMULATION_MODE"] = "hw_emu"
        try:
            self.assertTrue(gate.image_ok(x)[0])
        finally:
            del os.environ["XCL_EMULATION_MODE"]

    def test_suggested_clock(self):
        self.assertEqual(suggest_clock(300, -0.4), 250)         # 1/(3.333+0.4) = 268 MHz, minus 5 percent, in 5 MHz steps
        self.assertLess(suggest_clock(300, -0.01), 300)
        self.assertGreaterEqual(suggest_clock(300, -50), 5)


class BuildFlowTest(unittest.TestCase):
    """The build script and its state, with stand-ins for the tools and a temporary root."""

    @classmethod
    def setUpClass(cls):
        if not shutil.which("bash") or not shutil.which("python3") and not sys.executable:
            raise unittest.SkipTest("needs bash")

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="kgbuild_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        bindir = os.path.join(self.tmp, "bin")
        os.makedirs(bindir)
        write_exec(os.path.join(bindir, "vitis_hls"), STUB_HLS)
        write_exec(os.path.join(bindir, "v++"), STUB_VPP)
        # the script runs `python3 gate.py`: make sure a python3 exists on the stand-in PATH
        write_exec(os.path.join(bindir, "python3"), '#!/usr/bin/env bash\nexec "%s" "$@"\n' % sys.executable)
        if not shutil.which("setsid"):
            write_exec(os.path.join(bindir, "setsid"), '#!/usr/bin/env bash\nexec "$@"\n')
        if not shutil.which("flock"):      # `flock -n 9`: the lock is on the open file, so it outlives this process
            write_exec(os.path.join(bindir, "flock"), '#!/usr/bin/env bash\nexec "%s" -c "import fcntl,sys; '
                       'fcntl.flock(int(sys.argv[2]), fcntl.LOCK_EX | fcntl.LOCK_NB)" "$@"\n' % sys.executable)
        self.report = os.path.join(self.tmp, "report.rpt")
        self.util = os.path.join(self.tmp, "util.rpt")
        self.hold = os.path.join(self.tmp, "hold")
        with open(self.util, "w") as f:
            f.write(UTIL)
        self.env = {"PATH": bindir + os.pathsep + os.environ["PATH"], "STUB_REPORT": self.report, "STUB_UTIL": self.util,
                    "STUB_HOLD": self.hold}
        self.saved = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        self.addCleanup(self._restore)
        self.root = os.path.join(self.tmp, "root")
        os.makedirs(self.root)
        self.builder = Builder(Remote(target=None, root=self.root), vitis_settings=None, platform_repo="/none", xrt="/none")

    def _restore(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def plan(self, *names, clock=300, target="hw"):
        outs = [emit_hls_program(os.path.join(EMBEDDED, n + ".kg"), None, clock_mhz=clock) for n in names]
        return Plan(outs, clock, target=target)

    def finish(self, dirname, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self.builder.state(dirname)
            if st["state"] != "running":
                return st
            time.sleep(0.2)
        self.fail("the stand-in build did not finish")

    def test_keys_and_names(self):
        a, b = self.plan("mavg"), self.plan("fir")
        both = self.plan("mavg", "fir")
        self.assertEqual(a.key, a.outputs[0].manifest["cache_key"])
        self.assertEqual(a.name, "mavg")
        self.assertNotEqual(both.key, a.key)
        self.assertNotEqual(both.key, self.plan("fir", "mavg").key)       # the order is part of the image
        self.assertEqual(self.plan("mavg").key, a.key)
        self.assertNotEqual(self.plan("mavg", clock=250).key, a.key)
        self.assertTrue(both.name.startswith("klong_"))
        self.assertEqual([e["top"] for _p, e in both.kernels], ["mavg_step_kernel", "fir_run_kernel"])
        self.assertEqual(self.plan("mavg", target="hw_emu").dirname, a.key + "-hw_emu")
        self.assertEqual(b.image, "fir.xclbin")

    def test_script_has_the_link_recipe_and_the_guard(self):
        s = self.builder.script(self.plan("mavg", "fir"))
        for want in ("v++ -l -t hw -R2", "--platform xilinx_u250_gen3x16_xdma_4_1_202210_1", "--kernel_frequency 300",
                     "--vivado.synth.jobs 8 --vivado.impl.jobs 8", "--save-temps", "--config link.cfg", "KG_EXPORT=1 vitis_hls",
                     "sp=mavg_step_kernel_1.in:DDR[0]", "sp=fir_run_kernel_1.status:DDR[0]", "flock -n 9", "pgrep -f",
                     "python3 gate.py"):
            self.assertIn(want, s)
        self.assertNotIn("sudo", s)
        self.assertNotIn("xbmgmt", s)

    def test_passing_build(self):
        with open(self.report, "w") as f:
            f.write(PASS_RPT)
        plan = self.plan("mavg", "fir")
        st = self.builder.start(plan)
        self.assertIn(st["state"], ("running", "passed"))
        st = self.finish(plan.dirname)
        self.assertEqual(st["state"], "passed", st)
        meta = st["build"]
        self.assertTrue(meta["passed"])
        self.assertEqual(meta["timing"]["wns_ns"], 0.057)
        self.assertEqual(meta["key"], plan.key)
        self.assertEqual(meta["sources_sha256"], plan.sources_sha)
        self.assertEqual(meta["utilization"]["lut"]["used"], 406298)
        self.assertEqual([k["top"] for m in meta["kernels"] for k in m["entries"]], ["mavg_step_kernel", "fir_run_kernel"])
        image = os.path.join(st["dir"], plan.image)
        self.assertTrue(gate.image_ok(image)[0])
        for top in ("mavg_step_kernel", "fir_run_kernel"):
            self.assertTrue(os.path.exists(os.path.join(st["dir"], "xo", top + ".xo")))
        with open(os.path.join(st["dir"], "vpp-args.txt")) as f:
            args = f.read()
        self.assertIn("--kernel_frequency 300", args)
        self.assertIn("mavg_step_kernel.xo", args)
        # the cache: a second request is answered without a build
        out, err = io.StringIO(), io.StringIO()
        self.assertEqual(self._main(["%s/mavg.kg" % EMBEDDED, "%s/fir.kg" % EMBEDDED], out, err), 0)
        self.assertIn("cached", out.getvalue())
        self.assertIn("WNS 0.057 ns", out.getvalue())
        self.assertNotIn("build started", out.getvalue())

    def _main(self, files, out, err, **kw):
        return hw_build_main(files, 32, kw.pop("clock_mhz", 300), 1, builder=self.builder, stdout=out, stderr=err, poll=0.2, **kw)

    def test_failing_build_is_kept_renamed_and_reported(self):
        with open(self.report, "w") as f:
            f.write(FAIL_RPT)
        out, err = io.StringIO(), io.StringIO()
        rc = self._main(["%s/mavg.kg" % EMBEDDED], out, err, wait=True)
        self.assertEqual(rc, 1, out.getvalue() + err.getvalue())
        text = out.getvalue() + err.getvalue()
        self.assertIn("WNS -1.597 ns", text)
        self.assertIn("NOT MET", text)
        self.assertIn("--clock-mhz", text)                 # the suggestion
        self.assertIn("not loadable", text)
        plan = self.plan("mavg")
        d = self.builder.dir_of(plan.dirname)
        self.assertTrue(os.path.exists(os.path.join(d, "mavg.timing-failed.xclbin")))
        self.assertFalse(os.path.exists(os.path.join(d, "mavg.xclbin")))
        # asking again does not rebuild, and does not retry at another clock
        out, err = io.StringIO(), io.StringIO()
        self.assertEqual(self._main(["%s/mavg.kg" % EMBEDDED], out, err), 1)
        self.assertIn("did not pass", err.getvalue())
        self.assertNotIn("build started", out.getvalue())
        # --force builds again
        with open(self.report, "w") as f:
            f.write(PASS_RPT)
        out, err = io.StringIO(), io.StringIO()
        self.assertEqual(self._main(["%s/mavg.kg" % EMBEDDED], out, err, wait=True, force=True), 0, out.getvalue() + err.getvalue())
        self.assertIn("build started", out.getvalue())

    def test_second_build_is_refused_while_one_runs(self):
        with open(self.report, "w") as f:
            f.write(PASS_RPT)
        with open(self.hold, "w") as f:
            f.write("x")
        first = self.plan("mavg")
        self.builder.start(first)
        try:
            time.sleep(1.0)
            st = self.builder.state(first.dirname)
            self.assertEqual(st["state"], "running", st)
            self.assertTrue(st["phase"].startswith("link"), st["phase"])
            with self.assertRaises(RemoteError) as cm:
                self.builder.start(self.plan("fir"))
            self.assertIn("is running", str(cm.exception))
            out, err = io.StringIO(), io.StringIO()
            self.assertEqual(self._main(["%s/fir.kg" % EMBEDDED], out, err), 1)
            self.assertIn("is running", err.getvalue())
            out, err = io.StringIO(), io.StringIO()
            self.assertEqual(hw_build_main([], 32, 300, 1, status=True, builder=self.builder, stdout=out, stderr=err), 0)
            self.assertIn("running", out.getvalue())
        finally:
            os.remove(self.hold)
        st = self.finish(first.dirname)
        self.assertEqual(st["state"], "passed", st)

    def test_script_refuses_when_the_lock_is_held(self):
        import subprocess
        plan = self.plan("mavg")
        with open(self.report, "w") as f:
            f.write(PASS_RPT)
        # hold the lock the way a running build does, then start the script directly
        holder = subprocess.Popen(["bash", "-c", 'exec 9>"%s/build.lock"; flock -n 9; sleep 20' % self.root],
                                  env=dict(os.environ))
        try:
            time.sleep(0.5)
            self.builder.remote._sh("true")
            B = self.builder.dir_of(plan.dirname)
            os.makedirs(B)
            self.builder.remote._sh_with("tar xf - -C %s" % B, b"", 10)
            with open(os.path.join(B, "build.sh"), "w") as f:
                f.write(self.builder.script(plan))
            p = subprocess.run(["bash", os.path.join(B, "build.sh")], capture_output=True)
            with open(os.path.join(B, "exit-code")) as f:
                self.assertEqual(f.read().strip(), "75")
            self.assertFalse(os.path.exists(os.path.join(B, "xo", "mavg_step_kernel.xo")))
        finally:
            holder.terminate()
            holder.wait()

    def test_link_failure_is_reported_with_the_log(self):
        with open(self.report, "w") as f:
            f.write(PASS_RPT)
        os.environ["STUB_EXIT"] = "1"
        try:
            plan = self.plan("mavg")
            self.builder.start(plan)
            st = self.finish(plan.dirname)
        finally:
            del os.environ["STUB_EXIT"]
        # v++ exits 1 but wrote an image and a passing report: the gate decides (it passes); the exit of v++ is logged
        self.assertIn("v++ exit 1", open(os.path.join(st["dir"], "build.log")).read())


class CliTest(unittest.TestCase):
    def test_flags(self):
        import subprocess
        py = sys.executable
        r = subprocess.run([py, "-W", "ignore", "-m", "klongpy.cli", "fpga", "build", "--lanes", "2", "x.kg"], capture_output=True, text=True,
                           env=dict(os.environ, KLONG_FPGA_BUILD=""))
        self.assertNotEqual(r.returncode, 0)
        r = subprocess.run([py, "-W", "ignore", "-m", "klongpy.cli", "fpga", "status"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("kgpy fpga", r.stderr)
        r = subprocess.run([py, "-W", "ignore", "-m", "klongpy.cli", "fpga", "status"], capture_output=True, text=True,
                           env=dict(os.environ, KLONG_FPGA_BUILD=""))
        self.assertEqual(r.returncode, 2)
        self.assertIn("KLONG_FPGA_BUILD", r.stderr)


if __name__ == "__main__":
    unittest.main()
