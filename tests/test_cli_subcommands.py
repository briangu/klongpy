"""The subcommands of kgpy: lint, check, compile c, compile fpga, fpga build, fpga status, ast."""
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MAVG = os.path.join("examples", "embedded", "mavg.kg")
PID = os.path.join("examples", "embedded", "pid.kg")


def kgpy(*argv, stdin=None, cwd=ROOT, env=None):
    return subprocess.run([sys.executable, "-W", "ignore", "-m", "klongpy.cli", *argv], input=stdin, capture_output=True,
                          text=True, timeout=180, cwd=cwd, env=env)


class TestSubcommands(unittest.TestCase):
    def test_lint(self):
        r = kgpy("lint", "-", stdin="my_var::5\n")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("W001", r.stdout)
        self.assertEqual(kgpy("lint", "-", stdin="1+\n").returncode, 1)

    def test_check(self):
        r = kgpy("check", MAVG, PID)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(kgpy("check", os.path.join("tests", "static", "reject", "where.kg")).returncode, 1)
        r = kgpy("check", MAVG, "--dump-ir")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout)

    def test_compile_c(self):
        with tempfile.TemporaryDirectory() as d:
            r = kgpy("compile", "c", MAVG, "-o", d, "--int64", "--float64")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(sorted(os.listdir(d)), ["klong_rt.h", "mavg.c", "mavg.h"])

    def test_fpga_source(self):
        with tempfile.TemporaryDirectory() as d:
            r = kgpy("fpga", "source", MAVG, "--lanes", "2", "--clock-mhz", "250", "-o", d)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("manifest.json", os.listdir(d))

    def test_fpga_build_and_status_need_a_build_machine(self):
        env = dict(os.environ, KLONG_FPGA_BUILD="")
        for argv in (["fpga", "build", MAVG], ["fpga", "status"], ["fpga", "build", "--wait"]):
            r = kgpy(*argv, env=env)
            self.assertEqual(r.returncode, 2, argv)
            self.assertIn("KLONG_FPGA_BUILD", r.stderr)

    def test_ast(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.kg")
            with open(p, "w") as f:
                f.write("1+2\n")
            r = kgpy("ast", p)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(r.stdout.strip())

    def test_options_before_and_after_files(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a.kg"), os.path.join(d, "b.kg")
            for p in (a, b):
                with open(p, "w") as f:
                    f.write("f::{x::5;x}\n")
            outs = {kgpy("lint", *argv).stdout.replace(d, "") for argv in
                    ([a, b, "-v", "--select", "W019"], ["-v", a, "--select", "W019", b], ["--select=W019", "-v", a, b])}
            self.assertEqual(len(outs), 1)
        for argv in (["check", "--int64", "-v", MAVG, PID], ["check", MAVG, "-v", PID, "--int64"]):
            self.assertEqual(kgpy(*argv).returncode, 0, argv)

    def test_option_of_another_subcommand_is_refused(self):
        for argv, word in ((["lint", "-", "--int64"], "--int64"), (["check", MAVG, "--lanes", "2"], "--lanes"),
                           (["compile", "c", MAVG, "--synth"], "--synth"), (["fpga", "source", MAVG, "--float64"], "--float64"),
                           (["fpga", "source", MAVG, "--wait"], "--wait"), (["fpga", "status", "--force"], "--force"), (["ast", MAVG, "-o", "x"], "-o")):
            r = kgpy(*argv, stdin="")
            self.assertEqual(r.returncode, 2, argv)
            self.assertIn("unrecognized arguments", r.stderr)
            self.assertIn(word, r.stderr)

    def test_old_flags_are_refused(self):
        for flag in ("--lint", "--check-static", "--emit-c", "--emit-hls", "--hw-build", "--status", "--dump-ast", "--synth"):
            r = kgpy(flag, MAVG, stdin="")
            self.assertEqual(r.returncode, 2, flag)
            self.assertIn("unrecognized arguments: " + flag, r.stderr)
            self.assertNotIn("Traceback", r.stderr)

    def test_missing_file_is_a_usage_error(self):
        for argv in (["lint"], ["check"], ["compile", "c"], ["fpga", "source"], ["ast"], ["compile"], ["fpga"], ["compile", "fpga", MAVG]):
            self.assertEqual(kgpy(*argv, stdin="").returncode, 2, argv)

    def test_help(self):
        r = kgpy("--help")
        self.assertEqual(r.returncode, 0)
        for name in ("lint FILE", "check FILE", "compile c FILE", "fpga source FILE", "fpga build", "fpga status", "ast FILE"):
            self.assertIn(name, r.stdout)
        self.assertNotIn("--emit-c", r.stdout)
        self.assertNotIn("--int64", r.stdout)
        r = kgpy("compile", "c", "--help")
        self.assertIn("--line-directives", r.stdout)
        self.assertNotIn("--select", r.stdout)
        self.assertIn("--select", kgpy("lint", "--help").stdout)
        self.assertIn("--wait", kgpy("fpga", "build", "--help").stdout)
        self.assertIn("build", kgpy("fpga", "--help").stdout)


class TestRunFormUnchanged(unittest.TestCase):
    def test_file_named_like_a_subcommand(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "lint"), "w") as f:
                f.write('.p("ran lint")\n')
            r = kgpy("./lint", cwd=d, env=dict(os.environ, PYTHONPATH=ROOT))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("ran lint", r.stdout)

    def test_file_with_arguments(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "args.kg")
            with open(p, "w") as f:
                f.write(".p(.os.argv)\n")
            r = kgpy(p, "--", "a", "b")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("a", r.stdout)
            self.assertIn("b", r.stdout)

    def test_expr(self):
        r = kgpy("-e", "1+2")
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "3"))


if __name__ == "__main__":
    unittest.main()
