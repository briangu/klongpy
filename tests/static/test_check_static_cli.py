"""
`kgpy check FILE...` (the CLI of the static profile checker), the
package's independence from optional packages, and a run over the shipped
.kg files (the checker must never fail on a program, whatever it holds).
"""
import glob
import io
import os
import subprocess
import sys
import tempfile
import unittest

from klongpy.static import check_file, check_main

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
EMBEDDED = os.path.join(ROOT, "examples", "embedded")


def run_main(*files, **kw):
    out, err = io.StringIO(), io.StringIO()
    rc = check_main(list(files), stdout=out, stderr=err, **kw)
    return rc, out.getvalue(), err.getvalue()


class TestCheckMain(unittest.TestCase):
    def test_a_program_inside_the_profile(self):
        path = os.path.join(EMBEDDED, "pid.kg")
        rc, out, err = run_main(path)
        self.assertEqual((rc, err), (0, ""))
        self.assertEqual(out, f"{path}: inside the static profile (2 functions, 1 entry, 2 externs, 8 globals)\n")

    def test_globals_computed_on_the_host_are_named(self):
        rc, out, _ = run_main(os.path.join(EMBEDDED, "fir.kg"))
        self.assertEqual(rc, 0)
        self.assertIn("3 globals (raw, taps, idx) are computed on the host", out)

    def test_a_program_outside_the_profile_is_rendered_like_any_error(self):
        path = os.path.join(HERE, "reject", "const-count.kg")
        rc, out, err = run_main(path)
        self.assertEqual((rc, out), (1, ""))
        self.assertEqual(err.split("\n")[:4], [
            f"{path}:3:5: static error: the count of n#a (Take) is not a compile-time constant: "
            "the result length would depend on run-time data [const-count]",
            "  f::{y#x}",
            "      ^^^",
            "  hint: the count must be a literal, #a or ^a, or a global that no function assigns",
        ])
        self.assertTrue(err.endswith("1 static error in 1 file\n"))

    def test_every_problem_of_every_file_and_a_summary(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a.kg")
            with open(a, "w") as f:
                f.write("f::{&x}\ng::{?x}\n.p(1)\n")
            rc, out, err = run_main(a, os.path.join(EMBEDDED, "mavg.kg"), os.path.join(HERE, "reject", "where.kg"))
            self.assertEqual(rc, 1)
            self.assertEqual(err.count("static error:"), 4)
            self.assertIn("4 static errors in 2 files", err)
            self.assertIn("mavg.kg: inside the static profile", out)

    def test_a_library_without_entries_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "lib.kg")
            with open(a, "w") as f:
                f.write("sq::{x*x}\n")
            rc, out, _ = run_main(a)
            self.assertEqual(rc, 0)
            self.assertIn("no .entry, so every function was checked with arguments of unknown type", out)

    def test_unreadable_file(self):
        rc, out, err = run_main("/no/such/file.kg", os.path.join(EMBEDDED, "mavg.kg"))
        self.assertEqual(rc, 2)
        self.assertIn("/no/such/file.kg: cannot read", err)
        self.assertIn("mavg.kg: inside the static profile", out)

    def test_verbose_lists_what_was_found(self):
        rc, out, _ = run_main(os.path.join(EMBEDDED, "pid.kg"), verbose=True)
        self.assertEqual(rc, 0)
        for line in ("  entry   step: float -> float", "  extern  board.adc: int -> float (C read_adc)",
                     "  global  kp: float (const)", "  global  integ: float (state)", "  fn      step(float) -> float"):
            self.assertIn(line + "\n", out)

    def test_int64(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "big.kg")
            with open(a, "w") as f:
                f.write("big::3000000000\n")
            self.assertEqual(run_main(a)[0], 1)
            self.assertEqual(run_main(a, int_bits=64)[0], 0)


class TestCli(unittest.TestCase):
    """By subprocess. stdin is closed, so a REPL would end at once."""

    def kgpy(self, *args):
        p = subprocess.run([sys.executable, "-W", "ignore", "-m", "klongpy.cli", *args], cwd=ROOT, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=120)
        return p.returncode, p.stdout, p.stderr

    def test_check_static(self):
        rc, out, err = self.kgpy("check", "examples/embedded/mavg.kg", "examples/embedded/pid.kg", "examples/embedded/fir.kg")
        self.assertEqual((rc, err), (0, ""))
        self.assertEqual(out.count("inside the static profile"), 3)

    def test_check_static_failure_and_flags(self):
        rc, out, err = self.kgpy("check", "tests/static/reject/where.kg")
        self.assertEqual(rc, 1)
        self.assertIn("tests/static/reject/where.kg:3:5: static error: ", err)
        self.assertIn("[where]", err)
        rc, out, err = self.kgpy("check", "tests/static/reject/int-range.kg", "--int64", "-v")
        self.assertEqual(rc, 0)
        self.assertIn("  global  big: int (const)", out)

    def test_the_file_is_not_run(self):
        with tempfile.TemporaryDirectory() as d:
            mark = os.path.join(d, "mark")
            path = os.path.join(d, "p.kg")
            with open(path, "w") as f:
                f.write(f't::.sys("touch {mark}")\nf::{{x}}\n')
            rc, out, err = self.kgpy("check", path)
            self.assertEqual(rc, 0, err)
            self.assertFalse(os.path.exists(mark))


class TestNoOptionalPackages(unittest.TestCase):
    def test_the_package_never_imports_torch(self):
        for path in glob.glob(os.path.join(ROOT, "klongpy", "static", "*.py")) + [os.path.join(ROOT, "klongpy", "sys_fn_static.py")]:
            with open(path) as f:
                text = f.read()
            self.assertNotIn("import torch", text, path)
            self.assertNotIn("from torch", text, path)

    def test_it_works_with_every_optional_package_blocked(self):
        code = (
            "import sys\n"
            "for m in ('torch','pandas','duckdb','aiohttp','websockets','colorama','readline','prompt_toolkit'):\n"
            "    sys.modules[m] = None\n"
            "import klongpy.static as s\n"
            "r = s.check_file('examples/embedded/pid.kg')\n"
            "assert r.ok, r.render()\n"
            "k = s.load_program('examples/embedded/pid.kg', externs={'read_adc': lambda c: 0.3, 'write_pwm': lambda c, u: 0})\n"
            "assert abs(s.call_entry(k, 'step', 0.31) - 0.12005) < 1e-12\n"
            "print('ok')\n"
        )
        p = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual((p.returncode, p.stdout.strip()), (0, "ok"), p.stderr)


class TestShippedFiles(unittest.TestCase):
    """The checker reads anything without failing; these files are mostly outside the profile, which is not the point here."""

    def test_library_and_examples(self):
        files = sorted(glob.glob(os.path.join(ROOT, "klongpy", "lib", "*.kg"))
                       + glob.glob(os.path.join(ROOT, "examples", "**", "*.kg"), recursive=True))
        self.assertTrue(files)
        for path in files:
            with self.subTest(path=os.path.relpath(path, ROOT)):
                r = check_file(path)
                for d in r.diagnostics:
                    start, end = d.span
                    if end > len(d.source.text):
                        # engine needs: a StaticDiag does not say which source its span lies in (check_report's `sources` index);
                        # a diagnostic in a file the program loads has that file's offsets, so it can only be the loading kind
                        self.assertIn(".l(", d.source.text)
                        continue
                    self.assertTrue(0 <= start <= end <= len(d.source.text))
                    d.render()

    def test_mutated_sources_never_make_the_checker_fail(self):
        import random
        from klongpy.static import check_source
        files = sorted(glob.glob(os.path.join(HERE, "accept", "*.kg")) + glob.glob(os.path.join(HERE, "reject", "*.kg"))
                       + glob.glob(os.path.join(EMBEDDED, "*.kg")))
        sources = []
        for path in files:
            with open(path) as f:
                sources.append(f.read())
        pieces = list("xyzfg01 2.5+-*%:#_@!&|<>=~^,'/\\()[]{};\"\n.$?") + [
            "::", ":[", ":|", ":~", ":*", ":'", ":=", ":^", ":+", ":%", "[;", ".entry(", ".extern(", '"int -> int"', "f::{", "{x}"]
        rng = random.Random(90)
        for _ in range(1500):
            s = list(rng.choice(sources))
            for _ in range(rng.randrange(1, 6)):
                i = rng.randrange(len(s) + 1)
                op = rng.random()
                if op < 0.4:
                    s[i:i] = [rng.choice(pieces)]
                elif s:
                    j = min(i, len(s) - 1)
                    if op < 0.7:
                        del s[j]
                    else:
                        s[j] = rng.choice(pieces)
            text = "".join(s)
            try:
                for d in check_source(text, "fuzz.kg", resolve=False).diagnostics:
                    d.render()
            except Exception as e:      # pragma: no cover
                self.fail(f"{type(e).__name__}: {e} on {text!r}")

    def test_test_suites(self):
        for path in sorted(glob.glob(os.path.join(ROOT, "tests", "kgtests", "language", "*.kg")))[:12]:
            with self.subTest(path=os.path.relpath(path, ROOT)):
                check_file(path)


if __name__ == "__main__":
    unittest.main()
