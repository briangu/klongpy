"""
The native runtime (native/) against the conformance suite and the golden parse trees.

Skipped unless the `klong` binary is built:

    cd native && cargo build --release

The binary is `native/target/release/kg` (`klong` while a build has only that name), or the path in the environment variable KLONG_NATIVE.
KLONG_NATIVE_TAGS overrides the tags the binary declares (for a build with the features
`int32,float32`: core,strings,dicts,unicode,sysio,strict1,strict2,numpy-only,device).
"""
import importlib.util
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _native_binary():
    """The native binary, `kg`; KLONG_NATIVE overrides."""
    given = os.environ.get("KLONG_NATIVE")
    if given:
        return given
    return os.path.join(ROOT, "native", "target", "release", "kg")
CONF = os.path.join(ROOT, "conformance")

BINARY = _native_binary()
# What the native runtime declares in the default build (64-bit integers, f64). Not declared:
# bigint (integers wrap at the build width), torch-only and kgpy-only (KlongPy's own), and the
# tiers timers, ipc and desktop.
TAGS = set(
    (os.environ.get("KLONG_NATIVE_TAGS")
     or "core,strings,dicts,unicode,sysio,float,string-fmt,width64,strict1,strict2,numpy-only,device,static").split(",")
)
JOBS = int(os.environ.get("KLONG_NATIVE_JOBS") or (os.cpu_count() or 2))
BUILT = os.path.isfile(BINARY) and os.access(BINARY, os.X_OK)


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(BUILT, f"native binary not built: {BINARY}")
class TestNativeConformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conf = _load("conformance_run", os.path.join(CONF, "run.py"))

    def test_cases(self):
        conf = self.conf
        executor = conf.External([BINARY, "--conformance"])
        failures = []
        ran = 0
        for path in conf.case_files(conf.CASES_DIR):
            cases = conf.selected(conf.load_file(path), TAGS)
            ran += len(cases)
            for case, _, problem in conf.run_cases(cases, executor, JOBS):
                if problem:
                    failures.append(f"{case.where} {case.name}\n    " + problem.replace("\n", "\n    "))
        self.assertGreater(ran, 0, "no case selected by the tags")
        self.assertFalse(failures, f"{len(failures)} of {ran} cases failed:\n" + "\n".join(failures))

    def test_golden_trees(self):
        conf = self.conf
        failures = []
        count = 0
        for name, src, want in conf.golden_pairs():
            count += 1
            problem = conf.run_tree([BINARY], src, want)
            if problem:
                failures.append(f"{name}: {problem}")
        self.assertGreater(count, 0)
        self.assertFalse(failures, f"{len(failures)} of {count} trees differ:\n" + "\n".join(failures))

    def test_limits_are_errors(self):
        """Recursion without end, a heap limit and a step limit end as Klong errors, not as a crash."""
        import subprocess

        def run(args, source):
            p = subprocess.run([BINARY, *args, "--conformance"], input=source.encode(), capture_output=True, timeout=60)
            return p.returncode, p.stdout.decode().strip()

        self.assertEqual(run([], "f::{f(x+1)};f(0)"), (1, "error: range"))
        self.assertEqual(run(["--heap-limit", "1M"], "#!10000000"), (1, "error: memory"))
        self.assertEqual(run(["--heap-limit", "1M"], "a::[];{a::a,x}'!1000000;#a"), (1, "error: memory"))
        self.assertEqual(run(["--steps", "100000"], "{1}{x+1}:~0"), (1, "error: interrupt"))

    def test_nesting_and_small_stacks(self):
        """Data nested too deeply, and recursion on a small native stack, are range errors."""
        import subprocess

        def run(args, source):
            p = subprocess.run([BINARY, *args, "--conformance"], input=source.encode(), capture_output=True, timeout=60)
            return p.returncode, p.stdout.decode().strip()

        small = ["--stack", "48K", "--stack-budget", "48K"]
        quote = chr(39)
        self.assertEqual(run([], "200{,x}:*1"), (1, "error: range"))
        self.assertEqual(run([], "#100{,x}:*1"), (0, "1"))
        self.assertEqual(run([], 'd:::{};d,("self",,d);d'), (1, "error: range"))
        self.assertEqual(run([], 'd:::{};d,("self",,d);d~d'), (0, "1"))
        self.assertEqual(run(small, "f::{:[x;f(x-1);0]};f(100000)"), (1, "error: range"))
        self.assertEqual(run(small, "f::{:[x;f(x-1);0]};f(5)"), (0, "0"))
        self.assertEqual(run(small, "(" * 20000 + "1" + ")" * 20000), (1, "error: range"))
        self.assertEqual(run(small, "a::60{,x}:*1;a+a"), (1, "error: range"))
        # found by fuzzing: a chain of adverbs recursed without a check
        self.assertEqual(run(small, "(-1){x+1}\\*" + quote * 5000 + "0"), (1, "error: range"))
        self.assertEqual(run([], "1+" + quote * 600 + "2"), (1, "error: range"))

    def test_allocation_failures_are_errors(self):
        """Programs that fuzzing found to abort or hang when an allocation failed."""
        import subprocess

        quote = chr(39)
        programs = [
            '?"' + quote * 513,  # the text of a string token was copied with every token
            "[1 2 3]" + '"' * 560,  # an error message quoted the whole program
            "d:::{[1 2] " + '"' * 690,  # the token list did not end after a failed allocation
            "[a " + '"' * 80 + "(" * 600,
            'd:::{[1 2] ["a" "b"]};.fmt((2000)$1;(""):$(d,(1e10),,:{}))',  # .fmt quoted its template
            "f::{x-y}\ng::f::f(;1)\ng" + quote + "[5 6]",  # the list of argument layers of a call
            "f" + quote * 300 + "::{x+1}",  # the list of adverbs of a chain
        ]
        for source in programs:
            for n in (1, 2, 3, 4, 5, 7):
                p = subprocess.run(
                    [BINARY, "--memfs", "--heap-limit", "64K", "--steps", "200000", "--fail-every", str(n), "--conformance"],
                    input=source.encode(), capture_output=True, timeout=60)
                self.assertIn(p.returncode, (0, 1), f"--fail-every {n}: {source[:60]!r}: {p.stderr.decode()[-200:]}")
                self.assertNotIn("internal", p.stdout.decode())

    def test_files_in_memory(self):
        """The file functions through the Platform file interface (a file system in memory)."""
        import subprocess

        source = (
            '.tc(.oc("t.kg"));.p("sq::{x*x}");.p("v::sq(7)");.cc(.tc(0));.l("t");'
            '.fc(.ic("t.kg"));a::.rl();.cc(.fc(0));.df("t.kg");v,,a'
        )
        p = subprocess.run([BINARY, "--memfs", "--conformance"], input=source.encode(), capture_output=True, timeout=60)
        self.assertEqual((p.returncode, p.stdout.decode().strip()), (0, '[49 "sq::{x*x}"]'))
        p = subprocess.run([BINARY, "--memfs", "--conformance"], input=b'.l("nstat");nstat.mu([1 2 3 4])', capture_output=True, timeout=60)
        self.assertEqual((p.returncode, p.stdout.decode().strip()), (0, "2.5"))

    def test_library_and_reference_suites(self):
        """klongpy/lib loads and tests/kgtests/language passes through runner.kg (the long generated suites: native/scripts/suites.py)."""
        import glob
        import subprocess

        lib = os.path.join(ROOT, "klongpy", "lib")
        for path in sorted(glob.glob(os.path.join(lib, "*.kg"))):
            name = os.path.basename(path)
            p = subprocess.run([BINARY, "-e", f'.helpdb::[];.l("{name}");"loaded"'], capture_output=True, text=True, cwd=lib, timeout=120)
            self.assertEqual((p.returncode, p.stdout.strip().splitlines()[-1:]), (0, ['"loaded"']), f"{name}: {p.stderr[-300:]}")
        ran = 0
        for path in sorted(glob.glob(os.path.join(ROOT, "tests", "kgtests", "language", "test_*.kg"))):
            if os.path.basename(path) == "test_reassign.kg":  # uses .pyf (Python interop)
                continue
            rel = os.path.relpath(path, ROOT)
            p = subprocess.run([BINARY, "-e", f'fullpath::"{rel}";.l("tests/kgtests/runner.kg");err'],
                               capture_output=True, text=True, cwd=ROOT, timeout=300)
            ran += 1
            self.assertEqual(p.returncode, 0, f"{rel}: {p.stderr[-300:]}")
            self.assertNotIn("failed: ", p.stdout, rel)
            self.assertTrue(p.stdout.strip().endswith("0"), f"{rel}: {p.stdout[-200:]}")
        self.assertGreater(ran, 10)


# The serial console (a simulated board behind a pseudo-terminal, the upload tool,
# the cases through the console): the classes of native/tests/test_console.py, skipped unless
# native/target/release/klong-sim is built.
_console = _load("native_test_console", os.path.join(ROOT, "native", "tests", "test_console.py"))
TestConsoleCheckComplete = _console.TestCheckComplete
TestConsoleUploadTool = _console.TestUploadTool
TestConsoleConformance = _console.TestConformanceOverConsole


if __name__ == "__main__":
    unittest.main()
