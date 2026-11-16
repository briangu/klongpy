"""
The native runtime (native/, ticket 102) against the conformance suite and the golden parse trees.

Skipped unless the `klong` binary is built:

    cd native && cargo build --release

The binary is `native/target/release/klong`, or the path in the environment variable KLONG_NATIVE.
KLONG_NATIVE_TAGS overrides the tags the binary declares (for a build with the features
`int32,float32`: core,strict1,strict2,numpy-only,device).
"""
import importlib.util
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONF = os.path.join(ROOT, "conformance")

BINARY = os.environ.get("KLONG_NATIVE") or os.path.join(ROOT, "native", "target", "release", "klong")
# What the spike implements: the core tier in the default build (64-bit integers, f64).
TAGS = set(
    (os.environ.get("KLONG_NATIVE_TAGS") or "core,float,string-fmt,width64,strict1,strict2,numpy-only,device").split(",")
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


if __name__ == "__main__":
    unittest.main()
