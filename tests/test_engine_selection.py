"""
Which engine KlongPy evaluates on: the native (Rust) engine by default, the Python engine with KLONGPY_ENGINE=python, and a clear
ImportError (no silent fallback) when the compiled extension cannot be found.
"""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROBE = "import klongpy; print(klongpy.KlongInterpreter.__name__)"


def run(code, **env):
    e = dict(os.environ)
    for k in ("KLONGPY_ENGINE", "KLONGPY_NATIVE_PATH"):
        e.pop(k, None)
    e.update({k: v for k, v in env.items() if v is not None})
    e["PYTHONPATH"] = ROOT + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    return subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=ROOT, env=e, capture_output=True, text=True, timeout=60)


def extension_directory():
    """The directory the running test session finds the extension in (KLONGPY_NATIVE_PATH), or None."""
    return os.environ.get("KLONGPY_NATIVE_PATH") or None


def extension_on_the_path():
    """True when the extension is importable without KLONGPY_NATIVE_PATH (installed, or on PYTHONPATH)."""
    r = run("import importlib.util as u, glob, os\n"
            "print(u.find_spec('klong_native') is not None or bool(glob.glob(os.path.join('klongpy', '_native*'))))")
    return r.stdout.strip() == "True"


class TestEngineSelection(unittest.TestCase):
    def test_the_python_engine_is_selected_by_the_environment(self):
        r = run(PROBE, KLONGPY_ENGINE="python")
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "KlongInterpreter"), r.stderr)

    def test_an_unknown_engine_is_an_error(self):
        r = run(PROBE, KLONGPY_ENGINE="fortran")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("KLONGPY_ENGINE", r.stderr)

    def test_the_native_engine_is_the_default_when_the_extension_is_found(self):
        d = extension_directory()
        if d is None and not extension_on_the_path():
            self.skipTest("the compiled extension is not available to this session")
        r = run(PROBE, KLONGPY_NATIVE_PATH=d)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "RustInterpreter"), r.stderr)

    def test_the_native_engine_can_be_named(self):
        d = extension_directory()
        if d is None and not extension_on_the_path():
            self.skipTest("the compiled extension is not available to this session")
        r = run(PROBE, KLONGPY_ENGINE="rust", KLONGPY_NATIVE_PATH=d)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "RustInterpreter"), r.stderr)

    def test_a_missing_extension_is_an_error_that_says_how_to_build_it(self):
        if extension_on_the_path():
            self.skipTest("the extension is installed in this environment")
        r = run(PROBE)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("ImportError", r.stderr)
        for word in ("compiled extension", "cargo build --release", "KLONGPY_NATIVE_PATH", "KLONGPY_ENGINE=python"):
            self.assertIn(word, r.stderr)
        self.assertNotIn("RustInterpreter", r.stdout)       # and no silent fallback to the Python engine
        self.assertNotIn("KlongInterpreter", r.stdout)

    def test_a_directory_without_the_extension_is_said_so(self):
        if extension_on_the_path():
            self.skipTest("the extension is installed in this environment")
        r = run(PROBE, KLONGPY_NATIVE_PATH=os.path.join(ROOT, "docs"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("which does not hold it", r.stderr)


if __name__ == "__main__":
    unittest.main()
