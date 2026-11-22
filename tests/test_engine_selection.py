"""
Which engine KlongPy evaluates on: the native (Rust) engine, and a clear ImportError (no silent fallback) when the compiled
extension cannot be found.
"""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROBE = "import klongpy; print(klongpy.KlongInterpreter.__name__)"
# Mask both supported import names in this subprocess, even inside an installed wheel.
MISSING_EXTENSION = "import sys; sys.modules.update({'klongpy._native': None, 'klong_native': None})\n"


def run(code, **env):
    e = dict(os.environ)
    for k in ("KLONGPY_NATIVE_PATH",):
        e.pop(k, None)
    e.update({k: v for k, v in env.items() if v is not None})
    e["PYTHONPATH"] = ROOT + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    return subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=ROOT, env=e, capture_output=True, text=True, timeout=60)


class TestEngineSelection(unittest.TestCase):
    def test_the_native_engine_is_the_interpreter_when_the_extension_is_found(self):
        r = run(PROBE, KLONGPY_NATIVE_PATH=os.environ.get("KLONGPY_NATIVE_PATH"))
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "KlongInterpreter"), r.stderr)

    def test_a_missing_extension_is_an_error_that_says_how_to_build_it(self):
        r = run(MISSING_EXTENSION + PROBE)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("ImportError", r.stderr)
        for word in ("compiled extension", "cargo build --release", "KLONGPY_NATIVE_PATH"):
            self.assertIn(word, r.stderr)
        self.assertNotIn("KlongInterpreter", r.stdout)       # and no silent fallback

    def test_a_directory_without_the_extension_is_said_so(self):
        r = run(MISSING_EXTENSION + PROBE, KLONGPY_NATIVE_PATH=os.path.join(ROOT, "docs"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("which does not hold it", r.stderr)


if __name__ == "__main__":
    unittest.main()
