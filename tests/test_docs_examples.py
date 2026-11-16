"""The ``?> `` transcript examples in README.md and docs/*.md must match what the interpreter prints.

Each document is checked by scripts/check_docs.py in a subprocess. Blocks that
need a package that is not installed (duckdb, pandas, aiohttp) are skipped by
the script, so the checks run in every environment.
"""
import glob
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "check_docs.py")
DOCS = [os.path.join(ROOT, "README.md")] + sorted(glob.glob(os.path.join(ROOT, "docs", "*.md")))


@unittest.skipUnless(os.path.exists(SCRIPT), "scripts/check_docs.py is not part of this source tree")
class TestDocsExamples(unittest.TestCase):
    def test_documents(self):
        for path in DOCS:
            with self.subTest(document=os.path.relpath(path, ROOT)):
                r = subprocess.run([sys.executable, SCRIPT, path], cwd=ROOT,
                                   capture_output=True, text=True, timeout=300)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
