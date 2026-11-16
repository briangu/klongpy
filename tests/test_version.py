import re
import unittest
from pathlib import Path

import klongpy
from klongpy import cli


class TestVersion(unittest.TestCase):
    def test_version_matches_pyproject(self):
        text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
        m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
        self.assertEqual(klongpy.__version__, m.group(1))

    def test_cli_falls_back_when_not_installed(self):
        import importlib.metadata as md
        orig = md.distribution
        def missing(name):
            raise md.PackageNotFoundError(name)
        md.distribution = missing
        try:
            self.assertEqual(cli._klongpy_version(), klongpy.__version__)
        finally:
            md.distribution = orig


if __name__ == "__main__":
    unittest.main()
