"""examples/modules: the backtest in twelve module files prints what EXPECTED.txt says, and each program of errors/ stops with its error."""
import os
import re
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(ROOT, "examples", "modules")


def kgpy(*args):
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""), NO_COLOR="1")
    env.pop("KLONGPATH", None)
    return subprocess.run([sys.executable, "-W", "ignore", "-m", "klongpy.cli", *args], cwd=EXAMPLE, env=env, capture_output=True,
                          text=True, timeout=120)


class TestModulesExample(unittest.TestCase):
    def test_main_prints_the_expected_text(self):
        r = kgpy("main.kg")
        with open(os.path.join(EXAMPLE, "EXPECTED.txt")) as f:
            self.assertEqual((r.returncode, r.stdout), (0, f.read()), r.stderr)

    def test_each_error_program_stops_with_its_error(self):
        with open(os.path.join(EXAMPLE, "EXPECTED-ERRORS.txt")) as f:
            text = f.read()
        want = dict(re.findall(r"^(\w+)\.kg\n\s+(\S.*)$", text, re.M))
        self.assertGreaterEqual(len(want), 8)
        for name, first in want.items():
            with self.subTest(program=name):
                r = kgpy(f"errors/{name}.kg")
                self.assertEqual((r.returncode, r.stdout), (1, ""), r.stderr)
                kind = re.search(r"(\w+) error: ", first).group(1)
                self.assertIn(f"{kind} error: ", r.stderr.split("\n")[0])

    def test_classic_load_binds_the_module_to_the_file_name(self):
        r = kgpy("classic_load.kg")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.strip())


if __name__ == "__main__":
    unittest.main()
