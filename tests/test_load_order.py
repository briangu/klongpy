"""One search order for ``.l``: run, ``kgpy lint``, ``kgpy check`` and ``kgpy compile c`` agree.

KLONGPATH directories (default ``.:lib``) from the working directory, then the directory of the program.
"""
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def write(base, name, text):
    path = os.path.join(base, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return path


class TestLoadOrder(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = os.path.realpath(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        env = dict(os.environ)
        env.pop("KLONGPATH", None)
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
        self.env = env
        self.wd = os.path.join(self.base, "wd")
        os.makedirs(self.wd)

    def kgpy(self, *args):
        r = subprocess.run([sys.executable, "-W", "ignore", "-m", "klongpy.cli", *args], cwd=self.wd, env=self.env,
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
        return r.returncode, r.stdout, r.stderr

    # -- run -----------------------------------------------------------------------------------------------
    def test_run(self):
        write(self.base, "prog/common.kg", "which::{x+1}\n")
        write(self.base, "prog/p.kg", '.l("common")\n.p(which(0))\n')
        rc, out, err = self.kgpy("../prog/p.kg")
        self.assertEqual((rc, out.strip()), (0, "1"), err)          # only beside the program
        write(self.wd, "common.kg", "which::{x+2}\n")
        rc, out, err = self.kgpy("../prog/p.kg")
        self.assertEqual((rc, out.strip()), (0, "2"), err)          # the working directory wins

    # -- lint ----------------------------------------------------------------------------------------------
    def test_lint(self):
        # the file beside the program takes one argument, the one in the working directory none
        write(self.base, "prog/common.kg", "k::{x+1}\n")
        write(self.base, "prog/p.kg", '.l("common")\n.p(k(1))\n')
        rc, out, err = self.kgpy("lint", "../prog/p.kg")
        self.assertEqual(rc, 0, out + err)                          # found beside the program
        write(self.wd, "common.kg", "k::{1}\n")
        rc, out, err = self.kgpy("lint", "../prog/p.kg")
        self.assertEqual(rc, 1, out + err)                          # the working directory wins: E002
        self.assertIn("E002", out)

    def test_lint_klongpath_before_program_directory(self):
        write(self.base, "prog/common.kg", "k::{x+1}\n")
        write(self.base, "kp/common.kg", "k::{1}\n")
        write(self.base, "prog/p.kg", '.l("common")\n.p(k(1))\n')
        self.env["KLONGPATH"] = os.path.join(self.base, "kp")
        rc, out, err = self.kgpy("lint", "../prog/p.kg")
        self.assertIn("E002", out)

    # -- check and compile ---------------------------------------------------------------------------------
    PROGRAM = '.l("common")\nf::{k(x)}\n.entry("f";"int -> int")\n'

    def test_check(self):
        write(self.base, "prog/q.kg", self.PROGRAM)
        write(self.base, "prog/common.kg", "k::{x+10}\n")
        rc, out, err = self.kgpy("check", "../prog/q.kg")
        self.assertEqual(rc, 0, out + err)                          # only beside the program
        os.remove(os.path.join(self.base, "prog/common.kg"))
        rc, out, err = self.kgpy("check", "../prog/q.kg")
        self.assertNotEqual(rc, 0)
        self.assertIn("cannot find the file", out + err)
        self.assertIn("then the directory of the program", out + err)
        # beside the program a file with an effect at top level; in the working directory a good one
        write(self.base, "prog/common.kg", "k::{x+10}\n.p(1)\n")
        rc, out, err = self.kgpy("check", "../prog/q.kg")
        self.assertNotEqual(rc, 0, out + err)
        write(self.wd, "common.kg", "k::{x+20}\n")
        rc, out, err = self.kgpy("check", "../prog/q.kg")
        self.assertEqual(rc, 0, out + err)                          # the working directory wins

    def test_compile_c(self):
        write(self.base, "prog/q.kg", self.PROGRAM)
        write(self.base, "prog/common.kg", "k::{x+10}\n")
        out1 = os.path.join(self.base, "out1")
        rc, out, err = self.kgpy("compile", "c", "../prog/q.kg", "-o", out1)
        self.assertEqual(rc, 0, out + err)
        with open(os.path.join(out1, "q.c")) as f:
            self.assertIn("kg_iadd(a0_x, 10)", f.read())            # only beside the program
        write(self.wd, "common.kg", "k::{x+20}\n")
        out2 = os.path.join(self.base, "out2")
        rc, out, err = self.kgpy("compile", "c", "../prog/q.kg", "-o", out2)
        self.assertEqual(rc, 0, out + err)
        with open(os.path.join(out2, "q.c")) as f:
            self.assertIn("kg_iadd(a0_x, 20)", f.read())            # the working directory wins

    def test_compile_does_not_change_directory(self):
        write(self.base, "prog/q.kg", self.PROGRAM)
        write(self.base, "prog/common.kg", "k::{x+10}\n")
        code = ("import os,sys; from klongpy.static.compile import compile_program;"
                "c=os.getcwd(); compile_program(sys.argv[1]); assert os.getcwd()==c")
        r = subprocess.run([sys.executable, "-W", "ignore", "-c", code, "../prog/q.kg"], cwd=self.wd, env=self.env,
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
