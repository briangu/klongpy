"""`.l` searches beside the calling file first, then KLONGPATH and the bundled library."""
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


class TestLoadNextToProgram(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = os.path.realpath(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        env = dict(os.environ)
        env.pop("KLONGPATH", None)
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
        self.env = env

    def kgpy(self, cwd, *args, stdin=None):
        r = subprocess.run([sys.executable, "-W", "ignore", "-m", "klongpy.cli", *args], cwd=cwd, env=self.env,
                           input=stdin, capture_output=True, text=True, timeout=60)
        return r.returncode, r.stdout, r.stderr

    def test_run_from_another_directory(self):
        write(self.base, "proj/common.kg", "inc::{x+1}\n")
        write(self.base, "proj/prog.kg", '.l("common")\n.p(inc(41))\n')
        elsewhere = os.path.join(self.base, "elsewhere")
        os.makedirs(elsewhere)
        for target in (os.path.join("..", "proj", "prog.kg"), os.path.join(self.base, "proj", "prog.kg")):
            rc, out, err = self.kgpy(elsewhere, target)
            self.assertEqual((rc, out.strip()), (0, "42"), err)

    def test_name_with_extension_and_subdirectory(self):
        write(self.base, "proj/lib2/util.kg", "twice::{2*x}\n")
        write(self.base, "proj/prog.kg", '.l("lib2/util.kg")\n.p(twice(4))\n')
        elsewhere = os.path.join(self.base, "elsewhere")
        os.makedirs(elsewhere)
        rc, out, err = self.kgpy(elsewhere, "../proj/prog.kg")
        self.assertEqual((rc, out.strip()), (0, "8"), err)

    def test_nested_load_uses_the_directory_of_the_loaded_file(self):
        write(self.base, "proj/a/outer.kg", '.l("inner")\nouter::{inner(x)+1}\n')
        write(self.base, "proj/a/inner.kg", "inner::{x*10}\n")
        write(self.base, "proj/prog.kg", '.l("a/outer.kg")\n.p(outer(4))\n')
        elsewhere = os.path.join(self.base, "elsewhere")
        os.makedirs(elsewhere)
        rc, out, err = self.kgpy(elsewhere, "../proj/prog.kg")
        self.assertEqual((rc, out.strip()), (0, "41"), err)

    def test_nested_load_does_not_look_in_the_main_program_directory(self):
        write(self.base, "proj/a/outer.kg", '.l("only_top")\n')
        write(self.base, "proj/only_top.kg", "t::1\n")
        write(self.base, "proj/prog.kg", '.l("a/outer.kg")\n')
        elsewhere = os.path.join(self.base, "elsewhere")
        os.makedirs(elsewhere)
        rc, out, err = self.kgpy(elsewhere, "../proj/prog.kg")
        self.assertNotEqual(rc, 0)
        self.assertIn("file does not exist: only_top", err)

    def test_program_directory_precedes_current_directory(self):
        write(self.base, "proj/common.kg", "which::{1}\n")
        write(self.base, "here/common.kg", "which::{2}\n")
        write(self.base, "proj/prog.kg", '.l("common")\n.p(which(0))\n')
        rc, out, err = self.kgpy(os.path.join(self.base, "here"), "../proj/prog.kg")
        self.assertEqual((rc, out.strip()), (0, "1"), err)

    def test_klongpath_is_the_fallback_after_the_program_directory(self):
        write(self.base, "proj/common.kg", "which::{1}\n")
        write(self.base, "kp/common.kg", "which::{3}\n")
        write(self.base, "proj/prog.kg", '.l("common")\n.p(which(0))\n')
        os.makedirs(os.path.join(self.base, "elsewhere"))
        self.env["KLONGPATH"] = os.path.join(self.base, "kp")
        rc, out, err = self.kgpy(os.path.join(self.base, "elsewhere"), "../proj/prog.kg")
        self.assertEqual((rc, out.strip()), (0, "1"), err)
        os.remove(os.path.join(self.base, "proj", "common.kg"))
        rc, out, err = self.kgpy(os.path.join(self.base, "elsewhere"), "../proj/prog.kg")
        self.assertEqual((rc, out.strip()), (0, "3"), err)

    def test_not_found_lists_the_places(self):
        write(self.base, "proj/prog.kg", '.l("nowhere")\n')
        os.makedirs(os.path.join(self.base, "elsewhere"))
        rc, out, err = self.kgpy(os.path.join(self.base, "elsewhere"), "../proj/prog.kg")
        self.assertNotEqual(rc, 0)
        self.assertIn("file does not exist: nowhere", err)
        self.assertIn(os.path.join("..", "proj", "nowhere"), err)
        self.assertIn("bundled library", err)

    def test_expression_has_no_program_directory(self):
        write(self.base, "common.kg", "inc::{x+1}\n")
        os.makedirs(os.path.join(self.base, "elsewhere"))
        rc, out, err = self.kgpy(os.path.join(self.base, "elsewhere"), "-e", '.l("common")')
        self.assertNotEqual(rc, 0)
        self.assertNotIn("directory of the file that calls .l", err)

    def test_load_option(self):
        write(self.base, "proj/common.kg", "inc::{x+1}\n")
        write(self.base, "proj/pre.kg", '.l("common")\n')
        os.makedirs(os.path.join(self.base, "elsewhere"))
        rc, out, err = self.kgpy(os.path.join(self.base, "elsewhere"), "-l", "../proj/pre.kg", stdin=".p(inc(1))\n")
        self.assertEqual(rc, 0, err)
        self.assertIn("2", out)


if __name__ == "__main__":
    unittest.main()
