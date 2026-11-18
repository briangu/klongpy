"""The ``?> `` transcript examples in README.md and docs/*.md must match what the interpreter prints.

Each document is checked by scripts/check_docs.py in a subprocess. Blocks that
need a package that is not installed (duckdb, pandas, aiohttp) are skipped by
the script, so the checks run in every environment.
"""
import glob
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "check_docs.py")
DOCS = ([os.path.join(ROOT, "README.md")] + sorted(glob.glob(os.path.join(ROOT, "docs", "*.md")))
        + sorted(glob.glob(os.path.join(ROOT, "docs", "native", "*.md"))))


@unittest.skipUnless(os.path.exists(SCRIPT), "scripts/check_docs.py is not part of this source tree")
class TestDocsExamples(unittest.TestCase):
    def test_documents(self):
        for path in DOCS:
            with self.subTest(document=os.path.relpath(path, ROOT)):
                r = subprocess.run([sys.executable, SCRIPT, path], cwd=ROOT,
                                   capture_output=True, text=True, timeout=300)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


STRICT_DOC = """<!-- check-docs: strict 2 -->
```kgpy
?> a
error: undefined: a
?> f::{x+1}
:monad
```

<!-- check-docs: continue -->
```kgpy
?> f(1)
2
?> a
error: undefined: a
```

```kgpy
?> a
:a
```
"""


@unittest.skipUnless(os.path.exists(SCRIPT), "scripts/check_docs.py is not part of this source tree")
class TestStrictDirective(unittest.TestCase):
    """``<!-- check-docs: strict N -->`` runs a block at that strict level and its continuations with it."""

    def run_doc(self, text):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "doc.md")
            with open(path, "w") as f:
                f.write(text)
            return subprocess.run([sys.executable, SCRIPT, path], cwd=ROOT, capture_output=True, text=True, timeout=120)

    def test_level_applies_to_block_and_continuation_only(self):
        r = self.run_doc(STRICT_DOC)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("5 examples checked, 0 mismatches", r.stdout)

    def test_without_the_directive_level_is_zero(self):
        r = self.run_doc(STRICT_DOC.replace("<!-- check-docs: strict 2 -->\n", ""))
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("MISMATCH", r.stdout)

    def test_a_continuation_cannot_change_the_level(self):
        text = STRICT_DOC.replace("check-docs: continue", "check-docs: continue; strict 1")
        r = self.run_doc(text)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("cannot change the strict level", r.stderr)

    def test_bad_level_is_rejected(self):
        r = self.run_doc(STRICT_DOC.replace("strict 2", "strict 3"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("strict needs a level", r.stderr)


RENDER_DOC = """<!-- check-docs: strict 2; render -->
```kgpy
?> lenght::3;lenth
1:11: name error: undefined: lenth
  lenght::3;lenth
            ^^^^^
  hint: did you mean lenght?
?> 1+1
2
```

```kgpy
?> 1+"a"
error: cannot apply + (Plus) to integer and string
```
"""


@unittest.skipUnless(os.path.exists(SCRIPT), "scripts/check_docs.py is not part of this source tree")
class TestRenderDirective(unittest.TestCase):
    """``<!-- check-docs: render -->`` shows an error with its position, source line, carets and hint."""

    def run_doc(self, text):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "doc.md")
            with open(path, "w") as f:
                f.write(text)
            return subprocess.run([sys.executable, SCRIPT, path], cwd=ROOT, capture_output=True, text=True, timeout=120)

    def test_rendered_error_and_plain_error(self):
        r = self.run_doc(RENDER_DOC)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("3 examples checked, 0 mismatches", r.stdout)

    def test_plain_form_does_not_match_a_rendered_expectation(self):
        r = self.run_doc(RENDER_DOC.replace("strict 2; render", "strict 2"))
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)


SHELL_DOC = """```console
$ cat > demo.kg <<'EOF'
> my_var::5
> EOF
$ kgpy lint demo.kg
demo.kg:1:1: W001 'my_var' reads as my _ var: _ is the Drop operator, names cannot contain underscores
0 errors, 1 warning in 1 file
$ kgpy -e '1+"a"'
<expr>:1:2: type error: cannot apply + (Plus) to integer and string
  1+"a"
   ^
```

<!-- check-docs: continue -->
```console
$ ls
demo.kg
$ echo done
done
```
"""


@unittest.skipUnless(os.path.exists(SCRIPT), "scripts/check_docs.py is not part of this source tree")
class TestShellBlocks(unittest.TestCase):
    """A ``console`` block of ``$ `` commands runs them in a temporary directory and compares stdout and stderr."""

    def run_doc(self, text):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "doc.md")
            with open(path, "w") as f:
                f.write(text)
            return subprocess.run([sys.executable, SCRIPT, path], cwd=ROOT, capture_output=True, text=True, timeout=120)

    def test_commands_output_and_shared_directory(self):
        r = self.run_doc(SHELL_DOC)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("5 examples checked, 0 mismatches", r.stdout)

    def test_a_different_output_is_a_mismatch(self):
        r = self.run_doc(SHELL_DOC.replace("0 errors, 1 warning", "0 errors, 2 warnings"))
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("MISMATCH", r.stdout)

    def test_a_block_without_continue_gets_a_new_directory(self):
        r = self.run_doc(SHELL_DOC.replace("<!-- check-docs: continue -->\n", ""))
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)

    def test_a_plain_block_that_starts_with_a_prompt_is_not_run(self):
        r = self.run_doc("```\n$ kgpy -e 'nothing that matches'\nwrong\n```\n")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("0 examples checked", r.stdout)


CMD_DOC = """<!-- check-docs: requires-cmd sh -->
```console
$ echo first
first
```

<!-- check-docs: continue; requires-cmd no-such-program-for-check-docs -->
```console
$ echo not run
this is not what it prints
```

<!-- check-docs: continue -->
```console
$ echo not run either
this is not what it prints
```
"""

LINK_DOC = """<!-- check-docs: link examples/embedded -->
```console
$ ls examples/embedded/mavg.kg
examples/embedded/mavg.kg
$ python -c 'import klongpy; print(klongpy.__name__)'
klongpy
```

<!-- check-docs: continue -->
```console
$ kgpy check examples/embedded/mavg.kg
examples/embedded/mavg.kg: inside the static profile (1 function, 1 entry, 0 externs, 2 globals)
```
"""


@unittest.skipUnless(os.path.exists(SCRIPT), "scripts/check_docs.py is not part of this source tree")
class TestShellDirectives(unittest.TestCase):
    """``requires-cmd`` skips a block when a program is missing; ``link`` shows repository paths to the commands."""

    def run_doc(self, text):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "doc.md")
            with open(path, "w") as f:
                f.write(text)
            return subprocess.run([sys.executable, SCRIPT, path], cwd=ROOT, capture_output=True, text=True, timeout=120)

    def test_missing_command_skips_the_block_and_its_continuations(self):
        r = self.run_doc(CMD_DOC)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("1 examples checked, 0 mismatches", r.stdout)
        self.assertIn("2 blocks skipped (needs no-such-program-for-check-docs)", r.stdout)

    def test_a_block_without_the_directive_runs(self):
        text = CMD_DOC.replace("; requires-cmd no-such-program-for-check-docs", "")
        text = text.replace("this is not what it prints", "wrong")
        r = self.run_doc(text)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertEqual(r.stdout.count("MISMATCH"), 2, r.stdout)

    @unittest.skipUnless(os.path.exists(os.path.join(ROOT, "examples", "embedded", "mavg.kg")), "examples are not part of this source tree")
    def test_link_and_python(self):
        r = self.run_doc(LINK_DOC)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("3 examples checked, 0 mismatches", r.stdout)

    @unittest.skipUnless(os.path.exists(os.path.join(ROOT, "examples", "embedded", "mavg.kg")), "examples are not part of this source tree")
    def test_without_link_the_path_is_missing(self):
        r = self.run_doc(LINK_DOC.replace("<!-- check-docs: link examples/embedded -->\n", ""))
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)

    def test_link_of_an_unknown_path_is_rejected(self):
        r = self.run_doc(LINK_DOC.replace("examples/embedded", "no/such/path"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("cannot link", r.stderr)

    def test_link_on_a_transcript_block_is_rejected(self):
        r = self.run_doc("<!-- check-docs: link examples -->\n```kgpy\n?> 1\n1\n```\n")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("console blocks only", r.stderr)


FILE_DOC = """<!-- check-docs: requires-file scripts/check_docs.py -->
```console
$ echo first
first
```

<!-- check-docs: continue -->
```console
$ echo second
second
```

<!-- check-docs: requires-file no/such/file/for/check-docs -->
```console
$ echo not run
this is not what it prints
```

<!-- check-docs: continue -->
```console
$ echo not run either
this is not what it prints
```
"""


@unittest.skipUnless(os.path.exists(SCRIPT), "scripts/check_docs.py is not part of this source tree")
class TestFileAndPathDirectives(unittest.TestCase):
    """``requires-file`` skips a block whose file is not there; ``path`` puts a directory of the repository on the PATH."""

    def run_doc(self, text):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "doc.md")
            with open(path, "w") as f:
                f.write(text)
            return subprocess.run([sys.executable, SCRIPT, path], cwd=ROOT, capture_output=True, text=True, timeout=120)

    def test_a_missing_file_skips_the_block_and_its_continuations(self):
        r = self.run_doc(FILE_DOC)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("2 examples checked, 0 mismatches", r.stdout)
        self.assertIn("2 blocks skipped (needs no/such/file/for/check-docs)", r.stdout)

    def test_a_file_that_is_there_lets_the_block_run(self):
        r = self.run_doc(FILE_DOC.replace("no/such/file/for/check-docs", "scripts/check_docs.py").replace("this is not what it prints", "wrong"))
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertEqual(r.stdout.count("MISMATCH"), 2, r.stdout)

    def test_the_directory_comes_first_on_the_path(self):
        text = "<!-- check-docs: path scripts -->\n```console\n$ echo $PATH | cut -d: -f1\n%s\n```\n" % os.path.join(ROOT, "scripts")
        r = self.run_doc(text)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("1 examples checked, 0 mismatches", r.stdout)

    def test_a_path_that_is_not_a_directory_is_rejected(self):
        r = self.run_doc("<!-- check-docs: path no/such/dir -->\n```console\n$ echo x\nx\n```\n")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("cannot put", r.stderr)

    def test_path_on_a_transcript_block_is_rejected(self):
        r = self.run_doc("<!-- check-docs: path scripts -->\n```kgpy\n?> 1\n1\n```\n")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("console blocks only", r.stderr)


if __name__ == "__main__":
    unittest.main()
