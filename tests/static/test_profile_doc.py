"""
The profile document generator:
``python -m klongpy.static.profile_doc`` prints docs/static-profile.md, whose
operator section is generated from the operator table; ``--check`` fails when
the checked-in text differs from the generated text.
"""
import io
import os
import tempfile
import unittest

from klongpy.static import FORMS, OPTABLE, RULES
from klongpy.static.profile_doc import BEGIN, DEFAULT_PATH, END, check_text, generated_block, main, render_page

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    return main(list(argv), out, err), out.getvalue(), err.getvalue()


class TestProfileDoc(unittest.TestCase):
    def test_the_page_holds_the_generated_block_between_the_markers(self):
        page = render_page()
        self.assertEqual(page.count(BEGIN), 1)
        self.assertEqual(page.count(END), 1)
        self.assertIn(generated_block(), page)
        self.assertTrue(check_text(page))
        self.assertTrue(page.startswith("# The static profile\n"))

    def test_the_block_lists_every_row_form_and_rule(self):
        block = generated_block()
        for op in OPTABLE:
            if op.status == "allowed":
                self.assertIn(f"`{op.id}`", block)
            else:
                self.assertIn(f"| {op.name} | `{op.rule}` |", block)
        for form in FORMS:
            self.assertIn(f"`{form.id}`", block)
        for rule in RULES:
            self.assertIn(f"| `{rule}` |", block)

    def test_tables_are_well_formed_markdown(self):
        for line in generated_block().split("\n"):
            if line.startswith("|"):
                self.assertTrue(line.endswith("|"), line)
        tables = [t for t in generated_block().split("\n\n") if t.startswith("|")]
        self.assertTrue(tables)
        for t in tables:
            rows = t.strip().split("\n")
            widths = {r.replace("\\|", "").count("|") for r in rows}
            self.assertEqual(len(widths), 1, rows[0])

    def test_the_page_contains_the_hand_written_sections(self):
        page = render_page()
        for heading in ("## Two stages", "## Types", "## What a run-time function may contain", "## Typing rules",
                        "## Host functions and entry points", "## Differences from KlongPy", "## Reference programs"):
            self.assertIn(heading, page)
        self.assertIn("| Integer overflow |", page)             # the differences table
        self.assertIn("`(-7):%2` is -3", page)

    def test_check_is_0_when_fresh_1_after_a_hand_edit_2_when_missing(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "static-profile.md")
            self.assertEqual(run("--check", path)[0], 2)
            self.assertEqual(run("--write", path)[0], 0)
            with open(path) as f:
                text = f.read()
            self.assertEqual(text, render_page())
            self.assertEqual(run("--check", path)[0], 0)
            with open(path, "w") as f:
                f.write(text.replace("| Plus |", "| Add |"))            # inside the generated block
            rc, _, err = run("--check", path)
            self.assertEqual(rc, 1)
            self.assertIn("out of date", err)
            with open(path, "w") as f:
                f.write(text.replace("## Two stages", "## Stages"))     # outside it: hand-written text is free
            self.assertEqual(run("--check", path)[0], 0)

    def test_write_refreshes_the_block_and_keeps_the_text_around_it(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.md")
            with open(path, "w") as f:
                f.write(f"# Mine\n\n{BEGIN}\nstale\n{END}\n\nmy notes\n")
            self.assertEqual(run("--write", path)[0], 0)
            with open(path) as f:
                text = f.read()
            self.assertTrue(text.startswith("# Mine\n"))
            self.assertTrue(text.endswith("\nmy notes\n"))
            self.assertNotIn("stale", text)
            self.assertEqual(run("--check", path)[0], 0)

    def test_printing_the_page(self):
        rc, out, _ = run()
        self.assertEqual((rc, out), (0, render_page()))
        self.assertEqual(run("--bogus")[0], 2)

    def test_the_checked_in_page_is_up_to_date(self):
        path = os.path.join(ROOT, DEFAULT_PATH)
        if not os.path.isfile(path):
            self.skipTest("docs/static-profile.md is not checked in yet (the docs are another change)")
        with open(path) as f:
            self.assertTrue(check_text(f.read()),
                            "run: python -m klongpy.static.profile_doc --write docs/static-profile.md")


if __name__ == "__main__":
    unittest.main()
