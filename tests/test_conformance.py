"""
The implementation-neutral conformance suite (conformance/README.md, ticket 101).

* every case in conformance/cases runs against KlongPy in process, one test per file;
  a failure names the file and line of the case;
* the same cases run through the external-executable path with the reference shim
  (a sample by default; CONFORMANCE_SHIM_FULL=1 runs all of them);
* the golden parse trees run through `--dump-ast` of the shim;
* conformance/pending holds cases KlongPy does not satisfy yet: each must still fail;
* the case files themselves are checked: format, unique names, honest tags.
"""
import importlib.util
import os
import re
import sys
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONF = os.path.join(ROOT, "conformance")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


conf = _load("conformance_run", os.path.join(CONF, "run.py"))
SHIM = os.path.join(CONF, "shim", "kgpy_shim.py")

CASE_FILES = conf.case_files(conf.CASES_DIR)
PENDING_FILES = conf.case_files(conf.PENDING_DIR)
# The backend the in-process interpreter runs on (tests/conftest.py, --backend) and the tags
# KlongPy declares there. The shim is a separate process and always runs the numpy backend.
BACKEND = os.environ.get("_KLONGPY_TEST_BACKEND", "numpy")
TAGS = conf.klongpy_tags(BACKEND)
COMMENT = re.compile(r':"(?:[^"]|"")*"')
FLOAT_TEXT = re.compile(r"\d\.\d|\d[eE][+-]?\d")


def _failures(cases, executor, jobs=1):
    rows = conf.run_cases(cases, executor, jobs)
    return [f"{c.where} {c.name}\n    " + p.replace("\n", "\n    ") for c, _, p in rows if p]


class TestKlongPyAgainstSuite(unittest.TestCase):
    """One generated test per case file (see below)."""

    executor = None

    @classmethod
    def setUpClass(cls):
        cls.executor = conf.InProcess()

    def check_file(self, path):
        cases = conf.selected(conf.load_file(path), TAGS)
        failed = _failures(cases, self.executor)
        if failed:
            self.fail(f"{len(failed)} of {len(cases)} cases failed in {os.path.relpath(path, ROOT)}:\n"
                      + "\n".join(failed))


def _make(path):
    def test(self):
        self.check_file(path)
    return test


for _p in CASE_FILES:
    setattr(TestKlongPyAgainstSuite, "test_" + re.sub(r"\W", "_", os.path.splitext(os.path.basename(_p))[0]),
            _make(_p))


class TestSuiteFiles(unittest.TestCase):

    def test_there_are_cases(self):
        cases = conf.load_dir(conf.CASES_DIR)
        self.assertGreater(len(cases), 800)

    def test_names_unique_across_files(self):
        seen = {}
        for c in conf.load_dir(conf.CASES_DIR) + conf.load_dir(conf.PENDING_DIR):
            self.assertNotIn(c.name, seen, f"{c.where} repeats {c.name} of {seen.get(c.name)}")
            seen[c.name] = c.where

    def test_name_starts_with_file_area(self):
        for p in CASE_FILES:
            area = re.sub(r"^\d+-", "", os.path.splitext(os.path.basename(p))[0])
            for c in conf.load_file(p):
                self.assertTrue(c.name.startswith(area + "."), f"{c.where}: {c.name} is not in area {area}")

    def test_no_unfilled_cases(self):
        for c in conf.load_dir(conf.CASES_DIR) + conf.load_dir(conf.PENDING_DIR):
            self.assertFalse(c.unfilled, f"{c.where} {c.name} still has a ? line")

    def test_tags_are_honest(self):
        for c in conf.load_dir(conf.CASES_DIR) + conf.load_dir(conf.PENDING_DIR):
            src = COMMENT.sub("", c.source)
            kind, text = c.expect
            exp = text if kind == "value" else ""
            tiers = {t for t in c.tags if t in conf.TIERS}
            uses_strings = ('"' in src or "0c" in src or "$" in src or '"' in exp or "0c" in exp
                            or "0c" in (c.expect_out or ""))
            uses_dicts = bool(re.search(r"(?<!:):\{|:::\{", src)) or ":{" in exp
            uses_float = bool(FLOAT_TEXT.search(src) or FLOAT_TEXT.search(exp))
            if uses_strings:
                self.assertIn("strings", tiers, f"{c.where} {c.name} uses strings or chars")
            if uses_dicts:
                self.assertIn("dicts", tiers, f"{c.where} {c.name} uses a dictionary")
            if uses_float:
                self.assertIn("float", c.tags, f"{c.where} {c.name} uses floats")
            if "string-fmt" in c.tags:
                self.assertIn("float", c.tags, f"{c.where} {c.name}: string-fmt is about float text")

    def test_strict_tags_only_on_core_and_strings(self):
        for c in conf.load_dir(conf.CASES_DIR):
            if c.strict_level:
                self.assertTrue(set(c.tags) & {"strict1", "strict2"}, c.name)
                self.assertFalse({"strict1", "strict2"} <= set(c.tags), f"{c.where} has both strict tags")

    def test_pending_cases_say_why(self):
        for c in conf.load_dir(conf.PENDING_DIR):
            self.assertTrue(c.meta.get("reason"), f"{c.where} {c.name} has no : reason")
            self.assertIn("actual", c.meta, f"{c.where} {c.name} has no : actual")

    def test_every_error_case_uses_a_known_kind(self):
        for c in conf.load_dir(conf.CASES_DIR):
            if c.error:
                self.assertIn(c.error, conf.KINDS)

    def test_every_operator_is_exercised(self):
        """A coarse coverage check: each operator glyph appears in a case source, monadic and dyadic."""
        sources = "\n".join(COMMENT.sub("", c.source) for c in conf.load_dir(conf.CASES_DIR))
        for op in ("@", ":#", "!", "&", "*", "_", "$", ">", "<", "=", ",", "-", "~", "%", "?", "|", "^", "#", "+", ":_",
                   ":=", ":-", "::", ":%", ":$", ":@", ":^", ":+", ":#", ":*", ":~", ":'", ":/", ":\\", "@'", "\\*", "\\~"):
            self.assertIn(op, sources, f"no case uses {op}")


class TestFormat(unittest.TestCase):

    def parse(self, text):
        return conf.parse_text(text, "<test>")

    def test_example(self):
        (c,) = self.parse("# c\n@ a.b core float\n> 1+1.5\n>\n> 2\n= 2.5\n~ x\n")
        self.assertEqual((c.name, c.tags, c.source), ("a.b", ("core", "float"), "1+1.5\n\n2"))
        self.assertEqual((c.expect, c.expect_out), (("value", "2.5"), "x"))

    def test_multiline_result_and_empty_lines(self):
        (c,) = self.parse('@ a.b strings\n> "a"\n= "x\n=\n= y"\n')
        self.assertEqual(c.expect, ("value", '"x\n\ny"'))

    def test_error_case(self):
        (c,) = self.parse("@ a.b core\n> 1+\n! syntax\n")
        self.assertEqual(c.expect, ("error", "syntax"))

    def test_no_value_case(self):
        (c,) = self.parse("@ a.b core\n> :\"c\"\n")
        self.assertEqual(c.expect, ("none", None))

    def test_rejects_unknown_tag(self):
        with self.assertRaises(conf.FormatError):
            self.parse("@ a.b core nosuchtag\n> 1\n")

    def test_rejects_missing_tier(self):
        with self.assertRaises(conf.FormatError):
            self.parse("@ a.b float\n> 1\n")

    def test_rejects_core_with_other_tier(self):
        with self.assertRaises(conf.FormatError):
            self.parse("@ a.b core strings\n> 1\n")

    def test_rejects_unknown_kind_sigil_and_duplicates(self):
        for bad in ("@ a.b core\n> 1\n! nokind\n", "@ a.b core\n> 1\n% x\n", "@ a.b core\n> 1\n@ a.b core\n> 2\n",
                    "> 1\n", "@ a.b core\n= 1\n", "@ a.b core\n> 1\n= 1\n! type\n"):
            with self.assertRaises(conf.FormatError, msg=bad):
                self.parse(bad)

    def test_tag_selection(self):
        cs = self.parse("@ a.x core\n> 1\n@ a.y strings\n> 1\n@ a.z core bigint\n> 1\n")
        self.assertEqual([c.name for c in conf.selected(cs, {"core"})], ["a.x"])
        self.assertEqual([c.name for c in conf.selected(cs, {"core", "strings", "bigint"})], ["a.x", "a.y", "a.z"])

    def test_response_parsing(self):
        r = conf.parse_response(0, "[1 2]\n", "")
        self.assertEqual(r.as_expect(), ("value", "[1 2]"))
        self.assertEqual(conf.parse_response(0, "", "").as_expect(), ("none", None))
        self.assertEqual(conf.parse_response(1, "error: range\n", "").as_expect(), ("error", "range"))
        self.assertEqual(conf.parse_response(2, "", "boom").kind, "crash")
        self.assertEqual(conf.parse_response(1, "oops", "").kind, "crash")


class TestRunnerDetectsFailures(unittest.TestCase):
    """The runner must fail a wrong implementation, or the suite proves nothing."""

    def test_wrong_value_and_kind(self):
        (c1, c2) = conf.parse_text("@ a.b core\n> 1+1\n= 2\n@ a.c core\n> 1+\n! syntax\n", "<t>")

        class Stub:
            name = "stub"

            def run(self, case):
                return conf.Result("value", "3") if case is c1 else conf.Result("error", "type")
        rows = conf.run_cases([c1, c2], Stub())
        self.assertTrue(all(p for _, _, p in rows))
        self.assertIn("expected 2", rows[0][2])
        self.assertIn("error: type", rows[1][2])

    def test_console_output_is_checked(self):
        (c,) = conf.parse_text("@ a.b core\n> .p(1);0\n= 0\n~ 1\n", "<t>")
        self.assertIsNone(conf.check(c, conf.Result("value", "0", "1\n")))
        self.assertIsNotNone(conf.check(c, conf.Result("value", "0", "2\n")))

    def test_crashing_executable(self):
        (c,) = conf.parse_text("@ a.b core\n> 1\n= 1\n", "<t>")
        ex = conf.External([sys.executable, "-c", "import sys; sys.exit(3)"])
        (row,) = conf.run_cases([c], ex)
        self.assertIn("crashed", row[2])

    def test_timeout(self):
        (c,) = conf.parse_text("@ a.b core\n> 1\n= 1\n", "<t>")
        ex = conf.External([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)
        (row,) = conf.run_cases([c], ex)
        self.assertIn("timeout", row[2])


class TestShimProtocol(unittest.TestCase):
    """KlongPy behind the external protocol: the path a native executable takes."""

    def external(self):
        return conf.External([sys.executable, "-W", "ignore", SHIM], timeout=60)

    def test_cases_through_shim(self):
        cases = conf.selected(conf.load_dir(conf.CASES_DIR), conf.KLONGPY_TAGS)
        if os.environ.get("CONFORMANCE_SHIM_FULL"):
            sample = cases
        else:
            # every 7th case, plus every case that exercises a protocol feature: errors, console, strict, unicode
            sample = [c for i, c in enumerate(cases)
                      if i % 7 == 0 or c.expect_out is not None or c.strict_level or "unicode" in c.tags
                      or "device" in c.tags or c.expect[0] in ("error", "none")]
        failed = _failures(sample, self.external(), jobs=8)
        if failed:
            self.fail(f"{len(failed)} of {len(sample)} cases failed through the shim:\n" + "\n".join(failed))

    def test_in_process_and_shim_agree_on_pending(self):
        cases = conf.selected(conf.load_dir(conf.PENDING_DIR), conf.KLONGPY_TAGS)
        # the shim process runs the numpy backend: compare it with numpy in process on every --backend
        ex_ext, ex_in = self.external(), conf.InProcess(backend="numpy")
        for c in cases[:6]:
            a, b = ex_ext.run(c), ex_in.run(c)
            self.assertEqual((a.kind, a.text), (b.kind, b.text), c.name)

    def test_dump_ast_trees(self):
        pairs = list(conf.golden_pairs())
        self.assertGreater(len(pairs), 100)
        sample = [p for i, p in enumerate(pairs) if i % 9 == 0 or p[2].startswith("(error ")]
        bad = []
        for name, src, want in sample:
            problem = conf.run_tree([sys.executable, "-W", "ignore", SHIM], src, want, timeout=60)
            if problem:
                bad.append(f"{name}: {problem}")
        self.assertEqual(bad, [])


class TestPending(unittest.TestCase):

    def test_pending_cases_still_fail(self):
        ex = conf.InProcess()
        cases = conf.selected(conf.load_dir(conf.PENDING_DIR), TAGS)
        self.assertGreater(len(cases), 0)
        passing = [c for c in cases if conf.check(c, ex.run(c)) is None]
        if passing:
            names = ", ".join(f"{c.where} {c.name}" for c in passing)
            if os.environ.get("CONFORMANCE_STRICT_PENDING"):
                self.fail(f"pending cases that now pass, move them to cases/: {names}")
            warnings.warn(f"conformance: pending cases that now pass, move them to cases/: {names}")

    def test_actual_is_current(self):
        """`: actual` documents what KlongPy answers; when it moves, update it (or the case passes now)."""
        ex = conf.InProcess()
        if BACKEND != "numpy":
            self.skipTest("`: actual` records the answer of the numpy backend (the reference)")
        for c in conf.selected(conf.load_dir(conf.PENDING_DIR), conf.KLONGPY_TAGS):
            r = ex.run(c)
            got = ("error: " + r.text) if r.kind == "error" else (r.text or "(no value)")
            if conf.check(c, r) is None:
                continue
            if got != c.meta["actual"]:
                warnings.warn(f"conformance: {c.name} now gives {got!r}, the file says {c.meta['actual']!r}")


if __name__ == "__main__":
    unittest.main()
