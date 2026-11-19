"""Tests for the REPL ]-commands and the docstring-based help index."""
import importlib.util
import os
import tempfile
import unittest

from klongpy import KlongInterpreter

if importlib.util.find_spec("colorama") is None:
    raise unittest.SkipTest("colorama not installed")

from klongpy import cli, help_index  # noqa: E402


class TestSysCommands(unittest.TestCase):
    def setUp(self):
        self.klong = KlongInterpreter()
        self.cmds = cli.create_sys_cmd_functions()

    def run_cmd(self, line):
        r = cli.run_sys_cmd(self.klong, self.cmds, line)
        self.assertTrue(r is None or isinstance(r, (str, int)), line)
        return r if isinstance(r, str) else ""

    def test_bare_bracket_lists_commands(self):
        for line in ("]", "] "):
            out = self.run_cmd(line)
            self.assertIn("]h", out)
            self.assertIn("]q", out)

    def test_timeit_arguments(self):
        self.assertIn("usage", self.run_cmd("]T"))
        self.assertIn("total:", self.run_cmd("]T1+1"))
        self.assertIn("total:", self.run_cmd("]T 1+1"))
        self.assertIn("usage", self.run_cmd("]T:3"))
        self.assertIn("total:", self.run_cmd("]T:3 1+1"))
        self.assertIn("iteration count", self.run_cmd("]T:x 1"))
        self.assertIn("iteration count", self.run_cmd("]T:0 1"))

    def test_load_with_space_and_quote(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.kg")
            with open(path, "w") as f:
                f.write("loadedval::42\n")
            self.run_cmd(f"]l {path}")
            self.assertEqual(self.klong("loadedval"), 42)
            self.klong("loadedval::0")
            self.run_cmd(f"]l   {path}  ")
            self.assertEqual(self.klong("loadedval"), 42)
        self.assertIn("usage", self.run_cmd("]l"))
        # a quote in the name must not break out of a generated string
        self.assertIsInstance(self.run_cmd(']l a"b.kg'), str)

    def test_inventory_missing_dir(self):
        self.assertIn("no such directory", self.run_cmd("]i /missing"))

    def test_unknown_command_message(self):
        out = self.run_cmd("]X")
        self.assertIn("unknown system command: ]X", out)
        self.assertNotIn("unkown", out)

    def test_shell_without_argument(self):
        self.assertIn("usage", self.run_cmd("]!"))

    def test_help_arguments_do_not_raise(self):
        for topic in ("", "all", ".py", "/", ":>", "strict", '"', "adverbs", "]", ".p", "nosuchtopic"):
            self.assertTrue(self.run_cmd(f"]h {topic}"))

    def test_help_glyph_shows_adverb(self):
        out = self.run_cmd("]h /")
        self.assertIn("[Over]", out)
        self.assertNotIn("[Max/Or]", out)

    def test_help_extensions_and_system(self):
        self.assertIn("[Autograd]", self.run_cmd("]h :>"))
        self.assertIn("[Python-From]", self.run_cmd("]h .pyf"))
        self.assertIn("[Strict]", self.run_cmd("]h .strict"))
        self.assertIn("[Strict]", self.run_cmd("]h strict"))
        self.assertIn("[Each]", self.run_cmd("]h adverbs"))
        self.assertIn("[Conditional]", self.run_cmd("]h :["))

    def test_help_unknown_topic_says_so(self):
        self.assertIn("no help for", self.run_cmd("]h zzzqq"))

    def test_help_grade_up_is_ascending(self):
        out = self.run_cmd("]h <")
        self.assertIn("[Grade-Up]", out)
        self.assertIn('"<" = ascending', out)

    def test_apropos(self):
        out = self.run_cmd("]a scan")
        for title in ("Scan-Over", "Scan-Converging", "Scan-While", "Scan-Iterating"):
            self.assertIn(title, out)
        self.assertIn("usage", self.run_cmd("]a"))
        self.assertIn("nothing found", self.run_cmd("]a zzzqq"))

    def test_transcript(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.txt")
            self.assertIn("appended", self.run_cmd(f"]t {path}"))
            cli.transcript_write("\t1+1")
            cli.transcript_write("2")
            self.assertIn("stopped", self.run_cmd("]t"))
            cli.transcript_write("not recorded")
            with open(path) as f:
                self.assertEqual(f.read(), "\t1+1\n2\n")
            self.assertIn("cannot open", self.run_cmd(f"]t {d}"))
        self.assertIn("no transcript", self.run_cmd("]t"))


class TestHelpIndex(unittest.TestCase):
    def test_every_documented_eval_function_is_reachable(self):
        import inspect
        import re
        from klongpy import sys_fn, sys_var
        from klongpy.operator_help import operator_docs
        from klongpy.sysdocs import engine_docs
        header = re.compile(r"^.*?\s{2,}\[([^\]]+)\]\s*$")
        titles = {e.title for e in help_index.entries()}
        texts = []      # (name, help text): the operators' and the engine's system functions' are data, the others' a docstring
        for module, prefix in (("monads", "eval_monad_"), ("dyads", "eval_dyad_"), ("adverbs", "eval_adverb_")):
            texts.extend((n, d) for n, d in operator_docs().get(module, {}).items() if n.startswith(prefix))
        texts.extend(engine_docs().items())
        for mod, prefix in ((sys_fn, "eval_sys_"), (sys_var, "eval_sys_var_")):
            texts.extend((n, f.__doc__) for n, f in inspect.getmembers(mod, inspect.isfunction) if n.startswith(prefix))
        checked = 0
        for name, doc in texts:
            if not doc:
                continue
            first = next((ln for ln in doc.splitlines() if ln.strip()), "").strip()
            m = header.match(first)
            if m:
                checked += 1
                self.assertIn(m.group(1), titles, name)
                self.assertIn(f"[{m.group(1)}]", help_index.help_text(m.group(1)), name)
        self.assertGreater(checked, 100)

    def test_no_control_characters(self):
        import re
        for e in help_index.entries():
            self.assertIsNone(re.search(r"[\x00-\x08\x0b-\x1f]", e.full()), e.header)

    def test_op_db_grade_rows(self):
        path = os.path.join(os.path.dirname(help_index.__file__), "lib", "help.kg")
        with open(path) as f:
            text = f.read()
        self.assertNotIn("dimenions", text)
        down = text[text.index('"Grade-Down"'):]
        self.assertIn("descending", down.split("]")[0])
        up = text[text.index('"Grade-Up"'):]
        self.assertIn("ascending", up.split("]")[0])


if __name__ == "__main__":
    unittest.main()
