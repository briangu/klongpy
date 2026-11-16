"""REPL loop driven by scripted input (tickets 070 and 071), no terminal."""
import asyncio
import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest

from klongpy import KlongInterpreter

if importlib.util.find_spec("colorama") is None:
    raise unittest.SkipTest("colorama not installed")

from klongpy import cli  # noqa: E402
from klongpy.repl_input import Completer, History, history_path  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


class Script:
    """A read_line replacement: items are strings, or exception classes/instances to raise."""

    def __init__(self, *items):
        self.items = list(items)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self.items:
            raise EOFError
        x = self.items.pop(0)
        if isinstance(x, BaseException) or (isinstance(x, type) and issubclass(x, BaseException)):
            raise x
        return x


def evaluate(klong, text):
    return asyncio.run(cli.repl_eval(klong, text, verbose=False))


def run(*items, klong=None, history=None, cancel_on_blank=True, evaluate=evaluate):
    klong = klong or KlongInterpreter()
    script = Script(*items)
    exit_state = {"code": None}
    out = io.StringIO()
    # errors go to stderr (ticket 042); one buffer keeps them in order with the values
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        cli.repl_loop(klong, evaluate, script, history=history, exit_state=exit_state,
                      cancel_on_blank=cancel_on_blank)
    text = out.getvalue()
    # colours are not under test
    for c in ("\x1b[32m", "\x1b[31m", "\x1b[0m"):
        text = text.replace(c, "")
    return text, script, exit_state, klong


class TestMultiLine(unittest.TestCase):
    def test_function_over_three_lines(self):
        out, script, _, _ = run("f::{", " x+1", "}", "f(2)")
        self.assertIn("3", out.split("\n"))
        self.assertEqual(script.prompts, [cli.PROMPT, cli.CONTINUATION_PROMPT, cli.CONTINUATION_PROMPT,
                                          cli.PROMPT, cli.PROMPT])

    def test_list_over_three_lines(self):
        out, *_ = run("arr::[1", "2", "3]", "arr")
        self.assertIn("[1 2 3]", out)

    def test_brackets_in_strings_and_comments_do_not_continue(self):
        for line in ('"a[b"', ':"a { b"', "0c["):
            _, script, *_ = run(line)
            self.assertEqual(script.prompts, [cli.PROMPT, cli.PROMPT], line)

    def test_trailing_dyad_is_evaluated_at_once(self):
        out, script, *_ = run("1+")
        self.assertEqual(script.prompts, [cli.PROMPT, cli.PROMPT])
        self.assertIn("syntax error", out)

    def test_mismatch_is_evaluated_at_once(self):
        out, script, *_ = run("(1]")
        self.assertEqual(script.prompts, [cli.PROMPT, cli.PROMPT])
        self.assertIn("syntax error", out)

    def test_blank_line_at_main_prompt_is_ignored(self):
        out, script, *_ = run("", "1+1")
        self.assertIn("2", out)
        self.assertEqual(len(script.prompts), 3)

    def test_empty_line_cancels_continuation(self):
        out, script, _, k = run("f::{", "", "1+1")
        self.assertEqual(script.prompts, [cli.PROMPT, cli.CONTINUATION_PROMPT, cli.PROMPT, cli.PROMPT])
        self.assertIn("2", out)
        self.assertNotIn("f", [x for x, _ in k._context])

    def test_ctrl_c_cancels_continuation(self):
        out, script, *_ = run("f::{", KeyboardInterrupt, "1+1")
        self.assertEqual(script.prompts, [cli.PROMPT, cli.CONTINUATION_PROMPT, cli.PROMPT, cli.PROMPT])
        self.assertIn("2", out)

    def test_ctrl_c_at_empty_main_prompt_does_not_quit(self):
        out, script, state, _ = run(KeyboardInterrupt, KeyboardInterrupt, "1+1")
        self.assertIn("2", out)
        self.assertIsNone(state["code"])
        self.assertEqual(out.count("bye"), 1)  # only the end of the script

    def test_ctrl_d_quits_at_main_prompt(self):
        out, script, state, _ = run("1+1")
        self.assertIn("bye!", out)
        self.assertEqual(len(script.prompts), 2)

    def test_ctrl_d_in_pending_entry_quits(self):
        out, script, *_ = run("f::{", EOFError, "1+1")
        self.assertIn("bye!", out)
        self.assertEqual(script.items, ["1+1"])

    def test_sys_command_only_at_primary_prompt(self):
        out, script, state, _ = run("]q")
        self.assertIn("bye!", out)
        self.assertEqual(state["code"], 0)
        # inside a pending entry a line starting with ] is entry text, not a command
        out, script, state, _ = run("[1", "]q")
        self.assertEqual(script.prompts, [cli.PROMPT, cli.CONTINUATION_PROMPT, cli.PROMPT])
        self.assertIsNone(state["code"])

    def test_blank_inside_entry_when_not_interactive(self):
        out, script, *_ = run("f::{", "", " x+1", "}", "f(2)", cancel_on_blank=False)
        self.assertIn("3", out.split("\n"))

    def test_error_in_evaluate_does_not_quit(self):
        def boom(klong, text):
            raise RuntimeError("boom")
        out, *_ = run("1", "2", evaluate=boom)
        self.assertEqual(out.count("internal error: RuntimeError: boom"), 2)

    def test_keyboard_interrupt_from_evaluate_propagates(self):
        def intr(klong, text):
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            run("1", evaluate=intr)


class FakeReadline:
    """The history half of the readline module, with 1-based items."""

    def __init__(self):
        self.items = []

    def add_history(self, s):
        self.items.append(s)

    def get_current_history_length(self):
        return len(self.items)

    def get_history_item(self, i):
        return self.items[i - 1]

    def remove_history_item(self, i):
        del self.items[i]


class RecordingScript(Script):
    """Behaves like input() with readline: each non-empty line typed lands in the history."""

    def __init__(self, rl, *items):
        super().__init__(*items)
        self.rl = rl

    def __call__(self, prompt):
        s = super().__call__(prompt)
        if s:
            self.rl.add_history(s)
        return s


class TestHistoryItem(unittest.TestCase):
    def test_multi_line_entry_is_one_item(self):
        rl = FakeReadline()
        hist = History(rl, None)
        script = RecordingScript(rl, "1+1", "f::{", " x+1", "}", "f(2)")
        with contextlib.redirect_stdout(io.StringIO()):
            cli.repl_loop(KlongInterpreter(), evaluate, script, history=hist)
        self.assertEqual(rl.items, ["1+1", "f::{\n x+1\n}", "f(2)"])

    def test_cancelled_entry_is_one_item(self):
        rl = FakeReadline()
        script = RecordingScript(rl, "f::{", " x", "")
        with contextlib.redirect_stdout(io.StringIO()):
            cli.repl_loop(KlongInterpreter(), evaluate, script, history=History(rl, None))
        self.assertEqual(rl.items, ["f::{\n x"])


class TestPiped(unittest.TestCase):
    def run_cli(self, stdin, *args):
        env = dict(os.environ, KGPY_HISTORY="", PYTHONPATH=ROOT)
        return subprocess.run([sys.executable, "-W", "ignore", "-m", "klongpy.cli", *args], input=stdin,
                              capture_output=True, text=True, cwd=ROOT, env=env, timeout=60)

    def test_piped_multi_line_definition(self):
        r = self.run_cli("f::{\n x+1\n}\nf(2)\narr::[1\n2\n3]\narr\n")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("3", r.stdout.replace("?> ", "").split())
        self.assertIn("[1 2 3]", r.stdout)
        self.assertNotIn("Error", r.stdout)

    def test_piped_blank_line_inside_function(self):
        r = self.run_cli("f::{\n\n x+1\n}\nf(2)\n")
        self.assertNotIn("Error", r.stdout)
        self.assertIn("3", r.stdout.replace("?> ", "").split())

    def test_piped_single_lines_as_before(self):
        r = self.run_cli("1+2\n\n3*4\n")
        out = r.stdout.replace("?> ", "")
        self.assertIn("3", out.split())
        self.assertIn("12", out.split())

    def test_piped_error_and_it(self):
        r = self.run_cli("1+2\nit*2\n")
        self.assertIn("6", r.stdout.replace("?> ", "").split())

    def test_expr_does_not_set_it(self):
        r = self.run_cli("", "-e", "1+2;it")
        self.assertEqual(r.stdout.strip(), ":it")


class TestIt(unittest.TestCase):
    def test_it_after_result(self):
        out, *_ = run("1+2", "it*2")
        self.assertIn("6", out.split("\n"))

    def test_it_assignment_then_overwritten(self):
        out, *_ = run("it::5", "1+1", "it")
        self.assertEqual(out.split("\n").count("2"), 2)

    def test_it_unchanged_after_error_and_command(self):
        _, _, _, k = run("7", "1+", "]h +", "(1]")
        self.assertEqual(k["it"], 7)

    def test_it_undefined_before_first_result(self):
        k = KlongInterpreter()
        self.assertEqual(str(k("it")), "it")

    def test_assignment_value_sets_it(self):
        _, _, _, k = run("a::3")
        self.assertEqual(k["it"], 3)


class TestCompleter(unittest.TestCase):
    def setUp(self):
        self.k = KlongInterpreter()
        self.k("alpha::1; alphabet::2; beta::3")
        self.c = Completer(self.k, cli.create_sys_cmd_functions())

    def test_system_functions(self):
        words = self.c.matches(".p")[1]
        for w in (".p", ".pc", ".py", ".pya", ".pyc", ".pyf"):
            self.assertIn(w, words)
        self.assertTrue(all(w.startswith(".p") for w in words))

    def test_defined_names(self):
        self.assertEqual(self.c.matches("alp"), (0, ["alpha", "alphabet"]))
        self.assertEqual(self.c.matches("1+be"), (2, ["beta"]))
        self.assertEqual(self.c.matches("f(alph"), (2, ["alpha", "alphabet"]))

    def test_no_prefix_no_names(self):
        self.assertEqual(self.c.matches("1+"), (2, []))

    def test_no_operator_glyphs(self):
        self.assertEqual(self.c.matches("zzz"), (0, []))

    def test_sys_commands(self):
        start, words = self.c.matches("]")
        self.assertEqual(start, 0)
        self.assertIn("]h", words)
        self.assertIn("]q", words)
        self.assertEqual(self.c.matches("]h")[1], ["]h"])
        self.assertEqual(self.c.matches("  ]t")[0], 2)

    def test_name_after_command(self):
        self.assertIn("alpha", self.c.matches("]h alp")[1])

    def test_files_after_load_command(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ("one.kg", "other.txt"):
                open(os.path.join(d, n), "w").close()
            os.mkdir(os.path.join(d, "sub"))
            start, words = self.c.matches("]l " + os.path.join(d, "o"))
            self.assertEqual(start, 3)
            self.assertEqual(words, [os.path.join(d, "one.kg"), os.path.join(d, "other.txt")])
            self.assertEqual(self.c.matches('.l("' + os.path.join(d, "s"))[1], [os.path.join(d, "sub") + "/"])

    def test_readline_protocol(self):
        got = []
        for state in range(5):
            w = self.c("alp", state, line="1+alp", begidx=2)
            if w is None:
                break
            got.append(w)
        self.assertEqual(got, ["alpha", "alphabet"])

    def test_protocol_when_word_starts_before_begidx(self):
        # readline split the word after "." (begidx inside the name): the candidate loses the head
        self.assertEqual(self.c("lpha", 0, line="alpha"[:0] + "alp", begidx=1), "lpha")

    def test_completion_does_not_evaluate(self):
        self.k("side::0")
        self.c.matches("side")
        self.assertEqual(self.k("side"), 0)


class TestHistoryFile(unittest.TestCase):
    def test_round_trip_with_newlines(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "h")
            rl = FakeReadline()
            items = ["1+1", "f::{\n x+1\n}", "a\\nb", 'x::"q"']
            rl.items = list(items)
            History(rl, path).save()
            rl2 = FakeReadline()
            History(rl2, path).load()
            self.assertEqual(rl2.items, items)

    def test_cap(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "h")
            rl = FakeReadline()
            rl.items = [str(i) for i in range(1500)]
            History(rl, path).save()
            rl2 = FakeReadline()
            History(rl2, path).load()
            self.assertEqual(len(rl2.items), 1000)
            self.assertEqual(rl2.items[-1], "1499")

    def test_env_override(self):
        self.assertEqual(history_path({"KGPY_HISTORY": "/x/y"}), "/x/y")
        self.assertIsNone(history_path({"KGPY_HISTORY": ""}))
        self.assertTrue(history_path({}).endswith(".kgpy_history"))

    def test_disabled_writes_nothing(self):
        rl = FakeReadline()
        rl.items = ["1"]
        History(rl, None).save()
        History(rl, None).load()

    def test_unwritable_directory_and_missing_file(self):
        rl = FakeReadline()
        rl.items = ["1"]
        History(rl, "/nonexistent-dir-kgpy/h").save()
        History(rl, "/nonexistent-dir-kgpy/h").load()
        self.assertEqual(rl.items, ["1"])

    @unittest.skipIf(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0), "needs a non-root POSIX user")
    def test_read_only_directory(self):
        with tempfile.TemporaryDirectory() as d:
            os.chmod(d, 0o500)
            try:
                rl = FakeReadline()
                rl.items = ["1"]
                History(rl, os.path.join(d, "h")).save()
            finally:
                os.chmod(d, 0o700)

    def test_no_readline(self):
        h = History(None, "/tmp/never")
        h.load()
        h.save()
        h.replace_entry(0, "x")
        self.assertEqual(h.mark(), 0)

    def test_setup_without_readline(self):
        saved = cli.readline
        cli.readline = None
        try:
            h = cli.setup_readline(KlongInterpreter(), {})
            h.load()
            h.save()
        finally:
            cli.readline = saved

    def test_cli_starts_without_readline_module(self):
        code = ("import sys; sys.modules['readline']=None; import importlib.util as u;"
                "_f=u.find_spec; u.find_spec=lambda n,*a: None if n=='readline' else _f(n,*a);"
                "sys.argv=['kgpy','-e','1+2']; from klongpy import cli; assert cli.readline is None; cli.main()")
        r = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True,
                           cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT), timeout=60)
        self.assertEqual(r.stdout.strip(), "3", r.stderr)


if __name__ == "__main__":
    unittest.main()
