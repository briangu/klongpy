"""Readable errors in the REPL, `-e`, file runs and piped input."""
import asyncio
import contextlib
import importlib.util
import io
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import warnings

from klongpy import KlongInterpreter

if importlib.util.find_spec("colorama") is None:
    raise unittest.SkipTest("colorama not installed")

from klongpy import cli  # noqa: E402
from klongpy.errors import KlongInternalError, KlongSyntaxWarning, KlongTypeError  # noqa: E402
from klongpy.source import Source  # noqa: E402
from klongpy.repl import format_error, format_warning, use_color  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# A Python exception that is not a KlongError, raised from inside the interpreter: the
# driver patches KlongInterpreter.__call__ and then runs the real command line.
DRIVER = """
import sys
import klongpy.cli as cli
from klongpy.interpreter import KlongInterpreter

def boom(self, x):
    raise ZeroDivisionError("kaboom")

KlongInterpreter.__call__ = boom
sys.argv = ["kgpy"] + sys.argv[1:]
cli.main()
"""


def _default_sigint():
    """Run in the child before exec: a test run started in the background has SIGINT
    ignored, which the child would inherit and Python would then not turn into KeyboardInterrupt."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def run_cli(*args, stdin="", env=None, driver=None, timeout=60, cwd=None):
    environ = dict(os.environ, KGPY_HISTORY="", PYTHONPATH=ROOT)
    environ.pop("NO_COLOR", None)
    environ.update(env or {})
    cmd = [sys.executable, "-W", "ignore"]
    cmd += ["-c", driver] if driver else ["-m", "klongpy.cli"]
    return subprocess.run(cmd + list(args), input=stdin, capture_output=True, text=True,
                          cwd=cwd or ROOT, env=environ, timeout=timeout)


class Files(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def write(self, name, text):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path


class TestExpr(Files):
    def test_name_error_is_rendered(self):
        r = run_cli("-e", "foo(1)")
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stdout, "")
        self.assertEqual(r.stderr.splitlines(), [
            "<expr>:1:1: name error: undefined function: foo",
            "  foo(1)",
            "  ^^^^^^",
        ])

    def test_syntax_error_has_a_caret(self):
        r = run_cli("-e", "(1]")
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stderr.splitlines(), [
            "<expr>:1:3: syntax error: expected ')' but found ']'",
            "  (1]",
            "    ^",
        ])

    def test_no_traceback_without_the_flag(self):
        r = run_cli("-e", "1+\"a\"")
        self.assertNotIn("Traceback", r.stderr)
        self.assertNotIn("('", r.stderr)

    def test_traceback_flags(self):
        for flag in ("-v", "--traceback"):
            r = run_cli(flag, "-e", "1+\"a\"")
            self.assertEqual(r.returncode, 1)
            self.assertTrue(r.stderr.startswith("<expr>:1:2: type error:"), r.stderr)
            self.assertIn("Traceback (most recent call last)", r.stderr)
            self.assertIn("The above exception was the direct cause", r.stderr)  # __cause__ chain

    def test_value_goes_to_stdout_and_exit_is_zero(self):
        r = run_cli("-e", "1+2")
        self.assertEqual((r.returncode, r.stdout, r.stderr), (0, "3\n", ""))

    def test_exit_function_keeps_its_status(self):
        self.assertEqual(run_cli("-e", ".x(1)").returncode, 1)
        self.assertEqual(run_cli("-e", ".x(0)").returncode, 0)


class TestSystemContext(Files):
    """-e and a file run start the same system context (.system: the loops behind .timer, .srv and .cli) as -l and the REPL."""

    def serve(self):
        """A KlongPy IPC server in this process; returns its port."""
        from klongpy.repl import cleanup_repl, create_repl
        from tests.ipc_helpers import free_port
        port = free_port()
        klong, loops = create_repl()
        self.addCleanup(cleanup_repl, loops)
        klong(f".srv({port})")
        return port

    def test_expr_can_create_a_timer(self):
        r = run_cli("-e", 'cb::{1};th::.timer("t";0.2;cb);1', timeout=30)
        self.assertEqual((r.returncode, r.stdout, r.stderr), (0, "1\n", ""))

    def test_expr_can_cancel_its_timer(self):
        r = run_cli("-e", 'cb::{1};th::.timer("t";0.2;cb);.timerc(th)', timeout=30)
        self.assertEqual((r.returncode, r.stderr), (0, ""), r.stderr)

    def test_expr_can_connect_to_a_server(self):
        port = self.serve()
        r = run_cli("-e", f'c::.cli({port});c("1+2")', timeout=30)
        self.assertEqual((r.returncode, r.stdout, r.stderr), (0, "3\n", ""))

    def test_expr_has_the_host_variables_of_a_file_run(self):
        r = run_cli("-e", ".os.argv", "--", "a", "b")
        self.assertEqual((r.returncode, r.stdout), (0, '["a" "b"]\n'))

    def test_file_with_a_timer_ends_with_the_script(self):
        path = self.write("timer.kg", 'cb::{.p("tick");1}\nth::.timer("t";0.2;cb)\n.p("end")\n')
        r = run_cli(path, timeout=30)
        self.assertEqual((r.returncode, r.stderr), (0, ""))
        self.assertTrue(r.stdout.startswith("end\n"), r.stdout)

    def test_file_that_closes_its_connection_ends(self):
        port = self.serve()
        path = self.write("closes.kg", f'c::.cli({port})\n.p(c("1+2"))\n.clic(c)\n.p("done")\n')
        r = run_cli(path, timeout=30)
        self.assertEqual((r.returncode, r.stdout, r.stderr), (0, "3\ndone\n", ""))

    @unittest.skipIf(sys.platform == "win32", "needs SIGINT")
    def test_file_holding_a_connection_prints_and_keeps_running_until_interrupted(self):
        # the pub-sub client in docs/ipc_capabilities.md stays up to receive updates
        port = self.serve()
        path = self.write("holds.kg", f'c::.cli({port})\n.p(c("1+2"))\n.p("done")\n')
        environ = dict(os.environ, KGPY_HISTORY="", PYTHONPATH=ROOT)
        p = subprocess.Popen([sys.executable, "-W", "ignore", "-m", "klongpy.cli", path], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, cwd=ROOT, env=environ, preexec_fn=_default_sigint)
        try:
            self.assertEqual(p.stdout.readline(), "3\n")      # visible although stdout is a pipe
            self.assertEqual(p.stdout.readline(), "done\n")
            time.sleep(0.5)
            self.assertIsNone(p.poll())
            p.send_signal(signal.SIGINT)
            p.communicate(timeout=30)
            self.assertEqual(p.returncode, 130)
        finally:
            if p.poll() is None:
                p.kill()
            p.wait()


class TestFileRun(Files):
    def test_error_on_line_three(self):
        path = self.write("e.kg", '1\n2\n1+"a"\n')
        r = run_cli(path)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stdout, "")
        self.assertEqual(r.stderr.splitlines(), [
            f'{path}:3:2: type error: cannot apply + (Plus) to integer and string',
            '  1+"a"',
            '   ^',
        ])

    def test_stops_at_the_first_error(self):
        path = self.write("e.kg", '.p("before")\n1+"a"\n.p("after")\n')
        r = run_cli(path)
        self.assertIn("before", r.stdout)
        self.assertNotIn("after", r.stdout)
        self.assertEqual(r.returncode, 1)

    def test_runtime_error_inside_a_function_loaded_from_a_file(self):
        lib = self.write("lib.kg", "g::{[a];\n  a::x;\n  a@5}\n")
        main = self.write("main.kg", f'.l("{lib}")\n\ng([1 2])\n')
        r = run_cli(main)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stderr.splitlines(), [
            f"{lib}:3:4: range error: index 5 out of range for list of 2",
            "    a@5}",
            "     ^",
        ])

    def test_function_defined_and_called_in_one_file(self):
        path = self.write("f.kg", "n::[1 2]\n\ng::{[a];\n  a::x;\n  a@5}\n\ng(n)\n")
        r = run_cli(path)
        self.assertTrue(r.stderr.startswith(f"{path}:5:4: range error:"), r.stderr)

    def test_syntax_error_in_a_file(self):
        path = self.write("s.kg", "1\n2\n(3]\n")
        r = run_cli(path)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stderr.splitlines()[0], f"{path}:3:3: syntax error: expected ')' but found ']'")
        self.assertEqual(r.stderr.splitlines()[2], "    ^")

    def test_success_exits_zero_and_stays_quiet(self):
        path = self.write("ok.kg", '.p("hi")\n')
        r = run_cli(path)
        self.assertEqual((r.returncode, r.stdout, r.stderr), (0, "hi\n", ""))

    def test_exit_function(self):
        path = self.write("x.kg", ".x(1)\n")
        self.assertEqual(run_cli(path).returncode, 1)

    def test_missing_file_is_one_line(self):
        r = run_cli(os.path.join(self.dir, "nope.kg"))
        self.assertEqual(r.returncode, 1)
        self.assertTrue(r.stderr.startswith("system error: cannot open "), r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_traceback_flag_on_a_file_run(self):
        path = self.write("e.kg", '1+"a"\n')
        r = run_cli("-v", path)
        self.assertIn(f"{path}:1:2: type error", r.stderr)
        self.assertIn("Traceback", r.stderr)


class TestTestMode(Files):
    def test_failing_line_does_not_stop_the_run(self):
        path = self.write("t.kg", '.p("one")\n\n:" a comment\n1+"a"\n.p("three")\n')
        r = run_cli("-t", path)
        self.assertEqual(r.returncode, 1)
        self.assertIn('.p("three")', r.stdout)
        self.assertIn("three", r.stdout.replace('.p("three")', ""))
        self.assertNotIn("Welcome", r.stdout)  # the REPL is not started
        self.assertEqual(r.stderr.splitlines()[0], f"{path}:4:2: type error: cannot apply + (Plus) to integer and string")
        self.assertEqual(r.stderr.splitlines()[1:], ['  1+"a"', '   ^'])

    def test_all_lines_pass(self):
        path = self.write("t.kg", "1+2\n3*4\n")
        r = run_cli("-t", path)
        self.assertEqual(r.returncode, 0)
        self.assertNotIn("Welcome", r.stdout)
        self.assertEqual(r.stderr, "")

    def test_every_failing_line_is_reported(self):
        path = self.write("t.kg", '1+"a"\n2\n(1]\n')
        r = run_cli("-t", path)
        self.assertEqual(r.returncode, 1)
        self.assertIn(f"{path}:1:2: type error", r.stderr)
        self.assertIn(f"{path}:3:3: syntax error", r.stderr)


class TestLoad(Files):
    def test_failing_load_exits_without_the_repl(self):
        path = self.write("bad.kg", '1\n3+"a"\n')
        r = run_cli("-l", path, stdin="1+1\n")
        self.assertEqual(r.returncode, 1)
        self.assertNotIn("Welcome", r.stdout)
        self.assertTrue(r.stderr.startswith(f"{path}:2:2: type error"), r.stderr)

    def test_good_load_starts_the_repl(self):
        path = self.write("good.kg", "a::41\n")
        r = run_cli("-l", path, stdin="a+1\n")
        self.assertEqual(r.returncode, 0)
        self.assertIn("42", r.stdout)


class TestPipedStdin(Files):
    def test_errors_to_stderr_values_to_stdout(self):
        r = run_cli(stdin='1+2\nfoo(1)\n3*4\n')
        out = r.stdout.replace("?> ", "")
        self.assertIn("3", out.split())
        self.assertIn("12", out.split())  # the loop went on after the error
        self.assertEqual(r.stderr.splitlines(), [
            "1:1: name error: undefined function: foo",
            "  foo(1)",
            "  ^^^^^^",
        ])
        self.assertEqual(r.returncode, 1)

    def test_clean_input_exits_zero(self):
        r = run_cli(stdin="1+2\n")
        self.assertEqual((r.returncode, r.stderr), (0, ""))

    def test_multi_line_entry_reports_the_line_in_the_entry(self):
        r = run_cli(stdin=':[1=1;\n2+"a";\n3]\n')
        self.assertEqual(r.stderr.splitlines(), [
            "2:2: type error: cannot apply + (Plus) to integer and string",
            '  2+"a";',
            "   ^",
        ])

    def test_exit_function_wins(self):
        r = run_cli(stdin="foo(1)\n.x(0)\n")
        self.assertEqual(r.returncode, 0)

    def test_no_mode_prints_an_args_tuple(self):
        r = run_cli(stdin='foo(1)\n1+"a"\n]l nothing.kg\n')
        self.assertNotIn("('", r.stderr + r.stdout)
        self.assertNotIn("Traceback", r.stderr + r.stdout)

    def test_missing_file_for_load_command_is_one_line(self):
        r = run_cli(stdin="]l nothing.kg\n")
        self.assertIn("cannot load nothing.kg", r.stdout)
        self.assertEqual(r.stderr, "")


class TestPythonException(Files):
    """An exception that is not a KlongError is a bug: one line, and how to get the traceback."""

    def check(self, r, label):
        self.assertEqual(r.returncode, 1, label)
        self.assertEqual(r.stderr.splitlines(), [
            "internal error: ZeroDivisionError: kaboom",
            "  hint: run with -v for the Python traceback",
        ], label)

    def test_expr(self):
        self.check(run_cli("-e", "1", driver=DRIVER), "-e")

    def test_file(self):
        self.check(run_cli(self.write("a.kg", "1\n"), driver=DRIVER), "file")

    def test_test_mode(self):
        self.check(run_cli("-t", self.write("a.kg", "1\n"), driver=DRIVER), "-t")

    def test_repl(self):
        r = run_cli(stdin="1\n", driver=DRIVER)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stderr.splitlines()[0], "internal error: ZeroDivisionError: kaboom")

    def test_traceback_with_the_flag(self):
        for args in (("-v", "-e", "1"), ("--traceback", "-e", "1")):
            r = run_cli(*args, driver=DRIVER)
            self.assertEqual(r.returncode, 1)
            self.assertTrue(r.stderr.startswith("internal error: ZeroDivisionError: kaboom\n"), r.stderr)
            self.assertIn("Traceback (most recent call last)", r.stderr)
            self.assertIn("ZeroDivisionError: kaboom", r.stderr.split("Traceback", 1)[1])
            self.assertNotIn("run with -v", r.stderr)

    def test_repl_with_the_flag(self):
        r = run_cli("-v", stdin="1\n", driver=DRIVER)
        self.assertIn("Traceback (most recent call last)", r.stderr)


class TestColour(unittest.TestCase):
    def error_text(self, env, tty):
        script = 'import sys; sys.argv=["kgpy","-e","1+\\"a\\""]; import klongpy.cli as c; c.main()'
        environ = dict(os.environ, PYTHONPATH=ROOT, KGPY_HISTORY="")
        environ.pop("NO_COLOR", None)
        environ.update(env)
        if not tty:
            r = subprocess.run([sys.executable, "-W", "ignore", "-c", script], capture_output=True, text=True,
                               env=environ, cwd=ROOT, timeout=60)
            return r.stderr
        import pty
        master, slave = pty.openpty()
        try:
            subprocess.run([sys.executable, "-W", "ignore", "-c", script], stderr=slave, stdout=subprocess.DEVNULL,
                           env=environ, cwd=ROOT, timeout=60)
            os.set_blocking(master, False)
            return os.read(master, 65536).decode()
        finally:
            os.close(master)
            os.close(slave)

    def test_not_a_tty_is_plain(self):
        self.assertNotIn("\x1b[", self.error_text({}, tty=False))

    @unittest.skipUnless(importlib.util.find_spec("pty") and sys.platform != "win32", "needs a pty")
    def test_tty_is_coloured_on_the_first_line_only(self):
        text = self.error_text({}, tty=True)
        lines = text.replace("\r", "").split("\n")
        self.assertTrue(lines[0].startswith("\x1b[31m<expr>:1:2: type error"), repr(text))
        self.assertTrue(lines[0].endswith("\x1b[0m"))
        self.assertNotIn("\x1b[", "\n".join(lines[1:]))

    @unittest.skipUnless(importlib.util.find_spec("pty") and sys.platform != "win32", "needs a pty")
    def test_no_color_on_a_tty(self):
        self.assertNotIn("\x1b[", self.error_text({"NO_COLOR": "1"}, tty=True))

    def test_use_color_rules(self):
        class Tty(io.StringIO):
            def isatty(self):
                return True
        old = os.environ.pop("NO_COLOR", None)
        try:
            self.assertTrue(use_color(Tty()))
            self.assertFalse(use_color(io.StringIO()))
            os.environ["NO_COLOR"] = "1"
            self.assertFalse(use_color(Tty()))
            os.environ["NO_COLOR"] = ""   # an empty value does not count (no-color.org)
            self.assertTrue(use_color(Tty()))
        finally:
            os.environ.pop("NO_COLOR", None)
            if old is not None:
                os.environ["NO_COLOR"] = old


class Script:
    def __init__(self, *items):
        self.items = list(items)

    def __call__(self, prompt):
        if not self.items:
            raise EOFError
        return self.items.pop(0)


def evaluate(klong, source, verbose=False):
    return asyncio.run(cli.repl_eval(klong, source, verbose=verbose))


def repl(*lines, klong=None, evaluate=evaluate, verbose=False):
    """Run the injected loop; returns (stdout, stderr, klong, failures)."""
    klong = klong or KlongInterpreter()
    if evaluate is globals()["evaluate"]:
        evaluate = lambda k, source: globals()["evaluate"](k, source, verbose)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        failures = cli.repl_loop(klong, evaluate, Script(*lines), history=None, exit_state={"code": None},
                                 verbose=verbose)
    return out.getvalue(), err.getvalue(), klong, failures


class TestReplLoop(unittest.TestCase):
    def test_error_is_rendered_to_stderr(self):
        out, err, _, failures = repl('1+"a"', "1+1")
        self.assertEqual(err.splitlines(), [
            "1:2: type error: cannot apply + (Plus) to integer and string",
            '  1+"a"',
            "   ^",
        ])
        self.assertIn("2", out.split("\n"))
        self.assertNotIn("error", out)
        self.assertEqual(failures, 1)

    def test_hint_is_shown(self):
        _, err, *_ = repl("[1 2 3]+[1 2]")
        self.assertIn("shape error: cannot apply + (Plus) to list of 3 and list of 2", err)

    def test_multi_line_entry_error_on_the_second_line(self):
        _, err, *_ = repl(":[1=1;", '2+"a";', "3]")
        self.assertEqual(err.splitlines(), [
            "2:2: type error: cannot apply + (Plus) to integer and string",
            '  2+"a";',
            "   ^",
        ])

    def test_multi_line_function_body(self):
        _, err, *_ = repl("g::{[a];", " a::x;", " a@5}", "g([1 2])")
        self.assertEqual(err.splitlines(), [
            "3:3: range error: index 5 out of range for list of 2",
            "   a@5}",
            "    ^",
        ])

    def test_entries_are_numbered(self):
        seen = []

        def spy(klong, source):
            seen.append((source.name, source.text))
            return None
        repl("1", "", "]h +", "2+2", "a::[1", "2]", evaluate=spy)
        self.assertEqual(seen, [("<repl:1>", "1"), ("<repl:2>", "2+2"), ("<repl:3>", "a::[1\n2]")])

    def test_error_source_has_the_entry_name(self):
        k = KlongInterpreter()
        errors = []

        def spy(klong, source):
            try:
                return klong(source)
            except Exception as e:
                errors.append(e)
        repl("1", '2+"a"', klong=k, evaluate=spy)
        self.assertEqual(errors[0].source.name, "<repl:2>")
        self.assertEqual(errors[0].position(), ("<repl:2>", 1, 2))

    def test_python_exception_is_one_line_plus_hint(self):
        def boom(klong, source):
            raise TypeError("message")
        _, err, _, failures = repl("1", evaluate=boom)
        self.assertEqual(err.splitlines(), [
            "internal error: TypeError: message",
            "  hint: run with -v for the Python traceback",
        ])
        self.assertEqual(failures, 1)

    def test_python_exception_with_traceback(self):
        def boom(klong, source):
            raise TypeError("message")
        _, err, *_ = repl("1", evaluate=boom, verbose=True)
        self.assertTrue(err.startswith("internal error: TypeError: message\n"))
        self.assertIn("Traceback (most recent call last)", err)
        self.assertNotIn("run with -v", err)

    def test_klong_error_traceback_only_with_the_flag(self):
        _, err, *_ = repl('1+"a"')
        self.assertNotIn("Traceback", err)
        _, err, *_ = repl('1+"a"', verbose=True)
        self.assertIn("Traceback", err)
        self.assertTrue(err.startswith("1:2: type error"))

    def test_python_error_inside_an_operator_is_a_klong_internal_error(self):
        k = KlongInterpreter()

        def bad(x):
            raise AttributeError("nope")
        k["bad"] = bad
        _, err, *_ = repl("bad(1)", klong=k)
        self.assertNotIn("('", err)
        self.assertNotIn("Traceback", err)
        self.assertIn("error", err)

    def test_nothing_prints_an_args_tuple(self):
        _, err, *_ = repl("foo(1)", "[1 2 3]+[1 2]", "1+", "(1]", "]l /no/such/file.kg", "]")
        self.assertNotIn("('", err)
        self.assertNotIn("Traceback", err)

    def test_sys_command_error_is_rendered(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.kg")
            with open(path, "w") as f:
                f.write('1\n3+"a"\n')
            _, err, *_ = repl(f"]l {path}")
        self.assertTrue(err.startswith(f"{path}:2:2: type error"), err)

    def test_it_is_not_set_by_an_error(self):
        _, _, k, _ = repl("7", '1+"a"')
        self.assertEqual(k["it"], 7)

    def test_failures_counted_and_returned(self):
        *_, failures = repl("1", "foo(1)", "1+", "2")
        self.assertEqual(failures, 2)


class TestInterrupt(unittest.TestCase):
    def test_keyboard_interrupt_during_evaluation_is_reported(self):
        k = KlongInterpreter()

        def stop(x):
            raise KeyboardInterrupt
        k["stop"] = stop
        out, err, *_ = repl("stop(1)", "1+1", klong=k)
        self.assertEqual(err, "interrupted\n")
        self.assertIn("2", out.split("\n"))

    @unittest.skipIf(sys.platform == "win32", "needs SIGINT")
    def test_ctrl_c_stops_a_long_evaluation_and_the_repl_goes_on(self):
        environ = dict(os.environ, KGPY_HISTORY="", PYTHONPATH=ROOT)
        p = subprocess.Popen([sys.executable, "-W", "ignore", "-m", "klongpy.cli"], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT, env=environ,
                             preexec_fn=_default_sigint)
        try:
            p.stdin.write('.p("ready")\n')
            p.stdin.flush()
            for line in p.stdout:     # the interpreter is up and reading
                if "ready" in line:
                    break
            p.stdin.write("{x+1}/1+!30000000\n")   # about 100 s if it ran to the end
            p.stdin.flush()
            time.sleep(1.5)
            self.assertIsNone(p.poll())
            started = time.time()
            p.send_signal(signal.SIGINT)
            p.stdin.write("1+1\n")
            p.stdin.flush()
            out, err = p.communicate(timeout=60)
        finally:
            p.kill()
            p.wait()
        self.assertLess(time.time() - started, 20)
        self.assertEqual(err.strip(), "interrupted")
        self.assertIn("2", out.replace("?> ", "").split())


    @unittest.skipIf(sys.platform == "win32", "needs SIGINT")
    def test_file_and_expr_runs_end_with_status_130(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "long.kg")
            with open(path, "w") as f:
                f.write("{x+1}/1+!30000000\n")
            for args in (["-e", "{x+1}/1+!30000000"], [path]):
                environ = dict(os.environ, KGPY_HISTORY="", PYTHONPATH=ROOT)
                p = subprocess.Popen([sys.executable, "-W", "ignore", "-m", "klongpy.cli"] + args,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT, env=environ,
                                     preexec_fn=_default_sigint)
                try:
                    time.sleep(3.0)
                    self.assertIsNone(p.poll())
                    p.send_signal(signal.SIGINT)
                    out, err = p.communicate(timeout=30)
                finally:
                    p.kill()
                    p.wait()
                self.assertEqual((p.returncode, err), (130, "interrupted\n"), args)


class KlongTestWarning(UserWarning):
    """Stands in for the parser's warning category: any Warning named Klong* is shown in the short form."""

    def __init__(self, message, span=None, source=None, hint=None):
        super().__init__(message)
        self.span, self.source, self.hint = span, source, hint


class TestWarnings(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")   # the process-wide state the REPL must override
        self.addCleanup(warnings.resetwarnings)

    def klong_with(self, **kw):
        k = KlongInterpreter()

        def warn(x):
            warnings.warn(KlongTestWarning(**kw))
            return 7
        k["warn"] = warn
        return k

    def test_short_form_without_a_span(self):
        k = self.klong_with(message="unexpected input after the expression")
        out, err, *_ = repl("warn(1)", klong=k)
        self.assertEqual(err, "warning: unexpected input after the expression\n")
        self.assertIn("7", out.split("\n"))

    def test_with_a_span_it_has_position_source_line_and_caret(self):
        src = Source("<repl:1>", "1 2 3+4 5")
        k = self.klong_with(message="input left over", span=(6, 9), source=src, hint="did you mean ,?")
        _, err, *_ = repl("warn(1)", klong=k)
        self.assertEqual(err.splitlines(), [
            "1:7: warning: input left over",
            "  1 2 3+4 5",
            "        ^^^",
            "  hint: did you mean ,?",
        ])

    def test_klong_syntax_warning_uses_its_own_render(self):
        src = Source("<repl:1>", "1 2 3+4 5")
        k = KlongInterpreter()

        def warn(x):
            warnings.warn(KlongSyntaxWarning("input left over", span=(6, 9), source=src, hint="use ,", code="trailing-input"))
            return 7
        k["warn"] = warn
        out, err, _, failures = repl("warn(1)", klong=k)
        self.assertEqual(err.splitlines(), [
            "1:7: warning: input left over",
            "  1 2 3+4 5",
            "        ^^^",
            "  hint: use ,",
        ])
        self.assertEqual(err, KlongSyntaxWarning("input left over", span=(6, 9), source=src, hint="use ,").render() + "\n")
        self.assertEqual(failures, 0)

    def test_file_name_in_the_position(self):
        src = Source("prog.kg", "a::1\nb::2 3 q\n")
        text = format_warning(KlongTestWarning("stray input", span=(11, 12), source=src))
        self.assertEqual(text.splitlines()[0], "prog.kg:2:7: warning: stray input")

    def test_shown_every_time_not_once_per_location(self):
        k = self.klong_with(message="again")
        _, err, *_ = repl("warn(1)", "warn(2)", "warn(3)", klong=k)
        self.assertEqual(err.count("warning: again"), 3)

    def test_a_warning_is_not_a_failure(self):
        k = self.klong_with(message="w")
        *_, failures = repl("warn(1)", klong=k)
        self.assertEqual(failures, 0)

    def test_expr_run_shows_the_warning_and_exits_zero(self):
        script = textwrap.dedent("""
            import sys, warnings
            import klongpy.cli as cli
            from klongpy.interpreter import KlongInterpreter

            class KlongTestWarning(UserWarning):
                pass

            def call(self, x):
                warnings.warn(KlongTestWarning("heads up"))
                return 5
            KlongInterpreter.__call__ = call
            sys.argv = ["kgpy", "-e", "1"]
            cli.main()
        """)
        r = run_cli("-e", "1", driver=script)
        self.assertEqual((r.returncode, r.stdout, r.stderr), (0, "5\n", "warning: heads up\n"))


class TestFormat(unittest.TestCase):
    def test_klong_error_is_render(self):
        try:
            KlongInterpreter()(Source("x.kg", '\n1+"a"'))
        except KlongTypeError as e:
            self.assertEqual(format_error(e), e.render())
            self.assertEqual(format_error(e, color=True), e.render(color=True))
            self.assertTrue(format_error(e, color=True).startswith("\x1b[31mx.kg:2:2:"))

    def test_other_exception(self):
        self.assertEqual(format_error(KeyError("k")),
                         "internal error: KeyError: 'k'\n  hint: run with -v for the Python traceback")
        self.assertEqual(format_error(RuntimeError()).splitlines()[0], "internal error: RuntimeError")

    def test_internal_klong_error_gets_the_hint(self):
        text = format_error(KlongInternalError("cannot apply + (Plus) to integer and integer: AttributeError: x"))
        self.assertEqual(text.splitlines()[1], "  hint: run with -v for the Python traceback")

    def test_multi_line_message_is_cut_to_one_line(self):
        self.assertEqual(format_error(ValueError("a\nb")).splitlines()[0], "internal error: ValueError: a")


if __name__ == "__main__":
    unittest.main()
