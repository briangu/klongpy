"""What keeps a `kgpy FILE` run up, what ends it, and what a callback may do.

A file run stays up while an IPC server listens, a connection opened with .cli is
open, a web server (.web) listens or a websocket connection (.ws) is open; it ends when
none is left. A timer fires while the run is up for one of those
reasons and does not keep it up by itself. .x in a callback ends the
process. Every program runs as a subprocess with a timeout, and every process
the test starts is ended by its handle.
"""
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from tests.cli_subprocess import cli_command

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, PYTHONPATH=ROOT, PYTHONUNBUFFERED="1")
COMMAND = cli_command(os.environ.get("_KLONGPY_TEST_BACKEND", "numpy"), os.environ.get("_KLONGPY_TEST_DEVICE"))
WINDOWS = sys.platform == "win32"
posix = unittest.skipIf(WINDOWS, "needs POSIX signals")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def default_sigint():
    """In the child, before it runs: SIGINT is the default action again. A test run started in the background of a shell has SIGINT
    ignored (the shell does that to its background jobs, and children inherit it), and Python then installs no KeyboardInterrupt handler:
    a Ctrl-C test would wait for a process that never sees the signal."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)


class Proc:
    """A kgpy process: its output is collected on a pipe, it is ended by its handle."""

    def __init__(self, tc, source, *args, name="p.kg", mode="file"):
        self.tc = tc
        self.dir = tempfile.TemporaryDirectory()
        tc.addCleanup(self.dir.cleanup)
        path = os.path.join(self.dir.name, name)
        with open(path, "w") as f:
            f.write(source)
        cmd = [*COMMAND, *args, path]
        self.p = subprocess.Popen(cmd, cwd=self.dir.name, env=ENV, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  preexec_fn=None if WINDOWS else default_sigint)
        self.started = time.monotonic()
        tc.addCleanup(self.stop)

    def stop(self):
        if self.p.poll() is None:
            self.p.kill()
        self.p.wait()
        if self.p.stdout:
            self.p.stdout.close()

    def stays_up(self, seconds=1.5):
        """True when the process is still running after `seconds`."""
        try:
            self.p.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            return True
        return False

    def finish(self, timeout=30):
        """Wait for the process to end on its own; return (status, output). A hang fails the test."""
        try:
            out, _ = self.p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.p.kill()
            out, _ = self.p.communicate()
            self.tc.fail(f"the process did not end within {timeout} s; output:\n{out}")
        return self.p.returncode, out

    def end_with(self, sig, timeout=20):
        """Send `sig` and wait for the process to end. A signal that arrives while CPython runs a weakref or garbage-collection callback
        has its exception ignored there (a loaded machine hits it now and then), so an unanswered signal is sent again every 3 s;
        a process that never ends still fails the test."""
        end = time.monotonic() + timeout
        while True:
            self.p.send_signal(sig)
            try:
                self.p.wait(timeout=min(3, max(0.1, end - time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= end:
                    break
        return self.finish(max(0.1, end - time.monotonic()) if self.p.poll() is None else 1)


class TestWhatKeepsARunUp(unittest.TestCase):

    def test_a_file_with_nothing_active_ends_at_once(self):
        status, out = Proc(self, '.p("hi")\n').finish()
        self.assertEqual((status, out), (0, "hi\n"))

    def test_a_server_alone_keeps_the_run_up(self):
        p = Proc(self, f'.srv({free_port()})\n.p("up")\n')
        self.assertTrue(p.stays_up(), "a listening server must keep the run up")

    def test_a_timer_alone_does_not_keep_the_run_up(self):
        src = 'cb::{.p("tick");1}\n.timer("t";0.2;cb)\n.p("end")\n'
        status, out = Proc(self, src).finish()
        self.assertEqual(status, 0)
        self.assertTrue(out.startswith("end\n"), out)

    def test_a_connection_alone_keeps_the_run_up_until_it_closes(self):
        port = free_port()
        server = Proc(self, f'.srv({port})\n.p("serving")\n', name="server.kg")
        self.assertEqual(server.p.stdout.readline(), "serving\n")
        client = Proc(self, f'c::.cli({port})\n.p(c("1+1"))\n', name="client.kg")
        self.assertEqual(client.p.stdout.readline(), "2\n")
        self.assertTrue(client.stays_up(), "an open connection must keep the run up")
        server.end_with(signal.SIGTERM if not WINDOWS else signal.SIGINT)
        status, out = client.finish()          # the peer went away: nothing is left
        self.assertEqual(status, 0, out)

    def test_a_timer_that_stops_the_server_ends_the_run(self):
        src = f'.srv({free_port()})\ncb::{{.p("stopping");.srv(0);0}}\n.timer("t";0.3;cb)\n'
        status, out = Proc(self, src).finish()
        self.assertEqual((status, out), (0, "stopping\n"))

    def test_a_timer_ticks_while_a_server_keeps_the_run_up_and_stops_with_it(self):
        src = (f'.srv({free_port()})\nn::0\ncb::{{n::n+1;.p(n);:[n=4;.srv(0);1]}}\n'
               '.timer("t";0.1;cb)\n')
        status, out = Proc(self, src).finish()
        self.assertEqual((status, out), (0, "1\n2\n3\n4\n"))

    @posix
    def test_ctrl_c_ends_a_waiting_run_with_130(self):
        p = Proc(self, f'.srv({free_port()})\n.p("up")\n')
        self.assertEqual(p.p.stdout.readline(), "up\n")
        status, out = p.end_with(signal.SIGINT)
        self.assertEqual(status, 130)
        self.assertIn("interrupted", out)

    @posix
    def test_sigterm_ends_a_waiting_run_with_143(self):
        p = Proc(self, f'.srv({free_port()})\ncb::{{1}}\n.timer("t";0.1;cb)\n.p("up")\n')
        self.assertEqual(p.p.stdout.readline(), "up\n")
        status, out = p.end_with(signal.SIGTERM)
        self.assertEqual(status, 143)
        self.assertIn("terminated", out)


try:
    import aiohttp  # noqa: F401
    HAS_WEB = True
except ImportError:
    HAS_WEB = False
try:
    import websockets
    HAS_WS = True
except ImportError:
    HAS_WS = False


@unittest.skipUnless(HAS_WEB, "requires aiohttp")
class TestAWebServerKeepsARunUp(unittest.TestCase):

    def program(self, port, tail=""):
        return (f'.py("klongpy.web")\nindex(r)::{{"hi"}}\nh::.web({port};:{{;"/";index}};:{{}})\n'
                f'.p("ready")\n{tail}')

    def test_a_web_server_alone_keeps_the_run_up_and_serves(self):
        port = free_port()
        p = Proc(self, self.program(port), name="web.kg")
        self.assertEqual(p.p.stdout.readline(), "ready\n")
        self.assertTrue(p.stays_up(), "a listening web server must keep the run up")
        import urllib.request
        for _ in range(50):
            try:
                body = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read()
                break
            except OSError:
                time.sleep(0.1)
        self.assertEqual(body, b"hi")

    def test_a_server_whose_handle_is_dropped_still_keeps_the_run_up(self):
        src = f'.py("klongpy.web")\n.web({free_port()};:{{}};:{{}})\n.p("ready")\n'
        self.assertTrue(Proc(self, src).stays_up())

    def test_stopping_the_web_server_ends_the_run_cleanly(self):
        tail = 'cb::{.p("stopping");.webc(h);0}\n.timer("t";0.5;cb)\n'
        status, out = Proc(self, self.program(free_port(), tail), name="web.kg").finish()
        self.assertEqual((status, out), (0, "ready\nstopping\n"))

    @posix
    def test_sigterm_and_ctrl_c_end_a_web_run_without_a_warning(self):
        for sig, code, word in ((signal.SIGTERM, 143, "terminated"), (signal.SIGINT, 130, "interrupted")):
            with self.subTest(sig=sig):
                p = Proc(self, self.program(free_port()), name="web.kg")
                self.assertEqual(p.p.stdout.readline(), "ready\n")
                status, out = p.end_with(sig)
                self.assertEqual(status, code)
                self.assertIn(word, out)
                self.assertNotIn("WARNING", out)
                self.assertNotIn("Traceback", out)

    def test_a_timer_does_not_change_this(self):
        src = f'.py("klongpy.web")\ncb::{{1}}\n.timer("t";0.1;cb)\n.p("end")\n'
        self.assertEqual(Proc(self, src).finish(), (0, "end\n"))


@unittest.skipUnless(HAS_WS, "requires websockets")
class TestAWebsocketConnectionKeepsARunUp(unittest.TestCase):

    def serve(self, close_after=None):
        """A websocket server in this process, on its own thread; returns its port."""
        import asyncio
        ready = threading.Event()
        box = {}

        async def handler(ws):
            if close_after is not None:
                await asyncio.sleep(close_after)
                await ws.close()
                return
            async for _ in ws:
                pass

        async def main():
            box["loop"] = asyncio.get_running_loop()
            box["stop"] = asyncio.Event()
            server = await websockets.serve(handler, "127.0.0.1", 0)
            box["port"] = server.sockets[0].getsockname()[1]
            ready.set()
            await box["stop"].wait()
            server.close()
            await server.wait_closed()

        t = threading.Thread(target=lambda: asyncio.run(main()), daemon=True)
        t.start()
        self.assertTrue(ready.wait(10))

        def stop():
            box["loop"].call_soon_threadsafe(box["stop"].set)
            t.join(10)
        self.addCleanup(stop)
        return box["port"]

    def program(self, port, tail=""):
        return f'.py("klongpy.ws")\nw::.ws("ws://127.0.0.1:{port}")\n.p("connected")\n{tail}'

    def test_an_open_connection_keeps_the_run_up_until_it_is_closed(self):
        p = Proc(self, self.program(self.serve(), 'cb::{.p("closing");.wsc(w);0}\n.timer("t";1.5;cb)\n'), name="ws.kg")
        self.assertEqual(p.p.stdout.readline(), "connected\n")
        self.assertEqual(p.p.stdout.readline(), "closing\n")
        status, out = p.finish()
        self.assertEqual((status, out), (0, ""))

    def test_the_peer_closing_the_connection_ends_the_run(self):
        p = Proc(self, self.program(self.serve(close_after=1.0)), name="ws.kg")
        self.assertEqual(p.p.stdout.readline(), "connected\n")
        status, out = p.finish()
        self.assertEqual(status, 0, out)
        self.assertNotIn("WARNING", out)

    @posix
    def test_sigterm_ends_a_websocket_run_without_a_warning(self):
        p = Proc(self, self.program(self.serve()), name="ws.kg")
        self.assertEqual(p.p.stdout.readline(), "connected\n")
        self.assertTrue(p.stays_up(), "an open websocket connection must keep the run up")
        status, out = p.end_with(signal.SIGTERM)
        self.assertEqual(status, 143)
        self.assertNotIn("WARNING", out)


class TestCallbacksEndARun(unittest.TestCase):

    def test_exit_from_a_timer_callback(self):
        for code, status in ((0, 0), (1, 1), (7, 1)):
            with self.subTest(code=code):
                src = f'.srv({free_port()})\nn::0\ncb::{{n::n+1;.p(n);:[n=3;.x({code});1]}}\n.timer("t";0.1;cb)\n'
                st, out = Proc(self, src).finish()
                self.assertEqual((st, out), (status, "1\n2\n3\n"))

    def test_exit_output_is_flushed(self):
        src = f'.srv({free_port()})\ncb::{{.d("no newline at the end");.x(0)}}\n.timer("t";0.1;cb)\n'
        status, out = Proc(self, src).finish()
        self.assertEqual((status, out), (0, "no newline at the end"))

    def test_exit_from_an_ipc_callback(self):
        port = free_port()
        server = Proc(self, f'.srv({port})\nquit::{{x;.p("bye");.x(1)}}\n.p("serving")\n', name="server.kg")
        self.assertEqual(server.p.stdout.readline(), "serving\n")
        client = Proc(self, f'c::.cli({port})\na::.async(c;{{x}})\na(:quit,,0)\n', name="client.kg")
        client.finish()
        status, out = server.finish()
        self.assertEqual((status, out), (1, "bye\n"))

    def test_stopping_the_server_from_a_request_ends_the_run(self):
        port = free_port()
        server = Proc(self, f'.srv({port})\nstop::{{x;.srv(0)}}\n.p("serving")\n', name="server.kg")
        self.assertEqual(server.p.stdout.readline(), "serving\n")
        client = Proc(self, f'c::.cli({port})\na::.async(c;{{x}})\na(:stop,,0)\n', name="client.kg")
        client.finish()
        status, out = server.finish()
        self.assertEqual(status, 0, out)

    def test_the_last_client_leaving_lets_the_server_end_the_run(self):
        port = free_port()
        server = Proc(self, f'.srv({port})\n.srv.c::{{x;.srv(0)}}\n.p("serving")\n', name="server.kg")
        self.assertEqual(server.p.stdout.readline(), "serving\n")
        client = Proc(self, f'c::.cli({port})\n.p(c("1+1"))\n.clic(c)\n', name="client.kg")
        self.assertEqual(client.finish(), (0, "2\n"))
        status, out = server.finish()
        self.assertEqual(status, 0, out)

    def test_closing_the_last_connection_from_a_callback_ends_the_run(self):
        port = free_port()
        server = Proc(self, f'.srv({port})\n.p("serving")\n', name="server.kg")
        self.assertEqual(server.p.stdout.readline(), "serving\n")
        client = Proc(self, f'c::.cli({port})\ncb::{{.p("closing");.clic(c);0}}\n.timer("t";0.3;cb)\n', name="client.kg")
        self.assertEqual(client.finish(), (0, "closing\n"))


class TestTimerFailures(unittest.TestCase):

    def check(self, body, expect, pre=""):
        # a server keeps the run up while the timer ticks; a second timer stops it
        src = (f'{pre}\n.srv({free_port()})\nn::0\ncb::{{n::n+1;.p(n);:[n=2;{body};1]}}\n.timer("tick";0.1;cb)\n'
               'q::{.srv(0);0}\n.timer("q";0.8;q)\n')
        status, out = Proc(self, src, name="timer.kg").finish()
        self.assertEqual(status, 0, out)
        self.assertEqual(out.count("timer tick callback failed; cancelling timer"), 1, out)
        self.assertIn(expect, out)
        self.assertNotIn("Traceback", out)
        self.assertTrue(out.startswith("1\n2\n"), out)
        self.assertNotIn("3\n", out, "the timer must stop after the failure")

    def test_klong_error(self):
        self.check('1+"a"', "type error")

    def test_python_error(self):
        self.check('truediv(1;0)', "ZeroDivisionError", pre='.pyf("operator";"truediv")')

    def test_traceback_with_dash_v(self):
        src = (f'.srv({free_port()})\nn::0\ncb::{{n::n+1;1+"a"}}\n.timer("tick";0.1;cb)\n'
               'q::{.srv(0);0}\n.timer("q";0.8;q)\n')
        status, out = Proc(self, src, "-v").finish()
        self.assertIn("timer tick callback failed; cancelling timer", out)
        self.assertIn("Traceback", out)


class TestCallingYourOwnServer(unittest.TestCase):

    def test_a_synchronous_call_to_its_own_server_is_an_error_at_once(self):
        port = free_port()
        src = f'.srv({port})\nc::.cli({port})\n.p(c("1+1"))\n.p("not reached")\n'
        t0 = time.monotonic()
        status, out = Proc(self, src).finish(timeout=20)
        self.assertEqual(status, 1, out)
        self.assertLess(time.monotonic() - t0, 15, "the error must not wait for the call timeout")
        self.assertIn("system error", out)
        self.assertIn(".async", out)
        self.assertNotIn("not reached", out)

    def test_the_same_from_a_timer_callback_stops_the_timer_with_a_message(self):
        port = free_port()
        src = (f'.srv({port})\nc::.cli({port})\ncb::{{.p(c("1+1"));1}}\n.timer("ask";0.1;cb)\n'
               'q::{.srv(0);.clic(c);0}\n.timer("q";1;q)\n')
        status, out = Proc(self, src).finish(timeout=20)
        self.assertEqual(status, 0, out)
        self.assertEqual(out.count("timer ask callback failed; cancelling timer"), 1, out)
        self.assertIn(".async", out)

    def test_a_server_pushing_to_a_client_in_its_own_process_is_an_error_at_once(self):
        port = free_port()
        src = (f'.srv.o::{{.p(x("1+1"))}}\n.srv({port})\nc::.cli({port})\n.p("connected")\n'
               'q::{.srv(0);.clic(c);0}\n.timer("q";1;q)\n')
        t0 = time.monotonic()
        status, out = Proc(self, src).finish(timeout=20)
        self.assertEqual(status, 0, out)
        self.assertLess(time.monotonic() - t0, 15)
        self.assertIn("connected", out)
        self.assertIn(".async", out)

    def test_async_to_its_own_server_works(self):
        port = free_port()
        src = (f'.srv({port})\nc::.cli({port})\na::.async(c;{{.p(x)}})\na("1+1")\n'
               'q::{.srv(0);.clic(c);0}\n.timer("q";1;q)\n')
        status, out = Proc(self, src).finish(timeout=20)
        self.assertEqual((status, out), (0, "2\n"))

    def test_a_call_from_expression_mode_still_works(self):
        # -e evaluates outside the loop thread, so the server is answered
        port = free_port()
        cmd = [*COMMAND, "-e", f'.srv({port});c::.cli({port});c("1+1")']
        r = subprocess.run(cmd, env=ENV, capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "2"), r.stderr)


if __name__ == "__main__":
    unittest.main()
