"""Ticket 019: IPC lifecycle (bind failures, shutdown, timeouts, per-interpreter server).

All tests use real sockets on localhost with dynamically chosen ports, and every
potentially blocking step runs under a bounded wait so that a regression fails
instead of hanging the suite.
"""
import asyncio
import os
import socket
import threading
import time
import unittest

from klongpy import KlongInterpreter
from klongpy.repl import create_repl, cleanup_repl
from klongpy.sys_fn_ipc import (NetworkClient, KlongIPCException,
                                eval_sys_fn_create_ipc_server)
from klongpy.core import KlongException
from klongpy.utils import CallbackEvent


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def bounded(fn, timeout=10.0):
    """Run fn in a daemon thread; fail (not hang) if it does not finish in time."""
    box = {}

    def run():
        try:
            box["r"] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised below
            box["e"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise AssertionError(f"did not finish within {timeout}s")
    if "e" in box:
        raise box["e"]
    return box.get("r")


class SilentServer:
    """Accepts connections and never replies. close_after: close each connection after n seconds."""

    def __init__(self, close_after=None):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(5)
        self.port = self.sock.getsockname()[1]
        self.close_after = close_after
        self.conns = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while True:
                c, _ = self.sock.accept()
                self.conns.append(c)
                if self.close_after is not None:
                    threading.Timer(self.close_after, c.close).start()
        except OSError:
            pass

    def stop(self):
        for c in self.conns:
            try:
                c.close()
            except OSError:
                pass
        self.sock.close()


class ReplCase(unittest.TestCase):
    def new_repl(self):
        klong, loops = create_repl()
        self.addCleanup(lambda: bounded(lambda: cleanup_repl(loops), 15))
        return klong, loops


class TestCallbackEvent(unittest.TestCase):
    def test_trigger_tolerates_self_unsubscribe(self):
        ev = CallbackEvent()
        seen = []

        def mk(n):
            def cb():
                seen.append(n)
                ev.unsubscribe(cb)
            return cb

        ev.subscribe(mk(1))
        ev.subscribe(mk(2))
        ev.trigger()
        self.assertEqual(sorted(seen), [1, 2])
        self.assertEqual(len(ev.subscribers), 0)


class TestShutdown(unittest.TestCase):
    def test_shutdown_with_two_clients_is_clean(self):
        for _ in range(20):
            klong, loops = create_repl()
            try:
                port = free_port()
                self.assertEqual(bounded(lambda: klong(f".srv({port})")), 1)
                bounded(lambda: klong(f"a::.cli({port})"))
                bounded(lambda: klong(f"b::.cli({port})"))
                ev = klong['.system']['closeEvent']
                bounded(ev.trigger, 20)
                # a second trigger after shutdown must not blow up either
                time.sleep(0.2)
                bounded(ev.trigger, 20)
            finally:
                bounded(lambda: cleanup_repl(loops), 15)


class TestServerStart(ReplCase):
    def test_srv_port_in_use_fails(self):
        klong, _ = self.new_repl()
        blocker = socket.socket()
        blocker.bind(("0.0.0.0", 0))
        blocker.listen(1)
        self.addCleanup(blocker.close)
        port = blocker.getsockname()[1]
        try:
            r = bounded(lambda: klong(f".srv({port})"))
        except (KlongException, KlongIPCException):
            return
        self.assertEqual(r, 0)

    def test_srv_can_restart_after_failed_bind(self):
        klong, _ = self.new_repl()
        blocker = socket.socket()
        blocker.bind(("0.0.0.0", 0))
        blocker.listen(1)
        busy = blocker.getsockname()[1]
        try:
            bounded(lambda: klong(f".srv({busy})"))
        except (KlongException, KlongIPCException):
            pass
        blocker.close()
        port = free_port()
        self.assertEqual(bounded(lambda: klong(f".srv({port})")), 1)

    def test_two_interpreters_serve_independently(self):
        k1, _ = self.new_repl()
        k2, _ = self.new_repl()
        p1, p2 = free_port(), free_port()
        self.assertEqual(bounded(lambda: k1(f".srv({p1})")), 1)
        self.assertEqual(bounded(lambda: k2(f".srv({p2})")), 1)
        for p in (p1, p2):
            s = socket.create_connection(("127.0.0.1", p), timeout=3)
            s.close()
        # closing one leaves the other serving
        self.assertEqual(bounded(lambda: k1(".srv(0)")), 1)
        s = socket.create_connection(("127.0.0.1", p2), timeout=3)
        s.close()
        self.assertEqual(bounded(lambda: k2(".srv(0)")), 1)

    def test_srv_close_releases_port(self):
        klong, _ = self.new_repl()
        port = free_port()
        bounded(lambda: klong(f".srv({port})"))
        bounded(lambda: klong(".srv(0)"))
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=3)
        self.assertEqual(bounded(lambda: klong(".srv(0)")), 0)

    def test_srv_docstring_has_trust_note(self):
        doc = eval_sys_fn_create_ipc_server.__doc__
        self.assertIn("trusted", doc)
        self.assertIn("127.0.0.1", doc)


class TestTimeouts(ReplCase):
    def test_call_times_out_on_silent_peer(self):
        klong, _ = self.new_repl()
        srv = SilentServer()
        self.addCleanup(srv.stop)
        bounded(lambda: klong(f"c::.cli({srv.port})"))
        nc = klong['c']
        self.assertEqual(nc.call_timeout, 30.0)
        nc.call_timeout = 0.5
        t0 = time.time()
        with self.assertRaises(Exception) as cm:
            bounded(lambda: klong('c("1+1")'), 8)
        self.assertLess(time.time() - t0, 5)
        self.assertIn("timed out", str(cm.exception))

    def test_connection_close_fails_pending_call(self):
        klong, _ = self.new_repl()
        srv = SilentServer(close_after=0.3)
        self.addCleanup(srv.stop)
        bounded(lambda: klong(f"c::.cli({srv.port})"))
        t0 = time.time()
        with self.assertRaises(Exception):
            bounded(lambda: klong('c("1+1")'), 8)
        self.assertLess(time.time() - t0, 5)

    def test_cleanup_pending_none_exception(self):
        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)
        nc = NetworkClient(loop, loop, None, None)
        pending = loop.create_future()
        done = loop.create_future()
        done.set_result(1)
        nc.pending_responses = {1: pending, 2: done}
        nc._cleanup_pending_responses(None)
        self.assertEqual(nc.pending_responses, {})
        self.assertIsInstance(pending.exception(), KlongIPCException)
        self.assertEqual(done.result(), 1)

    def test_pending_requests_share_one_error(self):
        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)
        nc = NetworkClient(loop, loop, None, None)
        f1, f2 = loop.create_future(), loop.create_future()
        nc.pending_responses = {1: f1, 2: f2}
        err = KlongIPCException("boom")
        nc._cleanup_pending_responses(err)
        self.assertIs(f1.exception(), err)
        self.assertIs(f2.exception(), err)


class TestReplLifecycle(unittest.TestCase):
    def test_cleanup_repl_with_pending_task_returns(self):
        klong, loops = create_repl()
        klong_loop = loops[3]

        async def forever():
            await asyncio.sleep(1000)

        klong_loop.call_soon_threadsafe(lambda: klong_loop.create_task(forever()))
        time.sleep(0.2)
        t0 = time.time()
        bounded(lambda: cleanup_repl(loops), 5)
        self.assertLess(time.time() - t0, 5)
        self.assertTrue(klong_loop.is_closed())

    def test_klongpath_idempotent(self):
        old = os.environ.get("KLONGPATH")
        self.addCleanup(lambda: os.environ.__setitem__("KLONGPATH", old)
                        if old is not None else os.environ.pop("KLONGPATH", None))
        os.environ.pop("KLONGPATH", None)
        _, l1 = create_repl()
        first = os.environ["KLONGPATH"]
        cleanup_repl(l1)
        for _ in range(3):
            _, l = create_repl()
            cleanup_repl(l)
        self.assertEqual(os.environ["KLONGPATH"], first)


class TestBareInterpreter(unittest.TestCase):
    def test_bare_interpreter_srv_message(self):
        k = KlongInterpreter()
        with self.assertRaises(Exception) as cm:
            k(".srv(0)")
        self.assertNotIsInstance(cm.exception, KeyError)
        self.assertIn(".system", str(cm.exception))

    def test_bare_interpreter_cli_message(self):
        k = KlongInterpreter()
        with self.assertRaises(Exception) as cm:
            k(".cli(1234)")
        self.assertNotIsInstance(cm.exception, KeyError)
        self.assertIn(".system", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
