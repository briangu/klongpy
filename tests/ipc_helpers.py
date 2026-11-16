"""Helpers shared by the IPC tests: free ports, bounded waits, a scripted kdb+ peer."""
import random
import socket
import threading
import time

from klongpy.ipc import Session
from klongpy.ipc.session import Message


_ports = random.Random()


def free_port():
    """A port that is free now and unlikely to be taken before the caller binds it.

    Ports the OS hands out for outgoing connections (the ephemeral range, 32768 and up) collide with
    a "free" port picked by binding port 0, which made a later bind fail now and then. These come
    from below that range.
    """
    for _ in range(100):
        port = _ports.randrange(12000, 30000)
        s = socket.socket()
        try:
            s.bind(("0.0.0.0", port))
        except OSError:
            continue
        finally:
            s.close()
        return port
    raise RuntimeError("no free port found")


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


def wait_until(pred, timeout=5.0, step=0.01):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(step)
    return pred()


class FakeKdbServer:
    """A scripted kdb+ protocol server on a real localhost socket.

    ``handler(value)`` is called for each request (sync or async) and returns the
    value to answer a sync request with; raising answers with an error object
    whose text is ``str(exception)``.  ``handler=None`` never answers.
    ``reply_handshake=False`` stays silent after the greeting; ``reject=True``
    closes after the greeting; ``close_after`` closes each connection after that
    many seconds.  ``greetings`` and ``requests`` record what arrived.
    """

    def __init__(self, handler=None, reply_handshake=True, reject=False, close_after=None, capability=3):
        self.handler = handler
        self.reply_handshake = reply_handshake
        self.reject = reject
        self.close_after = close_after
        self.capability = capability
        self.greetings = []
        self.requests = []
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(5)
        self.port = self.sock.getsockname()[1]
        self.conns = []
        self.thread = threading.Thread(target=self._accept, daemon=True)
        self.thread.start()

    def _accept(self):
        try:
            while True:
                c, _ = self.sock.accept()
                self.conns.append(c)
                if self.close_after is not None:
                    threading.Timer(self.close_after, self._close, args=(c,)).start()
                threading.Thread(target=self._serve, args=(c,), daemon=True).start()
        except OSError:
            pass

    @staticmethod
    def _close(c):
        try:
            c.close()
        except OSError:
            pass

    def _serve(self, c):
        sess = Session.server()
        try:
            while True:
                data = c.recv(65536)
                if not data:
                    return
                for ev in sess.feed(data):
                    kind = type(ev).__name__
                    if kind == "Handshake":
                        self.greetings.append((ev.user, ev.password, ev.capability))
                        if self.reject:
                            sess.reject()
                            c.close()
                            return
                        if self.reply_handshake:
                            c.sendall(sess.accept())
                            for e2 in sess.feed(b""):
                                self._on_event(c, sess, e2)
                    else:
                        self._on_event(c, sess, ev)
        except OSError:
            pass

    def _on_event(self, c, sess, ev):
        if not isinstance(ev, Message) or ev.kind == "response":
            return
        self.requests.append((ev.kind, ev.value))
        if self.handler is None:
            return
        svc = sess.begin_service(ev)
        try:
            try:
                result = self.handler(ev.value)
                err = None
            except Exception as e:  # noqa: BLE001
                result, err = None, str(e)
            if ev.kind == "sync":
                c.sendall(sess.send_error(svc, err) if err is not None else sess.send_response(svc, result))
        finally:
            sess.end_service(svc)

    def stop(self):
        for c in self.conns:
            self._close(c)
        self.sock.close()
