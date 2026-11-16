"""Helpers for ``test_ipc_peachq.py``: finding PeachQ, running it as a server, running q scripts.

PeachQ (https://github.com/peachq-org/peachq, MIT) is an open-source implementation of the q language.
The interop tests are opt-in: set ``KLONG_TEST_PEACHQ`` to the path of its ``q`` executable.
Everything here is bounded in time; nothing writes outside a temporary directory.
"""
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time

from ipc_helpers import free_port

ENV = "KLONG_TEST_PEACHQ"

#: the PeachQ release the tests were written and last run against (skip texts and notes name it)
TESTED_VERSION = "0.92.20261115"


def peachq_exe():
    """The PeachQ executable named by ``KLONG_TEST_PEACHQ``, or None."""
    path = os.environ.get(ENV, "")
    if path and os.path.isfile(path) and os.access(path, os.X_OK):
        return path
    return None


_version = {}


def peachq_version(exe=None):
    """PeachQ's own version string (``.pq.version``, e.g. ``0.92.20261115``), or ``unknown``."""
    exe = exe or peachq_exe()
    if exe is None:
        return "unknown"
    if exe not in _version:
        try:
            out = subprocess.run([exe, "-q"], input=".pq.version\n\\\\\n", capture_output=True, text=True,
                                 timeout=20, cwd=tempfile.gettempdir()).stdout
            m = re.search(r'"([0-9][^"]*)"', out)
            _version[exe] = m.group(1) if m else "unknown"
        except (OSError, subprocess.SubprocessError):
            _version[exe] = "unknown"
    return _version[exe]


def lan_address():
    """An IPv4 address of this host that is not loopback (PeachQ, like kdb+, compresses only to such peers), or None."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))              # a documentation address; UDP connect sends nothing
        addr = s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()
    if addr.startswith("127.") or addr == "0.0.0.0":
        return None
    try:                                         # make sure it can actually be bound and reached
        t = socket.socket()
        t.bind((addr, 0))
        t.close()
    except OSError:
        return None
    return addr


def wait_listening(port, host="127.0.0.1", timeout=15.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            socket.create_connection((host, port), timeout=0.5).close()
            return True
        except OSError:
            time.sleep(0.05)
    return False


class PeachQServer:
    """``q -p PORT`` in a child process, working in its own temporary directory.

    ``stop()`` terminates it (and kills it if it ignores that).  ``stdout()`` is what it printed so far.
    """

    def __init__(self, exe=None, port=None, extra_args=()):
        self.exe = exe or peachq_exe()
        self.port = port or free_port()
        self.dir = tempfile.mkdtemp(prefix="peachq-")
        self._out = open(os.path.join(self.dir, "stdout.txt"), "w+b")
        self.proc = subprocess.Popen([self.exe, "-q", "-p", str(self.port), *extra_args], cwd=self.dir,
                                     stdin=subprocess.DEVNULL, stdout=self._out, stderr=subprocess.STDOUT)
        if not wait_listening(self.port):
            self.stop()
            raise AssertionError(f"PeachQ did not listen on {self.port}: {self.stdout()!r}")

    def stdout(self):
        self._out.flush()
        with open(self._out.name, "rb") as f:
            return f.read().decode("utf-8", "replace")

    def alive(self):
        return self.proc.poll() is None

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(5)
        self._out.close()
        shutil.rmtree(self.dir, ignore_errors=True)


def run_q(script, timeout=40, exe=None):
    """Run q source as a script under PeachQ; returns ``(returncode, stdout, stderr)``.

    The script should end with ``\\\\`` so that q exits.  A script that runs past ``timeout`` seconds is killed
    and the call raises ``AssertionError``.
    """
    exe = exe or peachq_exe()
    d = tempfile.mkdtemp(prefix="peachq-script-")
    try:
        path = os.path.join(d, "t.q")
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)
        try:
            p = subprocess.run([exe, path, "-q"], cwd=d, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                               timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise AssertionError(f"q script did not finish within {timeout}s; output so far: {e.stdout!r}") from None
        return p.returncode, p.stdout, p.stderr
    finally:
        shutil.rmtree(d, ignore_errors=True)


def parse_results(stdout):
    """Lines ``R|name|status|type|text`` (or ``R|name|err|text``) printed by a q script, as a dict.

    A value is ``("ok", type_number, text)`` or ``("err", text)``.
    """
    out = {}
    for line in stdout.splitlines():
        if not line.startswith("R|"):
            continue
        parts = line.split("|", 4)
        if parts[2] == "ok":
            out[parts[1]] = ("ok", int(parts[3]), parts[4])
        else:
            out[parts[1]] = ("err", "|".join(parts[3:]))
    return out


#: q preamble for client scripts.  ``rep[name;request]`` makes one sync request on handle ``h`` and prints
#: ``R|name|ok|<type>|<-3! text>`` or ``R|name|err|<signal text>``; ``asy[request]`` sends an async message.
Q_PRELUDE = r'''
tr:{[e] @[{(1b;h x)};e;{(0b;x)}]};
rep:{[n;e] r:tr e; $[first r; -1 "R|",n,"|ok|",(string type last r),"|",(-3!last r); -1 "R|",n,"|err|",last r]};
asy:{[e] neg[h] e};
'''
