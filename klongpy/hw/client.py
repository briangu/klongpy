"""
The client of klongpy/hw/server.py (ticket 111): starts the server next to the card and moves numpy arrays.

    with HwSession("brian@xilinx") as hw:          # None (or $KLONG_HW_HOST unset): a server on this machine
        info = hw.load("/home/brian/klong-hw/builds/<key>/fir.xclbin")
        k = hw.kernel("fir_run_kernel")             # from the build.json of the loaded image
        y = k.run(x)                                # x: (n, 256) float32 -> (n, 241) float32
        k.last                                      # {"transfer_s", "kernel_s", "bytes_in", "bytes_out", ...}

The server is started as ``ssh HOST python server.py``; its two files (server.py and gate.py) are copied to
``<root>/server/<hash>/`` on the host first. There is no listening socket. When the ssh connection drops or the
server stops answering, calls raise HwConnectionError (a HwError) instead of hanging.
"""
import hashlib
import json
import os
import queue
import shlex
import struct
import subprocess
import sys
import threading
import time

import numpy as np

from .remote import REMOTE_ROOT, clean_stderr

__all__ = ["HwSession", "HwKernel", "HwError", "HwConnectionError", "HwRefused", "HwTrap", "Handle", "host_target"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
MAX_FRAME = 1 << 31


class HwError(RuntimeError):
    """The server answered with an error: ``kind`` (BadRequest, NoImage, ImageRefused, CardBusy, ...) and the message."""
    def __init__(self, kind, message):
        RuntimeError.__init__(self, "%s: %s" % (kind, message))
        self.kind = kind
        self.message = message


class HwConnectionError(HwError):
    def __init__(self, message):
        HwError.__init__(self, "ConnectionLost", message)


class HwRefused(HwError):
    """The server refused to load an image (timing not met, no build.json, card busy)."""


class HwTrap(HwError):
    """A run set the trap bit of the kernel's status word (index out of range, integer division by zero)."""
    def __init__(self, status, kernel):
        why = []
        if status & 2:
            why.append("index out of range")
        if status & 4:
            why.append("integer division by zero")
        HwError.__init__(self, "Trap", "kernel %s: %s (status 0x%x)" % (kernel, ", ".join(why) or "trap", status))
        self.status = status


def host_target(env=None):
    """The ssh destination of the card host from $KLONG_HW_HOST, or None (this machine)."""
    v = (env if env is not None else os.environ).get("KLONG_HW_HOST", "").strip()
    return v or None


class Handle:
    """A buffer in card memory: ``hw.put(array)`` makes one, kernels accept it in place of an array, ``hw.get`` reads it."""
    def __init__(self, session, hid, nbytes, dtype=None, shape=None):
        self.session, self.id, self.nbytes, self.dtype, self.shape = session, hid, nbytes, dtype, shape

    def __repr__(self):
        return "Handle(%d, %d bytes)" % (self.id, self.nbytes)


def frame(header, data=b""):
    h = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<II", 4 + len(h) + len(data), len(h)) + h + bytes(data)


class HwSession:
    """
    host      ssh destination (user@host) or None for this machine (default: $KLONG_HW_HOST)
    command   the command list that starts the server (tests give their own); default: see ``start``
    timeout   seconds to wait for any reply other than a run (a run waits as long as its own timeout plus a margin)
    """
    def __init__(self, host="env", command=None, timeout=120, root=REMOTE_ROOT, fake=False, python=None, env=None, cwd=None, setup=None):
        self.host = host_target() if host == "env" else host
        self.command = command
        self.timeout = timeout
        self.root = root
        self.fake = fake
        self.python = python
        self.env = dict(env or {})          # exported before the server starts (on the host)
        self.cwd = cwd
        self.setup = setup                  # a shell line run on the host first (hardware emulation sources the Vitis settings)
        self.proc = None
        self._q = queue.Queue()
        self._stderr = []
        self._next_id = 1
        self.hello = None
        self.image = None
        self.entries = {}
        self.last = None

    # -- the connection -------------------------------------------------------------------
    def _server_files(self):
        out = {}
        for name in ("server.py", "gate.py"):
            with open(os.path.join(_HERE, name), "rb") as f:
                out[name] = f.read()
        return out

    def _install(self):
        """Copy server.py and gate.py to <root>/server/<hash>/ on the host; returns the path of server.py there."""
        files = self._server_files()
        h = hashlib.sha256()
        for n in sorted(files):
            h.update(files[n])
        d = "%s/server/%s" % (self.root, h.hexdigest()[:12])
        import io
        import tarfile
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as t:
            for n, data in files.items():
                ti = tarfile.TarInfo(n)
                ti.size = len(data)
                ti.mode = 0o644
                ti.mtime = int(time.time())
                t.addfile(ti, io.BytesIO(data))
        p = subprocess.run(_SSH + [self.host, "mkdir -p %s && tar xf - -C %s" % (shlex.quote(d), shlex.quote(d))],
                           input=buf.getvalue(), capture_output=True, timeout=120)
        if p.returncode != 0:
            raise HwConnectionError("cannot copy the server to %s: %s" % (self.host, clean_stderr(p.stderr.decode(errors="replace")).strip()))
        return d + "/server.py"

    def start(self):
        if self.proc is not None:
            return self
        fake = ["--fake"] if self.fake else []
        try:
            if self.command is not None:
                cmd = list(self.command)
            elif self.host is None:
                cmd = [sys.executable, os.path.join(_HERE, "server.py")] + fake
                if self.env or self.cwd:
                    raise HwConnectionError("env and cwd are for a session on another host")
            else:
                path = self._install()
                py = self.python or (self.root + "/venv/bin/python")
                exports = "".join("export %s=%s; " % (k, shlex.quote(v)) for k, v in sorted(self.env.items()))
                remote = ("%s%ssource /opt/xilinx/xrt/setup.sh >/dev/null 2>&1; export PYTHONPATH=/opt/xilinx/xrt/python${PYTHONPATH:+:$PYTHONPATH}; "
                          "%sexec %s -u %s %s" % ((self.setup + "; ") if self.setup else "", exports, ("cd %s; " % shlex.quote(self.cwd)) if self.cwd else "",
                                                  shlex.quote(py), shlex.quote(path), " ".join(fake)))
                cmd = _SSH + [self.host, remote]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as e:
            raise HwConnectionError("cannot start the server: %s" % e)
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=self._pump_err, daemon=True).start()
        self.hello = self._reply(self.timeout)[0]
        return self

    def _pump(self):
        out = self.proc.stdout
        try:
            while True:
                first = out.read(4)
                if len(first) < 4:
                    break
                (n,) = struct.unpack("<I", first)
                if n < 4 or n > MAX_FRAME:
                    break
                body = out.read(n)
                while len(body) < n:
                    more = out.read(n - len(body))
                    if not more:
                        raise EOFError
                    body += more
                (hl,) = struct.unpack_from("<I", body, 0)
                self._q.put((json.loads(body[4:4 + hl].decode("utf-8")), bytes(body[4 + hl:])))
        except (EOFError, ValueError, OSError):
            pass
        self._q.put(None)

    def _pump_err(self):
        try:
            for line in self.proc.stderr:
                self._stderr.append(line.decode(errors="replace").rstrip())
                del self._stderr[:-40]
        except (OSError, ValueError):
            pass

    def _lost(self, why):
        time.sleep(0.2)
        tail = "\n".join(l for l in self._stderr if not l.startswith("** "))[-800:]
        rc = self.proc.poll() if self.proc else None
        return HwConnectionError("%s%s%s" % (why, " (server exit status %s)" % rc if rc is not None else "", (": " + tail) if tail else ""))

    def _reply(self, timeout):
        try:
            item = self._q.get(timeout=timeout)
        except queue.Empty:
            raise self._lost("no answer from the server in %s s" % timeout)
        if item is None:
            self._q.put(None)
            raise self._lost("the connection to the server was lost")
        return item

    def call(self, cmd, data=b"", timeout=None, **args):
        """One request; returns (reply header, data). Raises HwError (the server's) or HwConnectionError."""
        if self.proc is None:
            self.start()
        rid = self._next_id
        self._next_id += 1
        header = dict(args, cmd=cmd, id=rid)
        try:
            self.proc.stdin.write(frame(header, data))
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            raise self._lost("cannot send to the server")
        h, d = self._reply(timeout or self.timeout)
        if not h.get("ok"):
            kind = h.get("kind", "Error")
            cls = HwRefused if kind in ("ImageRefused", "CardBusy") else HwError
            raise cls(kind, h.get("message", ""))
        return h, d

    def close(self):
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            proc.stdin.write(frame({"cmd": "bye", "id": 0}))
            proc.stdin.flush()
            proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.terminate()            # our own child, by its Popen object
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    # -- commands -------------------------------------------------------------------------
    def info(self):
        return self.call("info")[0]

    def status(self):
        return self.call("status")[0]

    def load(self, path):
        """Load a built image on the card (the reply says what it replaces). Refused when it failed timing."""
        h, _ = self.call("load", path=path, timeout=max(self.timeout, 300))
        self.image = path
        self.entries = {e["top"]: e for e in h.get("entries", [])}
        return h

    def put(self, array):
        a = np.ascontiguousarray(array)
        h, _ = self.call("put", data=a.tobytes(), nbytes=a.nbytes)
        return Handle(self, h["handle"], a.nbytes, a.dtype, a.shape), h

    def get(self, handle, dtype=None, shape=None, count=None):
        h, d = self.call("get", handle=handle.id, nbytes=count) if count is not None else self.call("get", handle=handle.id)
        dt = np.dtype(dtype or handle.dtype or np.uint8)
        a = np.frombuffer(d, dtype=dt).copy()
        if shape is not None:
            a = a.reshape(shape)
        self.last_get = h
        return a

    def free(self, handle="all"):
        self.call("free", handle="all" if handle == "all" else handle.id)

    def kernel(self, name):
        if name not in self.entries:
            raise HwError("NoSuchKernel", "the loaded image has no kernel %s (it has %s)" % (name, ", ".join(sorted(self.entries)) or "none"))
        return HwKernel(self, self.entries[name])


class HwKernel:
    """One entry of the loaded image: arrays in, arrays out. ``last`` holds the transfer and kernel times of the last run."""
    def __init__(self, session, entry):
        self.s = session
        self.e = entry
        words_in, words_out = entry["in_words"], entry["out_words"]
        if entry["in_type"] == "uint32_t":
            raise HwError("BadRequest", "kernel %s has records that mix int and float: not supported by this client" % entry["top"])
        self.in_dtype = self._dtype(entry["in_type"], entry["in_bytes"], words_in)
        self.out_dtype = self._dtype(entry["out_type"], entry["out_bytes"], words_out)
        self.out_shape = tuple(entry["result"]["shape"])
        self.params = entry["params"]
        self.summary = entry.get("output") == "summary"        # one result record whatever n is (a summary pair)
        self.registers = entry.get("registers") or []
        self.last = None

    def set_register(self, name, values):
        """
        Set the register ``name`` of this kernel: ``values`` is a numpy array of one value per lane (or one value),
        already of the register's type (int, scaled integers of a fixed-point type, or float). Kept by the server and
        passed at every run, so it stays until it is set again.
        """
        r = next((x for x in self.registers if x["name"] == name), None)
        if r is None:
            raise HwError("NoSuchRegister", "kernel %s has no register %r" % (self.e["top"], name))
        dt = np.dtype("<f4") if r["elem"] == "float" else np.dtype("<i%d" % r["bytes"])
        a = np.asarray(values).astype(dt).reshape(-1)
        self.s.call("set", data=a.tobytes(), kernel=self.e["top"], name=name)

    @staticmethod
    def _dtype(ctype, nbytes, words):
        if ctype == "kg_float":
            return np.dtype("<f4")
        return np.dtype("<i%d" % (nbytes // words))

    def pack(self, args):
        """One array per argument, each (n, *shape) (or the bare shape for one call), to (n, in_words)."""
        if len(args) != len(self.params):
            raise HwError("BadRequest", "kernel %s takes %d argument(s), %d given" % (self.e["top"], len(self.params), len(args)))
        cols, n = [], None
        for a, p in zip(args, self.params):
            a = np.asarray(a, dtype=self.in_dtype)
            shape = tuple(p["shape"])
            if a.shape == shape:
                a = a.reshape((1,) + shape)
            if a.shape[1:] != shape:
                raise HwError("BadRequest", "argument %s of kernel %s must have shape %s or (n,)+%s, got %s" %
                              (p["name"], self.e["top"], shape, shape, a.shape))
            if n is not None and a.shape[0] != n:
                raise HwError("BadRequest", "the arguments hold %d and %d calls" % (n, a.shape[0]))
            n = a.shape[0]
            cols.append(a.reshape(n, -1))
        return np.ascontiguousarray(np.hstack(cols) if len(cols) > 1 else cols[0]), n

    def run(self, *args, reset=False, timeout=600, keep=False, n=None):
        """
        Run the entry on n calls. Arrays in, array out (n, *result shape); with a Handle as the only argument the
        input stays on the card (``n``: only its first n calls); with keep=True the result is a Handle too. For a
        summary kernel the result is the one summary record, and no argument reads the state without running a call.
        Raises HwTrap when a trap flag was set.
        """
        t0 = time.perf_counter()
        if not args and self.summary:
            # read the summary of the state as it is: no calls (n = 0); the input buffer is not read
            flat, n = np.zeros(max(1, self.e["in_bytes"] // self.in_dtype.itemsize), dtype=self.in_dtype), 0
            hin, ph = self.s.put(flat)
            put_s, put_b = ph["seconds"], 0
        elif len(args) == 1 and isinstance(args[0], Handle):
            hin = args[0]
            held = hin.nbytes // self.e["in_bytes"]
            n = held if n is None else n
            if n > held:
                raise HwError("TooBig", "the buffer holds %d calls and %d were asked for" % (held, n))
            put_s, put_b = 0.0, 0
        else:
            flat, n = self.pack(args)
            hin, ph = self.s.put(flat)
            put_s, put_b = ph["seconds"], flat.nbytes
        try:
            h, _ = self.s.call("run", timeout=timeout + 30, kernel=self.e["top"], n=n, reset=bool(reset), ktimeout=timeout,
                               **{"in": hin.id})
            lead = () if self.summary else (n,)
            out = Handle(self.s, h["out"], h["out_nbytes"], self.out_dtype, lead + self.out_shape)
            result, get_s = out, 0.0
            if not keep:
                data = self.s.get(out)
                get_s = self.s.last_get["seconds"]
                result = data.reshape(lead + self.out_shape)
                self.s.free(out)
        finally:
            if not (len(args) == 1 and isinstance(args[0], Handle)):
                try:
                    self.s.free(hin)
                except HwError:
                    pass
        self.last = {"n": n, "kernel_s": h["seconds"], "put_s": put_s, "get_s": get_s, "transfer_s": put_s + get_s,
                     "total_s": time.perf_counter() - t0, "bytes_in": put_b, "bytes_out": h["out_nbytes"], "status": h["status"]}
        if h["status"] & 1:
            raise HwTrap(h["status"], self.e["top"])
        return result
