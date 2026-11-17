"""
The server that runs next to the Alveo card (ticket 111). One process per session, started over ssh by
klongpy/hw/client.py; it talks over stdin and stdout and opens no socket (access control is ssh).

    PYTHONPATH=/opt/xilinx/xrt/python python3 server.py [--device N] [--fake]

Python 3.8, the standard library, numpy and pyxrt only. It does not import klongpy. The one file it needs next to
itself is ``gate.py`` (the rule of which images may be loaded); both files travel together.

Frames, both directions: ``u32 length`` (little endian, the number of bytes after it), then
``u32 header_length``, the header (UTF-8 JSON, an object), and the rest of the frame, the binary data (may be empty).
Numbers in binary data are little endian (the card and the hosts we use are). The server first sends a frame
``{"ok": true, "hello": "klong-hw-server", "protocol": 1, ...}``. A request header holds ``cmd`` and ``id``;
the reply header echoes ``id`` and holds ``ok``: true with the results, or false with ``kind`` and ``message``
(kinds: BadRequest, NoImage, ImageRefused, CardBusy, NoSuchKernel, NoSuchHandle, TooBig, DeviceError, Timeout).

Commands (arguments in the header):
    info                      the device, the shell, the image loaded on the card (not opened by this session
                              until ``load``), the image of this session
    load  path                load a user image with XRT. Refuses ``*.timing-failed.xclbin`` and any image without a
                              build.json that records passed routed timing; refuses when another process uses the
                              card; the reply says what the card held before ("replaced")
    put   nbytes + data       upload bytes to card memory; reply: handle
    get   handle [offset, nbytes]   download; data in the reply
    run   kernel in n reset [out] [ktimeout]   n calls of an entry on the records in buffer ``in``; reply: out handle,
                              status word, seconds (the kernel from submit to done)
    free  handle | all
    status                    handles, bytes, image, counters
    bye                       end the session

No command flashes or programs the shell; ``load`` loads a user image with ``xrt.device.load_xclbin`` only.
"""
import glob
import json
import os
import struct
import subprocess
import sys
import time

PROTOCOL = 1
MAX_FRAME = 1 << 31
MAX_DEVICE_BYTES = 8 << 30
XBUTIL = "/opt/xilinx/xrt/bin/xbutil"
HERE = os.path.dirname(os.path.abspath(__file__))


class ServerError(Exception):
    def __init__(self, kind, message):
        Exception.__init__(self, message)
        self.kind = kind
        self.message = message


# ---------------------------------------------------------------------------------------
# framing
# ---------------------------------------------------------------------------------------

def read_exact(f, n):
    chunks, got = [], 0
    while got < n:
        b = f.read(n - got)
        if not b:
            raise EOFError("the other side closed the connection")
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def read_frame(f):
    """(header dict, data bytes); EOFError at the end of the stream, ServerError('BadRequest') for a malformed frame."""
    first = f.read(4)
    if not first:
        raise EOFError("end of input")
    if len(first) < 4:
        first += read_exact(f, 4 - len(first))
    (length,) = struct.unpack("<I", first)
    if length < 4 or length > MAX_FRAME:
        raise ServerError("BadRequest", "frame length %d is not acceptable" % length)
    body = read_exact(f, length)
    (hl,) = struct.unpack_from("<I", body, 0)
    if 4 + hl > length:
        raise ServerError("BadRequest", "header length %d does not fit the frame" % hl)
    try:
        header = json.loads(body[4:4 + hl].decode("utf-8"))
    except ValueError:
        raise ServerError("BadRequest", "the header is not JSON")
    if not isinstance(header, dict):
        raise ServerError("BadRequest", "the header is not an object")
    return header, bytes(body[4 + hl:])


def encode_frame(header, data=b""):
    h = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<II", 4 + len(h) + len(data), len(h)) + h + bytes(data)


def write_frame(f, header, data=b""):
    f.write(encode_frame(header, data))
    f.flush()


# ---------------------------------------------------------------------------------------
# the device
# ---------------------------------------------------------------------------------------

def emulating():
    """True under hardware emulation (XCL_EMULATION_MODE): the real card is then neither asked nor touched."""
    return bool(os.environ.get("XCL_EMULATION_MODE"))


def _xbutil(args, timeout=60):
    if emulating():
        return "emulation: the card is not consulted\n"
    try:
        p = subprocess.run([XBUTIL] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
                           universal_newlines=True)
        return p.stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return "xbutil failed: %s" % e


def _own_fds_on_device():
    """Pids other than ours that hold the card's device nodes open, as far as this user can see (/proc/<pid>/fd)."""
    mine = os.getpid()
    out = []
    if emulating():
        return out
    for fd_dir in glob.glob("/proc/[0-9]*/fd"):
        pid = int(fd_dir.split("/")[2])
        if pid == mine:
            continue
        try:
            for fd in os.listdir(fd_dir):
                try:
                    target = os.readlink(os.path.join(fd_dir, fd))
                except OSError:
                    continue
                if target.startswith("/dev/dri/") or "xocl" in target or "xclmgmt" in target:
                    try:
                        with open("/proc/%d/comm" % pid) as c:
                            name = c.read().strip()
                    except OSError:
                        name = "?"
                    out.append((pid, name))
                    break
        except OSError:
            continue
    return out


class XrtDevice:
    """The card through pyxrt. Every method is what the server needs; FakeDevice has the same ones."""
    name = "xrt"

    def __init__(self, index=0):
        self.index = index
        self.xrt = None
        self.dev = None
        self.uuid = None
        self.kernels = {}

    # -- what is there, without opening the device ------------------------------------------
    def describe(self):
        text = _xbutil(["examine"])
        info = {"examine": text.strip().splitlines()[-6:], "kind": "xrt"}
        for line in text.splitlines():
            if line.startswith("[") and "]" in line:
                head, _, rest = line.partition("]")
                info["device"] = head.strip("[")
                words = rest.replace(":", " ").split()
                if words:
                    info["shell"] = words[0]
                break
        info.update(self.loaded())
        return info

    def loaded(self):
        """The image on the card now: {"uuid": ..., "kernels": [...], "busy": bool} from xbutil (nothing is opened)."""
        # xbutil needs the device named when the machine could have several; the first card's address is looked up once
        if getattr(self, "bdf", None) is None:
            self.bdf = ""
            for line in _xbutil(["examine"]).splitlines():
                if line.startswith("[") and "]" in line:
                    self.bdf = line.split("]")[0].strip("[")
                    break
        text = _xbutil(["examine"] + (["-d", self.bdf] if self.bdf else []) + ["-r", "dynamic-regions"])
        uuid, cus, busy = None, [], False
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("Xclbin UUID:"):
                uuid = s.split(":", 1)[1].strip()
            elif ":" in s and "(" in s and s.split()[0].isdigit():
                cus.append(s.split()[1])
                if "(IDLE)" not in s:
                    busy = True
        return {"uuid": uuid, "compute_units": cus, "busy": busy}

    def users(self):
        return _own_fds_on_device()

    # -- loading ----------------------------------------------------------------------------
    def load(self, path):
        import pyxrt
        self.xrt = pyxrt
        if self.dev is None:
            self.dev = pyxrt.device(self.index)
        self.uuid = self.dev.load_xclbin(pyxrt.xclbin(path))
        self.kernels = {}
        return str(self.uuid)

    def kernel(self, name):
        if name not in self.kernels:
            try:
                self.kernels[name] = self.xrt.kernel(self.dev, self.uuid, name)
            except Exception as e:
                raise ServerError("NoSuchKernel", "the image has no kernel %s (%s)" % (name, e))
        return self.kernels[name]

    # -- memory ------------------------------------------------------------------------------
    def alloc(self, nbytes, kernel_name):
        k = self.kernel(kernel_name)
        return self.xrt.bo(self.dev, nbytes, self.xrt.bo.normal, k.group_id(0))

    def write(self, bo, data, offset=0):
        bo.write(data, offset)
        bo.sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def read(self, bo, nbytes, offset=0):
        bo.sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return bytes(bo.read(nbytes, offset))

    def run(self, name, bo_in, bo_out, n, reset, bo_status, timeout):
        k = self.kernel(name)
        t0 = time.perf_counter()
        r = k(bo_in, bo_out, n, reset, bo_status)
        state = r.wait(int(timeout * 1000)) if timeout else r.wait()
        dt = time.perf_counter() - t0
        if state != self.xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise ServerError("DeviceError", "the kernel did not complete: %s" % state)
        return dt


class FakeDevice:
    """
    A device made of numpy for tests (and ``--fake``): no card. Its kernels are ``fake_kernels``
    {name: fn(in_words ndarray of uint32, n, reset) -> (out ndarray, status)} ; the default kernel ``copy`` echoes records.
    """
    name = "fake"

    def __init__(self, fake_kernels=None, busy_users=None, loaded=None):
        import numpy as np
        self.np = np
        self.fake_kernels = fake_kernels or {}
        self.busy_users = busy_users or []
        self.was = loaded or {"uuid": "00000000-0000-0000-0000-000000000000", "compute_units": ["other:other_1"], "busy": False}
        self.loaded_paths = []
        self.state = {}
        self.uuid = None

    def describe(self):
        d = {"kind": "fake", "device": "fake:00", "shell": "fake_shell"}
        d.update(self.loaded())
        return d

    def loaded(self):
        if self.uuid:
            return {"uuid": self.uuid, "compute_units": sorted(self.fake_kernels), "busy": False}
        return dict(self.was)

    def users(self):
        return list(self.busy_users)

    def load(self, path):
        self.loaded_paths.append(path)
        self.uuid = "fake-%d" % len(self.loaded_paths)
        self.state = {}
        return self.uuid

    def kernel(self, name):
        if name not in self.fake_kernels:
            raise ServerError("NoSuchKernel", "the image has no kernel %s" % name)
        return self.fake_kernels[name]

    def alloc(self, nbytes, kernel_name):
        return bytearray(nbytes)

    def write(self, bo, data, offset=0):
        bo[offset:offset + len(data)] = data

    def read(self, bo, nbytes, offset=0):
        return bytes(bo[offset:offset + nbytes])

    def run(self, name, bo_in, bo_out, n, reset, bo_status, timeout):
        fn = self.kernel(name)
        np = self.np
        words = np.frombuffer(bytes(bo_in), dtype=np.uint32)
        t0 = time.perf_counter()
        out, status = fn(words, n, reset, self.state)
        raw = np.ascontiguousarray(out).tobytes()
        bo_out[:len(raw)] = raw
        bo_status[:4] = struct.pack("<I", status)
        return time.perf_counter() - t0


# ---------------------------------------------------------------------------------------
# the server
# ---------------------------------------------------------------------------------------

class Server:
    def __init__(self, device, image_ok=None, now=time.perf_counter):
        self.dev = device
        self.image_ok = image_ok or _load_gate()
        self.now = now
        self.handles = {}          # handle -> (bo, nbytes)
        self.next_handle = 1
        self.image = None          # {"path", "uuid", "build", "kernels": {top: entry}}
        self.bytes_in_use = 0
        self.counters = {"runs": 0, "records": 0, "put_bytes": 0, "get_bytes": 0}
        self.times = {"put_s": 0.0, "get_s": 0.0, "run_s": 0.0}
        self.last_kernel = None

    def handle_request(self, header, data):
        cmd = header.get("cmd")
        fn = getattr(self, "cmd_" + cmd, None) if isinstance(cmd, str) and cmd.isidentifier() else None
        if fn is None or cmd.startswith("_"):
            raise ServerError("BadRequest", "unknown command %r" % (cmd,))
        return fn(header, data)

    # -- commands ------------------------------------------------------------------------------
    def cmd_hello(self, h, data):
        return {"protocol": PROTOCOL}, b""

    def cmd_info(self, h, data):
        d = self.dev.describe()
        d["session_image"] = self.image and {"path": self.image["path"], "uuid": self.image["uuid"],
                                              "kernels": sorted(self.image["kernels"])}
        d["protocol"] = PROTOCOL
        d["card_users"] = [{"pid": p, "name": n} for p, n in self.dev.users()]
        return d, b""

    def cmd_load(self, h, data):
        path = h.get("path")
        if not isinstance(path, str) or not path:
            raise ServerError("BadRequest", "load needs a path")
        path = os.path.abspath(path)
        ok, meta = self.image_ok(path)
        if not ok:
            raise ServerError("ImageRefused", "%s: %s" % (path, meta))
        before = self.dev.loaded()
        users = self.dev.users()
        if users:
            raise ServerError("CardBusy", "the card is in use by %s; nothing was loaded" %
                              ", ".join("%s (pid %d)" % (n, p) for p, n in users))
        if before.get("busy"):
            raise ServerError("CardBusy", "a compute unit of the image on the card (%s) is not idle; nothing was loaded" %
                              before.get("uuid"))
        if self.handles:
            self._free_all()
        try:
            uuid = self.dev.load(path)
        except ServerError:
            raise
        except Exception as e:
            raise ServerError("DeviceError", "loading %s failed: %s" % (path, e))
        kernels = {}
        for m in meta.get("kernels", []):
            for e in m.get("entries", []):
                kernels[e["top"]] = e
        self.image = {"path": path, "uuid": uuid, "build": meta, "kernels": kernels}
        replaced = {"uuid": before.get("uuid"), "compute_units": before.get("compute_units")}
        msg = ("loaded %s; it replaces the image that was on the card (uuid %s, compute units %s)" %
               (path, replaced["uuid"], ", ".join(replaced["compute_units"] or []) or "none"))
        return {"uuid": uuid, "replaced": replaced, "message": msg, "kernels": sorted(kernels),
                "entries": [kernels[k] for k in sorted(kernels)], "timing": meta.get("timing"), "key": meta.get("key"),
                "clock_mhz": meta.get("clock_mhz_image")}, b""

    def _need_image(self):
        if self.image is None:
            raise ServerError("NoImage", "no image is loaded in this session: load one first")
        return self.image

    def _new_handle(self, bo, nbytes):
        if self.bytes_in_use + nbytes > MAX_DEVICE_BYTES:
            raise ServerError("TooBig", "%d bytes more would exceed the %d bytes this server uses on the card" % (nbytes, MAX_DEVICE_BYTES))
        hid = self.next_handle
        self.next_handle += 1
        self.handles[hid] = (bo, nbytes)
        self.bytes_in_use += nbytes
        return hid

    def _handle(self, h, key="handle"):
        v = h.get(key)
        if not isinstance(v, int) or v not in self.handles:
            raise ServerError("NoSuchHandle", "no buffer %r" % (v,))
        return v, self.handles[v]

    def _any_kernel(self):
        return sorted(self._need_image()["kernels"])[0]

    def cmd_put(self, h, data):
        self._need_image()
        n = h.get("nbytes", len(data))
        if n != len(data):
            raise ServerError("BadRequest", "put says %r bytes and the frame holds %d" % (n, len(data)))
        if n == 0:
            raise ServerError("BadRequest", "put of zero bytes")
        t0 = self.now()
        bo = self.dev.alloc(n, self._any_kernel())
        self.dev.write(bo, data, 0)
        dt = self.now() - t0
        self.counters["put_bytes"] += n
        self.times["put_s"] += dt
        return {"handle": self._new_handle(bo, n), "nbytes": n, "seconds": dt}, b""

    def cmd_get(self, h, data):
        hid, (bo, nbytes) = self._handle(h)
        off = h.get("offset", 0)
        n = h.get("nbytes", nbytes - off)
        if not (isinstance(off, int) and isinstance(n, int) and off >= 0 and n >= 0 and off + n <= nbytes):
            raise ServerError("BadRequest", "get %r+%r is outside the buffer of %d bytes" % (off, n, nbytes))
        t0 = self.now()
        out = self.dev.read(bo, n, off)
        dt = self.now() - t0
        self.counters["get_bytes"] += n
        self.times["get_s"] += dt
        return {"nbytes": n, "seconds": dt}, out

    def cmd_run(self, h, data):
        image = self._need_image()
        name = h.get("kernel")
        ent = image["kernels"].get(name)
        if ent is None:
            raise ServerError("NoSuchKernel", "the image has no kernel %r (it has %s)" % (name, ", ".join(sorted(image["kernels"]))))
        n = h.get("n")
        if not isinstance(n, int) or n < 1 or n >= 1 << 31:
            raise ServerError("BadRequest", "run needs n, a count of records from 1")
        reset = 1 if h.get("reset") else 0
        hin, (bo_in, in_n) = self._handle(h, "in")
        need = n * ent["in_bytes"]
        if need > in_n:
            raise ServerError("TooBig", "%d records of %d bytes need %d bytes and the input buffer holds %d" %
                              (n, ent["in_bytes"], need, in_n))
        out_bytes = n * ent["out_bytes"]
        if h.get("out") is not None:
            hout, (bo_out, out_n) = self._handle(h, "out")
            if out_n < out_bytes:
                raise ServerError("TooBig", "the output buffer holds %d bytes and %d are needed" % (out_n, out_bytes))
        else:
            bo_out = self.dev.alloc(out_bytes, name)
            hout = self._new_handle(bo_out, out_bytes)
        bo_status = self.dev.alloc(4, name)
        self.dev.write(bo_status, b"\0\0\0\0", 0)
        timeout = h.get("ktimeout", 600)
        try:
            dt = self.dev.run(name, bo_in, bo_out, n, reset, bo_status, timeout)
        except ServerError:
            raise
        except Exception as e:
            raise ServerError("DeviceError", "the run failed: %s" % e)
        status = struct.unpack("<I", self.dev.read(bo_status, 4, 0))[0]
        self.counters["runs"] += 1
        self.counters["records"] += n
        self.times["run_s"] += dt
        self.last_kernel = name
        return {"out": hout, "out_nbytes": out_bytes, "status": status, "seconds": dt, "n": n}, b""

    def _free_all(self):
        self.handles.clear()
        self.bytes_in_use = 0

    def cmd_free(self, h, data):
        if h.get("handle") == "all":
            n = len(self.handles)
            self._free_all()
            return {"freed": n}, b""
        hid, (bo, nbytes) = self._handle(h)
        del self.handles[hid]
        self.bytes_in_use -= nbytes
        return {"freed": 1}, b""

    def cmd_status(self, h, data):
        return {"image": self.image and {"path": self.image["path"], "uuid": self.image["uuid"],
                                         "key": self.image["build"].get("key"), "kernels": sorted(self.image["kernels"])},
                "handles": {str(k): v[1] for k, v in self.handles.items()}, "bytes_in_use": self.bytes_in_use,
                "counters": self.counters, "times": self.times, "device": getattr(self.dev, "name", "?")}, b""

    def cmd_bye(self, h, data):
        return {"bye": True}, b""


def _load_gate():
    """The image rule of gate.py, from the directory of this file; without it nothing is loaded."""
    sys.path.insert(0, HERE)
    try:
        import gate
        return gate.image_ok
    except ImportError:
        def refuse(path):
            return False, "gate.py is missing next to server.py, so no image can be checked"
        return refuse
    finally:
        sys.path.pop(0)


def serve(stdin, stdout, server):
    """Answer requests until ``bye`` or the end of the input. Returns the number of requests."""
    write_frame(stdout, {"ok": True, "hello": "klong-hw-server", "protocol": PROTOCOL, "pid": os.getpid(),
                         "python": sys.version.split()[0], "device": getattr(server.dev, "name", "?")})
    count = 0
    while True:
        try:
            header, data = read_frame(stdin)
        except EOFError:
            return count
        except ServerError as e:
            write_frame(stdout, {"ok": False, "kind": e.kind, "message": e.message})
            return count          # the stream cannot be trusted any more
        count += 1
        reply_id = header.get("id")
        try:
            result, out = server.handle_request(header, data)
            result = dict(result)
            result.update({"ok": True, "id": reply_id})
            write_frame(stdout, result, out)
        except ServerError as e:
            write_frame(stdout, {"ok": False, "id": reply_id, "kind": e.kind, "message": e.message})
        except Exception as e:                       # a bug here must not end the session silently
            write_frame(stdout, {"ok": False, "id": reply_id, "kind": "DeviceError", "message": "%s: %s" % (type(e).__name__, e)})
        if header.get("cmd") == "bye":
            return count


def _fake_copy(words, n, reset, state):
    """The kernel of --fake: result record = first word of each input record, ten words: for protocol tests over ssh."""
    import numpy as np
    return np.asarray(words[:n], dtype=np.uint32), 0


def main(argv):
    device_index = 0
    fake = False
    it = iter(argv)
    for a in it:
        if a == "--device":
            device_index = int(next(it))
        elif a == "--fake":
            fake = True
        else:
            sys.stderr.write("usage: server.py [--device N] [--fake]\n")
            return 2
    # stdout carries the protocol: nothing else may print there
    out = sys.stdout.buffer
    sys.stdout = sys.stderr
    dev = FakeDevice({"copy": _fake_copy}) if fake else XrtDevice(device_index)
    serve(sys.stdin.buffer, out, Server(dev))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
