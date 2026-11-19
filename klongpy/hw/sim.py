"""
The simulator behind ``KLONG_FPGA=sim``: the same session interface as the card, with no card.

    s = SimSession()                    # an HwSession (klongpy/hw/client.py) that answers in this process
    path, build = s.prepare(["fir.kg"])  # compile the kernels; returns the pseudo image path and a build.json-like dict
    s.load(path)
    s.kernel("fir_run_kernel").run(x)

What runs is what the hardware kernel runs, on this machine's CPU: for every entry the generated HLS top function
(``<prog>_<entry>_top.cpp``, the code that Vitis HLS turns into the kernel: the load, core and store steps, the trap
counter, the ``reset`` flag, state kept between runs) is compiled with the system C++ compiler (the HLS pragmas are
ignored) next to the C of the program, and called through ctypes with the same record layout and the same status word
as the card. The requests go through the server of klongpy/hw/server.py with a simulated device in place of pyxrt, so
buffers, handles, errors and bookkeeping are the card's too. It is a simulator: the numbers it produces are not
timing, and a result is never "hardware".

A C compiler (cc, clang or gcc) and a C++ compiler (c++, clang++ or g++) must be on the path. Compiled code is cached
under the temporary directory by the cache key of the program, so a second session does not compile again.
"""
import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np

from .client import HwError, HwRefused, HwSession
from .server import FakeDevice, Server, ServerError

__all__ = ["SimSession", "SimUnavailable", "compilers"]

SIM_UUID = "simulator"


class SimUnavailable(HwError):
    """The simulator cannot be built here (no C or C++ compiler, or the compile failed)."""
    def __init__(self, message):
        HwError.__init__(self, "NoSimulator", message)


def compilers():
    """(C compiler, C++ compiler) from the path, or raises SimUnavailable. $KG_TEST_CC / $KG_TEST_CXX are honoured."""
    def find(env, names):
        for cand in (os.environ.get(env),) + names:
            if cand and shutil.which(cand):
                return shutil.which(cand)
        return None
    cc = find("KG_TEST_CC", ("cc", "clang", "gcc"))
    cxx = find("KG_TEST_CXX", ("c++", "clang++", "g++"))
    if not cc or not cxx:
        raise SimUnavailable("KLONG_FPGA=sim needs a C compiler and a C++ compiler on the path (cc and c++, clang and clang++, "
                             "or gcc and g++); %s not found" % ("neither was" if not cc and not cxx else
                                                                  "the C compiler was" if not cc else "the C++ compiler was"))
    return cc, cxx


def _cache_root():
    try:
        uid = os.getuid()
    except AttributeError:
        uid = 0
    d = os.path.join(tempfile.gettempdir(), "klongpy-hw-sim-%d" % uid)
    os.makedirs(d, exist_ok=True)
    return d


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SimUnavailable("the simulator build failed: %s\n%s" % (" ".join(cmd), r.stderr[-1500:]))


def build_library(out):
    """
    Compile one program's HlsOutput ``out`` (klongpy.static.native.emit_hls_program) to a shared library of its
    kernels, cached by cache key; returns the path of the library.
    """
    key = out.manifest["cache_key"]
    root = _cache_root()
    lib = os.path.join(root, "%s-%s.so" % (out.prefix, key[:24]))
    if os.path.exists(lib):
        return lib
    cc, cxx = compilers()
    work = tempfile.mkdtemp(prefix="build-", dir=root)
    try:
        out.write(work)
        flags = list(out.manifest["cflags"]) + ["-I" + work, "-O1", "-ffp-contract=off", "-fPIC"]
        objs = []
        o = os.path.join(work, "core.o")
        _run([cc, "-std=c99", "-c", os.path.join(work, out.prefix + ".c"), "-o", o] + flags)
        objs.append(o)
        for e in out.entries:               # a kernel with one process per lane has the program's C once more per lane
            for lf in e.get("lane_files", ()):
                o = os.path.join(work, lf[:-2] + ".o")
                _run([cc, "-std=c99", "-c", os.path.join(work, lf), "-o", o] + flags)
                objs.append(o)
        for e in out.entries:
            o = os.path.join(work, e["c_name"] + "_top.o")
            _run([cxx, "-std=c++14", "-Wno-unknown-pragmas", "-c", os.path.join(work, e["files"]["top"]), "-o", o] + flags)
            objs.append(o)
        tmp = os.path.join(work, "lib.so")
        _run([cxx, "-shared"] + objs + ["-lm", "-o", tmp])
        os.replace(tmp, lib)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return lib


class SimDevice(FakeDevice):
    """The server's FakeDevice with the compiled kernels of the programs of the image being loaded."""
    name = "simulator"

    def __init__(self):
        FakeDevice.__init__(self, loaded={"uuid": SIM_UUID, "compute_units": [], "busy": False})
        self.images = {}            # pseudo path -> list of (library path, [entry dicts])
        self._libs = []

    def describe(self):
        d = FakeDevice.describe(self)
        d.update({"kind": "simulator", "device": "simulator", "shell": "none (C compiled for this machine)"})
        return d

    def load(self, path):
        uuid = FakeDevice.load(self, path)
        self.fake_kernels = {}
        for lib_path, entries in self.images[path]:
            # a private copy, so that this load starts every program from its initial state
            fresh = "%s.%d.%d.so" % (lib_path[:-3], os.getpid(), len(self._libs))
            shutil.copyfile(lib_path, fresh)
            lib = ctypes.CDLL(fresh)
            os.unlink(fresh)
            self._libs.append(lib)
            for e in entries:
                self.fake_kernels[e["top"]] = _kernel_fn(lib, e)
        return uuid


def _kernel_fn(lib, entry):
    fn = getattr(lib, entry["top"])
    fn.restype = None
    # the register arguments between reset and status: one table pointer (lanes), or scalars in manifest order
    regs = entry.get("registers") or []
    if entry.get("register_table"):
        reg_types = [ctypes.c_void_p]
    else:
        reg_types = [{"float": ctypes.c_float}.get(r["elem"], ctypes.c_int64 if r["bytes"] == 8 else ctypes.c_int32) for r in regs]
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint] + reg_types + [ctypes.POINTER(ctypes.c_uint)]
    in_bytes, out_bytes = entry["in_bytes"], entry["out_bytes"]
    summary = entry.get("output") == "summary"

    def run(words, n, reset, state, regvals=None):
        data = np.array(words, dtype=np.uint32, copy=True)
        if data.nbytes < n * in_bytes:
            raise ServerError("TooBig", "the input holds %d bytes and %d records need %d" % (data.nbytes, n, n * in_bytes))
        out = np.zeros(out_bytes if summary else n * out_bytes, dtype=np.uint8)
        status = ctypes.c_uint(0)
        extra = []
        if regs:
            if regvals is None:
                raise ServerError("BadRequest", "kernel %s has registers and the server passed none" % entry["top"])
            if entry.get("register_table"):
                table = np.frombuffer(bytes(regvals["args"][0]), dtype=np.uint8).copy()
                extra = [table.ctypes.data]
            else:
                extra = []
                for r, v in zip(regs, regvals["args"]):
                    if r["elem"] == "float":
                        v = float(np.array([v], dtype="<i4").view("<f4")[0])
                    extra.append(v)
        fn(data.ctypes.data, out.ctypes.data, n, 1 if reset else 0, *extra, ctypes.byref(status))
        return out, status.value
    return run


class SimSession(HwSession):
    """An HwSession that answers in this process (see the module docstring)."""
    simulated = True

    def __init__(self, timeout=120):
        HwSession.__init__(self, host=None, timeout=timeout)
        self.dev = SimDevice()
        self.server = None
        self._meta = {}
        self._dir = None

    # -- the "connection" ------------------------------------------------------------------
    def start(self):
        if self.server is None:
            self.server = Server(self.dev, image_ok=self._image_ok)
            self.hello = {"ok": True, "hello": "klong-hw-server", "protocol": 1, "device": "simulator"}
            self.proc = None
        return self

    def _image_ok(self, path):
        meta = self._meta.get(path)
        if meta is None:
            return False, "no simulated image %s" % path
        return True, meta

    def call(self, cmd, data=b"", timeout=None, **args):
        if self.server is None:
            self.start()
        rid = self._next_id
        self._next_id += 1
        header = dict(args, cmd=cmd, id=rid)
        try:
            res, out = self.server.handle_request(header, bytes(data))
        except ServerError as e:
            raise (HwRefused if e.kind in ("ImageRefused", "CardBusy") else HwError)(e.kind, e.message)
        res = dict(res)
        res["ok"] = True
        res["id"] = rid
        return res, out

    def close(self):
        self.server = None

    # -- images ----------------------------------------------------------------------------
    def prepare(self, outputs):
        """
        Compile the HlsOutputs of an image's programs and register them as one simulated image. Returns
        (pseudo path, build dict shaped like a build.json: key, passed, kernels = the manifests).
        """
        libs = [(build_library(o), [dict(e, registers=o.manifest["registers"]) if o.manifest.get("registers") else e
                                    for e in o.entries]) for o in outputs]
        keys = [o.manifest["cache_key"] for o in outputs]
        key = keys[0] if len(keys) == 1 else hashlib.sha256("\n".join(keys).encode()).hexdigest()
        if self._dir is None:
            self._dir = tempfile.mkdtemp(prefix="sim-", dir=_cache_root())
        path = os.path.abspath(os.path.join(self._dir, key[:12] + ".sim"))
        build = {"format": 1, "key": key, "target": "simulator", "passed": True, "xclbin": os.path.basename(path),
                 "timing": {"met": True, "wns_ns": None, "note": "simulator: no timing, no hardware"},
                 "clock_mhz_requested": outputs[0].manifest["clock_mhz"], "clock_mhz_image": None,
                 "kernels": [o.manifest for o in outputs], "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        self.dev.images[path] = libs
        self._meta[path] = build
        return path, build

    def load(self, path):
        h = HwSession.load(self, path)
        h["message"] = "simulator: loaded %s; no card is involved" % os.path.basename(path)
        return h
