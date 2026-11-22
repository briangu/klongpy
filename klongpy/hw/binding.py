"""
The Python half of ``.fpga`` (klongpy/sys_fn_hw.py is the Klong half): which hardware is used, which image holds a
program, and calls of its entries with numpy arrays. Nothing here imports torch or klongpy's interpreter, and nothing
is imported at KlongPy start.

    hw = bind(["examples/embedded/fir.kg"])       # Binding: compiles the program, finds and loads its image
    hw.entries["run"].call([x])                   # numpy in, numpy out; x of shape (256,) or (n, 256)

Where the kernels run (``choose_mode``):

    KLONG_FPGA=sim            the simulator (klongpy/hw/sim.py): the HLS top functions compiled for this machine
    KLONG_FPGA_HOST=user@host the card on that machine, over ssh (the server of klongpy/hw/server.py)
    pyxrt importable        the card on this machine
    otherwise               HwNotAvailable, with the three ways to get hardware

One session (one server process, one image on the card) is shared by every interpreter of the process; it is closed
when the process ends.
"""
import atexit
import importlib.util
import json
import os
import shlex
import threading

import numpy as np

from ..static.signature import Mismatch, Ty, conform, parse_signature
from .client import HwConnectionError, HwError

__all__ = ["bind", "choose_mode", "close_all", "Binding", "BoundEntry", "Buffer", "HwNotAvailable", "NoImage", "BadCall",
           "session_for", "find_image_for"]

DEFAULT_CLOCK = 300
BUILD_COMMAND = "kgpy fpga build"


class HwNotAvailable(HwError):
    """No simulator, no card and no pyxrt: nothing to run on."""
    def __init__(self, message):
        HwError.__init__(self, "NotAvailable", message)


class NoImage(HwError):
    """There is no usable image for the program: the message names the command that builds it."""
    def __init__(self, message):
        HwError.__init__(self, "NoImage", message)


class BadCall(HwError):
    """An argument does not fit the entry: ``kind`` is "type", "shape" or "range" (the Klong error it becomes)."""
    def __init__(self, kind, message):
        HwError.__init__(self, "BadCall", message)
        self.what = kind


# ---------------------------------------------------------------------------------------
# which hardware
# ---------------------------------------------------------------------------------------

def choose_mode(env=None):
    """"sim", "card" (a machine over ssh), "local" (pyxrt here) or None."""
    env = os.environ if env is None else env
    if env.get("KLONG_FPGA", "").strip().lower() == "sim":
        return "sim"
    if env.get("KLONG_FPGA_HOST", "").strip():
        return "card"
    if importlib.util.find_spec("pyxrt") is not None:
        return "local"
    return None


_LOCK = threading.RLock()
_SESSIONS = {}
_TEMPLATE_HINT = ("set KLONG_FPGA_HOST=user@host to use the Alveo card on that machine over ssh, run KlongPy on the machine "
                  "with the card (pyxrt importable), or set KLONG_FPGA=sim for the simulator (no hardware)")


def session_for(mode):
    """The shared session of a mode (started on first use)."""
    with _LOCK:
        s = _SESSIONS.get(mode)
        if s is not None and (mode == "sim" or s.proc is not None):
            return s
        if mode == "sim":
            from .sim import SimSession
            s = SimSession()
        else:
            from .client import HwSession, host_target
            s = HwSession(host_target() if mode == "card" else None, timeout=300)
            s.mode = mode
        s.mode = mode
        s.start()
        _SESSIONS[mode] = s
        return s


def drop_session(mode):
    with _LOCK:
        s = _SESSIONS.pop(mode, None)
    if s is not None:
        s.close()


def close_all():
    """Close every session: buffers are freed with them. Registered with atexit."""
    with _LOCK:
        items = list(_SESSIONS.items())
        _SESSIONS.clear()
    for _, s in items:
        try:
            s.close()
        except Exception:
            pass


atexit.register(close_all)


# ---------------------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------------------

def _program_outputs(files, clock_mhz=DEFAULT_CLOCK, lanes=None):
    from ..static.native import emit_hls_program
    return [emit_hls_program(f, None, int_bits=32, clock_mhz=clock_mhz, lanes=lanes if lanes and lanes > 1 else None)
            for f in files]


def _images_script(root):
    return ("python3 - <<'PY'\nimport glob, json\n"
            "for p in sorted(glob.glob(%s + '/builds/*/build.json')):\n"
            "    try:\n        d = json.load(open(p))\n    except Exception:\n        continue\n"
            "    t = d.get('timing') or {}\n"
            "    print(json.dumps({'dir': p.rsplit('/', 1)[0], 'xclbin': d.get('xclbin'), 'passed': bool(d.get('passed')), "
            "'met': bool(t.get('met')), 'target': d.get('target'), 'built_at': d.get('built_at'), "
            "'programs': [k.get('cache_key') for k in d.get('kernels', [])], "
            "'clocks': [k.get('clock_mhz') for k in d.get('kernels', [])]}))\nPY\n" % json.dumps(root))


def find_image_for(files, host, clock_mhz=DEFAULT_CLOCK, lanes=None):
    """
    The image of exactly these programs (kgpy fpga build FILE... [--lanes N]), else, for programs built together with
    others, the newest passing image that holds every one of them (same source hash). Returns (path, build.json dict).
    ``clock_mhz`` None looks at the default clock first and then at every clock an image on the machine was built for.
    Raises NoImage, with the build command, when there is none.
    """
    from .build import Builder, StaleImage, find_image, plan_for
    from .remote import Remote, RemoteError
    first_clock = DEFAULT_CLOCK if clock_mhz is None else clock_mhz
    try:
        return find_image(files, host=host, clock_mhz=first_clock, lanes=lanes)
    except StaleImage as first:
        exact = first
    except RemoteError as e:
        raise HwConnectionError(str(e))
    if "still being built" in str(exact):
        raise NoImage(str(exact))
    try:
        b = Builder(Remote(target=host))
        rc, out, err = b.sh(_images_script(b.root))
    except RemoteError as e:
        raise HwConnectionError(str(e))
    if rc != 0:
        raise HwConnectionError("cannot list the images on %s: %s" % (host or "this machine", err.strip()))
    listed = []
    for line in out.splitlines():
        try:
            listed.append(json.loads(line))
        except ValueError:
            continue
    clocks = [first_clock]
    if clock_mhz is None:
        for d in listed:
            for c in d.get("clocks") or []:
                if isinstance(c, int) and c not in clocks:
                    clocks.append(c)
    plans = {c: plan_for(files, 32, c, lanes=lanes if lanes and lanes > 1 else None) for c in clocks}
    best = None
    for d in listed:
        if d.get("target") != "hw" or not d.get("passed") or not d.get("met"):
            continue
        if any(all(k in d["programs"] for k in pl.program_keys) for pl in plans.values()):
            if best is None or (d.get("built_at") or "") > (best.get("built_at") or ""):
                best = d
    if best is None:
        raise NoImage(str(exact))
    rc, out, err = b.sh("cat %s/build.json" % shlex.quote(best["dir"]))
    try:
        meta = json.loads(out)
    except ValueError:
        raise NoImage("the build.json of %s cannot be read" % best["dir"])
    return "%s/%s" % (best["dir"], meta["xclbin"]), meta


def build_command(files, clock_mhz=DEFAULT_CLOCK, lanes=None):
    clock_mhz = DEFAULT_CLOCK if clock_mhz is None else clock_mhz
    return "%s %s%s%s" % (BUILD_COMMAND, " ".join(files), "" if clock_mhz == DEFAULT_CLOCK else " --clock-mhz %d" % clock_mhz,
                          " --lanes %d" % lanes if lanes and lanes > 1 else "")


# ---------------------------------------------------------------------------------------
# a bound image
# ---------------------------------------------------------------------------------------

class Buffer:
    """A resident buffer: the packed argument records of ``n`` calls of one entry, in card memory."""
    def __init__(self, binding, entry, handle, n, many=True):
        self.binding, self.entry, self.handle, self.n, self.many = binding, entry, handle, n, many
        self.freed = False

    @property
    def nbytes(self):
        return self.handle.nbytes

    def __repr__(self):
        return "<hw buffer %d: %d call(s) of %s, %d bytes%s>" % (self.handle.id, self.n, self.entry.qualified, self.nbytes,
                                                                  ", freed" if self.freed else "")


def _types(entry):
    """The argument types (static Ty) of an entry dict of a manifest."""
    return [Ty(p["elem"], tuple(p["shape"])) for p in entry["params"]]


def _np_in(elem, int_bits):
    return np.dtype("<f4") if elem == "float" else np.dtype("<i%d" % (int_bits // 8))


class BoundEntry:
    """
    One entry of a bound image: ``call`` runs it on the card (or the simulator).

    ``summary``: the entry is the driver of a summary pair (``step: T -> void`` with ``summary: -> R``): a call runs its
    n calls and returns the one summary record of the reader. ``reader``: the entry is that reader, a call of no
    arguments that runs no ticks and returns the summary of the state as it is.
    """
    def __init__(self, binding, program, entry, reader=False):
        self.binding, self.program, self.e = binding, program, entry
        self.reader = reader
        self.summary = entry.get("output") == "summary"
        if reader:
            self.name = entry["summary"]["entry"]
            self.sig = parse_signature(entry["summary"]["signature"], allow_str=False)
        else:
            self.name = entry["name"]
            self.sig = parse_signature(entry["signature"], allow_str=False)
        self.qualified = "%s.%s" % (program["program"], self.name)
        self.top = entry["top"]
        self.types = list(self.sig.params)
        self.arity = len(self.types)
        self.int_bits = program.get("int_bits", 32)
        self.mixed = len({t.elem for t in self.types}) > 1
        self.last = None
        # the result as the kernel returns it: the lane dimension (--lanes) in front, the summary's type for a driver
        r = entry["result"]
        self.result_ty = Ty(r["elem"], tuple(r["shape"]))

    @property
    def pending_reset(self):
        """The first call of a kernel after .fpga (or .fpga.reset) starts the program from its initial state; shared by the
        entries of one kernel (a driver and its reader)."""
        return self.binding.pending.get(self.top, True)

    @pending_reset.setter
    def pending_reset(self, v):
        self.binding.pending[self.top] = v

    def __repr__(self):
        return "<hw entry %s: %s>" % (self.qualified, self.e["signature"])

    # -- arguments ------------------------------------------------------------------------
    @staticmethod
    def _as_np(a):
        if isinstance(a, (list, tuple)):
            try:
                return np.asarray(a)
            except Exception:
                return a
        if hasattr(a, "detach"):
            return np.asarray(a.detach().cpu())
        return a

    def fits(self, args):
        """True when ``args`` have this entry's argument shapes (used to choose between entries of one name)."""
        if len(args) != self.arity:
            return False
        for ty, a in zip(self.types, args):
            if isinstance(a, Buffer):
                return a.entry is self
            a = self._as_np(a)
            nd = np.ndim(a)
            if nd not in (len(ty.shape), len(ty.shape) + 1) or tuple(np.shape(a))[nd - len(ty.shape):] != ty.shape:
                return False
        return True

    def prepare(self, args):
        """
        Check ``args`` against the signature, as call_entry does, and return (arrays, n, many): one array of shape
        (n,) + parameter shape per argument. An argument with one extra leading dimension holds n calls; all arguments must agree on n.
        A Buffer is the whole input: returned as (buffer, n, True). ``many`` is True when an argument had the extra dimension.
        """
        sig = self.sig
        if self.reader:
            if args:
                raise BadCall("arity", "entry %s is declared \"%s\" and takes no argument, got %d" % (self.name, sig, len(args)))
            return [], 0, False
        if any(isinstance(a, Buffer) for a in args):
            if len(args) != 1:
                raise BadCall("type", "a buffer holds all the arguments of its calls: pass it alone, as %s(buffer)" % self.qualified)
            b = args[0]
            if b.freed:
                raise BadCall("type", "the buffer was freed")
            if b.entry is not self:
                raise BadCall("type", "the buffer holds calls of %s, not of %s" % (b.entry.qualified, self.qualified))
            return b, b.n, True
        if len(args) != self.arity:
            raise BadCall("arity", "entry %s is declared \"%s\" and takes %d argument(s), got %d" %
                          (self.name, sig, self.arity, len(args)))
        cols, n, many_any = [], None, False
        for i, (ty, a) in enumerate(zip(self.types, args)):
            what = "argument %d of entry %s" % (i + 1, self.name)
            a = self._as_np(a)
            many = np.ndim(a) == len(ty.shape) + 1
            many_any = many_any or many
            try:
                if many:
                    k = int(np.shape(a)[0])
                    if k < 1:
                        raise Mismatch("shape", "%s holds no calls" % what)
                    v = conform(Ty(ty.elem, (k,) + ty.shape), a, what)
                else:
                    k = 1
                    v = conform(ty, a, what)
            except Mismatch as m:
                raise BadCall(m.kind, "%s (declared \"%s\"; an extra leading dimension means several calls)" % (m.message, sig))
            if n is not None and k != n:
                raise BadCall("shape", "the arguments hold %d and %d calls (declared \"%s\")" % (n, k, sig))
            n = k
            v = np.asarray(v)
            if ty.elem == "int":
                lim = 1 << (self.int_bits - 1)
                if v.size and (v.min() < -lim or v.max() >= lim):
                    raise BadCall("range", "%s does not fit a %d-bit int (the kernel's int)" % (what, self.int_bits))
            cols.append(v.reshape((n,) + ty.shape))
        return cols, n, many_any

    # -- running --------------------------------------------------------------------------
    def kernel(self):
        if self.mixed:
            raise BadCall("type", "entry %s mixes int and float arguments in one record, which the host side does not support yet" % self.name)
        return self.binding.live_session().kernel(self.top)

    def put(self, args):
        """Upload the records of ``args`` once; returns a Buffer."""
        cols, n, many = self.prepare(args)
        if isinstance(cols, Buffer):
            raise BadCall("type", "that is a buffer already")
        k = self.kernel()
        recs, _ = k.pack(cols)
        h, _ = k.s.put(recs)
        h.dtype, h.shape = recs.dtype, recs.shape
        return Buffer(self.binding, self, h, n, many)

    def call(self, args, reset=False):
        """
        Run on the calls in ``args``: (array of shape (n,) + result shape, many). ``many`` is False for one plain call.
        Raises HwTrap when a trap flag was set (the results are not returned). The first call after a ``.fpga`` binding,
        and the first after ``reset_next``, starts the program from its initial state.
        """
        cols, n, many = self.prepare(args)
        k = self.kernel()
        reset = reset or self.pending_reset
        try:
            if isinstance(cols, Buffer):
                out = k.run(cols.handle, reset=reset)
            else:
                out = k.run(*cols, reset=reset)
        finally:
            self.last = k.last
        self.pending_reset = False
        return out, (False if self.summary else many)

    def reset_next(self):
        self.pending_reset = True


class Binding:
    """The programs of one ``.fpga`` call, the image that holds them, and the session that runs them."""
    def __init__(self, session, mode, files, outputs, path, build, loaded):
        self.session, self.mode, self.files, self.outputs = session, mode, list(files), outputs
        self.path, self.build, self.loaded = path, build, loaded
        self.entries = {}                    # "prog.entry" -> BoundEntry
        self.by_name = {}                    # "entry" -> [BoundEntry]
        self.pending = {}                    # kernel -> True while its next run must reset the program
        for o in outputs:
            m = o.manifest
            for e in m["entries"]:
                made = [BoundEntry(self, m, e)]
                if e.get("output") == "summary":
                    made.append(BoundEntry(self, m, e, reader=True))
                for be in made:
                    self.entries[be.qualified + ("" if len(self.by_name.get(be.name, ())) == 0 else "#%d" % (len(self.by_name[be.name]) + 1))] = be
                    self.by_name.setdefault(be.name, []).append(be)
        self.buffers = []

    def registers(self):
        """[(manifest, register dict)] of every program of the binding."""
        return [(o.manifest, r) for o in self.outputs for r in o.manifest.get("registers", [])]

    def set_register(self, name, value):
        """
        Set the ``.param`` register ``name`` of every program of this binding that declares it (the kernels keep the
        value until it is set again). ``value`` is converted as the interpreter does (``set_param``: int, float or
        fixed-point from a number or an array); with ``--lanes N`` it is an array of N values, one per lane.
        Returns the number of kernels set. Raises BadCall (type, shape, range) and NoSuchRegister.
        """
        found = [(m, r) for m, r in self.registers() if r["name"] == name]
        if not found:
            have = sorted({r["name"] for _m, r in self.registers()})
            raise HwError("NoSuchRegister", "%s is not a parameter of the program%s" %
                          (name, "; the parameters are " + ", ".join(have) if have else " (it declares none with .param)"))
        s = self.live_session()
        count = 0
        for m, r in found:
            lanes = m.get("lanes", 1) if m.get("lane_dimension") else 0
            ty = Ty(r["elem"], (lanes,) if lanes else ())
            what = "the value of parameter %s" % name
            try:
                v = conform(ty, value, what)
            except Mismatch as e:
                raise BadCall(e.kind, "%s (declared %s%s)" % (e.message, r["elem"], ", one value per lane: an array of %d" % lanes if lanes else ""))
            arr = np.asarray(v.raw if hasattr(v, "raw") else v)
            if r["elem"] == "int" and arr.size:
                lim = 1 << (8 * r["bytes"] - 1)
                if arr.min() < -lim or arr.max() >= lim:
                    raise BadCall("range", "%s does not fit a %d-bit int (the kernel's int)" % (what, 8 * r["bytes"]))
            for e in m["entries"]:
                s.kernel(e["top"]).set_register(name, arr)
                count += 1
        return count

    @property
    def simulated(self):
        return self.mode == "sim"

    def live_session(self):
        """The session, if this binding's image is still the one loaded in it."""
        s = self.session
        if getattr(s, "image", None) != self.path or _SESSIONS.get(self.mode) is not s:
            raise HwError("ImageReplaced", "the image of this binding is no longer loaded (another .fpga call loaded a different "
                          "image, or the session ended); call .fpga again")
        return s

    def info(self):
        s = self.session
        d = {}
        try:
            d = s.info()
        except HwError:
            pass
        return d


def bind(files, clock_mhz=None, env=None, lanes=None):
    """
    Compile ``files`` (static-profile programs), find the image that holds them, load it and return the Binding.
    ``lanes``: the image was built with ``--lanes N``. ``clock_mhz`` None: the clock of the image that is found
    (300 MHz for the simulator). Raises HwNotAvailable, NoImage, HwRefused (timing, card busy), HwConnectionError,
    and the static checker's errors.
    """
    mode = choose_mode(env)
    if mode is None:
        raise HwNotAvailable("no hardware to run on: " + _TEMPLATE_HINT)
    outputs = _program_outputs(files, DEFAULT_CLOCK if clock_mhz is None else clock_mhz, lanes)
    try:
        s = session_for(mode)
        if mode == "sim":
            path, build = s.prepare(outputs)
        else:
            from .client import host_target
            host = host_target() if mode == "card" else None
            try:
                path, build = find_image_for(files, host, clock_mhz, lanes)
            except NoImage as e:
                raise NoImage("%s" % _with_command(str(e), files, clock_mhz, lanes))
        if getattr(s, "image", None) == path and getattr(s, "loaded_reply", None):
            loaded = s.loaded_reply          # this image is the one this session loaded: nothing to replace
        else:
            loaded = s.loaded_reply = s.load(path)
    except HwConnectionError:
        drop_session(mode)
        raise
    b = Binding(s, mode, files, outputs, path, build, loaded)
    return b


def _with_command(msg, files, clock_mhz, lanes=None):
    cmd = build_command(files, clock_mhz, lanes)
    return msg if cmd in msg else "%s. Build it with: %s" % (msg.rstrip("."), cmd)
