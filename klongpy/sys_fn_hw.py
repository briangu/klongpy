"""
.fpga: call a built FPGA kernel from Klong like a function.

    fir::.fpga("fir.kg")             bind the kernels of the image built from fir.kg (kgpy fpga build fir.kg)
    y::fir.run(x)                  runs on the card; the same program without .fpga runs interpreted

See docs/hardware.md. The work is in klongpy/hw/binding.py; nothing from klongpy/hw is imported until .fpga is used,
and torch is never imported here: arrays cross to the card as numpy and come back through the session's backend.
"""
import os
import sys

import numpy as np

from .errors import (KlongArityError, KlongError, KlongNameError, KlongRangeError, KlongShapeError, KlongSystemError,
                     KlongTypeError)
from .types import KGLambda, KGSym

__all__ = ["KlongHwError", "KlongHwImageError", "KlongHwBusyError", "KlongHwConnectionError", "create_system_functions_hw"]


class KlongHwError(KlongSystemError):
    """.fpga cannot run: no hardware to run on, or the hardware answered with an error (kind "system")."""


class KlongHwImageError(KlongHwError):
    """There is no usable image for the program (not built, built from other source, timing failed, replaced)."""


class KlongHwBusyError(KlongHwError):
    """The card is in use by another process, or a compute unit is not idle; nothing was loaded."""


class KlongHwConnectionError(KlongHwError):
    """The connection to the machine with the card was lost, or could not be made."""


def _translate(e, entry=None):
    """The KlongError for an error of klongpy/hw."""
    from .hw.binding import BadCall, HwNotAvailable, NoImage
    from .hw.client import HwConnectionError, HwRefused, HwTrap
    if isinstance(e, KlongError):
        return e
    if isinstance(e, BadCall):
        cls = {"type": KlongTypeError, "shape": KlongShapeError, "range": KlongRangeError, "arity": KlongArityError}[e.what]
        return cls(e.message)
    if isinstance(e, HwTrap):
        name = entry.qualified if entry is not None else "?"
        why = []
        if e.status & 2:
            why.append("an index out of range")
        if e.status & 4:
            why.append("an integer division by zero")
        return KlongRangeError("entry %s trapped on the hardware: %s (status 0x%x); the results of this run are not returned"
                               % (name, " and ".join(why) or "a trap", e.status),
                               hint="the interpreted program raises the same error at the first call that does it")
    if isinstance(e, NoImage):
        return KlongHwImageError(e.message)
    if isinstance(e, HwNotAvailable):
        return KlongHwError(e.message)
    if isinstance(e, HwConnectionError):
        return KlongHwConnectionError("%s; the session is closed, call .fpga again to reconnect" % e.message)
    if isinstance(e, HwRefused):
        if e.kind == "CardBusy":
            return KlongHwBusyError("%s" % e.message)
        return KlongHwImageError("the image was refused: %s" % e.message)
    kind = getattr(e, "kind", "")
    if kind == "NoSuchRegister":
        return KlongNameError(e.message, hint='declare it in the program: .param("name";"int")')
    if kind in ("ImageReplaced",):
        return KlongHwImageError(e.message)
    if kind in ("NoSuchHandle",):
        return KlongTypeError("the buffer is not on the card any more (the image was replaced or the session ended)")
    if kind in ("TooBig",):
        return KlongRangeError(e.message)
    if kind in ("NoSimulator",):
        return KlongHwError(e.message)
    if kind == "Timeout":
        return KlongHwError(e.message)
    return KlongHwError(getattr(e, "message", str(e)))


def _guard(entry=None):
    """Context manager: errors of klongpy/hw become KlongErrors."""
    class G:
        def __enter__(self):
            return self

        def __exit__(self, et, ev, tb):
            if ev is None or isinstance(ev, KlongError):
                return False
            from .hw.client import HwError
            from .hw.remote import RemoteError
            if isinstance(ev, (HwError,)):
                raise _translate(ev, entry) from None
            if isinstance(ev, RemoteError):
                raise KlongHwConnectionError(str(ev)) from None
            if isinstance(ev, (OSError, EOFError)) and not isinstance(ev, FileNotFoundError):
                raise KlongHwConnectionError("%s: %s" % (type(ev).__name__, ev)) from None
            return False
    return G()


class HwEntryFn(KGLambda):
    """The Klong value of one kernel: calling it runs the entry on the hardware."""
    def __init__(self, bound):
        def call(klong, *args):
            return _call(klong, bound, args)
        KGLambda.__init__(self, call, wildcard=True)
        self.name = bound.qualified
        self.hw_entry = bound


def _values(elem, raw):
    """Words from the card as Klong numbers: ints as int64, floats as float64, fixed-point as float64 (exact to 53 bits)."""
    from .static import fixed as fx
    from .static.signature import is_fixed
    raw = np.asarray(raw)
    if is_fixed(elem):
        return fx.to_float(raw.astype(np.int64), fx.spec_of(elem))
    return raw.astype(np.int64 if elem == "int" else np.float64)


def _result_value(klong, bound, out, many):
    ty = bound.result_ty                 # the type the kernel returns: lanes in front, the summary for a driver
    out = _values(ty.elem, out)
    if not many:
        if not bound.summary:
            out = out[0]
        if not ty.shape:
            return int(out) if ty.elem == "int" else float(out)
    return klong._backend.kg_asarray(out)


def _call(klong, bound, args):
    # entries of one name with different shapes: the one whose parameters fit
    group = bound.binding.by_name.get(bound.name, [bound])
    if len(group) > 1:
        for cand in group:
            if cand.fits(args):
                bound = cand
                break
    with _guard(bound):
        out, many = bound.call(list(args))
    return _result_value(klong, bound, out, many)


def _paths(x):
    if isinstance(x, str):
        files = [x]
    else:
        try:
            files = list(x)
        except TypeError:
            files = []
        if not files or not all(type(f) is str for f in files):
            raise KlongTypeError(".fpga: the argument must be the path of a Klong program or a list of paths")
    for f in files:
        if not os.path.isfile(f):
            raise KlongSystemError(".fpga: no such file: %s" % f)
    return files


def _plain(v):
    """A build.json value as a Klong value: dictionaries of scalars (lists are dropped, booleans are 0 and 1)."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items() if not isinstance(x, (list, tuple))}
    return v


def _entry_of(x, who):
    b = getattr(x, "hw_entry", None)
    if b is None:
        if isinstance(x, dict) and len(x) == 1:
            b = getattr(next(iter(x.values())), "hw_entry", None)
        if b is None:
            raise KlongTypeError("%s: the first argument must be a kernel from .fpga, for example fir.run" % who)
    return b


def eval_sys_hw(klong, x):
    """

        .fpga(x)                                                  [FPGA]

        Bind the FPGA kernels of a built program. "x" is the path of
        a Klong program of the static profile that has .entry
        declarations, or a list of paths for an image that was
        built from several programs (kgpy fpga build a.kg b.kg).
        The program is checked and compiled, the image built from
        exactly this source is looked up on the card machine
        ($KLONG_FPGA_HOST; the build cache must be there) and loaded, and a dictionary
        from entry name to function is returned. Each entry is also
        bound as "program.entry", where "program" is the file name
        without ".kg": fir::.fpga("fir.kg") makes fir.run, which is
        also fir@"run".

        A call checks its arguments against the entry's signature
        and runs on the card; an argument with one extra leading
        dimension means that many calls in one run, and the result
        has that dimension too. The program's state is kept between
        calls and starts fresh at the first call after .fpga.

        With KLONG_FPGA=sim the kernels run on this machine's CPU: the
        generated HLS code compiled with the C++ compiler. That is a
        simulator, not hardware. There is no image to build then.

        Without KLONG_FPGA=sim, KLONG_FPGA_HOST or pyxrt, .fpga is an
        error that says so. Loading an image replaces what the card
        held. See docs/hardware.md.

        Examples:   fir::.fpga("fir.kg")
                    fir.run(256#1.0)

    """
    return _bind(klong, _paths(x), None)


def _bind(klong, files, lanes):
    with _guard():
        from .hw import binding
        b = binding.bind(files, lanes=lanes)
    table, qualified = {}, {}
    for be in b.entries.values():
        fn = HwEntryFn(be)
        table[be.name if be.name not in table else be.qualified] = fn
        qualified[be.qualified] = fn
    for q, fn in qualified.items():
        klong[KGSym(q)] = fn
    klong.hw_binding = b
    return table


def eval_sys_hw_lanes(klong, x, y):
    """

        .fpga.lanes(x;y)                                    [FPGA-Lanes]

        Bind the kernels of a program built with --lanes y
        (kgpy fpga build FILE --lanes y). Like .fpga: "x" is a path
        or a list of paths. The kernel then runs y copies of the
        program on the same stream of calls: every .param is set
        with .fpga.set as an array of y values, and a result has a
        leading dimension y.

        Example:   s::.fpga.lanes("crossover.kg";64)

    """
    if isinstance(y, bool) or not isinstance(y, (int, np.integer)) or y < 1:
        raise KlongTypeError(".fpga.lanes: the number of lanes must be an integer from 1")
    return _bind(klong, _paths(x), int(y))


def _binding_of(x, who):
    b = getattr(x, "hw_entry", None)
    if b is None and isinstance(x, dict):
        for v in x.values():
            b = getattr(v, "hw_entry", None)
            if b is not None:
                break
    if b is None:
        raise KlongTypeError("%s: the first argument must be what .fpga returned, or a kernel from it (fir.run)" % who)
    return b.binding


def eval_sys_hw_set(klong, x, y, z):
    """

        .fpga.set(x;y;z)                                      [FPGA-Set]

        Set the parameter "y" (declared in the program with
        .param) of the kernels of "x" to the value "z", between
        runs. "x" is what .fpga returned, or one of its kernels.
        The value is converted as the interpreter converts it
        (a fixed-point parameter takes a number, truncating toward
        minus infinity); for a kernel bound with .fpga.lanes it is
        an array with one value per lane. The kernel keeps the
        value, through resets, until it is set again. Returns the
        number of kernels set.

        Example:   .fpga.set(s;"fast";[3 5 8])

    """
    binding = _binding_of(x, ".fpga.set")
    if type(y) is not str:
        raise KlongTypeError(".fpga.set: the name of the parameter must be a string")
    with _guard():
        return binding.set_register(str(y), z)


def eval_sys_hw_put(klong, x, y):
    """

        .fpga.put(x;y)                                        [FPGA-Put]

        Upload the arguments of calls of the kernel "x" to the
        card once and return a buffer. "y" is the argument of the
        kernel (with the extra leading dimension for many calls)
        or, for a kernel with several arguments, a list of them.
        A buffer is accepted by the kernel in place of its
        arguments: fir.run(b). It stays on the card until it is
        freed with .fpga.free or the image is unloaded or the session
        ends.

        Example:   b::.fpga.put(fir.run;x)
                   fir.run(b)

    """
    bound = _entry_of(x, ".fpga.put")
    if bound.arity == 1:
        args = [y]
    else:
        try:
            args = list(y)
        except TypeError:
            raise KlongTypeError(".fpga.put: %s takes %d arguments: pass them as a list" % (bound.qualified, bound.arity)) from None
    with _guard(bound):
        buf = bound.put(args)
    bound.binding.buffers.append(buf)
    return buf


def eval_sys_hw_get(klong, x):
    """

        .fpga.get(x)                                          [FPGA-Get]

        Download a buffer made by .fpga.put as an array of the
        arguments it holds (for a kernel with several arguments,
        the records: one row per call, the arguments flattened one
        after the other).

        Example:   .fpga.get(b)

    """
    from .hw.binding import Buffer
    if not isinstance(x, Buffer):
        raise KlongTypeError(".fpga.get: the argument must be a buffer from .fpga.put")
    if x.freed:
        raise KlongTypeError(".fpga.get: the buffer was freed")
    with _guard(x.entry):
        x.entry.binding.live_session()
        k = x.entry.kernel()
        data = k.s.get(x.handle)
    e = x.entry
    recs = np.asarray(data).reshape(x.n, -1)
    if e.arity == 1:
        ty = e.types[0]
        arr = _values(ty.elem, recs.reshape((x.n,) + ty.shape))
        if not x.many:
            arr = arr[0]
            if not ty.shape:
                return int(arr) if ty.elem == "int" else float(arr)
        return klong._backend.kg_asarray(arr)
    return klong._backend.kg_asarray(recs.astype(np.float64 if recs.dtype.kind == "f" else np.int64))


def eval_sys_hw_free(klong, x):
    """

        .fpga.free(x)                                        [FPGA-Free]

        Free a buffer made by .fpga.put. Returns 1. Freeing a buffer
        twice is an error.

        Example:   .fpga.free(b)

    """
    from .hw.binding import Buffer
    if not isinstance(x, Buffer):
        raise KlongTypeError(".fpga.free: the argument must be a buffer from .fpga.put")
    if x.freed:
        raise KlongTypeError(".fpga.free: the buffer was already freed")
    x.freed = True
    b = x.entry.binding
    if x in b.buffers:
        b.buffers.remove(x)
    with _guard(x.entry):
        try:
            x.entry.binding.live_session().free(x.handle)
        except Exception as e:
            from .hw.client import HwError
            if not (isinstance(e, HwError) and e.kind in ("NoSuchHandle", "ImageReplaced")):
                raise
    return 1


def eval_sys_hw_reset(klong, x):
    """

        .fpga.reset(x)                                      [FPGA-Reset]

        Make the next call of the kernel "x" start the program from
        its initial state (the program's globals as initialised
        when the program was loaded). Returns 1.

        Example:   .fpga.reset(mavg.step)

    """
    bound = _entry_of(x, ".fpga.reset")
    bound.reset_next()
    return 1


def eval_sys_hw_info(klong):
    """

        .fpga.info()                                         [FPGA-Info]

        A dictionary describing the hardware in use: "mode"
        ("card", "local" or "sim"), "device", "shell", the loaded
        "image" (path), its "key", "kernels", the "build" record
        of the image (timing, utilisation, build time), and the
        resident "buffers" and "buffer-bytes". It does not load
        anything. Without a hardware to use it is an error.

        Example:   .fpga.info()

    """
    from .hw import binding
    with _guard():
        mode = binding.choose_mode()
        if mode is None:
            raise KlongHwError("no hardware to run on: " + binding._TEMPLATE_HINT)
        s = binding.session_for(mode)
        info = s.info()
        status = s.status()
    b = getattr(klong, "hw_binding", None)
    d = {"mode": mode, "device": str(info.get("device", "")), "shell": str(info.get("shell", "")),
         "card-image": str((info.get("session_image") or {}).get("uuid") or info.get("uuid") or ""), "image": "", "key": "", "kernels": "",
         "buffers": len(status.get("handles", {})), "buffer-bytes": int(status.get("bytes_in_use", 0))}
    if mode == "sim":
        d["note"] = "simulator: the generated HLS code compiled for this machine; not hardware"
    img = info.get("session_image") or status.get("image")
    if img:
        d["image"] = str(img.get("path", ""))
        d["kernels"] = ", ".join(img.get("kernels", []))
    if b is not None and b.session is s and img:
        d["key"] = str(b.build.get("key", ""))
        d["build"] = _plain({k: v for k, v in b.build.items() if k != "kernels"})
    return d


def create_system_functions_hw():
    def _get_name(s):
        i = s.index(".")
        return s[i : i + s[i:].index("(")]

    registry = {}
    m = sys.modules[__name__]
    for x in filter(lambda n: n.startswith("eval_sys_"), dir(m)):
        fn = getattr(m, x)
        registry[_get_name(fn.__doc__)] = fn
    return registry
