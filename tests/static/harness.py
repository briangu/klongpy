"""
The differential harness of the static compiler: the same
program runs under KlongPy and as compiled C, and the two must agree.

    cc = find_cc()                       None when there is no C compiler (tests then skip)
    b = Build(text, name="t", int_bits=32, float_bits=32, defines={...}, externs={...})
    b.call("f", 1.5)                     -> ("ok", value) or ("trap", code)
    b.state("buf")                       the value of a state global of the C instance
    h = Host(text, externs={...})
    h.call("f", 1.5)                     -> ("ok", value) or ("error", exception)
    same(host_value, c_value, ty, float_bits)    None, or a description of the difference

How the C side runs, and why

The emitted C is compiled for the host into a shared library together with
a small generated shim and loaded with ctypes. A generated main() that
prints results was the other candidate; the library was chosen because

  * the externs of both worlds are then the *same Python objects* (the
    shim forwards each extern to a ctypes callback), so a stand-in with
    state or a call log cannot drift between a Python and a C version of it;
  * values cross as binary ints and floats, with no printing and parsing
    in between, so floats can be compared bit for bit;
  * one compile serves any number of calls, which is what makes thousands
    of generated inputs affordable.

The shim defines kg_trap (the strong definition replaces the weak default)
to longjmp back to a guard around each entry, so a trap is an outcome the
test can compare, and gives accessors for the state globals.

How values are compared

Ints must be equal. Floats are compared bit for bit (any nan equals any
nan, and -0.0 differs from 0.0) when the build's kg_float is 64 bits wide
and the test does not pass a tolerance: the host computes in float64, the C
is built with -ffp-contract=off, so the same operations in the same order
give the same bits. A 32-bit kg_float cannot reproduce float64 results, so
those builds are compared within a relative tolerance (1e-5 by default,
plus the same absolute tolerance), unless the test says the values are
exact in both widths.

The reference is the numpy backend at strict level 2, whatever backend the
test session selected: that is the interpreter the compiler promises to
match (the torch backend computes in float32).
"""
import ctypes
import os
import shutil
import subprocess
import tempfile

import numpy as np

from klongpy.errors import KlongError
from klongpy.static import fixed as fx
from klongpy.static.native import call_entry, emit_program, ir_json, load_program
from klongpy.static.signature import Ty, c_identifier, is_fixed, parse_signature

WARN = ["-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic"]

_KEEP = []      # temp dirs and libraries stay alive for the test session


def find_cc():
    """The host C compiler: $KG_TEST_CC, else cc, clang or gcc on the path. None when there is none."""
    for cand in (os.environ.get("KG_TEST_CC"), "cc", "clang", "gcc"):
        if cand and shutil.which(cand):
            return shutil.which(cand)
    return None


def sanitize_flags():
    """Extra flags when $KG_TEST_SANITIZE is set (for example "undefined"): reports abort the test process."""
    kinds = os.environ.get("KG_TEST_SANITIZE")
    if not kinds:
        return []
    return [f"-fsanitize={kinds}", "-fno-sanitize-recover=all", "-g"]


def ctype_of(elem):
    """The C type of an element: kg_float, kg_int, or the integer that holds a fixed-point number."""
    if elem == "float":
        return "kg_float"
    if is_fixed(elem):
        return "int32_t" if fx.spec_of(elem).W <= 32 else "int64_t"
    return "kg_int"


class _Decl:
    def __init__(self, name, signature):
        self.name, self.signature = name, signature


class _Entry:
    def __init__(self, d):
        self.decl = _Decl(d["name"], parse_signature(d["signature"]))
        self.c_name = d["c_name"]


class _Extern:
    def __init__(self, d):
        self.name = d["name"]
        self.signature = parse_signature(d["signature"])
        self.c_name = d["c_name"]


class _Global:
    def __init__(self, d):
        self.name, self.ty, self.state = d["name"], Ty(d["ty"]["elem"], tuple(d["ty"]["shape"])), d["state"]


class _Program:
    """What the typed IR of the extension says about a program: its entries, externs and globals (ir_json)."""

    def __init__(self, ir):
        self.entries = [_Entry(e) for e in ir["entries"]]
        self.externs = [_Extern(x) for x in ir["externs"]]
        self.globals = [_Global(g) for g in ir["globals"]]


def _ct(ty, float_bits, int_bits):
    if is_fixed(ty.elem):
        return ctypes.c_int32 if fx.spec_of(ty.elem).W <= 32 else ctypes.c_int64
    if ty.elem == "float":
        return ctypes.c_double if float_bits == 64 else ctypes.c_float
    return ctypes.c_int64 if int_bits == 64 else ctypes.c_int32


def _np(ty, float_bits, int_bits):
    if is_fixed(ty.elem):
        return np.int32 if fx.spec_of(ty.elem).W <= 32 else np.int64
    if ty.elem == "float":
        return np.float64 if float_bits == 64 else np.float32
    return np.int64 if int_bits == 64 else np.int32


class Build:
    """A program compiled to C, built as a shared library for the host and loaded."""

    def __init__(self, text, name="t", int_bits=32, float_bits=32, defines=None, externs=None, opt="-O1",
                 freestanding=False, path=None):
        cc = find_cc()
        if cc is None:
            raise RuntimeError("no C compiler")
        self.int_bits = int_bits
        self.float_bits = float_bits
        self.dir = tempfile.mkdtemp(prefix="kgdiff_")
        _KEEP.append(self.dir)
        src = path or os.path.join(self.dir, name + ".kg")
        self.emitted = emit_program(src, self.dir, text=text, int_bits=int_bits, float_bits=float_bits, prefix=name,
                                    externs=externs)
        self.program = _Program(ir_json(src, text=text, int_bits=int_bits, prefix=name))
        self.prefix = P = self.emitted.prefix
        self.externs = dict(externs or {})
        shim = self._shim()
        with open(os.path.join(self.dir, "shim.c"), "w") as f:
            f.write(shim)
        lib = os.path.join(self.dir, f"lib{P}.so")
        flags = list(WARN) + [opt, "-ffp-contract=off", "-fPIC", "-shared"] + sanitize_flags()
        if freestanding:
            flags += ["-DKG_FREESTANDING", "-DKG_NO_MEM_BUILTINS"]
        for k, v in (defines or {}).items():
            flags.append(f"-D{k}={v}")
        cmd = [cc] + flags + [os.path.join(self.dir, P + ".c"), os.path.join(self.dir, "shim.c"), "-o", lib]
        if not freestanding:
            cmd.append("-lm")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError(f"the emitted C does not compile:\n{' '.join(cmd)}\n{r.stderr}\n--- {P}.c\n{self.emitted.c}")
        self.lib = ctypes.CDLL(lib)
        _KEEP.append(self.lib)
        self.lib[f"{P}_size"].restype = ctypes.c_size_t
        self.size = self.lib[f"{P}_size"]()
        self._callbacks = []
        self._bind_externs()
        self.entries = {}
        for e in self.program.entries:
            self.entries.setdefault(e.decl.name, []).append(e)
        self.reset()

    # -- the shim ------------------------------------------------------------------
    def _shim(self):
        P = self.prefix
        out = ["#include <setjmp.h>", f'#include "{P}.h"', "",
               "static jmp_buf kgt_jb;", "static int kgt_code;", "int kgt_line;", "",
               "void kg_trap(int code, int line) {", "    kgt_code = code;", "    kgt_line = line;",
               "    longjmp(kgt_jb, 1);", "}", ""]
        ct = lambda t: ctype_of(t.elem)  # noqa: E731
        for x in self.program.externs:
            sig = x.signature
            decl, args = [], []
            for i, ty in enumerate(sig.params):
                if ty.elem == "str":
                    decl.append(f"const char *a{i}")
                elif ty.shape:
                    decl.append(f"const {ct(ty)} *a{i}")
                else:
                    decl.append(f"{ct(ty)} a{i}")
                args.append(f"a{i}")
            r = sig.result
            if r.shape:
                decl.append(f"{ct(r)} *out")
                args.append("out")
            rtype = "void" if (r.shape or r.elem == "void") else ct(r)
            plist = ", ".join(decl) or "void"
            out += [f"static {rtype} (*kgt_fp_{x.c_name})({plist});",
                    f"void kgt_set_{x.c_name}({rtype} (*f)({plist})) {{", f"    kgt_fp_{x.c_name} = f;", "}",
                    f"{rtype} {x.c_name}({plist}) {{",
                    f"    {'return ' if rtype != 'void' else ''}kgt_fp_{x.c_name}({', '.join(args)});", "}", ""]
        for e in self.program.entries:
            sig = e.decl.signature
            decl, args = [f"{P}_t *p"], ["p"]
            for i, ty in enumerate(sig.params):
                decl.append(f"const {ct(ty)} *a{i}" if ty.shape else f"{ct(ty)} a{i}")
                args.append(f"a{i}")
            r = sig.result
            if r.elem != "void":
                decl.append(f"{ct(r)} *res")
            call = f"{P}_{e.c_name}({', '.join(args + (['res'] if r.shape else []))})"
            out += [f"int kgt_{e.c_name}({', '.join(decl)}) {{",
                    "    if (setjmp(kgt_jb)) {", "        return kgt_code;", "    }",
                    f"    {'*res = ' if (r.elem != 'void' and not r.shape) else ''}{call};",
                    "    return 0;", "}", ""]
        for g in self.program.globals:
            if not g.state:
                continue
            cid = g.name.replace(".", "_")
            n = g.ty.size
            out += [f"void kgt_get_{cid}({P}_t *p, {ct(g.ty)} *out) {{"]
            if g.ty.shape:
                out += [f"    for (kg_int i = 0; i < {n}; i++) {{", f"        out[i] = p->g_{cid}[i];", "    }"]
            else:
                out += [f"    *out = p->g_{cid};"]
            out += ["}", ""]
        return "\n".join(out)

    def _bind_externs(self):
        fb, ib = self.float_bits, self.int_bits
        for x in self.program.externs:
            sig = x.signature
            fn = self.externs.get(x.c_name)        # stand-ins are keyed by C symbol
            if fn is None:
                continue
            argtypes = []
            for ty in sig.params:
                if ty.elem == "str":
                    argtypes.append(ctypes.c_char_p)
                elif ty.shape:
                    argtypes.append(ctypes.POINTER(_ct(ty, fb, ib)))
                else:
                    argtypes.append(_ct(ty, fb, ib))
            r = sig.result
            if r.shape:
                argtypes.append(ctypes.POINTER(_ct(r, fb, ib)))
            restype = None if (r.shape or r.elem == "void") else _ct(r, fb, ib)

            def make(fn=fn, sig=sig, r=r):
                def cb(*cargs):
                    args = []
                    for ty, a in zip(sig.params, cargs):
                        if ty.elem == "str":
                            args.append(a.decode())
                        elif ty.shape:
                            arr = np.ctypeslib.as_array(a, shape=(ty.size,)).reshape(ty.shape)
                            args.append(arr.astype(np.float64 if ty.elem == "float" else np.int64))
                        else:
                            args.append(float(a) if ty.elem == "float" else int(a))
                    v = fn(*args)
                    if r.shape:
                        flat = np.asarray(v, dtype=_np(r, fb, ib)).reshape(-1)
                        for i in range(r.size):
                            cargs[-1][i] = flat[i]
                        return None
                    if r.elem == "void":
                        return None
                    return float(v) if r.elem == "float" else int(v)
                return cb
            proto = ctypes.CFUNCTYPE(restype, *argtypes)
            cb = proto(make())
            self._callbacks.append(cb)
            setter = self.lib[f"kgt_set_{x.c_name}"]
            setter.argtypes = [proto]
            setter.restype = None
            setter(cb)

    # -- running ---------------------------------------------------------------------
    def reset(self):
        """A fresh instance, initialised."""
        self.mem = ctypes.create_string_buffer(max(self.size, 8) + 64)
        self.lib[f"{self.prefix}_init"](self.mem)

    def _entry(self, name, args):
        cands = self.entries[name]
        if len(cands) == 1:
            return cands[0]
        for e in cands:
            ps = e.decl.signature.params
            if len(ps) == len(args) and all(tuple(np.shape(a)) == tuple(p.shape) for a, p in zip(args, ps)):
                return e
        raise KeyError(name)

    def call(self, name, *args):
        e = self._entry(name, args)
        sig = e.decl.signature
        fb, ib = self.float_bits, self.int_bits
        cargs = [self.mem]
        keep = []
        for ty, a in zip(sig.params, args):
            if ty.shape:
                arr = np.ascontiguousarray(np.asarray(a).astype(_np(ty, fb, ib)))
                keep.append(arr)
                cargs.append(arr.ctypes.data_as(ctypes.POINTER(_ct(ty, fb, ib))))
            else:
                cargs.append(_ct(ty, fb, ib)(a))
        r = sig.result
        res = None
        if r.shape:
            res = np.zeros(r.shape, dtype=_np(r, fb, ib))
            cargs.append(res.ctypes.data_as(ctypes.POINTER(_ct(r, fb, ib))))
        elif r.elem != "void":
            res = _ct(r, fb, ib)()
            cargs.append(ctypes.byref(res))
        fn = self.lib[f"kgt_{e.c_name}"]
        fn.restype = ctypes.c_int
        code = fn(*cargs)
        if code:
            return ("trap", code)
        if r.elem == "void":
            return ("ok", None)
        return ("ok", res if r.shape else res.value)

    def state(self, name):
        g = next(g for g in self.program.globals if g.name == name)
        fb, ib = self.float_bits, self.int_bits
        fn = self.lib[f"kgt_get_{name.replace('.', '_')}"]
        fn.restype = None
        if g.ty.shape:
            out = np.zeros(g.ty.shape, dtype=_np(g.ty, fb, ib))
            fn(self.mem, out.ctypes.data_as(ctypes.POINTER(_ct(g.ty, fb, ib))))
            return out
        out = _ct(g.ty, fb, ib)()
        fn(self.mem, ctypes.byref(out))
        return out.value


class Host:
    """The same program under KlongPy (numpy backend, strict level 2)."""

    def __init__(self, text, name="t.kg", externs=None, path=None):
        self.text, self.name, self.externs, self.path = text, name, externs, path
        self.reset()

    def reset(self):
        self.klong = load_program(self.path or self.name, externs=self.externs, text=self.text, strict=2, backend="numpy")

    def call(self, name, *args, raw=False):
        try:
            return ("ok", call_entry(self.klong, name, *args, raw=raw))
        except KlongError as e:
            return ("error", e)

    def state(self, name):
        v = self.klong[name]
        return v.item() if getattr(v, "ndim", None) == 0 else v


# ---------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------

def same(host, c, ty, float_bits, rtol=None, atol=None, int_bits=64, zero_sign=True):
    """
    None when the C value equals the host value, else a one-line description.
    rtol None means: bit for bit when kg_float is 64 bits wide, 1e-5 when it is 32.
    rtol 0 means bit for bit after rounding the host value to the width of kg_float.
    zero_sign=False lets 0.0 equal -0.0 in a bit-for-bit comparison (for programs
    that sum with +/: numpy's sum of negative zeros is +0.0, see test_differential).
    """
    if ty.elem == "void":
        return None
    h = np.asarray(host)
    v = np.asarray(c)
    if h.shape != v.shape:
        return f"shape {h.shape} on the host, {v.shape} in C"
    if ty.elem == "int" or is_fixed(ty.elem):
        if h.dtype.kind == "f":
            if not np.all(h == np.floor(h)):
                return f"host {host!r} is not whole, C {c!r}"
        if not np.array_equal(h.astype(object), v.astype(object)):
            return f"host {host!r}, C {c!r}"
        return None
    h = h.astype(np.float64)
    v = v.astype(np.float64)
    if rtol is None and float_bits == 64:
        rtol = 0
    if rtol == 0:
        if float_bits == 32:
            with np.errstate(over="ignore"):
                h = h.astype(np.float32).astype(np.float64)
        ok = ((h == v) & ((np.signbit(h) == np.signbit(v)) | (not zero_sign))) | (np.isnan(h) & np.isnan(v))
        return None if np.all(ok) else f"host {host!r}, C {c!r} (bits differ)"
    rtol = 1e-5 if rtol is None else rtol
    atol = rtol if atol is None else atol
    if float_bits == 32:
        with np.errstate(over="ignore"):
            h = h.astype(np.float32).astype(np.float64)    # what is beyond float32 is an infinity there too
    ok = np.isclose(h, v, rtol=rtol, atol=atol, equal_nan=True)
    return None if np.all(ok) else f"host {host!r}, C {c!r} (beyond rtol {rtol})"


def wrap(v, bits):
    """A Python int wrapped to a signed integer of this width (what compiled arithmetic gives)."""
    m = 1 << bits
    v = int(v) % m
    return v - m if v >= m // 2 else v
