"""
The native (Rust) Klong interpreter as KlongPy's evaluation engine, the only one (klongpy/interpreter.py exports it as
klongpy.KlongInterpreter).

Needs the extension module (native/klong-py), found as klongpy._native, as klong_native on the import path, or in the directory
named by KLONGPY_NATIVE_PATH; without it importing klongpy is an ImportError that says how to build it.

KlongInterpreter has the public surface that programs and tests use: __call__, exec, apply, __getitem__,
__setitem__, __delitem__, __contains__, strict, names, backend, np, console, current_module.

What Python supplies (the rest is the engine's): the system functions that are Python by nature (.py .pyf, .web .ws, .srv .cli
IPC and timers on asyncio, tables and SQL, FPGA card I/O, the autograd functions), bridged as host functions; the backend's arrays
through the host table (rust_host.py); and the engine's hooks: the live console (.p .d .w .pf .oc .ic .cc .tc .fc .r .rl .mi are the
engine's own, on sys.stdout and sys.stdin or on what console() names), the warning hook (syntax warnings before the first statement,
through the warnings module), the stand-in lookup of .extern and .fixed (klong.externs), and one HostLambda for each Python callable.
The parse tree is not an interpreter's: `kgpy ast`, lint and the static tools are the extension's (klongpy/tools.py).
"""
import contextlib
import functools
import types
import sys
import os
import re
import time
import warnings
from weakref import ref

import numpy as np

from .backends import get_backend
from .errors import (compat, ERROR_KINDS, KlongArityError, KlongError, KlongInternalError, KlongSyntaxError, KlongSyntaxWarning, UnexpectedChar, UnexpectedEOF, python_error,
                     system_error)
from .source import Source
from .utils import check_strict_level
from .types import (KGSym, KGChar, KLONG_UNDEFINED, KGCall, KGFn, KGFnWrapper, KGLambda, KGOp, reserved_fn_args,
                    _arity_error, _count, _named_arity_error)

from .native_ext import load_extension as _load_extension
from .ipc.codec import KdbTable, KdbKeyedTable
from .table_bridge import _WireTableView


klong_native = _load_extension()
_KFn = klong_native.KFn
from .rust_host import result_for as _result_for, HostObject
_TYPES = {"KGSym": KGSym, "KGChar": KGChar, "undefined": KLONG_UNDEFINED}

class _Body:
    """What KGFn.a is for a native function: the function itself, so that apply(fn.a, ...) calls it (KGFnWrapper does).
    It carries the parameters and locals as KGBody does."""

    def __init__(self, fn):
        self.fn = fn

    @property
    def params(self):
        return self.fn.params

    @property
    def locals(self):
        return self.fn.locals

    def __repr__(self):
        return f"<body of {self.fn}>"


def _build_function(k, text, named):
    """The function a literal's text defines, built in the native interpreter k without defining anything: the plain
    literal as `fn::{...}`, the named form as written (`fn(a;b)::{...}`); `fn` is restored afterwards."""
    name = "fn"
    had = name in k
    saved = k[name] if had else None
    try:
        k.run(text if named else f"{name}::{text}")
        return k[name]
    finally:
        if had:
            k[name] = saved
        elif name in k:
            del k[name]


_scratch_klong = None


class _Bound:
    """A projection's argument that the engine did not report (see NativeFn.args)."""

    def __repr__(self):
        return "<bound>"


_BOUND = _Bound()


class _Detached:
    """The owner of a function that was copied or unpickled: it belongs to no interpreter until it goes into one."""

    def _out(self, v):
        return v

    def _call_value(self, kfn, args):
        raise TypeError("a copied function of the native engine runs only once it is put into an interpreter")


def _detached_fn(text, params):
    """What unpickling or copying a NativeFn gives: the function as a value of a scratch interpreter (its arity, params
    and locals can be read), to be rebuilt from its text in the interpreter it is assigned to."""
    global _scratch_klong
    if _scratch_klong is None:
        _scratch_klong = klong_native.Klong(0, _TYPES)
    named = params is not None
    if named:
        text = "fn(%s)::%s" % (";".join(params), text)
    kfn = _build_function(_scratch_klong, text, named)
    f = (NativeCall if kfn.is_projection else NativeFn)(kfn, _Detached())
    f._detached = (text, named)
    return f


def _syms(names):
    return None if names is None else tuple(KGSym(n) for n in names)


class NativeFn(KGFn):
    """A Klong function value of the native engine, with the face of KlongPy's KGFn, built on the accessors of the
    extension's KFn: arity (the holes of a projection), params, locals, the source text of the literal, the bound
    arguments of a projection (args: the value, None for a hole) and what it calls (callee).

    Not available from the native engine, so None: the parse-tree fields max_arg and uses_self and the spans of the
    original program (span, source). `source_text` is the literal as written.
    """
    span = None
    source = None
    user = False        # a function of the engine is never user Python (KGLambda.user)
    _detached = None    # (text, named) for a function that was copied or unpickled

    def __init__(self, kfn, owner):
        self.a = _Body(self)
        self.global_params = set()
        self._kfn = kfn
        self._owner = owner
        self._cache = {}

    def _get(self, name):
        try:
            return self._cache[name]
        except KeyError:
            v = self._cache[name] = getattr(self._kfn, name)
            return v

    @property
    def arity(self):
        return self._get("arity")

    @property
    def kind(self):
        return self._get("kind")

    @property
    def is_projection(self):
        return self._get("is_projection")

    @property
    def args(self):
        """None for a function; for a projection the list of its arguments in order, None where a hole is."""
        if not self.is_projection:
            return None
        v = self._cache.get("_args")
        if v is None:
            v = [self._owner._out(q) for q in self._kfn.bound_args()]
            holes = self._get("arity")
            if sum(1 for q in v if q is None) != holes:
                # engine need: bound_args reports a value written in the call (`r(10)`) as None, like a hole. The holes are the
                # engine's own count (arity); the bound values are not known, and stand as _BOUND. The holes are taken as the last ones.
                v = [_BOUND] * (len(v) - holes) + [None] * holes
            self._cache["_args"] = v
        return v

    @property
    def locals(self):
        return _syms(self._get("locals")) if self.kind == "function" and not self.is_projection else None

    @property
    def params(self):
        return _syms(self._get("params")) if self.kind == "function" and not self.is_projection else None

    @property
    def source_text(self):
        """The text of the function literal as it was written, or None (a projection, an operator, a system function)."""
        if self.kind != "function" or self.is_projection:
            return None
        return self._get("source")

    def callee(self):
        c = self._kfn.callee()
        return None if c is None else self._owner._out(c)

    def is_op(self):
        return self.kind == "operator"

    def is_adverb_chain(self):
        return False

    def __str__(self):
        return str(self._kfn)

    __repr__ = __str__

    def __call__(self, *args, **kwargs):
        if kwargs:
            params = self.params
            names = [str.__str__(p).split('`')[0] for p in params] if params is not None else reserved_fn_args[:self.arity]
            slots = list(args)
            for k, v in kwargs.items():
                if k not in names:
                    raise _named_arity_error(f"the function has no parameter named {k}")
                i = names.index(k)
                if i < len(args):
                    raise _named_arity_error(f"the function got its parameter {k} twice")
                slots.extend([None] * (i + 1 - len(slots)))
                slots[i] = v
            args = tuple(slots)
        return self._owner._call_value(self._kfn, args)

    def __reduce__(self):
        """A function copies and pickles as its source text (and its parameter names), as KlongPy's KGFn does without
        its spans. Only a function literal has a text: a projection or a system function does not."""
        text = self.source_text
        if text is None:
            raise TypeError("cannot pickle a projection, an operator or a system function of the native engine")
        params = self.params
        return (_detached_fn, (text, None if params is None else [str.__str__(p) for p in params]))


class NativeCall(NativeFn, KGCall):
    """A projection (`f(1;)`, `{x+y}:(1;)`): what the Python engine holds as a KGCall."""


class NativeFnWrapper(KGFnWrapper):
    """`klong["f"]`: the function bound to a name, looked up again at each call so that a redefinition takes effect."""

    def __init__(self, klong, fn, sym=None):
        self.klong = klong
        self.fn = fn
        self._sym = sym

    def __reduce__(self):
        return self.fn.__reduce__()

    def __getattr__(self, name):
        if name in ("fn", "klong", "_sym") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.fn, name)


class NativeLambdaWrapper(KGFnWrapper):
    """`klong["f"]` for a Python callable that was assigned to a name: calling it calls the Python function, with the arity the
    Python engine checks (a KGLambda is never called from Python as a bare value: it wants a frame)."""

    def __init__(self, klong, kgl, sym=None):
        self.klong = klong
        self.fn = kgl
        self._sym = sym

    def __getattr__(self, name):
        if name in ("fn", "klong", "_sym") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.fn, name)

    def __call__(self, *args, **kwargs):
        kgl = self.fn
        if kgl._direct and not kwargs:
            bad = kgl.count_error(self._sym, len(args))
            if bad:
                raise _named_arity_error(bad)     # also a RuntimeError, as a wrong count has always been here
        elif not kgl._wildcard and not kwargs and len(args) != kgl.get_arity():
            raise _arity_error(len(args), kgl.get_arity())
        return self.klong.apply(kgl, list(args)) if not kwargs else kgl.call_with_kwargs(self.klong, _Frame(args), kwargs)


class _Frame(dict):
    """The frame a call pushes, as a KGLambda reads its arguments from it: x, y and z (and every argument, when there are more)."""

    def __init__(self, args):
        from .types import reserved_fn_symbols, reserved_all_args_symbol
        super().__init__(zip(reserved_fn_symbols, args))
        if len(args) > len(reserved_fn_symbols):
            self[reserved_all_args_symbol] = list(args)
        self._context = [self]


_PLAIN_CALLABLES = (types.FunctionType, types.MethodType, types.BuiltinFunctionType)


def _callback_owner(preferred, retained):
    if retained is not None:
        return retained
    owner = preferred()
    if owner is None:
        owner = klong_native.active_host()
    if owner is None:
        raise ReferenceError("a Python callback needs a hosted Klong interpreter")
    return owner


class HostLambda:
    """A KGLambda object (a network client, an async call, a database handle) as a callable the native engine can hold.

    Klong calls it with the arguments written; it runs the KGLambda against a frame of them. When such a value comes
    back out of the engine it should be the KGLambda again (`.kgl`): the extension gives a Python callable back as the
    same object (see the report, "engine needs").
    """

    def __init__(self, owner, kgl, bare=None):
        self._preferred_owner = ref(owner)
        # Plain functions need the session only for injection. Call adapters
        # that implement their own protocol always receive the original session.
        self._callback_backend = owner._backend
        uses_owner = type(kgl).__call__ is not KGLambda.__call__ or kgl._provide_klong
        self._retained_owner = owner if uses_owner else None
        self.kgl = kgl
        self.bare = bare    # the Python callable this stands for, when a bare callable was handed in (it goes back out as itself)

    # A function given directly declares its arity to the engine through these (the engine reads them off a KGLambda)
    @property
    def fn(self):
        if not self.kgl._direct:
            raise AttributeError("fn")
        return self

    @property
    def _provide_klong(self):
        if not self.kgl._direct:
            raise AttributeError("_provide_klong")
        return False

    @property
    def _wildcard(self):
        if not self.kgl._direct:
            raise AttributeError("_wildcard")
        return self.kgl._wildcard

    @property
    def args(self):
        if not self.kgl._direct:
            raise AttributeError("args")
        return self.kgl.args

    def __call__(self, *args):
        o = _callback_owner(self._preferred_owner, self._retained_owner)
        bad = self.kgl.count_error(self.kgl.name or "the function", len(args))
        if bad:
            raise _named_arity_error(bad)
        try:
            return o._in(self.kgl(o, _Frame([o._out(a, self._callback_backend) for a in args])))
        except KlongError:
            raise
        except Exception as e:
            err = python_error(e) if self.kgl.user is True else system_error(e)
            if err is e:
                raise
            raise err from e

    def __str__(self):
        return str(self.kgl)


def _error_for(f):
    from .errors import KlongTypeError
    return KlongTypeError(f"cannot apply {type(f).__name__}")


def _compat_base(kind, message, entry=None):
    """The builtin exception the Python engine's error of this kind also is (it let numpy's through: `except IndexError`
    keeps working). The native engine does not carry the original exception, so the base is chosen by kind and message."""
    if kind == "type":
        if message.startswith("!: enumerate needs an integer, not ") and not message.endswith("dictionary"):
            return RuntimeError
        return TypeError
    if kind == "shape":
        return ValueError
    if kind == "range":
        if "nested too deeply" in message:
            return RecursionError
        if "index" in message or "out of range" in message:
            return IndexError
        return None     # a range error of KlongPy's own (a count, a segment size) is not a numpy error
    if kind == "syntax" and message.startswith("end of comment not found"):
        return RuntimeError
    if kind == "system" and "does not exist" in message:
        return FileNotFoundError
    if kind == "syntax" and message.startswith("unknown operator"):
        return KeyError
    if kind == "syntax" and message.startswith("malformed number"):
        return ValueError
    if kind == "syntax" and "dictionary" in message:
        # what the reader raised before KlongPy had a syntax error: a bad entry that is a list is an IndexError, an atom a TypeError
        # (the native span is the literal, so `entry` is its text from the opening `:{`: the first entry decides)
        body = entry[2:].lstrip() if entry else ""
        return IndexError if "must be a [key value] pair" in message and (not body or body[0] in "[;") else TypeError
    return None


_EXPECTED = re.compile(r"expected '(.*)' but found ")


def _with_cause(err, base, message):
    """The Python engine's operator errors carry the exception numpy raised as __cause__ (`raise ... from`): here the
    native engine has no such exception, so the cause is a builtin of the class the error also is, with the message."""
    if base is not None and err.__cause__ is None and issubclass(base, Exception):
        try:
            err.__cause__ = base(f"shape mismatch: {message}" if base is ValueError and "cannot apply" in message else message)
        except Exception:
            pass
    return err


def _error(e, origin=None):
    err = _make_error(e, origin)
    err.__suppress_context__ = True     # as `raise ... from None`: the native error is not shown as the context
    return err


def _make_error(e, origin=None):
    """A native error as the KlongPy exception of its kind, with the span and the source. `origin` is the name of the
    Source the program came from (the file, <expr>): the native engine names the text it was given <string>."""
    kind, message, start, end, name, text, hint = e.args
    cls = compat(ERROR_KINDS.get(kind, KlongInternalError),
                 _compat_base(kind, message, text[start:end].lstrip() if start >= 0 else None))
    span = (start, end) if start >= 0 else None
    if not name:
        name = origin
    source = Source(name or "<string>", text) if span is not None else None
    if kind == "syntax" and span is not None:
        # the parser's two legacy classes, for code that catches them: "expected 'c' but found ..." is UnexpectedChar
        m = _EXPECTED.match(message)
        if m:
            cls = compat(UnexpectedChar, _compat_base(kind, message))
            e = cls.__new__(cls)
            KlongSyntaxError.__init__(e, message, span=span, source=source, hint=hint)
            e.t, e.i, e.expected = text, start, m.group(1)
            e.legacy_text = f"t: {text[start-10:start+10]} pos: {start} char: {m.group(1)}"
            return e
    if kind == "syntax" and span is not None and message.startswith("expected a character after 0c"):
        cls = compat(UnexpectedEOF, None)
        e = cls.__new__(cls)
        KlongSyntaxError.__init__(e, message, span=span, source=source, hint=hint)
        e.t, e.i = text, end
        e.legacy_text = f"t: {text[end-10:]} pos: {end}"
        return e
    err = cls(message, span=span, source=source, hint=hint)
    if kind == "syntax" and message.startswith("malformed number '"):
        err.legacy_text = f"could not convert string to float: {message[len('malformed number '):]}"
    base = _compat_base(kind, message)
    if kind == "syntax" and message.startswith("missing right operand"):
        _with_cause(err, TypeError, message)    # the operator's own complaint, as the Python engine keeps it
    if kind in ("type", "shape", "system") or base is RecursionError:
        _with_cause(err, base, message)
    return err


def _place(e, origin=None):
    """The position of a host function's exception: the engine records on it `_klong_span` = (start, end, source name, source
    text) of the call, once, innermost. A KlongError that has no position of its own gets it."""
    sp = getattr(e, "__dict__", {}).get("_klong_span")
    if sp is not None and isinstance(e, KlongError) and getattr(e, "span", None) is None and sp[0] >= 0 and sp[3]:
        e.span = (sp[0], sp[1])
        e.source = Source(sp[2] if sp[2] and sp[2] != "<string>" else (origin or "<string>"), sp[3])
    return e


def _program(x):
    """The text of a program and the name of its source: a str is <string>, a Source is what it says."""
    if isinstance(x, str):
        return x, None
    return getattr(x, "text", x), getattr(x, "name", None)


class KlongInterpreter:
    """The interpreter on the native engine (a drop-in for the parts that do not depend on Python internals)."""

    is_native = True
    DID_YOU_MEAN_CUTOFF = 0.7     # the engine's own rule of "did you mean": a difflib ratio of at least this

    def __init__(self, backend=None, device=None, strict=0):
        check_strict_level(strict)
        self._backend = get_backend(backend, device=device)
        self._k = klong_native.Klong(strict, _TYPES)
        from .sys_fn_static import ExternTable
        self.externs = ExternTable(self)
        self._sources = []
        self._pyfns = {}
        self._wrapped = set()   # names of Python callables that were wrapped as Klong functions on assignment
        self._host_vars = {}   # a Python object the engine cannot hold (a tuple of classes), kept on the Python side
        self._lambdas = {}      # id of a KGLambda -> (the KGLambda, its HostLambda): a callable is one dictionary key
        self._start_time = time.time()
        self._warn_registry = {}
        k = self._k
        k.set_host_owner(self)
        k.set_console(True)                         # .p .d .w .pf .oc .ic .cc .tc .fc .r .rl .mi: the engine's, on sys.stdout/stdin
        k.set_warning_hook(self._issue_warnings)    # syntax warnings, after the parse and before the first statement
        k.set_stand_in_lookup(self._stand_in)       # the Python function behind an .extern, and .fixed's array maker
        self._bridge_system_functions()
        k.set_converters(self._out, self._in)
        from . import rust_host
        rust_host.install(self, self._backend)
        # a backend that computes in float32 (torch): the engine computes its floats as float32 too (every float atom is the float32
        # nearest to what the operation gave), so that a scalar and an element of an array agree: 0.1+0.2 is 0.3 and equals the
        # element of [0.1]+0.2, and a float that leaves the engine is a float32
        self._float = None
        if self._backend.name != "numpy":
            w = rust_host.host_float(self._backend)
            if w == np.float32:
                k.set_float_width(32)
                self._float = np.float32

    @contextlib.contextmanager
    def wide_floats(self):
        """Float64 for what the engine computes and the arrays it makes, while the block runs: the finite differences of the numeric
        gradient need a step of 1e-6, which float32 cannot hold (3+1e-6 is not a float32 number). Nothing changes on a backend that
        computes in float64. On a device without float64 (MPS) the arrays are made on the CPU for the block."""
        b = self._backend
        if self._float is None:
            yield
            return
        with contextlib.ExitStack() as stack:
            if not b.supports_float64():
                cpu = getattr(b, "compute_on_cpu", None)
                if cpu is None:
                    yield
                    return
                stack.enter_context(cpu())
            was = b._host_float
            b._host_float = np.dtype(np.float64)
            self._k.set_float_width(64)
            try:
                yield
            finally:
                b._host_float = was
                self._k.set_float_width(32)

    def _bridge_system_functions(self):
        """KlongPy's Python system functions: those the native runtime does not have (.py .pyf .pya .pyc .sys .bkd ...) and the
        ones that are Python's by nature even though it has one of its own (the events: asyncio loops, the FPGA card), as callbacks."""
        for create, replaces in _system_function_factories():
            for name, kgl in create().items():
                if name in self._k and not replaces:
                    continue
                if not isinstance(kgl, KGLambda):
                    kgl = KGLambda(kgl)
                self._register_kglambda(name, kgl)
        from .sys_fn_ipc import create_system_var_ipc
        for name, v in create_system_var_ipc().items():
            self._k[name] = v

    def _warning_objects(self, issued, origin):
        out = []
        for code, message, hint, start, end, name, text in issued:
            source = Source(name if name and name != "<string>" else (origin or "<string>"), text)
            out.append(KlongSyntaxWarning(message, span=(start, end), source=source, hint=hint, code=code))
        return out

    def _issue_warnings(self, issued, origin=None):
        """The syntax warnings the native parser recorded for the program about to run, issued through the warnings module as
        the Python engine did: the name of the source as file name, the line of the span as line number, once per location. A
        warning filter set to error raises here and the program does not run."""
        if origin is None:
            origin = self._sources[-1].name if self._sources else None
        regs = self._warn_registry
        if len(regs) > 256:
            regs.clear()
        for w in self._warning_objects(issued, origin):
            warnings.warn_explicit(w, KlongSyntaxWarning, w.source.name, w.source.line_col(w.span[0])[0], module="klongpy",
                                   registry=regs.setdefault(w.source, {}))

    def _stand_in(self, name):
        """The Python callable behind a declared extern (None: there is none), and the maker of fixed-point arrays for .fixed."""
        if name == ".fixed":
            from .sys_fn_static import make_fixed
            return make_fixed
        return self.externs.get(name)

    def names(self):
        """The names of every global that is defined: variables and functions, system functions, the host's variables."""
        return sorted(set(self._k.names()) | set(self._host_vars))

    def current_function(self, sym):
        """The function the global `sym` is bound to now (not a projection), or None: what KGFnWrapper calls, so that a
        redefinition of the name takes effect."""
        try:
            v = self._k[str(sym)]
        except KeyError:
            return None
        if type(v) is _KFn:
            now = self._wrap_fn(v)
            if not isinstance(now, NativeCall):
                return now
        return None

    def name_of(self, fn):
        """The KGSym of the (non-system) global a function value is bound to, or None."""
        kfn = getattr(fn, "_kfn", None)
        if kfn is None:
            return None
        for name in self._k.names():
            if not name.startswith(".") and name not in ("x", "y", "z"):
                try:
                    v = self._k[name]
                except KeyError:
                    continue
                if type(v) is _KFn and v == kfn:
                    return KGSym(name)
        return None

    def process_start_time(self):
        return self._start_time

    def _register_kglambda(self, name, kgl, wrapped=False):
        self._pyfns[name] = kgl
        if wrapped:
            self._wrapped.add(name)
        else:
            self._wrapped.discard(name)
        fn = kgl.fn
        arity = 255 if kgl._wildcard else len(kgl.args)
        preferred = ref(self)
        # the native engine hands a nilad its unit argument: a system function declared without parameters takes none
        n = arity if arity != 255 else None
        user = kgl.user is True
        provide = kgl._provide_klong
        retained = self if provide else None
        backend = self._backend

        def host(*a):
            bad = kgl.count_error(name, len(a))
            if bad:
                raise _named_arity_error(bad)
            if n is not None and not user and len(a) < n:
                raise KlongArityError(f"{name} takes {n} argument{'s' if n != 1 else ''}, got {len(a)}")
            try:
                owner = _callback_owner(preferred, retained)
                args = [owner._out(q, backend) for q in a[:n]]
                return owner._in(fn(owner, *args) if provide else fn(*args))
            except KlongError:
                raise
            except Exception as e:
                # as the Python engine: user Python (.py, .pyf, .pya) fails as a KlongPythonError, a system function's
                # OSError as a KlongSystemError, anything else as it is
                err = python_error(e) if user else system_error(e)
                if err is e:
                    raise
                raise err from e

        self._k.register(name, host, arity, kgl)

    # -- function values and the crossing --------------------------------------------------------------------------
    def _wrap_fn(self, kfn):
        return NativeCall(kfn, self) if kfn.is_projection else NativeFn(kfn, self)

    def _out(self, v, backend=None):
        """A native result as KlongPy has it: function values are NativeFn (also inside lists and object arrays)."""
        backend = self._backend if backend is None else backend
        if backend.name != "numpy":
            v = _result_for(backend, v)
        t = type(v)
        if t is _KFn:
            return self._wrap_fn(v)
        if t is HostLambda:
            return v.kgl if v.bare is None else v.bare
        if t is HostObject:
            return v.obj
        if isinstance(v, KGLambda):
            return v       # a system function registered from a KGLambda, handed back as that object
        if isinstance(v, np.ndarray) and (t is np.ndarray or backend.is_backend_array(v)):
            if v.dtype == object and v.size:
                flat = v.ravel()
                if any(type(q) in (_KFn, np.ndarray, list, HostObject) for q in flat):
                    r = v.copy()
                    rr = r.reshape(-1)
                    for i, q in enumerate(flat):
                        rr[i] = self._out(q, backend)
                    return r
            return v
        if t is list:
            return [self._out(q, backend) for q in v]
        if t is dict:
            return {k: self._out(q, backend) for k, q in v.items()}
        return v

    def _host_lambda(self, kgl):
        """The one HostLambda of a KGLambda: a callable is one dictionary key."""
        e = self._lambdas.get(id(kgl))
        if e is None or e[0] is not kgl:
            e = self._lambdas[id(kgl)] = (kgl, HostLambda(self, kgl))
        return e[1]

    def _result(self, v):
        """A value on its way out to the caller of the interpreter, as what a program evaluates to or a variable holds: a float is the
        backend's float type (float32 on torch). Not for what goes to a Python function the program calls, and not for what apply()
        gives (the gradient code reads those as they are)."""
        f = self._float
        if f is None:
            return v
        t = type(v)
        if t is float:
            return f(v)
        if t is list:
            return [self._result(q) for q in v]
        if t is dict:
            return {k: self._result(q) for k, q in v.items()}
        return v

    def _native_fn(self, f):
        """The native function of a NativeFn; a copied or unpickled one is rebuilt from its text here."""
        if f._detached is None:
            return f._kfn
        text, named = f._detached
        return _build_function(self._k, text, named)

    def _in(self, v):
        """A value on its way into the native engine: NativeFn and its wrapper give back their native function."""
        t = type(v)
        if isinstance(v, NativeFn):
            return self._native_fn(v)
        if t is NativeFnWrapper:
            return self._native_fn(v.fn)
        if t is _WireTableView:
            return v._native_table
        if isinstance(v, KGLambda):
            return self._host_lambda(v)
        if t in _PLAIN_CALLABLES or (t is functools.partial and not hasattr(v, "dtype")):
            # a Python function handed in as a value (a callback, a member of a dictionary): it takes the arguments by position,
            # as a function assigned to a name does
            return HostLambda(self, KGLambda.from_callable(v), bare=v)
        if t not in (dict, KdbTable, KdbKeyedTable) and isinstance(v, dict) and not hasattr(v, "_h"):
            return HostObject(v)
        if t is list:
            return [self._in(q) for q in v]
        if t is tuple:
            return tuple(self._in(q) for q in v)
        if t is dict:
            return {k: self._in(q) for k, q in v.items()}
        if t is np.ndarray and v.dtype == object and v.size and any(isinstance(q, (NativeFn, NativeFnWrapper)) for q in v.ravel()):
            r = np.empty(v.shape, dtype=object)
            rr = r.reshape(-1)
            for i, q in enumerate(v.ravel()):
                rr[i] = self._in(q)
            return r
        return v

    def _call_value(self, kfn, args):
        try:
            return self._out(kfn(*[self._in(a) for a in args]))
        except klong_native.NativeError as e:
            raise _error(e)
        except KlongError as e:
            raise _place(e)

    # -- evaluation ---------------------------------------------------------------------------------------------
    def _evaluate(self, run, x):
        x, origin = _program(x)
        sources = self._sources
        sources.append(Source(origin or "<string>", x))     # current_source(), and what .l looks next to
        try:
            r = self._result(self._out(run(x, origin) if origin else run(x)))
        except klong_native.NativeError as e:
            raise _error(e, origin)
        except KlongError as e:
            raise _place(e, origin)
        finally:
            sources.pop()
        # engine needs: the hook runs for the program at the top level only; the warnings of a program that .E or .l parsed
        # while it ran are left to take, and are issued here once it has ended
        left = self._k.take_warnings()
        if left:
            self._issue_warnings(left, origin)
        return r

    def console(self, write=None, error=None, read=None):
        """Point the console channels at Python callables: what the program writes with .p .d .w .pf goes to write(text) (default:
        sys.stdout.write of that moment), console errors to error(text) (default sys.stderr.write), and .rl .r .mi read from
        read(n), the next input of at most n characters (default: sys.stdin.readline of that moment)."""
        self._k.set_console(True, write, error, read)

    def current_module(self):
        """The name of the module the program is in (`.module(:name)`), or None outside any."""
        return self._k.module()

    def start_module(self, name):
        """`.module(:name)` for the host: the programs run from now on are read in the module `name` until stop_module()."""
        self._k.start_module(str(name))

    def stop_module(self):
        """`.module(0)` for the host: leave the module."""
        self._k.stop_module()

    def current_source(self):
        """The klongpy.source.Source of the program being run (the innermost one), or None."""
        return self._sources[-1] if self._sources else None

    def __call__(self, x):
        return self._evaluate(self._k.eval, x)

    def exec(self, x):
        return self._evaluate(self._k.exec, x)

    def _function_from_text(self, text, named):
        """The function a source text defines (IPC: a function arrives as text), built by the native engine without defining
        anything: `named` is whether the text is the named form `fn(a;b)::{...}` rather than a plain literal (the extension's
        function_form says which, see klongpy.ipc.endpoint.parse_function)."""
        kfn = _build_function(self._k, text, named)
        return self._wrap_fn(kfn) if type(kfn) is _KFn else None

    def apply(self, f, values, arity=None):
        """Call a function value (what `klong['f']` returned, or a Klong function, a symbol naming one, a KGLambda or an
        operator) with a list of already evaluated arguments."""
        if isinstance(f, KGSym):
            try:
                    return self._out(self._k.call_sym(str(f), *[self._in(a) for a in values]))
            except klong_native.NativeError as e:
                raise _error(e)
        if isinstance(f, KGFnWrapper):
            f = f.fn
        if isinstance(f, _Body):
            f = f.fn
        if isinstance(f, NativeFn):
            return self._call_value(f._kfn, values)
        if isinstance(f, KGLambda):
            bad = f.count_error(f.name, len(values))
            if bad:
                raise _named_arity_error(bad)
            args = [self._out(q) for q in values]
            return self._out(f.fn(self, *args) if f._provide_klong else f.fn(*args))
        if isinstance(f, KGOp):
            text = f"{{{f.a}x}}" if len(values) == 1 else "{x" + f.a + "y}"
            return self._call_value(self._k.eval(text), values)
        if type(f) is _KFn:
            return self._call_value(f, values)
        if callable(f):
            return f(*values)
        raise _error_for(f)

    # -- variables ---------------------------------------------------------------------------------------------
    def __getitem__(self, k):
        return self._get_item(k, public=True)

    def _get_item(self, k, *, public=False):
        """Read a name before the public scalar conversion when serving a typed lookup."""
        if str(k) in self._host_vars:
            return self._host_vars[str(k)]
        try:
            v = self._k[str(k)]
        except KeyError:
            raise KeyError(k) from None
        if type(v) is _KFn:
            return NativeFnWrapper(self, self._wrap_fn(v), KGSym(str(k)))
        if isinstance(v, KGLambda):
            if str(k) in self._wrapped:
                return NativeLambdaWrapper(self, v, KGSym(str(k)))
            return v   # the KGLambda that was put there, as KlongPy hands it back
        v = self._out(v)
        return self._result(v) if public else v

    def _apply_ipc(self, fn, target, args):
        if self._float is not None and isinstance(target, NativeFn):
            return self._call_value(self._native_fn(target)._call_ipc, args)
        return self.apply(fn, args)

    def __setitem__(self, k, v):
        if type(v) is KGLambda:
            self._register_kglambda(str(k), v)
            return
        if (callable(v) and not isinstance(v, (KGLambda, NativeFn, NativeFnWrapper, KGFn, HostLambda)) and type(v) is not _KFn
                and not hasattr(v, "dtype")):
            # a Python function or class assigned to a name is a Klong function, as in the Python engine (context_value)
            kgl = KGLambda.from_callable(v)
            kgl.name = str(k)
            self._register_kglambda(str(k), kgl, wrapped=True)
            return
        self._host_vars.pop(str(k), None)
        try:
            self._k[str(k)] = self._in(v)
        except klong_native.NativeError as e:
            raise _error(e)
        except (TypeError, ValueError, AttributeError):
            # a Python object the engine cannot hold (a tuple of classes: numpy.ScalarType, whose .dtype is a descriptor the
            # extension reads .kind of: engine needs, an AttributeError there is "not an array"): kept on the Python side
            self._host_vars[str(k)] = v
            if str(k) in self._k:
                del self._k[str(k)]
            return
        self._pyfns.pop(str(k), None)

    def __delitem__(self, k):
        if str(k) in self._host_vars:
            del self._host_vars[str(k)]
            return
        try:
            del self._k[str(k)]
        except KeyError:
            raise KeyError(k) from None

    def __contains__(self, k):
        return str(k) in self._k or str(k) in self._host_vars

    # -- properties --------------------------------------------------------------------------------------------
    @property
    def strict(self):
        return self._k.strict

    @strict.setter
    def strict(self, n):
        check_strict_level(n)
        self._k.call(".strict", int(n))

    @property
    def backend(self):
        return self._backend

    @property
    def np(self):
        return self._backend.np


def _system_function_factories():
    from .sys_fn import create_system_functions
    from .sys_fn_autograd import create_system_functions_autograd
    from .sys_fn_static import create_system_functions_static
    from .sys_fn_hw import create_system_functions_hw
    from .sys_fn_ipc import create_system_functions_ipc
    from .sys_fn_timer import create_system_functions_timer
    # (factory, replaces the native function of the same name): the events are KlongPy's own (asyncio loops), the
    # native IPC and timers advance only while the native engine is evaluating
    return [(create_system_functions, False), (create_system_functions_autograd, False),
            (create_system_functions_static, False), (create_system_functions_hw, True),
            (create_system_functions_ipc, True), (create_system_functions_timer, True)]
