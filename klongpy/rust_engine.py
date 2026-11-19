"""
The native (Rust) Klong interpreter as KlongPy's evaluation engine: the default engine (klongpy/interpreter.py binds the name
KlongInterpreter to RustInterpreter at import time; KLONGPY_ENGINE=python selects the Python engine instead).

Needs the extension module (native/klong-py), found as klongpy._native, as klong_native on the import path, or in the directory
named by KLONGPY_NATIVE_PATH; without it importing klongpy is an ImportError that says how to build it.

RustInterpreter has the public surface of KlongInterpreter that programs and tests use: __call__, exec, apply, __getitem__,
__setitem__, __delitem__, __contains__, strict, names, backend, np.

What Python supplies (the rest is the engine's): the system functions that are Python by nature (.py .pyf, .web .ws, .srv .cli
IPC and timers on asyncio, tables and SQL, FPGA card I/O, the autograd functions), bridged as host functions, and the few of the
engine's own that stay KlongPy's until the engine does what the tests ask of them (_PYTHON_OWNED); the backend's arrays through the
host table (rust_host.py). The parse tree is not an interpreter's: `kgpy ast`, lint and the static tools are the extension's (klongpy/tools.py).
"""
import sys
import os
import re
import time
import warnings
from types import FunctionType as _FunctionType

import numpy as np

from .backends import get_backend
from .errors import (compat, ERROR_KINDS, KlongError, KlongInternalError, KlongSyntaxError, KlongSyntaxWarning, UnexpectedChar, UnexpectedEOF, python_error,
                     system_error)
from .source import Source
from .utils import check_strict_level
from .types import (KGSym, KGChar, KLONG_UNDEFINED, KGCall, KGFn, KGFnWrapper, KGLambda, KGOp, reserved_fn_args,
                    _arity_error, _count, _named_arity_error)

from .native_ext import load_extension as _load_extension


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
    """A projection's argument that the engine did not report (see RustFn.args)."""

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
    """What unpickling or copying a RustFn gives: the function as a value of a scratch interpreter (its arity, params
    and locals can be read), to be rebuilt from its text in the interpreter it is assigned to."""
    global _scratch_klong
    if _scratch_klong is None:
        _scratch_klong = klong_native.Klong(0, _TYPES)
    named = params is not None
    if named:
        text = "fn(%s)::%s" % (";".join(params), text)
    kfn = _build_function(_scratch_klong, text, named)
    f = (RustCall if kfn.is_projection else RustFn)(kfn, _Detached())
    f._detached = (text, named)
    return f


def _syms(names):
    return None if names is None else tuple(KGSym(n) for n in names)


class RustFn(KGFn):
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


class RustCall(RustFn, KGCall):
    """A projection (`f(1;)`, `{x+y}:(1;)`): what the Python engine holds as a KGCall."""


class RustFnWrapper(KGFnWrapper):
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

    def __call__(self, *args, **kwargs):
        fn = self.fn
        if self._sym is not None:
            try:
                current = self.klong._k[str(self._sym)]
                if type(current) is _KFn:
                    now = self.klong._wrap_fn(current)
                    if not isinstance(now, RustCall):
                        fn = now        # the current definition of the name (a projection is not taken over)
            except KeyError:
                pass
        if type(fn.args) is list and not kwargs:
            # a projection: it takes one argument for each empty slot
            holes = sum(1 for q in fn.args if q is None)
            if holes:
                if len(args) != holes:
                    raise _named_arity_error(f"{self._name()} takes {_count(holes)}, got {len(args)}")
                return self.klong.apply(fn, self._as_values(args))
        params = fn.params
        if kwargs:
            args = self._with_keywords(fn, params, args, kwargs)
        if len(args) != fn.arity:
            if params is not None:
                raise _named_arity_error(f"{self._name()} takes {_count(fn.arity)}, got {len(args)}")
            raise _arity_error(len(args), fn.arity)
        return self.klong._call_value(fn._kfn, self._as_values(args))


class RustLambdaWrapper(KGFnWrapper):
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
        if not kgl._wildcard and not kwargs and len(args) != kgl.get_arity():
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


class HostLambda:
    """A KGLambda object (a network client, an async call, a database handle) as a callable the native engine can hold.

    Klong calls it with the arguments written; it runs the KGLambda against a frame of them. When such a value comes
    back out of the engine it should be the KGLambda again (`.kgl`): the extension gives a Python callable back as the
    same object (see the report, "engine needs").
    """

    def __init__(self, owner, kgl):
        self.owner = owner
        self.kgl = kgl

    def __call__(self, *args):
        o = self.owner
        try:
            return o._in(self.kgl(o, _Frame([o._out(a) for a in args])))
        except KlongError:
            raise
        except Exception as e:
            err = python_error(e) if self.kgl.user is True else system_error(e)
            if err is e:
                raise
            raise err from e

    def __str__(self):
        return str(self.kgl)


# System functions the native runtime has and that stay KlongPy's own, because the engine's version lacks something the tests (the
# specification) ask for; each is in the report as an "engine needs" item:
#   channels and output  .cc .d .fc .ic .oc .p .pf .r .rl .tc .w .mi  write to and read from the channel at klong['.sys.cout'] /
#                        ['.sys.cin'], which a program or test sets to any Python file-like object (KGChannel); the engine's own
#                        write to its console only, and its .mi / .rl read the process's stdin
#   .l                   the search order of KLONGPATH and the directory of the loading program, the error text that lists the places,
#                        a path with a directory and an extension
#   .extern .entry .param  the stand-in is a Python function looked up at each call, and klong.externs.declared / .entries / .params hold
#                        what the program declared (the static tools' API)
#   .fixed               fixed-point arrays (the native one is an error)
_PYTHON_OWNED = frozenset({".cc", ".d", ".entry", ".extern", ".fc", ".fixed", ".ic", ".l", ".mi", ".oc", ".p", ".param", ".pf", ".r", ".rl", ".tc", ".w"})


def sys_load(klong, x):
    """.l as KlongPy's eval_sys_load (the search order of KLONGPATH and the directory of the program that loads, the file
    as the Source of what runs) without its manoeuvre with the Python engine's context stack."""
    from .sys_fn import find_load_file, _program_dir
    from .errors import KlongException, KlongSystemError
    home = _program_dir(klong)
    found, tried = find_load_file(x, home)
    if found is None:
        where = f", then in the directory of the file that calls .l, {home}" if home else ""
        raise FileNotFoundError(f"file does not exist: {x} (looked for {', '.join(tried)}; a relative path is "
                                f"searched in the directories of KLONGPATH, taken from the current directory, "
                                f"{os.getcwd()}{where})")
    try:
        with open(found, "r") as f:
            return klong(Source(found, f.read()))
    except IOError as e:
        if isinstance(e, KlongException):
            raise
        print(e)
        raise compat(KlongSystemError, RuntimeError)(f"file could not be opened: {found}") from e



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


class RustInterpreter:
    """KlongInterpreter on the native engine (a drop-in for the parts that do not depend on Python internals)."""

    is_native = True
    DID_YOU_MEAN_CUTOFF = 0.7     # the engine's own rule of "did you mean": a difflib ratio of at least this

    def __init__(self, backend=None, device=None, parser=None, strict=0):
        check_strict_level(strict)
        if parser not in (None, "token", "legacy"):
            raise ValueError(f"parser must be 'token' or 'legacy', not {parser!r}")
        self._backend = get_backend(backend, device=device)
        self._k = klong_native.Klong(strict, _TYPES)
        self._parser = "native"
        from .sys_fn_static import ExternTable
        self.externs = ExternTable()
        self._sources = []
        self._pyfns = {}
        self._wrapped = set()   # names of Python callables that were wrapped as Klong functions on assignment
        self._host_vars = {}   # a Python object the engine cannot hold (a tuple of classes), kept on the Python side
        self._start_time = time.time()
        self._warn_registry = {}
        self._bridge_system_functions()
        self._k.set_converters(self._out, self._in)
        from . import rust_host
        rust_host.install(self, self._backend)

    def _bridge_system_functions(self):
        """KlongPy's Python system functions that the native runtime does not have (.py .pyf .pya .pyc .sys ...), and
        those it has but that must be KlongPy's own because Python objects are involved (the channels: _PYTHON_OWNED),
        as callbacks."""
        for create, replaces in _system_function_factories():
            for name, kgl in create().items():
                if name in self._k and not replaces and name not in _PYTHON_OWNED:
                    continue
                if name == ".l":
                    kgl = sys_load     # KlongPy's search order and sources, loading through this interpreter
                if not isinstance(kgl, KGLambda):
                    kgl = KGLambda(kgl)
                self._register_kglambda(name, kgl)
        from .sys_fn_ipc import create_system_var_ipc
        for name, v in create_system_var_ipc().items():
            self._k[name] = v
        # the channels are Python objects (KGChannel) wherever they are visible
        from .sys_var import eval_sys_var_cin, eval_sys_var_cout, eval_sys_var_cerr
        for name, ch in (("cin", eval_sys_var_cin()), ("cout", eval_sys_var_cout()), ("cerr", eval_sys_var_cerr())):
            self._k["." + name] = ch
            self._k[".sys." + name] = ch

    def names(self):
        """The names of every global that is defined: variables and functions, system functions, the host's variables."""
        return sorted(set(self._k.names()) | set(self._host_vars))

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
        out, into = self._out, self._in
        # the native engine hands a nilad its unit argument: a system function declared without parameters takes none
        n = arity if arity != 255 else None
        user = kgl.user is True
        provide = kgl._provide_klong

        def host(*a):
            try:
                args = [out(q) for q in a[:n]]
                return into(fn(self, *args) if provide else fn(*args))
            except KlongError:
                raise
            except Exception as e:
                # as the Python engine: user Python (.py, .pyf, .pya) fails as a KlongPythonError, a system function's
                # OSError as a KlongSystemError, anything else as it is
                err = python_error(e) if user else system_error(e)
                if err is e:
                    raise
                raise err from e

        host._kgl = kgl
        self._k.register(name, host, arity)

    # -- function values and the crossing --------------------------------------------------------------------------
    def _wrap_fn(self, kfn):
        return RustCall(kfn, self) if kfn.is_projection else RustFn(kfn, self)

    def _out(self, v):
        """A native result as KlongPy has it: function values are RustFn (also inside lists and object arrays)."""
        if self._backend.name != "numpy":
            v = _result_for(self, v)
        t = type(v)
        if t is _KFn:
            return self._wrap_fn(v)
        if t is HostLambda:
            return v.kgl
        if t is HostObject:
            return v.obj
        if t is _FunctionType:
            return getattr(v, "_kgl", v)   # a system function registered from a KGLambda, handed back
        if t is np.ndarray:
            if v.dtype == object and v.size:
                flat = v.ravel()
                if any(type(q) is _KFn or type(q) in (np.ndarray, list) for q in flat):
                    r = np.empty(v.shape, dtype=object)
                    rr = r.reshape(-1)
                    for i, q in enumerate(flat):
                        rr[i] = self._out(q)
                    return r
            return v
        if t is list:
            return [self._out(q) for q in v]
        if t is dict:
            return {k: self._out(q) for k, q in v.items()}
        return v

    def _native_fn(self, f):
        """The native function of a RustFn; a copied or unpickled one is rebuilt from its text here."""
        if f._detached is None:
            return f._kfn
        text, named = f._detached
        return _build_function(self._k, text, named)

    def _in(self, v):
        """A value on its way into the native engine: RustFn and its wrapper give back their native function."""
        t = type(v)
        if isinstance(v, RustFn):
            return self._native_fn(v)
        if t is RustFnWrapper:
            return self._native_fn(v.fn)
        if isinstance(v, KGLambda):
            return HostLambda(self, v)
        if t is not dict and isinstance(v, dict) and not hasattr(v, "_h"):
            return HostObject(v)
        if t is list:
            return [self._in(q) for q in v]
        if t is tuple:
            return tuple(self._in(q) for q in v)
        if t is dict:
            return {k: self._in(q) for k, q in v.items()}
        if t is np.ndarray and v.dtype == object and v.size and any(isinstance(q, (RustFn, RustFnWrapper)) for q in v.ravel()):
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
    def _warn(self, origin):
        """The syntax warnings the native parser recorded, issued through the warnings module as the Python engine does:
        the name of the source as file name, the line of the span as line number, once per location."""
        issued = self._k.take_warnings()
        if not issued:
            return
        regs = self._warn_registry
        if len(regs) > 256:
            regs.clear()
        for code, message, hint, start, end, name, text in issued:
            source = Source(name if name and name != "<string>" else (origin or "<string>"), text)
            w = KlongSyntaxWarning(message, span=(start, end), source=source, hint=hint, code=code)
            warnings.warn_explicit(w, KlongSyntaxWarning, source.name, source.line_col(start)[0], module="klongpy",
                                   registry=regs.setdefault(source, {}))

    def _evaluate(self, run, x):
        x, origin = _program(x)
        sources = self._sources
        sources.append(Source(origin or "<string>", x))     # current_source(), and what .l looks next to
        try:
            r = self._out(run(x, origin) if origin else run(x))
        except klong_native.NativeError as e:
            err = _error(e, origin)
            if isinstance(err, KlongSyntaxError):
                self._k.take_warnings()   # a program that did not parse issues none
            else:
                self._warn(origin)
            raise err
        except KlongError as e:
            raise _place(e, origin)
        finally:
            sources.pop()
        self._warn(origin)
        return r

    def current_source(self):
        """The klongpy.lexer.Source of the program being run (the innermost one), or None."""
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
        if isinstance(f, RustFn):
            return self._call_value(f._kfn, values)
        if isinstance(f, KGLambda):
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
        if str(k) in self._host_vars:
            return self._host_vars[str(k)]
        try:
            v = self._k[str(k)]
        except KeyError:
            raise KeyError(k) from None
        if type(v) is _KFn:
            return RustFnWrapper(self, self._wrap_fn(v), KGSym(str(k)))
        if type(v) is _FunctionType:
            py = getattr(v, "_kgl", None)
            if isinstance(py, KGLambda):
                if str(k) in self._wrapped:
                    return RustLambdaWrapper(self, py, KGSym(str(k)))
                return py   # the KGLambda that was put there, as KlongPy hands it back
        return self._out(v)

    def __setitem__(self, k, v):
        if type(v) is KGLambda:
            self._register_kglambda(str(k), v)
            return
        if (callable(v) and not isinstance(v, (KGLambda, RustFn, RustFnWrapper, KGFn, HostLambda)) and type(v) is not _KFn
                and not hasattr(v, "dtype")):
            # a Python function or class assigned to a name is a Klong function, as in the Python engine (context_value)
            kgl = KGLambda(v)
            kgl.name = str(k)
            self._register_kglambda(str(k), kgl, wrapped=True)
            return
        self._host_vars.pop(str(k), None)
        try:
            self._k[str(k)] = self._in(v)
        except klong_native.NativeError as e:
            raise _error(e)
        except (TypeError, ValueError):
            # a Python object the engine cannot hold (a tuple of classes: numpy.ScalarType): kept on the Python side
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
