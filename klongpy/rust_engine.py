"""
The native (Rust) Klong interpreter as KlongPy's evaluation engine.

Switched on with the environment variable KLONGPY_ENGINE=rust (klongpy/interpreter.py replaces the name
KlongInterpreter with RustInterpreter at import time). Needs the extension module `klong_native`
(native/klong-py). RustInterpreter has the public surface of KlongInterpreter that programs and tests use:
__call__, exec, apply, __getitem__, __setitem__, __delitem__, __contains__, strict, backend, np.

What is NOT the same (see the report): everything that KlongPy implements in Python against the interpreter's internals
(.py/.pyf, .sys, .web ... system functions, IPC, timers, autograd, compiler, lint, REPL) is bridged as host functions
or served by a Python-engine interpreter that only parses.
"""
import sys
import os
import re
import threading
import time
import warnings
from types import FunctionType as _FunctionType

import numpy as np

from .backends import get_backend
from .errors import (compat, ERROR_KINDS, KlongError, KlongInternalError, KlongSyntaxError, KlongSyntaxWarning, UnexpectedChar, UnexpectedEOF, python_error,
                     system_error)
from .lexer import Source
from .types import (KGSym, KGChar, KLONG_UNDEFINED, KGCall, KGFn, KGFnWrapper, KGLambda, KGOp, reserved_fn_args,
                    _MISSING, _arity_error, _count, _named_arity_error)

try:
    import klong_native
except ImportError:  # the extension lives outside the package in the spike
    _ext = os.environ.get("KLONGPY_NATIVE_PATH")
    if _ext:
        sys.path.insert(0, _ext)
    import klong_native

_KFn = klong_native.KFn
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
            v = self._cache["_args"] = [self._owner._out(q) for q in self._kfn.bound_args()]
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


# system functions that exist natively and are replaced by KlongPy's own: they read and write the channels
# (klong['.sys.cin'], klong['.sys.cout']), which are Python file objects (KGChannel), so a program can be given any file-like object
_PYTHON_OWNED = frozenset({".ac", ".bkd", ".bkp", ".cc", ".d", ".fc", ".fixed", ".fl", ".ic", ".l", ".mi", ".oc", ".p", ".pf", ".r", ".rl",
                           ".rls", ".strict", ".tc", ".unsafe", ".w"})


# system functions of the native runtime that KlongPy does not have: a program that uses one gets KlongPy's "undefined function"
_NATIVE_ONLY = (".err",)


def _sys_load(klong, x):
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


# variables that hold Python objects the host mutates (the loops, events and registries of .system): kept on the Python side
_HOST_VARS = frozenset({".system"})


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


_SOURCE_NAMES = {}      # text of a named Source that ran -> its name, for errors in functions it defined (bounded)


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
    if kind == "syntax" and hint is None and '\\' in text:
        try:    # a string that ends in a backslash: the writer meant \" for a quote
            from .lexer import Lexer
            from .parser import tk_escaped_quote_hint
            hint = tk_escaped_quote_hint(Lexer(text))
        except Exception:
            pass
    if not name or name == "<string>":
        # the native engine names every text it was given <string>; the text of the error says which program it was
        name = _SOURCE_NAMES.get(text, origin)
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


def _program(x):
    """The text of a program and the name of its source: a str is <string>, a Source is what it says."""
    if isinstance(x, str):
        return x, None
    return getattr(x, "text", x), getattr(x, "name", None)


class RustInterpreter:
    """KlongInterpreter on the native engine (a drop-in for the parts that do not depend on Python internals)."""

    DID_YOU_MEAN_CUTOFF = 0.7     # the engine's own rule of "did you mean": a difflib ratio of at least this

    def __init__(self, backend=None, device=None, parser=None, strict=0):
        from .interpreter import check_strict_level
        check_strict_level(strict)
        if parser not in (None, "token", "legacy"):
            raise ValueError(f"parser must be 'token' or 'legacy', not {parser!r}")
        self._backend = get_backend(backend, device=device)
        self._k = klong_native.Klong(strict, _TYPES)
        self._parser = "native"
        self._parser_choice = parser   # which parser of the Python engine serves what is about the parse tree (prog, dump)
        self.externs = {}
        self._sources = []
        self._pyfns = {}
        self._wrapped = set()   # names of Python callables that were wrapped as Klong functions on assignment
        self._host_vars = {}   # variables that are Python objects owned by the host (.system), seen live from Python
        self._py_engine = None
        self._strict_level = strict
        self._start_time = time.time()
        self._primed = None
        self._warn_registry = {}
        for name in _NATIVE_ONLY:
            if name in self._k:
                del self._k[name]
        self._native_strict = self._k[".strict"]    # the native function of the level: `.strict` itself is KlongPy's
        self._bridge_system_functions()

    def _bridge_system_functions(self):
        """KlongPy's Python system functions that the native runtime does not have (.py .pyf .pya .pyc .sys ...), and
        those it has but that must be KlongPy's own because Python objects are involved (the channels: _PYTHON_OWNED),
        as callbacks."""
        for create, replaces in _system_function_factories():
            for name, kgl in create().items():
                if name in self._k and not replaces and name not in _PYTHON_OWNED:
                    continue
                if name == ".l":
                    kgl = _sys_load     # KlongPy's search order and sources, loading through this interpreter
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

    def __getattr__(self, name):
        # Everything that is about the PARSER (prog, _module, _vd, _context, lint, dump, the compiler caches) is served by a
        # Python-engine interpreter, created on first use. It shares no state with the native one: it only parses.
        if name.startswith("__") or name in ("_py_engine", "_k", "_pyfns", "_strict_level", "_host_vars", "_start_time", "_primed", "_wrapped", "_parser_choice", "_warn_registry", "_native_strict"):
            raise AttributeError(name)
        return getattr(self._python_engine(), name)

    def process_start_time(self):
        return self._start_time

    def _python_engine(self):
        if self._py_engine is None:
            from .interpreter import PyKlongInterpreter
            self._py_engine = PyKlongInterpreter(strict=self._strict_level, parser=self._parser_choice)
        return self._py_engine

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
        t = type(v)
        if t is _KFn:
            return self._wrap_fn(v)
        if t is HostLambda:
            return v.kgl
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

    def _prime(self):
        """The native stack budget is measured from the point where the last evaluation began (Interp::apply does not set it),
        so a call from another thread than the last evaluation's must begin with an evaluation of its own: an empty program.
        Until the extension sets the base in apply (see the report, "engine needs")."""
        me = threading.get_ident()
        if self._primed != me:
            self._k.run("")
            self._primed = me

    def _call_value(self, kfn, args):
        try:
            self._prime()
            return self._out(kfn(*[self._in(a) for a in args]))
        except klong_native.NativeError as e:
            raise _error(e)

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
        self._primed = None
        sources = self._sources
        sources.append(Source(origin or "<string>", x))     # current_source(), and what .l looks next to
        if origin is not None and origin != "<string>":
            if len(_SOURCE_NAMES) > 512:
                _SOURCE_NAMES.clear()
            _SOURCE_NAMES[x] = origin
        try:
            r = self._out(run(x))
        except klong_native.NativeError as e:
            err = _error(e, origin)
            if isinstance(err, KlongSyntaxError):
                self._k.take_warnings()   # a program that did not parse issues none
            else:
                self._warn(origin)
            raise err
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

    def _function_from_text(self, text, tree):
        """The function a source text defines (IPC: a function arrives as text), built by the native engine without
        defining anything: `tree` is the parse of the text (from the Python parser), which says whether it is one function
        literal, plain or the named form `fn(a;b)::{...}` (klongpy.ipc.endpoint.function_source_text)."""
        from .ipc.endpoint import _named_function
        from .types import KGCall
        named = _named_function(tree) is not None
        if not named and (not isinstance(tree, KGFn) or isinstance(tree, KGCall) or tree.args is not None or tree.is_op()):
            return None
        kfn = _build_function(self._k, text, named)
        return self._wrap_fn(kfn) if type(kfn) is _KFn else None

    def apply(self, f, values, arity=None):
        """Call a function value (what `klong['f']` returned, or a Klong function, a symbol naming one, a KGLambda or an
        operator) with a list of already evaluated arguments."""
        if isinstance(f, KGSym):
            try:
                self._prime()
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
            py = self._pyfns.get(str(k))
            if isinstance(py, KGLambda) and str(k) not in self._wrapped:
                return py   # the KGLambda that was put there, as KlongPy hands it back
            return RustFnWrapper(self, self._wrap_fn(v), KGSym(str(k)))
        return self._out(v)

    def __setitem__(self, k, v):
        if str(k) in _HOST_VARS:
            self._host_vars[str(k)] = v
            return
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
        return self._strict_level

    @strict.setter
    def strict(self, n):
        from .interpreter import check_strict_level
        check_strict_level(n)
        self._strict_level = n
        self._native_strict(int(n))

    @property
    def _context(self):
        return _ContextProxy(self, self._python_engine()._context)

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


class _ContextProxy:
    """The parts of KlongInterpreter._context that tests and tools use; the strict level is the native one."""

    def __init__(self, owner, ctx):
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_ctx", ctx)

    def __getattr__(self, name):
        if name == "_strict_mode":
            return self._owner._strict_level
        return getattr(self._ctx, name)

    def __setattr__(self, name, value):
        if name == "_strict_mode":
            self._owner.strict = value
        else:
            setattr(self._ctx, name, value)

    def __getitem__(self, k):
        return self._owner[k]

    def __setitem__(self, k, v):
        self._owner[k] = v

    def __delitem__(self, k):
        del self._owner[k]

    def __iter__(self):
        return iter(self._ctx)
