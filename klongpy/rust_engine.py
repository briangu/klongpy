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
import threading
import time
from types import FunctionType as _FunctionType

import numpy as np

from .backends import get_backend
from .errors import compat, ERROR_KINDS, KlongError, KlongInternalError, python_error, system_error
from .lexer import Source
from .types import (KGSym, KGChar, KLONG_UNDEFINED, KGFn, KGFnWrapper, KGLambda, KGOp, reserved_fn_args,
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

_ARITY = {":nilad": 0, ":monad": 1, ":dyad": 2, ":triad": 3}


class _Body:
    """What KGFn.a is for a native function: the function itself, so that apply(fn.a, ...) calls it (KGFnWrapper does)."""

    def __init__(self, fn):
        self.fn = fn

    def __repr__(self):
        return f"<body of {self.fn}>"


class RustFn(KGFn):
    """A Klong function value of the native engine, with the face of KlongPy's KGFn.

    What the extension tells about a function today is its display (`:monad`, `:dyad`, `:fn/5`, which counts the open
    slots of a projection) and that it can be called; the arity comes from the display. The parse-tree fields of KGFn
    (locals, params, max_arg) are not available: they are None.
    """

    def __init__(self, kfn, owner):
        shown = str(kfn)
        arity = _ARITY.get(shown)
        if arity is None:
            try:
                arity = int(shown.rsplit("/", 1)[1])
            except (IndexError, ValueError):
                arity = 1
        super().__init__(_Body(self), None, arity)
        self._kfn = kfn
        self._owner = owner

    def __str__(self):
        return str(self._kfn)

    __repr__ = __str__

    def __call__(self, *args, **kwargs):
        if kwargs:
            names = reserved_fn_args[:self.arity]
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
        raise TypeError("cannot pickle a function of the native engine")


class RustFnWrapper(KGFnWrapper):
    """`klong["f"]`: the function bound to a name, looked up again at each call so that a redefinition takes effect."""

    def __init__(self, klong, fn, sym=None):
        self.klong = klong
        self.fn = fn
        self._sym = sym

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
                    fn = self.klong._wrap_fn(current)
            except KeyError:
                pass
        if kwargs:
            names = reserved_fn_args[:fn.arity]
            slots = list(args) + [_MISSING] * (len(names) - len(args))
            for k, v in kwargs.items():
                if k not in names:
                    raise _named_arity_error(f"{self._name()} has no parameter named {k}")
                i = names.index(k)
                if i < len(args):
                    raise _named_arity_error(f"{self._name()} got its parameter {k} twice")
                slots[i] = v
            if len(args) > len(names) or any(q is _MISSING for q in slots):
                raise _named_arity_error(f"{self._name()} takes {_count(len(names))}, got {len(args) + len(kwargs)}")
            args = tuple(slots)
        if len(args) != fn.arity:
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
_PYTHON_OWNED = frozenset({".ac", ".bkd", ".bkp", ".cc", ".d", ".fc", ".fl", ".ic", ".mi", ".oc", ".p", ".pf", ".r", ".rl",
                           ".rls", ".tc", ".w"})


# variables that hold Python objects the host mutates (the loops, events and registries of .system): kept on the Python side
_HOST_VARS = frozenset({".system"})


def _error_for(f):
    from .errors import KlongTypeError
    return KlongTypeError(f"cannot apply {type(f).__name__}")


def _compat_base(kind, message):
    """The builtin exception the Python engine's error of this kind also is (it let numpy's through: `except IndexError`
    keeps working). The native engine does not carry the original exception, so the base is chosen by kind and message."""
    if kind == "type":
        return TypeError
    if kind == "shape":
        return ValueError
    if kind == "range":
        if "nested too deeply" in message:
            return RecursionError
        if "index" in message or "out of range" in message:
            return IndexError
        return ValueError
    if kind == "syntax" and message.startswith("unknown operator"):
        return KeyError
    return None


def _error(e, origin=None):
    """A native error as the KlongPy exception of its kind, with the span and the source. `origin` is the name of the
    Source the program came from (the file, <expr>): the native engine names the text it was given <string>."""
    kind, message, start, end, name, text, hint = e.args
    cls = compat(ERROR_KINDS.get(kind, KlongInternalError), _compat_base(kind, message))
    span = (start, end) if start >= 0 else None
    if origin is not None and (not name or name == "<string>"):
        name = origin
    source = Source(name or "<string>", text) if span is not None else None
    return cls(message, span=span, source=source, hint=hint)


def _program(x):
    """The text of a program and the name of its source: a str is <string>, a Source is what it says."""
    if isinstance(x, str):
        return x, None
    return getattr(x, "text", x), getattr(x, "name", None)


class RustInterpreter:
    """KlongInterpreter on the native engine (a drop-in for the parts that do not depend on Python internals)."""

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
        self._bridge_system_functions()

    def _bridge_system_functions(self):
        """KlongPy's Python system functions that the native runtime does not have (.py .pyf .pya .pyc .sys ...), and
        those it has but that must be KlongPy's own because Python objects are involved (the channels: _PYTHON_OWNED),
        as callbacks."""
        for create, replaces in _system_function_factories():
            for name, kgl in create().items():
                if name in self._k and not replaces and name not in _PYTHON_OWNED:
                    continue
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
        if name.startswith("__") or name in ("_py_engine", "_k", "_pyfns", "_strict_level", "_host_vars", "_start_time", "_primed", "_wrapped", "_parser_choice"):
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
        return RustFn(kfn, self)

    def _out(self, v):
        """A native result as KlongPy has it: function values are RustFn (also inside lists and object arrays)."""
        t = type(v)
        if t is _KFn:
            return RustFn(v, self)
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

    def _in(self, v):
        """A value on its way into the native engine: RustFn and its wrapper give back their native function."""
        t = type(v)
        if t is RustFn:
            return v._kfn
        if t is RustFnWrapper:
            return v.fn._kfn
        if isinstance(v, KGLambda):
            return HostLambda(self, v)
        if t is list:
            return [self._in(q) for q in v]
        if t is tuple:
            return tuple(self._in(q) for q in v)
        if t is dict:
            return {k: self._in(q) for k, q in v.items()}
        if t is np.ndarray and v.dtype == object and v.size and any(type(q) in (RustFn, RustFnWrapper) for q in v.ravel()):
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
            raise _error(e) from None

    # -- evaluation ---------------------------------------------------------------------------------------------
    def __call__(self, x):
        x, origin = _program(x)
        self._primed = None
        try:
            return self._out(self._k.eval(x))
        except klong_native.NativeError as e:
            raise _error(e, origin) from None

    def exec(self, x):
        x, origin = _program(x)
        self._primed = None
        try:
            return self._out(self._k.exec(x))
        except klong_native.NativeError as e:
            raise _error(e, origin) from None

    def apply(self, f, values, arity=None):
        """Call a function value (what `klong['f']` returned, or a Klong function, a symbol naming one, a KGLambda or an
        operator) with a list of already evaluated arguments."""
        if isinstance(f, KGSym):
            f = self[f]
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
        if str(k) in _HOST_VARS and str(k) in self._host_vars:
            return self._host_vars[str(k)]
        try:
            v = self._k[str(k)]
        except KeyError:
            raise KeyError(k) from None
        if type(v) is _KFn:
            py = self._pyfns.get(str(k))
            if isinstance(py, KGLambda) and str(k) not in self._wrapped:
                return py   # the KGLambda that was put there, as KlongPy hands it back
            return RustFnWrapper(self, RustFn(v, self), KGSym(str(k)))
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
        try:
            self._k[str(k)] = self._in(v)
        except klong_native.NativeError as e:
            raise _error(e) from None
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
        self._k.run(f".strict({int(n)})")

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
            (create_system_functions_static, False), (create_system_functions_hw, False),
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
