"""
SPIKE: the native (Rust) Klong interpreter as KlongPy's evaluation engine.

Switched on with the environment variable KLONGPY_ENGINE=rust (klongpy/interpreter.py replaces the name
KlongInterpreter with RustInterpreter at import time). Needs the extension module `klong_native`
(native/klong-py). RustInterpreter has the public surface of KlongInterpreter that programs and tests use:
__call__, exec, __getitem__, __setitem__, __delitem__, __contains__, strict, backend, np.

What is NOT the same (see the report): everything that KlongPy implements in Python against the interpreter's internals
(.py/.pyf, .sys, .web ... system functions, IPC, timers, autograd, compiler, lint, REPL) is missing or only partly bridged.
"""
import sys
import os

import numpy as np

from .backends import get_backend
from .errors import ERROR_KINDS, KlongError, KlongInternalError
from .lexer import Source
from .types import KGSym, KGChar, KLONG_UNDEFINED

try:
    import klong_native
except ImportError:  # the extension lives outside the package in the spike
    _ext = os.environ.get("KLONGPY_NATIVE_PATH")
    if _ext:
        sys.path.insert(0, _ext)
    import klong_native

_TYPES = {"KGSym": KGSym, "KGChar": KGChar, "undefined": KLONG_UNDEFINED}


def _error(e):
    """A native error as the KlongPy exception of its kind, with the span and the source."""
    kind, message, start, end, name, text, hint = e.args
    cls = ERROR_KINDS.get(kind, KlongInternalError)
    span = (start, end) if start >= 0 else None
    source = Source(name or "<string>", text) if span is not None else None
    return cls(message, span=span, source=source, hint=hint)


class RustInterpreter:
    """KlongInterpreter on the native engine (a drop-in for the parts that do not depend on Python internals)."""

    def __init__(self, backend=None, device=None, parser=None, strict=0):
        self._backend = get_backend(backend, device=device)
        self._k = klong_native.Klong(strict, _TYPES)
        self._parser = "native"
        self.externs = {}
        self._sources = []
        self._pyfns = {}
        self._py_engine = None
        self._strict_level = strict
        self._bridge_system_functions()

    def _bridge_system_functions(self):
        """KlongPy's Python system functions that the native runtime does not have (.py .pyf .pya .pyc .sys ...), as callbacks."""
        from .types import KGLambda
        for create in _system_function_factories():
            for name, kgl in create().items():
                if name in self._k:
                    continue
                if not isinstance(kgl, KGLambda):
                    kgl = KGLambda(kgl)
                self._register_kglambda(name, kgl)

    def __getattr__(self, name):
        # Everything that is about the PARSER (prog, _module, _vd, _context, lint, dump, the compiler caches) is served by a
        # Python-engine interpreter, created on first use. It shares no state with the native one: it only parses.
        if name.startswith("__") or name in ("_py_engine", "_k", "_pyfns", "_strict_level"):
            raise AttributeError(name)
        return getattr(self._python_engine(), name)

    def _python_engine(self):
        if self._py_engine is None:
            from .interpreter import PyKlongInterpreter
            self._py_engine = PyKlongInterpreter(strict=self._strict_level)
        return self._py_engine

    def _register_kglambda(self, name, kgl):
        self._pyfns[name] = kgl
        fn = kgl.fn
        arity = 255 if kgl._wildcard else len(kgl.args)
        if kgl._provide_klong:
            self._k.register(name, lambda *a, fn=fn: fn(self, *a), arity)
        else:
            self._k.register(name, fn, arity)

    # -- evaluation ---------------------------------------------------------------------------------------------
    def __call__(self, x):
        if not isinstance(x, str):
            x = getattr(x, "text", x)
        try:
            return self._k.eval(x)
        except klong_native.NativeError as e:
            raise _error(e) from None

    def exec(self, x):
        if not isinstance(x, str):
            x = getattr(x, "text", x)
        try:
            return self._k.exec(x)
        except klong_native.NativeError as e:
            raise _error(e) from None

    def apply(self, f, values, arity=None):
        """Call a function value (what `klong['f']` returned, or a Klong function) with a list of arguments."""
        return f(*values)

    # -- variables ---------------------------------------------------------------------------------------------
    def __getitem__(self, k):
        try:
            v = self._k[str(k)]
        except KeyError:
            raise KeyError(k) from None
        if str(k) in self._pyfns and type(v).__name__ == "KFn":
            return self._pyfns[str(k)]   # the Python callable that was put there, as KlongPy hands it back
        return v

    def __setitem__(self, k, v):
        from .types import KGLambda
        if isinstance(v, KGLambda):
            self._register_kglambda(str(k), v)
            return
        try:
            self._k[str(k)] = v
        except klong_native.NativeError as e:
            raise _error(e) from None
        if callable(v) and not isinstance(v, KGLambda) and type(v).__name__ != "KFn" and not hasattr(v, "dtype"):
            self._pyfns[str(k)] = v
        else:
            self._pyfns.pop(str(k), None)

    def __delitem__(self, k):
        try:
            del self._k[str(k)]
        except KeyError:
            raise KeyError(k) from None

    def __contains__(self, k):
        return str(k) in self._k

    # -- properties --------------------------------------------------------------------------------------------
    @property
    def strict(self):
        return self._strict_level

    @strict.setter
    def strict(self, n):
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
    return [create_system_functions, create_system_functions_autograd, create_system_functions_static,
            create_system_functions_hw]


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
