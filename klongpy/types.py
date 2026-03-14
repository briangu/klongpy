"""
KlongPy type definitions and type checking functions.

This module contains all the core types used by the KlongPy interpreter:
- KGSym: Symbol type
- KGChar: Character type
- KGFn, KGCall: Function types
- KGLambda: Lambda wrapper
- KGOp, KGAdverb: Operator types
- KGChannel: I/O channel type
- Type checking utilities
"""
import inspect
import weakref
from enum import Enum
import numpy

from .backend import np


# python3.11 support
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec


class KlongException(Exception):
    pass


class KGSym(str):
    _intern = {}
    def __new__(cls, s):
        existing = cls._intern.get(s)
        if existing is not None:
            return existing
        obj = str.__new__(cls, s)
        cls._intern[s] = obj
        return obj
    def __repr__(self):
        return f":{super().__str__()}"
    def __eq__(self, o):
        return self is o or (type(o) is KGSym and str.__eq__(self, o))
    __hash__ = str.__hash__


def get_fn_arity_str(arity):
    if arity == 0:
        return ":nilad"
    elif arity == 1:
        return ":monad"
    elif arity == 2:
        return ":dyad"
    return ":triad"


import operator as _op
import math as _math

# Fast scalar dispatch tables — pre-resolved on KGFn at construction time
def _int_lt(a, b): return 1 if a < b else 0
def _int_gt(a, b): return 1 if a > b else 0
def _int_eq(a, b): return 1 if a == b else 0
def _int_not(x): return 1 if x == 0 else 0
def _fast_floor(x): return x if type(x) is int else _math.floor(x)

_FAST_SCALAR_OPS = {'+': _op.add, '*': _op.mul, '-': _op.sub, '^': _op.pow, '<': _int_lt, '>': _int_gt, '=': _int_eq, '|': max, '&': min}
_FAST_SCALAR_MONADS = {'-': _op.neg, '~': _int_not, '_': _fast_floor}


class KGFn:
    __slots__ = ('a', 'args', 'arity', 'global_params', '_is_op', '_is_adverb_chain', '_op_a', '_op_arity', '_fast_op', '_fast_monad')

    def __init__(self, a, args, arity, global_params=None):
        self.a = a
        self.args = args
        self.arity = arity
        self.global_params = global_params
        _is_op = type(a) is KGOp
        self._is_op = _is_op
        self._is_adverb_chain = type(a) is list and len(a) > 0 and type(a[0]) is KGAdverb
        if _is_op:
            _op_a = a.a
            self._op_a = _op_a
            self._op_arity = a.arity
            self._fast_op = _FAST_SCALAR_OPS.get(_op_a)
            self._fast_monad = _FAST_SCALAR_MONADS.get(_op_a)
        else:
            self._op_a = None
            self._op_arity = 0
            self._fast_op = None
            self._fast_monad = None

    def __str__(self):
        return get_fn_arity_str(self.arity)


class KGFnWrapper:
    """
    Wrapper for KGFn that enables calling from Python with dynamic symbol resolution.

    When a KGFn is stored and later invoked, this wrapper automatically re-resolves
    the symbol to use the current function definition. This matches k4 behavior and
    provides REPL-friendly semantics where function redefinitions take effect immediately.

    Example:
        fn = klong['callback']  # Returns KGFnWrapper
        klong('callback::{new implementation}')
        fn(args)  # Uses the NEW implementation
    """
    __slots__ = ('klong', 'fn', '_sym')

    def __init__(self, klong, fn, sym=None):
        self.klong = klong
        self.fn = fn
        # Use provided symbol name (cached) or search for it
        self._sym = sym if sym is not None else self._find_symbol(fn)

    def _find_symbol(self, fn):
        """Find which symbol this function is currently bound to"""
        if type(fn) is not KGFn:
            return None

        # Search the context for this function
        # Skip reserved symbols (x, y, z, .f) which are function parameters, not stored callbacks
        for sym, value in self.klong._context:
            # Skip reserved symbols - use the module constants
            if sym in reserved_fn_symbols_set or sym == reserved_dot_f_symbol:
                continue
            if value is fn:
                return sym
        return None

    def __call__(self, *args, **kwargs):
        # Try to resolve dynamically first if we have a symbol
        if self._sym is not None:
            try:
                current = self.klong._context[self._sym]
                if type(current) is KGFn:
                    # Use the current definition
                    if len(args) != current.arity:
                        raise RuntimeError(f"Klong function called with {len(args)} but expected {current.arity}")
                    fn_args = [np.asarray(x) if isinstance(x, list) else x for x in args]
                    return self.klong.call(KGCall(current.a, [*fn_args], current.arity))
            except KeyError:
                # Symbol was deleted, fall through to original function
                pass

        if len(args) != self.fn.arity:
            raise RuntimeError(f"Klong function called with {len(args)} but expected {self.fn.arity}")
        fn_args = [np.asarray(x) if isinstance(x, list) else x for x in args]
        return self.klong.call(KGCall(self.fn.a, [*fn_args], self.fn.arity))


class KGCall(KGFn):
    __slots__ = ('_cached_body', '_cached_body_arity', '_cached_body_type', '_cached_version', '_cached_cond_is_dyad_op', '_nargs', '_f_args', '_arg0_is_dyad_op')

    def __init__(self, a, args, arity, global_params=None):
        super().__init__(a, args, arity, global_params)
        self._cached_body = None
        self._cached_body_arity = 0
        self._cached_body_type = None
        self._cached_version = -1
        self._cached_cond_is_dyad_op = False
        self._nargs = 0 if args is None else (len(args) if type(args) is list else 1)
        _f_args = args if args is None or type(args) is list else [args]
        self._f_args = _f_args
        # Pre-compute arg dispatch strategy for nargs==1
        if _f_args is not None and len(_f_args) == 1:
            _a0 = _f_args[0]
            _ta0 = type(_a0)
            self._arg0_is_dyad_op = (_ta0 is KGFn or _ta0 is KGCall) and _a0._is_op and _a0._op_arity == 2
        else:
            self._arg0_is_dyad_op = False

    def __str__(self):
        return self.a.__str__() if issubclass(type(self.a), KGLambda) else super().__str__()


class KGOp:
    __slots__ = ('a', 'arity')

    def __init__(self, a, arity):
        self.a = a
        self.arity = arity


class KGAdverb:
    __slots__ = ('a', 'arity')

    def __init__(self, a, arity):
        self.a = a
        self.arity = arity


class KGChar(str):
    pass


class KGCond(list):
    pass


class KGUndefined:
    def __repr__(self):
        return ":undefined"

    def __str__(self):
        return ":undefined"


KLONG_UNDEFINED = KGUndefined()


_inspect_cache = {}

def safe_inspect(fn, follow_wrapped=True):
    code = getattr(fn, '__code__', None)
    if code is not None:
        cached = _inspect_cache.get(id(code))
        if cached is not None:
            return cached
    try:
        result = inspect.signature(fn, follow_wrapped=follow_wrapped).parameters
    except ValueError:
        result = {"args":[]}
    if code is not None:
        _inspect_cache[id(code)] = result
    return result


class KGLambda:
    """
    KGLambda wraps a lambda and make it available to Klong, allowing for direct
    integration of python functions in Klong.

    Introspection is used to infer which parameters should be collected from the
    current context and passed to the lambda. Parameter names must be x,y, or z
    according to the klong convention.  The value for these parameters are
    extracted directly from the currentcontext.

    If a lambda requires access to klong itself, that must be the first parameter.

    Function arity is computed by examining the arguments.

    e.g.

    lambda x,y: x + y
    lambda klong, x: klong(x)

    """
    __slots__ = ('fn', 'args', '_provide_klong', '_wildcard')

    def __init__(self, fn, args=None, provide_klong=False, wildcard=False):
        self.fn = fn
        params = args or safe_inspect(fn)
        self.args = [reserved_fn_symbol_map[x] for x in reserved_fn_args if x in params]
        self._provide_klong = provide_klong or 'klong' in params
        self._wildcard = wildcard

    def _get_pos_args(self, ctx):
        if self._wildcard:
            pos_args = []
            for sym in reserved_fn_symbols:
                try:
                    pos_args.append(ctx[sym])
                except KeyError:
                    break
            return pos_args
        return [ctx[x] for x in self.args]

    def __call__(self, klong, ctx):
        # Fast path for common arity-1 non-klong case
        args = self.args
        if not self._wildcard and not self._provide_klong and len(args) == 1:
            return self.fn(ctx[args[0]])
        pos_args = self._get_pos_args(ctx)
        return self.fn(klong, *pos_args) if self._provide_klong else self.fn(*pos_args)

    def call_with_kwargs(self, klong, ctx, kwargs):
        pos_args = self._get_pos_args(ctx)
        return self.fn(klong, *pos_args, **kwargs) if self._provide_klong else self.fn(*pos_args, **kwargs)

    def get_arity(self):
        return len(self.args)

    def __str__(self):
        return get_fn_arity_str(self.get_arity())


class KGChannelDir(Enum):
    INPUT=1
    OUTPUT=2


class KGChannel:
    class FileHandler:
        def __init__(self, raw, parent):
            self._raw = raw
            self._ref = weakref.ref(parent, self.close)

        def close(self, *args):
            self._raw.close()

    def __init__(self, raw, channel_dir):
        self.channel_dir = channel_dir
        self.raw = raw
        self._fh = KGChannel.FileHandler(raw, self)
        self.at_eof = False

    def __enter__(self):
        return self

    def __exit__(self, ext_type, exc_value, traceback):
        self._fh.close()


class RangeError(Exception):
    def __init__(self, i):
        self.i = i
        super().__init__()


# Reserved function argument names and symbols
reserved_fn_args = ['x','y','z']
reserved_fn_symbols = [KGSym(n) for n in reserved_fn_args]
reserved_fn_symbols_set = frozenset(reserved_fn_symbols)
reserved_fn_symbol_map = {n:KGSym(n) for n in reserved_fn_args}
reserved_dot_f_symbol = KGSym('.f')


# Type checking functions

# Cache which types are integer-like and float-like to avoid repeated issubclass
_integer_types = set()
_float_types = set()

def is_list(x):
    # Check for list or any array-like with ndim > 0 (works for numpy and torch)
    t = type(x)
    if t is list:
        return True
    if t is numpy.ndarray:
        return x.ndim > 0
    return hasattr(x, 'ndim') and x.ndim > 0


def is_iterable(x):
    t = type(x)
    if t is list:
        return True
    if t is str:
        return True
    if t is numpy.ndarray:
        return x.ndim > 0
    if isinstance(x, str) and not isinstance(x, (KGSym, KGChar)):
        return True
    return hasattr(x, 'ndim') and x.ndim > 0


def is_empty(a):
    t = type(a)
    if t is list or t is str:
        return len(a) == 0
    return is_iterable(a) and len(a) == 0


def is_dict(x):
    return isinstance(x, dict)


def to_list(a):
    return a if isinstance(a, list) else a.tolist() if np.isarray(a) else [a]


def is_integer(x, backend):
    tx = type(x)
    if tx is int:
        return True
    if tx in _integer_types:
        return True
    if issubclass(tx, (int, numpy.integer)):
        _integer_types.add(tx)
        return True
    # Handle 0-dim numpy arrays
    if tx is numpy.ndarray and x.ndim == 0:
        return numpy.issubdtype(x.dtype, numpy.integer)
    # Handle backend-specific scalar integers (e.g., torch tensors)
    return backend.is_scalar_integer(x)


def is_float(x, backend):
    tx = type(x)
    if tx is float or tx is int:
        return True
    if tx in _float_types:
        return True
    if issubclass(tx, (float, numpy.floating)):
        _float_types.add(tx)
        return True
    # Handle 0-dim numpy arrays
    if tx is numpy.ndarray and x.ndim == 0:
        return numpy.issubdtype(x.dtype, numpy.floating)
    # Handle backend-specific scalar floats (e.g., torch tensors)
    return backend.is_scalar_float(x)


def is_number(a, backend):
    ta = type(a)
    if ta is int or ta is float:
        return True
    if is_float(a, backend) or is_integer(a, backend):
        return True
    # Handle 0-dim numpy arrays
    if ta is numpy.ndarray and a.ndim == 0:
        return numpy.issubdtype(a.dtype, numpy.number)
    # Handle 0-dim backend tensors as numbers
    if backend.is_backend_array(a) and hasattr(a, 'ndim') and a.ndim == 0:
        return True
    return False


def str_is_float(b):
    try:
        float(b)
        return True
    except ValueError:
        return False


def is_symbolic(c):
    return isinstance(c, str) and (c.isalnum() or c == '.')


def is_char(x):
    # Check for both core and backend KGChar classes
    if isinstance(x, KGChar):
        return True
    # Also check for backend KGChar (in case they're different classes)
    return type(x).__name__ == 'KGChar' and isinstance(x, str)


def is_atom(x):
    """ All objects except for non-empty lists and non-empty strings are atoms. """
    t = type(x)
    if t is list or t is str:
        return len(x) == 0
    if t is numpy.ndarray:
        return x.ndim == 0 or len(x) == 0
    return is_empty(x) if is_iterable(x) else True


def kg_truth(x):
    return x*1


def str_to_chr_arr(s, backend):
    """
    Convert string to character array.

    Parameters
    ----------
    s : str
        The string to convert.
    backend : BackendProvider
        The backend to use.

    Returns
    -------
    array
        Array of KGChar objects.

    Raises
    ------
    UnsupportedDtypeError
        If the backend doesn't support string operations.
    """
    return backend.str_to_char_array(s)


def get_dtype_kind(arr, backend):
    """
    Get the dtype 'kind' character for an array (numpy or torch).

    Returns:
        'O' for object dtype
        'i' for integer types
        'f' for float types
        'u' for unsigned integer
        'b' for boolean
        'c' for complex
    """
    return backend.get_dtype_kind(arr)


# Utility functions

def safe_eq(a, b):
    if a is b:
        return True
    return type(a) is type(b) and a == b


def in_map(x, v):
    try:
        return x in v
    except Exception:
        return False


def has_none(a):
    if a is None or type(a) is not list:
        return False
    for q in a:
        if q is None:
            return True
    return False


def rec_flatten(a):
    if is_list(a) and len(a) > 0:
        return np.concatenate([rec_flatten(x) if is_list(x) else np.array([x]) for x in a]).ravel()
    return a


# Adverb utilities

_ADVERBS = frozenset({
    "'",
    ':\\',
    ":'",
    ':/',
    '/',
    ':~',
    ':*',
    '\\',
    '\\~',
    '\\*'
})

def is_adverb(s):
    return s in _ADVERBS


_ADVERB_ARITIES = {
    ':\\': 2,
    ":'": 2,
    ':/': 2,
    '/': 2,
    ':~': 1,
    ':*': 1,
    '\\': 2,
    '\\~': 1,
    '\\*': 1,
}

def get_adverb_arity(s, ctx):
    if s == "'":
        return ctx
    r = _ADVERB_ARITIES.get(s)
    if r is not None:
        return r
    raise RuntimeError(f"unknown adverb: {s}")


# Function utilities

def merge_projections(arr):
    """
    A projection is a new function that is created by projecting an
    existing function onto at least one of its arguments, resulting
    in the partial application of the original function.
    """
    if len(arr) == 0:
        return arr
    if len(arr) == 1 or not has_none(arr[0]):
        return arr[0]
    sparse_fa = np.copy(arr[0])
    i = 0
    k = 1
    while i < len(sparse_fa) and k < len(arr):
        fa = arr[k]
        j = 0
        while i < len(sparse_fa) and j < len(fa):
            if sparse_fa[i] is None:
                sparse_fa[i] = fa[j]
                j += 1
                while j < len(fa) and fa[j] is None:
                    j += 1
            i += 1
        k += 1
    return sparse_fa


def get_fn_arity(f):
    """
    Examine a function AST and infer arity by looking for x,y and z.
    This arity is needed to populate the KGFn.

    NOTE: TODO: it maybe easier / better to do this at parse time vs late.
    """
    tf = type(f)
    if (tf is KGFn or tf is KGCall) and type(f.a) is KGSym and f.a not in reserved_fn_symbols_set:
       return sum(1 for x in set(f.args) if x in reserved_fn_symbols_set or (x is None))
    def _e(f, level=0):
        tf = type(f)
        if tf is KGFn or tf is KGCall:
            x = _e(f.a, level=1)
            if type(f.args) is list:
                for q in f.args:
                    x.update(_e(q, level=1))
        elif tf is list:
            x = set()
            for q in f:
                x.update(_e(q, level=1))
        elif tf is KGSym:
            x = set([f]) if f in reserved_fn_symbols_set else set()
        else:
            x = set()
        return x if level else len(x)
    return _e(f)
