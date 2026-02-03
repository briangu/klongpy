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
import sys
import numpy

from .backend import np


# python3.11 support
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec


class KlongException(Exception):
    pass


class KGSym(str):
    def __repr__(self):
        return f":{super().__str__()}"
    def __eq__(self, o):
        return isinstance(o,KGSym) and self.__str__() == o.__str__()
    def __hash__(self):
        return super().__hash__()


def get_fn_arity_str(arity):
    if arity == 0:
        return ":nilad"
    elif arity == 1:
        return ":monad"
    elif arity == 2:
        return ":dyad"
    return ":triad"


class KGFn:
    def __init__(self, a, args, arity):
        self.a = a
        self.args = args
        self.arity = arity

    def __str__(self):
        return get_fn_arity_str(self.arity)

    def is_op(self):
        return isinstance(self.a, KGOp)

    def is_adverb_chain(self):
        return isinstance(self.a, list) and isinstance(self.a[0], KGAdverb)


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

    def __init__(self, klong, fn, sym=None):
        self.klong = klong
        self.fn = fn
        # Use provided symbol name (cached) or search for it
        self._sym = sym if sym is not None else self._find_symbol(fn)

    def _find_symbol(self, fn):
        """Find which symbol this function is currently bound to"""
        if not isinstance(fn, KGFn) or isinstance(fn, KGCall):
            return None

        # Search the context for this function
        # Skip reserved symbols (x, y, z, .f) which are function parameters, not stored callbacks
        for sym, value in self.klong._context:
            # Skip reserved symbols - use the module constants
            if sym in reserved_fn_symbols or sym == reserved_dot_f_symbol:
                continue
            if value is fn:
                return sym
        return None

    def __call__(self, *args, **kwargs):
        # Try to resolve dynamically first if we have a symbol
        if self._sym is not None:
            try:
                current = self.klong._context[self._sym]
                if isinstance(current, KGFn) and not isinstance(current, KGCall):
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
    def __str__(self):
        return self.a.__str__() if issubclass(type(self.a), KGLambda) else super().__str__()


class KGOp:
    def __init__(self, a, arity):
        self.a = a
        self.arity = arity


class KGAdverb:
    def __init__(self, a, arity):
        self.a = a
        self.arity = arity


class KGChar(str):
    pass


class KGCond(list):
    pass


def safe_inspect(fn, follow_wrapped=True):
    try:
        return inspect.signature(fn, follow_wrapped=follow_wrapped).parameters
    except ValueError:
        return {"args":[]}


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
        else:
            pos_args = [ctx[x] for x in self.args]
        return pos_args

    def __call__(self, klong, ctx):
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
reserved_fn_symbol_map = {n:KGSym(n) for n in reserved_fn_args}
reserved_dot_f_symbol = KGSym('.f')


# Type checking functions

def is_list(x):
    # Check for list or any array-like with ndim > 0 (works for numpy and torch)
    return isinstance(x, list) or (hasattr(x, 'ndim') and x.ndim > 0)


def is_iterable(x):
    return is_list(x) or (isinstance(x, str) and not isinstance(x, (KGSym, KGChar)))


def is_empty(a):
    return is_iterable(a) and len(a) == 0


def is_dict(x):
    return isinstance(x, dict)


def to_list(a):
    return a if isinstance(a, list) else a.tolist() if np.isarray(a) else [a]


def is_integer(x, backend):
    if issubclass(type(x), (int, numpy.integer)):
        return True
    # Handle 0-dim numpy arrays
    if isinstance(x, numpy.ndarray) and x.ndim == 0:
        return numpy.issubdtype(x.dtype, numpy.integer)
    # Handle backend-specific scalar integers (e.g., torch tensors)
    return backend.is_scalar_integer(x)


def is_float(x, backend):
    if issubclass(type(x), (float, numpy.floating, int)):
        return True
    # Handle 0-dim numpy arrays
    if isinstance(x, numpy.ndarray) and x.ndim == 0:
        return numpy.issubdtype(x.dtype, numpy.floating)
    # Handle backend-specific scalar floats (e.g., torch tensors)
    return backend.is_scalar_float(x)


def is_number(a, backend):
    if is_float(a, backend) or is_integer(a, backend):
        return True
    # Handle 0-dim numpy arrays
    if isinstance(a, numpy.ndarray) and a.ndim == 0:
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
    return isinstance(c, str) and (c.isalpha() or c.isdigit() or c == '.')


def is_char(x):
    # Check for both core and backend KGChar classes
    if isinstance(x, KGChar):
        return True
    # Also check for backend KGChar (in case they're different classes)
    return type(x).__name__ == 'KGChar' and isinstance(x, str)


def is_atom(x):
    """ All objects except for non-empty lists and non-empty strings are atoms. """
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
    return isinstance(a, type(b)) and a == b


def in_map(x, v):
    try:
        return x in v
    except Exception:
        return False


def has_none(a):
    if isinstance(a, list):
        for q in a:
            if q is None:
                return True
    return False


def rec_flatten(a):
    if is_list(a) and len(a) > 0:
        return np.concatenate([rec_flatten(x) if is_list(x) else np.array([x]) for x in a]).ravel()
    return a


# Adverb utilities

def is_adverb(s):
    return s in {
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
    }


def get_adverb_arity(s, ctx):
    if s == "'":
        return ctx
    elif s == ':\\':
        return 2
    elif s == ':\'':
        return 2
    elif s == ':/':
        return 2
    elif s == '/':
        return 2
    elif s == ':~':
        return 1
    elif s == ':*':
        return 1
    elif s == '\\':
        return 2
    elif s == '\\~':
        return 1
    elif s == '\\*':
        return 1
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
                while j < len(fa) and safe_eq(fa[j], None):
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
    if isinstance(f, KGFn) and isinstance(f.a, KGSym) and not in_map(f.a, reserved_fn_symbols):
       return sum(1 for x in set(f.args) if in_map(x, reserved_fn_symbols) or (x is None))
    def _e(f, level=0):
        if isinstance(f, KGFn):
            x = _e(f.a, level=1)
            if isinstance(f.args, list):
                for q in f.args:
                    x.update(_e(q, level=1))
        elif isinstance(f, list):
            x = set()
            for q in f:
                x.update(_e(q, level=1))
        elif isinstance(f, KGSym):
            x = set([f]) if f in reserved_fn_symbols else set()
        else:
            x = set()
        return x if level else len(x)
    return _e(f)
