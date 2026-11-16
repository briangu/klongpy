"""
KlongPy type definitions and type checking functions.

This module contains all the core types used by the KlongPy interpreter:
- KGSym: Symbol type
- KGQSym: Quoted symbol (`:foo`) tree node
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

from .backend import np, KGChar
from .errors import (  # noqa: F401  (re-exported: klongpy.types and klongpy.core are where callers import them from)
    KlongError, KlongException, KlongSyntaxError, KlongNameError, KlongArityError,
    KlongTypeError, KlongShapeError, KlongRangeError, KlongSystemError,
    KlongPythonError, KlongInternalError, KlongMemoryError, KlongInterrupt,
    KlongStaticError, KlongRemoteError, RangeError,
)


# python3.11 support
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec


class KGSym(str):
    def __repr__(self):
        return f":{super().__str__()}"
    def __eq__(self, o):
        return isinstance(o,KGSym) and self.__str__() == o.__str__()
    def __hash__(self):
        return super().__hash__()


class KGQSym(KGSym):
    """
    Tree node for a quoted symbol (`:foo`), as opposed to the bare name `foo`.

    It is equal and hash-equal to the plain KGSym, so dictionary keys and
    comparisons do not care which one they hold. It exists only in parsed
    programs: KlongInterpreter._eval_quoted_symbol decides what it evaluates to,
    and evaluation never yields a KGQSym. List and dictionary literals hold
    plain KGSym.

    Pickling and copying produce a plain KGSym, so a quoted symbol never
    crosses IPC as anything but a symbol.
    """
    def __reduce__(self):
        return (KGSym, (str.__str__(self),))

    def __reduce_ex__(self, protocol):
        return self.__reduce__()


def get_fn_arity_str(arity):
    if arity == 0:
        return ":nilad"
    elif arity == 1:
        return ":monad"
    elif arity == 2:
        return ":dyad"
    return ":triad"


class KGFn:
    # Source positions, recorded by the token parser and by nothing else. They
    # are plain attributes that default to None, so nodes built by hand (or by
    # the legacy parser, or at run time) simply have none. A span is a
    # (start, end) pair of offsets into the text of a klongpy.lexer.Source.
    #
    #   span        the whole expression
    #   op_span     the operator, the callee, or the verb with its adverbs
    #   arg_spans   one span per argument, in order (None for a projection
    #               hole or a missing operand): parallel to args for a call or
    #               a dyad, one element for a monad, the operands for an
    #               adverb chain
    #   body_spans  function literals only: one span per statement of the body
    #   source      the Source the spans refer to (the text the node was read from)
    #
    # missing_operand is set by the parser on an operator node written without
    # its (right) operand ("1+", "-" alone). Such a node is evaluated as
    # before; if that fails, the error reported is the syntax error (see
    # KlongInterpreter._op_error and tickets/design/frontend.md section 6).
    span = None
    op_span = None
    arg_spans = None
    body_spans = None
    source = None
    missing_operand = False

    # Function metadata (ticket 032), set once by the token parser on the node
    # of a function literal ({...}, with or without an argument list after
    # it). None means "not computed": the node is not a function literal, or
    # it was built by hand or by the legacy parser. fn_metadata(node) computes
    # and stores the fields for such a node. Nothing is evaluated to read them.
    #
    #   locals     tuple of KGSym: the names declared by a leading list of
    #              symbols ({[a b];...}). The declaration is not part of the
    #              stored body (see KGBody).
    #   params     named parameters; None for a function of x, y and z
    #              (ticket 061 sets it)
    #   max_arg    the highest of x=1, y=2, z=3 the body mentions, 0 for none.
    #              Nested function bodies count, as they do for arity: a nested
    #              function called with fewer arguments reads the outer ones.
    #   uses_self  the body mentions .f (outside nested function literals,
    #              where .f is the nested function)
    #
    # arity keeps its legacy meaning: the number of distinct names among x, y
    # and z that get_fn_arity finds, which is not always max_arg ({y} has
    # arity 1 and max_arg 2).
    locals = None
    params = None
    max_arg = None
    uses_self = None

    def __init__(self, a, args, arity, global_params=None):
        self.a = a
        self.args = args
        self.arity = arity
        self.global_params = global_params or set()

    def __getstate__(self):
        # Positions refer to a source text that does not travel with the node:
        # a pickled or copied function carries none.
        d = self.__dict__
        if 'span' in d or 'source' in d:
            d = {k: v for k, v in d.items() if k not in _SPAN_ATTRS}
        return d

    def __str__(self):
        return get_fn_arity_str(self.arity)

    def is_op(self):
        return isinstance(self.a, KGOp)

    def is_adverb_chain(self):
        return isinstance(self.a, list) and isinstance(self.a[0], KGAdverb)


_SPAN_ATTRS = frozenset(('span', 'op_span', 'arg_spans', 'body_spans', 'source', 'locals_span'))


class KGBody(list):
    """
    The statements of a function body that declares locals, without the
    declaration: {[a b];a::x;a+b} is stored as KGBody([a::x, a+b]) with
    locals (a, b).

        locals   tuple of KGSym, in the order written (the legacy rule: the
                 first statement is a non-empty list that, ignoring ";", holds
                 only symbols, and at least one statement follows it)
        decl     the declaration as it was parsed (the data list)
        plain    the same statements as a plain list: what is evaluated, and
                 what .f is bound to. A call through .f therefore declares no
                 locals, as before this class existed.

    The body carries its locals so that a caller that holds only the body
    (KGFnWrapper, IPC and timer callbacks pass fn.a to apply) still gets them.
    """
    def __init__(self, stmts, locals, decl=None):
        list.__init__(self, stmts)
        self.locals = tuple(locals)
        self.decl = decl
        self.plain = list(stmts)

    def __reduce__(self):
        return (KGBody, (list(self), self.locals, self.decl))

    def full(self):
        """The body with its declaration as first statement (the form before ticket 032)."""
        return [self.decl if self.decl is not None else list(self.locals)] + list(self)


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
                        raise _arity_error(len(args), current.arity)
                    fn_args = [np.asarray(x) if isinstance(x, list) else x for x in args]
                    return self.klong.apply(current.a, fn_args, current.arity)
            except KeyError:
                # Symbol was deleted, fall through to original function
                pass

        if len(args) != self.fn.arity:
            raise _arity_error(len(args), self.fn.arity)
        fn_args = [np.asarray(x) if isinstance(x, list) else x for x in args]
        return self.klong.apply(self.fn.a, fn_args, self.fn.arity)


def _arity_error(given, expected):
    # also a RuntimeError, which is what this raised before the hierarchy existed
    from .errors import compat
    return compat(KlongArityError, RuntimeError)(f"Klong function called with {given} but expected {expected}")


class KGCall(KGFn):
    def __str__(self):
        return self.a.__str__() if issubclass(type(self.a), KGLambda) else super().__str__()


class KGOp:
    def __init__(self, a, arity):
        self.a = a
        self.arity = arity

    def __eq__(self, other):
        if not isinstance(other, KGOp):
            return NotImplemented
        return self.a == other.a and self.arity == other.arity

    def __hash__(self):
        return hash((KGOp, self.a, self.arity))


class KGAdverb:
    def __init__(self, a, arity):
        self.a = a
        self.arity = arity


class KGCond(list):
    # set by the token parser, see KGFn
    span = None
    arg_spans = None
    source = None


class KGUndefined:
    def __repr__(self):
        return ":undefined"

    def __str__(self):
        return ":undefined"


KLONG_UNDEFINED = KGUndefined()


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
    def __init__(self, fn, args=None, provide_klong=False, wildcard=False, user=False):
        self.fn = fn
        # user: the function is user Python brought in through .py, .pyf or
        # .pya (as opposed to one of KlongPy's own system functions). An
        # exception it raises is reported as a KlongPythonError.
        self.user = user or getattr(fn, '_klong_user', False)
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
    return isinstance(x, KGChar)


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
        '\\*',
        "@'"
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
    elif s == "@'":
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
    Examine a function body and infer arity by looking for x, y and z: the
    number of distinct ones found. The parser calls this once for each
    function literal and stores the result on the node (KGFn.arity); nothing
    calls it at run time.

    The count is the historical one and is kept as it is, because arity
    decides when a call is complete: an operand of a monad is not looked at
    ({-x} has arity 0), and a body that is a single call of a named function
    counts only arguments that are x, y, z or omitted ({g(x+1)} has arity 0).
    KGFn.max_arg is the complete answer.

    Argument values are never hashed (a call whose argument is a list used to
    fail here with "unhashable type").
    """
    if isinstance(f, KGFn) and isinstance(f.a, KGSym) and not in_map(f.a, reserved_fn_symbols) and isinstance(f.args, list):
        # distinct arguments that are x, y, z or a projection hole
        seen = []
        for q in f.args:
            if q is None or (isinstance(q, str) and in_map(q, reserved_fn_symbols)):
                if not any(q is s or (q is not None and s is not None and q == s) for s in seen):
                    seen.append(q)
        return len(seen)
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


_ARG_INDEX = {reserved_fn_symbols[0]: 1, reserved_fn_symbols[1]: 2, reserved_fn_symbols[2]: 3}


def _scan_args(f):
    """
    (highest of x=1 y=2 z=3 mentioned in the expression f, whether .f is
    mentioned). Data (arrays, dictionary literals) is not code and is skipped.
    A nested function literal contributes its max_arg and never its .f.
    """
    if isinstance(f, KGSym):
        i = _ARG_INDEX.get(f)
        if i is not None:
            return i, False
        return 0, f == reserved_dot_f_symbol
    if isinstance(f, KGFn):
        if f.max_arg is not None:
            m, s = f.max_arg, False
        else:
            m, s = _scan_args(f.a)
        args = f.args
        if isinstance(args, list):
            for q in args:
                if q is not None:
                    m2, s2 = _scan_args(q)
                    if m2 > m:
                        m = m2
                    s = s or s2
        elif args is not None and not isinstance(args, dict):
            m2, s2 = _scan_args(args)
            if m2 > m:
                m = m2
            s = s or s2
        return m, s
    if isinstance(f, list):
        m, s = 0, False
        for q in f:
            m2, s2 = _scan_args(q)
            if m2 > m:
                m = m2
            s = s or s2
        return m, s
    if isinstance(f, KGAdverb):
        return _scan_args(f.a)
    return 0, False


def split_locals(body):
    """
    The legacy rule for a locals declaration, applied to a function body:
    returns (locals, declaration, statements after it), or None when the body
    declares none. The body must be a statement list of more than one
    statement whose first is a non-empty list that, leaving out ";", holds
    only symbols ({[a 1];x} declares nothing: [a 1] is data).
    """
    if is_list(body) and len(body) > 1 and is_list(body[0]) and len(body[0]) > 0:
        decl = body[0]
        non_sep = [q for q in decl if q != ';']
        names = [q for q in non_sep if isinstance(q, KGSym)]
        if len(names) > 0 and len(names) == len(non_sep):
            return tuple(names), decl, body[1:]
    return None


def fn_metadata(node):
    """
    Compute and store the function metadata of the node of a function (see
    KGFn): arity is left alone; locals, max_arg and uses_self are set, and a
    locals declaration is moved out of the stored body (node.a becomes a
    KGBody). Returns the node. Does nothing if the node already has them.

    The parser does this for every function literal. A KGFn built by hand has
    it done on its first call.

    locals stays None (decided at each call, as before) in two cases nobody
    writes on purpose: the body is a single data list whose first row is a
    list of symbols, and a body on which the declaration test itself fails.
    """
    if node.locals is not None:
        return node
    body = node.a
    node.max_arg, node.uses_self = _scan_args(body)
    if type(body) is KGBody:
        node.locals = body.locals
    elif isinstance(body, list):
        sp = split_locals(body)
        if sp is None:
            node.locals = ()
        else:
            node.locals = sp[0]
            node.a = body = KGBody(sp[2], sp[0], sp[1])
            node.max_arg, node.uses_self = _scan_args(body)
    elif not is_list(body) or split_locals(body) is None:
        node.locals = ()
    return node
