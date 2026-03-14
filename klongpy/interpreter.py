import time

import numpy

from .adverbs import get_adverb_fn
from .backends import get_backend
from .core import *
from .dyads import create_dyad_functions
from .monads import create_monad_functions
from .sys_fn import create_system_functions
from .sys_fn_autograd import create_system_functions_autograd
from .sys_fn_ipc import create_system_functions_ipc, create_system_var_ipc
from .sys_fn_timer import create_system_functions_timer
from .sys_var import *
from .utils import ReadonlyDict


_UNEVALUATED_OPS = frozenset(['::','∇'])

# Import fast dispatch tables from types (pre-resolved on KGFn at construction time)
from .types import _FAST_SCALAR_OPS, _FAST_SCALAR_MONADS
import operator as _op
# Safe for numpy arrays (numpy handles div-by-zero via inf/nan)
_FAST_DYAD_OPS = {'+': _op.add, '*': _op.mul, '-': _op.sub, '%': _op.truediv, '^': _op.pow}

# Pre-resolve individual reserved symbols to avoid list indexing in hot path
_sym_x = reserved_fn_symbols[0]
_sym_y = reserved_fn_symbols[1]

# Specialized closure generators for fib-like recursive patterns
_CMP_SYM = {'<': '<', '>': '>', '=': '=='}
_ARITH_SYM = {'+': '+', '-': '-', '*': '*'}

def _make_recursive_closure(cond_op, cond_lit, arg0_op, arg0_lit, arg1_op, arg1_lit, branch_op):
    """Generate a specialized Python function for {:[x OP N; x; self(x OP A) BOP self(x OP B)]}."""
    _cs = _CMP_SYM.get(cond_op)
    _a0s = _ARITH_SYM.get(arg0_op)
    _a1s = _ARITH_SYM.get(arg1_op)
    _bs = _ARITH_SYM.get(branch_op)
    if not (_cs and _a0s and _a1s and _bs):
        return None
    # Generate a pure Python function via exec — uses LOAD_CONST for literals
    # and LOAD_GLOBAL for self-reference, matching native Python performance.
    # Parameters are all numeric literals (int/float), no injection risk.
    _ns = {}
    exec(f"def _f(x):\n if x {_cs} {cond_lit!r}:\n  return x\n return _f(x {_a0s} {arg0_lit!r}) {_bs} _f(x {_a1s} {arg1_lit!r})", _ns)
    return _ns['_f']

# Cache which types are KGLambda subclasses to avoid repeated issubclass calls
_kglambda_types = {KGLambda}
_non_kglambda_types = {int, float, str, list, KGFn, KGCall, KGOp, KGAdverb, KGSym, KGCond}

def _is_kglambda_type(tx):
    if tx in _kglambda_types:
        return True
    if tx in _non_kglambda_types:
        return False
    if issubclass(tx, KGLambda):
        _kglambda_types.add(tx)
        return True
    _non_kglambda_types.add(tx)
    return False

# Cache which types are numpy scalar types to avoid repeated issubclass calls
# Pre-populate with common numpy scalar types so each-adverb loops don't fall through to call()
_numpy_scalar_types = {
    numpy.int64, numpy.int32, numpy.int16, numpy.int8,
    numpy.float64, numpy.float32, numpy.float16,
    numpy.uint64, numpy.uint32, numpy.uint16, numpy.uint8,
    numpy.intp,
}

def _is_numpy_scalar_type(tx):
    if tx in _numpy_scalar_types:
        return True
    if issubclass(tx, (numpy.integer, numpy.floating)):
        _numpy_scalar_types.add(tx)
        return True
    return False


def set_context_var(d, sym, v):
    """
    Sets a context variable, wrapping Python lambda/functions as appropriate.
    """
    assert type(sym) is KGSym
    if callable(v) and type(v) not in _kglambda_types and not _is_kglambda_type(type(v)):
        x = KGLambda(v)
        v = KGCall(x,args=None,arity=x.get_arity())
    d[sym] = v


class KGModule(dict):
    """
    A module class that is used for optimizing when to scan for namespaced keys.
    """
    def __init__(self, name=None):
        self.name = name
        super().__init__()


class KlongContext():
    """

    Maintains current symbol and module context for the interpeter.

    TODO: Support 'it' system variable

        it                                                          [It]

    This variable holds the result of the most recent successful
    computation, so you do not have to re-type or copy-paste the
    previous result. E.g.:

            {(x+2%x)%2}:~2
    1.41421356237309504
            it^2
    1.99999999999999997

    """
    __slots__ = ('_context', '_min_ctx_count', '_strict_mode', '_lookup_cache', '_lookup_version', '_fast_x')

    def __init__(self, system_contexts, strict_mode=1):
        # Use list instead of deque for better cache locality
        # Convention: append = push (end is innermost scope), pop = pop from end
        self._context = [*reversed(system_contexts), {}]
        self._lookup_cache = {}
        self._min_ctx_count = len(system_contexts)
        self._strict_mode = strict_mode
        self._lookup_version = 0
        self._fast_x = None

    def start_module(self, name):
        self.push(KGModule(name))
        self._min_ctx_count = len(self._context)

    def stop_module(self):
        self.push({})

    def current_module(self):
        return self._module

    def __setitem__(self, k, v):
        if k not in reserved_fn_symbols_set:
            # Check if variable exists in any scope (iterate newest to oldest)
            for i in range(len(self._context) - 1, -1, -1):
                d = self._context[i]
                if k in d:
                    d[k] = v
                    # Invalidate lookup cache for this key
                    self._lookup_cache.pop(k, None)
                    self._lookup_version += 1
                    return k

        # Variable doesn't exist - check strict mode
        if self._strict_mode >= 1:
            # Check if we're inside a function (more than just global scope)
            in_function = len(self._context) > self._min_ctx_count + 1

            if in_function:
                # Inside function - disallow creating new variables
                raise KlongException(
                    f"undefined variable: {k}\n"
                    f"  To create a local variable, declare it in the parameter list: {{[{k}]; ...}}\n"
                    f"  To modify an existing global, ensure it exists before calling the function"
                )

        # Create new variable in current scope (end of list = innermost)
        set_context_var(self._context[-1], k, v)
        self._lookup_cache.pop(k, None)
        self._lookup_version += 1
        return k

    def __getitem__(self, k):
        # Fast path for reserved symbols (x, y, z, .f) — always in innermost scope
        if k in reserved_fn_symbols_set or k is reserved_dot_f_symbol:
            v = self._context[-1].get(k)
            if v is not None:
                return v
            # Fallback: specialized path stores x in _fast_x (no dict push)
            if k is _sym_x:
                v = self._fast_x
                if v is not None:
                    return v
        else:
            # Fast path: check lookup cache for non-reserved symbols
            cached = self._lookup_cache.get(k)
            if cached is not None:
                return cached
        # Iterate newest to oldest (end to start of list)
        ctx = self._context
        for i in range(len(ctx) - 1, -1, -1):
            d = ctx[i]
            v = d.get(k)
            if v is not None:
                if k not in reserved_fn_symbols_set and k is not reserved_dot_f_symbol:
                    self._lookup_cache[k] = v
                return v
            if type(d) is KGModule:
                if  '`' in k:
                    p = k.split('`')
                    if KGSym(p[1]) == d.name:
                        k = KGSym(p[0])
                else:
                    tk = k + '`'
                    for dk in d.keys():
                        if dk.startswith(tk):
                            return d[dk]
        raise KeyError(k)

    def __delitem__(self, k):
        ctx = self._context
        for i in range(len(ctx) - 1, -1, -1):
            d = ctx[i]
            if k in d and not isinstance(d, ReadonlyDict):
                del d[k]
                self._lookup_cache.pop(k, None)
                self._lookup_version += 1
                return
        raise KeyError(k)

    def push(self, d):
        self._context.append(d)
        cache = self._lookup_cache
        if cache:
            # KGModule uses wildcard matching — must clear entire cache
            if type(d) is KGModule:
                cache.clear()
                self._lookup_version += 1
            else:
                # Skip reserved symbols (x, y, z, .f) — they're always in innermost scope
                # and not worth caching since they change on every function call
                _invalidated = False
                for k in d:
                    if k not in reserved_fn_symbols_set and k is not reserved_dot_f_symbol:
                        cache.pop(k, None)
                        _invalidated = True
                if _invalidated:
                    self._lookup_version += 1

    def push_fn_ctx(self, d):
        """Fast push for function contexts with only reserved symbols (x/y/z/.f)."""
        self._context.append(d)

    def pop_fn_ctx(self):
        """Fast pop for function contexts with only reserved symbols — skip cache invalidation."""
        if len(self._context) > self._min_ctx_count:
            self._context.pop()

    def pop(self):
        if len(self._context) > self._min_ctx_count:
            r = self._context.pop()
            cache = self._lookup_cache
            if cache:
                if type(r) is KGModule:
                    cache.clear()
                    self._lookup_version += 1
                else:
                    _invalidated = False
                    for k in r:
                        if k not in reserved_fn_symbols_set and k is not reserved_dot_f_symbol:
                            cache.pop(k, None)
                            _invalidated = True
                    if _invalidated:
                        self._lookup_version += 1
            return r
        return None

    def is_defined_sym(self, k):
        if type(k) is KGSym:
            ctx = self._context
            for i in range(len(ctx) - 1, -1, -1):
                d = ctx[i]
                if k in d:
                    return True
                if type(d) is KGModule and '`' not in k:
                    tk = k + '`'
                    for dk in d.keys():
                        if dk.startswith(tk):
                            return True
        return False

    def __iter__(self):
        seen = set()
        ctx = self._context
        for i in range(len(ctx) - 1, -1, -1):
            d = ctx[i]
            for x in d.items():
                if x[0] not in seen:
                    yield x
                    seen.add(x[0])


def add_context_key_values(d, context):
    for k,fn in context.items():
        set_context_var(d, KGSym(k), fn)


_cached_sys_d = None
_cached_sys_var = None
_io_sym_cin = KGSym('.cin')
_io_sym_cout = KGSym('.cout')
_io_sym_cerr = KGSym('.cerr')
_io_sym_sys_cin = KGSym('.sys.cin')
_io_sym_sys_cout = KGSym('.sys.cout')
_io_sym_sys_cerr = KGSym('.sys.cerr')

def create_system_contexts():
    global _cached_sys_d, _cached_sys_var

    cin = eval_sys_var_cin()
    cout = eval_sys_var_cout()
    cerr = eval_sys_var_cerr()

    if _cached_sys_d is not None:
        # Reuse pre-built KGLambda/KGCall objects, just update I/O channels
        sys_d = dict(_cached_sys_d)
        sys_d[_io_sym_cin] = cin
        sys_d[_io_sym_cout] = cout
        sys_d[_io_sym_cerr] = cerr

        sys_var = dict(_cached_sys_var)
        sys_var[_io_sym_sys_cin] = cin
        sys_var[_io_sym_sys_cout] = cout
        sys_var[_io_sym_sys_cerr] = cerr

        return [sys_var, ReadonlyDict(sys_d)]

    sys_d = {}
    add_context_key_values(sys_d, create_system_functions())
    add_context_key_values(sys_d, create_system_functions_autograd())
    add_context_key_values(sys_d, create_system_functions_ipc())
    add_context_key_values(sys_d, create_system_functions_timer())
    set_context_var(sys_d, KGSym('.e'), eval_sys_var_epsilon()) # TODO: support lambda
    set_context_var(sys_d, _io_sym_cin, cin)
    set_context_var(sys_d, _io_sym_cout, cout)
    set_context_var(sys_d, _io_sym_cerr, cerr)

    sys_var = {}
    add_context_key_values(sys_var, create_system_var_ipc())
    set_context_var(sys_var, _io_sym_sys_cin, cin)
    set_context_var(sys_var, _io_sym_sys_cout, cout)
    set_context_var(sys_var, _io_sym_sys_cerr, cerr)

    # Cache for subsequent interpreter creations
    _cached_sys_d = dict(sys_d)
    _cached_sys_var = dict(sys_var)

    return [sys_var, ReadonlyDict(sys_d)]


_KLONG_OP_TO_PY = {'+': '+', '-': '-', '*': '*', '%': '/', '>': '>', '<': '<', '=': '==', '^': '**'}

# Monad ops are handled in _compile_arg_fn (not source) to use correct klong semantics

# Adverb reduction/scan ops that can be compiled to numpy source
_KLONG_REDUCE_TO_PY = {'+': '_np.sum', '*': '_np.prod', '|': '_np.max', '&': '_np.min'}
_KLONG_SCAN_TO_PY = {'+': '_np.cumsum', '*': '_np.cumprod', '|': '_np.maximum.accumulate', '&': '_np.minimum.accumulate'}

# Efficient rank: argsort + inverse permutation (O(n) instead of O(n log n) for second argsort)
def _rank(a):
    idx = numpy.argsort(a)
    rank = numpy.empty(len(idx), dtype=numpy.intp)
    rank[idx] = numpy.arange(len(idx))
    return rank

# BLAS-optimized dot-sum: np.dot for 1D arrays, np.sum(a*b) fallback for higher dims
def _dotsum(a, b):
    return numpy.dot(a, b) if a.ndim == 1 else numpy.sum(a * b)

# Globals dict for eval of compiled source that references numpy
_EVAL_GLOBALS = {'_np': numpy, '_rank': _rank, '_dotsum': _dotsum}

# Axis-based reduce/scan functions for stacked 2D arrays (used by _axis_fn on compiled fns)
_AXIS_REDUCE_KEEPDIMS = {
    '+': lambda a: numpy.sum(a, axis=1, keepdims=True),
    '*': lambda a: numpy.prod(a, axis=1, keepdims=True),
    '|': lambda a: numpy.max(a, axis=1, keepdims=True),
    '&': lambda a: numpy.min(a, axis=1, keepdims=True),
}
_AXIS_SCAN_2D = {
    '+': lambda a: numpy.cumsum(a, axis=1),
    '*': lambda a: numpy.cumprod(a, axis=1),
    '|': lambda a: numpy.maximum.accumulate(a, axis=1),
    '&': lambda a: numpy.minimum.accumulate(a, axis=1),
}

def _expr_to_source(expr, klong, dyadic=False, var_refs=None):
    """Try to convert an expression tree to a Python source string.
    Returns (source_str, is_const) or None if not possible.
    Handles arithmetic ops, monad ops (#), and simple adverb chains (+/, */, etc.).
    If var_refs is a dict, collects variable references (KGSym -> var_name) for
    top-level expression compilation instead of requiring variables to be constants.
    """
    te = type(expr)
    if te is KGSym:
        if expr is _sym_x:
            return ('x', False)
        if dyadic and expr is _sym_y:
            return ('y', False)
        try:
            val = klong._context[expr]
            tv = type(val)
            if tv is int or tv is float:
                if var_refs is not None:
                    # Top-level mode: capture as var_ref so compiled fn always reads current values
                    if expr in var_refs:
                        return (var_refs[expr], False)
                    var_name = f'_v{len(var_refs)}'
                    var_refs[expr] = var_name
                    return (var_name, False)
                return (repr(val), True)
            if var_refs is not None and tv is numpy.ndarray:
                if expr in var_refs:
                    return (var_refs[expr], False)
                var_name = f'_v{len(var_refs)}'
                var_refs[expr] = var_name
                return (var_name, False)
        except KeyError:
            pass
        return None
    if te is int or te is float:
        return (repr(expr), True)
    if (te is KGCall or te is KGFn) and expr._is_op:
        if expr._op_arity == 2:
            py_op = _KLONG_OP_TO_PY.get(expr._op_a)
            if py_op is None:
                # Special handling for @ (index-at) in top-level compilation
                if var_refs is not None and expr._op_a == '@':
                    fa = expr.args
                    if type(fa) is not list:
                        fa = [fa] if fa is not None else fa
                    if fa is not None and len(fa) == 2:
                        s0 = _expr_to_source(fa[0], klong, dyadic=dyadic, var_refs=var_refs)
                        s1 = _expr_to_source(fa[1], klong, dyadic=dyadic, var_refs=var_refs)
                        if s0 is not None and s1 is not None:
                            # Detect x@<x → np.sort(x) optimization
                            if s1[0] == f'_np.argsort({s0[0]})':
                                return (f'_np.sort({s0[0]})', False)
                            return (f'{s0[0]}[{s1[0]}]', False)
                return None
            fa = expr.args
            if type(fa) is not list:
                fa = [fa] if fa is not None else fa
            if fa is None or len(fa) != 2:
                return None
            s0 = _expr_to_source(fa[0], klong, dyadic=dyadic, var_refs=var_refs)
            s1 = _expr_to_source(fa[1], klong, dyadic=dyadic, var_refs=var_refs)
            if s0 is not None and s1 is not None:
                is_const = s0[1] and s1[1]
                if is_const:
                    # Both const: pre-evaluate to avoid runtime computation
                    try:
                        val = eval(f'({s0[0]}{py_op}{s1[0]})')
                        return (repr(val), True)
                    except Exception:
                        pass
                # Pre-fold constant sub-expressions (e.g., (1-0.06) → 0.94)
                if s0[1] and '(' in s0[0]:
                    try: s0 = (repr(eval(s0[0])), True)
                    except Exception: pass
                if s1[1] and '(' in s1[0]:
                    try: s1 = (repr(eval(s1[0])), True)
                    except Exception: pass
                return (f'({s0[0]}{py_op}{s1[0]})', is_const)
        if expr._op_arity == 1:
            # Monad ops: & (where/flatnonzero), # (length), < (argsort), - (negate)
            op_a = expr._op_a
            # Note: #, <, & have type-dependent behavior (e.g., # = length for arrays, ordinal for chars)
            # so they require var_refs (top-level compilation where types are known).
            # Negate (-) is always -x regardless of type, so it's safe without var_refs.
            if op_a == '-' or (var_refs is not None and op_a in ('&', '#', '<')):
                fa = expr.args
                arg = fa[0] if type(fa) is list else fa
                if op_a == '<':
                    # Detect <<x (rank) pattern: use O(n) inverse permutation
                    ta = type(arg)
                    if (ta is KGCall or ta is KGFn) and arg._is_op and arg._op_arity == 1 and arg._op_a == '<':
                        inner_arg = arg.args
                        inner = inner_arg[0] if type(inner_arg) is list else inner_arg
                        s = _expr_to_source(inner, klong, dyadic=dyadic, var_refs=var_refs)
                        if s is not None:
                            return (f'_rank({s[0]})', False)
                elif op_a == '#':
                    # Detect #&(expr) → count_nonzero(expr) optimization
                    ta = type(arg)
                    if (ta is KGCall or ta is KGFn) and arg._is_op and arg._op_arity == 1 and arg._op_a == '&':
                        inner_arg = arg.args
                        inner = inner_arg[0] if type(inner_arg) is list else inner_arg
                        s = _expr_to_source(inner, klong, dyadic=dyadic, var_refs=var_refs)
                        if s is not None:
                            return (f'_np.count_nonzero({s[0]})', False)
                s = _expr_to_source(arg, klong, dyadic=dyadic, var_refs=var_refs)
                if s is not None:
                    if op_a == '&':
                        return (f'_np.flatnonzero({s[0]})', False)
                    elif op_a == '#':
                        return (f'len({s[0]})', False)
                    elif op_a == '<':
                        return (f'_np.argsort({s[0]})', False)
                    else:  # '-' = negate
                        return (f'(-{s[0]})', False)
        # Monad ops (arity 1) handled by _compile_arg_fn for correct semantics
    # Handle simple adverb chains: op/x → _np.sum(x), op\x → _np.cumsum(x), etc.
    if te is KGCall and expr._is_adverb_chain:
        chain = expr.a
        if type(chain) is list and len(chain) == 3:
            verb_adv = chain[0]
            adverb = chain[1]
            arg = chain[2]
            if type(verb_adv) is KGAdverb and type(verb_adv.a) is KGOp and type(adverb) is KGAdverb:
                op_char = verb_adv.a.a
                adv_char = adverb.a
                py_fn = None
                if adv_char == '/':
                    py_fn = _KLONG_REDUCE_TO_PY.get(op_char)
                    # Detect +/(x*y) → BLAS dot product
                    if op_char == '+' and py_fn is not None:
                        ta = type(arg)
                        if (ta is KGCall or ta is KGFn) and arg._is_op and arg._op_arity == 2 and arg._op_a == '*':
                            fa_arg = arg.args
                            if type(fa_arg) is list and len(fa_arg) == 2:
                                s0 = _expr_to_source(fa_arg[0], klong, dyadic=dyadic, var_refs=var_refs)
                                s1 = _expr_to_source(fa_arg[1], klong, dyadic=dyadic, var_refs=var_refs)
                                if s0 is not None and s1 is not None:
                                    return (f'_dotsum({s0[0]},{s1[0]})', False)
                elif adv_char == '\\':
                    py_fn = _KLONG_SCAN_TO_PY.get(op_char)
                elif adv_char == ":'" and var_refs is not None:
                    # Each-pair: op:'var → var[:-1] op var[1:]
                    py_op = _KLONG_OP_TO_PY.get(op_char)
                    if py_op is not None:
                        s = _expr_to_source(arg, klong, dyadic=dyadic, var_refs=var_refs)
                        if s is not None:
                            return (f'({s[0]}[:-1]{py_op}{s[0]}[1:])', False)
                    # Max/min each-pair: |:'var → np.maximum(var[:-1], var[1:])
                    if op_char == '|':
                        s = _expr_to_source(arg, klong, dyadic=dyadic, var_refs=var_refs)
                        if s is not None:
                            return (f'_np.maximum({s[0]}[:-1],{s[0]}[1:])', False)
                    if op_char == '&':
                        s = _expr_to_source(arg, klong, dyadic=dyadic, var_refs=var_refs)
                        if s is not None:
                            return (f'_np.minimum({s[0]}[:-1],{s[0]}[1:])', False)
                if py_fn is not None:
                    s = _expr_to_source(arg, klong, dyadic=dyadic, var_refs=var_refs)
                    if s is not None:
                        return (f'{py_fn}({s[0]})', s[1])
    return None


def _compile_arg_fn(expr, klong, dyadic=False):
    """Try to compile a sub-expression to a fast Python function of x (or x,y if dyadic).

    Returns a callable or None. The returned callable has:
    - _vectorizable: if it can be applied to numpy arrays element-wise
    - _is_const: if the result is independent of x/y
    - _axis_fn: if set, can be called on a stacked 2D array for batch processing
    """
    te = type(expr)
    if te is KGSym:
        if expr is _sym_x:
            fn = (lambda x, y: x) if dyadic else (lambda x: x)
            fn._vectorizable = True
            fn._is_const = False
            fn._axis_fn = None if dyadic else (lambda a: a)
            return fn
        if dyadic and expr is _sym_y:
            fn = lambda x, y: y
            fn._vectorizable = True
            fn._is_const = False
            fn._axis_fn = None
            return fn
        # Try to resolve global variable to a constant value
        try:
            val = klong._context[expr]
            tv = type(val)
            if tv is int or tv is float or tv is numpy.ndarray:
                fn = (lambda x, y, c=val: c) if dyadic else (lambda x, c=val: c)
                fn._vectorizable = tv is not numpy.ndarray
                fn._is_const = True
                fn._axis_fn = None if dyadic else (lambda a, c=val: c)
                return fn
        except KeyError:
            pass
        return None
    if te is int or te is float:
        fn = (lambda x, y, c=expr: c) if dyadic else (lambda x, c=expr: c)
        fn._vectorizable = True
        fn._is_const = True
        fn._axis_fn = None if dyadic else (lambda a, c=expr: c)
        return fn
    if (te is KGCall or te is KGFn) and expr._is_op and expr._op_arity == 2:
        op_a = expr._op_a
        _fast = _FAST_DYAD_OPS.get(op_a)
        _op = _fast if _fast is not None else klong._vd[op_a]
        fa = expr.args
        if type(fa) is not list:
            fa = [fa] if fa is not None else fa
        if fa is None or len(fa) != 2:
            return None
        a0, a1 = fa[0], fa[1]
        # Recursively compile sub-args (allows deeper nesting)
        c0 = _compile_arg_fn(a0, klong, dyadic=dyadic)
        c1 = _compile_arg_fn(a1, klong, dyadic=dyadic)
        if c0 is not None and c1 is not None:
            # Constant folding: if both children are constants, pre-compute
            if c0._is_const and c1._is_const:
                try:
                    _cv0 = c0(0, 0) if dyadic else c0(0)
                    _cv1 = c1(0, 0) if dyadic else c1(0)
                    _const = _op(_cv0, _cv1)
                    fn = (lambda x, y, c=_const: c) if dyadic else (lambda x, c=_const: c)
                    fn._vectorizable = True
                    fn._is_const = True
                    fn._axis_fn = None if dyadic else (lambda a, c=_const: c)
                    return fn
                except Exception:
                    pass
            if dyadic:
                fn = lambda x, y, op=_op, g0=c0, g1=c1: op(g0(x, y), g1(x, y))
            else:
                fn = lambda x, op=_op, g0=c0, g1=c1: op(g0(x), g1(x))
            fn._vectorizable = _fast is not None and c0._vectorizable and c1._vectorizable
            fn._is_const = False
            # Compose axis functions if both children support it and op is a fast Python op
            _af0 = getattr(c0, '_axis_fn', None)
            _af1 = getattr(c1, '_axis_fn', None)
            if not dyadic and _fast is not None and _af0 is not None and _af1 is not None:
                fn._axis_fn = lambda a, op=_fast, g0=_af0, g1=_af1: op(g0(a), g1(a))
            else:
                fn._axis_fn = None
            # Tag constant*x pattern for matmul optimization in reduce
            if not dyadic and op_a == '*':
                if c0._is_const and not c1._is_const:
                    try:
                        _cv = c0(0)
                        if type(_cv) is numpy.ndarray:
                            fn._const_mul_val = _cv
                    except Exception:
                        pass
                elif c1._is_const and not c0._is_const:
                    try:
                        _cv = c1(0)
                        if type(_cv) is numpy.ndarray:
                            fn._const_mul_val = _cv
                    except Exception:
                        pass
            return fn
    # Handle monad ops (arity 1): e.g., #x → count(x)
    if (te is KGCall or te is KGFn) and expr._is_op and expr._op_arity == 1:
        op_a = expr._op_a
        _op = klong._vm.get(op_a)
        if _op is not None:
            fa = expr.args
            _fa = fa if type(fa) is not list else fa[0]
            c = _compile_arg_fn(_fa, klong, dyadic=dyadic)
            if c is not None:
                if dyadic:
                    fn = lambda x, y, op=_op, g=c: op(g(x, y))
                else:
                    fn = lambda x, op=_op, g=c: op(g(x))
                fn._vectorizable = False
                fn._is_const = c._is_const
                # For count monad (#), axis version uses shape[-1]
                _c_axis = getattr(c, '_axis_fn', None)
                if not dyadic and op_a == '#' and _c_axis is not None:
                    fn._axis_fn = lambda a, g=_c_axis: numpy.asarray(g(a)).shape[-1]
                else:
                    fn._axis_fn = None
                return fn
    # Handle simple adverb chains: op/x → reduce, op\x → accumulate
    if te is KGCall and expr._is_adverb_chain:
        chain = expr.a
        if type(chain) is list and len(chain) == 3:
            verb_adv = chain[0]
            adverb = chain[1]
            arg = chain[2]
            if type(verb_adv) is KGAdverb and type(verb_adv.a) is KGOp and type(adverb) is KGAdverb:
                op_char = verb_adv.a.a
                adv_char = adverb.a
                c_arg = _compile_arg_fn(arg, klong, dyadic=dyadic)
                if c_arg is not None:
                    np_fn = None
                    _axis_fn_2d = None
                    if adv_char == '/':
                        np_fn = _KLONG_REDUCE_TO_PY.get(op_char)
                        _axis_fn_2d = _AXIS_REDUCE_KEEPDIMS.get(op_char)
                    elif adv_char == '\\':
                        np_fn = _KLONG_SCAN_TO_PY.get(op_char)
                        _axis_fn_2d = _AXIS_SCAN_2D.get(op_char)
                    if np_fn is not None:
                        # Resolve to actual numpy function
                        _resolved = eval(np_fn, _EVAL_GLOBALS)
                        if dyadic:
                            fn = lambda x, y, rf=_resolved, g=c_arg: rf(g(x, y))
                        else:
                            fn = lambda x, rf=_resolved, g=c_arg: rf(g(x))
                        fn._vectorizable = False
                        fn._is_const = c_arg._is_const
                        # Compose axis function if child supports it
                        _c_axis = getattr(c_arg, '_axis_fn', None)
                        if not dyadic and _axis_fn_2d is not None and _c_axis is not None:
                            # Optimization: +/(constant*x) → matrix-vector multiply via BLAS
                            _cmv = getattr(c_arg, '_const_mul_val', None)
                            if op_char == '+' and _cmv is not None:
                                fn._axis_fn = lambda a, c=_cmv: a @ c
                            else:
                                fn._axis_fn = lambda a, af=_axis_fn_2d, g=_c_axis: af(g(a))
                        else:
                            fn._axis_fn = None
                        return fn
                    # Each-pair: op:'x → x[:-1] op x[1:]
                    if adv_char == ":'" and not dyadic:
                        _fast = _FAST_DYAD_OPS.get(op_char)
                        _c_axis = getattr(c_arg, '_axis_fn', None)
                        if _fast is not None:
                            if c_arg._is_const:
                                return None  # each-pair on constant makes no sense
                            def _ep_fn(x, op=_fast, g=c_arg):
                                v = g(x)
                                return op(v[:-1], v[1:])
                            _ep_fn._vectorizable = False
                            _ep_fn._is_const = False
                            if _c_axis is not None:
                                def _ep_axis(a, op=_fast, g=_c_axis):
                                    v = g(a)
                                    return op(v[:, :-1], v[:, 1:])
                                _ep_fn._axis_fn = _ep_axis
                            else:
                                _ep_fn._axis_fn = None
                            return _ep_fn
                        # Max/min each-pair: |:'x, &:'x
                        if op_char == '|' or op_char == '&':
                            _np_fn = numpy.maximum if op_char == '|' else numpy.minimum
                            def _ep_fn(x, npf=_np_fn, g=c_arg):
                                v = g(x)
                                return npf(v[:-1], v[1:])
                            _ep_fn._vectorizable = False
                            _ep_fn._is_const = False
                            if _c_axis is not None:
                                def _ep_axis(a, npf=_np_fn, g=_c_axis):
                                    v = g(a)
                                    return npf(v[:, :-1], v[:, 1:])
                                _ep_fn._axis_fn = _ep_axis
                            else:
                                _ep_fn._axis_fn = None
                            return _ep_fn
    return None


def chain_adverbs(klong, arr):
    """

        Multiple Adverbs

        Multiple adverbs can be attached to a verb. In this case, the
        first adverb modifies the verb, giving a new verb, and the next
        adverb modifies the new verb. Note that subsequent adverbs must
        be adverbs of monadic verbs, because the first verb-adverb
        combination in a chain of adverbs forms a monad. So ,/' (Join-
        Over-Each) would work, but ,/:' (Join-Over-Each-Pair) would not,
        because :' expects a dyadic verb.

        Examples:

        +/' (Plus-Over-Each) would apply Plus-Over to each member of a
        list:

        +/'[[1 2 3] [4 5 6] [7 8 9]]  -->  [6 15 24]

        ,/:~ (Flatten-Over-Converging) would apply Flatten-Over until a
        fixpoint is reached:

        ,/:~[1 [2 [3 [4] 5] 6] 7]  -->  [1 2 3 4 5 6 7]

        ,/\\~ (Flatten-Over-Scan-Converging) explains why ,/:~ flattens
        any object:

        ,/\\~[1 [2 [3 [4] 5] 6] 7]  -->  [[1 [2 [3 [4] 5] 6] 7]
                                          [1 2 [3 [4] 5] 6 7]
                                           [1 2 3 [4] 5 6 7]
                                            [1 2 3 4 5 6 7]]

    """
    _specialized = False
    _vectorizable = False
    _resolved_op = None
    if arr[0].arity == 1:
        if type(arr[0].a) is KGOp:
            # Direct dispatch for built-in operators — pre-resolved function avoids dict lookup
            _fn = klong._vm[arr[0].a.a]
            f = lambda x,fn=_fn: fn(x)
        else:
            # Try to specialize for simple op bodies (e.g., {x+1}, {x*2})
            # This avoids context push/pop and _eval_fn overhead entirely
            _specialized = False
            verb = arr[0].a
            # Resolve symbol to actual function at chain-creation time
            if type(verb) is KGSym:
                try:
                    _resolved = klong._context[verb]
                except KeyError:
                    _resolved = None
                if _resolved is not None and type(_resolved) is KGFn and not _resolved._is_op and not _resolved._is_adverb_chain and _resolved.args is None:
                    body = _resolved.a
                    tb = type(body)
                else:
                    body = verb
                    tb = type(body)
            else:
                body = verb
                tb = type(body)
            # Unwrap inline lambda KGFn: {x*x} → body is KGFn wrapping the op KGCall
            if tb is KGFn and not body._is_op and not body._is_adverb_chain and body.args is None:
                body = body.a
                tb = type(body)
            if (tb is KGCall or tb is KGFn) and body._is_op:
                op_a = body._op_a
                if body._op_arity == 2:
                    # Use fast Python operators for common arithmetic, fall back to cached_fn
                    _fast_op = _FAST_DYAD_OPS.get(op_a)
                    _op_fn = _fast_op if _fast_op is not None else klong._vd[op_a]
                    fa = body.args
                    if type(fa) is not list:
                        fa = [fa] if fa is not None else fa
                    fa0, fa1 = fa[0], fa[1]
                    t0, t1 = type(fa0), type(fa1)
                    # {x op literal}: e.g., {x+1}, {x*2}
                    if t0 is KGSym and fa0 is _sym_x and (t1 is int or t1 is float):
                        _c = fa1
                        f = lambda x, fn=_op_fn, c=_c: fn(x, c)
                        _specialized = True
                        _vectorizable = _fast_op is not None
                    # {literal op x}: e.g., {1+x}
                    elif t1 is KGSym and fa1 is _sym_x and (t0 is int or t0 is float):
                        _c = fa0
                        f = lambda x, fn=_op_fn, c=_c: fn(c, x)
                        _specialized = True
                        _vectorizable = _fast_op is not None
                    # {x op x}: e.g., {x*x}
                    elif t0 is KGSym and fa0 is _sym_x and t1 is KGSym and fa1 is _sym_x:
                        f = lambda x, fn=_op_fn: fn(x, x)
                        _specialized = True
                        _vectorizable = _fast_op is not None
                    else:
                        # Try flat source compilation first (fastest)
                        _src = _expr_to_source(body, klong, dyadic=False)
                        if _src is not None:
                            src_str, is_const = _src
                            if not is_const:
                                try:
                                    _code = compile(f'lambda x:{src_str}', '<klong>', 'eval')
                                    f = eval(_code, _EVAL_GLOBALS)
                                    _specialized = True
                                    _vectorizable = _fast_op is not None
                                    # Get axis function for batch processing via _compile_arg_fn
                                    _body_cf = _compile_arg_fn(body, klong, dyadic=False)
                                    if _body_cf is not None and getattr(_body_cf, '_axis_fn', None) is not None:
                                        f._axis_fn = _body_cf._axis_fn
                                except Exception:
                                    pass
                        if not _specialized:
                            # Fallback: nested compound lambda chain
                            _cf0 = _compile_arg_fn(fa0, klong)
                            _cf1 = _compile_arg_fn(fa1, klong)
                            if _cf0 is not None and _cf1 is not None:
                                f = lambda x, fn=_op_fn, g0=_cf0, g1=_cf1: fn(g0(x), g1(x))
                                _specialized = True
                                _vectorizable = _fast_op is not None and _cf0._vectorizable and _cf1._vectorizable
                                # Compose axis functions for stacked batch processing
                                _af0 = getattr(_cf0, '_axis_fn', None)
                                _af1 = getattr(_cf1, '_axis_fn', None)
                                if _fast_op is not None and _af0 is not None and _af1 is not None:
                                    f._axis_fn = lambda a, op=_fast_op, g0=_af0, g1=_af1: op(g0(a), g1(a))
                elif body._op_arity == 1:
                    # {monad_op x}: e.g., {!x}, {-x}
                    fa = body.args
                    _fa = fa if type(fa) is not list else fa[0]
                    if type(_fa) is KGSym and _fa is _sym_x:
                        _op_fn = klong._vm[op_a]
                        f = lambda x, fn=_op_fn: fn(x)
                        _specialized = True
            if not _specialized:
                # Try _compile_arg_fn on full body (handles adverb chains, monads, compositions)
                _body_cf = _compile_arg_fn(body, klong, dyadic=False)
                if _body_cf is not None:
                    f = _body_cf
                    _specialized = True
            if not _specialized:
                # General case: use KGCall + _eval_fn (skip eval dispatch)
                _call = KGCall(verb, [None], arity=1)
                _args = _call.args
                def f(x, k=klong, c=_call, a=_args):
                    a[0] = x
                    return k._eval_fn(c)
    else:
        if type(arr[0].a) is KGOp:
            # Direct dispatch for built-in operators — pre-resolved function avoids dict lookup
            _fn = klong._vd[arr[0].a.a]
            f = lambda x,y,fn=_fn: fn(x,y)
            _vectorizable = True
        else:
            # Try to unwrap inline lambda and resolve to direct op
            _dyad_verb = arr[0].a
            _resolved_op = None  # Will be set to KGOp if we can vectorize each-2
            _dtb = type(_dyad_verb)
            # Unwrap KGFn wrapper: {x+y} → body is KGFn wrapping op KGCall
            if _dtb is KGFn and not _dyad_verb._is_op and not _dyad_verb._is_adverb_chain and _dyad_verb.args is None:
                _inner = _dyad_verb.a
                _dit = type(_inner)
                if (_dit is KGCall or _dit is KGFn) and _inner._is_op and _inner._op_arity == 2:
                    _dop_a = _inner._op_a
                    _dfast = _FAST_DYAD_OPS.get(_dop_a)
                    _dop = _dfast if _dfast is not None else klong._vd[_dop_a]
                    _dfa = _inner.args
                    if type(_dfa) is not list:
                        _dfa = [_dfa] if _dfa is not None else _dfa
                    if _dfa is not None and len(_dfa) == 2:
                        _da0, _da1 = _dfa[0], _dfa[1]
                        # {x op y}: direct dyadic
                        if type(_da0) is KGSym and _da0 is _sym_x and type(_da1) is KGSym and _da1 is _sym_y:
                            f = lambda x,y,fn=_dop: fn(x,y)
                            _resolved_op = KGOp(_dop_a, arity=2)
                            _dyad_verb = None  # skip general case
                        # {y op x}: swapped args
                        elif type(_da0) is KGSym and _da0 is _sym_y and type(_da1) is KGSym and _da1 is _sym_x:
                            f = lambda x,y,fn=_dop: fn(y,x)
                            _dyad_verb = None
            if _dyad_verb is not None:
                # Try to compile the full body as a flat Python lambda (fastest)
                _compiled = None
                if _dtb is KGFn and not _dyad_verb._is_op and not _dyad_verb._is_adverb_chain and _dyad_verb.args is None:
                    _src = _expr_to_source(_dyad_verb.a, klong, dyadic=True)
                    if _src is not None:
                        src_str, is_const = _src
                        if is_const:
                            try:
                                _const = eval(src_str)
                                _compiled = lambda x, y, c=_const: c
                            except Exception:
                                pass
                        else:
                            try:
                                _code = compile(f'lambda x,y:{src_str}', '<klong>', 'eval')
                                _compiled = eval(_code, _EVAL_GLOBALS)
                            except Exception:
                                pass
                    if _compiled is None:
                        _compiled = _compile_arg_fn(_dyad_verb.a, klong, dyadic=True)
                if _compiled is not None:
                    f = _compiled
                    _dyad_verb = None
                else:
                    # General case: use KGCall + _eval_fn
                    _call = KGCall(arr[0].a, [None, None], arity=2)
                    _args = _call.args
                    def f(x, y, k=klong, c=_call, a=_args):
                        a[0] = x
                        a[1] = y
                        return k._eval_fn(c)
    # Axis-based reduction dispatch for over-each / scan-each on list of arrays
    _AXIS_REDUCE = {
        '+': lambda a: numpy.sum(a, axis=1),
        '*': lambda a: numpy.prod(a, axis=1),
        '|': lambda a: numpy.max(a, axis=1),
        '&': lambda a: numpy.min(a, axis=1),
        '-': lambda a: numpy.subtract.reduce(a, axis=1),
        '%': lambda a: numpy.divide.reduce(a, axis=1),
    }
    _AXIS_SCAN = {
        '+': lambda a: numpy.cumsum(a, axis=1),
        '*': lambda a: numpy.cumprod(a, axis=1),
        '-': lambda a: numpy.subtract.accumulate(a, axis=1),
        '%': lambda a: numpy.divide.accumulate(a, axis=1),
    }
    for i in range(1,len(arr)-1):
        # For each-adverb (') with a vectorizable arithmetic function,
        # apply directly to the array instead of iterating element by element.
        # This is safe for +, *, - which are element-wise on numpy arrays.
        if arr[i].a == "'" and _vectorizable and arr[i].arity == 1:
            _prev_f = f
            _be = klong._backend
            # Check if previous adverb was over(/) or scan(\) for axis optimization
            _verb_op = arr[0].a.a if type(arr[0].a) is KGOp else None
            _prev_adv = arr[i-1].a if i > 1 else None
            _axis_fn = None
            if _verb_op is not None:
                if _prev_adv == '/':
                    _axis_fn = _AXIS_REDUCE.get(_verb_op)
                elif _prev_adv == '\\':
                    _axis_fn = _AXIS_SCAN.get(_verb_op)
            # Also check for compiled axis function from _compile_arg_fn
            if _axis_fn is None:
                _axis_fn = getattr(f, '_axis_fn', None)
            def f(x, f=_prev_f, be=_be, axis_fn=_axis_fn):
                tx = type(x)
                if tx is numpy.ndarray and x.ndim > 0:
                    return f(x)
                if tx is list:
                    if axis_fn is not None and len(x) > 0 and type(x[0]) is numpy.ndarray:
                        try:
                            stacked = numpy.concatenate(x).reshape(len(x), -1)
                            result = axis_fn(stacked)
                            if type(result) is numpy.ndarray:
                                if result.ndim == 1:
                                    return result
                                # 2D: keepdims scalar-per-row → ravel, else list of arrays
                                if result.shape[1] == 1:
                                    return result.ravel()
                                return list(result)
                            return result
                        except (ValueError, TypeError):
                            pass
                    return be.kg_asarray([f(e) for e in x])
                if isinstance(x, str):
                    return be.kg_asarray([f(e) for e in be.str_to_char_array(x)])
                return f(x)
            _vectorizable = False  # Only vectorize the innermost each
            continue
        # Axis-fn fast path for compiled functions with _axis_fn on list of arrays
        if arr[i].a == "'" and arr[i].arity == 1:
            _af = getattr(f, '_axis_fn', None)
            if _af is not None:
                _prev_f = f
                _be = klong._backend
                def f(x, af=_af, pf=_prev_f, be=_be):
                    if type(x) is list and len(x) > 0 and type(x[0]) is numpy.ndarray:
                        try:
                            stacked = numpy.concatenate(x).reshape(len(x), -1)
                            if stacked.ndim == 2:
                                r = af(stacked)
                                if type(r) is numpy.ndarray:
                                    if r.ndim == 2:
                                        return r.ravel() if r.shape[1] == 1 else list(r)
                                    return r
                                return r
                        except (ValueError, TypeError):
                            pass
                    # Fallback to per-element
                    if hasattr(x, '__iter__') and not isinstance(x, str):
                        return be.kg_asarray([pf(e) for e in x])
                    return pf(x)
                continue
        o = get_adverb_fn(klong, arr[i].a, arity=arr[i].arity)
        if arr[i].arity == 1:
            _scan_op = _resolved_op if _resolved_op is not None else arr[0].a
            f = lambda x,f=f,o=o,_op=_scan_op: o(f,x,op=_op)
        else:
            if arr[i].a == "'":
                _each2_op = _resolved_op if _resolved_op is not None else arr[0].a
                f = lambda x,y,f=f,o=o,_op=_each2_op: o(f,x,y,op=_op)
            else:
                f = lambda x,y,f=f,o=o: o(f,x,y)
    if arr[-2].arity == 1:
        f = lambda a=arr[-1],f=f,k=klong: f(k.eval(a))
    else:
        f = lambda a=arr[-1],f=f,k=klong: f(k.eval(a[0]),k.eval(a[1]))
    return f


class KlongInterpreter():
    __slots__ = ('_backend', '_context', '_vd', '_vm', '_start_time', '_module',
                 '_parse_cache', '_adverb_cache', '_result_cache', '_result_cache_ok',
                 '_compiled_expr_cache')

    def __init__(self, backend=None, device=None):
        """
        Initialize a Klong interpreter.

        Parameters
        ----------
        backend : str, optional
            Backend name ('numpy' or 'torch'). Defaults to 'numpy'.
        device : str, optional
            Device for torch backend ('cpu', 'cuda', 'mps'). Only applies
            when backend='torch'. If None, auto-selects best available device.
        """
        self._backend = get_backend(backend, device=device)
        strict_mode = 0  # 0=unsafe (default for backward compat), 1=strict, 2=pedantic
        self._context = KlongContext(create_system_contexts(), strict_mode=strict_mode)
        self._vd = create_dyad_functions(self)
        self._vm = create_monad_functions(self)
        self._start_time = time.time()
        self._module = None
        self._parse_cache = {}
        self._adverb_cache = {}
        self._result_cache = {}
        self._result_cache_ok = True
        self._compiled_expr_cache = {}

    @property
    def backend(self):
        """Return the backend provider for this interpreter."""
        return self._backend

    @property
    def np(self):
        """Return the numpy-compatible array module for this interpreter."""
        return self._backend.np

    def __setitem__(self, k, v):
        k = k if type(k) is KGSym else KGSym(k)
        self._context[k] = v
        self._result_cache.clear()
        # Only clear adverb cache when assigning functions (not data)
        # since specialized adverb closures capture resolved function bodies
        tv = type(v)
        if tv is KGFn or tv is KGCall:
            self._adverb_cache.clear()
        self._result_cache_ok = False

    def __getitem__(self, k):
        k = k if type(k) is KGSym else KGSym(k)
        r = self._context[k]
        # Pass the symbol name to avoid O(n) context search
        tr = type(r)
        return KGFnWrapper(self, r, sym=k) if tr is KGFn or tr is KGCall else r

    def __delitem__(self, k):
        k = k if type(k) is KGSym else KGSym(k)
        del self._context[k]
        self._result_cache.clear()
        self._adverb_cache.clear()
        self._result_cache_ok = False

    def _get_op_fn(self, s, arity):
        return self._vm[s] if arity == 1 else self._vd[s]

    def _is_monad(self, s):
        return type(s) is KGOp and s.a in self._vm

    def _is_dyad(self, s):
        return type(s) is KGOp and s.a in self._vd

    def start_module(self, name):
        self._context.start_module(name)

    def stop_module(self):
        self._context.stop_module()

    def parse_module(self, name):
        self._module = None if safe_eq(name,0) or is_empty(name) else name

    def current_module(self):
        return self._module

    def process_start_time(self):
        return self._start_time

    def _apply_adverbs(self, t, i, a, aa, arity, dyad=False, dyad_value=None):
        aa_arity = get_adverb_arity(aa, arity)
        if type(a) is KGOp:
            a.arity = aa_arity
        a = KGAdverb(a, aa_arity)
        arr = [a, KGAdverb(aa, arity)]
        ii,aa = peek_adverb(t, i)
        while aa is not None:
            arr.append(KGAdverb(aa, 1))
            i = ii
            ii,aa = peek_adverb(t, i)
        i, aa = self._expr(t, i)
        arr.append([dyad_value,aa] if dyad else aa)
        return i,KGCall(arr,args=None,arity=2 if dyad else 1)

    def _read_fn_args(self, t, i=0):
        """

        # Arguments are delimited by parentheses and separated by
        # semicolons. There are up to three arguments.

        a := '(' ')'
           | '(' e ')'
           | '(' e ';' e ')'
           | '(' e ':' e ';' e ')'

        # Projected argument lists are like argument lists (a), but at
        # least one argument must be omitted.

        P := '(' ';' e ')'
           | '(' e ';' ')'
           | '(' ';' e ';' e ')'
           | '(' e ';' ';' e ')'
           | '(' e ';' e ';' ')'
           | '(' ';' ';' e ')'
           | '(' ';' e ';' ')'
           | '(' e ';' ';' ')'

        """
        if cmatch(t,i,'('):
            i += 1
        elif cmatch2(t,i,':', '('):
            i += 2
        else:
            raise UnexpectedChar(t,i,t[i])
        arr = []
        if cmatch(t, i, ')'): # nilad application
            return i+1, arr

        last_was_separator = False
        while True:
            i = skip(t, i, ignore_newline=True)
            if cmatch(t, i, ';'):
                arr.append(None)
                i += 1
                last_was_separator = True
                continue
            if cmatch(t, i, ')'):
                if last_was_separator and len(arr) > 0:
                    arr.append(None)
                return i + 1, arr

            i, a = self._expr(t, i, ignore_newline=True)
            if a is None:
                break
            arr.append(a)
            last_was_separator = False

            i = skip(t, i, ignore_newline=True)
            if cmatch(t, i, ';'):
                i += 1
                last_was_separator = True
                continue
            if cmatch(t, i, ')'):
                return i + 1, arr
            raise UnexpectedChar(t, i, t[i])

        raise UnexpectedEOF(t, i)

    def _read_index_args(self, t, i=0):
        """
        Parse postfix index syntax like a[0] and a[1 2].

        A single index is returned as a scalar; multiple indices are returned
        as a backend array so the existing @ operator semantics apply.
        """
        i, arr = read_list(t, ']', i=i+1, module=self.current_module())
        if len(arr) == 0:
            return i, self._backend.kg_asarray([])
        if len(arr) == 1:
            q = arr[0]
            return i, self._backend.kg_asarray(q) if type(q) is list else q
        return i, self._backend.kg_asarray(arr)


    def _factor(self, t, i=0, ignore_newline=False):
        """

        # A factor is a lexeme class (C) or a variable (V) applied to
        # arguments (a) or a function (f) or a function applied to
        # arguments or a monadic operator (m) applied to an expression
        # or a parenthesized expression or a conditional expression (c)
        # or a list (L) or a dictionary (D).

        x := C
           | V a
           | f
           | f a
           | m e
           | '(' e ')'
           | c
           | L
           | D

        # A function is a program delimited by braces. Deja vu? A
        # function may be projected onto on some of its arguments,
        # giving a projection. A variable can also be used to form
        # a projection.

        f := '{' p '}'
           | '{' p '}' P
           | V P

       """
        i,a = kg_read_array(t, i, self._backend, ignore_newline=ignore_newline, module=self.current_module())
        if a is None:
            return i,a
        if type(a) is str and a == '{': # read fn
            i,a = self.prog(t, i, ignore_newline=True)
            a = a[0] if len(a) == 1 else a
            i = skip(t, i, ignore_newline=True)
            i = cexpect(t, i, '}')
            arity = get_fn_arity(a)
            if cmatch(t, i, '(') or cmatch2(t,i,':','('):
                i,fa = self._read_fn_args(t,i)
                a = KGFn(a, fa, arity) if has_none(fa) else KGCall(a, fa, arity)
            else:
                a = KGFn(a, args=None, arity=arity)
            ii, aa = peek_adverb(t, i)
            if aa:
                i,a = self._apply_adverbs(t, ii, a, aa, arity=1)
        elif type(a) is KGSym:
            if cmatch(t,i,'(') or cmatch2(t,i,':','('):
                i,fa = self._read_fn_args(t,i)
                a = KGFn(a, fa, arity=len(fa)) if has_none(fa) else KGCall(a, fa, arity=len(fa))
                if a.a == KGSym(".comment"):
                    i = read_sys_comment(t,i,a.args[0])
                    return self._factor(t,i, ignore_newline=ignore_newline)
                elif a.a == KGSym('.module'):
                    self.parse_module(fa[0])
            ii, aa = peek_adverb(t, i)
            if aa:
                i,a = self._apply_adverbs(t, ii, a, aa, arity=1)
        elif self._is_monad(a):
            a.arity = 1
            ii, aa = peek_adverb(t, i)
            if aa:
                i,a = self._apply_adverbs(t, ii, a, aa, arity=1)
            else:
                i, aa = self._expr(t, i, ignore_newline=ignore_newline)
                a = KGFn(a, aa, arity=1)
        elif type(a) is str and a == '(':
            i,a = self._expr(t, i, ignore_newline=ignore_newline)
            i = cexpect(t, i, ')')
        elif type(a) is str and a == ':[':
            return read_cond(self, t, i)
        while cmatch(t, i, '['):
            i, index = self._read_index_args(t, i)
            a = KGFn(KGOp('@', arity=2), [a, index], arity=2)
        return i, a

    def _expr(self, t, i=0, ignore_newline=False):
        """

        # An expression is a factor or a dyadic operation applied to
        # a factor and an expression. I.e. dyads associate to the right.

        e := x
           | x d e

        """
        i, a = self._factor(t, i, ignore_newline=ignore_newline)
        if a is None or (type(a) is str and a == ';'):
            return i,a
        ii, aa = kg_read(t, i, ignore_newline=ignore_newline, module=self.current_module())
        if self._is_dyad(aa):
            aa.arity = 2
        while type(aa) is KGOp or type(aa) is KGSym or aa == '{':
            i = ii
            if aa == '{': # read fn
                i,aa = self.prog(t, i, ignore_newline=True)
                aa = aa[0] if len(aa) == 1 else aa
                i = skip(t, i, ignore_newline=True)
                i = cexpect(t, i, '}')
                arity = get_fn_arity(aa)
                if cmatch(t, i, '(') or cmatch2(t,i,':','('):
                    i,fa = self._read_fn_args(t,i)
                    aa = KGFn(aa, fa, arity=arity) if has_none(fa) else KGCall(aa, fa, arity=arity)
                else:
                    aa = KGFn(aa, args=None, arity=arity)
            elif type(aa) is KGSym and (cmatch(t, i, '(') or cmatch2(t,i,':','(')):
                i,fa = self._read_fn_args(t,i)
                aa = KGFn(aa, fa, arity=len(fa)) if has_none(fa) else KGCall(aa, fa, arity=len(fa))
            ii, aaa = peek_adverb(t, i)
            if aaa:
                i,a = self._apply_adverbs(t, ii, aa, aaa, arity=2, dyad=True, dyad_value=a)
            else:
                i, aaa = self._expr(t, i, ignore_newline=ignore_newline)
                a = KGFn(aa, [a, aaa], arity=2)
            ii, aa = kg_read(t, i, ignore_newline=ignore_newline, module=self.current_module())
        if ignore_newline and type(a) is str and a == '\n':
            i = skip(t, i, ignore_newline=True)
        return i, a

    def prog(self, t, i=0, ignore_newline=False):
        """

        Parse a Klong expression string into a Klong program, which can then be evaluated.

        # A program is a ';'-separated sequence of expressions.

        p := e
           | e ';' p

        """
        arr = []
        while i < len(t):
            i, q = self._expr(t,i, ignore_newline=ignore_newline)
            if q is None or (type(q) is str and q == ';'):
                continue
            arr.append(q)
            ii, c = kg_read(t, i, ignore_newline=ignore_newline, module=self.current_module())
            if c != ';':
                break
            i = ii
        return i, arr

    def _resolve_fn(self, f, f_args, f_arity):
        """

        Resolve a Klong function to its final form and update its arguments and arity.

        This helper function is used to resolve function references, projections, and invocations
        while considering the complexities involved, such as:
        * Functions may contain references to symbols which need to be resolved.
        * Functions may be projections or an invocation of a projection, and arguments
        need to be "flattened".

        Args:
            f (KGFn or KGSym): The Klong function or symbol to resolve.
            f_args (list): The list of arguments currently being associated with the function for its evaluation.
            f_arity (int): The arity of the function.

        Returns:
            tuple: A tuple containing the resolved function, updated arguments list, and updated arity.

        Details:
        - f_args is the list of arguments provided to the function during the current invocation.
        - f.args (if f is an instance of KGFn) represents predefined arguments associated with the function.
        These are arguments that are already set by virtue of the function being a projection of another function.
        - During the resolution process, if the function f is a KGFn with predefined arguments (projections),
        these arguments are appended to the f_args list.
        - The resolution process ensures that if there are placeholders in the predefined arguments, they are
        filled in with values from the provided arguments. However, if the function itself is being projected,
        f_args can still contain a None indicating an empty projection slot.
        - By the end of the resolution, f_args contains the arguments that will be passed to the function for
        evaluation. It is possible for f_args to finally contain None if the function is itself being projected.
        - If `f.a` (the main function or symbol) resolves to another function `_f`, it might indicate a higher-order function scenario.
        - In such a case, `f_args` supplies a function as an argument to `f.a`.
        - The `f.args` are then arguments for the function provided by `f_args`.
        - Even if this function (`_f`) corresponds to a reserved symbol, it should still undergo the projection flattening process in subsequent recursive calls.

        """
        tf = type(f)
        if tf is KGSym:
            try:
                _f = self._context[f]
                t_f = type(_f)
                if t_f is KGFn or t_f is KGCall or t_f in _kglambda_types or _is_kglambda_type(t_f) or f not in reserved_fn_symbols_set:
                    f = _f
                    tf = t_f
                else:
                    return f, f_args, f_arity
            except KeyError:
                if f not in reserved_fn_symbols_set:
                    raise KlongException(f"undefined: {f}")
        if f_arity > 0 and (tf is KGFn or tf is KGCall) and not f._is_op and not f._is_adverb_chain:
            if f.args is None:
                # if f.args is None, then there are no projections in place and we use f_args entirely for the function.
                return f.a, f_args, f.arity
            elif has_none(f.args):
                f_args.append(f.args if type(f.args) is list else [f.args])
                return f.a, f_args, f.arity
        return f, f_args, f_arity

    def _eval_fn(self, x: KGFn):
        """

        Evaluate a Klong function.

        The incoming "x" argument is the description of the function to be evaluated.

        x.a may be a symbol or an actual function. If it's a symbol and a reserved symbol, then it should be resolved to a function.
        x.args contains the arguments to be passed to the function.
        x.arity contains the arity of the function.

        Processing before function execution is to build the local context for the function so that
        when it references x,y and z it retreives the arguments provided to the function.

        In practice, its often way more complex to derived these arguemnts because:

        * Functions may contain references to symbols which need to be resolved
        * Functions may be projections or a invocation of a projection and
            arguments need to be "flattened".

        To address these points:

        * Projections are recursively merged with arguments and mapped to x,y, and z as appropriate.
        * A new runtime context is prepared and the function is invoked.
        * Locals are populated with identity first in the local context.
        * The system var .f is populated in the local context and made available to the function.

        Notes:

            In higher-order function scenarios, the x.args contains the function referenced by the symbol in by x.a.
            Subsequent processing will then use the arguments attached to the referenced function as the basis for projection flattening.

        """

        # Fast path: use cached resolution if available and still valid
        _ctx = self._context
        _ctx_list = _ctx._context
        _tx = type(x)
        if _tx is KGCall and x._cached_version == _ctx._lookup_version:
            f = x._cached_body
            # Specialized hot path: _cached_cond_fast implies nargs_ok, KGCond body, nargs==1
            if x._cached_cond_fast:
                # Inline arg eval for nargs==1
                if x._arg0_dyad_fast:
                    # Pre-cached: arg sym, literal, op_a
                    _ay = x._arg0_literal
                    _ax = _ctx._fast_x
                    if _ax is None:
                        _ax = self.eval(x._arg0_sym)
                    if type(_ax) is int:
                        _aop = x._arg0_op_a
                        if _aop == '-':
                            _xval = _ax - _ay
                        elif _aop == '+':
                            _xval = _ax + _ay
                        elif _aop == '*':
                            _xval = _ax * _ay
                        else:
                            _xval = self._vd[_aop](_ax, _ay)
                    else:
                        _xval = self._vd[x._arg0_op_a](_ax, _ay)
                elif x._arg0_is_dyad_op:
                    q = x._f_args[0]
                    _afa = q.args
                    if type(_afa) is not list:
                        _afa = [_afa] if _afa is not None else _afa
                    _afa1 = _afa[1]
                    _at1 = type(_afa1)
                    _ay = _afa1 if _at1 is int or _at1 is float else self.eval(_afa1)
                    _afa0 = _afa[0]
                    _at0 = type(_afa0)
                    if _at0 is KGSym and _afa0 in reserved_fn_symbols_set:
                        _ax = _ctx._fast_x if _afa0 is _sym_x else _ctx_list[-1].get(_afa0)
                        if _ax is None:
                            _ax = self.eval(_afa0)
                    elif _at0 is int or _at0 is float:
                        _ax = _afa0
                    else:
                        _ax = self.eval(_afa0)
                    if type(_ax) is int and type(_ay) is int:
                        _aop = q._op_a
                        if _aop == '-':
                            _xval = _ax - _ay
                        elif _aop == '+':
                            _xval = _ax + _ay
                        elif _aop == '*':
                            _xval = _ax * _ay
                        else:
                            _afast = q._fast_op
                            _xval = _afast(_ax, _ay) if _afast is not None else self._vd[_aop](_ax, _ay)
                    else:
                        _afast = q._fast_op
                        if _afast is not None and (type(_ax) is int or type(_ax) is float) and (type(_ay) is int or type(_ay) is float):
                            _xval = _afast(_ax, _ay)
                        else:
                            _xval = self._vd[q._op_a](_ax, _ay)
                else:
                    q = x._f_args[0]
                    tq = type(q)
                    if tq is int or tq is float or tq is numpy.ndarray or tq in _numpy_scalar_types:
                        _xval = q
                    elif (tq is KGFn or tq is KGCall) and q._is_op:
                        _xval = self.eval(q)
                    elif tq is KGCall:
                        _xval = self._eval_fn(q)
                    else:
                        _xval = self.call(q)
                # Check for specialized closure (fib-like self-recursive pattern)
                _sfn = x._specialized_fn
                if _sfn is not None:
                    return _sfn(_xval)
                _saved_x = _ctx._fast_x
                _ctx._fast_x = _xval
                try:
                    # Condition eval — pre-cached literal, op_a, fast_op
                    _cy = x._cond_literal
                    _cx = _xval
                    if type(_cx) is int:
                        _cop_a = x._cond_op_a
                        if _cop_a == '<':
                            p = _cx < _cy
                        elif _cop_a == '>':
                            p = _cx > _cy
                        elif _cop_a == '=':
                            p = _cx == _cy
                        else:
                            _cfast = x._cond_fast_op
                            q = _cfast(_cx, _cy) if _cfast is not None else self._vd[_cop_a](_cx, _cy)
                            p = q != 0
                    else:
                        _cop_a = x._cond_op_a
                        _cfast = x._cond_fast_op
                        if _cfast is not None and (type(_cx) is int or type(_cx) is float):
                            q = _cfast(_cx, _cy)
                        else:
                            q = self._vd[_cop_a](_cx, _cy)
                        tq = type(q)
                        if tq is int or tq is float:
                            p = q != 0
                        else:
                            p = not ((self._backend.is_number(q) and q == 0) or is_empty(q))
                    xb = f[1] if p else f[2]
                    if p:
                        if x._cached_true_is_sym:
                            if xb is _sym_x:
                                return _xval
                            return self.eval(xb)
                    else:
                        _fbc0 = x._cached_fb_call0
                        if _fbc0 is not None:
                            # Ultra-fast: both false branch args are KGCall
                            _by = self._eval_fn(x._cached_fb_call1)
                            _bx = self._eval_fn(_fbc0)
                            _fbop = x._cached_fb_op_a
                            if type(_bx) is int and type(_by) is int:
                                if _fbop == '+':
                                    return _bx + _by
                                elif _fbop == '-':
                                    return _bx - _by
                                elif _fbop == '*':
                                    return _bx * _by
                                else:
                                    return self._vd[_fbop](_bx, _by)
                            else:
                                return self._vd[_fbop](_bx, _by)
                        elif x._cached_false_is_dyad_op:
                            _bfa = xb.args
                            if type(_bfa) is not list:
                                _bfa = [_bfa] if _bfa is not None else _bfa
                            _bfa1 = _bfa[1]
                            _bt1 = type(_bfa1)
                            if _bt1 is int or _bt1 is float:
                                _by = _bfa1
                            elif _bt1 is KGSym and _bfa1 in reserved_fn_symbols_set:
                                _by = _xval if _bfa1 is _sym_x else self.eval(_bfa1)
                            elif (_bt1 is KGFn or _bt1 is KGCall) and _bfa1._is_op:
                                _by = self.eval(_bfa1)
                            elif _bt1 is KGCall and not _bfa1._is_adverb_chain:
                                _by = self._eval_fn(_bfa1)
                            else:
                                _by = self.eval(_bfa1)
                            _bop_a = xb._op_a
                            if _bop_a in _UNEVALUATED_OPS:
                                _bx = _bfa[0]
                            else:
                                _bfa0 = _bfa[0]
                                _bt0 = type(_bfa0)
                                if _bt0 is KGSym and _bfa0 in reserved_fn_symbols_set:
                                    _bx = _xval if _bfa0 is _sym_x else self.eval(_bfa0)
                                elif _bt0 is int or _bt0 is float:
                                    _bx = _bfa0
                                elif (_bt0 is KGFn or _bt0 is KGCall) and _bfa0._is_op:
                                    _bx = self.eval(_bfa0)
                                elif _bt0 is KGCall and not _bfa0._is_adverb_chain:
                                    _bx = self._eval_fn(_bfa0)
                                else:
                                    _bx = self.eval(_bfa0)
                            if type(_bx) is int and type(_by) is int:
                                if _bop_a == '+':
                                    return _bx + _by
                                elif _bop_a == '-':
                                    return _bx - _by
                                elif _bop_a == '*':
                                    return _bx * _by
                                else:
                                    _bfast = xb._fast_op
                                    if _bfast is not None:
                                        return _bfast(_bx, _by)
                                    return self._vd[_bop_a](_bx, _by)
                            else:
                                _bfast = xb._fast_op
                                if _bfast is not None and (type(_bx) is int or type(_bx) is float) and (type(_by) is int or type(_by) is float):
                                    return _bfast(_bx, _by)
                                return self._vd[_bop_a](_bx, _by)
                    txb = type(xb)
                    if txb is int or txb is float:
                        return xb
                    if txb is KGSym:
                        if xb in reserved_fn_symbols_set:
                            if xb is _sym_x:
                                return _xval
                        return self.eval(xb)
                    if (txb is KGCall or txb is KGFn) and xb._is_op:
                        return self.eval(xb)
                    if txb is KGCall and not xb._is_adverb_chain:
                        return self._eval_fn(xb)
                    return self.call(xb)
                finally:
                    _ctx._fast_x = _saved_x
            if not x._cached_nargs_ok:
                return x
            tf = x._cached_body_type
            f_args = x._f_args
            nargs = x._nargs
        else:
            f_arity = x.arity
            f = x.a
            f_args = [None] if x.args is None else [x.args if type(x.args) is list else [x.args]]
            # Fast path: inline first resolve for the common case
            tf = type(f)
            if tf is KGSym:
                try:
                    # Fast path: check lookup cache directly for non-reserved symbols
                    _f = _ctx._lookup_cache.get(f) if f not in reserved_fn_symbols_set else None
                    if _f is None:
                        _f = _ctx[f]
                    t_f = type(_f)
                    if t_f is KGFn or t_f is KGCall or t_f in _kglambda_types or _is_kglambda_type(t_f) or f not in reserved_fn_symbols_set:
                        f = _f
                        tf = t_f
                        # Check if we can unwrap the function directly
                        if f_arity > 0 and (tf is KGFn or tf is KGCall) and not f._is_op and not f._is_adverb_chain:
                            if f.args is None:
                                f, f_arity = f.a, f.arity
                                tf = type(f)
                            elif has_none(f.args):
                                f_args.append(f.args if type(f.args) is list else [f.args])
                                f, f_arity = f.a, f.arity
                                tf = type(f)
                except KeyError:
                    if f not in reserved_fn_symbols_set:
                        raise KlongException(f"undefined: {f}")
            elif tf is KGFn or tf is KGCall:
                if f_arity > 0 and not f._is_op and not f._is_adverb_chain:
                    if f.args is None:
                        f, f_arity = f.a, f.arity
                        tf = type(f)
                    elif has_none(f.args):
                        f_args.append(f.args if type(f.args) is list else [f.args])
                        f, f_arity = f.a, f.arity
                        tf = type(f)
            # Continue with remaining passes if needed (skip ops — they're already resolved)
            if tf is KGSym or ((tf is KGFn or tf is KGCall) and not f._is_op):
                f, f_args, f_arity = self._resolve_fn(f, f_args, f_arity)
                tf = type(f)
                if tf is KGSym or ((tf is KGFn or tf is KGCall) and not f._is_op):
                    f, f_args, f_arity = self._resolve_fn(f, f_args, f_arity)
                    tf = type(f)
            # Cache the resolution for next time (only when no projection merging occurred)
            if _tx is KGCall and len(f_args) == 1:
                x._cached_body = f
                x._cached_body_arity = f_arity
                x._cached_body_type = tf
                x._cached_version = _ctx._lookup_version
                x._cached_nargs_ok = x._nargs >= f_arity
                # Pre-compute KGCond body dispatch hints
                if tf is KGCond:
                    _x0 = f[0]
                    _tx0 = type(_x0)
                    _cond_is_dyad = (_tx0 is KGCall or _tx0 is KGFn) and _x0._is_op and _x0._op_arity == 2
                    x._cached_cond_is_dyad_op = _cond_is_dyad
                    # Pre-compute fast condition path: dyad op with list args, arg0 IS _sym_x, literal arg1, nargs==1
                    if _cond_is_dyad:
                        _cfa = _x0.args
                        _cond_fast = (x._nargs == 1 and type(_cfa) is list and
                            _cfa[0] is _sym_x and
                            (type(_cfa[1]) is int or type(_cfa[1]) is float))
                        x._cached_cond_fast = _cond_fast
                        if _cond_fast:
                            x._cond_literal = _cfa[1]
                            x._cond_op_a = _x0._op_a
                            x._cond_fast_op = _x0._fast_op
                    else:
                        x._cached_cond_fast = False
                    # Pre-compute branch types
                    _f1 = f[1]
                    _f2 = f[2]
                    x._cached_true_is_sym = type(_f1) is KGSym and _f1 in reserved_fn_symbols_set
                    _tf2 = type(_f2)
                    _f2_is_dyad = (_tf2 is KGCall or _tf2 is KGFn) and _f2._is_op and _f2._op_arity == 2
                    x._cached_false_is_dyad_op = _f2_is_dyad
                    # Pre-cache false branch dual-call pattern: op(call0, call1)
                    if _f2_is_dyad:
                        _f2a = _f2.args
                        if type(_f2a) is list and len(_f2a) == 2:
                            _fb0, _fb1 = _f2a[0], _f2a[1]
                            _tfb0, _tfb1 = type(_fb0), type(_fb1)
                            if (_tfb0 is KGCall and not _fb0._is_op and not _fb0._is_adverb_chain and
                                _tfb1 is KGCall and not _fb1._is_op and not _fb1._is_adverb_chain):
                                x._cached_fb_call0 = _fb0
                                x._cached_fb_call1 = _fb1
                                x._cached_fb_op_a = _f2._op_a
                                # Detect fib-like self-recursive pattern for closure generation
                                # Pattern: {:[x OP N; x; self(x OP A) BOP self(x OP B)]}
                                _xa = x.a
                                if (_cond_fast and
                                    type(_f1) is KGSym and _f1 is _sym_x and
                                    type(_xa) is KGSym and
                                    type(_fb0.a) is KGSym and _fb0.a is _xa and
                                    type(_fb1.a) is KGSym and _fb1.a is _xa and
                                    _fb0._arg0_dyad_fast and _fb1._arg0_dyad_fast):
                                    _sfn = _make_recursive_closure(
                                        _x0._op_a, _cfa[1],
                                        _fb0._arg0_op_a, _fb0._arg0_literal,
                                        _fb1._arg0_op_a, _fb1._arg0_literal,
                                        _f2._op_a)
                                    if _sfn is not None:
                                        x._specialized_fn = _sfn
                else:
                    x._cached_cond_is_dyad_op = False
                    x._cached_cond_fast = False
                    x._cached_true_is_sym = False
                    x._cached_false_is_dyad_op = False

            if len(f_args) == 1:
                f_args = f_args[0]
                nargs = 0 if f_args is None else len(f_args)
                if nargs < f_arity:
                    return x
            else:
                f_args.reverse()
                f_args = merge_projections(f_args)
                nargs = 0 if f_args is None else len(f_args)
                if nargs < f_arity or has_none(f_args):
                    return x

        if f_args is None:
            ctx = {}
        else:
            # Inline call() dispatch for op args to skip method call overhead
            if nargs == 1:
                q = f_args[0]
                if x._arg0_dyad_fast:
                    # Ultra-fast: args is list, arg0 is reserved sym, arg1 is literal
                    _afa = q.args
                    _ay = _afa[1]
                    _ax = _ctx_list[-1].get(_afa[0])
                    if _ax is None:
                        _ax = self.eval(_afa[0])
                    if type(_ax) is int:
                        _aop = q._op_a
                        if _aop == '-':
                            _xval = _ax - _ay
                        elif _aop == '+':
                            _xval = _ax + _ay
                        elif _aop == '*':
                            _xval = _ax * _ay
                        else:
                            _afast = q._fast_op
                            _xval = _afast(_ax, _ay) if _afast is not None else self._vd[_aop](_ax, _ay)
                    else:
                        _afast = q._fast_op
                        if _afast is not None and (type(_ax) is int or type(_ax) is float):
                            _xval = _afast(_ax, _ay)
                        else:
                            _xval = self._vd[q._op_a](_ax, _ay)
                elif x._arg0_is_dyad_op:
                    # Medium path: dyad op but args need full type checking
                    _afa = q.args
                    if type(_afa) is not list:
                        _afa = [_afa] if _afa is not None else _afa
                    _afa1 = _afa[1]
                    _at1 = type(_afa1)
                    _ay = _afa1 if _at1 is int or _at1 is float else self.eval(_afa1)
                    _afa0 = _afa[0]
                    _at0 = type(_afa0)
                    if _at0 is KGSym and _afa0 in reserved_fn_symbols_set:
                        _ax = _ctx_list[-1].get(_afa0)
                        if _ax is None:
                            _ax = self.eval(_afa0)
                    elif _at0 is int or _at0 is float:
                        _ax = _afa0
                    else:
                        _ax = self.eval(_afa0)
                    if type(_ax) is int and type(_ay) is int:
                        _aop = q._op_a
                        if _aop == '-':
                            _xval = _ax - _ay
                        elif _aop == '+':
                            _xval = _ax + _ay
                        elif _aop == '*':
                            _xval = _ax * _ay
                        else:
                            _afast = q._fast_op
                            _xval = _afast(_ax, _ay) if _afast is not None else self._vd[_aop](_ax, _ay)
                    else:
                        _afast = q._fast_op
                        if _afast is not None and (type(_ax) is int or type(_ax) is float) and (type(_ay) is int or type(_ay) is float):
                            _xval = _afast(_ax, _ay)
                        else:
                            _xval = self._vd[q._op_a](_ax, _ay)
                else:
                    tq = type(q)
                    if tq is int or tq is float or tq is numpy.ndarray or tq in _numpy_scalar_types:
                        _xval = q
                    elif tq is KGSym:
                        # Fast path: skip call() overhead for symbol args
                        if q in reserved_fn_symbols_set:
                            _xval = _ctx_list[-1].get(q)
                            if _xval is None:
                                _xval = self.eval(q)
                        else:
                            _xval = self.eval(q)
                    elif (tq is KGFn or tq is KGCall) and q._is_op:
                        _xval = self.eval(q)
                    elif tq is KGCall:
                        _xval = self._eval_fn(q)
                    else:
                        _xval = self.call(q)
                ctx = {_sym_x: _xval}
            elif nargs == 2:
                q0, q1 = f_args[0], f_args[1]
                tq0, tq1 = type(q0), type(q1)
                v0 = q0 if tq0 is int or tq0 is float or tq0 is numpy.ndarray or tq0 in _numpy_scalar_types else (self.eval(q0) if (tq0 is KGFn or tq0 is KGCall) and q0._is_op else self.call(q0))
                v1 = q1 if tq1 is int or tq1 is float or tq1 is numpy.ndarray or tq1 in _numpy_scalar_types else (self.eval(q1) if (tq1 is KGFn or tq1 is KGCall) and q1._is_op else self.call(q1))
                ctx = {_sym_x: v0, _sym_y: v1}
            else:
                ctx = {}
                for sym, q in zip(reserved_fn_symbols, f_args):
                    tq = type(q)
                    ctx[sym] = q if tq is int or tq is float or tq is numpy.ndarray or tq in _numpy_scalar_types else self.call(q)

        has_locals = (tf is list or (tf is numpy.ndarray and f.ndim > 0)) and len(f) > 1 and is_list(f[0]) and len(f[0]) > 0
        if has_locals:
            # Filter out semicolons — remaining elements are local variable declarations
            params = [q for q in f[0] if type(q) is KGSym]
            if len(params) > 0:
                for q in params:
                    # Don't overwrite function parameters (x, y, z)
                    if q not in ctx:
                        ctx[q] = q
                f = f[1:]
            else:
                has_locals = False

        ctx[reserved_dot_f_symbol] = f

        # Inline push/pop for speed — avoid method call overhead
        if has_locals:
            _ctx.push(ctx)
        else:
            _ctx_list.append(ctx)
        try:
            # Recompute tf only if has_locals modified f (f = f[1:] changes type)
            if has_locals:
                tf = type(f)
            # KGCond first: most common for recursive/conditional function bodies
            if tf is KGCond:
                x0 = f[0]
                if x._cached_cond_fast:
                    # Ultra-fast path: nargs==1, arg0 IS _sym_x, arg1 is literal
                    # _xval is the computed x value — use it directly (no ctx.get)
                    _cfa = x0.args
                    _cy = _cfa[1]
                    _cx = _xval
                    if type(_cx) is int and type(_cy) is int:
                        _cop_a = x0._op_a
                        if _cop_a == '<':
                            p = _cx < _cy
                        elif _cop_a == '>':
                            p = _cx > _cy
                        elif _cop_a == '=':
                            p = _cx == _cy
                        else:
                            _cfast = x0._fast_op
                            q = _cfast(_cx, _cy) if _cfast is not None else self._vd[_cop_a](_cx, _cy)
                            p = q != 0
                    else:
                        _cop_a = x0._op_a
                        _cfast = x0._fast_op
                        if _cfast is not None and (type(_cx) is int or type(_cx) is float) and (type(_cy) is int or type(_cy) is float):
                            q = _cfast(_cx, _cy)
                        else:
                            q = self._vd[_cop_a](_cx, _cy)
                        tq = type(q)
                        if tq is int or tq is float:
                            p = q != 0
                        else:
                            p = not ((self._backend.is_number(q) and q == 0) or is_empty(q))
                    xb = f[1] if p else f[2]
                    # Pre-computed fast paths for branch dispatch
                    if p:
                        if x._cached_true_is_sym:
                            if xb is _sym_x:
                                return _xval
                            _v = ctx.get(xb)
                            if _v is not None:
                                return _v
                            return self.eval(xb)
                    else:
                        _fbc0 = x._cached_fb_call0
                        if _fbc0 is not None:
                            _by = self._eval_fn(x._cached_fb_call1)
                            _bx = self._eval_fn(_fbc0)
                            _fbop = x._cached_fb_op_a
                            if type(_bx) is int and type(_by) is int:
                                if _fbop == '+':
                                    return _bx + _by
                                elif _fbop == '-':
                                    return _bx - _by
                                elif _fbop == '*':
                                    return _bx * _by
                                else:
                                    return self._vd[_fbop](_bx, _by)
                            else:
                                return self._vd[_fbop](_bx, _by)
                        elif x._cached_false_is_dyad_op:
                            # Inline dyad op for KGCond branch (e.g., (fib(x-1))+(fib(x-2)))
                            _bfa = xb.args
                            if type(_bfa) is not list:
                                _bfa = [_bfa] if _bfa is not None else _bfa
                            _bfa1 = _bfa[1]
                            _bt1 = type(_bfa1)
                            if _bt1 is int or _bt1 is float:
                                _by = _bfa1
                            elif _bt1 is KGSym and _bfa1 in reserved_fn_symbols_set:
                                _by = ctx.get(_bfa1)
                                if _by is None:
                                    _by = self.eval(_bfa1)
                            elif (_bt1 is KGFn or _bt1 is KGCall) and _bfa1._is_op:
                                _by = self.eval(_bfa1)
                            elif _bt1 is KGCall and not _bfa1._is_adverb_chain:
                                _by = self._eval_fn(_bfa1)
                            else:
                                _by = self.eval(_bfa1)
                            _bop_a = xb._op_a
                            if _bop_a in _UNEVALUATED_OPS:
                                _bx = _bfa[0]
                            else:
                                _bfa0 = _bfa[0]
                                _bt0 = type(_bfa0)
                                if _bt0 is KGSym and _bfa0 in reserved_fn_symbols_set:
                                    _bx = ctx.get(_bfa0)
                                    if _bx is None:
                                        _bx = self.eval(_bfa0)
                                elif _bt0 is int or _bt0 is float:
                                    _bx = _bfa0
                                elif (_bt0 is KGFn or _bt0 is KGCall) and _bfa0._is_op:
                                    _bx = self.eval(_bfa0)
                                elif _bt0 is KGCall and not _bfa0._is_adverb_chain:
                                    _bx = self._eval_fn(_bfa0)
                                else:
                                    _bx = self.eval(_bfa0)
                            if type(_bx) is int and type(_by) is int:
                                if _bop_a == '+':
                                    return _bx + _by
                                elif _bop_a == '-':
                                    return _bx - _by
                                elif _bop_a == '*':
                                    return _bx * _by
                                else:
                                    _bfast = xb._fast_op
                                    if _bfast is not None:
                                        return _bfast(_bx, _by)
                                    return self._vd[_bop_a](_bx, _by)
                            else:
                                _bfast = xb._fast_op
                                if _bfast is not None and (type(_bx) is int or type(_bx) is float) and (type(_by) is int or type(_by) is float):
                                    return _bfast(_bx, _by)
                                return self._vd[_bop_a](_bx, _by)
                    # General branch dispatch (fallback for ultra-fast path)
                    txb = type(xb)
                    if txb is int or txb is float:
                        return xb
                    if txb is KGSym:
                        if xb in reserved_fn_symbols_set:
                            _v = ctx.get(xb)
                            if _v is not None:
                                return _v
                        return self.eval(xb)
                    if (txb is KGCall or txb is KGFn) and xb._is_op:
                        return self.eval(xb)
                    if txb is KGCall and not xb._is_adverb_chain:
                        return self._eval_fn(xb)
                    return self.call(xb)
                elif x._cached_cond_is_dyad_op:
                    # Medium path: condition is a dyad op but args need type checking
                    _cfa = x0.args
                    if type(_cfa) is not list:
                        _cfa = [_cfa] if _cfa is not None else _cfa
                    _cfa1 = _cfa[1]
                    _ct1 = type(_cfa1)
                    _cy = _cfa1 if _ct1 is int or _ct1 is float else self.eval(_cfa1)
                    _cfa0 = _cfa[0]
                    _ct0 = type(_cfa0)
                    if _ct0 is KGSym and _cfa0 in reserved_fn_symbols_set:
                        _cx = ctx.get(_cfa0)
                        if _cx is None:
                            _cx = self.eval(_cfa0)
                    elif _ct0 is int or _ct0 is float:
                        _cx = _cfa0
                    else:
                        _cx = self.eval(_cfa0)
                    _cop_a = x0._op_a
                    if type(_cx) is int and type(_cy) is int:
                        if _cop_a == '<':
                            q = 1 if _cx < _cy else 0
                        elif _cop_a == '>':
                            q = 1 if _cx > _cy else 0
                        elif _cop_a == '=':
                            q = 1 if _cx == _cy else 0
                        else:
                            _cfast = x0._fast_op
                            q = _cfast(_cx, _cy) if _cfast is not None else self._vd[_cop_a](_cx, _cy)
                    else:
                        _cfast = x0._fast_op
                        if _cfast is not None and (type(_cx) is int or type(_cx) is float) and (type(_cy) is int or type(_cy) is float):
                            q = _cfast(_cx, _cy)
                        else:
                            q = self._vd[_cop_a](_cx, _cy)
                else:
                    tx0 = type(x0)
                    if tx0 is int or tx0 is float:
                        q = x0
                    elif (tx0 is KGCall or tx0 is KGFn) and x0._is_op:
                        q = self.eval(x0)
                    else:
                        q = self.call(x0)
                tq = type(q)
                if tq is int or tq is float:
                    p = q != 0
                else:
                    p = not ((self._backend.is_number(q) and q == 0) or is_empty(q))
                xb = f[1] if p else f[2]
                # Pre-computed fast paths for branch dispatch
                if p:
                    if x._cached_true_is_sym:
                        _v = ctx.get(xb)
                        if _v is not None:
                            return _v
                        return self.eval(xb)
                else:
                    _fbc0 = x._cached_fb_call0
                    if _fbc0 is not None:
                        _by = self._eval_fn(x._cached_fb_call1)
                        _bx = self._eval_fn(_fbc0)
                        _fbop = x._cached_fb_op_a
                        if type(_bx) is int and type(_by) is int:
                            if _fbop == '+':
                                return _bx + _by
                            elif _fbop == '-':
                                return _bx - _by
                            elif _fbop == '*':
                                return _bx * _by
                            else:
                                return self._vd[_fbop](_bx, _by)
                        else:
                            return self._vd[_fbop](_bx, _by)
                    elif x._cached_false_is_dyad_op:
                        # Inline dyad op for KGCond branch (e.g., (fib(x-1))+(fib(x-2)))
                        _bfa = xb.args
                        if type(_bfa) is not list:
                            _bfa = [_bfa] if _bfa is not None else _bfa
                        _bfa1 = _bfa[1]
                        _bt1 = type(_bfa1)
                        if _bt1 is int or _bt1 is float:
                            _by = _bfa1
                        elif _bt1 is KGSym and _bfa1 in reserved_fn_symbols_set:
                            _by = ctx.get(_bfa1)
                            if _by is None:
                                _by = self.eval(_bfa1)
                        elif (_bt1 is KGFn or _bt1 is KGCall) and _bfa1._is_op:
                            _by = self.eval(_bfa1)
                        elif _bt1 is KGCall and not _bfa1._is_adverb_chain:
                            _by = self._eval_fn(_bfa1)
                        else:
                            _by = self.eval(_bfa1)
                        _bop_a = xb._op_a
                        if _bop_a in _UNEVALUATED_OPS:
                            _bx = _bfa[0]
                        else:
                            _bfa0 = _bfa[0]
                            _bt0 = type(_bfa0)
                            if _bt0 is KGSym and _bfa0 in reserved_fn_symbols_set:
                                _bx = ctx.get(_bfa0)
                                if _bx is None:
                                    _bx = self.eval(_bfa0)
                            elif _bt0 is int or _bt0 is float:
                                _bx = _bfa0
                            elif (_bt0 is KGFn or _bt0 is KGCall) and _bfa0._is_op:
                                _bx = self.eval(_bfa0)
                            elif _bt0 is KGCall and not _bfa0._is_adverb_chain:
                                _bx = self._eval_fn(_bfa0)
                            else:
                                _bx = self.eval(_bfa0)
                        if type(_bx) is int and type(_by) is int:
                            if _bop_a == '+':
                                return _bx + _by
                            elif _bop_a == '-':
                                return _bx - _by
                            elif _bop_a == '*':
                                return _bx * _by
                            else:
                                _bfast = xb._fast_op
                                if _bfast is not None:
                                    return _bfast(_bx, _by)
                                return self._vd[_bop_a](_bx, _by)
                        else:
                            _bfast = xb._fast_op
                            if _bfast is not None and (type(_bx) is int or type(_bx) is float) and (type(_by) is int or type(_by) is float):
                                return _bfast(_bx, _by)
                            return self._vd[_bop_a](_bx, _by)
                # General branch dispatch (fallback when pre-computed flags are False)
                txb = type(xb)
                if txb is int or txb is float:
                    return xb
                if txb is KGSym:
                    if xb in reserved_fn_symbols_set:
                        _v = ctx.get(xb)
                        if _v is not None:
                            return _v
                    return self.eval(xb)
                if (txb is KGCall or txb is KGFn) and xb._is_op and xb._op_arity == 2:
                    _bfa = xb.args
                    if type(_bfa) is not list:
                        _bfa = [_bfa] if _bfa is not None else _bfa
                    _bfa1 = _bfa[1]
                    _bt1 = type(_bfa1)
                    if _bt1 is int or _bt1 is float:
                        _by = _bfa1
                    elif _bt1 is KGSym and _bfa1 in reserved_fn_symbols_set:
                        _by = ctx.get(_bfa1)
                        if _by is None:
                            _by = self.eval(_bfa1)
                    elif (_bt1 is KGFn or _bt1 is KGCall) and _bfa1._is_op:
                        _by = self.eval(_bfa1)
                    elif _bt1 is KGCall and not _bfa1._is_adverb_chain:
                        _by = self._eval_fn(_bfa1)
                    else:
                        _by = self.eval(_bfa1)
                    _bop_a = xb._op_a
                    if _bop_a in _UNEVALUATED_OPS:
                        _bx = _bfa[0]
                    else:
                        _bfa0 = _bfa[0]
                        _bt0 = type(_bfa0)
                        if _bt0 is KGSym and _bfa0 in reserved_fn_symbols_set:
                            _bx = ctx.get(_bfa0)
                            if _bx is None:
                                _bx = self.eval(_bfa0)
                        elif _bt0 is int or _bt0 is float:
                            _bx = _bfa0
                        elif (_bt0 is KGFn or _bt0 is KGCall) and _bfa0._is_op:
                            _bx = self.eval(_bfa0)
                        elif _bt0 is KGCall and not _bfa0._is_adverb_chain:
                            _bx = self._eval_fn(_bfa0)
                        else:
                            _bx = self.eval(_bfa0)
                    if type(_bx) is int and type(_by) is int:
                        if _bop_a == '+':
                            return _bx + _by
                        elif _bop_a == '-':
                            return _bx - _by
                        elif _bop_a == '*':
                            return _bx * _by
                        else:
                            _bfast = xb._fast_op
                            if _bfast is not None:
                                return _bfast(_bx, _by)
                            return self._vd[_bop_a](_bx, _by)
                    else:
                        _bfast = xb._fast_op
                        if _bfast is not None and (type(_bx) is int or type(_bx) is float) and (type(_by) is int or type(_by) is float):
                            return _bfast(_bx, _by)
                        return self._vd[_bop_a](_bx, _by)
                if (txb is KGCall or txb is KGFn) and xb._is_op:
                    return self.eval(xb)
                if txb is KGCall and not xb._is_adverb_chain:
                    return self._eval_fn(xb)
                return self.call(xb)
            # Op body (e.g., x+1)
            if (tf is KGCall or tf is KGFn) and f._is_op:
                op_a = f._op_a
                fa = f.args
                if f._op_arity == 2:
                    if type(fa) is not list:
                        fa = [fa] if fa is not None else fa
                    fa1 = fa[1]
                    t1 = type(fa1)
                    _y = fa1 if t1 is int or t1 is float or t1 is numpy.ndarray else self.eval(fa1)
                    if op_a in _UNEVALUATED_OPS:
                        _x = fa[0]
                    else:
                        fa0 = fa[0]
                        t0 = type(fa0)
                        if t0 is KGSym and fa0 in reserved_fn_symbols_set:
                            _x = ctx.get(fa0)
                            if _x is None:
                                _x = self.eval(fa0)
                        elif t0 is int or t0 is float or t0 is numpy.ndarray:
                            _x = fa0
                        else:
                            _x = self.eval(fa0)
                    # Fast path: inline operators for int-int case
                    if type(_x) is int and type(_y) is int:
                        if op_a == '+':
                            return _x + _y
                        elif op_a == '-':
                            return _x - _y
                        elif op_a == '*':
                            return _x * _y
                        elif op_a == '<':
                            return 1 if _x < _y else 0
                        elif op_a == '>':
                            return 1 if _x > _y else 0
                        elif op_a == '=':
                            return 1 if _x == _y else 0
                        else:
                            _fast_op = f._fast_op
                            if _fast_op is not None:
                                return _fast_op(_x, _y)
                            return self._vd[op_a](_x, _y)
                    else:
                        _fast_op = f._fast_op
                        if _fast_op is not None and (type(_x) is int or type(_x) is float) and (type(_y) is int or type(_y) is float):
                            return _fast_op(_x, _y)
                        return self._vd[op_a](_x, _y)
                else:
                    _x = fa if type(fa) is not list else fa[0]
                    if op_a not in _UNEVALUATED_OPS:
                        tx_x = type(_x)
                        if tx_x is KGSym and _x in reserved_fn_symbols_set:
                            v = ctx.get(_x)
                            if v is not None:
                                _x = v
                            else:
                                _x = self.eval(_x)
                        elif tx_x is not int and tx_x is not float and tx_x is not numpy.ndarray:
                            _x = self.eval(_x)
                    # Fast path: use pre-cached Python operators for scalar int/float monad
                    _fast_monad = f._fast_monad
                    if _fast_monad is not None and (type(_x) is int or type(_x) is float):
                        return _fast_monad(_x)
                    return self._vm[op_a](_x)
            if tf is int or tf is float:
                return f
            if tf in _kglambda_types or _is_kglambda_type(tf):
                return f(self, _ctx)
            # Route KGFn (recursive function calls) through _eval_fn, everything else through eval directly
            if tf is KGFn and not f._is_op and not f._is_adverb_chain:
                return self._eval_fn(f)
            return self.eval(f)
        finally:
            if has_locals:
                _ctx.pop()
            else:
                if len(_ctx_list) > _ctx._min_ctx_count:
                    _ctx_list.pop()

    def call(self, x):
        """

        Invoke a Klong program (as produced by prog()), causing functions to be called and evaluated.

        """
        tx = type(x)
        if tx is int or tx is float:
            return x
        if tx is KGCall:
            return self.eval(x)
        if tx is KGFn:
            # eval already handles KGFn for ops/adverbs via (tx is KGCall or tx is KGFn)
            if x._is_op or x._is_adverb_chain:
                return self.eval(x)
            return self._eval_fn(x)
        return self.eval(x)

    def eval(self, x):
        """

        Evaluate a Klong program.

        The fundamental design ideas include:

        * Python lists contain programs and NumPy arrays contain data.
        * Functions (KGFn) are not invoked unless they are KGCall instances, allowing for function definitions to be differentiated from invocations.

        """
        tx = type(x)
        if tx is KGSym:
            # Fast path: reserved symbols (x, y, z) are always in innermost scope
            if x in reserved_fn_symbols_set:
                v = self._context._context[-1].get(x)
                if v is not None:
                    return v
                # Fallback: specialized path stores x in _fast_x (no dict push)
                if x is _sym_x:
                    v = self._context._fast_x
                    if v is not None:
                        return v
            try:
                return self._context[x]
            except KeyError:
                if x not in reserved_fn_symbols_set:
                    self._context[x] = x
                return x
        elif tx is KGCall or tx is KGFn:
            if x._is_op:
                op_a = x._op_a
                fa = x.args
                if x._op_arity == 2:
                    if type(fa) is not list:
                        fa = [fa] if fa is not None else fa
                    fa1 = fa[1]
                    t1 = type(fa1)
                    # Inline dispatch: skip eval overhead for common arg types
                    if t1 is int or t1 is float or t1 is numpy.ndarray:
                        _y = fa1
                    elif t1 is KGSym and fa1 in reserved_fn_symbols_set:
                        _y = self._context._context[-1].get(fa1)
                        if _y is None:
                            _y = self.eval(fa1)
                    elif t1 is KGCall and not fa1._is_op and not fa1._is_adverb_chain:
                        _y = self._eval_fn(fa1)
                    else:
                        _y = self.eval(fa1)
                    if op_a in _UNEVALUATED_OPS:
                        _x = fa[0]
                    else:
                        fa0 = fa[0]
                        t0 = type(fa0)
                        if t0 is int or t0 is float or t0 is numpy.ndarray:
                            _x = fa0
                        elif t0 is KGSym and fa0 in reserved_fn_symbols_set:
                            _x = self._context._context[-1].get(fa0)
                            if _x is None:
                                _x = self.eval(fa0)
                        elif t0 is KGCall and not fa0._is_op and not fa0._is_adverb_chain:
                            _x = self._eval_fn(fa0)
                        else:
                            _x = self.eval(fa0)
                    # Fast path: use pre-cached Python operators for scalar int/float
                    _fast_op = x._fast_op
                    if _fast_op is not None and (type(_x) is int or type(_x) is float) and (type(_y) is int or type(_y) is float):
                        return _fast_op(_x, _y)
                    return self._vd[op_a](_x, _y)
                else:
                    _x = fa if type(fa) is not list else fa[0]
                    if op_a not in _UNEVALUATED_OPS:
                        tx_x = type(_x)
                        if tx_x is not int and tx_x is not float and tx_x is not numpy.ndarray:
                            _x = self.eval(_x)
                    # Fast path: use pre-cached Python operators for scalar int/float monad
                    _fast_monad = x._fast_monad
                    if _fast_monad is not None and (type(_x) is int or type(_x) is float):
                        return _fast_monad(_x)
                    return self._vm[op_a](_x)
            elif x._is_adverb_chain:
                xa_id = id(x.a)
                cached = self._adverb_cache.get(xa_id)
                if cached is None:
                    cached_fn = chain_adverbs(self, x.a)
                    # Store both closure and chain list reference to prevent id reuse after GC
                    self._adverb_cache[xa_id] = (cached_fn, x.a)
                else:
                    cached_fn = cached[0]
                return cached_fn()
            elif tx is KGCall:
                return self._eval_fn(x)
        elif tx is KGCond:
            # Inline condition eval: skip call() overhead for common op case
            x0 = x[0]
            tx0 = type(x0)
            if tx0 is int or tx0 is float:
                q = x0
            elif (tx0 is KGCall or tx0 is KGFn) and x0._is_op:
                q = self.eval(x0)
            else:
                q = self.call(x0)
            tq = type(q)
            if tq is int or tq is float:
                p = q != 0
            else:
                p = not ((self._backend.is_number(q) and q == 0) or is_empty(q))
            # Inline branch eval: skip call() overhead for common cases
            xb = x[1] if p else x[2]
            txb = type(xb)
            if txb is int or txb is float:
                return xb
            if txb is KGSym:
                return self.eval(xb)
            if (txb is KGCall or txb is KGFn) and xb._is_op:
                return self.eval(xb)
            return self.call(xb)
        elif tx is list and len(x) > 0:
            return [self.call(y) for y in x][-1]
        return x

    def __call__(self, x):
        """

        Convience method for executing Klong programs.

        If the result only contains one entry, it's directly returned for convenience.

        Example:

        klong = KlongInterpreter()
        r = klong("1+1")
        assert r == 2

        or more succinctly

        assert 2 == KlongInterpreter()("1+1")

        """
        # Fast path: return cached result if available
        result = self._result_cache.get(x)
        if result is not None:
            return result
        # Check parse cache; if miss, parse inline (avoids redundant exec→parse_cache check)
        cached = self._parse_cache.get(x)
        _was_cached = cached is not None
        if not _was_cached:
            i, prog = self.prog(x)
            i = skip(x, i)
            if i < len(x) and x[i] == '}':
                raise UnexpectedChar(x, i, x[i])
            cached = prog[0] if len(prog) == 1 else prog
            self._parse_cache[x] = cached
        # Eval the cached AST
        self._result_cache_ok = True
        # Fast path: compiled expression cache (compiled lambda with variable lookups)
        _compiled = self._compiled_expr_cache.get(x)
        if _compiled is not None:
            if type(_compiled) is tuple:
                _cfn, _cvar_syms = _compiled
                result = _cfn(*(self._context[s] for s in _cvar_syms))
            else:
                # Negative cache sentinel (False) — skip compilation, use normal eval
                x0 = cached
                tx0 = type(x0)
                if tx0 is int or tx0 is float:
                    result = x0
                elif tx0 is KGCall:
                    result = self.eval(x0) if x0._is_op or x0._is_adverb_chain else self._eval_fn(x0)
                elif tx0 is KGFn and x0._is_op:
                    result = self.eval(x0)
                else:
                    result = self.call(x0)
        elif type(cached) is not list:
            x0 = cached
            tx0 = type(x0)
            # Inline call dispatch for common cases to avoid function call overhead
            if tx0 is int or tx0 is float:
                result = x0
            elif tx0 is KGCall:
                # KGCall: dispatch directly to avoid call() → eval() → _eval_fn() chain
                result = self.eval(x0) if x0._is_op or x0._is_adverb_chain else self._eval_fn(x0)
            elif tx0 is KGFn and x0._is_op:
                result = self.eval(x0)
            else:
                result = self.call(x0)
            # Try to compile top-level expression for future calls
            if _was_cached and type(cached) is not list:
                var_refs = {}
                src = _expr_to_source(cached, self, var_refs=var_refs)
                if src is not None and var_refs:
                    var_syms = list(var_refs.keys())
                    var_names = [var_refs[s] for s in var_syms]
                    fn_src = f'lambda {",".join(var_names)}: {src[0]}'
                    try:
                        fn = eval(fn_src, _EVAL_GLOBALS)
                        self._compiled_expr_cache[x] = (fn, var_syms)
                    except Exception:
                        self._compiled_expr_cache[x] = False
                else:
                    self._compiled_expr_cache[x] = False
        else:
            if not cached:
                return None
            result = [self.call(y) for y in cached][-1]
        # Only cache results on warm path (parse cache hit) — cold path may contain
        # side-effectful expressions whose results shouldn't be cached
        if _was_cached and self._result_cache_ok:
            rt = type(result)
            if rt is int or rt is float:
                self._result_cache[x] = result
            elif rt is numpy.ndarray:
                if result.ndim > 0 and result.flags.writeable:
                    result.flags.writeable = False
                self._result_cache[x] = result
            elif rt in _numpy_scalar_types or _is_numpy_scalar_type(rt):
                self._result_cache[x] = result
            elif hasattr(result, 'flags'):
                if result.flags.writeable:
                    result.flags.writeable = False
                self._result_cache[x] = result
        return result

    def exec(self, x):
        """

        Execute a Klong program.

        Each subprogram is executed in order and the resulting array contains the resulst of each sub-program.

        """
        cached = self._parse_cache.get(x)
        if cached is not None:
            # Single-expression programs stored unwrapped
            return [self.call(cached)] if type(cached) is not list else [self.call(y) for y in cached]
        i, prog = self.prog(x)
        i = skip(x, i)
        if i < len(x) and x[i] == '}':
            raise UnexpectedChar(x, i, x[i])
        # Store single-expression programs unwrapped for fast __call__ dispatch
        self._parse_cache[x] = prog[0] if len(prog) == 1 else prog
        return [self.call(y) for y in prog]
