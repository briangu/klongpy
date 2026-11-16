"""
Expression compiler for KlongPy.

Compiles Klong expression ASTs to a backend-neutral IR (tuple tree),
then delegates to the active backend for platform-specific code generation.
Falls back to None (interpreter handles it) for anything it can't compile.
"""
import logging
from collections import OrderedDict

from .types import KGSym, KGFn, KGCall, KGOp, KGAdverb

_log = logging.getLogger(__name__)

# Returned by run_compiled when the compiled function must not be used for
# the current values (see run_compiled). The caller evaluates normally.
NOT_COMPILED = object()

# Exceptions that mean "this AST cannot be compiled". They are raised while
# the AST is walked or the source is generated, before any Klong code runs,
# so falling back to the interpreter for them cannot repeat a side effect.
# Nothing is caught once a compiled function is running: see run_compiled.
_UNSUPPORTED = (AttributeError, IndexError, KeyError, TypeError, ValueError, NotImplementedError, SyntaxError)

CACHE_LIMIT = 2048


class LRUCache(OrderedDict):
    """A dict that keeps at most `maxsize` entries, dropping the least recently used."""

    def __init__(self, maxsize=CACHE_LIMIT):
        super().__init__()
        self.maxsize = maxsize

    def get(self, key, default=None):
        try:
            self.move_to_end(key)
        except KeyError:
            return default
        return OrderedDict.__getitem__(self, key)

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self.move_to_end(key)
        while len(self) > self.maxsize:
            self.popitem(last=False)


class CompiledExpr:
    """A compiled function with the operand kinds it was compiled for."""
    __slots__ = ('fn', 'var_syms', 'kinds')

    def __init__(self, fn, var_syms, kinds):
        self.fn = fn
        self.var_syms = var_syms
        self.kinds = kinds


# Python integers outside this range are exact in Klong (ticket 025) but are not
# numpy integers; a compiled expression never sees one.
_INT64_MIN = -2 ** 63
_INT64_MAX = 2 ** 63 - 1


def _kind(val, klong):
    """Operand class a compiled function is valid for, or None if not compilable."""
    tv = type(val)
    if tv is int:
        return int if _INT64_MIN <= val <= _INT64_MAX else None
    if tv is float:
        return tv
    if isinstance(val, klong._backend.np.ndarray):
        dt = val.dtype
        k = getattr(dt, 'kind', None) or str(dt)
        return None if k == 'u' else ('array', k)
    return None


# Arithmetic ops the IR supports
_ARITH_OPS = {'+', '-', '*', '%', '^'}

# Comparison ops
_CMP_OPS = {'>', '<', '='}

# Supported reduce/scan ops
_REDUCE_SCAN_OPS = {'+', '*', '|', '&'}


def compile_expr(ast, klong):
    """Try to compile an AST node to a callable.

    Returns (compiled_fn, [var_syms]) or None.
    """
    try:
        var_refs = {}
        ir = _ast_to_ir(ast, klong, var_refs)
        if ir is None:
            return None
        if not var_refs:
            return None  # pure constant — not worth compiling

        var_syms = list(var_refs.keys())
        return klong._backend.compile_expr_ir(ir, var_syms)
    except _UNSUPPORTED:
        _log.debug("expression not compilable", exc_info=True)
        return None


def compile_guarded(ast, klong):
    """Like compile_expr, but returns a CompiledExpr that remembers the kind
    (int, float, array dtype kind) of every variable it was compiled for, or None.
    """
    r = compile_expr(ast, klong)
    if r is None:
        return None
    fn, var_syms = r
    try:
        kinds = [_kind(klong._context[s], klong) for s in var_syms]
    except KeyError:
        return None
    if None in kinds:
        return None
    return CompiledExpr(fn, var_syms, kinds)


def run_compiled(klong, c):
    """Run a CompiledExpr on the current values of its variables.

    Returns NOT_COMPILED, without running anything, if a variable has been
    rebound to a different kind of value since compilation (or removed): the
    compiled function assumes its kinds, and Python operators mean different
    things on other types. Otherwise the result is returned and any exception
    the function raises propagates (except ZeroDivisionError, see below): the function is the whole expression, so
    re-running it on the interpreter would only repeat the work and the error.
    """
    ctx = klong._context
    args = []
    try:
        for s, kind in zip(c.var_syms, c.kinds):
            v = ctx[s]
            if type(v) is kind:
                if kind is int and not _INT64_MIN <= v <= _INT64_MAX:
                    return NOT_COMPILED  # an integer beyond 64 bits: the interpreter's exact arithmetic
            elif _kind(v, klong) != kind:
                return NOT_COMPILED
            args.append(v)
    except KeyError:
        return NOT_COMPILED
    try:
        return c.fn(*args)
    except ZeroDivisionError:
        # The one run-time fallback. Python raises this for scalar a%b with
        # b == 0 where the interpreter returns :undefined. The compiled
        # function is only literals, variable values already fetched above and
        # the arithmetic/reduce/scan operators of the IR: it never calls Klong
        # code and has no side effect, so the interpreter can re-evaluate it.
        return NOT_COMPILED


def _is_scalar(ir, var_refs, klong):
    """True if the IR certainly evaluates to a number rather than an array.

    Decided from the values the variables hold now; run_compiled's kind check
    keeps those kinds. False means "an array or not known".
    """
    t = ir[0]
    if t == 'literal':
        return True
    if t == 'var':
        for sym, name in var_refs.items():
            if name == ir[1]:
                return type(klong._context[sym]) in (int, float)
        return False
    if t == 'binop' or t == 'cmp':
        return _is_scalar(ir[2], var_refs, klong) and _is_scalar(ir[3], var_refs, klong)
    if t == 'negate':
        return _is_scalar(ir[1], var_refs, klong)
    return False


def _ast_to_ir(node, klong, var_refs):
    """Walk AST and emit IR tuples. Returns IR tree or None."""
    t = type(node)

    # Literals (an integer beyond 64 bits is left to the interpreter, see _kind)
    if t is int:
        return ('literal', node) if _INT64_MIN <= node <= _INT64_MAX else None
    if t is float:
        return ('literal', node)

    # Symbols -> variable references
    # Exact type on purpose: a quoted symbol (KGQSym) is not a plain variable
    # reference and is left to the interpreter (KlongInterpreter._eval_quoted_symbol).
    if t is KGSym:
        try:
            val = klong._context[node]
        except KeyError:
            return None
        tv = type(val)
        if tv is float or (tv is int and _INT64_MIN <= val <= _INT64_MAX):
            if node not in var_refs:
                var_refs[node] = f'_v{len(var_refs)}'
            return ('var', var_refs[node])
        if isinstance(val, klong._backend.np.ndarray):
            if node not in var_refs:
                var_refs[node] = f'_v{len(var_refs)}'
            return ('var', var_refs[node])
        return None

    # Operator expressions
    if isinstance(node, KGFn) and node.is_op():
        op_char = node.a.a
        arity = node.a.arity

        if arity == 2:
            args = node.args
            if type(args) is not list:
                args = [args] if args is not None else None
            if args is None or len(args) != 2:
                return None
            left = _ast_to_ir(args[0], klong, var_refs)
            right = _ast_to_ir(args[1], klong, var_refs)
            if left is None or right is None:
                return None
            if op_char in _ARITH_OPS:
                return ('binop', op_char, left, right)
            if op_char in _CMP_OPS:
                return ('cmp', op_char, left, right)
            return None

        if arity == 1 and op_char == '-':
            arg = node.args
            if type(arg) is list:      # a conditional or evaluated list is a list subclass: one operand
                arg = arg[0]
            child = _ast_to_ir(arg, klong, var_refs)
            if child is None:
                return None
            return ('negate', child)

    # Adverb chains: op/arg (reduce) and op\arg (scan)
    if isinstance(node, KGCall) and node.is_adverb_chain():
        chain = node.a
        if isinstance(chain, list) and len(chain) == 3:
            verb_adv = chain[0]
            adverb = chain[1]
            arg = chain[2]
            if (isinstance(verb_adv, KGAdverb) and isinstance(verb_adv.a, KGOp)
                    and isinstance(adverb, KGAdverb)):
                op_char = verb_adv.a.a
                adv_char = adverb.a
                if op_char not in _REDUCE_SCAN_OPS:
                    return None
                arg_ir = _ast_to_ir(arg, klong, var_refs)
                if arg_ir is None:
                    return None
                if adv_char == '/':
                    return ('reduce', op_char, arg_ir)
                elif adv_char == '\\':
                    if _is_scalar(arg_ir, var_refs, klong):
                        return None  # the interpreter returns a scalar for op\scalar, not a list
                    return ('scan', op_char, arg_ir)

    return None
