"""
Expression compiler for KlongPy.

Compiles Klong expression ASTs to a backend-neutral IR (tuple tree),
then delegates to the active backend for platform-specific code generation.
Falls back to None (interpreter handles it) for anything it can't compile.
"""
from .types import KGSym, KGFn, KGCall, KGOp, KGAdverb, reserved_fn_symbols


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
    var_refs = {}
    ir = _ast_to_ir(ast, klong, var_refs)
    if ir is None:
        return None
    if not var_refs:
        return None  # pure constant — not worth compiling

    var_syms = list(var_refs.keys())
    return klong._backend.compile_expr_ir(ir, var_syms)


def _ast_to_ir(node, klong, var_refs):
    """Walk AST and emit IR tuples. Returns IR tree or None."""
    t = type(node)

    # Literals
    if t is int or t is float:
        return ('literal', node)

    # Symbols -> variable references
    if t is KGSym:
        try:
            val = klong._context[node]
        except KeyError:
            return None
        tv = type(val)
        if tv is int or tv is float:
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
            if not isinstance(args, list):
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
            if isinstance(arg, list):
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
                    return ('scan', op_char, arg_ir)

    return None
