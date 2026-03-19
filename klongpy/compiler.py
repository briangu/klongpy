"""
Torch expression compiler for KlongPy.

Compiles Klong expression ASTs to Python functions using torch-compatible
operators. Falls back to None (interpreter handles it) for anything it
can't compile.
"""
from .types import KGSym, KGFn, KGOp, reserved_fn_symbols


def compile_expr(ast, klong):
    """Try to compile an AST node to a callable.

    Returns (compiled_fn, [var_syms]) or None.
    """
    # Check for torch backend
    if not hasattr(klong._backend, '_torch_backend'):
        return None

    var_refs = {}
    source = _ast_to_source(ast, klong, var_refs)
    if source is None:
        return None
    if not var_refs:
        return None  # pure constant — not worth compiling

    var_syms = list(var_refs.keys())
    param_names = [var_refs[s] for s in var_syms]

    fn_source = f"def _expr({', '.join(param_names)}): return {source}"
    ns = {}
    try:
        exec(fn_source, ns)
    except SyntaxError:
        return None

    return (ns['_expr'], var_syms)


# Klong operator -> Python operator
_KLONG_TO_PY = {
    '+': '+', '-': '-', '*': '*',
    '%': '/',   # Klong % is division
    '^': '**',
}

# Comparison ops return bool in Python but numeric 0/1 in Klong
_KLONG_CMP_OPS = {'>', '<', '='}


def _ast_to_source(node, klong, var_refs):
    """Walk AST and emit Python source. Returns source string or None."""
    t = type(node)

    # Literals
    if t is int or t is float:
        return repr(node)

    # Symbols -> variable references
    if t is KGSym:
        if node in reserved_fn_symbols:
            return None
        try:
            val = klong._context[node]
        except KeyError:
            return None
        tv = type(val)
        if tv is int or tv is float:
            if node not in var_refs:
                var_refs[node] = f'_v{len(var_refs)}'
            return var_refs[node]
        # Only accept backend-native arrays (torch.Tensor when on torch backend).
        # Uses backend.np.ndarray which is torch.Tensor for torch, numpy.ndarray
        # for numpy. This avoids importing torch directly.
        if isinstance(val, klong._backend.np.ndarray):
            if node not in var_refs:
                var_refs[node] = f'_v{len(var_refs)}'
            return var_refs[node]
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
            left = _ast_to_source(args[0], klong, var_refs)
            right = _ast_to_source(args[1], klong, var_refs)
            if left is None or right is None:
                return None
            py_op = _KLONG_TO_PY.get(op_char)
            if py_op is not None:
                return f'({left}{py_op}{right})'
            # Comparison ops: wrap with *1 to convert bool -> numeric 0/1
            if op_char in _KLONG_CMP_OPS:
                py_cmp = {'=': '==', '>': '>', '<': '<'}[op_char]
                return f'(({left}{py_cmp}{right})*1)'
            return None

        if arity == 1 and op_char == '-':
            arg = node.args
            if isinstance(arg, list):
                arg = arg[0]
            s = _ast_to_source(arg, klong, var_refs)
            if s is None:
                return None
            return f'(-{s})'

    return None
