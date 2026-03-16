"""
Expression compiler for KlongPy.

Converts Klong expression trees to compiled Python source for vectorized evaluation.
Includes chain_adverbs which specializes adverb closures for performance.
"""
import itertools
import numpy
import operator as _op

from .adverbs import get_adverb_fn
from .core import KGSym, KGFn, KGCall, KGOp, KGAdverb, KGCond, KGLambda, reserved_fn_symbols, reserved_fn_symbols_set
from .cffi_accel import (
    _EVAL_GLOBALS, _python_expr_to_c,
    _cumsum, _cumprod, _running_max, _running_min,
    _prefix_scan_linear, _compile_cffi_fused, _compile_cffi_reduce,
    _compile_cffi_fused_running, _parallel_eval_2,
    _is_elementwise_source,
)

# Pre-resolve individual reserved symbols
_sym_x = reserved_fn_symbols[0]
_sym_y = reserved_fn_symbols[1]

# Safe for numpy arrays (numpy handles div-by-zero via inf/nan)
_FAST_DYAD_OPS = {'+': _op.add, '*': _op.mul, '-': _op.sub, '%': _op.truediv, '^': _op.pow}

# Klong op to Python op mapping
_KLONG_OP_TO_PY = {'+': '+', '-': '-', '*': '*', '%': '/', '>': '>', '<': '<', '=': '==', '^': '**'}

# Adverb reduction/scan ops that can be compiled to numpy source
_KLONG_REDUCE_TO_PY = {'+': '_np.sum', '*': '_np.prod', '|': '_np.max', '&': '_np.min'}
_KLONG_SCAN_TO_PY = {'+': '_cumsum', '*': '_cumprod', '|': '_running_max', '&': '_running_min'}

# Axis-based reduce/scan functions for stacked 2D arrays
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
                            if s1[0] == f'_argsort({s0[0]})':
                                return (f'_fast_sort({s0[0]})', False)
                            _s1str = s1[0]
                            if _s1str.startswith(f'_fused_where({s0[0]},'):
                                return (f'_fused_filter({_s1str[13:]}', False)
                            return (f'{s0[0]}[{s1[0]}]', False)
                # Special handling for # dyad (take)
                if var_refs is not None and expr._op_a == '#':
                    fa = expr.args
                    if type(fa) is not list:
                        fa = [fa] if fa is not None else fa
                    if fa is not None and len(fa) == 2:
                        s0 = _expr_to_source(fa[0], klong, dyadic=dyadic, var_refs=var_refs)
                        s1 = _expr_to_source(fa[1], klong, dyadic=dyadic, var_refs=var_refs)
                        if s0 is not None and s1 is not None and s0[1]:
                            return (f'{s1[0]}[:{s0[0]}]', False)
                # |/& (max/min) dyads
                if expr._op_a == '|' or expr._op_a == '&':
                    _np_fn = '_np.maximum' if expr._op_a == '|' else '_np.minimum'
                    fa = expr.args
                    if type(fa) is not list:
                        fa = [fa] if fa is not None else fa
                    if fa is not None and len(fa) == 2:
                        s0 = _expr_to_source(fa[0], klong, dyadic=dyadic, var_refs=var_refs)
                        s1 = _expr_to_source(fa[1], klong, dyadic=dyadic, var_refs=var_refs)
                        if s0 is not None and s1 is not None:
                            is_const = s0[1] and s1[1]
                            return (f'{_np_fn}({s0[0]},{s1[0]})', is_const)
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
                    try:
                        val = eval(f'({s0[0]}{py_op}{s1[0]})')
                        return (repr(val), True)
                    except Exception:
                        pass
                if s0[1] and '(' in s0[0]:
                    try: s0 = (repr(eval(s0[0])), True)
                    except Exception: pass
                if s1[1] and '(' in s1[0]:
                    try: s1 = (repr(eval(s1[0])), True)
                    except Exception: pass
                return (f'({s0[0]}{py_op}{s1[0]})', is_const)
        if expr._op_arity == 1:
            op_a = expr._op_a
            if op_a == '-' or (var_refs is not None and op_a in ('&', '#', '<', '?')):
                fa = expr.args
                arg = fa[0] if type(fa) is list else fa
                if op_a == '<':
                    ta = type(arg)
                    if (ta is KGCall or ta is KGFn) and arg._is_op and arg._op_arity == 1 and arg._op_a == '<':
                        inner_arg = arg.args
                        inner = inner_arg[0] if type(inner_arg) is list else inner_arg
                        s = _expr_to_source(inner, klong, dyadic=dyadic, var_refs=var_refs)
                        if s is not None:
                            return (f'_rank({s[0]})', False)
                elif op_a == '#':
                    ta = type(arg)
                    if (ta is KGCall or ta is KGFn) and arg._is_op and arg._op_arity == 1 and arg._op_a == '&':
                        inner_arg = arg.args
                        inner = inner_arg[0] if type(inner_arg) is list else inner_arg
                        ti = type(inner)
                        if (ti is KGCall or ti is KGFn) and inner._is_op and inner._op_arity == 2:
                            cmp_op = inner._op_a
                            if cmp_op in ('<', '>', '='):
                                cmp_args = inner.args
                                if type(cmp_args) is list and len(cmp_args) == 2:
                                    s0 = _expr_to_source(cmp_args[0], klong, dyadic=dyadic, var_refs=var_refs)
                                    s1 = _expr_to_source(cmp_args[1], klong, dyadic=dyadic, var_refs=var_refs)
                                    if s0 is not None and s1 is not None:
                                        py_cmp = {'<': '<', '>': '>', '=': '=='}[cmp_op]
                                        return (f'_fused_count({s0[0]},{py_cmp!r},{s1[0]})', False)
                        s = _expr_to_source(inner, klong, dyadic=dyadic, var_refs=var_refs)
                        if s is not None:
                            return (f'_np.count_nonzero({s[0]})', False)
                if op_a == '&':
                    ta = type(arg)
                    if (ta is KGCall or ta is KGFn) and arg._is_op and arg._op_arity == 2:
                        cmp_op = arg._op_a
                        if cmp_op in ('<', '>', '='):
                            cmp_args = arg.args
                            if type(cmp_args) is list and len(cmp_args) == 2:
                                s0 = _expr_to_source(cmp_args[0], klong, dyadic=dyadic, var_refs=var_refs)
                                s1 = _expr_to_source(cmp_args[1], klong, dyadic=dyadic, var_refs=var_refs)
                                if s0 is not None and s1 is not None:
                                    py_cmp = {'<': '<', '>': '>', '=': '=='}[cmp_op]
                                    return (f'_fused_where({s0[0]},{py_cmp!r},{s1[0]})', False)
                s = _expr_to_source(arg, klong, dyadic=dyadic, var_refs=var_refs)
                if s is not None:
                    if op_a == '&':
                        return (f'_flatnonzero({s[0]})', False)
                    elif op_a == '#':
                        return (f'len({s[0]})', False)
                    elif op_a == '<':
                        return (f'_argsort({s[0]})', False)
                    elif op_a == '?':
                        return (f'_unique({s[0]})', False)
                    else:
                        return (f'(-{s[0]})', False)
    # Handle simple adverb chains
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
                    py_op = _KLONG_OP_TO_PY.get(op_char)
                    if py_op is not None:
                        s = _expr_to_source(arg, klong, dyadic=dyadic, var_refs=var_refs)
                        if s is not None:
                            return (f'({s[0]}[:-1]{py_op}{s[0]}[1:])', False)
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
                        if adv_char == '/' and not s[1] and var_refs is not None:
                            inner = s[0]
                            has_v0 = '_v0' in inner
                            has_v1 = '_v1' in inner
                            if has_v0 and not has_v1:
                                try:
                                    _python_expr_to_c(inner, 1)
                                    return (f"_cffi_reduce_1({op_char!r},{inner!r},_v0)", False)
                                except (ValueError, SyntaxError):
                                    pass
                            elif not has_v0 and has_v1:
                                remapped = inner.replace('_v1', '_v0')
                                try:
                                    _python_expr_to_c(remapped, 1)
                                    return (f"_cffi_reduce_1({op_char!r},{remapped!r},_v1)", False)
                                except (ValueError, SyntaxError):
                                    pass
                            elif has_v0 and has_v1:
                                try:
                                    _python_expr_to_c(inner, 2)
                                    return (f"_cffi_reduce_2({op_char!r},{inner!r},_v0,_v1)", False)
                                except (ValueError, SyntaxError):
                                    pass
                        return (f'{py_fn}({s[0]})', s[1])
    return None


def _compile_arg_fn(expr, klong, dyadic=False):
    """Try to compile a sub-expression to a fast Python function of x (or x,y if dyadic)."""
    te = type(expr)
    if te is KGSym:
        if expr is _sym_x:
            fn = (lambda x, y: x) if dyadic else (lambda x: x)
            fn._vectorizable = True
            fn._is_const = False
            fn._axis_fn = (lambda X, Y: X) if dyadic else (lambda a: a)
            if dyadic:
                fn._is_x = True
            return fn
        if dyadic and expr is _sym_y:
            fn = lambda x, y: y
            fn._vectorizable = True
            fn._is_const = False
            fn._axis_fn = lambda X, Y: Y
            fn._is_y = True
            return fn
        try:
            val = klong._context[expr]
            tv = type(val)
            if tv is int or tv is float or tv is numpy.ndarray:
                fn = (lambda x, y, c=val: c) if dyadic else (lambda x, c=val: c)
                fn._vectorizable = tv is not numpy.ndarray
                fn._is_const = True
                fn._axis_fn = (lambda X, Y, c=val: c) if dyadic else (lambda a, c=val: c)
                return fn
        except KeyError:
            pass
        return None
    if te is int or te is float:
        fn = (lambda x, y, c=expr: c) if dyadic else (lambda x, c=expr: c)
        fn._vectorizable = True
        fn._is_const = True
        fn._axis_fn = (lambda X, Y, c=expr: c) if dyadic else (lambda a, c=expr: c)
        return fn
    if (te is KGCall or te is KGFn) and expr._is_op and expr._op_arity == 2:
        op_a = expr._op_a
        _fast = _FAST_DYAD_OPS.get(op_a)
        _op_fn = _fast if _fast is not None else klong._vd[op_a]
        fa = expr.args
        if type(fa) is not list:
            fa = [fa] if fa is not None else fa
        if fa is None or len(fa) != 2:
            return None
        a0, a1 = fa[0], fa[1]
        c0 = _compile_arg_fn(a0, klong, dyadic=dyadic)
        c1 = _compile_arg_fn(a1, klong, dyadic=dyadic)
        if c0 is not None and c1 is not None:
            if c0._is_const and c1._is_const:
                try:
                    _cv0 = c0(0, 0) if dyadic else c0(0)
                    _cv1 = c1(0, 0) if dyadic else c1(0)
                    _const = _op_fn(_cv0, _cv1)
                    fn = (lambda x, y, c=_const: c) if dyadic else (lambda x, c=_const: c)
                    fn._vectorizable = True
                    fn._is_const = True
                    fn._axis_fn = None if dyadic else (lambda a, c=_const: c)
                    return fn
                except Exception:
                    pass
            if dyadic:
                fn = lambda x, y, op=_op_fn, g0=c0, g1=c1: op(g0(x, y), g1(x, y))
            else:
                fn = lambda x, op=_op_fn, g0=c0, g1=c1: op(g0(x), g1(x))
            fn._vectorizable = _fast is not None and c0._vectorizable and c1._vectorizable
            fn._is_const = False
            _af0 = getattr(c0, '_axis_fn', None)
            _af1 = getattr(c1, '_axis_fn', None)
            if _fast is not None and _af0 is not None and _af1 is not None:
                if dyadic:
                    fn._axis_fn = lambda X, Y, op=_fast, g0=_af0, g1=_af1: op(g0(X, Y), g1(X, Y))
                else:
                    fn._axis_fn = lambda a, op=_fast, g0=_af0, g1=_af1: op(g0(a), g1(a))
                if getattr(_af0, '_uses_scan', False) or getattr(_af1, '_uses_scan', False):
                    fn._axis_fn._uses_scan = True
            else:
                fn._axis_fn = None
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
            if dyadic and op_a == '*':
                _x0 = getattr(c0, '_is_x', False)
                _x1 = getattr(c1, '_is_x', False)
                _y0 = getattr(c0, '_is_y', False)
                _y1 = getattr(c1, '_is_y', False)
                if _x0 and c1._is_const:
                    try: fn._x_coeff = c1(0, 0)
                    except: pass
                elif _x1 and c0._is_const:
                    try: fn._x_coeff = c0(0, 0)
                    except: pass
                elif _y0 and c1._is_const:
                    try: fn._y_coeff = c1(0, 0)
                    except: pass
                elif _y1 and c0._is_const:
                    try: fn._y_coeff = c0(0, 0)
                    except: pass
            if dyadic and op_a == '+':
                _xc0 = getattr(c0, '_x_coeff', 1.0 if getattr(c0, '_is_x', False) else None)
                _xc1 = getattr(c1, '_x_coeff', 1.0 if getattr(c1, '_is_x', False) else None)
                _yc0 = getattr(c0, '_y_coeff', 1.0 if getattr(c0, '_is_y', False) else None)
                _yc1 = getattr(c1, '_y_coeff', 1.0 if getattr(c1, '_is_y', False) else None)
                if _xc0 is not None and _yc1 is not None:
                    fn._linear_recurrence = (float(_xc0), float(_yc1))
                elif _xc1 is not None and _yc0 is not None:
                    fn._linear_recurrence = (float(_xc1), float(_yc0))
            return fn
    if (te is KGCall or te is KGFn) and expr._is_op and expr._op_arity == 1:
        op_a = expr._op_a
        _op_fn = klong._vm.get(op_a)
        if _op_fn is not None:
            fa = expr.args
            _fa = fa if type(fa) is not list else fa[0]
            c = _compile_arg_fn(_fa, klong, dyadic=dyadic)
            if c is not None:
                if dyadic:
                    fn = lambda x, y, op=_op_fn, g=c: op(g(x, y))
                else:
                    fn = lambda x, op=_op_fn, g=c: op(g(x))
                fn._vectorizable = False
                fn._is_const = c._is_const
                _c_axis = getattr(c, '_axis_fn', None)
                if not dyadic and op_a == '#' and _c_axis is not None:
                    fn._axis_fn = lambda a, g=_c_axis: numpy.asarray(g(a)).shape[-1]
                else:
                    fn._axis_fn = None
                return fn
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
                        _resolved = eval(np_fn, _EVAL_GLOBALS)
                        if dyadic:
                            fn = lambda x, y, rf=_resolved, g=c_arg: rf(g(x, y))
                        else:
                            fn = lambda x, rf=_resolved, g=c_arg: rf(g(x))
                        fn._vectorizable = False
                        fn._is_const = c_arg._is_const
                        _c_axis = getattr(c_arg, '_axis_fn', None)
                        if _axis_fn_2d is not None and _c_axis is not None:
                            if dyadic:
                                fn._axis_fn = lambda X, Y, af=_axis_fn_2d, g=_c_axis: af(g(X, Y))
                            else:
                                _cmv = getattr(c_arg, '_const_mul_val', None)
                                if op_char == '+' and _cmv is not None:
                                    fn._axis_fn = lambda a, c=_cmv: a @ c
                                else:
                                    fn._axis_fn = lambda a, af=_axis_fn_2d, g=_c_axis: af(g(a))
                            if adv_char == '\\':
                                fn._axis_fn._uses_scan = True
                        else:
                            fn._axis_fn = None
                        return fn
                    if adv_char == ":'" and not dyadic:
                        _fast = _FAST_DYAD_OPS.get(op_char)
                        _c_axis = getattr(c_arg, '_axis_fn', None)
                        if _fast is not None:
                            if c_arg._is_const:
                                return None
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
    """Build adverb chain closure with specialization for performance."""
    _specialized = False
    _vectorizable = False
    _resolved_op = None
    if arr[0].arity == 1:
        if type(arr[0].a) is KGOp:
            _fn = klong._vm[arr[0].a.a]
            f = lambda x,fn=_fn: fn(x)
        else:
            _specialized = False
            verb = arr[0].a
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
            if tb is KGFn and not body._is_op and not body._is_adverb_chain and body.args is None:
                body = body.a
                tb = type(body)
            if (tb is KGCall or tb is KGFn) and body._is_op:
                op_a = body._op_a
                if body._op_arity == 2:
                    _fast_op = _FAST_DYAD_OPS.get(op_a)
                    _op_fn = _fast_op if _fast_op is not None else klong._vd[op_a]
                    fa = body.args
                    if type(fa) is not list:
                        fa = [fa] if fa is not None else fa
                    fa0, fa1 = fa[0], fa[1]
                    t0, t1 = type(fa0), type(fa1)
                    if t0 is KGSym and fa0 is _sym_x and (t1 is int or t1 is float):
                        _c = fa1
                        f = lambda x, fn=_op_fn, c=_c: fn(x, c)
                        _specialized = True
                        _vectorizable = _fast_op is not None
                    elif t1 is KGSym and fa1 is _sym_x and (t0 is int or t0 is float):
                        _c = fa0
                        f = lambda x, fn=_op_fn, c=_c: fn(c, x)
                        _specialized = True
                        _vectorizable = _fast_op is not None
                    elif t0 is KGSym and fa0 is _sym_x and t1 is KGSym and fa1 is _sym_x:
                        f = lambda x, fn=_op_fn: fn(x, x)
                        _specialized = True
                        _vectorizable = _fast_op is not None
                    else:
                        _src = _expr_to_source(body, klong, dyadic=False)
                        if _src is not None:
                            src_str, is_const = _src
                            if not is_const:
                                try:
                                    _code = compile(f'lambda x:{src_str}', '<klong>', 'eval')
                                    f = eval(_code, _EVAL_GLOBALS)
                                    _specialized = True
                                    _vectorizable = _fast_op is not None
                                    _body_cf = _compile_arg_fn(body, klong, dyadic=False)
                                    if _body_cf is not None and getattr(_body_cf, '_axis_fn', None) is not None:
                                        f._axis_fn = _body_cf._axis_fn
                                except Exception:
                                    pass
                        if not _specialized:
                            _cf0 = _compile_arg_fn(fa0, klong)
                            _cf1 = _compile_arg_fn(fa1, klong)
                            if _cf0 is not None and _cf1 is not None:
                                f = lambda x, fn=_op_fn, g0=_cf0, g1=_cf1: fn(g0(x), g1(x))
                                _specialized = True
                                _vectorizable = _fast_op is not None and _cf0._vectorizable and _cf1._vectorizable
                                _af0 = getattr(_cf0, '_axis_fn', None)
                                _af1 = getattr(_cf1, '_axis_fn', None)
                                if _fast_op is not None and _af0 is not None and _af1 is not None:
                                    f._axis_fn = lambda a, op=_fast_op, g0=_af0, g1=_af1: op(g0(a), g1(a))
                                    if getattr(_af0, '_uses_scan', False) or getattr(_af1, '_uses_scan', False):
                                        f._axis_fn._uses_scan = True
                elif body._op_arity == 1:
                    fa = body.args
                    _fa = fa if type(fa) is not list else fa[0]
                    if type(_fa) is KGSym and _fa is _sym_x:
                        _op_fn = klong._vm[op_a]
                        f = lambda x, fn=_op_fn: fn(x)
                        _specialized = True
            if not _specialized:
                _body_cf = _compile_arg_fn(body, klong, dyadic=False)
                if _body_cf is not None:
                    f = _body_cf
                    _specialized = True
            if not _specialized:
                _call = KGCall(verb, [None], arity=1)
                _args = _call.args
                def f(x, k=klong, c=_call, a=_args):
                    a[0] = x
                    return k._eval_fn(c)
    else:
        if type(arr[0].a) is KGOp:
            _fn = klong._vd[arr[0].a.a]
            f = lambda x,y,fn=_fn: fn(x,y)
            _vectorizable = True
        else:
            _dyad_verb = arr[0].a
            _resolved_op = None
            _dtb = type(_dyad_verb)
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
                        if type(_da0) is KGSym and _da0 is _sym_x and type(_da1) is KGSym and _da1 is _sym_y:
                            f = lambda x,y,fn=_dop: fn(x,y)
                            _resolved_op = KGOp(_dop_a, arity=2)
                            _dyad_verb = None
                        elif type(_da0) is KGSym and _da0 is _sym_y and type(_da1) is KGSym and _da1 is _sym_x:
                            f = lambda x,y,fn=_dop: fn(y,x)
                            _dyad_verb = None
            if _dyad_verb is not None:
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
                    if getattr(f, '_axis_fn', None) is None or getattr(f, '_linear_recurrence', None) is None:
                        _body_cf = _compile_arg_fn(_dyad_verb.a, klong, dyadic=True)
                        if _body_cf is not None:
                            if getattr(f, '_axis_fn', None) is None and getattr(_body_cf, '_axis_fn', None) is not None:
                                f._axis_fn = _body_cf._axis_fn
                            _lr = getattr(_body_cf, '_linear_recurrence', None)
                            if _lr is not None:
                                f._linear_recurrence = _lr
                    if _dtb is KGFn:
                        _vbody = _dyad_verb.a
                        _vbt = type(_vbody)
                        if (_vbt is KGCall or _vbt is KGFn) and _vbody._is_op and _vbody._op_arity == 2:
                            _vop = _vbody._op_a
                            if _vop == '+' or _vop == '*':
                                _vfa = _vbody.args
                                if type(_vfa) is list and len(_vfa) == 2:
                                    _y_dep = None
                                    if type(_vfa[0]) is KGSym and _vfa[0] is _sym_x:
                                        _y_dep = _vfa[1]
                                    elif type(_vfa[1]) is KGSym and _vfa[1] is _sym_x:
                                        _y_dep = _vfa[0]
                                    if _y_dep is not None:
                                        _gy_src = _expr_to_source(_y_dep, klong, dyadic=True)
                                        if _gy_src is not None and 'x' not in _gy_src[0]:
                                            try:
                                                _gy_fn = eval(compile(f'lambda y:{_gy_src[0]}', '<klong>', 'eval'), _EVAL_GLOBALS)
                                                if _vop == '+':
                                                    f._scan_cumsum_fn = _gy_fn
                                                else:
                                                    f._scan_cumprod_fn = _gy_fn
                                            except Exception:
                                                pass
                    _dyad_verb = None
                else:
                    _call = KGCall(arr[0].a, [None, None], arity=2)
                    _args = _call.args
                    def f(x, y, k=klong, c=_call, a=_args):
                        a[0] = x
                        a[1] = y
                        return k._eval_fn(c)
    # Axis-based reduction dispatch
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
        if arr[i].a == "'" and _vectorizable and arr[i].arity == 1:
            _prev_f = f
            _be = klong._backend
            _verb_op = arr[0].a.a if type(arr[0].a) is KGOp else None
            _prev_adv = arr[i-1].a if i > 1 else None
            _axis_fn = None
            if _verb_op is not None:
                if _prev_adv == '/':
                    _axis_fn = _AXIS_REDUCE.get(_verb_op)
                elif _prev_adv == '\\':
                    _axis_fn = _AXIS_SCAN.get(_verb_op)
            if _axis_fn is None:
                _axis_fn = getattr(f, '_axis_fn', None)
            _af_uses_scan = getattr(_axis_fn, '_uses_scan', False) if _axis_fn is not None else False
            def f(x, f=_prev_f, be=_be, axis_fn=_axis_fn, uses_scan=_af_uses_scan):
                tx = type(x)
                if tx is numpy.ndarray and x.ndim > 0:
                    return f(x)
                if tx is list:
                    if uses_scan and len(x) > 0 and type(x[0]) is numpy.ndarray and len(x[0]) >= 300:
                        return be.kg_asarray([f(e) for e in x])
                    if axis_fn is not None and len(x) > 0 and type(x[0]) is numpy.ndarray:
                        try:
                            stacked = numpy.concatenate(x).reshape(len(x), -1)
                            result = axis_fn(stacked)
                            if type(result) is numpy.ndarray:
                                if result.ndim == 1:
                                    return result
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
            _vectorizable = False
            continue
        if arr[i].a == "'" and arr[i].arity == 1:
            _af = getattr(f, '_axis_fn', None)
            if _af is not None:
                _prev_f = f
                _be = klong._backend
                _af_uses_scan = getattr(_af, '_uses_scan', False)
                def f(x, af=_af, pf=_prev_f, be=_be, uses_scan=_af_uses_scan):
                    if type(x) is list and len(x) > 0 and type(x[0]) is numpy.ndarray:
                        if uses_scan and len(x[0]) >= 300:
                            return be.kg_asarray([pf(e) for e in x])
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
                    if hasattr(x, '__iter__') and not isinstance(x, str):
                        return be.kg_asarray([pf(e) for e in x])
                    return pf(x)
                continue
        if arr[i].a == '/' and arr[i].arity == 1:
            _csf = getattr(f, '_scan_cumsum_fn', None)
            _cpf = getattr(f, '_scan_cumprod_fn', None)
            if _csf is not None:
                def f(x, csf=_csf):
                    if type(x) is numpy.ndarray and len(x) > 0:
                        if len(x) == 1:
                            return x[0]
                        return x[0] + numpy.sum(csf(x[1:]))
                    from functools import reduce
                    return reduce(lambda a, b, _csf=csf: a + _csf(b), x)
                continue
            if _cpf is not None:
                def f(x, cpf=_cpf):
                    if type(x) is numpy.ndarray and len(x) > 0:
                        if len(x) == 1:
                            return x[0]
                        return x[0] * numpy.prod(cpf(x[1:]))
                    from functools import reduce
                    return reduce(lambda a, b, _cpf=cpf: a * _cpf(b), x)
                continue
        if arr[i].a == '\\' and arr[i].arity == 1:
            _lr = getattr(f, '_linear_recurrence', None)
            if _lr is not None:
                _c_x, _c_y = _lr
                def f(x, cx=_c_x, cy=_c_y):
                    if type(x) is numpy.ndarray:
                        return _prefix_scan_linear(x, cx, cy)
                    return numpy.fromiter(itertools.accumulate(
                        x.tolist() if type(x) is numpy.ndarray else x,
                        lambda a, b, _cx=cx, _cy=cy: _cx * a + _cy * b
                    ), dtype=numpy.float64, count=len(x))
                continue
            _csf = getattr(f, '_scan_cumsum_fn', None)
            _cpf = getattr(f, '_scan_cumprod_fn', None)
            if _csf is not None:
                _be = klong._backend
                def f(x, csf=_csf, be=_be):
                    if type(x) is numpy.ndarray and len(x) > 0:
                        n = len(x)
                        if n == 1:
                            return x.copy()
                        g = csf(x[1:])
                        cs = numpy.cumsum(g)
                        cs += x[0]
                        result = numpy.empty(n, dtype=cs.dtype)
                        result[0] = x[0]
                        result[1:] = cs
                        return result
                    return be.kg_asarray(list(itertools.accumulate(x, lambda a, b, _csf=csf: a + _csf(b))))
                continue
            if _cpf is not None:
                _be = klong._backend
                def f(x, cpf=_cpf, be=_be):
                    if type(x) is numpy.ndarray and len(x) > 0:
                        n = len(x)
                        if n == 1:
                            return x.copy()
                        g = cpf(x[1:])
                        cp = numpy.cumprod(g)
                        cp *= x[0]
                        result = numpy.empty(n, dtype=cp.dtype)
                        result[0] = x[0]
                        result[1:] = cp
                        return result
                    return be.kg_asarray(list(itertools.accumulate(x, lambda a, b, _cpf=cpf: a * _cpf(b))))
                continue
        o = get_adverb_fn(klong, arr[i].a, arity=arr[i].arity)
        if arr[i].arity == 1:
            _scan_op = _resolved_op if _resolved_op is not None else arr[0].a
            f = lambda x,f=f,o=o,_op=_scan_op: o(f,x,op=_op)
        else:
            if arr[i].a == "'":
                _daf = getattr(f, '_axis_fn', None)
                if _daf is not None:
                    _prev_f = f
                    _be = klong._backend
                    _each2_op = _resolved_op if _resolved_op is not None else arr[0].a
                    def f(x, y, daf=_daf, pf=_prev_f, be=_be, o=o, _op=_each2_op):
                        if type(x) is list and type(y) is list and len(x) > 0 and type(x[0]) is numpy.ndarray:
                            try:
                                X = numpy.concatenate(x).reshape(len(x), -1)
                                Y = numpy.concatenate(y).reshape(len(y), -1)
                                result = daf(X, Y)
                                if type(result) is numpy.ndarray:
                                    if result.ndim == 1:
                                        return result
                                    if result.ndim == 2:
                                        return result.ravel() if result.shape[1] == 1 else list(result)
                                return result
                            except (ValueError, TypeError):
                                pass
                        return o(pf, x, y, op=_op)
                else:
                    _each2_op = _resolved_op if _resolved_op is not None else arr[0].a
                    f = lambda x,y,f=f,o=o,_op=_each2_op: o(f,x,y,op=_op)
            else:
                f = lambda x,y,f=f,o=o: o(f,x,y)
    if arr[-2].arity == 1:
        f = lambda a=arr[-1],f=f,k=klong: f(k.eval(a))
    else:
        f = lambda a=arr[-1],f=f,k=klong: f(k.eval(a[0]),k.eval(a[1]))
    return f
