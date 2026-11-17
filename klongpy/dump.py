"""
Canonical dump of a KlongPy parse tree (conformance/README.md).

    dump(tree, compat=False, spans=False) -> str
    node_spans(tree) -> [(node, span, op_span, arg_spans), ...]
    parse_program(klong, text) -> list of statements (parse only, no evaluation)

    python -m klongpy.dump FILE      print the dump of a file, one form per line

The dump is defined on the legacy tree classes. It does not depend on object
identity or interpreter state. A top-level Python ``list`` is a program: one
line per statement. Any other object is dumped as one form.

Mapping from tree shape to dump form
------------------------------------

Expression mode (the object is something that will be evaluated):

    int, float (also numpy scalars)         (num 1) (num 1.5)   floats via repr()
    KGChar                                  (char "x")
    str (plain)                             (str "a\\"b")        JSON escapes
    KGQSym                                  (qsym a)            (sym a) if compat
    KGSym x, y, z (reserved)                (arg x)
    KGSym other                             (sym a) (sym a`mod)
    numpy array / data list                 (list D...)         elements in data mode
    KGExprArray                             (elist E...)
    KGExprDict                              (edict K1 V1 K2 V2 ...)    :{;k1;v1;k2;v2}
    KGCond [c, t, e]                        (cond C T E)        e may be a KGCond
    KGCall(copy_lambda, args=dict)          (dict (K V) ...)    insertion order
    KGCall(copy_lambda, args=array)         (list D...)         a list literal that holds a dictionary literal
    KGCall([KGAdverb v, KGAdverb a1, ..., X], args=None), X not a list
                                            (adv (verb V) (adverbs "a1" ...) X)
    same, X = [L, R] (a Python list)        (adv2 (verb V) (adverbs ...) L R)
        V is "+" for an operator verb, otherwise the verb's expression form
    KGFn(KGOp, args=E) (args not a list)    (m "-" E)
    KGFn(KGOp, args=None)                   (m "-" (missing))   monad without operand
    KGFn(KGOp, [L, None])                   (d "@" L (missing)) dyad without right operand
    KGFn(KGOp "::", [L, R])                 (def L R)
    KGFn(KGOp "::", [KGDestructure, R])     (destructure (a b c) R)    [a b c]::R
    KGFn(KGOp, [L, R])                      (d "+" L R)
    KGFn(F, [L, R]) no hole, F not a KGOp   (infix F L R)       F = KGSym or KGFn
    KGFn(body, None)                        (fn ARITY (locals a b) BODY...)
    same, defined as f(a;b)::{...}          (fn ARITY (params a b) (locals ...) BODY...)   
    KGCall(KGSym, [A...])                   (call (sym f) A...)
    KGCall(body, [A...])                    (call (fn ARITY ...) A...)
    KGFn(F, [A...]) with a None hole        (proj F A ... _ ...)
    KGOp (bare)                             (opval "+")
    None                                    _

Function bodies: KGFn.a / KGCall.a of a literal is the body. A body that is a
Python list is a statement sequence. The locals declaration is printed as
(locals ...) and is not part of BODY: the token parser stores it apart from
the statements (a KGBody, printed from its ``locals``, which also
hold the targets of a destructuring that heads the body); in a tree
of the legacy parser it is the first element of the list, a non-empty data
list holding only symbols (and ";"), found by the same test as at run time.
ARITY is KGFn.arity / KGCall.arity (for a function literal: the highest of x, y, z
its body mentions, see klongpy.types.get_fn_arity).

Data mode (inside lists and dict literals): KGOp -> (op "+"), the string ";"
-> (str ";"), every KGSym (reserved or not, KGQSym too) -> (sym a), nested
arrays -> (list ...), KGCall dict literal -> (dict ...).

Where the legacy tree cannot tell two forms apart (design list vs tree)
-----------------------------------------------------------------------
* ``{x}(1)`` and ``x(1)`` are the same tree (a call whose callee is the body
  ``x``). Dumped as ``(call (arg x) (num 1))`` in both cases; likewise ``{a}(1)``
  and ``a(1)``. ``(call (fn ...) ...)`` appears only when the body is not a
  bare name.
* Infix ``1 f 2`` versus a dyad: the tree has no flag. KGFn with a non-KGOp
  callee, two arguments and no hole is printed ``infix``. ``f(1;2)`` is a
  KGCall and prints as ``call``.
* Whether a monad/dyad was written with parentheses: parentheses are not in the
  tree.
* ``.comment("m")`` calls are removed by the parser and do not appear;
  ``.module(:m)`` does appear as an ordinary call.
* A dyad adverb chain with a missing-operand slot is not distinguishable
  from a normal one; (adv2) always has two operands.
* The design's node flags (``unterminated`` etc.) do not exist on the tree
  (the token parser builds the same tree as the legacy one), so ``compat``
  only affects qsym.
* Unrecognised objects print as ``(unknown ClassName)``.
"""
import json
import os
import sys

from klongpy.types import (KGAdverb, KGBody, KGCall, KGChar, KGCond, KGDestructure, KGFn, KGLambda, KGOp, KGQSym, KGSym,
                           reserved_fn_symbol_map)
from klongpy.parser import KGExprArray, KGExprDict


def _str(s):
    return json.dumps(str.__str__(s))


def _scalar(x):
    if hasattr(x, "item") and getattr(x, "ndim", None) == 0:
        return x.item()
    return x


def _num(x):
    if isinstance(x, bool):
        return f"(num {int(x)})"
    if isinstance(x, int):
        return f"(num {x})"
    return f"(num {repr(float(x))})"


def _is_dict_literal(x):
    return isinstance(x, KGCall) and isinstance(x.a, KGLambda) and isinstance(x.args, dict)


def _is_seq(x):
    return isinstance(x, list) and type(x) is list or (hasattr(x, "ndim") and x.ndim > 0)


def _dict(x, compat):
    pairs = " ".join(f"({_data(k, compat)} {_data(v, compat)})" for k, v in x.args.items())
    return f"(dict {pairs})" if pairs else "(dict)"


def _data(x, compat):
    x = _scalar(x)
    if x is None:
        return "_"
    if isinstance(x, KGOp):
        return f"(op {_str(x.a)})"
    if isinstance(x, KGSym):
        return f"(sym {str.__str__(x)})"
    if isinstance(x, KGChar):
        return f"(char {_str(x)})"
    if isinstance(x, str):
        return f"(str {_str(x)})"
    if isinstance(x, (int, float)):
        return _num(x)
    if _is_dict_literal(x):
        return _dict(x, compat)
    if isinstance(x, dict):
        return _dict(KGCall(None, x, 0), compat)
    if isinstance(x, (list, tuple)) or (hasattr(x, "ndim") and x.ndim > 0):
        inner = " ".join(_data(e, compat) for e in x)
        return f"(list {inner})" if inner else "(list)"
    return f"(unknown {type(x).__name__})"


def _locals(body):
    """Return (names, rest) if body starts with a locals declaration."""
    if type(body) is list and len(body) > 1:
        f0 = body[0]
        if _is_seq(f0) and len(f0) > 0:
            non_sep = [q for q in f0 if not (isinstance(q, str) and not isinstance(q, KGSym) and q == ";")]
            if non_sep and all(isinstance(q, KGSym) for q in non_sep):
                return [str.__str__(q) for q in non_sep], body[1:]
    return None, body


def _fn(body, arity, compat):
    params = None
    if type(body) is KGBody:
        # the token parser stores the declaration apart from the body
        loc, stmts = [str.__str__(q) for q in body.locals], list(body)
        params = body.params
    elif type(body) is list:
        loc, stmts = _locals(body)
    else:
        loc, stmts = None, [body]
    parts = [f"(fn {arity}"]
    if params is not None:
        # f(a;b)::{...}; "(params)" for a function of no parameters
        parts.append("(params" + "".join(" " + str.__str__(q) for q in params) + ")")
    if loc:
        parts.append("(locals " + " ".join(loc) + ")")
    parts.extend(_e(s, compat) for s in stmts)
    return " ".join(parts) + ")"


def _verb(v, compat):
    if isinstance(v, KGOp):
        return _str(v.a)
    return _e(v, compat)


def _adverb_chain(x, compat):
    arr = x.a
    verb = arr[0]
    advs = " ".join(_str(q.a) for q in arr[1:-1])
    operand = arr[-1]
    head = f"(verb {_verb(verb.a, compat)}) (adverbs {advs})"
    if type(operand) is list:
        return f"(adv2 {head} {_e(operand[0], compat)} {_e(operand[1], compat)})"
    return f"(adv {head} {_e(operand, compat)})"


def _callee(x, compat):
    """Callee of a KGFn/KGCall with a list of arguments."""
    a = x.a
    if isinstance(a, (KGSym, KGFn)):
        return _e(a, compat)
    return _fn(a, x.arity, compat)


def _operand(x, compat):
    """Operand of an operator application; None is a missing operand."""
    return "(missing)" if x is None else _e(x, compat)


_SPANS = False


def _e(x, compat):
    s = _e0(x, compat)
    if _SPANS:
        sp = getattr(x, "span", None)
        if sp is not None:
            s += f"@{sp[0]}:{sp[1]}"
    return s


def _e0(x, compat):
    x = _scalar(x)
    if x is None:
        return "_"
    if isinstance(x, KGQSym):
        return f"(sym {str.__str__(x)})" if compat else f"(qsym {str.__str__(x)})"
    if isinstance(x, KGSym):
        s = str.__str__(x)
        if reserved_fn_symbol_map.get(s) == x:
            return f"(arg {s})"
        return f"(sym {s})"
    if isinstance(x, KGChar):
        return f"(char {_str(x)})"
    if isinstance(x, str):
        return f"(str {_str(x)})"
    if isinstance(x, (int, float)):
        return _num(x)
    if isinstance(x, KGOp):
        return f"(opval {_str(x.a)})"
    if isinstance(x, KGCond):
        return "(cond " + " ".join(_e(q, compat) for q in x) + ")"
    if isinstance(x, KGExprDict):
        return "(edict" + "".join(" " + _e(q, compat) for q in x) + ")"
    if isinstance(x, KGExprArray):
        return "(elist" + "".join(" " + _e(q, compat) for q in x) + ")"
    if _is_dict_literal(x):
        return _dict(x, compat)
    if isinstance(x, KGCall) and isinstance(x.a, KGLambda) and hasattr(x.args, "ndim"):
        # a list literal that holds a dictionary literal (copied when evaluated, like a dictionary literal)
        return _data(x.args, compat)
    if isinstance(x, KGCall) and isinstance(x.a, list) and x.a and isinstance(x.a[0], KGAdverb):
        return _adverb_chain(x, compat)
    if isinstance(x, KGFn):
        if type(x.args) is not list:
            if isinstance(x.a, KGOp):
                return f"(m {_str(x.a.a)} {_operand(x.args, compat)})"
            if x.args is None:
                return _fn(x.a, x.arity, compat)
            return f"(unknown {type(x).__name__})"
        if isinstance(x.a, KGOp) and len(x.args) == 2 and type(x.args[0]) is KGDestructure:
            names = " ".join(str.__str__(q) for q in x.args[0])
            return f"(destructure ({names}) {_operand(x.args[1], compat)})"
        if isinstance(x.a, KGOp) and not isinstance(x, KGCall):
            args = " ".join(_operand(q, compat) for q in x.args)
        else:
            args = " ".join(_e(q, compat) for q in x.args)
        if isinstance(x, KGCall):
            return f"(call {_callee(x, compat)}{' ' if args else ''}{args})"
        if isinstance(x.a, KGOp):
            if len(x.args) == 1:
                return f"(m {_str(x.a.a)} {args})"
            if x.a.a == "::" and len(x.args) == 2:
                return f"(def {args})"
            return f"(d {_str(x.a.a)} {args})"
        if any(q is None for q in x.args):
            return f"(proj {_callee(x, compat)}{' ' if args else ''}{args})"
        if len(x.args) == 2 and isinstance(x.a, (KGSym, KGFn)):
            return f"(infix {_e(x.a, compat)} {args})"
        return f"(proj {_callee(x, compat)}{' ' if args else ''}{args})"
    if type(x) is list:
        # a statement sequence that is not a function body (should not occur)
        return "(seq" + "".join(" " + _e(q, compat) for q in x) + ")"
    if hasattr(x, "ndim") and x.ndim > 0:
        return _data(x, compat)
    return f"(unknown {type(x).__name__})"


def dump(tree, compat=False, spans=False):
    """
    Canonical text of a parse tree. A top-level list is a program (one line per statement).

    With spans=True every node that carries a source span (set by the token
    parser on KGFn, KGCall, KGCond, KGExprArray and dictionary literals) is
    followed by ``@start:end``. Leaves have no span of their own; their
    positions are in the enclosing node's arg_spans (see node_spans).
    """
    global _SPANS
    saved = _SPANS
    _SPANS = bool(spans)
    try:
        if type(tree) is list:
            return "".join(_e(s, compat) + "\n" for s in tree)
        return _e(tree, compat) + "\n"
    finally:
        _SPANS = saved


def node_spans(tree):
    """
    Every node of a parse tree that carries a span, depth first, as a list of
    (node, span, op_span, arg_spans). For tests and tools.
    """
    out = []

    def walk(x):
        if isinstance(x, (KGFn, KGCond, KGExprArray)):
            if getattr(x, "span", None) is not None:
                out.append((x, x.span, getattr(x, "op_span", None), getattr(x, "arg_spans", None)))
            if isinstance(x, KGFn):
                a = x.a
                if type(a) is list and a and isinstance(a[0], KGAdverb):
                    walk(a[0].a)
                    walk(a[-1])
                else:
                    walk(a)
                if not isinstance(x.args, dict):
                    walk(x.args)
            else:
                for q in x:
                    walk(q)
        elif type(x) is list or type(x) is KGBody:
            for q in x:
                walk(q)

    walk(tree)
    return out


def parse_program(klong, text):
    """
    Parse text without evaluating it and return the list of statements.

    Uses the interpreter's own ``prog`` (what ``__call__``/``exec`` run before
    evaluating). Parse-time effects: ``.comment`` is consumed by the parser, and
    ``.module`` switches the parser's active module while parsing; the module in
    force before the call is restored afterwards, because nothing is evaluated.
    """
    saved = klong._module
    try:
        _, prog = klong.prog(text)
    finally:
        klong._module = saved
    return prog


def dump_text(text, compat=False, parser=None, spans=False):
    """
    Parse text on a fresh interpreter and dump it; on a parse-time exception return ``(error ClassName)``.

    parser selects the front end ("token" or "legacy"); None is the interpreter's default.
    """
    from klongpy import KlongInterpreter
    try:
        return dump(parse_program(KlongInterpreter(parser=parser), text), compat, spans)
    except Exception as e:
        return f"(error {type(e).__name__})\n"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    compat = "--compat" in argv
    files = [a for a in argv if not a.startswith("--")]
    if len(files) != 1:
        print("usage: python -m klongpy.dump [--compat] FILE", file=sys.stderr)
        return 2
    with open(files[0], "r") as f:
        text = f.read()
    from klongpy import KlongInterpreter
    sys.stdout.write(dump(parse_program(KlongInterpreter(), text), compat))
    return 0


if __name__ == "__main__":
    sys.exit(main())
