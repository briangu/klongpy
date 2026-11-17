"""
The typed IR of the static compiler (docs/static-profile.md).

Every node has ``ty`` (a Ty: element type and shape) and ``span`` (offsets into
the source of the function it came from, or None). The tree of one FnSpec is
what lowering (klongpy/static/infer.py) produces for one function at one tuple
of argument types; the emitter (klongpy/static/emit_c.py) walks it.

    Program(name, globals, fns, entries, externs, int_bits, float_bits)
    GlobalDef(name, ty, value, state)
    FnSpec(name, index, kind, param_tys, locals, body, result_ty, ...)
    Entry(decl, fn, c_suffix)

Evaluation order is part of the IR's meaning (it is Klong's, observed on the
interpreter), and each node class documents the order of its operands in
``ORDER``: the fields named there are evaluated in that order. A dyad
evaluates its right operand first; a call, an evaluated list and the operands
of an adverb go left to right.

    dump(program) -> str      the stable text form used by the golden tests
    effects(node) -> bool     the node assigns, or calls an extern or a function that does
"""
from dataclasses import dataclass, field

import numpy as np

from .signature import Ty

__all__ = [
    "Node", "Const", "GlobalRef", "LocalRef", "ParamRef", "Unary", "Binary", "Convert", "Match", "Index", "Gather",
    "Take", "Drop", "Rotate", "Reverse", "Transpose", "Reshape", "Join", "Iota", "MakeList", "Amend", "Cond",
    "Call", "ExternCall", "Map", "Map2", "Reduce", "Scan", "EachPair", "Loop", "Assign", "Seq",
    "FnSpec", "GlobalDef", "Entry", "Program", "dump", "effects", "walk", "children",
]


@dataclass(frozen=True, eq=False)
class Node:
    ty: Ty
    span: object = field(default=None, compare=False)

    ORDER = ()          # names of the operand fields, in evaluation order

    @property
    def cval(self):
        """The compile-time value when lowering knows it (an int, a float or a numpy array), else None."""
        return None


def _node(order=()):
    def wrap(cls):
        cls = dataclass(frozen=True, eq=False)(cls)
        cls.ORDER = tuple(order)
        return cls
    return wrap


@_node()
class Const(Node):
    """A scalar constant, a constant table (a numpy array), or a string literal passed to an extern."""
    value: object = None

    @property
    def cval(self):
        return None if self.ty.elem == "str" else self.value


@_node()
class GlobalRef(Node):
    """A global, either state (assigned by some compiled function) or a constant table. ``known`` is its value when constant."""
    name: str = ""
    state: bool = False
    known: object = None

    @property
    def cval(self):
        return self.known


@_node()
class LocalRef(Node):
    slot: int = 0
    name: str = ""


@_node()
class ParamRef(Node):
    index: int = 0
    name: str = ""


@_node(["a"])
class Unary(Node):
    """Atomic, elementwise: op is a row id of the operator table (neg, floor, recip, not)."""
    op: str = ""
    a: Node = None
    known: object = None

    @property
    def cval(self):
        return self.known


@_node(["b", "a"])
class Binary(Node):
    """
    Atomic, elementwise: op is a row id (add, sub, mul, div, idiv, rem, pow, max, min, lt, gt, eq).
    The operands already have the element type the operation works in; of
    two arrays the shorter shape is a prefix of the longer.
    """
    op: str = ""
    a: Node = None
    b: Node = None


@_node(["a"])
class Convert(Node):
    """The conversion of a to the element type of ``ty``: int to float, int, float or fixed-point to fixed-point, fixed-point to float."""
    a: Node = None


@_node(["b", "a"])
class Match(Node):
    """a~b on operands of one shape: an int scalar."""
    a: Node = None
    b: Node = None


@_node(["i", "a"])
class Index(Node):
    """a@i with a scalar i: the item of a. ``checked``: i is a constant known to be in range (and not negative)."""
    a: Node = None
    i: Node = None
    checked: bool = False


@_node(["idx", "a"])
class Gather(Node):
    """a@idx with an int array: shape of idx followed by the item shape of a. A fusible leaf when a is materialised."""
    a: Node = None
    idx: Node = None
    checked: bool = False


@_node(["a"])
class Take(Node):
    """n#a, rank 1, constant n: |n| items, cycling; a negative n counts from the end."""
    a: Node = None
    n: int = 0


@_node(["a"])
class Drop(Node):
    a: Node = None
    n: int = 0


@_node(["a", "n"])
class Rotate(Node):
    """n:+a, rank 1: item j of the result is item (j-n) mod len of a."""
    a: Node = None
    n: Node = None


@_node(["a"])
class Reverse(Node):
    a: Node = None


@_node(["a"])
class Transpose(Node):
    a: Node = None


@_node(["a"])
class Reshape(Node):
    """The row-major data of a, cycled, in the shape of ``ty``."""
    a: Node = None


@_node(["b", "a"])
class Join(Node):
    """a,b along the first dimension; a scalar operand is one item."""
    a: Node = None
    b: Node = None


@_node()
class Iota(Node):
    """!n: 0 .. n-1."""
    n: int = 0

    @property
    def cval(self):
        return np.arange(self.n, dtype=np.int64) if self.n <= 4096 else None


@_node(["items"])
class MakeList(Node):
    """[;a;b;c]: items of one type and shape, evaluated left to right."""
    items: tuple = ()


@_node(["idx", "value", "a"])
class Amend(Node):
    """a:=value,idx on a rank-1 array: a scalar value stored at a scalar index or at each index of a rank-1 list."""
    a: Node = None
    value: Node = None
    idx: Node = None
    checked: bool = False


@_node(["c"])
class Cond(Node):
    """:[c;t;e]: only the branch taken is evaluated."""
    c: Node = None
    t: Node = None
    e: Node = None


@_node(["args"])
class Call(Node):
    fn: object = None       # FnSpec
    args: tuple = ()


@_node(["args"])
class ExternCall(Node):
    ext: object = None      # ExternDecl
    args: tuple = ()


@_node(["a", "captures"])
class Map(Node):
    """f'a. ``captures`` are the variables of the enclosing function the verb reads, passed after its own arguments."""
    fn: object = None
    a: Node = None
    captures: tuple = ()


@_node(["a", "b", "captures"])
class Map2(Node):
    fn: object = None
    a: Node = None
    b: Node = None
    captures: tuple = ()


@_node(["seed", "a", "captures"])
class Reduce(Node):
    """f/a and seed f/a: acc = f(acc, item) from the first item or the seed. ``ty`` is the accumulator's type."""
    fn: object = None
    a: Node = None
    seed: Node = None
    captures: tuple = ()


@_node(["seed", "a", "captures"])
class Scan(Node):
    """f\\a and seed f\\a: every value of the accumulator ([n] items, [n+1] with a seed)."""
    fn: object = None
    a: Node = None
    seed: Node = None
    captures: tuple = ()
    acc_ty: Ty = None


@_node(["a", "captures"])
class EachPair(Node):
    """f:'a: f(item j, item j+1)."""
    fn: object = None
    a: Node = None
    captures: tuple = ()


@_node(["count", "init", "captures", "pred_captures"])
class Loop(Node):
    """kind "iterate": count f:*init. kind "while": pred f:~init (pred is a FnSpec; count is None)."""
    kind: str = ""
    fn: object = None
    init: Node = None
    count: Node = None
    pred: object = None
    captures: tuple = ()
    pred_captures: tuple = ()


@_node(["value"])
class Assign(Node):
    """target::value. target is a LocalRef or a GlobalRef; the node's value is the value assigned."""
    target: Node = None
    value: Node = None


@_node(["stmts"])
class Seq(Node):
    """Statements in order; the value is that of the last."""
    stmts: tuple = ()

    @property
    def cval(self):
        return self.stmts[-1].cval if self.stmts else None


@dataclass(eq=False)
class FnSpec:
    """
    One function at one tuple of argument types.

        name        the Klong name; "f.fn1" for the first function literal written in f; the glyph for an operator
        index       unique in the program; the C name is kg_<name>__<index>
        kind        "fn" (a top-level function), "lambda" (a function literal that is the verb of an
                    adverb) or "op" (an operator used as a verb)
        param_tys   own parameters, then the captured variables
        n_own       how many of param_tys are the function's own parameters
        locals      Ty of each local slot; local_names names them (hidden slots start with "%")
        body        a Seq
    """
    name: str
    index: int
    kind: str
    param_tys: tuple
    n_own: int
    param_names: tuple
    locals: tuple
    local_names: tuple
    body: Node
    result_ty: Ty
    span: object = None
    source: object = None
    stmt_spans: tuple = ()      # the source span of each statement of body (None for generated ones)

    @property
    def label(self):
        return f"{self.name}#{self.index}"


@dataclass(eq=False)
class GlobalDef:
    """A global reached from an entry: ``state`` (assigned at run time, lives in the instance) or a constant."""
    name: str
    ty: Ty
    value: object
    state: bool
    param: bool = False     # declared with .param: an input register, set by the host between runs


@dataclass(eq=False)
class Entry:
    decl: object            # EntryDecl
    fn: FnSpec
    c_name: str             # without the program prefix: "step", or "run__f256" when the function has several entries


@dataclass(eq=False)
class Program:
    name: str
    globals: list
    fns: list               # callees before callers
    entries: list
    externs: list           # ExternDecl, those reached
    int_bits: int = 32
    source: object = None


# ---------------------------------------------------------------------------
# Walking
# ---------------------------------------------------------------------------

_FN_FIELDS = ("fn", "pred")


def children(n):
    """The operand nodes of n in evaluation order."""
    out = []
    for name in n.ORDER:
        v = getattr(n, name)
        if v is None:
            continue
        if isinstance(v, tuple):
            out.extend(v)
        else:
            out.append(v)
    if isinstance(n, Cond):
        out += [n.t, n.e]
    return out


def walk(n):
    """n and every node below it (not into the FnSpecs it names)."""
    yield n
    for c in children(n):
        yield from walk(c)


def fnspecs(n):
    """The FnSpecs a node names directly."""
    return [getattr(n, f) for f in _FN_FIELDS if isinstance(getattr(n, f, None), FnSpec)]


_EFFECTS = {}


def effects(n):
    """Does evaluating n assign a variable or call an extern (directly or through a function)?"""
    if isinstance(n, FnSpec):
        key = id(n)
        if key not in _EFFECTS:
            _EFFECTS[key] = (n, effects(n.body))
        return _EFFECTS[key][1]
    if isinstance(n, (Assign, ExternCall)):
        return True
    if any(effects(f) for f in fnspecs(n)):
        return True
    return any(effects(c) for c in children(n))


# ---------------------------------------------------------------------------
# The text form
# ---------------------------------------------------------------------------

def _num(v):
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return str(int(v))


def _table(v):
    flat = np.asarray(v).reshape(-1)
    shown = " ".join(_num(x) for x in flat[:8])
    return f"[{shown}{' ...' if flat.size > 8 else ''}]"


def _fn(f):
    return f"<{f.kind} {f.label}>"


def _expr(n):
    k = type(n).__name__
    t = f":{n.ty}"
    if isinstance(n, Const):
        if n.ty.elem == "str":
            return f'(str "{n.value}")'
        return f"({_num(n.value)}{t})" if not n.ty.shape else f"(table {_table(n.value)}{t})"
    if isinstance(n, GlobalRef):
        return f"({'state' if n.state else 'const'} {n.name}{t})"
    if isinstance(n, LocalRef):
        return f"(local {n.slot} {n.name}{t})"
    if isinstance(n, ParamRef):
        return f"(param {n.index} {n.name}{t})"
    if isinstance(n, (Unary, Binary)):
        ops = [n.a] if isinstance(n, Unary) else [n.a, n.b]
        return f"({n.op}{t} {' '.join(_expr(q) for q in ops)})"
    if isinstance(n, Convert):
        return f"(convert{t} {_expr(n.a)})" if n.ty.elem != "float" or n.a.ty.elem != "int" else f"(float{t} {_expr(n.a)})"
    if isinstance(n, Match):
        return f"(match{t} {_expr(n.a)} {_expr(n.b)})"
    if isinstance(n, Index):
        return f"(index{t}{' checked' if n.checked else ''} {_expr(n.a)} {_expr(n.i)})"
    if isinstance(n, Gather):
        return f"(gather{t}{' checked' if n.checked else ''} {_expr(n.a)} {_expr(n.idx)})"
    if isinstance(n, (Take, Drop)):
        return f"({k.lower()}{t} {n.n} {_expr(n.a)})"
    if isinstance(n, Rotate):
        return f"(rotate{t} {_expr(n.n)} {_expr(n.a)})"
    if isinstance(n, (Reverse, Transpose, Reshape)):
        return f"({k.lower()}{t} {_expr(n.a)})"
    if isinstance(n, Join):
        return f"(join{t} {_expr(n.a)} {_expr(n.b)})"
    if isinstance(n, Iota):
        return f"(iota{t})"
    if isinstance(n, MakeList):
        return f"(list{t} {' '.join(_expr(q) for q in n.items)})"
    if isinstance(n, Amend):
        return f"(amend{t}{' checked' if n.checked else ''} {_expr(n.a)} {_expr(n.value)} {_expr(n.idx)})"
    if isinstance(n, Cond):
        return f"(cond{t} {_expr(n.c)} {_expr(n.t)} {_expr(n.e)})"
    if isinstance(n, Call):
        return f"(call{t} {_fn(n.fn)}{''.join(' ' + _expr(q) for q in n.args)})"
    if isinstance(n, ExternCall):
        return f"(extern{t} {n.ext.name}{''.join(' ' + _expr(q) for q in n.args)})"
    cap = lambda c: (" captures" + "".join(" " + _expr(q) for q in c)) if c else ""  # noqa: E731
    if isinstance(n, Map):
        return f"(each{t} {_fn(n.fn)} {_expr(n.a)}{cap(n.captures)})"
    if isinstance(n, Map2):
        return f"(each2{t} {_fn(n.fn)} {_expr(n.a)} {_expr(n.b)}{cap(n.captures)})"
    if isinstance(n, EachPair):
        return f"(each-pair{t} {_fn(n.fn)} {_expr(n.a)}{cap(n.captures)})"
    if isinstance(n, (Reduce, Scan)):
        seed = f" seed {_expr(n.seed)}" if n.seed is not None else ""
        return f"({'over' if isinstance(n, Reduce) else 'scan'}{t} {_fn(n.fn)}{seed} {_expr(n.a)}{cap(n.captures)})"
    if isinstance(n, Loop):
        head = f"count {_expr(n.count)}" if n.kind == "iterate" else f"pred {_fn(n.pred)}"
        return f"({n.kind}{t} {head} {_fn(n.fn)} {_expr(n.init)}{cap(n.captures)}{cap(n.pred_captures)})"
    if isinstance(n, Assign):
        return f"(set{t} {_expr(n.target)} {_expr(n.value)})"
    if isinstance(n, Seq):
        return f"(seq{t} {' '.join(_expr(q) for q in n.stmts)})"
    raise TypeError(k)


def dump(program):
    """
    The program as text, stable across runs: globals, externs, entries, then
    each specialisation with its storage-planning data (array locals, the
    array-valued intermediates and the functions it calls).
    """
    out = [f"program {program.name} (kg_int: {program.int_bits} bits)"]
    for g in program.globals:
        kind = "param" if g.param else "state" if g.state else "const"
        shown = _table(g.value) if g.ty.shape else _num(g.value)
        out.append(f"global {kind} {g.name}: {g.ty} = {shown}")
    for x in program.externs:
        out.append(f"extern {x.name}: {x.signature} (C {x.c_name}, module {x.module})")
    for e in program.entries:
        out.append(f"entry {e.c_name}: {e.decl.signature} -> {e.fn.label}")
    for f in program.fns:
        params = "; ".join(f"{n}: {t}" for n, t in zip(f.param_names, f.param_tys))
        out.append("")
        out.append(f"fn {f.label} [{f.kind}] ({params}) -> {f.result_ty}"
                   + (f"  captures {len(f.param_tys) - f.n_own}" if len(f.param_tys) > f.n_own else ""))
        for i, (n, t) in enumerate(zip(f.local_names, f.locals)):
            out.append(f"  local {i} {n}: {t}")
        stmts = f.body.stmts if isinstance(f.body, Seq) else (f.body,)
        for s in stmts:
            out.append(f"  {_expr(s)}")
        plan = storage_inputs(f)
        if plan["arrays"]:
            out.append("  arrays: " + ", ".join(plan["arrays"]))
        if plan["calls"]:
            out.append("  calls: " + ", ".join(plan["calls"]))
        out.append(f"  effects: {'yes' if effects(f) else 'no'}")
    return "\n".join(out) + "\n"


_LEAVES = (Const, GlobalRef, LocalRef, ParamRef)
_FUSIBLE = (Unary, Binary, Convert, Index, Gather, Take, Drop, Rotate, Reverse, Transpose, Reshape, Join, Iota)


def storage_inputs(f):
    """
    What the storage planner needs to know about one FnSpec:

        arrays   the array-valued intermediates, as "kind:type" in evaluation order, each marked
                 "fused" (an elementwise or index-remapping node: no storage of its own) or "slot"
        calls    labels of the FnSpecs called, in order of first mention
    """
    arrays, calls = [], []
    for n in walk(f.body):
        for g in fnspecs(n):
            if g.label not in calls:
                calls.append(g.label)
        if n.ty.shape and not isinstance(n, _LEAVES + (Assign, Seq)):
            arrays.append(f"{type(n).__name__.lower()}:{n.ty} {'fused' if isinstance(n, _FUSIBLE) else 'slot'}")
    return {"arrays": arrays, "calls": calls}
