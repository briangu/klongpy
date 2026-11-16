"""
The C emitter (tickets/design/static-compile.md sections 6 to 8; ticket 092).

    emit(program, prefix=None, float_bits=32, line_directives=False) -> Emitted

``program`` is an ir.Program (klongpy.static.infer.lower). The result holds
the text of ``<prefix>.c`` and ``<prefix>.h`` and the size report; the driver
(klongpy/static/compile.py) writes them next to a copy of klong_rt.h.

How code is generated

Every expression is compiled to a *value*: a scalar (a C expression) or an
array view (a function from index expressions to the C expression of one
element). Atomic operators, conversions and every index-remapping operator
(index, gather, take, drop, rotate, reverse, transpose, enumerate, join)
compose views and emit nothing, so a tree of them becomes one loop nest when
it is finally stored: that is fusion. Storage is only touched by

    * an assignment to an array variable (a loop into the variable),
    * the result of a function (a loop into ``out``),
    * the nodes that need a place: an evaluated list, amend, an array
      conditional, a call that returns an array, each, scan, an array
      accumulator. They get a *slot*, a field of the instance struct.

Evaluation order is Klong's. Statements are emitted in the order the
operands are evaluated (ir.Node.ORDER); a value computed before a later
operand that has a side effect is first *pinned* (copied to a C local or a
slot) if it reads anything that can change. Each view records which variables
it reads and whether element i of the view reads only element i of the
variable; an assignment whose value reads its target out of step goes
through a slot and a copy, otherwise it is done in place.

Slots are fields of ``struct <prefix>``; those of one function whose
lifetimes do not overlap (they are released at the end of each statement)
share a field. A function's slots are its own, so a callee never touches a
caller's. Scalars and loop counters are C locals.

Nothing here names a board, a chip or a vendor.
"""
import re

import numpy as np

from ..errors import KlongStaticError
from . import ir
from .optable import C_EXPANSION
from .signature import _C_KEYWORDS, Ty, c_identifier

__all__ = ["emit", "Emitted", "HANDLED"]

# the rows of the operator table this emitter handles (a test compares it with the table)
HANDLED = frozenset(C_EXPANSION)

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z|\(?-?\d+\)?\Z")


# ---------------------------------------------------------------------------
# Index arithmetic on C expressions, folding literals
# ---------------------------------------------------------------------------

def _lit(e):
    try:
        return int(e)
    except (TypeError, ValueError):
        return None


def _add(a, b):
    la, lb = _lit(a), _lit(b)
    if la is not None and lb is not None:
        return str(la + lb)
    if la == 0:
        return str(b)
    if lb == 0:
        return str(a)
    if lb is not None and lb < 0:
        return f"({a} - {-lb})"
    return f"({a} + {b})"


def _sub(a, b):
    la, lb = _lit(a), _lit(b)
    if la is not None and lb is not None:
        return str(la - lb)
    if lb == 0:
        return str(a)
    return f"({a} - {b})"


def _mul(a, b):
    la, lb = _lit(a), _lit(b)
    if la is not None and lb is not None:
        return str(la * lb)
    if la == 0 or lb == 0:
        return "0"
    if la == 1:
        return str(b)
    if lb == 1:
        return str(a)
    return f"({a} * {b})"


def _flat(idx, shape):
    e = str(idx[0])
    for k in range(1, len(shape)):
        e = _add(_mul(e, shape[k]), idx[k])
    return e


def _size(shape):
    n = 1
    for d in shape:
        n *= d
    return n


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------

class _S:
    """A scalar: a C expression. reads: the variables it reads that can change (empty: it can be used at any later point)."""
    __slots__ = ("elem", "c", "reads")

    def __init__(self, elem, c, reads=()):
        self.elem = elem
        self.c = c
        self.reads = frozenset(reads)

    shape = ()

    @property
    def atomic(self):
        return bool(_IDENT.match(self.c))


class _A:
    """
    An array view.

        ty      element type and shape
        at      function from a tuple of C index expressions to the C expression of that element
        reads   {storage key: aligned}; aligned means element i reads only element i of that storage
        ptr     the C pointer of its storage when the view is exactly a stored array, else None
        key     the storage key of ptr
    """
    __slots__ = ("ty", "at", "reads", "ptr", "key")

    def __init__(self, ty, at, reads=None, ptr=None, key=None):
        self.ty = ty
        self.at = at
        self.reads = dict(reads or {})
        self.ptr = ptr
        self.key = key

    @property
    def elem(self):
        return self.ty.elem

    @property
    def shape(self):
        return self.ty.shape


def _stored(ty, ptr, key=None, mutable=False):
    """The view of an array stored at ptr."""
    shape = ty.shape
    return _A(ty, lambda idx: f"{ptr}[{_flat(idx, shape)}]", {key: True} if mutable else {}, ptr, key)


class _Dest:
    """Where an array result may be written: fresh means nothing else can read or alias it while it is filled."""
    __slots__ = ("ptr", "key", "ty", "fresh")

    def __init__(self, ptr, key, ty, fresh):
        self.ptr = ptr
        self.key = key
        self.ty = ty
        self.fresh = fresh


def _unaligned(reads):
    return {k: False for k in reads}


def _merge(*parts):
    out = {}
    for reads in parts:
        for k, v in reads.items():
            out[k] = out.get(k, True) and v
    return out


class Emitted:
    """
    The result of emit(): ``c`` and ``h`` are the texts of <prefix>.c and
    <prefix>.h; ``report`` is the size report (a dict); ``prefix`` the C prefix.
    """
    def __init__(self, prefix, c, h, report):
        self.prefix = prefix
        self.c = c
        self.h = h
        self.report = report

    def report_text(self):
        r = self.report
        lines = [
            f"size report for {self.prefix} (kg_int {r['int_bits']} bits, kg_float {r['float_bits']} bits)",
            f"  constants          {r['const_bytes']} bytes (static const data; flash on a microcontroller)",
            f"  sizeof({self.prefix}_t)  {r['instance_bytes']} bytes"
            f" ({r['state_bytes']} state, {r['slot_bytes']} array locals and temporaries)",
            f"  deepest call chain {r['call_depth']} C function(s): {' > '.join(r['call_chain'])}",
            f"  stack estimate     {r['stack_bytes']} bytes (scalars, loop counters and call frames along that chain;"
            " externs not counted)",
            f"  C source           {r['c_lines']} lines",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# One C function being written
# ---------------------------------------------------------------------------

class _Code:
    def __init__(self):
        self.lines = []
        self.depth = 1
        self.n = 0
        self.scalars = 0        # C locals declared (for the stack estimate)

    def emit(self, text):
        self.lines.append("    " * self.depth + text)

    def name(self, stem):
        self.n += 1
        return f"{stem}{self.n}"


_TEMP_DECL = re.compile(r"(\s*)(?:const )?kg_\w+ (t\d+)( = .*)?;\Z")


def _mark_unused(lines):
    """
    A temporary that is computed for its effect only (the result of a call
    that is discarded, an index that is checked but selects nothing) gets a
    (void) use, so the function compiles without warnings.
    """
    text = "\n".join(line for line in lines if not line.lstrip().startswith(("/*", "#line")))
    out = []
    for line in lines:
        out.append(line)
        m = _TEMP_DECL.match(line)
        if m and len(re.findall(rf"\b{m.group(2)}\b", text)) == 1:
            out.append(f"{m.group(1)}(void){m.group(2)};")
    return out


class _Fn:
    """Compiles the body of one FnSpec into C statements."""

    def __init__(self, em, spec):
        self.em = em
        self.spec = spec
        self.code = _Code()
        self.params = []            # stack of parameter values (the function's own, then those of inlined verbs)
        self.locals = {}
        self.free = []              # released temp slots: (elem, field)
        self.live = []
        self.calls = []             # FnSpecs called as C functions
        self.tag = f"f{spec.index}"
        self.line = 0

    # -- small helpers ------------------------------------------------------------
    def ctype(self, elem):
        return "kg_float" if elem == "float" else "kg_int"

    def fail(self, node, message):
        raise KlongStaticError(f"the C emitter cannot compile this: {message} [unsupported-op]",
                               span=getattr(node, "span", None), source=self.spec.source)

    def line_of(self, node):
        sp = getattr(node, "span", None)
        src = self.spec.source
        if sp is None or src is None:
            return self.line
        try:
            return int(src.line_col(sp[0])[0])
        except Exception:
            return self.line

    def temp(self, elem, expr, const=True):
        """Declare a C local holding expr; returns its name."""
        name = self.code.name("t")
        self.code.scalars += 1
        self.code.emit(f"{'const ' if const else ''}{self.ctype(elem)} {name} = {expr};")
        return name

    def slot(self, ty):
        """A temp slot for an array of this type: a _Dest that is fresh."""
        need = _size(ty.shape)
        for i, (elem, field) in enumerate(self.free):
            if elem == ty.elem:
                del self.free[i]
                break
        else:
            field = self.em.new_slot(self.tag, ty.elem)
        self.em.grow_slot(field, need)
        self.live.append((ty.elem, field))
        return _Dest(f"p->{field}", f"t:{field}", ty, True)

    def release(self):
        self.free.extend(self.live)
        self.live = []

    def open_loops(self, shape):
        idx = []
        for d in shape:
            i = self.code.name("i")
            self.code.scalars += 1
            self.code.emit(f"for (kg_int {i} = 0; {i} < {d}; {i}++) {{")
            self.code.depth += 1
            idx.append(i)
        return tuple(idx)

    def close(self, n=1):
        for _ in range(n):
            self.code.depth -= 1
            self.code.emit("}")

    # -- values ---------------------------------------------------------------------
    def conv(self, v, elem):
        if v.elem == elem or elem != "float":
            return v
        if isinstance(v, _S):
            return _S("float", f"(kg_float)({v.c})" if not v.atomic else f"(kg_float){v.c}", v.reads)
        return _A(Ty("float", v.shape), lambda idx: f"(kg_float)({v.at(idx)})", v.reads)

    def pin(self, v):
        """v as a value that later statements cannot change."""
        if v is None:
            return v
        if isinstance(v, _S):
            if not v.reads:
                return v
            return _S(v.elem, self.temp(v.elem, v.c))
        if not v.reads:
            return v
        return self.store(v, self.slot(v.ty))

    def hoist(self, v):
        """A scalar that is cheap to repeat inside a loop."""
        if isinstance(v, _S) and not v.atomic and v.elem != "str":
            return _S(v.elem, self.temp(v.elem, v.c))
        return v

    def materialize(self, v):
        """An array view as a stored array (a pointer)."""
        if v.ptr is not None:
            return v
        return self.store(v, self.slot(v.ty))

    def store(self, v, dest):
        """Fill dest with the elements of v; returns the view of dest."""
        out = _stored(dest.ty, dest.ptr, dest.key, mutable=not dest.fresh)
        if isinstance(v, _A) and v.ptr == dest.ptr:
            return out
        if isinstance(v, _A) and v.reads.get(dest.key) is False:
            v = self.store(v, self.slot(v.ty))          # reads its own target out of step: through a slot
        v = self.conv(v, dest.ty.elem)
        shape = dest.ty.shape
        if isinstance(v, _S):
            v = self.hoist(v)
        idx = self.open_loops(shape)
        rhs = v.c if isinstance(v, _S) else v.at(idx)
        self.code.emit(f"{dest.ptr}[{_flat(idx, shape)}] = {rhs};")
        self.close(len(shape))
        return out

    def operands(self, nodes):
        """Compile nodes in this (evaluation) order, pinning a value when a later one has a side effect."""
        vals = []
        for k, n in enumerate(nodes):
            v = self.compile(n)
            if any(ir.effects(m) for m in nodes[k + 1:]):
                v = self.pin(v)
            vals.append(v)
        return vals

    def row(self, v, j):
        """Item j (a C index expression) of an array value: a scalar or a view."""
        rest = v.shape[1:]
        if not rest:
            return _S(v.elem, v.at((j,)), v.reads)
        ptr = None
        if v.ptr is not None:
            off = _mul(j, _size(rest))
            ptr = v.ptr if off == "0" else f"({v.ptr} + {off})"
        return _A(Ty(v.elem, rest), lambda idx: v.at((j,) + tuple(idx)), _unaligned(v.reads), ptr, v.key if ptr else None)

    def row_dest(self, dest, j):
        rest = dest.ty.shape[1:]
        off = _mul(j, _size(rest))
        return _Dest(dest.ptr if off == "0" else f"({dest.ptr} + {off})", dest.key, Ty(dest.ty.elem, rest), dest.fresh)

    def target(self, dest, ty, vals=(), fns=()):
        """dest if it may be filled while vals are still being read and fns are being called, else a new slot."""
        if dest is not None and dest.ty.shape == ty.shape and dest.ty.elem == ty.elem:
            if dest.fresh:
                return dest
            ok = all(v is None or dest.key not in v.reads for v in vals)
            if ok and dest.key.startswith("g:"):
                name = dest.key[2:]
                ok = all(name not in self.em.touches(f) for f in fns)
            if ok:
                return dest
        return self.slot(ty)

    # -- dispatch -----------------------------------------------------------------------
    def compile(self, node, dest=None):
        return getattr(self, "c_" + type(node).__name__)(node, dest)

    def c_Const(self, n, dest):
        if n.ty.elem == "str":
            return _S("str", '"' + n.value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"')
        if not n.ty.shape:
            return _S(n.ty.elem, self.em.literal(n.ty.elem, n.value))
        return _stored(n.ty, self.em.table(n.ty, n.value))

    def c_GlobalRef(self, n, dest):
        cid = c_identifier(n.name)
        if not n.state:
            return _stored(n.ty, f"kg_c_{cid}")
        key = f"g:{n.name}"
        if not n.ty.shape:
            return _S(n.ty.elem, f"p->g_{cid}", [key])
        return _stored(n.ty, f"p->g_{cid}", key, mutable=True)

    def c_LocalRef(self, n, dest):
        return self.local(n.slot)

    def local(self, slot):
        ty = self.spec.locals[slot]
        kind, c = self.locals[slot]
        if kind == "scalar":
            return _S(ty.elem, c, [f"v:{c}"])
        return _stored(ty, c, f"l:{slot}", mutable=True)

    def c_ParamRef(self, n, dest):
        return self.params[-1][n.index]

    def c_Convert(self, n, dest):
        return self.conv(self.compile(n.a), "float")

    def expand(self, op, elem, a, b, node):
        form = C_EXPANSION[op][2 if elem == "float" else 1]
        if form is None:
            self.fail(node, f"{op} on {elem}")
        if op == "pow" and elem == "float":
            self.em.need_math = True
        return form.format(a=a, b=b, line=self.line_of(node))

    def c_Unary(self, n, dest):
        a = self.compile(n.a)
        work = a.elem
        if isinstance(a, _S):
            return _S(n.ty.elem, self.expand(n.op, work, a.c, None, n), a.reads)
        return _A(n.ty, lambda idx: self.expand(n.op, work, a.at(idx), None, n), a.reads)

    def c_Binary(self, n, dest):
        b, a = self.operands([n.b, n.a])
        work = a.elem
        if isinstance(a, _S) and isinstance(b, _S):
            return _S(n.ty.elem, self.expand(n.op, work, a.c, b.c, n), a.reads | b.reads)
        rank = len(n.ty.shape)
        parts = []
        reads = {}
        for v in (a, b):
            if isinstance(v, _S):
                v = self.pin(self.hoist(v))
                parts.append(lambda idx, v=v: v.c)
            else:
                k = len(v.shape)
                parts.append(lambda idx, v=v, k=k: v.at(tuple(idx[:k])))
                reads = _merge(reads, v.reads if k == rank else _unaligned(v.reads))
        return _A(n.ty, lambda idx: self.expand(n.op, work, parts[0](idx), parts[1](idx), n), reads)

    def c_Match(self, n, dest):
        b, a = self.operands([n.b, n.a])
        form = C_EXPANSION["match"][2 if a.elem == "float" else 1]
        if isinstance(a, _S):
            return _S("int", form.format(a=f"({a.c})", b=f"({b.c})"), a.reads | b.reads)
        t = self.temp("int", "1", const=False)
        idx = self.open_loops(a.shape)
        eq = "kg_feq" if a.elem == "float" else "kg_ieq"
        self.code.emit(f"if (!{eq}({a.at(idx)}, {b.at(idx)})) {{")
        self.code.emit(f"    {t} = 0;")
        self.code.emit("}")
        self.close(len(a.shape))
        return _S("int", t)

    def index_expr(self, i, n, checked, node):
        return i if checked else f"kg_ix({i}, {n}, {self.line_of(node)})"

    def c_Index(self, n, dest):
        i, a = self.operands([n.i, n.a])
        j = i.c if n.checked else self.temp("int", self.index_expr(i.c, n.a.ty.shape[0], False, n))
        return self.row(a, j)

    def c_Gather(self, n, dest):
        idx, a = self.operands([n.idx, n.a])
        r = len(idx.shape)
        size = n.a.ty.shape[0]

        def at(I):
            return a.at((self.index_expr(idx.at(tuple(I[:r])), size, n.checked, n),) + tuple(I[r:]))
        return _A(n.ty, at, _unaligned(_merge(idx.reads, a.reads)))

    def c_Take(self, n, dest):
        a = self.compile(n.a)
        size = n.a.ty.shape[0]
        count = abs(n.n)
        start = 0 if n.n > 0 else (-count) % size
        if start + count <= size:
            return _A(n.ty, lambda idx: a.at((_add(idx[0], start),)), _unaligned(a.reads))
        return _A(n.ty, lambda idx: a.at((f"({_add(idx[0], start)} % {size})",)), _unaligned(a.reads))

    def c_Drop(self, n, dest):
        a = self.compile(n.a)
        off = n.n if n.n > 0 else 0
        return _A(n.ty, lambda idx: a.at((_add(idx[0], off),) + tuple(idx[1:])), _unaligned(a.reads))

    def c_Rotate(self, n, dest):
        a, k = self.operands([n.a, n.n])
        size = n.ty.shape[0]
        lit = _lit(k.c)
        if lit is not None:
            back = (size - lit % size) % size
        else:
            back = self.temp("int", f"{size} - kg_rot({k.c}, {size})")
        return _A(n.ty, lambda idx: a.at((f"({_add(idx[0], back)} % {size})",)), _unaligned(a.reads))

    def c_Reverse(self, n, dest):
        a = self.compile(n.a)
        last = n.ty.shape[0] - 1
        return _A(n.ty, lambda idx: a.at((_sub(last, idx[0]),) + tuple(idx[1:])), _unaligned(a.reads))

    def c_Transpose(self, n, dest):
        a = self.compile(n.a)
        return _A(n.ty, lambda idx: a.at((idx[1], idx[0])), _unaligned(a.reads))

    def c_Reshape(self, n, dest):
        a = self.compile(n.a)
        shape = n.ty.shape
        if isinstance(a, _S):
            a = self.pin(self.hoist(a))
            return _A(n.ty, lambda idx: a.c)
        if shape == (1,) + tuple(a.shape):
            return _A(n.ty, lambda idx: a.at(tuple(idx[1:])), _unaligned(a.reads))
        a = self.materialize(a)
        have = _size(a.shape)
        if _size(shape) <= have:
            return _A(n.ty, lambda idx: f"{a.ptr}[{_flat(idx, shape)}]", _unaligned(a.reads))
        return _A(n.ty, lambda idx: f"{a.ptr}[{_flat(idx, shape)} % {have}]", _unaligned(a.reads))

    def c_Join(self, n, dest):
        b, a = self.operands([n.b, n.a])
        na = a.shape[0] if a.shape else 1
        a = self.pin(self.hoist(a)) if isinstance(a, _S) else a
        b = self.pin(self.hoist(b)) if isinstance(b, _S) else b

        def part(v, j, rest):
            return v.c if isinstance(v, _S) else v.at((j,) + tuple(rest))

        def at(idx):
            j, rest = idx[0], idx[1:]
            lit = _lit(j)
            if lit is not None:
                return part(a, j, rest) if lit < na else part(b, str(lit - na), rest)
            return f"({j} < {na} ? {part(a, j, rest)} : {part(b, _sub(j, na), rest)})"
        return _A(n.ty, at, _unaligned(_merge(a.reads if isinstance(a, _A) else {}, b.reads if isinstance(b, _A) else {})))

    def c_Iota(self, n, dest):
        return _A(n.ty, lambda idx: str(idx[0]))

    def c_MakeList(self, n, dest):
        t = dest if (dest is not None and dest.fresh and dest.ty == n.ty) else self.slot(n.ty)
        for k, item in enumerate(n.items):
            v = self.compile(item)
            if isinstance(v, _S):
                self.code.emit(f"{t.ptr}[{k}] = {v.c};")
            else:
                self.store(v, self.row_dest(t, str(k)))
        return _stored(n.ty, t.ptr, t.key)

    def c_Amend(self, n, dest):
        idx, value, a = self.operands([n.idx, n.value, n.a])
        value = self.pin(self.hoist(value))
        if dest is not None and isinstance(idx, _A) and dest.key in idx.reads:
            idx = self.pin(idx) if not dest.fresh else idx
            if dest.key in idx.reads:
                idx = self.store(idx, self.slot(idx.ty))
        if isinstance(idx, _S):
            idx = self.pin(self.hoist(idx))
        t = self.target(dest, n.ty, [idx])
        self.store(a, t)
        size = n.ty.shape[0]
        if isinstance(idx, _S):
            self.code.emit(f"{t.ptr}[{self.index_expr(idx.c, size, n.checked, n)}] = {value.c};")
        else:
            loop = self.open_loops(idx.shape)
            self.code.emit(f"{t.ptr}[{self.index_expr(idx.at(loop), size, n.checked, n)}] = {value.c};")
            self.close(len(loop))
        return _stored(n.ty, t.ptr, t.key, mutable=not t.fresh)

    def branch(self, node):
        """Compile node into a separate list of lines (one block deeper). Returns (lines, value)."""
        saved = self.code.lines
        self.code.lines = []
        self.code.depth += 1
        try:
            v = self.compile(node)
            return self.code.lines, v
        finally:
            self.code.depth -= 1
            self.code.lines = saved

    def c_Cond(self, n, dest):
        c = self.compile(n.c)
        cond = f"{c.c} != 0"
        lt, vt = self.branch(n.t)
        le, ve = self.branch(n.e)
        if n.ty.elem == "void":
            self.code.emit(f"if ({cond}) {{")
            self.code.lines += lt
            self.code.emit("} else {")
            self.code.lines += le
            self.code.emit("}")
            return None
        if not n.ty.shape:
            if not lt and not le:
                return _S(n.ty.elem, f"({cond} ? {vt.c} : {ve.c})", c.reads | vt.reads | ve.reads)
            r = self.code.name("t")
            self.code.scalars += 1
            self.code.emit(f"{self.ctype(n.ty.elem)} {r};")
            self.code.emit(f"if ({cond}) {{")
            self.code.lines += lt
            self.code.emit(f"    {r} = {vt.c};")
            self.code.emit("} else {")
            self.code.lines += le
            self.code.emit(f"    {r} = {ve.c};")
            self.code.emit("}")
            return _S(n.ty.elem, r)
        t = self.target(dest, n.ty, [vt, ve]) if dest is not None else self.slot(n.ty)
        self.code.emit(f"if ({cond}) {{")
        self.code.lines += lt
        self.code.depth += 1
        self.store(vt, t)
        self.code.depth -= 1
        self.code.emit("} else {")
        self.code.lines += le
        self.code.depth += 1
        self.store(ve, t)
        self.code.depth -= 1
        self.code.emit("}")
        return _stored(n.ty, t.ptr, t.key, mutable=not t.fresh)

    # -- calls ----------------------------------------------------------------------------
    def arg(self, v, fn):
        if isinstance(v, _S):
            return v.c
        v = self.materialize(v)
        if fn is not None and v.key is not None and v.key.startswith("g:") and v.key[2:] in self.em.writes(fn):
            v = self.store(v, self.slot(v.ty))      # the callee assigns the global this argument points into
        return v.ptr

    def call_c(self, name, with_p, result, vals, dest, fn):
        args = (["p"] if with_p else []) + [self.arg(v, fn) for v in vals]
        if result.elem == "void":
            self.code.emit(f"{name}({', '.join(args)});")
            return None
        if not result.shape:
            t = self.temp(result.elem, f"{name}({', '.join(args)})")
            return _S(result.elem, t)
        t = dest if (dest is not None and dest.fresh and dest.ty == result) else self.slot(result)
        self.code.emit(f"{name}({', '.join(args + [t.ptr])});")
        return _stored(result, t.ptr, t.key)

    def c_Call(self, n, dest):
        vals = self.operands(list(n.args))
        self.calls.append(n.fn)
        return self.call_c(self.em.fn_name(n.fn), True, n.fn.result_ty, vals, dest, n.fn)

    def c_ExternCall(self, n, dest):
        vals = self.operands(list(n.args))
        return self.call_c(n.ext.c_name, False, n.ty, vals, dest, None)

    def apply(self, fn, vals, dest=None):
        """The value of fn applied to these argument values: inlined when fn is a small pure verb, else a C call."""
        if self.em.inline(fn):
            uses = self.em.param_uses(fn)
            bound = []
            for i, v in enumerate(vals):
                if isinstance(v, _S) and uses.get(i, 0) > 1:
                    v = self.hoist(v)
                bound.append(v)
            self.params.append(bound)
            try:
                return self.compile(fn.body.stmts[0], dest)
            finally:
                self.params.pop()
        self.calls.append(fn)
        return self.call_c(self.em.fn_name(fn), True, fn.result_ty, vals, dest, fn)

    def captured(self, nodes):
        return [self.pin(self.hoist(self.compile(c))) if not c.ty.shape else self.compile(c) for c in nodes]

    def settle(self, vals, fns):
        """Operands of an adverb whose verb has side effects are evaluated in full before the first call."""
        if any(ir.effects(f) for f in fns):
            return [self.pin(v) for v in vals]
        return vals

    def each(self, n, vals, items_of, count):
        fn = n.fn
        caps = self.captured(n.captures)
        vals = self.settle(vals, [fn])
        r = fn.result_ty
        t = None if r.elem == "void" else self.target(dest_of(n), n.ty, list(vals) + caps, [fn])
        (j,) = self.open_loops((count,))
        res = self.apply(fn, items_of(j) + caps, None if t is None or not r.shape else self.row_dest(t, j))
        if t is not None:
            if r.shape:
                self.store(res, self.row_dest(t, j))
            else:
                self.code.emit(f"{t.ptr}[{j}] = {self.conv(res, n.ty.elem).c};")
        self.close()
        return None if t is None else _stored(n.ty, t.ptr, t.key, mutable=not t.fresh)

    def c_Map(self, n, dest):
        _DEST[id(n)] = dest
        (a,) = self.operands([n.a])
        return self.each(n, [a], lambda j: [self.row(a, j)], n.a.ty.shape[0])

    def c_Map2(self, n, dest):
        _DEST[id(n)] = dest
        a, b = self.operands([n.a, n.b])
        return self.each(n, [a, b], lambda j: [self.row(a, j), self.row(b, j)], n.a.ty.shape[0])

    def c_EachPair(self, n, dest):
        _DEST[id(n)] = dest
        (a,) = self.operands([n.a])
        return self.each(n, [a], lambda j: [self.row(a, j), self.row(a, _add(j, 1))], n.a.ty.shape[0] - 1)

    def fold(self, n, dest, acc_ty, scan):
        """Over and Scan."""
        fn = n.fn
        nodes = ([n.seed] if n.seed is not None else []) + [n.a]
        vals = self.operands(nodes)
        caps = self.captured(n.captures)
        vals = self.settle(vals, [fn])
        a = vals[-1]
        count = n.a.ty.shape[0]
        init = vals[0] if n.seed is not None else self.row(a, "0")
        first = 0 if n.seed is not None else 1
        out = self.target(dest, n.ty, vals + caps, [fn]) if scan else None
        k0 = 1 if scan else 0       # row of the scan result the first step writes
        if not acc_ty.shape:
            acc = self.temp(acc_ty.elem, self.conv(init, acc_ty.elem).c, const=False)
            if scan:
                self.code.emit(f"{out.ptr}[0] = {acc};")
            if count > first:
                j = self.code.name("i")
                self.code.scalars += 1
                self.code.emit(f"for (kg_int {j} = {first}; {j} < {count}; {j}++) {{")
                self.code.depth += 1
                res = self.apply(fn, [_S(acc_ty.elem, acc), self.row(a, j)] + caps)
                self.code.emit(f"{acc} = {self.conv(res, acc_ty.elem).c};")
                if scan:
                    self.code.emit(f"{out.ptr}[{_add(j, k0 - first)}] = {acc};")
                self.close()
            return _stored(n.ty, out.ptr, out.key, mutable=not out.fresh) if scan else _S(acc_ty.elem, acc)
        if scan:
            self.store(init, self.row_dest(out, "0"))
            if count > first:
                j = self.code.name("i")
                self.code.scalars += 1
                self.code.emit(f"for (kg_int {j} = {first}; {j} < {count}; {j}++) {{")
                self.code.depth += 1
                prev = self.row_dest(out, _add(j, k0 - first - 1))
                cur = self.row_dest(out, _add(j, k0 - first))
                res = self.apply(fn, [_stored(acc_ty, prev.ptr), self.row(a, j)] + caps)
                self.store(res, _Dest(cur.ptr, None, acc_ty, True))
                self.close()
            return _stored(n.ty, out.ptr, out.key, mutable=not out.fresh)
        t = self.target(dest, acc_ty, vals + caps, [fn])
        self.store(init, t)
        if count > first:
            j = self.code.name("i")
            self.code.scalars += 1
            self.code.emit(f"for (kg_int {j} = {first}; {j} < {count}; {j}++) {{")
            self.code.depth += 1
            res = self.apply(fn, [_A(acc_ty, _stored(acc_ty, t.ptr).at, {t.key: True}, t.ptr, t.key), self.row(a, j)] + caps)
            self.store(res, _Dest(t.ptr, t.key, acc_ty, False))
            self.close()
        return _stored(acc_ty, t.ptr, t.key, mutable=not t.fresh)

    def c_Reduce(self, n, dest):
        return self.fold(n, dest, n.ty, False)

    def c_Scan(self, n, dest):
        return self.fold(n, dest, n.acc_ty, True)

    def c_Loop(self, n, dest):
        fn = n.fn
        if n.kind == "iterate":
            count, init = self.operands([n.count, n.init])
            count = self.pin(self.hoist(count))
        else:
            (init,) = self.operands([n.init])
            count = None
        caps = self.captured(n.captures)
        pcaps = self.captured(n.pred_captures)
        fns = [fn] + ([n.pred] if n.pred is not None else [])
        (init,) = self.settle([init], fns)
        ty = n.ty
        if ty.shape:
            t = self.target(dest, ty, [init] + caps + pcaps, fns)
            self.store(init, t)

            def state():
                return _A(ty, _stored(ty, t.ptr).at, {t.key: True}, t.ptr, t.key)
        else:
            var = self.temp(ty.elem, init.c, const=False)

            def state():
                return _S(ty.elem, var)
        if n.kind == "iterate":
            k = self.code.name("i")
            self.code.scalars += 1
            self.code.emit(f"for (kg_int {k} = 0; {k} < {count.c}; {k}++) {{")
            self.code.depth += 1
        else:
            self.code.emit("for (;;) {")
            self.code.depth += 1
            c = self.apply(n.pred, [state()] + pcaps)
            self.code.emit(f"if (!({c.c} != 0)) {{")
            self.code.emit("    break;")
            self.code.emit("}")
        res = self.apply(fn, [state()] + caps)
        if ty.shape:
            self.store(res, _Dest(t.ptr, t.key, ty, False))
        else:
            self.code.emit(f"{var} = {res.c};")
        self.close()
        if ty.shape:
            return _stored(ty, t.ptr, t.key, mutable=not t.fresh)
        return _S(ty.elem, var)

    # -- statements ---------------------------------------------------------------------------
    def c_Assign(self, n, dest):
        target = n.target
        if isinstance(target, ir.LocalRef):
            kind, c = self.locals[target.slot]
            key = f"v:{c}" if kind == "scalar" else f"l:{target.slot}"
        else:
            c = f"p->g_{c_identifier(target.name)}"
            key = f"g:{target.name}"
        if not n.ty.shape:
            v = self.compile(n.value)
            self.code.emit(f"{c} = {v.c};")
            return _S(n.ty.elem, c, [key])
        d = _Dest(c, key, n.ty, False)
        v = self.compile(n.value, d)
        return self.store(v, d)

    def c_Seq(self, n, dest):
        v = None
        for i, s in enumerate(n.stmts):
            v = self.compile(s, dest if i == len(n.stmts) - 1 else None)
        return v

    # -- the function ---------------------------------------------------------------------------
    def build(self):
        spec = self.spec
        em = self.em
        own = []
        decl = [f"{em.prefix}_t *p"]
        for i, (name, ty) in enumerate(zip(spec.param_names, spec.param_tys)):
            c = f"a{i}_{c_identifier(name)}"
            if ty.shape:
                decl.append(f"const {self.ctype(ty.elem)} *{c}")
                own.append(_stored(ty, c))
            else:
                decl.append(f"{self.ctype(ty.elem)} {c}")
                own.append(_S(ty.elem, c))
        self.params.append(own)
        r = spec.result_ty
        out = None
        if r.shape:
            decl.append(f"{self.ctype(r.elem)} *out")
            out = _Dest("out", "out", r, True)
        head = []
        for slot, (name, ty) in enumerate(zip(spec.local_names, spec.locals)):
            cid = re.sub(r"[^A-Za-z0-9_]", "_", c_identifier(name))
            if ty.shape:
                field = em.new_local(self.tag, slot, cid, ty)
                self.locals[slot] = ("array", f"p->{field}")
            else:
                c = f"v{slot}_{cid}"
                self.locals[slot] = ("scalar", c)
                head.append(f"    {self.ctype(ty.elem)} {c} = 0;")
                head.append(f"    (void){c};")
                self.code.scalars += 1
        stmts = spec.body.stmts
        src = spec.source
        for i, s in enumerate(stmts):
            last = i == len(stmts) - 1
            self.comment(spec.stmt_spans[i] if i < len(spec.stmt_spans) and spec.stmt_spans[i] else getattr(s, "span", None), src)
            v = self.compile(s, out if last else None)
            if last:
                if r.shape:
                    self.store(v, out)
                elif r.elem != "void":
                    self.code.emit(f"return {v.c};")
            self.release()
        rtype = "void" if (r.shape or r.elem == "void") else self.ctype(r.elem)
        sig = f"static {rtype} {em.fn_name(spec)}({', '.join(decl)})"
        unused = [f"    (void)a{i}_{c_identifier(name)};" for i, name in enumerate(spec.param_names)]
        body = ["    (void)p;"] + unused + head + self.code.lines
        return sig, _mark_unused(body)

    def comment(self, sp, src):
        if sp is None or src is None:
            return
        try:
            line = int(src.line_col(sp[0])[0])
        except Exception:
            return
        self.line = line
        text = " ".join(src.text[sp[0]:sp[1]].split())
        if len(text) > 72:
            text = text[:69] + "..."
        text = text.replace("*/", "* /").replace("/*", "/ *").replace("\\", "\\\\")
        name = self.em.source_name
        if self.em.line_directives:
            self.code.lines.append(f'#line {line} "{name}"')
        self.code.emit(f"/* {name}:{line}: {text} */")


_DEST = {}


def dest_of(n):
    return _DEST.pop(id(n), None)


# ---------------------------------------------------------------------------
# The whole program
# ---------------------------------------------------------------------------

class _Emitter:
    def __init__(self, program, prefix, float_bits, line_directives):
        self.program = program
        self.prefix = prefix
        self.int_bits = program.int_bits
        self.float_bits = float_bits
        self.line_directives = line_directives
        src = program.source
        self.source_name = (getattr(src, "name", None) or program.name).replace("\\", "/").split("/")[-1]
        self.tables = []            # (name, ty, array)
        self.table_keys = {}
        self.slots = {}             # field -> [elem, size, comment]
        self.order = []
        self.need_math = False
        self._writes = {}
        self._reads = {}
        self._uses = {}
        self.frames = {}            # fn index -> (scalars, calls)

    def isize(self, elem):
        return (self.float_bits if elem == "float" else self.int_bits) // 8

    # -- literals and tables --------------------------------------------------------------
    def literal(self, elem, v):
        if elem == "float":
            v = float(v)
            if v != v:
                return "KG_NAN"
            if v in (float("inf"), float("-inf")):
                return "KG_INF" if v > 0 else "(-KG_INF)"
            return f"((kg_float){v!r})"
        v = int(v)
        lo = -(1 << (self.int_bits - 1))
        if v == lo:
            return f"({v + 1} - 1)"
        return str(v) if v >= 0 else f"({v})"

    def table(self, ty, value, name=None):
        arr = np.asarray(value, dtype=np.float64 if ty.elem == "float" else np.int64)
        key = (ty.elem, tuple(arr.shape), arr.tobytes())
        if name is None and key in self.table_keys:
            return self.table_keys[key]
        if name is None:
            name = f"kg_k{len(self.tables)}"
            self.table_keys[key] = name
        self.tables.append((name, ty, arr))
        return name

    def table_text(self, name, ty, arr):
        ctype = "kg_float" if ty.elem == "float" else "kg_int"
        flat = arr.reshape(-1)
        if ty.elem == "float":
            items = [self.literal("float", x).replace("((kg_float)", "").rstrip(")") if x == x and abs(x) != float("inf")
                     else self.literal("float", x) for x in flat]
        else:
            items = [self.literal("int", x) for x in flat]
        lines = []
        row = "   "
        for it in items:
            if len(row) + len(it) + 2 > 110:
                lines.append(row)
                row = "   "
            row += " " + it + ","
        lines.append(row)
        dims = ",".join(str(d) for d in ty.shape)
        return [f"static const {ctype} {name}[{flat.size}] = {{ /* {ty.elem}[{dims}] */"] + lines + ["};"]

    # -- slots ------------------------------------------------------------------------------------
    def new_slot(self, tag, elem):
        field = f"{tag}_t{sum(1 for f in self.slots if f.startswith(tag + '_t'))}"
        self.slots[field] = [elem, 0, "temporary"]
        return field

    def grow_slot(self, field, size):
        self.slots[field][1] = max(self.slots[field][1], size)

    def new_local(self, tag, slot, cid, ty):
        field = f"{tag}_l{slot}_{cid}"
        self.slots[field] = [ty.elem, _size(ty.shape), f"local {cid}: {ty}"]
        return field

    # -- facts about functions -----------------------------------------------------------------------
    def fn_name(self, f):
        cid = re.sub(r"[^A-Za-z0-9_]", "_", c_identifier(f.name)) if f.kind != "op" else "op"
        return f"kg_{cid}__{f.index}"

    def inline(self, f):
        return (f.kind in ("op", "lambda") and not f.locals and len(f.body.stmts) == 1 and not ir.effects(f)
                and not any(isinstance(n, ir.ExternCall) for n in ir.walk(f.body)))

    def _collect(self, f, cache, pick):
        got = cache.get(id(f))
        if got is None:
            got = set()
            cache[id(f)] = got
            for n in ir.walk(f.body):
                got |= pick(n)
                for g in ir.fnspecs(n):
                    got |= self._collect(g, cache, pick)
        return got

    def writes(self, f):
        """Names of the globals f assigns, directly or through what it calls."""
        return self._collect(f, self._writes,
                             lambda n: {n.target.name} if isinstance(n, ir.Assign) and isinstance(n.target, ir.GlobalRef) else set())

    def touches(self, f):
        """Names of the state globals f reads or assigns, directly or through what it calls."""
        return self._collect(f, self._reads, lambda n: {n.name} if isinstance(n, ir.GlobalRef) and n.state else set())

    def param_uses(self, f):
        u = self._uses.get(id(f))
        if u is None:
            u = {}
            for n in ir.walk(f.body):
                if isinstance(n, ir.ParamRef):
                    u[n.index] = u.get(n.index, 0) + 1
            self._uses[id(f)] = u
        return u

    # -- output ----------------------------------------------------------------------------------------
    def run(self):
        prog = self.program
        P = self.prefix
        fns = []
        for f in prog.fns:
            if self.inline(f):
                continue
            ctx = _Fn(self, f)
            sig, body = ctx.build()
            fns.append((f, sig, body))
            self.frames[f.index] = (ctx.code.scalars, [g for g in ctx.calls])
        state = [g for g in prog.globals if g.state]
        consts = [g for g in prog.globals if not g.state and g.ty.shape]
        for g in consts:
            self.table(g.ty, g.value, f"kg_c_{c_identifier(g.name)}")
        for g in state:
            if g.ty.shape:
                self.table(g.ty, g.value, f"kg_i_{c_identifier(g.name)}")

        # ---- the entries
        wrappers = []
        protos = []
        entry_slots = 0
        for e in prog.entries:
            sig = e.decl.signature
            f = e.fn
            want = sig.result
            ctype = lambda t: "kg_float" if t.elem == "float" else "kg_int"  # noqa: E731
            decl = [f"{P}_t *p"]
            args = ["p"]
            names = []
            for i, (name, ty) in enumerate(zip(f.param_names, sig.params)):
                c = re.sub(r"[^A-Za-z0-9_]", "_", c_identifier(name))
                c = c if c not in ("p", "out", "i") and c not in names and c not in _C_KEYWORDS else f"a{i}_{c}"
                names.append(c)
                dims = ",".join(str(d) for d in ty.shape)
                decl.append(f"const {ctype(ty)} *{c} /* [{dims}] */" if ty.shape else f"{ctype(ty)} {c}")
                args.append(c)
            if want.shape:
                dims = ",".join(str(d) for d in want.shape)
                decl.append(f"{ctype(want)} *out /* [{dims}] */")
            rtype = "void" if (want.shape or want.elem == "void") else ctype(want)
            proto = f"{rtype} {P}_{e.c_name}({', '.join(decl)})"
            call = f"{self.fn_name(f)}({', '.join(args + (['out'] if f.result_ty.shape and f.result_ty.elem == want.elem else []))})"
            body = []
            if want.elem == "void":
                if f.result_ty.shape:
                    field = f"e{entry_slots}"
                    entry_slots += 1
                    self.slots[field] = [f.result_ty.elem, _size(f.result_ty.shape), f"discarded result of {e.c_name}"]
                    call = f"{self.fn_name(f)}({', '.join(args + ['p->' + field])})"
                body.append(f"    {'(void)' if f.result_ty.elem != 'void' and not f.result_ty.shape else ''}{call};")
            elif not want.shape:
                body.append(f"    return {'(kg_float)' if want.elem != f.result_ty.elem else ''}{call};")
            elif f.result_ty.elem == want.elem:
                body.append(f"    {call};")
            else:
                field = f"e{entry_slots}"
                entry_slots += 1
                size = _size(want.shape)
                self.slots[field] = [f.result_ty.elem, size, f"int result of {e.c_name} before conversion"]
                body.append(f"    {self.fn_name(f)}({', '.join(args + ['p->' + field])});")
                body.append(f"    for (kg_int i = 0; i < {size}; i++) {{")
                body.append(f"        out[i] = (kg_float)p->{field}[i];")
                body.append("    }")
            wrappers.append((e, proto, body))
            protos.append((proto, f"entry {e.decl.name}: {sig}"))

        # ---- the instance struct: 8-byte fields first, so there is no padding between fields
        fields = []
        for g in state:
            cid = c_identifier(g.name)
            fields.append((g.ty.elem, f"g_{cid}", _size(g.ty.shape) if g.ty.shape else 0, f"state {g.name}: {g.ty}", "state"))
        for field, (elem, size, note) in self.slots.items():
            fields.append((elem, field, max(size, 1), note, "slot"))
        fields.sort(key=lambda f: -self.isize(f[0]))
        state_bytes = sum(self.isize(f[0]) * max(f[2], 1) for f in fields if f[4] == "state")
        slot_bytes = sum(self.isize(f[0]) * max(f[2], 1) for f in fields if f[4] == "slot")
        total = state_bytes + slot_bytes
        if not fields:
            total = self.int_bits // 8
        widest = max([self.isize(f[0]) for f in fields] or [self.int_bits // 8])
        total = (total + widest - 1) // widest * widest
        const_bytes = sum(self.isize(ty.elem) * arr.size for _, ty, arr in self.tables)

        guard = f"{P.upper()}_H"
        h = [f"/* {P}.h - generated by KlongPy (kgpy --emit-c) from {self.source_name}. Do not edit. */",
             f"#ifndef {guard}", f"#define {guard}", ""]
        h.append("/* The element types this program was compiled for. */")
        if self.int_bits == 64:
            h += ["#ifndef KG_INT64", "#define KG_INT64 1", "#endif"]
        else:
            h += ["#ifdef KG_INT64", f"#error \"{P} was compiled for a 32-bit kg_int\"", "#endif"]
        if self.float_bits == 64:
            h += ["#ifndef KG_FLOAT64", "#define KG_FLOAT64 1", "#endif"]
        else:
            h += ["#ifdef KG_FLOAT64", f"#error \"{P} was compiled for a 32-bit kg_float\"", "#endif"]
        if self.need_math:
            h += ["/* The program uses ^ with a float result: pow from <math.h>, or kg_pow of klong_rt.h under KG_FREESTANDING. */",
                  "#ifndef KG_NEED_MATH", "#define KG_NEED_MATH 1", "#endif"]
        h += ['#include "klong_rt.h"', ""]
        h += ["/*",
              f" * One instance of the program: its state and the storage of its array temporaries.",
              " * The caller chooses where it lives (a static, a heap block, external RAM) and may",
              " * keep several. One instance must not be entered from two threads at once. The fields",
              " * are private to the generated code.",
              " */",
              f"struct {P} {{"]
        if not fields:
            h.append("    kg_int kg_unused; /* the program has no state and needs no array storage */")
        for elem, name, size, note, _ in fields:
            ctype = "kg_float" if elem == "float" else "kg_int"
            h.append(f"    {ctype} {name}{f'[{size}]' if size else ''}; /* {note} */")
        h += ["};", f"typedef struct {P} {P}_t;", f"typedef struct {P} {P}_storage_t; /* for static allocation */",
              f"#define {P.upper()}_SIZE (sizeof(struct {P}))", ""]
        h += [f"/* Sets the state to its initial values. Call it once before the first entry. */",
              f"void {P}_init({P}_t *p);",
              f"/* sizeof({P}_t), for a host that allocates the instance itself. */",
              f"size_t {P}_size(void);", ""]
        if protos:
            h.append("/* Entries. An array argument and the out array must not overlap. */")
        for proto, note in protos:
            h.append(f"{proto}; /* {note} */")
        if prog.externs:
            h += ["", "/* Externs: functions the target must define. */"]
        for x in prog.externs:
            h.append(f"{self.extern_proto(x)}; /* Klong name {x.name}: {x.signature} (module {x.module}) */")
        h += ["", f"#endif /* {guard} */", ""]

        c = [f"/* {P}.c - generated by KlongPy (kgpy --emit-c) from {self.source_name}. Do not edit. */",
             "#define KG_RT_IMPLEMENTATION /* the weak kg_trap (and freestanding memcpy, memset) are defined in this file */",
             f'#include "{P}.h"', ""]
        for name, ty, arr in self.tables:
            c += self.table_text(name, ty, arr)
            c.append("")
        for f, sig, body in fns:
            c.append(f"{sig};")
        if fns:
            c.append("")
        for f, sig, body in fns:
            params = "; ".join(str(t) for t in f.param_tys)
            c.append(f"/* {f.name}({params}) -> {f.result_ty} */")
            c.append(sig + " {")
            c += body
            c.append("}")
            c.append("")
        c.append(f"void {P}_init({P}_t *p) {{")
        c.append("    (void)p;")
        for g in state:
            cid = c_identifier(g.name)
            if g.ty.shape:
                c.append(f"    for (kg_int i = 0; i < {_size(g.ty.shape)}; i++) {{")
                c.append(f"        p->g_{cid}[i] = kg_i_{cid}[i];")
                c.append("    }")
            else:
                c.append(f"    p->g_{cid} = {self.literal(g.ty.elem, g.value)};")
        c += ["}", "", f"size_t {P}_size(void) {{", f"    return sizeof({P}_t);", "}", ""]
        for e, proto, body in wrappers:
            c.append(proto + " {")
            c += body
            c.append("}")
            c.append("")
        ctext = "\n".join(c)
        if self.line_directives:
            pass
        chain, stack = self.stack(prog)
        report = {
            "int_bits": self.int_bits, "float_bits": self.float_bits,
            "const_bytes": const_bytes, "instance_bytes": total, "state_bytes": state_bytes, "slot_bytes": slot_bytes,
            "call_depth": len(chain), "call_chain": chain, "stack_bytes": stack,
            "c_lines": ctext.count("\n") + 1, "need_math": self.need_math,
            "entries": [f"{P}_{e.c_name}" for e in prog.entries],
            "externs": [x.c_name for x in prog.externs],
        }
        return Emitted(P, ctext, "\n".join(h), report)

    def extern_proto(self, x):
        sig = x.signature
        ctype = lambda t: "kg_float" if t.elem == "float" else "kg_int"  # noqa: E731
        decl = []
        for i, ty in enumerate(sig.params):
            if ty.elem == "str":
                decl.append(f"const char *a{i}")
            elif ty.shape:
                decl.append(f"const {ctype(ty)} *a{i} /* [{','.join(str(d) for d in ty.shape)}] */")
            else:
                decl.append(f"{ctype(ty)} a{i}")
        r = sig.result
        if r.shape:
            decl.append(f"{ctype(r)} *out /* [{','.join(str(d) for d in r.shape)}] */")
        rtype = "void" if (r.shape or r.elem == "void") else ctype(r)
        return f"{rtype} {x.c_name}({', '.join(decl) or 'void'})"

    def stack(self, prog):
        """
        The deepest chain of C calls from an entry and an estimate of its stack
        use: per frame, a fixed 16 bytes (return address and frame pointer),
        one word per argument and one per scalar local or loop counter.
        """
        word = max(self.int_bits, self.float_bits, 32) // 8
        by_index = {f.index: f for f in prog.fns}
        memo = {}

        def depth(f):
            if f.index not in memo:
                scalars, calls = self.frames.get(f.index, (0, []))
                own = 16 + word * (len(f.param_tys) + 2 + scalars)
                best = ([], 0)
                for g in calls:
                    cand = depth(by_index[g.index])
                    if cand[1] > best[1] or (cand[1] == best[1] and len(cand[0]) > len(best[0])):
                        best = cand
                memo[f.index] = ([self.fn_name(f)] + best[0], own + best[1])
            return memo[f.index]

        best = ([], 0)
        for e in prog.entries:
            chain, size = depth(e.fn)
            own = 16 + word * (len(e.fn.param_tys) + 2)
            cand = ([f"{self.prefix}_{e.c_name}"] + chain, size + own)
            if cand[1] > best[1]:
                best = cand
        return best


def emit(program, prefix=None, float_bits=32, line_directives=False):
    """
    Emit C for a lowered program. prefix is the C prefix of every public name
    (default: the program's name): <prefix>_t, <prefix>_init, <prefix>_<entry>.
    Raises KlongStaticError when a public name would clash.
    """
    if float_bits not in (32, 64):
        raise ValueError("float_bits must be 32 or 64")
    prefix = prefix or program.name
    if not re.match(r"[A-Za-z][A-Za-z0-9_]*\Z", prefix) or prefix == "kg" or prefix.startswith("kg_") or prefix in _C_KEYWORDS:
        raise ValueError(f"'{prefix}' cannot be the C prefix of a program: it must be a C identifier that is not a C keyword "
                         "and does not start with kg_")
    public = {f"{prefix}_{e.c_name}" for e in program.entries} | {f"{prefix}_init", f"{prefix}_size", f"{prefix}_t", f"{prefix}_storage_t"}
    for x in program.externs:
        if x.c_name in public or x.c_name == prefix:
            raise KlongStaticError(f"the C name of extern {x.name} is '{x.c_name}', which this program generates itself [extern]",
                                   span=x.span, source=x.source,
                                   hint="rename the extern, or compile with another --prefix")
    return _Emitter(program, prefix, float_bits, line_directives).run()
