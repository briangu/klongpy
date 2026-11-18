"""
Lowering to the typed IR, with type and shape inference and specialisation
(docs/static-profile.md).

    lower(report, values, name="prog", int_bits=32) -> ir.Program

``report`` is the StaticReport of a program the checker accepted with every
global resolved (klongpy.static.checker.check_source(..., globals=values));
``values`` are the stage-one values of its globals. Lowering walks the same
parse trees the checker walked, starting from each ``.entry`` with the
argument types of its signature, and builds one FnSpec per function and
tuple of argument types. It never evaluates anything.

The type and shape of every operator come from the columns of the operator
table (klongpy.static.optable: ``type_rule`` and ``shape_rule``); the code
below applies the rule a row names and has no rule of its own. The checker
has already reported everything that is outside the profile, so an error
here is rare: it is raised as a KlongStaticError with the rule id, at the
first problem, for the cases the checker cannot decide without constant
values it does not compute.

What lowering decides that the checker does not record

    * conversions: int operands of a float operation, int values stored in a
      float variable, int branches of a float conditional are wrapped in
      Convert, so the emitter never promotes on its own;
    * captured variables: a function literal used as a verb that reads a
      variable of the function it is written in gets that variable as an
      extra parameter, and the adverb node passes it (``captures``);
    * parameters that are assigned are copied into a local on entry;
    * constants: #a, ^a, !n, literals and arithmetic on int constants are
      folded, constant indices are range-checked and made non-negative.
"""
import numpy as np

from ..parser import KGExprArray
from ..types import KGCall, KGChar, KGCond, KGDestructure, KGFn, KGOp, KGQSym, KGSym
from . import ir
from .checker import (_arg_span, _args, _body, _is_chain, _is_fn_literal, _is_plain_str, _locals, _name, _params,
                      validate_global)
from .diagnostics import StaticDiagnostic
from .optable import lookup
from .rules import RULES
from . import fixed as fx
from .signature import INT_RANGE, MAX_RANK, Ty, is_fixed, parse_type

__all__ = ["lower"]

INT = Ty("int")
FLOAT = Ty("float")
VOID = Ty("void")
STR = Ty("str")

_MAX_PASSES = 6


def _is_int(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _promote(a, b):
    if is_fixed(a):
        return a
    if is_fixed(b):
        return b
    return "float" if "float" in (a, b) else "int"


class _Var:
    __slots__ = ("kind", "index", "ty", "hint", "name")

    def __init__(self, kind, index, ty, name, hint=None):
        self.kind = kind        # "param" or "local"
        self.index = index
        self.ty = ty
        self.name = name
        self.hint = hint


class _Frame:
    def __init__(self, parent, fname, source, n_own):
        self.parent = parent
        self.fname = fname
        self.source = source
        self.vars = {}
        self.locals = []        # _Var of kind local, by slot
        self.captures = []      # (name, node in the parent's frame)
        self.n_own = n_own
        self.cap_tys = []
        self.n_literals = 0

    def new_local(self, name, ty, hint=None):
        v = _Var("local", len(self.locals), ty, name, hint)
        self.locals.append(v)
        return v


def _assigned(stmts):
    """Names a function body assigns (not looking into the function literals written in it)."""
    out = set()

    def go(n):
        if isinstance(n, (KGCond, KGExprArray)) or type(n) is list:
            for q in n:
                go(q)
            return
        if not isinstance(n, KGFn) or isinstance(n, KGSym):
            return
        if _is_chain(n):
            operand = n.a[-1]
            for q in (operand if type(operand) is list else [operand]):
                go(q)
            return
        if isinstance(n.a, KGOp):
            args = n.args if type(n.args) is list else [n.args]
            if n.a.a == "::" and len(args) == 2:
                left = args[0]
                for t in (left if type(left) is KGDestructure else [left] if isinstance(left, KGSym) else []):
                    out.add(_name(t))
                go(args[1])
                return
            for q in args:
                go(q)
            return
        if _is_fn_literal(n):
            return
        for q in _args(n):
            go(q)

    for s in stmts:
        go(s)
    return out


class _Lower:
    def __init__(self, report, values, name, int_bits):
        self.report = report
        self.values = values
        self.name = name
        self.int_bits = int_bits
        self.lo, self.hi = INT_RANGE[int_bits]
        self.memo = {}
        self.gdefs = {}
        self.used_externs = {}
        self.src = report.source

    # -- errors ---------------------------------------------------------------
    def fail(self, rule, sp, message):
        assert rule in RULES, rule
        d = StaticDiagnostic(rule, tuple(sp) if sp is not None else (0, 0), self.src, message, RULES[rule].hint)
        e = d.error()
        e.diagnostics = [d]
        raise e

    # -- constants --------------------------------------------------------------
    def const(self, v, sp=None):
        if isinstance(v, np.ndarray):
            if v.dtype.kind == "f":
                return ir.Const(Ty("float", tuple(int(d) for d in v.shape)), sp, value=v.astype(np.float64))
            return ir.Const(Ty("int", tuple(int(d) for d in v.shape)), sp, value=v.astype(np.int64))
        if isinstance(v, (float, np.floating)):
            return ir.Const(FLOAT, sp, value=float(v))
        return ir.Const(INT, sp, value=int(v))

    def in_range(self, v):
        return self.lo <= v <= self.hi

    def conv(self, n, elem):
        """n as the element type elem (an int converts to float, and an int constant to fixed-point; the checker allowed no more)."""
        if is_fixed(elem) and n.ty.elem == "int":
            return self.to_fixed(n, elem)
        if n.ty.elem == elem or elem != "float" or n.ty.elem != "int":
            return n
        if isinstance(n, ir.Const):
            return self.const(np.asarray(n.value, dtype=np.float64) if n.ty.shape else float(n.value), n.span)
        return ir.Convert(Ty("float", n.ty.shape), n.span, a=n)

    def to_fixed(self, n, elem):
        """n (an int, a float or a fixed-point value) as fixed-point elem: a constant when n is one."""
        spec = fx.spec_of(elem)
        src = n.ty.elem
        if src == elem:
            return n
        if isinstance(n, ir.Const) and not isinstance(n.value, str):
            v = np.asarray(n.value)
            if is_fixed(src):
                raw = fx.convert(v, fx.spec_of(src), spec)
            elif src == "float":
                raw = fx.from_float(v, spec)
            else:
                raw = fx.from_int(v, spec)
            return ir.Const(Ty(elem, n.ty.shape), n.span, value=raw if n.ty.shape else int(raw))
        return ir.Convert(Ty(elem, n.ty.shape), n.span, a=n)

    def mk(self, elem, shape, sp):
        if any(d <= 0 for d in shape):
            self.fail("array-shape", sp, "the result would have a dimension of 0: every dimension is a positive integer")
        if len(shape) > MAX_RANK:
            self.fail("array-shape", sp, f"the result would have rank {len(shape)}; the rank is 1 to {MAX_RANK}")
        return Ty(elem, tuple(int(d) for d in shape))

    # -- globals ------------------------------------------------------------------
    def gdef(self, name):
        g = self.gdefs.get(name)
        if g is None:
            info = self.report.globals[name]
            ty, rule, message = validate_global(self.values[name], self.int_bits)
            if rule:
                self.fail(rule, info.span, f"{name}: {message}")
            v = self.values[name]
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            if is_fixed(ty.elem):
                v = np.asarray(v.raw if hasattr(v, "raw") else v, dtype=np.int64)
                v = v.copy() if ty.shape else int(v)
            elif ty.shape:
                v = np.asarray(v).astype(np.float64 if ty.elem == "float" else np.int64)
            else:
                v = v.item() if hasattr(v, "item") else v
                v = float(v) if ty.elem == "float" else int(v)
            g = self.gdefs[name] = ir.GlobalDef(name, ty, v, bool(info.assigned), param=bool(getattr(info, "param", False)))
        return g

    def global_ref(self, name, sp):
        g = self.gdef(name)
        if g.state:
            return ir.GlobalRef(g.ty, sp, name=name, state=True)
        if not g.ty.shape:
            if is_fixed(g.ty.elem):
                return ir.Const(g.ty, sp, value=int(g.value))
            return self.const(g.value, sp)
        return ir.GlobalRef(g.ty, sp, name=name, state=False, known=g.value if g.value.size <= 4096 else None)

    # -- functions ------------------------------------------------------------------
    def spec(self, name, arg_tys):
        key = (name, tuple(arg_tys))
        f = self.memo.get(key)
        if f is None:
            info = self.report.functions[name]
            saved = self.src
            self.src = info.source
            try:
                f, _ = self.literal(info.node, tuple(arg_tys), None, name, "fn", info.span)
            finally:
                self.src = saved
            self.memo[key] = f
        return f

    def literal(self, node, arg_tys, parent, name, kind, span):
        """Lower a function literal at these argument types. Returns (FnSpec, capture nodes in the parent's frame)."""
        params = _params(node)
        names = _locals(node)
        stmts, spans = _body(node)
        if len(params) != len(arg_tys):
            self.fail("arity", span, f"{name} takes {len(params)} argument(s); here it is given {len(arg_tys)}")
        assigned = _assigned(stmts)
        default = tuple(node.span) if node.span is not None else span
        carried = {}
        for _ in range(_MAX_PASSES):
            fr = _Frame(parent, name, self.src, len(params))
            pre = []
            for i, (p, t) in enumerate(zip(params, arg_tys)):
                if p in assigned:
                    lt = Ty("float" if carried.get(p) == "float" else t.elem, t.shape)
                    v = fr.new_local(p, lt)
                    fr.vars[p] = v
                    pre.append(ir.Assign(lt, default, target=ir.LocalRef(lt, default, slot=v.index, name=p),
                                         value=self.conv(ir.ParamRef(t, default, index=i, name=p), lt.elem)))
                else:
                    fr.vars[p] = _Var("param", i, t, p)
            for q in names:
                if q not in fr.vars:
                    fr.vars[q] = fr.new_local(q, None, carried.get(q))
            body = list(pre)
            sps = [None] * len(pre)
            for s, sp in zip(stmts, spans):
                body.append(self.ev(s, tuple(sp) if sp is not None else default, fr))
                sps.append(tuple(sp) if sp is not None else None)
            now = {v.name: v.ty.elem for v in fr.locals if v.ty is not None}
            if now == carried:
                break
            carried = now
        else:
            self.fail("shape-change", default, f"the types of the locals of {name} do not settle")
        result = body[-1].ty if body else VOID
        cap_tys = tuple(n.ty for _, n in fr.captures)
        cap_names = tuple(n for n, _ in fr.captures)
        f = ir.FnSpec(name=name, index=-1, kind=kind, param_tys=tuple(arg_tys) + cap_tys, n_own=len(params),
                      param_names=tuple(params) + cap_names,
                      locals=tuple(v.ty if v.ty is not None else INT for v in fr.locals),
                      local_names=tuple(v.name for v in fr.locals),
                      body=ir.Seq(result, default, stmts=tuple(body)), result_ty=result, span=span, source=self.src,
                      stmt_spans=tuple(sps))
        return f, tuple(n for _, n in fr.captures)

    def op_spec(self, glyph, arg_tys, sp):
        """An operator used as a verb, as a FnSpec whose body is the operator applied to its parameters."""
        ps = [ir.ParamRef(t, sp, index=i, name="xyz"[i]) for i, t in enumerate(arg_tys)]
        body = self.monad(glyph, ps[0], sp) if len(ps) == 1 else self.dyad(glyph, ps[0], ps[1], sp)
        return ir.FnSpec(name=glyph, index=-1, kind="op", param_tys=tuple(arg_tys), n_own=len(ps),
                         param_names=tuple("xyz"[:len(ps)]), locals=(), local_names=(),
                         body=ir.Seq(body.ty, sp, stmts=(body,)), result_ty=body.ty, span=sp, source=self.src)

    def extern_spec(self, decl, arg_tys, sp):
        ps = [ir.ParamRef(t, sp, index=i, name="xyz"[i] if i < 3 else f"a{i}") for i, t in enumerate(arg_tys)]
        body = self.extern_call(decl, ps, sp)
        return ir.FnSpec(name=decl.name, index=-1, kind="lambda", param_tys=tuple(arg_tys), n_own=len(ps),
                         param_names=tuple(p.name for p in ps), locals=(), local_names=(),
                         body=ir.Seq(body.ty, sp, stmts=(body,)), result_ty=body.ty, span=sp, source=self.src)

    def verb(self, v, arg_tys, sp, fr):
        """The verb of an adverb at these argument types: (FnSpec, captures)."""
        if isinstance(v, KGOp):
            if v.a in ("::", ":="):
                self.fail("unsupported-op", sp, f"{v.a} cannot be the verb of an adverb")
            return self.op_spec(v.a, arg_tys, sp), ()
        if isinstance(v, KGSym) and not isinstance(v, KGQSym):
            name = _name(v)
            if self.lookup_var(name, fr) is not None:
                self.fail("fn-value", sp, f"{name} is a variable used as the verb of an adverb")
            if name in self.report.functions:
                if self.report.functions[name].arity != len(arg_tys):
                    self.fail("arity", sp, f"{name} takes {self.report.functions[name].arity} argument(s); here it is given {len(arg_tys)}")
                return self.spec(name, arg_tys), ()
            if name in self.report.externs:
                return self.extern_spec(self.report.externs[name], arg_tys, sp), ()
            self.fail("undefined-name", sp, f"the function {name} is not defined")
        if _is_fn_literal(v):
            fr.n_literals += 1
            root = fr
            while root.parent is not None:
                root = root.parent
            if fr is not root:
                root.n_literals += 1
            return self.literal(v, tuple(arg_tys), fr, f"{root.fname}.fn{root.n_literals}", "lambda", sp)
        self.fail("fn-value", sp, "the verb of the adverb must be an operator, a function name or a function literal")

    # -- variables --------------------------------------------------------------------
    def lookup_var(self, name, fr):
        f = fr
        while f is not None:
            if name in f.vars:
                return f
            f = f.parent
        return None

    def read(self, name, sp, fr):
        v = fr.vars.get(name)
        if v is not None:
            if v.ty is None:
                self.fail("unset-local", sp, f"the local {name} is read before it is assigned")
            if v.kind == "param":
                return ir.ParamRef(v.ty, sp, index=v.index, name=name)
            return ir.LocalRef(v.ty, sp, slot=v.index, name=name)
        if fr.parent is not None and self.lookup_var(name, fr.parent) is not None:
            outer = self.read(name, sp, fr.parent)
            idx = fr.n_own + len(fr.captures)
            fr.captures.append((name, outer))
            fr.vars[name] = _Var("param", idx, outer.ty, name)
            return ir.ParamRef(outer.ty, sp, index=idx, name=name)
        if name in self.report.globals:
            return self.global_ref(name, sp)
        if name in self.report.functions or name in self.report.externs:
            self.fail("fn-value", sp, f"the function {name} is used as a value")
        self.fail("undefined-name", sp, f"{name} is not defined")

    def store(self, name, value, sp, fr):
        if value.ty.elem == "void":
            self.fail("void-value", sp, "the result of a void extern is assigned")
        v = fr.vars.get(name)
        if v is not None and v.kind == "local":
            if v.ty is None:
                v.ty = Ty("float" if v.hint == "float" else value.ty.elem, value.ty.shape)
            else:
                if v.ty.shape != value.ty.shape:
                    self.fail("shape-change", sp, f"{name} was {v.ty} and is assigned {value.ty}: the shape of a variable is fixed")
                if value.ty.elem == "float" and v.ty.elem == "int":
                    v.ty = Ty("float", v.ty.shape)      # the next pass starts from float
            return ir.Assign(v.ty, sp, target=ir.LocalRef(v.ty, sp, slot=v.index, name=name),
                             value=self.conv(value, v.ty.elem))
        if v is not None or self.lookup_var(name, fr) is not None:
            self.fail("closure-assign", sp, f"the nested function assigns {name}, a variable of the function it is written in")
        if name in self.report.globals:
            g = self.gdef(name)
            g.state = True
            if g.ty != value.ty:
                self.fail("shape-change", sp, f"the global {name} is {g.ty} and is assigned {value.ty}: "
                                              "the type and shape of a global are fixed")
            return ir.Assign(g.ty, sp, target=ir.GlobalRef(g.ty, sp, name=name, state=True), value=value)
        self.fail("undefined-name", sp, f"{name} is assigned but is neither a local nor a global")

    # -- expressions ------------------------------------------------------------------------
    def ev(self, n, sp, fr):
        nsp = getattr(n, "span", None)
        if nsp is not None:
            sp = tuple(nsp)
        if isinstance(n, KGQSym):
            self.fail("symbol", sp, f"the symbol :{_name(n)} is not a run-time value")
        if isinstance(n, KGSym):
            return self.read(_name(n), sp, fr)
        if isinstance(n, (KGChar, str)):
            self.fail("string", sp, "a string is not a run-time value")
        if isinstance(n, (bool, np.bool_)):
            return self.const(int(n), sp)
        if isinstance(n, (int, np.integer)):
            return self.int_literal(int(n), sp)
        if isinstance(n, (float, np.floating)):
            return self.const(float(n), sp)
        if isinstance(n, np.ndarray):
            if n.dtype.kind not in "iubf" or n.size == 0:
                self.fail("list-literal", sp, "the list is not a rectangular list of numbers")
            return self.const(n, sp)
        if isinstance(n, KGCond):
            return self.cond(n, sp, fr)
        if isinstance(n, KGExprArray):
            return self.elist(n, sp, fr)
        if not isinstance(n, KGFn):
            self.fail("unsupported-op", sp, f"a {type(n).__name__} is not part of the profile")
        a = n.a
        if _is_chain(n):
            return self.adverb(n, sp, fr)
        if isinstance(a, KGOp):
            if type(n.args) is not list:
                if a.a == "-" and _is_int(n.args):
                    return self.int_literal(-n.args, sp)
                return self.monad(a.a, self.ev(n.args, _arg_span(n, 0, sp), fr), sp)
            if len(n.args) != 2:
                self.fail("unsupported-op", sp, f"the operator {a.a} is applied to {len(n.args)} operands")
            if a.a == "::":
                return self.assign(n, sp, fr)
            if a.a == ":=":
                return self.amend(n, sp, fr)
            right = self.ev(n.args[1], _arg_span(n, 1, sp), fr)     # the right operand is evaluated first
            left = self.ev(n.args[0], _arg_span(n, 0, sp), fr)
            return self.dyad(a.a, left, right, sp)
        if _is_fn_literal(n):
            self.fail("fn-value", sp, "a function literal is used as a value")
        return self.call(n, sp, fr)

    def int_literal(self, v, sp):
        if not self.in_range(v):
            self.fail("int-range", sp, f"{v} is outside the range of a {self.int_bits}-bit kg_int")
        return self.const(v, sp)

    def cond(self, n, sp, fr):
        if len(n) != 3:
            self.fail("unsupported-op", sp, "a conditional has a condition and two branches")
        c = self.ev(n[0], _arg_span(n, 0, sp), fr)
        if c.ty.shape or c.ty.elem == "void":
            self.fail("cond-scalar", _arg_span(n, 0, sp), f"the condition is {c.ty}; it must be a scalar")
        t = self.ev(n[1], _arg_span(n, 1, sp), fr)
        e = self.ev(n[2], _arg_span(n, 2, sp), fr)
        if t.ty.elem == "void" and e.ty.elem == "void":
            return ir.Cond(VOID, sp, c=c, t=t, e=e)
        if t.ty.elem == "void" or e.ty.elem == "void" or t.ty.shape != e.ty.shape:
            self.fail("cond-shape", sp, f"the branches of the conditional are {t.ty} and {e.ty}; they must have the same shape")
        elem = _promote(t.ty.elem, e.ty.elem)
        return ir.Cond(Ty(elem, t.ty.shape), sp, c=c, t=self.conv(t, elem), e=self.conv(e, elem))

    def elist(self, n, sp, fr):
        items = [self.ev(q, _arg_span(n, i, sp), fr) for i, q in enumerate(n)]
        if not items:
            self.fail("array-shape", sp, "an empty list has a dimension of 0")
        elem = "int"
        for it in items:
            if it.ty.elem == "void":
                self.fail("void-value", sp, "a list element is the result of a void extern")
            if it.ty.shape != items[0].ty.shape:
                self.fail("list-literal", sp, f"the elements of the list are {items[0].ty} and {it.ty}; they must have the same shape")
            elem = _promote(elem, it.ty.elem)
        ty = self.mk(elem, (len(items),) + tuple(items[0].ty.shape), sp)
        return ir.MakeList(ty, sp, items=tuple(self.conv(it, elem) for it in items))

    # -- assignment ---------------------------------------------------------------------------
    def assign(self, n, sp, fr):
        left, right = n.args
        lsp = _arg_span(n, 0, sp)
        if _is_fn_literal(right):
            self.fail("fn-value", _arg_span(n, 1, sp), "a function is assigned to a variable inside a function")
        v = self.ev(right, _arg_span(n, 1, sp), fr)
        if type(left) is KGDestructure:
            names = [_name(q) for q in left]
            if not v.ty.shape or v.ty.shape[0] != len(names):
                self.fail("shape-mismatch", sp, f"cannot destructure {v.ty} into {len(names)} name(s)")
            hidden = fr.new_local(f"%d{len(fr.locals)}", v.ty)
            ref = ir.LocalRef(v.ty, sp, slot=hidden.index, name=hidden.name)
            stmts = [ir.Assign(v.ty, sp, target=ref, value=v)]
            item = Ty(v.ty.elem, v.ty.shape[1:])
            for i, q in enumerate(names):
                stmts.append(self.store(q, ir.Index(item, sp, a=ref, i=self.const(i, sp), checked=True), lsp, fr))
            stmts.append(ref)
            return ir.Seq(v.ty, sp, stmts=tuple(stmts))
        if not isinstance(left, KGSym) or isinstance(left, KGQSym):
            self.fail("strict-syntax", lsp, "the left side of :: must be a name")
        return self.store(_name(left), v, lsp, fr)

    # -- calls ------------------------------------------------------------------------------------
    def call(self, n, sp, fr):
        callee = n.a
        args = _args(n)
        if any(q is None for q in args):
            self.fail("projection", sp, "the call leaves an argument out: that is a projection, a function value")
        if not isinstance(callee, KGSym) or isinstance(callee, KGQSym):
            self.fail("fn-value", sp, "a function literal is called in place")
        name = _name(callee)
        if self.lookup_var(name, fr) is not None:
            self.fail("fn-value", sp, f"{name} is a variable called as a function")
        if name in self.report.functions:
            f = self.report.functions[name]
            if len(args) != f.arity:
                self.fail("projection" if len(args) < f.arity else "arity", sp, f"{name} takes {f.arity} argument(s), got {len(args)}")
            vals = [self.ev(q, _arg_span(n, i, sp), fr) for i, q in enumerate(args)]
            for v in vals:
                if v.ty.elem == "void":
                    self.fail("void-value", sp, f"an argument of {name} is the result of a void extern")
            g = self.spec(name, [v.ty for v in vals])
            return ir.Call(g.result_ty, sp, fn=g, args=tuple(vals))
        if name in self.report.externs:
            decl = self.report.externs[name]
            sig = decl.signature
            if len(args) != sig.arity:
                self.fail("extern-call", sp, f"{name} is declared \"{sig}\" and takes {sig.arity} argument(s), got {len(args)}")
            vals = []
            for i, (q, ty) in enumerate(zip(args, sig.params)):
                if ty.elem == "str":
                    if not _is_plain_str(q):
                        self.fail("extern-call", _arg_span(n, i, sp), f"argument {i + 1} of {name} is declared str: it must be a string literal")
                    vals.append(ir.Const(STR, _arg_span(n, i, sp), value=str(q)))
                else:
                    vals.append(self.ev(q, _arg_span(n, i, sp), fr))
            return self.extern_call(decl, vals, sp)
        if name == ".f":
            self.fail("recursion", sp, ".f is the running function: calling it is recursion")
        if name == ".fixed":
            ty = parse_type(args[0])
            v = self.ev(args[1], _arg_span(n, 1, sp), fr)
            return self.to_fixed(v, ty.elem)
        if name == ".float":
            v = self.ev(args[0], _arg_span(n, 0, sp), fr)
            if v.ty.elem == "float":
                return v
            if isinstance(v, ir.Const) and not is_fixed(v.ty.elem):
                return self.const(np.asarray(v.value, dtype=np.float64) if v.ty.shape else float(v.value), sp)
            return ir.Convert(Ty("float", v.ty.shape), sp, a=v)
        if name.startswith("."):
            self.fail("sys-fn", sp, f"{name} is a system function; it does not exist in a compiled program")
        self.fail("undefined-name", sp, f"the function {name} is not defined")

    def extern_call(self, decl, vals, sp):
        sig = decl.signature
        out = []
        for i, (v, ty) in enumerate(zip(vals, sig.params)):
            if ty.elem == "str":
                if v.ty.elem != "str":
                    self.fail("extern-call", sp, f"argument {i + 1} of {decl.name} is declared str: it must be a string literal")
                out.append(v)
                continue
            if v.ty.shape != ty.shape or (v.ty.elem == "float" and ty.elem == "int") or v.ty.elem in ("void", "str"):
                self.fail("extern-call", sp, f"argument {i + 1} of {decl.name} is declared {ty}, got {v.ty}")
            out.append(self.conv(v, ty.elem))
        self.used_externs[decl.name] = decl
        return ir.ExternCall(sig.result, sp, ext=decl, args=tuple(out))

    # -- adverbs ----------------------------------------------------------------------------------------
    def adverb(self, n, sp, fr):
        chain = n.a
        verb = chain[0].a
        advs = [q.a for q in chain[1:-1]]
        operand = chain[-1]
        dyadic = type(operand) is list
        vsp = tuple(n.op_span) if n.op_span is not None else sp
        if len(advs) != 1:
            self.fail("unsupported-op", vsp, f"the adverb chain {''.join(advs)} is not in the profile")
        glyph = advs[0]
        row = lookup(glyph, "adverb2" if dyadic else "adverb1")
        if row is None or row.status != "allowed":
            self.fail(row.rule if row is not None else "unsupported-op", vsp, f"the adverb {glyph} is not in the profile")
        kind = row.id
        if kind in ("while", "iterate"):
            init = self.ev(operand[1], _arg_span(n, 1, sp), fr)
            if init.ty.elem == "void":
                self.fail("void-value", sp, "the starting value is the result of a void extern")
            if kind == "iterate":
                count = self.ev(operand[0], _arg_span(n, 0, sp), fr)
                if count.ty != INT:
                    self.fail("operand", _arg_span(n, 0, sp), f"the count of Iterate is {count.ty}; it must be an int scalar")
                pred, pcap = None, ()
            else:
                count = None
                pred, pcap = self.verb(operand[0], [init.ty], _arg_span(n, 0, sp), fr)
                if pred.result_ty.shape or pred.result_ty.elem == "void":
                    self.fail("cond-scalar", _arg_span(n, 0, sp), f"the predicate of While returns {pred.result_ty}; it must be a scalar")
            f, cap = self.verb(verb, [init.ty], vsp, fr)
            if f.result_ty != init.ty:
                self.fail("loop-state", sp, f"the state starts as {init.ty} and the verb returns {f.result_ty}; "
                                            "its type and shape must not change")
            return ir.Loop(init.ty, sp, kind=kind, fn=f, init=init, count=count, pred=pred, captures=cap, pred_captures=pcap)
        if dyadic:
            b = self.ev(operand[1], _arg_span(n, 1, sp), fr)
            a = self.ev(operand[0], _arg_span(n, 0, sp), fr)
        else:
            b = self.ev(operand, _arg_span(n, 0, sp), fr)
            a = None
        for v in (a, b):
            if v is not None and v.ty.elem == "void":
                self.fail("void-value", sp, f"an operand of {glyph} ({row.name}) is the result of a void extern")
        if not b.ty.shape:
            self.fail("operand", sp, f"{glyph} ({row.name}) needs an array; the operand is {b.ty}")
        item = Ty(b.ty.elem, b.ty.shape[1:])
        count = b.ty.shape[0]

        def stacked(r, items):
            if r.elem == "void":
                return VOID
            return self.mk(r.elem, (items,) + tuple(r.shape), sp)

        if kind == "each":
            f, cap = self.verb(verb, [item], vsp, fr)
            return ir.Map(stacked(f.result_ty, count), sp, fn=f, a=b, captures=cap)
        if kind == "each2":
            if not a.ty.shape:
                self.fail("operand", sp, f"{glyph} ({row.name}) needs two arrays; the left operand is {a.ty}")
            if a.ty.shape[0] != count:
                self.fail("shape-mismatch", sp, f"{glyph} ({row.name}) pairs the items of {a.ty} and {b.ty}: the lengths differ")
            f, cap = self.verb(verb, [Ty(a.ty.elem, a.ty.shape[1:]), item], vsp, fr)
            return ir.Map2(stacked(f.result_ty, count), sp, fn=f, a=a, b=b, captures=cap)
        if kind == "each-pair":
            if count < 2:
                self.fail("array-shape", sp, f"{glyph} ({row.name}) of {b.ty} has no pairs: the result would be empty")
            f, cap = self.verb(verb, [item, item], vsp, fr)
            return ir.EachPair(stacked(f.result_ty, count - 1), sp, fn=f, a=b, captures=cap)
        # over and scan: the accumulator is the fixed point of acc = f(acc, item), with one promotion
        acc = a.ty if a is not None else item
        if a is not None and acc.elem == "int" and is_fixed(item.elem):
            acc = Ty(item.elem, acc.shape)
        f, cap = self.verb(verb, [acc, item], vsp, fr)
        for _ in range(2):
            r = f.result_ty
            if r.elem == "void":
                self.fail("void-value", sp, f"the verb of {glyph} ({row.name}) returns nothing")
            if r.shape != acc.shape:
                self.fail("loop-state", sp, f"{glyph} ({row.name}): the accumulator is {acc} and the verb returns {r}; "
                                            "its shape must not change")
            if r.elem == "float" and acc.elem == "int":
                acc = Ty("float", acc.shape)
                f, cap = self.verb(verb, [acc, item], vsp, fr)
                continue
            break
        acc = Ty(_promote(acc.elem, f.result_ty.elem), acc.shape)
        seed = self.conv(a, acc.elem) if a is not None else None
        if kind in ("over", "over-seed"):
            return ir.Reduce(acc, sp, fn=f, a=b, seed=seed, captures=cap)
        ty = self.mk(acc.elem, (count + (1 if a is not None else 0),) + tuple(acc.shape), sp)
        return ir.Scan(ty, sp, fn=f, a=b, seed=seed, captures=cap, acc_ty=acc)

    # -- operators ----------------------------------------------------------------------------------------
    def row(self, glyph, valence, sp):
        row = lookup(glyph, valence)
        if row is None:
            self.fail("unsupported-op", sp, f"the operator {glyph} is not in the profile")
        if row.status != "allowed":
            self.fail(row.rule, sp, f"{glyph} ({row.name}): {row.note}")
        return row

    def count(self, a, sp, what):
        """The constant int count of take, drop or enumerate."""
        if a.ty != INT:
            self.fail("operand", sp, f"the count of {what} is {a.ty}; it must be an int scalar")
        c = a.cval
        if not _is_int(c) or ir.effects(a):
            self.fail("const-count", sp, f"the count of {what} is not a compile-time constant: "
                                         "the result length would depend on run-time data")
        return int(c)

    def monad(self, glyph, a, sp):
        row = self.row(glyph, "monad", sp)
        if a.ty.elem == "void":
            self.fail("void-value", sp, f"the operand of {glyph} ({row.name}) is the result of a void extern")
        k = row.id
        shape = a.ty.shape
        if k == "neg":
            c = a.cval
            if is_fixed(a.ty.elem):
                return ir.Unary(a.ty, sp, op="neg", a=a)
            if isinstance(a, ir.Const) and not shape and (a.ty.elem == "float" or self.in_range(-c)):
                return self.const(-c, sp)
            return ir.Unary(a.ty, sp, op="neg", a=a, known=None if c is None else -c)
        if k == "floor":
            return a if a.ty.elem == "int" else ir.Unary(Ty("int", shape), sp, op="floor", a=a)
        if k == "recip":
            return ir.Unary(Ty("float", shape), sp, op="recip", a=self.conv(a, "float"))
        if k == "not":
            return ir.Unary(Ty("int", shape), sp, op="not", a=a)
        if k in ("size", "shape"):
            if not shape:
                self.fail("size-of-atom", sp, f"{glyph} ({row.name}) of {a.ty}: the operand is an atom")
            c = self.const(int(shape[0]) if k == "size" else np.array(shape, dtype=np.int64), sp)
            return ir.Seq(c.ty, sp, stmts=(a, c)) if ir.effects(a) else c
        if k == "first":
            return ir.Index(Ty(a.ty.elem, shape[1:]), sp, a=a, i=self.const(0, sp), checked=True) if shape else a
        if k == "reverse":
            return ir.Reverse(a.ty, sp, a=a) if shape else a
        if k == "transpose":
            if len(shape) != 2:
                self.fail("operand", sp, f"+ (Transpose) needs a rank-2 array; the operand is {a.ty}")
            return ir.Transpose(Ty(a.ty.elem, (shape[1], shape[0])), sp, a=a)
        if k == "enumerate":
            c = self.count(a, sp, "!n (Enumerate)")
            if c <= 0:
                self.fail("array-shape", sp, f"!{c} would be empty: n must be positive")
            return ir.Iota(Ty("int", (c,)), sp, n=c)
        if k == "list":
            ty = self.mk(a.ty.elem, (1,) + tuple(shape), sp)
            return ir.Reshape(ty, sp, a=a) if shape else ir.MakeList(ty, sp, items=(a,))
        self.fail("unsupported-op", sp, f"{glyph} ({row.name}) is not emitted")

    def prefix(self, a, b, sp, row):
        sa, sb = a.ty.shape, b.ty.shape
        if not sa:
            return sb
        if not sb:
            return sa
        short, long_ = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
        if long_[:len(short)] != short:
            self.fail("shape-mismatch", sp, f"cannot apply {row.glyph} ({row.name}) to {a.ty} and {b.ty}")
        return long_

    def fold(self, k, x, y):
        """The value of an int operation on two int constants, or None when it is not folded."""
        if k == "add":
            return x + y
        if k == "sub":
            return x - y
        if k == "mul":
            return x * y
        if k == "max":
            return max(x, y)
        if k == "min":
            return min(x, y)
        if k in ("idiv", "rem") and y != 0:
            q = abs(x) // abs(y)
            q = q if (x < 0) == (y < 0) else -q
            return q if k == "idiv" else x - q * y
        if k == "pow" and 0 <= y <= 64:
            return x ** y
        if k == "lt":
            return int(x < y)
        if k == "gt":
            return int(x > y)
        if k == "eq":
            return int(x == y)
        return None

    def dyad(self, glyph, a, b, sp):
        row = self.row(glyph, "dyad", sp)
        if a.ty.elem == "void" or b.ty.elem == "void":
            self.fail("void-value", sp, f"an operand of {glyph} ({row.name}) is the result of a void extern")
        k = row.id
        ea, eb = a.ty.elem, b.ty.elem
        if row.shape_rule == "prefix":
            shape = self.prefix(a, b, sp, row)
            ca, cb = a.cval, b.cval
            if _is_int(cb) and cb == 0 and ea == "int" and eb == "int" and (k in ("idiv", "rem") or (k == "div" and ca is not None)):
                self.fail("div-zero", sp, f"{glyph} ({row.name}) divides an int by the constant 0")
            rule = row.type_rule
            fxe = ea if is_fixed(ea) else eb if is_fixed(eb) else None
            if fxe is not None:
                elem, work = ("int" if rule == "int" else fxe), fxe
                if len(shape) > MAX_RANK:
                    self.mk(elem, shape, sp)
                return ir.Binary(Ty(elem, shape), sp, op=k, a=self.conv(a, work), b=self.conv(b, work))
            if rule == "promote":
                elem = work = _promote(ea, eb)
            elif rule == "float":
                elem = work = "float"
            elif rule == "int":
                elem, work = "int", _promote(ea, eb)
            elif rule == "int-only":
                if "float" in (ea, eb):
                    self.fail("operand", sp, f"{glyph} ({row.name}) needs int operands; they are {a.ty} and {b.ty}")
                elem = work = "int"
            else:       # power
                elem = work = "int" if (ea == "int" and eb == "int" and _is_int(cb) and cb >= 0) else "float"
            if len(shape) > MAX_RANK:
                self.mk(elem, shape, sp)
            if _is_int(ca) and _is_int(cb) and elem == "int" and not shape and isinstance(a, ir.Const) and isinstance(b, ir.Const):
                v = self.fold(k, int(ca), int(cb))
                if v is not None and self.in_range(v):
                    return self.const(v, sp)
            return ir.Binary(Ty(elem, shape), sp, op=k, a=self.conv(a, work), b=self.conv(b, work))
        if k == "match":
            if a.ty.shape != b.ty.shape:
                self.fail("shape-mismatch", sp, f"~ (Match) compares {a.ty} with {b.ty}: the shapes differ")
            work = _promote(ea, eb)
            return ir.Match(INT, sp, a=self.conv(a, work), b=self.conv(b, work))
        if k == "index":
            if not a.ty.shape:
                self.fail("operand", sp, f"cannot index {a.ty}: @ (Index) needs an array on the left")
            if eb != "int":
                self.fail("operand", sp, f"the index is {b.ty}; an index must be an int")
            n = a.ty.shape[0]
            b, checked = self.const_index(b, n, sp, f"{a.ty}")
            if not b.ty.shape:
                return ir.Index(Ty(ea, a.ty.shape[1:]), sp, a=a, i=b, checked=checked)
            return ir.Gather(self.mk(ea, tuple(b.ty.shape) + tuple(a.ty.shape[1:]), sp), sp, a=a, idx=b, checked=checked)
        if k == "take":
            c = self.count(a, sp, "n#a (Take)")
            if len(b.ty.shape) != 1:
                self.fail("take-rank", sp, f"n#a (Take) from {b.ty}: take works on rank-1 arrays only")
            if c == 0:
                self.fail("array-shape", sp, "0#a would be empty: every dimension is a positive integer")
            return ir.Take(Ty(eb, (abs(c),)), sp, a=b, n=c)
        if k == "drop":
            c = self.count(a, sp, "n_a (Drop)")
            if not b.ty.shape:
                self.fail("operand", sp, f"n_a (Drop) from {b.ty}: the right operand must be an array")
            left = b.ty.shape[0] - abs(c)
            if left <= 0:
                self.fail("array-shape", sp, f"dropping {abs(c)} from {b.ty} leaves nothing")
            return b if c == 0 else ir.Drop(Ty(eb, (left,) + tuple(b.ty.shape[1:])), sp, a=b, n=c)
        if k == "join":
            elem = _promote(ea, eb)
            sa, sb = tuple(a.ty.shape), tuple(b.ty.shape)
            a, b = self.conv(a, elem), self.conv(b, elem)
            if not sa and not sb:
                # a dyad evaluates its right operand first: a Join node keeps that order
                return ir.Join(Ty(elem, (2,)), sp, a=a, b=b)
            if not sa or not sb:
                arr = sb if not sa else sa
                if len(arr) != 1:
                    self.fail("shape-mismatch", sp, f"joining {a.ty} with {b.ty} would give a ragged list")
                return ir.Join(Ty(elem, (arr[0] + 1,)), sp, a=a, b=b)
            if len(sa) != len(sb) or sa[1:] != sb[1:]:
                self.fail("shape-mismatch", sp, f"joining {a.ty} with {b.ty} would give a ragged list")
            return ir.Join(Ty(elem, (sa[0] + sb[0],) + sa[1:]), sp, a=a, b=b)
        if k == "rotate":
            if a.ty != INT:
                self.fail("operand", sp, f"the count of n:+a (Rotate) is {a.ty}; it must be an int scalar")
            if len(b.ty.shape) != 1:
                self.fail("operand", sp, f"n:+a (Rotate) of {b.ty}: the profile rotates rank-1 arrays only")
            return ir.Rotate(b.ty, sp, a=b, n=a)
        if k == "reshape":
            c = a.cval
            if ea != "int":
                self.fail("operand", sp, f"the shape of s:^a (Reshape) is {a.ty}; it must hold ints")
            if c is None or ir.effects(a) or len(a.ty.shape) > 1:
                self.fail("const-count", sp, "the shape of s:^a (Reshape) is not a compile-time constant list of ints")
            if not a.ty.shape and len(b.ty.shape) > 1:
                self.fail("operand", sp, f"n:^a (Reshape) of {b.ty} with a scalar n: write the shape as a list, [n]:^a")
            dims = tuple(int(d) for d in np.asarray(c).reshape(-1))
            return ir.Reshape(self.mk(eb, dims, sp), sp, a=b)
        self.fail("unsupported-op", sp, f"{glyph} ({row.name}) cannot be used here")

    def const_index(self, b, n, sp, what):
        """Range-check a constant index against length n and make it non-negative. Returns (node, checked)."""
        c = b.cval
        if c is None:
            return b, False
        idx = np.asarray(c)
        if idx.size and (int(idx.min()) < -n or int(idx.max()) >= n):
            worst = int(idx.max()) if int(idx.max()) >= n else int(idx.min())
            self.fail("index-range", sp, f"index {worst} is out of range for {what}")
        if isinstance(b, ir.Const):
            if not b.ty.shape:
                return self.const(int(c) % n, b.span), True
            return self.const(np.mod(idx, n), b.span), True
        return b, bool(idx.min() >= 0)

    def amend(self, n, sp, fr):
        target, pair = n.args
        psp = _arg_span(n, 1, sp)
        if not (isinstance(pair, KGFn) and not isinstance(pair, KGCall) and isinstance(pair.a, KGOp) and pair.a.a == ","
                and type(pair.args) is list and len(pair.args) == 2):
            self.fail("amend-form", psp, "the right side of := (Amend) must be written v,i: one value, then the index or indices")
        if pair.span is not None:
            psp = tuple(pair.span)
        idx = self.ev(pair.args[1], _arg_span(pair, 1, psp), fr)
        val = self.ev(pair.args[0], _arg_span(pair, 0, psp), fr)
        arr = self.ev(target, _arg_span(n, 0, sp), fr)
        if "void" in (idx.ty.elem, val.ty.elem, arr.ty.elem):
            self.fail("void-value", sp, "an operand of := (Amend) is the result of a void extern")
        if len(arr.ty.shape) != 1:
            self.fail("amend-form", sp, f":= (Amend) of {arr.ty}: the profile amends rank-1 arrays only")
        if val.ty.shape:
            self.fail("amend-form", psp, f"the value of := (Amend) is {val.ty}; it must be one scalar")
        if idx.ty.elem != "int":
            self.fail("operand", psp, f"the index of := (Amend) is {idx.ty}; an index must be an int")
        if len(idx.ty.shape) > 1:
            self.fail("amend-form", psp, f"the index of := (Amend) is {idx.ty}; it must be a scalar or a rank-1 list")
        idx, checked = self.const_index(idx, arr.ty.shape[0], psp, f"{arr.ty}")
        elem = _promote(arr.ty.elem, val.ty.elem)
        return ir.Amend(Ty(elem, arr.ty.shape), sp, a=self.conv(arr, elem), value=self.conv(val, elem), idx=idx, checked=checked)

    # -- the program ------------------------------------------------------------------------------------------
    def program(self):
        rep = self.report
        entries = []
        seen = set()
        per_fn = {}
        for e in rep.entries:
            key = (e.name, str(e.signature))
            if key in seen:
                continue
            seen.add(key)
            per_fn.setdefault(e.name, []).append(e)
        for e in rep.entries:
            if (e.name, str(e.signature)) not in seen:
                continue
            seen.discard((e.name, str(e.signature)))
            sig = e.signature
            self.src = rep.functions[e.name].source
            f = self.spec(e.name, sig.params)
            r, want = f.result_ty, sig.result
            if want.elem != "void":
                if r.elem == "void" or r.shape != want.shape or (r.elem == "float" and want.elem == "int") \
                        or ((is_fixed(r.elem) or is_fixed(want.elem)) and r.elem != want.elem):
                    self.src = e.source
                    self.fail("entry", e.span, f"entry {e.name} is declared to return {want}, but the function returns {r}")
            c_name = e.name.replace(".", "_")
            if len(per_fn[e.name]) > 1:
                c_name += "__" + _suffix(sig)
            entries.append(ir.Entry(e, f, c_name))
        names = [x.c_name for x in entries]
        for x in entries:
            if names.count(x.c_name) > 1:
                self.src = x.decl.source
                self.fail("entry", x.decl.span, f"two entries have the C name {x.c_name}")
        # order the specialisations callees first and number them
        order = []
        done = set()

        def visit(f):
            if id(f) in done:
                return
            done.add(id(f))
            for n in ir.walk(f.body):
                for g in ir.fnspecs(n):
                    visit(g)
            order.append(f)

        for x in entries:
            visit(x.fn)
        for i, f in enumerate(order):
            f.index = i
        used = set()
        for f in order:
            for n in ir.walk(f.body):
                if isinstance(n, ir.GlobalRef):
                    used.add(n.name)
        gl = []
        for name in rep.globals:
            g = self.gdefs.get(name)
            if g is None or (g.ty.shape and name not in used):
                continue
            gl.append(g)
        externs = [d for name, d in rep.externs.items()
                   if any(isinstance(n, ir.ExternCall) and n.ext is d for f in order for n in ir.walk(f.body))]
        return ir.Program(self.name, gl, order, entries, externs, self.int_bits, rep.source)


def _suffix(sig):
    """The C name suffix that tells apart the entries of one function: the parameter types, f256 or i_f3x2."""
    parts = []
    for p in sig.params:
        parts.append(p.elem[0] + "x".join(str(d) for d in p.shape))
    return "_".join(parts) if parts else "void"


def lower(report, values, name="prog", int_bits=32):
    """
    Lower a checked program to the typed IR. ``report`` must be the result of
    check_source(..., globals=values) with ``report.ok`` and no unresolved
    global. Raises KlongStaticError (with ``rule`` and ``diagnostics``).
    """
    if not report.ok:
        report.raise_first()
    if report.unresolved:
        raise ValueError("lower() needs the value of every global; missing " + ", ".join(report.unresolved))
    if not report.entries:
        d = StaticDiagnostic("entry", (0, 0), report.source, "the program declares no .entry: there is nothing to compile",
                             "mark the functions the host calls: .entry(\"step\";\"float -> float\")")
        e = d.error()
        e.diagnostics = [d]
        raise e
    return _Lower(report, values, name, int_bits).program()
