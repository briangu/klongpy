"""
An evaluator of the typed IR: the interpreter stand-in for fixed-point programs (ticket 113).

KlongPy has no fixed-point numbers, so a program that uses ``fixed<W,I>`` cannot run as plain
Klong with exact results. ``klongpy.static.call_entry`` runs such a program here instead: the
IR that the compiler lowers (klongpy/static/infer.py) is evaluated on numpy arrays, with fixed-point
values as scaled int64 (klongpy/static/fixed.py) and every integer wrapped at the width of kg_int,
which is what compiled code does. The semantics of each node are those of klong_rt.h; where compiled
code traps (an index out of range, an integer or fixed-point division by zero) this raises Trap.

    ev = IrEvaluator(program)
    ev.state                                   {name: value} the state globals (and parameters)
    ev.run(entry, [arg, ...]) -> value         values are numpy arrays or numbers: int64, float64, or raw int64
"""
import numpy as np

from . import fixed as fx
from . import ir
from .signature import is_fixed

__all__ = ["IrEvaluator", "Trap", "zero_of"]

TRAP_BOUNDS, TRAP_DIV0 = 1, 2


class Trap(Exception):
    """What compiled code reports with kg_trap: code 1 an index out of range, 2 a division by zero."""
    def __init__(self, code, message):
        Exception.__init__(self, message)
        self.code = code


def _dtype(elem):
    return np.float64 if elem == "float" else np.int64


def zero_of(ty):
    return np.zeros(ty.shape, dtype=_dtype(ty.elem))


class _Env:
    __slots__ = ("params", "locals")

    def __init__(self, params, locals_):
        self.params = params
        self.locals = locals_


class IrEvaluator:
    def __init__(self, program):
        self.program = program
        self.bits = program.int_bits
        self.consts = {g.name: self._val(g.ty, g.value) for g in program.globals if not g.state}
        self.initial = {g.name: self._val(g.ty, g.value) for g in program.globals if g.state}
        self.state = {k: v.copy() for k, v in self.initial.items()}

    @staticmethod
    def _val(ty, v):
        return np.array(np.asarray(v, dtype=_dtype(ty.elem)))

    def reset(self):
        self.state = {k: v.copy() for k, v in self.initial.items()}

    # -- integers ----------------------------------------------------------------
    def wrap(self, a):
        a = np.asarray(a, dtype=np.int64)
        if self.bits == 64:
            return a
        h, m = np.int64(1 << (self.bits - 1)), np.int64((1 << self.bits) - 1)
        return ((a + h) & m) - h

    def cast(self, v, src, dst):
        """A value of element type src as dst (the conversions of the IR)."""
        if src == dst:
            return v
        if dst == "float":
            if is_fixed(src):
                return fx.to_float(v, fx.spec_of(src))
            return np.asarray(v, dtype=np.float64)
        if is_fixed(dst):
            d = fx.spec_of(dst)
            if is_fixed(src):
                return fx.convert(v, fx.spec_of(src), d)
            if src == "float":
                return fx.from_float(v, d)
            return fx.from_int(v, d)
        return v

    # -- running ---------------------------------------------------------------------
    def run(self, entry, args):
        fn = entry.fn if hasattr(entry, "fn") else entry
        vals = [np.array(np.asarray(a, dtype=_dtype(t.elem))) for a, t in zip(args, fn.param_tys)]
        return self.call(fn, vals)

    def call(self, fn, args):
        env = _Env(list(args), [zero_of(t) for t in fn.locals])
        out = None
        for s in fn.body.stmts:
            out = self.ev(s, env)
        return out

    def ev(self, n, env):
        return getattr(self, "e_" + type(n).__name__)(n, env)

    def e_Const(self, n, env):
        return np.array(np.asarray(n.value, dtype=_dtype(n.ty.elem)))

    def e_GlobalRef(self, n, env):
        if n.state:
            return self.state[n.name]
        return self.consts[n.name]

    def e_LocalRef(self, n, env):
        return env.locals[n.slot]

    def e_ParamRef(self, n, env):
        return env.params[n.index]

    def e_Convert(self, n, env):
        return self.cast(self.ev(n.a, env), n.a.ty.elem, n.ty.elem)

    def e_Unary(self, n, env):
        a = self.ev(n.a, env)
        e = n.a.ty.elem
        if is_fixed(e):
            s = fx.spec_of(e)
            if n.op == "neg":
                return fx.neg(a, s)
            if n.op == "floor":
                return fx.floor_int(a, s, self.bits)
            if n.op == "not":
                return (a == 0).astype(np.int64)
        if n.op == "neg":
            return self.wrap(-a) if e == "int" else -a
        if n.op == "floor":
            f = np.floor(np.nan_to_num(a, nan=0.0, posinf=2.0 ** 70, neginf=-2.0 ** 70))
            lim = float(1 << (self.bits - 1))
            return np.where(f >= lim, (1 << (self.bits - 1)) - 1, np.where(f < -lim, -(1 << (self.bits - 1)), f)).astype(np.int64)
        if n.op == "recip":
            with np.errstate(all="ignore"):
                return 1.0 / a
        if n.op == "not":
            return (a == 0).astype(np.int64)
        raise NotImplementedError(n.op)

    def e_Binary(self, n, env):
        b = self.ev(n.b, env)
        a = self.ev(n.a, env)
        a, b = _align(a, b)
        e = n.a.ty.elem
        op = n.op
        if is_fixed(e):
            s = fx.spec_of(e)
            if op == "add":
                return fx.add(a, b, s)
            if op == "sub":
                return fx.sub(a, b, s)
            if op == "mul":
                return fx.mul(a, b, s)
            if op == "div":
                if np.any(np.asarray(b) == 0):
                    raise Trap(TRAP_DIV0, "fixed-point division by zero")
                return fx.div(a, b, s)
            if op == "max":
                return fx.fmax(a, b)
            if op == "min":
                return fx.fmin(a, b)
            return fx.compare(op, a, b)
        if e == "int":
            if op == "add":
                return self.wrap(a + b)
            if op == "sub":
                return self.wrap(a - b)
            if op == "mul":
                return self.wrap(a * b)
            if op in ("idiv", "rem"):
                if np.any(b == 0):
                    raise Trap(TRAP_DIV0, "integer division by zero")
                q = np.abs(a) // np.abs(b)
                q = np.where((a < 0) == (b < 0), q, -q)
                return self.wrap(q if op == "idiv" else a - q * b)
            if op == "pow":
                return _ipow(a, b, self.bits)
        else:
            with np.errstate(all="ignore"):
                if op == "add":
                    return a + b
                if op == "sub":
                    return a - b
                if op == "mul":
                    return a * b
                if op == "div":
                    return a / b
                if op == "rem":
                    return np.fmod(a, b)
                if op == "pow":
                    return np.power(a, b)
        if op == "div":
            with np.errstate(all="ignore"):
                return a / b
        if op == "max":
            return np.maximum(a, b)
        if op == "min":
            return np.minimum(a, b)
        if op == "lt":
            return (a < b).astype(np.int64)
        if op == "gt":
            return (a > b).astype(np.int64)
        if op == "eq":
            return (a == b).astype(np.int64)
        raise NotImplementedError(op)

    def e_Match(self, n, env):
        b = self.ev(n.b, env)
        a = self.ev(n.a, env)
        if n.a.ty.elem == "float" and not n.a.ty.shape:
            return np.array(int(np.isclose(a, b, rtol=1e-5, atol=1e-8)))
        return np.array(int(np.array_equal(a, b)))

    def _index(self, i, size):
        i = np.asarray(i, dtype=np.int64)
        i = np.where(i < 0, i + size, i)
        if np.any((i < 0) | (i >= size)):
            raise Trap(TRAP_BOUNDS, "index out of range")
        return i

    def e_Index(self, n, env):
        i = self.ev(n.i, env)
        a = self.ev(n.a, env)
        return np.array(a[int(self._index(i, a.shape[0]))])

    def e_Gather(self, n, env):
        idx = self.ev(n.idx, env)
        a = self.ev(n.a, env)
        return a[self._index(idx, a.shape[0])]

    def e_Take(self, n, env):
        a = self.ev(n.a, env)
        size = a.shape[0]
        count = abs(n.n)
        start = 0 if n.n > 0 else (-count) % size
        return a[(start + np.arange(count)) % size]

    def e_Drop(self, n, env):
        a = self.ev(n.a, env)
        return a[n.n:] if n.n > 0 else a[:n.n]

    def e_Rotate(self, n, env):
        k = int(self.ev(n.n, env))
        a = self.ev(n.a, env)
        size = a.shape[0]
        return a[(np.arange(size) - k) % size]

    def e_Reverse(self, n, env):
        return self.ev(n.a, env)[::-1].copy()

    def e_Transpose(self, n, env):
        return self.ev(n.a, env).T.copy()

    def e_Reshape(self, n, env):
        a = self.ev(n.a, env)
        total = int(np.prod(n.ty.shape))
        flat = np.resize(np.asarray(a).reshape(-1), total)
        return flat.reshape(n.ty.shape)

    def e_Join(self, n, env):
        b = self.ev(n.b, env)
        a = self.ev(n.a, env)
        a = a.reshape((1,) + a.shape[:0]) if a.ndim == 0 else a
        b = b.reshape((1,) + b.shape[:0]) if b.ndim == 0 else b
        return np.concatenate([a, b], axis=0)

    def e_Iota(self, n, env):
        return np.arange(n.n, dtype=np.int64)

    def e_MakeList(self, n, env):
        return np.stack([self.ev(q, env) for q in n.items])

    def e_Amend(self, n, env):
        idx = self.ev(n.idx, env)
        value = self.ev(n.value, env)
        a = self.ev(n.a, env).copy()
        a[self._index(idx, a.shape[0])] = value
        return a

    def e_Cond(self, n, env):
        c = self.ev(n.c, env)
        return self.ev(n.t if c != 0 else n.e, env)

    def e_Call(self, n, env):
        args = [self.ev(q, env) for q in n.args]
        return self.call(n.fn, args)

    def e_ExternCall(self, n, env):
        raise NotImplementedError("externs are not available to the fixed-point evaluator")

    def _caps(self, n, env):
        return [self.ev(c, env) for c in n.captures]

    def _stack(self, rows, ty):
        return np.stack(rows) if rows else np.zeros((0,) + tuple(ty.shape[1:]))

    def e_Map(self, n, env):
        a = self.ev(n.a, env)
        caps = self._caps(n, env)
        return np.stack([self.call(n.fn, [a[j]] + caps) for j in range(a.shape[0])])

    def e_Map2(self, n, env):
        a = self.ev(n.a, env)
        b = self.ev(n.b, env)
        caps = self._caps(n, env)
        return np.stack([self.call(n.fn, [a[j], b[j]] + caps) for j in range(a.shape[0])])

    def e_EachPair(self, n, env):
        a = self.ev(n.a, env)
        caps = self._caps(n, env)
        return np.stack([self.call(n.fn, [a[j], a[j + 1]] + caps) for j in range(a.shape[0] - 1)])

    def _fold(self, n, env, acc_ty, scan):
        seed = self.ev(n.seed, env) if n.seed is not None else None
        a = self.ev(n.a, env)
        caps = self._caps(n, env)
        if seed is None:
            acc, first = a[0].copy(), 1
        else:
            acc, first = seed.copy(), 0
        acc = self.cast(acc, (n.seed.ty.elem if n.seed is not None else n.a.ty.elem), acc_ty.elem)
        out = [acc]
        for j in range(first, a.shape[0]):
            r = self.call(n.fn, [acc, a[j]] + caps)
            acc = self.cast(r, n.fn.result_ty.elem, acc_ty.elem)
            out.append(acc)
        return np.stack(out) if scan else acc

    def e_Reduce(self, n, env):
        return self._fold(n, env, n.ty, False)

    def e_Scan(self, n, env):
        return self._fold(n, env, n.acc_ty, True)

    def e_Loop(self, n, env):
        count = int(self.ev(n.count, env)) if n.kind == "iterate" else None
        state = self.ev(n.init, env)
        caps = self._caps(n, env)
        if n.kind == "iterate":
            for _ in range(max(count, 0)):
                state = self.call(n.fn, [state] + caps)
            return state
        pcaps = [self.ev(c, env) for c in n.pred_captures]
        while self.call(n.pred, [state] + pcaps) != 0:
            state = self.call(n.fn, [state] + caps)
        return state

    def e_Assign(self, n, env):
        v = self.ev(n.value, env)
        t = n.target
        if isinstance(t, ir.LocalRef):
            env.locals[t.slot] = np.array(v)
        else:
            self.state[t.name] = np.array(v)
        return v

    def e_Seq(self, n, env):
        out = None
        for s in n.stmts:
            out = self.ev(s, env)
        return out


def _align(a, b):
    """Klong's atomic extension: the shorter shape is a prefix of the longer, so it gains trailing axes (numpy would add leading ones)."""
    if a.ndim < b.ndim:
        a = a.reshape(a.shape + (1,) * (b.ndim - a.ndim))
    elif b.ndim < a.ndim:
        b = b.reshape(b.shape + (1,) * (a.ndim - b.ndim))
    return a, b


def _ipow(a, b, bits):
    out = np.empty(np.broadcast(a, b).shape, dtype=np.int64)
    m = (1 << bits) - 1
    h = 1 << (bits - 1)
    for idx in np.ndindex(out.shape):
        x = int(np.broadcast_to(a, out.shape)[idx])
        y = int(np.broadcast_to(b, out.shape)[idx])
        out[idx] = ((pow(x, y, 1 << bits) + h) & m) - h if y >= 0 else 0
    return out
