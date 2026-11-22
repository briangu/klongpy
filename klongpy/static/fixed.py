"""
Fixed-point numbers of the static profile (docs/hardware.md).

``fixed<W,I>`` is a signed number of W bits of which I are integer bits (the sign
bit counts): the value of a W-bit integer ``raw`` is ``raw * 2^-(W-I)``. This
module is the reference semantics; the C emitter, the hardware code and the
typed-IR evaluator (klongpy/static/eval_ir.py) all implement exactly these
functions, and the tests hold them to bit for bit equality.

    add, sub, neg, max, min      raw arithmetic, the result wrapped to W bits
    mul                          floor(a * b / 2^F), wrapped to W bits
    div                          floor(a * 2^F / b), wrapped to W bits (b is not 0; W <= 32)
    compare                      on the raw values
    floor to int                 raw >> F (floor), then wrapped to kg_int
    int to fixed                 raw = v * 2^F, wrapped to W bits
    float to fixed               raw = floor(v * 2^F), wrapped; nan and anything outside the
                                 range of a 64-bit integer give 0
    fixed to float               raw * 2^-F (rounded to float32 in compiled code)
    fixed to fixed               raw scaled to the new F by a floor shift or an exact shift, wrapped

"Wrapped to W bits" is two's complement: the low W bits, sign extended. F = W - I.

Values here are numpy int64 arrays (or Python ints for scalars) holding raw. A
product of two operands of more than 32 bits needs 128 bits; those are done on
Python integers. ``FixedArray`` is the container that carries (W, I) for the
values the interpreter produces: ``.fixed("fixed<32,16>";0.5)`` returns one.
"""

import numpy as np

from .signature import fixed_elem, fixed_spec

__all__ = ["FixedArray", "Spec", "spec_of", "wrap", "from_float", "to_float", "from_int", "convert", "add", "sub", "neg",
           "mul", "div", "floor_int", "compare", "fmax", "fmin", "is_fixed_value", "raw_of", "make"]


class Spec:
    """W, I and F = W - I of a fixed-point type."""
    __slots__ = ("W", "I", "F")

    def __init__(self, W, I):
        self.W, self.I, self.F = int(W), int(I), int(W) - int(I)

    @property
    def elem(self):
        return fixed_elem(self.W, self.I)

    def __eq__(self, o):
        return isinstance(o, Spec) and (self.W, self.I) == (o.W, o.I)

    def __hash__(self):
        return hash((self.W, self.I))

    def __repr__(self):
        return self.elem

    @property
    def lo(self):
        return -(1 << (self.W - 1))

    @property
    def hi(self):
        return (1 << (self.W - 1)) - 1


def spec_of(elem_or_spec):
    if isinstance(elem_or_spec, Spec):
        return elem_or_spec
    w, i = fixed_spec(elem_or_spec)
    return Spec(w, i)


def _obj(a):
    """a as an object array of Python ints (exact, any width)."""
    return np.asarray(a).astype(object)


def wrap(v, spec):
    """Two's complement wrap of integers (numpy int64 or object arrays, or Python ints) to W bits."""
    W = spec.W
    if isinstance(v, (int, np.integer)):
        v = int(v)
        m = (1 << W) - 1
        v &= m
        return v - (1 << W) if v >> (W - 1) else v
    v = np.asarray(v)
    if v.dtype == object:
        m = (1 << W) - 1
        h = 1 << (W - 1)
        return np.vectorize(lambda x: ((int(x) + h) & m) - h, otypes=[object])(v) if v.size else v
    if W == 64:
        return v.astype(np.int64)
    h = np.int64(1 << (W - 1))
    m = np.int64((1 << W) - 1)
    return ((v.astype(np.int64) + h) & m) - h


def _fin(v, spec):
    """Wrap and bring back to int64."""
    v = wrap(v, spec)
    return v.astype(np.int64) if isinstance(v, np.ndarray) else np.int64(v)


def add(a, b, spec):
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    if spec.W >= 63:
        return _fin(_obj(a) + _obj(b), spec)
    return _fin(a + b, spec)


def sub(a, b, spec):
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    if spec.W >= 63:
        return _fin(_obj(a) - _obj(b), spec)
    return _fin(a - b, spec)


def neg(a, spec):
    a = np.asarray(a, dtype=np.int64)
    if spec.W >= 63:
        return _fin(-_obj(a), spec)
    return _fin(-a, spec)


def mul(a, b, spec):
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    if spec.W <= 32:
        return _fin((a * b) >> spec.F, spec)
    return _fin((_obj(a) * _obj(b)) >> spec.F, spec)


def div(a, b, spec):
    """floor(a * 2^F / b), wrapped. The caller has checked that b is not 0 and that W <= 32."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    if spec.W > 32:
        raise ValueError("fixed-point division is defined for W <= 32")
    return _fin((a << spec.F) // b, spec)


def floor_int(a, spec, int_bits):
    """_x: floor to an int, wrapped to kg_int."""
    a = np.asarray(a, dtype=np.int64) >> spec.F
    if int_bits == 64:
        return a
    h, m = np.int64(1 << (int_bits - 1)), np.int64((1 << int_bits) - 1)
    return ((a + h) & m) - h


def compare(op, a, b):
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    r = {"lt": a < b, "gt": a > b, "eq": a == b}[op]
    return r.astype(np.int64)


def fmax(a, b):
    return np.maximum(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))


def fmin(a, b):
    return np.minimum(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))


def from_int(v, spec):
    """Int to fixed: raw = v * 2^F wrapped (v is exact in int64)."""
    v = np.asarray(v, dtype=np.int64)
    if spec.F == 0:
        return _fin(v, spec)
    return _fin(_obj(v) * (1 << spec.F), spec)


def from_float(v, spec):
    """Float to fixed: floor(v * 2^F) wrapped; nan and anything beyond a 64-bit integer give 0."""
    v = np.asarray(v, dtype=np.float64)
    scaled = np.floor(v * (2.0 ** spec.F))
    ok = np.isfinite(scaled) & (scaled >= -9223372036854775808.0) & (scaled < 9223372036854775808.0)
    safe = np.where(ok, scaled, 0.0)
    return _fin(safe.astype(np.int64), spec) if np.ndim(safe) else _fin(np.int64(safe), spec)


def to_float(raw, spec):
    """Fixed to float64 (exact for W <= 53): raw * 2^-F."""
    return np.asarray(raw, dtype=np.int64).astype(np.float64) * (2.0 ** -spec.F)


def convert(raw, src, dst):
    """Fixed to another fixed type: floor shift to the new fraction, then wrap."""
    raw = np.asarray(raw, dtype=np.int64)
    d = dst.F - src.F
    if d >= 0:
        return _fin(_obj(raw) * (1 << d), dst) if d else _fin(raw, dst)
    return _fin(raw >> (-d), dst)


# ---------------------------------------------------------------------------
# The interpreter's container
# ---------------------------------------------------------------------------

class FixedArray(np.ndarray):
    """
    An int64 array of raw fixed-point values that knows its type (``spec``). It is what
    ``.fixed(type;x)`` returns under KlongPy. The interpreter has no fixed-point
    arithmetic of its own: a program that uses fixed numbers runs through
    klongpy.static.call_entry, which evaluates the typed program on the raw values
    (klongpy/static/eval_ir.py). ufuncs on a FixedArray follow the reference semantics for
    the operations that have one and refuse the rest, so a result is never silently wrong.
    """
    def __new__(cls, raw, spec):
        o = np.asarray(raw, dtype=np.int64).view(cls)
        o.spec = spec
        return o

    def __array_finalize__(self, obj):
        if obj is not None:
            self.spec = getattr(obj, "spec", None)

    @property
    def raw(self):
        return np.asarray(self).view(np.ndarray)

    def to_float(self):
        return to_float(self.raw, self.spec)

    def __repr__(self):
        return f"{self.spec.elem}({np.asarray(self).tolist()} raw)"

    def __array_ufunc__(self, ufunc, method, *inputs, **kw):
        spec = self.spec
        for i in inputs:
            if isinstance(i, FixedArray) and i.spec != spec:
                raise TypeError(f"cannot mix {spec.elem} and {i.spec.elem}")
        if method == "__call__":
            ops = [i.raw if isinstance(i, FixedArray) else i for i in inputs]
            name = ufunc.__name__
            if name in ("add", "subtract", "multiply", "divide", "true_divide", "maximum", "minimum", "less", "greater",
                        "equal"):
                a, b = ops
                if not isinstance(inputs[0], FixedArray) or not isinstance(inputs[1], FixedArray):
                    other = 1 if isinstance(inputs[0], FixedArray) else 0
                    x = ops[other]
                    if isinstance(x, (float, np.floating)) or (isinstance(x, np.ndarray) and x.dtype.kind == "f"):
                        raise TypeError(f"cannot mix {spec.elem} with a float")
                    conv = from_int(x, spec)
                    a, b = (ops[0], conv) if other == 1 else (conv, ops[1])
                if name == "add":
                    return FixedArray(add(a, b, spec), spec)
                if name == "subtract":
                    return FixedArray(sub(a, b, spec), spec)
                if name == "multiply":
                    return FixedArray(mul(a, b, spec), spec)
                if name in ("divide", "true_divide"):
                    return FixedArray(div(a, b, spec), spec)
                if name == "maximum":
                    return FixedArray(fmax(a, b), spec)
                if name == "minimum":
                    return FixedArray(fmin(a, b), spec)
                return compare({"less": "lt", "greater": "gt", "equal": "eq"}[name], a, b)
            if name == "negative":
                return FixedArray(neg(ops[0], spec), spec)
            if name == "floor":
                return floor_int(ops[0], spec, 64)
        if method in ("reduce", "accumulate") and ufunc.__name__ in ("add", "maximum", "minimum"):
            raw = inputs[0].raw
            fn = {"add": lambda a, b: add(a, b, spec), "maximum": fmax, "minimum": fmin}[ufunc.__name__]
            axis = kw.get("axis", 0)
            moved = np.moveaxis(raw, axis, 0)
            out = [moved[0]]
            for k in range(1, len(moved)):
                out.append(fn(out[-1], moved[k]))
            if method == "reduce":
                return FixedArray(out[-1], spec)
            return FixedArray(np.moveaxis(np.array(out), 0, axis), spec)
        return NotImplemented


def is_fixed_value(v):
    return isinstance(v, FixedArray)


def raw_of(v):
    return v.raw if isinstance(v, FixedArray) else np.asarray(v, dtype=np.int64)


def make(elem, raw):
    return FixedArray(raw, spec_of(elem))
