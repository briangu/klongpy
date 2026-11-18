"""
Backend arrays and the gradient operators on the native (Rust) engine.

The native interpreter (native/klong-py) holds a value it does not understand as an opaque Python object (a HOST value: a NumPy
array of some size, a torch tensor on any device, a 0-d tensor that requires gradients) and asks a TABLE, supplied here, to do
what Klong does with it. The extension knows nothing of the backend's array type: it calls the methods below.

    install(engine, backend=None)       the one entry point: RustInterpreter calls it once its extension and backend exist

The table (`HostTable`; the extension calls these, with the operands as Python values: a native vector is a NumPy array, a
number a Python number, a host value itself, a Klong function a callable):

    is_host(o)                  is this Python object a host value? (the backend's own array type: a torch tensor)
    monad(glyph, x)             a primitive monad with a host value as the operand: KlongPy's own eval_monad_* (monads.py)
    dyad(glyph, x, y)           a primitive dyad with a host value among the operands: KlongPy's eval_dyad_* (dyads.py)
    adverb(glyph, f, op, x, y)  an adverb applied to a host value: KlongPy's eval_adverb_* (adverbs.py), f being the function the
                                adverb modifies, op the verb's operator text (or None), y None for a monadic form
    to_native(o, what)          the host value as a NumPy array, to be copied into a native vector. REFUSES (a Klong error that
                                names the operation) a tensor that requires gradients: a copy cuts the graph
    truthy(o), format(o, display), collect(items)   the condition of :[, the text of .p, a list of values stacked

All of them answer NotImplemented (or None) when the backend has nothing to say; the extension then copies the value into a native
one (and so never silently for a tensor that requires gradients: to_native raises).

The same functions serve the Python engine and this one, so a primitive means the same thing on both (this module has no
implementation of a primitive of its own). The gradient operators (the glyphs of the Python engine's dyads.py/monads.py, which
the native core lacks) are one more callable, `HostOps`: monadic and dyadic ∇, dyadic ∂ and :> (with the autograd functions of
klongpy/autograd.py, which call a Klong function value and the backend's methods).
"""
import numpy
import operator

from .autograd import grad_of_fn
from .dyads import (create_dyad_functions, eval_dyad_autograd, eval_dyad_grad, eval_dyad_jacobian)
from .monads import create_monad_functions
from .adverbs import get_adverb_fn, _converged
from .errors import KlongTypeError
from .types import KGOp, KGSym, KGFn, KLONG_UNDEFINED
from .autograd import multi_grad_of_fn
from .writer import kg_write

def _array(backend, a):
    """A NumPy array as the backend holds one. A native vector is float64; the backend's own float (the torch backend's float32:
    its literals are float32) is what a number of the Klong program is there, so the array is made of that width."""
    if a.dtype.kind == "f":
        w = getattr(backend, "_host_float", None)
        if w is None:
            probe = backend.kg_asarray([0.5])
            w = backend._host_float = numpy.dtype(str(probe.dtype).replace("torch.", "")) if hasattr(probe, "dtype") else numpy.dtype("float64")
        if a.dtype != w:
            a = a.astype(w)
    return backend.kg_asarray(a)


_FAST = {"+": operator.add, "-": operator.sub, "*": operator.mul}

# operators that are not primitives of the table: the host operators and `::`
_NOT_PRIMITIVE = frozenset(("::", "∇", "∂", ":>"))


class HostTable:
    """The callable table of one interpreter's backend (see the module text)."""

    def __init__(self, engine, backend):
        self.engine = engine
        self.backend = backend
        self._dyads = {g: f for g, f in create_dyad_functions(engine).items() if g not in _NOT_PRIMITIVE}
        self._monads = {g: f for g, f in create_monad_functions(engine).items() if g not in _NOT_PRIMITIVE}
        self._is_backend_array = backend.is_backend_array
        self._np = numpy
        self._asarray = backend.kg_asarray
        probe = backend.kg_asarray([0.5])
        # the backend's array type, when it is not NumPy's: floating arrays meeting numbers or arrays of their own shape take the
        # operator directly (the answer is the same as eval_dyad_* gives: those add the checks Klong's lists need)
        self._tensor = type(probe) if backend.is_backend_array(probe) else None

    # -- what is a host value -----------------------------------------------------------------------------------------
    def is_host(self, o):
        return self._is_backend_array(o)

    def _lift(self, x, y):
        """A NumPy array that meets a backend array is the backend's array (the Python engine holds both as one kind)."""
        ba = self._is_backend_array
        nd = self._np.ndarray
        if ba(x):
            if type(y) is nd and y.dtype.kind in "ifb":
                y = self._asarray(y)
        elif ba(y) and type(x) is nd and x.dtype.kind in "ifb":
            x = self._asarray(x)
        return x, y

    # -- primitives ---------------------------------------------------------------------------------------------------
    def monad(self, g, x):
        f = self._monads.get(g)
        if f is None:
            return NotImplemented
        return f(x)

    def dyad(self, g, x, y):
        fast = _FAST.get(g)
        T = self._tensor
        if fast is not None and T is not None:
            tx, ty = type(x), type(y)
            if tx is T:
                if ty is T:
                    if x.is_floating_point() and y.is_floating_point() and (x.shape == y.shape or x.ndim == 0 or y.ndim == 0):
                        return fast(x, y)
                elif (ty is float or ty is int) and x.is_floating_point():
                    return fast(x, y)
            elif ty is T and (tx is float or tx is int) and y.is_floating_point():
                return fast(x, y)
        f = self._dyads.get(g)
        if f is None:
            return NotImplemented
        x, y = self._lift(x, y)
        return f(x, y)

    def adverb(self, g, f, op, x, y):
        if y is None:
            return get_adverb_fn(self.engine, g, 1)(f, x, KGOp(op, 2) if op is not None else f)
        x, y = self._lift(x, y)
        return get_adverb_fn(self.engine, g, 2)(f, x, y)

    # -- conversions --------------------------------------------------------------------------------------------------
    def to_native(self, o, what):
        b = self.backend
        if b.has_gradient(o):
            raise KlongTypeError(
                f"{what}: a tensor that requires gradients cannot be copied into a native value (the graph would be cut); "
                f"the operation has no counterpart on the {b.name} backend")
        return b.to_numpy(o)

    def adopt(self, a):
        """A native numeric vector (a NumPy array) as the backend's array: what the Python engine holds every numeric array as."""
        r = _array(self.backend, a)
        return r if self._is_backend_array(r) else None

    def truthy(self, o):
        if not self._is_backend_array(o):
            return None
        if o.ndim == 0:
            return bool(o != 0)
        return o.shape[0] > 0

    def items(self, o):
        return list(o) if o.ndim > 0 else None

    def converged(self, p, q):
        return bool(_converged(self.backend, p, q))

    def format(self, o, display):
        return kg_write(o, self.backend, display)

    def collect(self, items):
        r = self._asarray(items)
        return r if self._is_backend_array(r) else None


class HostOps:
    """The operators the native core does not implement: the gradient operators, on the engine's own function values."""

    def __init__(self, engine):
        self.engine = engine

    def __call__(self, glyph, *args):
        e = self.engine
        out, into = e._out, e._in
        args = [out(a) for a in args]
        if glyph == "∇" and len(args) == 1:
            fn = args[0]
            return into(lambda x, fn=fn, k=e: grad_of_fn(k, fn, x))
        if glyph == "∇":
            return into(eval_dyad_grad(e, args[0], args[1]))
        if glyph == "∂":
            return into(eval_dyad_jacobian(e, args[0], args[1]))
        if glyph == ":>":
            a, b = args[0], args[1]
            named = args[2] if len(args) > 2 else None
            if isinstance(named, KGSym) and isinstance(a, KGFn) and a.arity == 0:
                # loss:>w with a niladic loss: w names the parameter to differentiate (as in loss:>[w]), not a point
                return into(multi_grad_of_fn(e, a, [named])[0])
            return into(eval_dyad_autograd(e, a, b))
        raise KlongTypeError(f"operator {glyph} is not available")


def result_for(engine, v):
    """A native engine's result as the backend holds it. The Python engine on the torch backend holds every numeric array as a
    tensor; the native engine computes on native vectors and hands them out as NumPy arrays. For a backend whose arrays are not
    NumPy's (`is_backend_array` is not a NumPy test), a numeric NumPy array (also inside a list or dict) leaves as the backend's
    array. Use it on the engine's results, where RustInterpreter._out converts."""
    if engine._backend.name == "numpy":
        return v
    t = type(v)
    if t is numpy.ndarray:
        if v.dtype.kind in "ifb" and v.ndim >= 1:
            return _array(engine._backend, v)
        if v.dtype == object:
            r = numpy.empty(v.shape, dtype=object)
            rr = r.reshape(-1)
            for i, q in enumerate(v.ravel()):
                rr[i] = result_for(engine, q)
            return r
    elif t is list:
        return [result_for(engine, q) for q in v]
    return v


def install(engine, backend=None):
    """Give the engine's extension the backend's table and the gradient operators.

    engine: a RustInterpreter (it has `_k`, the native interpreter, `_backend`, `_in`, `_out`, `apply`)."""
    backend = backend or engine._backend
    engine._k.set_host_table(HostTable(engine, backend))
    if backend.name != "numpy":
        # the torch value model: every numeric vector the engine makes is a tensor on the device, as on the Python engine
        engine._k.set_adopt(1)
    engine._k.set_host_ops(HostOps(engine))
