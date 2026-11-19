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
from types import FunctionType as _FunctionType

from .autograd import grad_of_fn
from .errors import KlongError, KlongTypeError, adverb_name, is_passthrough, operator_error, operator_name
from .types import KGOp, KGSym, KGFn, KLONG_UNDEFINED
from .autograd import multi_grad_of_fn
from .writer import kg_write


class HostObject:
    """A Python dict subclass with a behaviour of its own (the handle of a remote dictionary, .clid) held by the engine as an opaque
    value: the engine copies a dict subclass into a native dictionary. The Klong operators on it (`,` `?` ...) go through the
    host table, which unwraps it. Engine need: leave a dict subclass alone."""
    __slots__ = ("obj",)

    def __init__(self, obj):
        self.obj = obj

    def __str__(self):
        return str(self.obj)

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
        from .rust_engine import _KFn
        self._kfn = _KFn
        self.engine = engine
        self.backend = backend
        # Engine need: the primitives on a backend's own arrays (a torch tensor) and on opaque host objects (the handle of a remote
        # dictionary, .clid) are still KlongPy's Python evaluators (dyads.py, monads.py, adverbs.py); the native engine has no
        # counterpart. They are loaded when there is such a value to compute on, so a program that has none never imports them.
        self._tables = None
        self._is_backend_array = backend.is_backend_array
        self._np = numpy
        self._asarray = backend.kg_asarray
        probe = backend.kg_asarray([0.5])
        # the backend's array type, when it is not NumPy's: floating arrays meeting numbers or arrays of their own shape take the
        # operator directly (the answer is the same as eval_dyad_* gives: those add the checks Klong's lists need)
        self._tensor = type(probe) if backend.is_backend_array(probe) else None

    @property
    def _dyads(self):
        return self._load()[0]

    @property
    def _monads(self):
        return self._load()[1]

    def _load(self):
        if self._tables is None:
            from .dyads import create_dyad_functions
            from .monads import create_monad_functions
            self._tables = ({g: f for g, f in create_dyad_functions(self.engine).items() if g not in _NOT_PRIMITIVE},
                            {g: f for g, f in create_monad_functions(self.engine).items() if g not in _NOT_PRIMITIVE})
        return self._tables

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
    def _fn(self, v):
        """A function value that reaches the table is the engine's raw one (or a Python callable): KlongPy's function object."""
        t = type(v)
        if t is self._kfn or t is _FunctionType or t is list:
            return self.engine._out(v)
        return v

    def _operator_error(self, e, g, arity, operands):
        """The Klong error an exception of a backend operation becomes: what the Python engine's _op_error makes."""
        if isinstance(e, KlongError) or is_passthrough(e):
            return e
        return operator_error(e, g, operator_name(g, arity), [self._fn(v) for v in operands])

    def monad(self, g, x):
        f = self._monads.get(g)
        if f is None:
            return NotImplemented
        if type(x) is HostObject:
            x = x.obj
        x = self._fn(x)
        try:
            return f(x)
        except Exception as e:
            err = self._operator_error(e, g, 1, (x,))
            if err is e:
                raise
            raise err from e

    def dyad(self, g, x, y):
        if type(x) is HostObject:
            # a function value on its way to the remote end is KlongPy's; what comes back is the engine's
            return self.engine._in(self._dyads[g](x.obj, self.engine._out(y))) if g in self._dyads else NotImplemented
        if type(y) is HostObject:
            return self.engine._in(self._dyads[g](self.engine._out(x), y.obj)) if g in self._dyads else NotImplemented
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
        x, y = self._lift(self._fn(x), self._fn(y))
        try:
            return f(x, y)
        except Exception as e:
            err = self._operator_error(e, g, 2, (x, y))
            if err is e:
                raise
            raise err from e

    def adverb(self, g, f, op, x, y):
        from .adverbs import get_adverb_fn
        f = self._fn(f)
        try:
            if y is None:
                x = self._fn(x)
                return get_adverb_fn(self.engine, g, 1)(f, x, KGOp(op, 2) if op is not None else f)
            x, y = self._lift(self._fn(x), self._fn(y))
            return get_adverb_fn(self.engine, g, 2)(f, x, y)
        except Exception as e:
            if isinstance(e, KlongError) or is_passthrough(e):
                raise
            name = "-".join(n for n in ((operator_name(op, 2) if op is not None else None), adverb_name(g, 1 if y is None else 2)) if n)
            err = operator_error(e, (op or "f") + g, name, [v for v in (x, y) if v is not None])
            raise err from e

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
        from .adverbs import _converged
        return bool(_converged(self.backend, p, q))

    def format(self, o, display):
        return kg_write(o.obj if type(o) is HostObject else o, self.backend, display)

    def collect(self, items):
        r = self._asarray(items)
        return r if self._is_backend_array(r) else None


class _Gradient:
    """The function `∇f` is: a callable of one argument that prints as a monad (Klong shows a function by its arity)."""

    def __init__(self, engine, fn):
        self.engine = engine
        self.fn = fn

    def __call__(self, x):
        return grad_of_fn(self.engine, self.fn, x)

    def __str__(self):
        return ":monad"

    __repr__ = __str__


class HostOps:
    """The operators the native core does not implement: the gradient operators, on the engine's own function values."""

    def __init__(self, engine):
        self.engine = engine

    def __call__(self, glyph, *args):
        from .dyads import eval_dyad_autograd, eval_dyad_grad, eval_dyad_jacobian
        e = self.engine
        out, into = e._out, e._in
        args = [out(a) for a in args]
        if glyph == "∇" and len(args) == 1:
            fn = args[0]
            return into(_Gradient(e, fn))
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
