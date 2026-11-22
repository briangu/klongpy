"""
.extern, .entry, .param, .fixed and .float: declaring the functions a compiled program takes from its host, the functions the
host calls and its tunable registers (docs/static-profile.md). The functions are the native engine's (their text is
klongpy/data/system_function_help.json); what is Python's here is the host's side of the declarations:

    klong = KlongInterpreter(strict=2)
    klong.externs["read_adc"] = lambda channel: 0.3       # the stand-in, by C symbol
    klong('adc::.extern("read_adc";"int -> float")')      # the function value
    klong('adc(0)')                                       # 0.3

A target's functions live in a module file (``adc::.extern("read_adc";"int -> float")`` and ``export:::{;"adc";adc}`` in ``read.kg``),
used as ``.l("read")`` and ``read.adc(3)``.

``klong.externs`` is an ExternTable: a dict from C symbol to a Python callable (the stand-in, looked up by the engine at each
call, so it can be bound before or after the ``.extern`` line runs and replaced at any time), which also shows what the
program declared (``declared``, ``entries``, ``params``: read from the engine). The engine checks arguments and result
against the signature on every call.
"""
from .static.signature import EntryDecl, ExternDecl, SignatureError, Ty, is_fixed, parse_signature, parse_type
from .errors import KlongError, KlongTypeError, python_error


class ParamDecl:
    """``.param(name; type[; default])``: a global that is an input register of the compiled program."""
    __slots__ = ("name", "ty", "default", "span", "source")

    def __init__(self, name, ty, default=0, span=None, source=None):
        self.name = name
        self.ty = ty
        self.default = default
        self.span = span
        self.source = source

    def __repr__(self):
        return f"<param {self.name}: {self.ty}>"


class ExternTable(dict):
    """
    ``klong.externs``: C symbol -> Python stand-in.

        declared   {symbol: ExternDecl}   what .extern has declared (C symbol, signature)
        entries    [EntryDecl]          what .entry has declared, in order
        params     {name: ParamDecl}    what .param has declared (tunable registers)

    The three are read from the interpreter that owns the table. A stand-in is called with one positional argument per
    parameter of the signature: a Python int for ``int``, a float for ``float``, a str for ``str``, a numpy int64 or float64
    array of the declared shape for an array. It returns a value of the result type (anything for ``void``).
    """

    def __init__(self, klong=None, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self._klong = klong

    def _native(self):
        return self._klong._k if self._klong is not None else None

    @property
    def declared(self):
        k = self._native()
        return {n: ExternDecl(n, parse_signature(s, allow_str=True), n) for n, s in (k.declared_externs() if k else ())}

    @property
    def entries(self):
        k = self._native()
        return [EntryDecl(n, parse_signature(s, allow_str=False)) for n, s in (k.declared_entries() if k else ())]

    @property
    def params(self):
        k = self._native()
        out = {}
        for name, ty, default, _value in (k.params() if k else ()):
            out[name] = ParamDecl(name, parse_type(ty), _host_value(default))
        return out

    def get(self, name, default=None):
        """The stand-in, as the engine calls it: an exception of it that is not a KlongError is a KlongPythonError."""
        fn = dict.get(self, name)
        if fn is None:
            return default
        if getattr(fn, "_klong_wrapped", False):
            return fn

        def call(*args):
            try:
                return fn(*args)
            except KlongError:
                raise
            except Exception as e:
                raise python_error(e) from e
        call._klong_wrapped = True
        return call

    def entry(self, name, signature=None):
        """The EntryDecl of name (with that signature text, when a function has several); KeyError if none."""
        for e in self.entries:
            if e.name == name and (signature is None or str(e.signature) == str(parse_signature(signature))):
                return e
        raise KeyError(name)


def _host_value(v):
    """A value of the extension as Python has it: a fixed-point (elem, raw) as a FixedArray."""
    if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str):
        import numpy as np
        from .static import fixed as fx
        return fx.FixedArray(np.array(v[1], dtype=np.int64), fx.spec_of(v[0]))
    return v


def fixed_value(value, elem, what):
    """A number, a list or a FixedArray as a FixedArray of the type elem (truncating toward minus infinity)."""
    import numpy as np
    from .static import fixed as fx
    spec = fx.spec_of(elem)
    v = value
    if isinstance(v, fx.FixedArray):
        return fx.FixedArray(fx.convert(v.raw, v.spec, spec), spec)
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    a = np.asarray(v)
    if a.dtype.kind in "iub":
        return fx.FixedArray(fx.from_int(a, spec), spec)
    if a.dtype.kind == "f":
        return fx.FixedArray(fx.from_float(a, spec), spec)
    raise KlongTypeError(f"{what} must be a number or an array of numbers, got {type(value).__name__}")


def make_fixed(type_text, value):
    """What the engine's ``.fixed(type;value)`` asks the host for: the fixed-point array of the type named by the string."""
    if type(type_text) is not str:
        raise KlongTypeError(".fixed: the type must be a string")
    try:
        ty = parse_type(type_text)
    except SignatureError as e:
        raise KlongTypeError(f".fixed: {e.message}", hint='a type is int, float or fixed<W,I>, for example "fixed<32,16>"') from None
    if not is_fixed(ty.elem) or ty.shape:
        raise KlongTypeError(".fixed: the type must be fixed<W,I>, for example \"fixed<32,16>\"")
    return fixed_value(value, ty.elem, ".fixed: the value")


def create_system_functions_static():
    """The Python system functions of this module: none, the declarations are the engine's."""
    return {}
