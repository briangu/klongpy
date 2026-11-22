"""
Signatures of host functions and entry points (docs/static-profile.md).

    parse_signature("int float[64] -> float")  -> Signature
    parse_type("float[16,241]")                -> Ty
    conform(ty, value, what)                   -> value, checked and converted
    c_identifier("read.adc")                   -> "read_adc"
    check_symbol(symbol), check_entry_name(name)

Grammar

    signature := params "->" result
    params    := type*                 separated by spaces, may be empty
    result    := type | "void"
    type      := ("int" | "float" | fixed) dims? | "str"
    fixed     := "fixed<" W "," I ">"  2 <= W <= 64, 1 <= I <= W
    dims      := "[" N ("," N)* "]"    1 to 4 dimensions, each a positive integer

``str`` is a parameter type only, and only of an extern. The parser, the
canonical text and the name rules are the native engine's (the extension's
parse_signature, format_signature, check_symbol, ...): the interpreter's
``.extern``/``.entry``, the static checker and this module cannot disagree
about a signature. What is Python's here is the shape of the values (Ty,
Signature, ExternDecl, EntryDecl) and ``conform``, the host half that checks a
Python value against a type.
"""
from dataclasses import dataclass

import numpy as np

MAX_RANK = 4



def _ext():
    from ..native_ext import native
    return native()


def fixed_elem(width, integer):
    """The element type text of a fixed-point number: fixed<32,16>."""
    return f"fixed<{int(width)},{int(integer)}>"


def fixed_spec(elem):
    """(W, I) of an element type such as ``fixed<32,16>``, or None when elem is not a fixed-point type."""
    return _ext().fixed_spec(elem) if isinstance(elem, str) else None


def is_fixed(elem):
    return isinstance(elem, str) and elem.startswith("fixed<")


def is_numeric(elem):
    return elem in ("int", "float") or is_fixed(elem)

INT_RANGE = {32: (-2**31, 2**31 - 1), 64: (-2**63, 2**63 - 1)}


@dataclass(frozen=True)
class Ty:
    """
    A static type: an element type and a shape.

        elem    "int", "float", "str" or "void"
        shape   () for a scalar, else 1 to 4 positive dimensions, row-major
    """
    elem: str
    shape: tuple = ()

    @property
    def rank(self):
        return len(self.shape)

    @property
    def is_array(self):
        return bool(self.shape)

    @property
    def size(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def __str__(self):
        if self.shape:
            return f"{self.elem}[{','.join(str(d) for d in self.shape)}]"
        return self.elem


INT = Ty("int")
FLOAT = Ty("float")
STR = Ty("str")
VOID = Ty("void")


@dataclass(frozen=True)
class Signature:
    """The parsed form of ``params -> result``. ``str(sig)`` is the canonical text."""
    params: tuple
    result: Ty

    @property
    def arity(self):
        return len(self.params)

    def __str__(self):
        left = " ".join(str(p) for p in self.params)
        return f"{left} -> {self.result}" if left else f"-> {self.result}"


class ExternDecl:
    """``name::.extern(c_symbol; signature)``: a function the target provides. ``name`` is what the compiler calls it (``prefix.name``
    for an extern a module file declares); ``c_symbol`` is the C function, what the host registers and the generated C calls."""
    __slots__ = ("name", "signature", "c_symbol", "span", "source")

    def __init__(self, name, signature, c_symbol=None, span=None, source=None):
        self.name = name
        self.signature = signature
        self.c_symbol = name if c_symbol is None else c_symbol
        self.span = span
        self.source = source

    @property
    def c_name(self):
        return self.c_symbol

    def __repr__(self):
        return f"<extern {self.name}: {self.signature} (C {self.c_symbol})>"


class EntryDecl:
    """``.entry(name; signature)``: a function the host calls, with its argument types."""
    __slots__ = ("name", "signature", "span", "source")

    def __init__(self, name, signature, span=None, source=None):
        self.name = name
        self.signature = signature
        self.span = span
        self.source = source

    def __repr__(self):
        return f"<entry {self.name}: {self.signature}>"


class SignatureError(ValueError):
    """
    A malformed signature. ``message`` is one line; ``span`` is the
    (start, end) pair of offsets into the signature text the message is about.
    """
    def __init__(self, message, span):
        ValueError.__init__(self, message)
        self.message = message
        self.span = span


def _ty(t):
    return Ty(t[0], tuple(t[1]))


def _wrap(e):
    message, start, end = e.args
    return SignatureError(message, (start, end))


def parse_type(text):
    """Parse one type such as ``float[16,241]``. Raises SignatureError."""
    if not isinstance(text, str):
        raise SignatureError("a type must be a string", (0, 0))
    ext = _ext()
    try:
        return _ty(ext.parse_type(text))
    except ext.SignatureError as e:
        raise _wrap(e) from None


def parse_signature(text, allow_str=True):
    """
    Parse ``params -> result``. Raises SignatureError (a ValueError) whose
    span points into text.

    allow_str=False is the rule for ``.entry``: ``str`` is a parameter type
    of externs only.
    """
    if not isinstance(text, str):
        raise SignatureError("a signature must be a string such as \"int -> float\"", (0, 0))
    ext = _ext()
    try:
        params, result = ext.parse_signature(text, allow_str)
    except ext.SignatureError as e:
        raise _wrap(e) from None
    return Signature(tuple(_ty(p) for p in params), _ty(result))


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

def c_identifier(name):
    """The C identifier of a Klong name: every '.' becomes '_'."""
    return _ext().c_identifier(str(name))


def check_symbol(symbol):
    """None if symbol can be the C symbol of an .extern, else the reason as one line."""
    return _ext().check_symbol(symbol) if isinstance(symbol, str) else f"an extern's C symbol must be a string (found {symbol!r})"


def check_entry_name(name):
    """None if name can be declared with .entry, else the reason as one line."""
    return _ext().check_entry_name(name) if isinstance(name, str) else f"an entry name must be a Klong name (found {name!r})"


# ---------------------------------------------------------------------------
# Values against types (the host half: what KlongPy checks when an extern is
# called and when an entry is called through klongpy.static.call_entry)
# ---------------------------------------------------------------------------

def _describe(v):
    if isinstance(v, bool):
        return "a boolean"
    if isinstance(v, (int, np.integer)):
        return "an integer"
    if isinstance(v, (float, np.floating)):
        return "a float"
    if isinstance(v, str):
        return "a string" if type(v) is str else "a symbol or a character"
    shape = getattr(v, "shape", None)
    if shape is not None and len(shape) > 0:
        kind = getattr(getattr(v, "dtype", None), "kind", None)
        elem = {"i": "int", "u": "int", "b": "int", "f": "float"}.get(kind)
        if elem is None and "int" in str(getattr(v, "dtype", "")):
            elem = "int"
        if elem is None and "float" in str(getattr(v, "dtype", "")):
            elem = "float"
        dims = ",".join(str(int(d)) for d in shape)
        return f"{elem}[{dims}]" if elem else f"a list of shape [{dims}]"
    if v is None:
        return "nothing"
    return f"a {type(v).__name__}"


def _scalar(v):
    """A 0-d array or tensor as a Python number; anything else unchanged."""
    if getattr(v, "ndim", None) == 0 and hasattr(v, "item"):
        return v.item()
    return v


class Mismatch(Exception):
    """conform() failed: ``kind`` is "type" or "shape", ``message`` one line."""
    def __init__(self, kind, message):
        Exception.__init__(self, message)
        self.kind = kind
        self.message = message


def _conform_fixed(ty, value, what):
    """A fixed-point value (a number, a list or a FixedArray of the type) as a FixedArray of the type."""
    from . import fixed as fx
    spec = fx.spec_of(ty.elem)
    if isinstance(value, fx.FixedArray):
        if value.spec != spec:
            raise Mismatch("type", f"{what} must be {ty}, got {value.spec.elem}{'[' + ','.join(str(d) for d in value.shape) + ']' if value.ndim else ''}")
        raw = value.raw
    else:
        v = value
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        if isinstance(v, (list, tuple)):
            try:
                v = np.asarray(v)
            except Exception:
                raise Mismatch("type", f"{what} must be {ty}, got a ragged list")
        a = np.asarray(v)
        if a.dtype.kind not in "iubf" or a.dtype == object:
            raise Mismatch("type", f"{what} must be {ty}, got {_describe(value)}")
        raw = fx.from_float(a, spec) if a.dtype.kind == "f" else fx.from_int(a, spec)
    raw = np.asarray(raw, dtype=np.int64)
    if tuple(raw.shape) != tuple(ty.shape):
        raise Mismatch("shape" if ty.shape and raw.ndim else "type", f"{what} must be {ty}, got {_describe(value)}")
    return fx.FixedArray(raw, spec)


def conform(ty, value, what):
    """
    Check value against ty and return it in the canonical host form:

        int        Python int        (an integer, numpy integer or boolean)
        float      Python float      (an int is accepted and converted)
        int[..]    numpy int64 array of exactly that shape
        float[..]  numpy float64 array of exactly that shape (ints converted)
        str        str
        void       None, whatever the value

    Raises Mismatch(kind, message); what names the value in the message
    ("argument 1 of read.adc").
    """
    if ty.elem == "void":
        return None
    if ty.elem == "str":
        if type(value) is str:      # not a symbol or a character, which are str subclasses
            return value
        raise Mismatch("type", f"{what} must be str, got {_describe(value)}")
    if is_fixed(ty.elem):
        return _conform_fixed(ty, value, what)
    v = _scalar(value)
    if not ty.shape:
        if isinstance(v, (bool, np.bool_)):
            v = int(v)
        if ty.elem == "int":
            if isinstance(v, (int, np.integer)):
                return int(v)
        elif isinstance(v, (int, np.integer, float, np.floating)):
            return float(v)
        raise Mismatch("type", f"{what} must be {ty}, got {_describe(v)}")
    if isinstance(v, (list, tuple)):
        try:
            v = np.asarray(v)
        except Exception:
            raise Mismatch("type", f"{what} must be {ty}, got a ragged list")
    shape = getattr(v, "shape", None)
    if shape is None or len(shape) == 0:
        raise Mismatch("type", f"{what} must be {ty}, got {_describe(v)}")
    if not isinstance(v, np.ndarray):
        try:
            v = np.asarray(v.detach().cpu() if hasattr(v, "detach") else v)
        except Exception:
            raise Mismatch("type", f"{what} must be {ty}, got {_describe(v)}")
    kind = v.dtype.kind
    if kind not in "iubf" or (ty.elem == "int" and kind == "f"):
        raise Mismatch("type", f"{what} must be {ty}, got {_describe(v)}")
    if tuple(v.shape) != tuple(ty.shape):
        raise Mismatch("shape", f"{what} must be {ty}, got {_describe(v)}")
    return v.astype(np.int64 if ty.elem == "int" else np.float64, copy=False)
