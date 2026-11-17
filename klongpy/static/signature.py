"""
Signatures of host functions and entry points (docs/static-profile.md).

    parse_signature("int float[64] -> float")  -> Signature
    parse_type("float[16,241]")                -> Ty
    conform(ty, value, what)                   -> value, checked and converted
    c_identifier("read.adc")                   -> "read_adc"
    check_extern_name(name), check_entry_name(name)

Grammar

    signature := params "->" result
    params    := type*                 separated by spaces, may be empty
    result    := type | "void"
    type      := ("int" | "float" | fixed) dims? | "str"
    fixed     := "fixed<" W "," I ">"  2 <= W <= 64, 1 <= I <= W
    dims      := "[" N ("," N)* "]"    1 to 4 dimensions, each a positive integer

``str`` is a parameter type only, and only of an extern. The same parser is
used by ``.extern``/``.entry`` in the interpreter (klongpy/sys_fn_static.py)
and by the static checker, so the two cannot disagree about a signature.

This module needs only numpy.
"""
import re
from dataclasses import dataclass

import numpy as np

MAX_RANK = 4

_FIXED = re.compile(r"fixed<(\d+),(\d+)>\Z")


def fixed_elem(width, integer):
    """The element type text of a fixed-point number: fixed<32,16>."""
    return f"fixed<{int(width)},{int(integer)}>"


def fixed_spec(elem):
    """(W, I) of an element type such as ``fixed<32,16>``, or None when elem is not a fixed-point type."""
    m = _FIXED.match(elem) if isinstance(elem, str) else None
    return (int(m.group(1)), int(m.group(2))) if m else None


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
    """``.extern(name; signature[; module])``: a function the target provides."""
    __slots__ = ("name", "signature", "module", "span", "source")

    def __init__(self, name, signature, module="env", span=None, source=None):
        self.name = name
        self.signature = signature
        self.module = module
        self.span = span
        self.source = source

    @property
    def c_name(self):
        return c_identifier(self.name)

    def __repr__(self):
        return f"<extern {self.name}: {self.signature} (module {self.module})>"


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


_TYPE = re.compile(r"([A-Za-z]+)(<[^<>\s]*>)?(\[([^\]]*)\])?")
_DIMS = re.compile(r"\s*\d+\s*(,\s*\d+\s*)*\Z")
_WORD = re.compile(r"\S+")


def _type_at(text, pos, end):
    """Read one type from text[pos:end]. Returns (Ty, next position)."""
    m = _TYPE.match(text, pos, end)
    if not m:
        w = _WORD.match(text, pos, end)
        stop = w.end() if w else min(pos + 1, end)
        raise SignatureError(f"expected a type but found '{text[pos:stop]}'", (pos, stop))
    word = m.group(1)
    angle = m.group(2)
    if word == "fixed":
        if angle is None:
            raise SignatureError("fixed needs its widths: fixed<W,I> (W total bits, I integer bits)", m.span(1))
        mm = re.match(r"<(\d+),(\d+)>\Z", angle)
        if not mm:
            raise SignatureError(f"a fixed-point type is written fixed<W,I> with two positive integers, found 'fixed{angle}'",
                                 m.span(2))
        w, i = int(mm.group(1)), int(mm.group(2))
        if not 2 <= w <= 64:
            raise SignatureError(f"fixed<{w},{i}>: the total width W must be 2 to 64", m.span(2))
        if not 1 <= i <= w:
            raise SignatureError(f"fixed<{w},{i}>: the integer bits I must be 1 to W ({w})", m.span(2))
        word = fixed_elem(w, i)
    else:
        if angle is not None:
            raise SignatureError(f"{word} takes no <...> (only fixed<W,I> does)", m.span(2))
        if word not in ("int", "float", "str", "void"):
            raise SignatureError(f"unknown type '{word}' (the types are int, float, fixed<W,I>, str and void)", m.span(1))
    shape = ()
    if m.group(3) is not None:
        inner = m.group(4)
        if word in ("str", "void"):
            raise SignatureError(f"{word} cannot have dimensions", m.span(3))
        if not _DIMS.match(inner):
            raise SignatureError(f"dimensions must be positive integers separated by commas, found '[{inner}]'", m.span(3))
        shape = tuple(int(d) for d in inner.split(","))
        if any(d <= 0 for d in shape):
            raise SignatureError("every dimension must be a positive integer", m.span(3))
        if len(shape) > MAX_RANK:
            raise SignatureError(f"an array has at most {MAX_RANK} dimensions, found {len(shape)}", m.span(3))
    stop = m.end()
    if stop < end and not text[stop].isspace():
        w = _WORD.match(text, pos, end)
        raise SignatureError(f"'{w.group(0)}' is not a type", w.span())
    return Ty(word, shape), stop


def parse_type(text):
    """Parse one type such as ``float[16,241]``. Raises SignatureError."""
    if not isinstance(text, str):
        raise SignatureError("a type must be a string", (0, 0))
    pos = len(text) - len(text.lstrip())
    end = len(text.rstrip())
    if pos >= end:
        raise SignatureError("expected a type but found nothing", (0, len(text)))
    ty, stop = _type_at(text, pos, end)
    if stop != end:
        raise SignatureError(f"unexpected '{text[stop:end].strip()}' after the type", (stop, end))
    return ty


def parse_signature(text, allow_str=True):
    """
    Parse ``params -> result``. Raises SignatureError (a ValueError) whose
    span points into text.

    allow_str=False is the rule for ``.entry``: ``str`` is a parameter type
    of externs only.
    """
    if not isinstance(text, str):
        raise SignatureError("a signature must be a string such as \"int -> float\"", (0, 0))
    arrow = text.find("->")
    if arrow < 0:
        raise SignatureError("a signature needs '->' between the parameter types and the result type", (0, len(text)))
    if text.find("->", arrow + 2) >= 0:
        second = text.find("->", arrow + 2)
        raise SignatureError("a signature has exactly one '->'", (second, second + 2))
    params = []
    pos = 0
    while True:
        while pos < arrow and text[pos].isspace():
            pos += 1
        if pos >= arrow:
            break
        start = pos
        ty, pos = _type_at(text, pos, arrow)
        if ty.elem == "void":
            raise SignatureError("void is a result type only; a function of no parameters is written \"-> int\"", (start, pos))
        if ty.elem == "str" and not allow_str:
            raise SignatureError("str is a parameter type of externs only", (start, pos))
        params.append(ty)
    pos = arrow + 2
    end = len(text.rstrip())
    while pos < end and text[pos].isspace():
        pos += 1
    if pos >= end:
        raise SignatureError("a signature needs a result type after '->' (void for none)", (arrow, len(text)))
    start = pos
    result, pos = _type_at(text, pos, end)
    if result.elem == "str":
        raise SignatureError("str cannot be a result type", (start, pos))
    if pos != end:
        rest = pos + len(text[pos:end]) - len(text[pos:end].lstrip())
        raise SignatureError(f"unexpected '{text[rest:end]}' after the result type", (rest, end))
    return Signature(tuple(params), result)


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9.]*\Z")

_C_KEYWORDS = frozenset("""
auto break case char const continue default do double else enum extern float
for goto if inline int long register restrict return short signed sizeof
static struct switch typedef union unsigned void volatile while
_Bool _Complex _Imaginary main memcpy memset NULL
""".split())

# prog_<name> names the emitter generates itself
RESERVED_ENTRY_NAMES = frozenset(("init", "t", "storage.t", "size"))


def c_identifier(name):
    """The C identifier of a Klong name: every '.' becomes '_'."""
    return str(name).replace(".", "_")


def _check_name(name, what):
    if not isinstance(name, str) or not _NAME.match(name):
        return (f"{what} name must be a Klong name of ASCII letters, digits and '.', starting with a letter"
                f" (found {name!r})")
    if name in ("x", "y", "z"):
        return f"{name} is a function argument and cannot be {what} name"
    return None


def check_extern_name(name):
    """None if name can be declared with .extern, else the reason as one line."""
    bad = _check_name(name, "an extern")
    if bad:
        return bad
    c = c_identifier(name)
    if c in _C_KEYWORDS:
        return f"the C name of extern {name} is '{c}', which is reserved in C"
    if c.startswith("kg_"):
        return f"the C name of extern {name} is '{c}': names that start with kg_ belong to the runtime"
    return None


def check_entry_name(name):
    """None if name can be declared with .entry, else the reason as one line."""
    bad = _check_name(name, "an entry")
    if bad:
        return bad
    if name in RESERVED_ENTRY_NAMES:
        return f"an entry cannot be called {name}: prog_{c_identifier(name)} is a name the compiler generates"
    return None


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
