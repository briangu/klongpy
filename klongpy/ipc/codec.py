"""kdb+ IPC object codec: Klong values to and from the bytes of one object.

Pure functions, no I/O.  Layout and semantics follow ``tickets/design/ipc-kdb.md``
sections 3, 4 and 9 and the byte-level reference ``ref-kdb-ipc.md`` section 3.

``encode(value, widths=64) -> bytes``
    one object (no message header), always little-endian.
``encode_call(items, widths=64) -> bytes``
    a call envelope: always a general list (type 0), even when every item is a
    symbol (design section 3).
``decode(data, little_endian=True, widths=64) -> value``
    one object.  Anything malformed raises :class:`ProtocolError` and nothing
    else, except :class:`IPCRangeError` when ``widths`` is 32 and a number does
    not fit.  Never executes anything.

Values that have no Klong equivalent decode to small placeholder objects
(:class:`KdbLambda`, :class:`KdbOpaque`, :class:`KdbError`); the endpoint layer
decides what to do with them (proxy, raise, pass back).  They all encode back.

Torch isolation (CLAUDE.md): this module uses plain numpy only.  Torch tensors
must be converted by the caller (backend ``to_numpy``) before ``encode``, and
decoded arrays are numpy arrays that the endpoint converts if the interpreter
runs a torch backend.

Decisions where the design leaves room (each also noted in the ticket):

* An empty general list decodes to an empty float64 array, which is what
  Klong's ``[]`` is; empty typed vectors decode to empty arrays of their
  element kind (int64, float64, object for symbols).
* Any 1-D numpy array and any Python list/tuple/dict-key/value sequence that is
  homogeneous (all ints, all floats or all symbols) is sent as a typed vector;
  everything else, including a list of strings, as a general list.
* General lists decode to int64/float64 arrays when all items are ints or all
  are floats, to an N-d array when all items are equally shaped int or float
  arrays of one dtype (design: ``kg_asarray`` rule), otherwise to an object
  array.  A mix of ints and floats stays an object array (lossless; the design
  does not say).
* A guid atom decodes to an int64 array of 16 and a guid vector to an (n, 16)
  int64 array; they re-encode as long vectors, not guids (design section 4;
  no guid is ever sent, so peers below capability 3 are safe).
* A dictionary whose keys or values are not lists, or a table that is not a
  dict of symbol vector and general list of equal-length columns, is a protocol
  error (the reference documents nothing else).  Unhashable keys (arrays) become
  tuples.
* Types 112, 20..97 and anything unknown are protocol errors (reference 3.4,
  UNCONFIRMED layouts: the safe default of design section 4).
"""
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..types import (
    KGSym, KGChar, KGUndefined, KLONG_UNDEFINED,
    KGFn, KGLambda, KGOp, KGAdverb, KGFnWrapper,
)
from .errors import ProtocolError, IPCTypeError, IPCRangeError

__all__ = [
    "encode", "encode_call", "decode", "KdbLambda", "KdbOpaque", "KdbError",
    "set_function_text_hook", "set_table_factory", "KdbTable", "MAX_DEPTH", "GENERIC_NULL",
]

#: nesting limit for general lists, dictionaries, tables and function objects
MAX_DEPTH = 64

#: the bytes of the generic null (::)
GENERIC_NULL = b"\x65\x00"

_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1

# ----------------------------------------------------------------------------
# placeholder values
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class KdbLambda:
    """A lambda (type 100): context name and source text.

    ``context`` is ``"kg"`` for Klong functions and ``""`` or a namespace name
    for q functions.  The raw bytes are kept so that a lambda received from a
    peer re-encodes exactly, whatever its text encoding.
    """
    context: str
    text: str
    context_raw: Optional[bytes] = field(default=None, compare=False, repr=False)
    text_raw: Optional[bytes] = field(default=None, compare=False, repr=False)

    def __str__(self):
        # what Klong prints for the value: the function's text (a q lambda
        # received from a peer stays a KdbLambda)
        return self.text

    @property
    def is_klong(self):
        return self.context == "kg"


@dataclass(frozen=True)
class KdbOpaque:
    """A function object 101 to 111 (other than ``(::)``) kept as received.

    ``raw`` holds the complete object, type byte first, normalised to
    little-endian, so ``encode`` reproduces it byte for byte.
    """
    type: int
    raw: bytes

    def __str__(self):
        return ":kdb-function"


@dataclass(frozen=True)
class KdbError:
    """The error object (type -128): the signal text.  Decoded, never raised here."""
    text: str
    text_raw: Optional[bytes] = field(default=None, compare=False, repr=False)


_fn_text_hook = None


class KdbTable(dict):
    """A decoded kdb+ table (type 98): a dict from column-name symbol to column.

    It encodes as a table again, so a table passes through unchanged.
    """
    __slots__ = ()


_table_factory = None


def set_table_factory(factory):
    """Install ``factory(names, columns) -> value`` for decoded tables.

    ``klongpy.db`` installs one when it is loaded so a table arrives as its
    ``Table``; without it a table is a :class:`KdbTable`.  ``None`` removes it.
    """
    global _table_factory
    _table_factory = factory


def set_function_text_hook(hook):
    """Install ``hook(fn) -> str | None`` giving the source text of a function.

    Function source text is not recorded by the parser yet (spans arrive with
    a later ticket); until then functions go out in the display form of design
    section 6.  ``None`` removes the hook.
    """
    global _fn_text_hook
    _fn_text_hook = hook


# ----------------------------------------------------------------------------
# width handling
# ----------------------------------------------------------------------------


def _norm_widths(widths):
    if widths is None:
        return 64, 64
    if isinstance(widths, (tuple, list)):
        i, f = widths
    else:
        i = f = widths
    i, f = int(i), int(f)
    if i not in (32, 64) or f not in (32, 64):
        raise ValueError("widths must be 32 or 64")
    return i, f


# ----------------------------------------------------------------------------
# encoder
# ----------------------------------------------------------------------------

_PACK_ATOM = {
    -1: struct.Struct("<bB"), -4: struct.Struct("<bB"), -5: struct.Struct("<bh"),
    -6: struct.Struct("<bi"), -7: struct.Struct("<bq"), -8: struct.Struct("<bf"),
    -9: struct.Struct("<bd"), -12: struct.Struct("<bq"), -16: struct.Struct("<bq"),
}
_VEC_HDR = struct.Struct("<bBi")

# kdb+ temporals sent from numpy: datetime64 is a timestamp (12), int64
# nanoseconds since 2000-01-01; timedelta64 is a timespan (16), int64
# nanoseconds; NaT is the null (INT64_MIN).  Any numpy unit is normalised to
# nanoseconds.  Decoding stays as the design says (the underlying integer):
# what was sent as a timestamp comes back as the int64 nanoseconds, not as
# datetime64 (see tickets/design/ipc-kdb.md section 4).
_KDB_EPOCH_NS = 946684800 * 10 ** 9          # 2000-01-01 in Unix nanoseconds
_FINER_THAN_NS = ("ps", "fs", "as")


def _temporal_ns(a):
    """datetime64/timedelta64 array as (int64 kdb+ nanoseconds, kdb+ type); NaT is INT64_MIN."""
    is_time = a.dtype.kind == "M"
    unit = np.datetime_data(a.dtype)[0]
    if not is_time and unit in ("Y", "M"):
        raise IPCTypeError("cannot send numpy %s array (a calendar unit has no fixed length)" % a.dtype)
    try:
        with np.errstate(all="ignore"):
            ns = a.astype("datetime64[ns]" if is_time else "timedelta64[ns]")
    except OverflowError:
        raise IPCRangeError("%s value outside the nanosecond range of a kdb+ temporal" % a.dtype) from None
    except (TypeError, ValueError):
        raise IPCTypeError("cannot send numpy %s array" % a.dtype) from None
    nat = np.isnat(a)
    if unit not in _FINER_THAN_NS and unit != "generic":
        # an overflow of the nanosecond range wraps silently in numpy: detect it
        try:
            with np.errstate(all="ignore"):
                back = ns.astype(a.dtype)
            same = np.array_equal(back[~nat], a[~nat])
        except OverflowError:
            same = False
        if not same:
            raise IPCRangeError("%s value outside the nanosecond range of a kdb+ temporal" % a.dtype)
    v = ns.view("i8")
    if is_time:
        if a.size and (~nat).any():
            live = v[~nat]
            if int(live.min()) < _INT64_MIN + _KDB_EPOCH_NS:
                raise IPCRangeError("datetime64 value before the kdb+ range")
        with np.errstate(over="ignore"):
            v = v - np.int64(_KDB_EPOCH_NS)
    v = np.where(nat, np.int64(_INT64_MIN), v)
    return v, (12 if is_time else 16)
_PY_FLOATS = {float, np.float64}


def _function_display(fn):
    if isinstance(fn, KGLambda):
        arity = fn.get_arity()
    else:
        arity = getattr(fn, "arity", 0)
        args = getattr(fn, "args", None)
        if type(args) is list:
            # a projection: the arguments it still takes
            holes = sum(1 for q in args if q is None)
            if holes:
                arity = holes
    if not isinstance(arity, int) or arity < 0:
        arity = 0
    names = (":nilad", ":monad", ":dyad", ":triad")
    return names[arity] if arity < 4 else ":fn/%d" % arity


class _Encoder:
    def __init__(self, widths, fn_text):
        self.ibits, self.fbits = _norm_widths(widths)
        self.fn_text = fn_text if fn_text is not None else _fn_text_hook
        self.out = []   # bytes or memoryviews, joined once at the end

    # -- helpers ------------------------------------------------------------

    def _count(self, n):
        if n > _INT32_MAX:
            raise IPCRangeError("too many elements for one kdb+ vector")
        return n

    def _bytes_text(self, s, what):
        try:
            return s.encode("utf-8")
        except UnicodeEncodeError:
            raise IPCTypeError("cannot send %s: not valid text" % what) from None

    def _sym_bytes(self, s):
        b = self._bytes_text(s, "symbol")
        if b"\x00" in b:
            raise IPCTypeError("cannot send a symbol containing NUL")
        return b

    def _charvec(self, b):
        self.out.append(_VEC_HDR.pack(10, 0, self._count(len(b))))
        self.out.append(b)

    # -- atoms --------------------------------------------------------------

    def _int_atom(self, v):
        v = int(v)
        if self.ibits == 32:
            if not _INT32_MIN <= v <= _INT32_MAX:
                raise IPCRangeError("integer %d does not fit 32 bits" % v)
            self.out.append(_PACK_ATOM[-6].pack(-6, v))
        else:
            if not _INT64_MIN <= v <= _INT64_MAX:
                raise IPCRangeError("integer does not fit 64 bits")
            self.out.append(_PACK_ATOM[-7].pack(-7, v))

    def _real_atom(self, v):
        with np.errstate(over="ignore"):
            self.out.append(_PACK_ATOM[-8].pack(-8, float(np.float32(v))))

    def _float_atom(self, v):
        if self.fbits == 32:
            self._real_atom(v)
        else:
            self.out.append(_PACK_ATOM[-9].pack(-9, float(v)))

    def _numpy_scalar(self, v):
        k = v.dtype.kind
        sz = v.dtype.itemsize
        if k == "b":
            self.out.append(_PACK_ATOM[-1].pack(-1, 1 if v else 0))
        elif k == "i":
            if sz == 8:
                self._int_atom(v)
            elif sz == 4:
                self.out.append(_PACK_ATOM[-6].pack(-6, int(v)))
            else:                       # int16, and int8 (kdb+ bytes are unsigned)
                self.out.append(_PACK_ATOM[-5].pack(-5, int(v)))
        elif k == "u":
            if sz == 1:
                self.out.append(_PACK_ATOM[-4].pack(-4, int(v)))
            elif sz == 2:
                self.out.append(_PACK_ATOM[-6].pack(-6, int(v)))
            else:
                self._int_atom(v)
        elif k == "f":
            if sz == 4:
                self._real_atom(v)
            else:
                self._float_atom(v)
        else:
            raise IPCTypeError("cannot send numpy %s value" % v.dtype)

    # -- vectors ------------------------------------------------------------

    def _vector(self, a):
        """1-D numeric/bool numpy array as a typed vector (one buffer copy)."""
        k = a.dtype.kind
        sz = a.dtype.itemsize
        if k == "b":
            t, wire = 1, "u1"
            a = a.view("u1")
        elif k == "i":
            if sz == 8:
                t, wire = (6, "<i4") if self.ibits == 32 else (7, "<i8")
            elif sz == 4:
                t, wire = 6, "<i4"
            elif sz == 2:
                t, wire = 5, "<i2"
            else:
                t, wire = 5, "<i2"      # int8 widens to short: bytes are unsigned
        elif k == "u":
            if sz == 1:
                t, wire = 4, "u1"
            elif sz == 2:
                t, wire = 6, "<i4"
            elif sz == 4:
                t, wire = (6, "<i4") if self.ibits == 32 else (7, "<i8")
            else:
                if a.size and int(a.max()) > _INT64_MAX:
                    raise IPCRangeError("unsigned integer does not fit 64 bits")
                t, wire = (6, "<i4") if self.ibits == 32 else (7, "<i8")
        elif k == "f":
            if sz == 4 or (sz == 8 and self.fbits == 32):
                t, wire = 8, "<f4"
            else:
                t, wire = 9, "<f8"
        else:
            raise IPCTypeError("cannot send numpy %s array" % a.dtype)
        if wire == "<i4" and sz > 4 and a.size:
            if int(a.min()) < _INT32_MIN or int(a.max()) > _INT32_MAX:
                raise IPCRangeError("integer array does not fit 32 bits")
        n = self._count(a.shape[0])
        # same dtype and contiguous: no copy here; the join below copies once
        with np.errstate(over="ignore"):
            w = np.ascontiguousarray(a, dtype=wire)
        self.out.append(_VEC_HDR.pack(t, 0, n))
        self.out.append(memoryview(w.view("u1")))

    def _sym_vector(self, items):
        self.out.append(_VEC_HDR.pack(11, 0, self._count(len(items))))
        self.out.append(b"".join(self._sym_bytes(s) + b"\x00" for s in items))

    def _py_vector(self, items, t):
        n = self._count(len(items))
        if t == 7:
            if self.ibits == 32:
                lo, hi, code, tt = _INT32_MIN, _INT32_MAX, "<i4", 6
            else:
                lo, hi, code, tt = _INT64_MIN, _INT64_MAX, "<i8", 7
            for x in items:
                if not lo <= x <= hi:
                    raise IPCRangeError("integer does not fit %d bits" % self.ibits)
            arr = np.array(items, dtype=code)
            self.out.append(_VEC_HDR.pack(tt, 0, n))
        else:
            code, tt = ("<f4", 8) if self.fbits == 32 else ("<f8", 9)
            with np.errstate(over="ignore"):
                arr = np.array(items, dtype=code)
            self.out.append(_VEC_HDR.pack(tt, 0, n))
        self.out.append(memoryview(arr.view("u1")))

    # -- lists --------------------------------------------------------------

    def _general(self, items, depth):
        if depth >= MAX_DEPTH:
            raise IPCRangeError("nesting deeper than %d" % MAX_DEPTH)
        self.out.append(_VEC_HDR.pack(0, 0, self._count(len(items))))
        for x in items:
            self.obj(x, depth + 1)

    def _seq(self, items, depth):
        """A Python sequence: typed vector when homogeneous, else general list."""
        if not isinstance(items, (list, tuple)):
            items = list(items)
        if items:
            types = {type(x) for x in items}
            if types == {int}:
                return self._py_vector(items, 7)
            if types <= _PY_FLOATS:
                return self._py_vector(items, 9)
            if types == {KGSym}:
                return self._sym_vector(items)
        self._general(items, depth)

    def _array(self, a, depth):
        if a.ndim == 0:
            return self.obj(a[()], depth)
        k = a.dtype.kind
        if k in "Mm" and a.ndim == 1:
            ns, t = _temporal_ns(a)
            self.out.append(_VEC_HDR.pack(t, 0, self._count(a.shape[0])))
            self.out.append(memoryview(np.ascontiguousarray(ns, dtype="<i8").view("u1")))
            return None
        if a.ndim == 1 and k in "biuf":
            if a.shape[0] == 0:
                # empty list: an empty general list, as for Klong's []
                return self._general((), depth)
            return self._vector(a)
        if k == "U" or k == "S":
            return self._seq(a.tolist(), depth)
        if k in "biufOMm":
            if a.ndim == 1:
                return self._seq(list(a), depth)
            # rank 2 or more: a general list of rows, encoded recursively
            if depth >= MAX_DEPTH:
                raise IPCRangeError("nesting deeper than %d" % MAX_DEPTH)
            self.out.append(_VEC_HDR.pack(0, 0, self._count(a.shape[0])))
            for row in a:
                self._array(row, depth + 1)
            return None
        raise IPCTypeError("cannot send numpy %s array" % a.dtype)

    def _dict(self, d, depth):
        if depth >= MAX_DEPTH:
            raise IPCRangeError("nesting deeper than %d" % MAX_DEPTH)
        self.out.append(b"\x63")
        self._seq(list(d.keys()), depth + 1)
        self._seq(list(d.values()), depth + 1)

    def _table(self, names, cols, depth):
        """A table: 98, attribute byte, then a dictionary of column names to columns."""
        if depth >= MAX_DEPTH:
            raise IPCRangeError("nesting deeper than %d" % MAX_DEPTH)
        rows = None
        arrays = []
        for c in cols:
            if isinstance(c, str):
                a = c
            else:
                a = np.asarray(c)
                if a.dtype.kind not in "biufMm":
                    a = np.asarray(c, dtype=object)
                if a.ndim != 1:
                    raise IPCTypeError("table column is not a list")
            if rows is None:
                rows = len(a)
            elif len(a) != rows:
                raise IPCTypeError("table columns differ in length")
            arrays.append(a)
        self.out.append(b"\x62\x00\x63")
        self._sym_vector([str(n) for n in names])
        self.out.append(_VEC_HDR.pack(0, 0, self._count(len(arrays))))
        for a in arrays:
            self.obj(a, depth + 1)

    def _function(self, fn):
        if isinstance(fn, KGFnWrapper):
            fn = fn.fn
        text = None
        if self.fn_text is not None:
            text = self.fn_text(fn)
        if text is None:
            text = _function_display(fn)
        self._lambda(b"kg", self._bytes_text(text, "function text"))

    def _lambda(self, ctx, text):
        if b"\x00" in ctx:
            raise IPCTypeError("lambda context contains NUL")
        self.out.append(b"\x64" + ctx + b"\x00")
        self._charvec(text)

    # -- dispatch -----------------------------------------------------------

    def obj(self, v, depth=0):
        if v is None:
            raise IPCTypeError("cannot send None")
        if isinstance(v, KGUndefined):
            self.out.append(GENERIC_NULL)
        elif isinstance(v, KGSym):
            self.out.append(b"\xf5" + self._sym_bytes(v) + b"\x00")
        elif isinstance(v, KGChar):
            b = self._bytes_text(v, "char")
            if len(v) == 1 and b[0] < 0x80:
                self.out.append(b"\xf6" + b)
            else:
                self._charvec(b)        # non-ASCII chars go as a char vector
        elif isinstance(v, str):
            self._charvec(self._bytes_text(v, "string"))
        elif isinstance(v, (np.datetime64, np.timedelta64)):    # before np.integer: timedelta64 is one
            ns, t = _temporal_ns(np.asarray(v))
            self.out.append(_PACK_ATOM[-t].pack(-t, int(ns[()])))
        elif isinstance(v, (bool, np.bool_)):
            self.out.append(_PACK_ATOM[-1].pack(-1, 1 if v else 0))
        elif isinstance(v, int):
            self._int_atom(v)
        elif isinstance(v, np.integer):
            self._numpy_scalar(v)
        elif isinstance(v, float):           # includes np.float64
            self._float_atom(v)
        elif isinstance(v, np.floating):
            self._numpy_scalar(v)
        elif isinstance(v, np.ndarray):
            self._array(v, depth)
        elif isinstance(v, KdbTable):
            self._table(list(v.keys()), list(v.values()), depth)
        elif hasattr(v, "get_dataframe") and hasattr(v, "columns"):
            # a klongpy.db Table, recognised by shape so pandas is not imported here
            df = v.get_dataframe()
            self._table(list(v.columns), [df[c].values for c in v.columns], depth)
        elif isinstance(v, Mapping):
            self._dict(v, depth)
        elif isinstance(v, (list, tuple)):
            self._seq(v, depth)
        elif isinstance(v, KdbLambda):
            ctx = v.context_raw if v.context_raw is not None else self._bytes_text(v.context, "context")
            txt = v.text_raw if v.text_raw is not None else self._bytes_text(v.text, "function text")
            self._lambda(ctx, txt)
        elif isinstance(v, KdbOpaque):
            self.out.append(v.raw)
        elif isinstance(v, KdbError):
            b = v.text_raw if v.text_raw is not None else self._bytes_text(v.text, "error text")
            if b"\x00" in b:
                raise IPCTypeError("error text contains NUL")
            self.out.append(b"\x80" + b + b"\x00")
        elif isinstance(v, (KGFn, KGLambda, KGOp, KGAdverb, KGFnWrapper)):
            self._function(v)
        else:
            raise IPCTypeError("cannot send a value of kind %s" % type(v).__name__)


def encode(value, widths=64, *, fn_text=None):
    """Encode one Klong value as a kdb+ object (bytes after the message header).

    ``widths`` is 64 (long/float; default), 32 (int/real, small builds) or an
    ``(int_bits, float_bits)`` pair.  ``fn_text(fn)`` may return source text
    for a function (default: the hook of :func:`set_function_text_hook`, else
    the display form).  Raises :class:`IPCTypeError` for values with no kdb+
    form and :class:`IPCRangeError` for numbers that do not fit.
    """
    e = _Encoder(widths, fn_text)
    e.obj(value, 0)
    return b"".join(e.out)


def encode_call(items, widths=64, *, fn_text=None):
    """Encode a call envelope: always a general list, never a symbol vector."""
    e = _Encoder(widths, fn_text)
    e._general(list(items), 0)
    return b"".join(e.out)


# ----------------------------------------------------------------------------
# decoder
# ----------------------------------------------------------------------------

# type -> (struct char, size) for atoms and for the elements of vectors
_ATOM_FMT = {
    1: ("B", 1), 4: ("B", 1), 5: ("h", 2), 6: ("i", 4), 7: ("q", 8),
    8: ("f", 4), 9: ("d", 8), 12: ("q", 8), 13: ("i", 4), 14: ("i", 4),
    15: ("d", 8), 16: ("q", 8), 17: ("i", 4), 18: ("i", 4), 19: ("i", 4),
}
_ELEM_SIZE = {1: 1, 2: 16, 4: 1, 5: 2, 6: 4, 7: 8, 8: 4, 9: 8, 10: 1, 11: 1,
              12: 8, 13: 4, 14: 4, 15: 8, 16: 8, 17: 4, 18: 4, 19: 4}
# numpy wire dtype (byte order added at runtime) and "is a float" per vector type
_NP_KIND = {1: "u1", 4: "u1", 5: "i2", 6: "i4", 7: "i8", 8: "f4", 9: "f8",
            12: "i8", 13: "i4", 14: "i4", 15: "f8", 16: "i8", 17: "i4",
            18: "i4", 19: "i4"}
_FLOAT_TYPES = {8, 9, 15}

_ERR_TRUNC = "message truncated"


class _Decoder:
    def __init__(self, data, little, widths):
        self.buf = data
        self.n = len(data)
        self.pos = 0
        self.little = bool(little)
        self.o = "<" if self.little else ">"
        self.ibits, self.fbits = _norm_widths(widths)
        self.i32 = struct.Struct(self.o + "i")
        self._atoms = {t: struct.Struct(self.o + f) for t, (f, _) in _ATOM_FMT.items()}

    # -- primitive reads ----------------------------------------------------

    def _byte(self):
        if self.pos >= self.n:
            raise ProtocolError(_ERR_TRUNC)
        b = self.buf[self.pos]
        self.pos += 1
        return b

    def _take(self, k):
        end = self.pos + k
        if k < 0 or end > self.n:
            raise ProtocolError(_ERR_TRUNC)
        chunk = self.buf[self.pos:end]
        self.pos = end
        return chunk

    def _count(self, min_elem):
        if self.pos + 4 > self.n:
            raise ProtocolError(_ERR_TRUNC)
        (c,) = self.i32.unpack_from(self.buf, self.pos)
        self.pos += 4
        # length checks before any allocation: the count must be possible in
        # the bytes that are left (design section 9)
        if c < 0 or c * min_elem > self.n - self.pos:
            raise ProtocolError("vector count %d exceeds the message" % c)
        return c

    def _cstring(self):
        """Bytes up to the next NUL (consumed)."""
        end = self.buf.find(b"\x00", self.pos)
        if end < 0:
            raise ProtocolError("unterminated string")
        chunk = bytes(self.buf[self.pos:end])
        self.pos = end + 1
        return chunk

    @staticmethod
    def _text(b):
        return b.decode("utf-8", "replace")

    def _fit(self, v):
        if self.ibits == 32 and not _INT32_MIN <= v <= _INT32_MAX:
            raise IPCRangeError("integer %d does not fit 32 bits" % v)
        return v

    # -- objects ------------------------------------------------------------

    def obj(self, depth):
        t = self._byte()
        if t >= 128:
            t -= 256
        if t < 0:
            return self._atom(-t)
        if t == 0:
            return self._make_list(self._general_items(depth))
        if t <= 19:
            return self._vector(t)
        if t == 98:
            return self._table(depth)
        if t == 99 or t == 127:
            return self._dict(depth)
        if t == 100:
            return self._lambda()
        if 101 <= t <= 111:
            if t == 101 and self.pos < self.n and self.buf[self.pos] == 0:
                self.pos += 1
                return KLONG_UNDEFINED
            self.pos -= 1
            out = []
            self.copy(out, depth)
            return KdbOpaque(t, b"".join(out))
        raise ProtocolError("unsupported type byte %d" % t)

    def _atom(self, t):
        if t == 128:                              # -128: error
            raw = self._cstring()
            return KdbError(self._text(raw), raw)
        if t == 11:
            return KGSym(self._text(self._cstring()))
        if t == 10:
            return KGChar(self._text(bytes(self._take(1))))
        if t == 2:
            raw = self._take(16)
            return np.frombuffer(raw, dtype="u1").astype(self._int_dtype())
        st = self._atoms.get(t)
        if st is None:
            raise ProtocolError("unsupported atom type %d" % t)
        if self.pos + st.size > self.n:
            raise ProtocolError(_ERR_TRUNC)
        (v,) = st.unpack_from(self.buf, self.pos)
        self.pos += st.size
        if t in _FLOAT_TYPES:
            return float(np.float32(v)) if self.fbits == 32 else float(v)
        return self._fit(int(v)) if t in (7, 12, 16) else int(v)

    def _int_dtype(self):
        return np.int32 if self.ibits == 32 else np.int64

    def _float_dtype(self):
        return np.float32 if self.fbits == 32 else np.float64

    def _vector(self, t):
        if self.pos >= self.n:
            raise ProtocolError(_ERR_TRUNC)
        self.pos += 1                              # attribute byte: ignored
        size = _ELEM_SIZE.get(t)
        if size is None:
            raise ProtocolError("unsupported vector type %d" % t)
        n = self._count(size)
        if t == 11:
            return self._symbols(n)
        if t == 10:
            return self._text(bytes(self._take(n)))
        start = self.pos
        if n * size > self.n - start:
            raise ProtocolError(_ERR_TRUNC)
        self.pos = start + n * size
        if t == 2:
            return np.frombuffer(self.buf, dtype="u1", count=16 * n, offset=start) \
                .reshape(n, 16).astype(self._int_dtype())
        kind = _NP_KIND[t]
        wire = np.dtype(kind) if kind == "u1" else np.dtype(self.o + kind)
        # one copy: straight from the receive buffer into the result dtype
        arr = np.frombuffer(self.buf, dtype=wire, count=n, offset=start)
        if t in _FLOAT_TYPES:
            return arr.astype(self._float_dtype())
        if self.ibits == 32 and size == 8 and n:
            if int(arr.min()) < _INT32_MIN or int(arr.max()) > _INT32_MAX:
                raise IPCRangeError("integer vector does not fit 32 bits")
        return arr.astype(self._int_dtype())

    def _symbols(self, n):
        out = np.empty(n, dtype=object)
        buf = self.buf
        pos = self.pos
        find = buf.find
        for i in range(n):
            end = find(b"\x00", pos)
            if end < 0:
                raise ProtocolError("unterminated symbol")
            out[i] = KGSym(buf[pos:end].decode("utf-8", "replace"))
            pos = end + 1
        self.pos = pos
        return out

    def _general_items(self, depth):
        if depth >= MAX_DEPTH:
            raise ProtocolError("nesting deeper than %d" % MAX_DEPTH)
        if self.pos >= self.n:
            raise ProtocolError(_ERR_TRUNC)
        self.pos += 1                              # attribute byte
        n = self._count(2)                         # every object is >= 2 bytes
        return [self.obj(depth + 1) for _ in range(n)]

    def _make_list(self, items):
        """General-list items to a Klong list (array), the ``kg_asarray`` rule."""
        if not items:
            return np.empty(0, dtype=np.float64)
        types = {type(x) for x in items}
        if len(types) == 1:
            ty = next(iter(types))
            if ty is int:
                return np.array(items, dtype=self._int_dtype())
            if ty is float:
                return np.array(items, dtype=self._float_dtype())
            if ty is np.ndarray:
                first = items[0]
                if first.dtype.kind in "if" and first.ndim >= 1 and all(
                        x.dtype == first.dtype and x.shape == first.shape for x in items):
                    return np.stack(items)
        out = np.empty(len(items), dtype=object)
        for i, x in enumerate(items):
            out[i] = x
        return out

    # -- sequences used as dictionary keys and values ------------------------

    def _seq_items(self, depth):
        """Next object, which must be a list (type 0..19), as a Python list."""
        if self.pos >= self.n:
            raise ProtocolError(_ERR_TRUNC)
        t = self.buf[self.pos]
        if t == 0:
            self.pos += 1
            return self._general_items(depth)
        if 1 <= t <= 19:
            self.pos += 1
            v = self._vector(t)
            if isinstance(v, str):
                return [KGChar(c) for c in v]
            if v.dtype == object:
                return list(v)
            if v.ndim == 1:
                return v.tolist()
            return list(v)
        raise ProtocolError("dictionary part is not a list (type %d)" % t)

    @staticmethod
    def _hashable(k):
        if isinstance(k, np.ndarray):
            k = list(k) if k.dtype == object else k.tolist()
        if isinstance(k, list):
            return tuple(_Decoder._hashable(x) for x in k)
        if isinstance(k, dict):
            raise ProtocolError("dictionary used as a dictionary key")
        return k

    def _dict(self, depth):
        if depth >= MAX_DEPTH:
            raise ProtocolError("nesting deeper than %d" % MAX_DEPTH)
        if self.pos >= self.n:
            raise ProtocolError(_ERR_TRUNC)
        if self.buf[self.pos] == 98:               # keyed table: table!table
            self.pos += 1
            keys = self._table(depth + 1)
            if self._byte() != 98:
                raise ProtocolError("keyed table values are not a table")
            vals = self._table(depth + 1)
            out = np.empty(2, dtype=object)
            out[0] = keys
            out[1] = vals
            return out
        keys = self._seq_items(depth + 1)
        vals = self._seq_items(depth + 1)
        if len(keys) != len(vals):
            raise ProtocolError("dictionary keys and values differ in length")
        try:
            return {self._hashable(k): v for k, v in zip(keys, vals)}
        except TypeError:
            raise ProtocolError("unhashable dictionary key") from None

    def _table(self, depth):
        if depth >= MAX_DEPTH:
            raise ProtocolError("nesting deeper than %d" % MAX_DEPTH)
        self._take(1)                              # attribute byte: ignored
        if self._byte() not in (99, 127):
            raise ProtocolError("table without its dictionary")
        if self._byte() != 11:
            raise ProtocolError("table column names are not a symbol vector")
        names = self._vector(11)
        if self._byte() != 0:
            raise ProtocolError("table columns are not a general list")
        cols = self._general_items(depth + 1)
        if len(cols) != len(names):
            raise ProtocolError("table has %d names and %d columns" % (len(names), len(cols)))
        rows = None
        for c in cols:
            if not isinstance(c, (np.ndarray, str)) or (isinstance(c, np.ndarray) and c.ndim == 0):
                raise ProtocolError("table column is not a list")
            if rows is None:
                rows = len(c)
            elif len(c) != rows:
                raise ProtocolError("table columns differ in length")
        if _table_factory is not None:
            return _table_factory(list(names), cols)
        return KdbTable((names[i], cols[i]) for i in range(len(names)))

    def _lambda(self):
        ctx = self._cstring()
        if self._byte() != 10:
            raise ProtocolError("lambda text is not a char vector")
        self._take(1)                              # attribute byte
        n = self._count(1)
        txt = bytes(self._take(n))
        return KdbLambda(self._text(ctx), self._text(txt), ctx, txt)

    # -- structural copy (opaque function objects) ---------------------------

    def _copy_swapped(self, out, nbytes, width):
        raw = self._take(nbytes)
        if self.little or width == 1:
            out.append(bytes(raw))
        else:
            a = np.frombuffer(raw, dtype=">u%d" % width)
            out.append(a.astype("<u%d" % width).tobytes())

    def copy(self, out, depth):
        """Validate one object and append its little-endian bytes to ``out``."""
        t = self._byte()
        out.append(bytes((t,)))
        if t >= 128:
            t -= 256
        if t < 0:
            t = -t
            if t == 128 or t == 11:
                out.append(self._cstring() + b"\x00")
            elif t == 2:
                out.append(bytes(self._take(16)))
            elif t in _ATOM_FMT:
                self._copy_swapped(out, _ATOM_FMT[t][1], _ATOM_FMT[t][1])
            elif t == 10:
                out.append(bytes(self._take(1)))
            else:
                raise ProtocolError("unsupported atom type %d" % t)
            return
        if t == 0 or 1 <= t <= 19:
            if t == 0 and depth >= MAX_DEPTH:
                raise ProtocolError("nesting deeper than %d" % MAX_DEPTH)
            out.append(bytes(self._take(1)))       # attribute
            if t == 0:
                n = self._count(2)
                out.append(struct.pack("<i", n))
                for _ in range(n):
                    self.copy(out, depth + 1)
            elif t == 11:
                n = self._count(1)
                out.append(struct.pack("<i", n))
                for _ in range(n):
                    out.append(self._cstring() + b"\x00")
            elif t in _ELEM_SIZE:
                size = _ELEM_SIZE[t]
                n = self._count(size)
                out.append(struct.pack("<i", n))
                self._copy_swapped(out, n * size, 1 if t in (1, 2, 4, 10) else size)
            else:
                raise ProtocolError("unsupported vector type %d" % t)
            return
        if depth >= MAX_DEPTH:
            raise ProtocolError("nesting deeper than %d" % MAX_DEPTH)
        if t == 98:
            out.append(bytes(self._take(1)))
            self.copy(out, depth + 1)
        elif t == 99 or t == 127:
            self.copy(out, depth + 1)
            self.copy(out, depth + 1)
        elif t == 100:
            out.append(self._cstring() + b"\x00")
            self.copy(out, depth + 1)
        elif 101 <= t <= 103:
            out.append(bytes(self._take(1)))
        elif t == 104 or t == 105:
            n = self._count(2)
            out.append(struct.pack("<i", n))
            for _ in range(n):
                self.copy(out, depth + 1)
        elif 106 <= t <= 111:
            self.copy(out, depth + 1)
        else:
            raise ProtocolError("unsupported type byte %d" % t)


def decode(data, little_endian=True, widths=64, offset=0):
    """Decode one object (the bytes after the 8-byte message header).

    ``little_endian`` is header byte 0 (1 = little).  ``offset`` is where the
    object starts (8 to skip a message header without copying).  The whole of
    ``data`` must be consumed.  Raises :class:`ProtocolError` for anything malformed;
    :class:`IPCRangeError` only when ``widths`` is 32 and a number does not fit.
    """
    if isinstance(data, memoryview):
        data = data.tobytes()
    elif not isinstance(data, (bytes, bytearray)):
        raise TypeError("decode needs bytes")
    if not 0 <= offset <= len(data):
        raise ValueError("offset out of range")
    d = _Decoder(data, little_endian, widths)
    d.pos = offset
    try:
        v = d.obj(0)
    except (ProtocolError, IPCRangeError):
        raise
    except (struct.error, IndexError, ValueError, OverflowError, MemoryError,
            RecursionError, UnicodeError, TypeError, KeyError) as exc:
        raise ProtocolError("malformed object: %s" % exc) from exc
    if d.pos != d.n:
        raise ProtocolError("%d bytes after the object" % (d.n - d.pos))
    return v
