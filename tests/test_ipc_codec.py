"""Tests of the kdb+ IPC codec and compression (steps 1 and 4).

Shared vectors: ``tests/ipc_vectors/vectors.json``.  Entry
fields: ``name``, ``klong`` (Klong source of the expected value, evaluated with
the interpreter), ``hex`` (the whole message, header included), ``mode``
(``roundtrip``, ``decode`` or ``reject``), ``widths`` (``64``, ``32`` or ``any``),
``source`` (``kx`` or ``derived``), optional ``type`` (message type, default 0),
optional ``error`` (``system`` on reject entries).  Values Klong source cannot
express carry ``expect``, a tagged JSON value (see ``from_expect``).

The codec is tested on the bytes after the header (byte order from header
byte 0, compressed messages through ``decompress`` first); header-level reject
entries are exercised by the session tests.

Hypothesis is not installed in this environment, so the property tests use
seeded random generation.
"""
import json
import math
import os
import random
import struct
import tracemalloc
import unittest

import numpy as np

from klongpy import KlongInterpreter
from klongpy.types import KGSym, KGChar, KGLambda, KGFn, KGOp, KLONG_UNDEFINED, KGUndefined
from klongpy.ipc import codec
from klongpy.ipc.codec import (
    encode, encode_call, decode, KdbLambda, KdbOpaque, KdbError, MAX_DEPTH,
)
from klongpy.ipc.compress import compress, decompress
from klongpy.ipc.errors import ProtocolError, IPCTypeError, IPCRangeError

VECTORS = os.path.join(os.path.dirname(__file__), "ipc_vectors", "vectors.json")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _kind(x):
    """Classify a scalar the way Klong tells them apart."""
    if isinstance(x, KGSym):
        return "sym"
    if isinstance(x, KGChar):
        return "char"
    if isinstance(x, str):
        return "str"
    if isinstance(x, (bool, int, np.integer)):
        return "int"
    if isinstance(x, (float, np.floating)):
        return "float"
    return None


def kg_equal(a, b):
    """Structural equality: lists and arrays compare as sequences, ints and
    floats are distinct, symbols/chars/strings are distinct, NaN equals NaN."""
    if isinstance(a, np.ndarray) and a.ndim == 0:
        a = a[()]
    if isinstance(b, np.ndarray) and b.ndim == 0:
        b = b[()]
    seq = (list, tuple, np.ndarray)
    if isinstance(a, seq) or isinstance(b, seq):
        if not (isinstance(a, seq) and isinstance(b, seq)) or len(a) != len(b):
            return False
        if isinstance(a, np.ndarray) and a.dtype != object:
            a = a.tolist()
        if isinstance(b, np.ndarray) and b.dtype != object:
            b = b.tolist()
        return all(kg_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)) or len(a) != len(b):
            return False
        for k, v in a.items():
            if k not in b or not kg_equal(v, b[k]):
                return False
        return True
    ka, kb = _kind(a), _kind(b)
    if ka is not None or kb is not None:
        if ka != kb:
            return False
        if ka == "float":
            return (math.isnan(a) and math.isnan(b)) or a == b
        return str.__eq__(a, b) if ka in ("sym", "char", "str") else a == b
    if isinstance(a, KGUndefined) or isinstance(b, KGUndefined):
        return a is b
    return a == b and type(a) is type(b)


def from_expect(e):
    """Tagged JSON value to a Klong value (the ``expect`` field of an entry)."""
    if "int" in e:
        return e["int"]
    if "float" in e:
        return {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}.get(e["float"], e["float"])
    if "str" in e:
        return e["str"]
    if "char" in e:
        return KGChar(e["char"])
    if "sym" in e:
        return KGSym(e["sym"])
    if "undefined" in e:
        return KLONG_UNDEFINED
    if "error" in e:
        return KdbError(e["error"])
    if "lambda" in e:
        return KdbLambda(*e["lambda"])
    if "opaque" in e:
        return KdbOpaque(e["opaque"][0], bytes.fromhex(e["opaque"][1]))
    if "dict" in e:
        return {from_expect(k): from_expect(v) for k, v in e["dict"]}
    if "array" in e:
        dt = e["dtype"]
        if dt == "object":
            out = np.empty(len(e["array"]), dtype=object)
            for i, x in enumerate(e["array"]):
                out[i] = from_expect(x)
            return out
        items = [from_expect({"float": x}) if isinstance(x, str) else x for x in e["array"]]
        return np.array(items, dtype=dt)
    raise ValueError(e)


# The codec is plain numpy on every backend (klongpy/ipc/codec.py, "Torch isolation"): the
# expected values of the vectors are numpy values, also when the suite runs with --backend torch.
# The endpoint converts tensors before encoding; tests/test_torch_regressions.py covers that.
_klong = KlongInterpreter(backend='numpy')


def expected_value(entry):
    if "expect" in entry:
        return from_expect(entry["expect"])
    return _klong(entry["klong"])


def load_vectors():
    with open(VECTORS) as f:
        return json.load(f)


def widths_of(entry):
    return (64, 32) if entry["widths"] == "any" else (int(entry["widths"]),)


def split(entry):
    msg = bytes.fromhex(entry["hex"])
    return msg, msg[0] == 1, msg[:8]


def body_of(entry):
    """Message body (after the header), decompressing when flagged."""
    msg, little, hdr = split(entry)
    if hdr[2] == 1:
        msg = decompress(msg)
    return msg[8:], little


def header_consistent(msg):
    return (len(msg) >= 8 and msg[0] in (0, 1) and msg[1] <= 2 and msg[2] <= 1 and msg[3] == 0
            and struct.unpack("<I" if msg[0] == 1 else ">I", msg[4:8])[0] == len(msg))


# ----------------------------------------------------------------------------
# vectors
# ----------------------------------------------------------------------------

class TestVectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vectors = load_vectors()

    def test_file_shape(self):
        names = [v["name"] for v in self.vectors]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sorted(v["name"] for v in self.vectors if v["source"] == "kx"),
                         ["kx-dict-sym-int", "kx-int-atom", "kx-int-vector", "kx-lambda-root"])
        for v in self.vectors:
            self.assertIn(v["mode"], ("roundtrip", "decode", "reject"), v["name"])
            self.assertIn(v["widths"], ("64", "32", "any"), v["name"])
            self.assertIn(v["source"], ("kx", "derived"), v["name"])
            if v["mode"] == "reject":
                self.assertEqual(v["error"], "system")
            else:
                self.assertTrue(header_consistent(bytes.fromhex(v["hex"])), v["name"])
                self.assertEqual(bytes.fromhex(v["hex"])[1], v.get("type", 0), v["name"])

    def test_roundtrip(self):
        for v in self.vectors:
            if v["mode"] != "roundtrip":
                continue
            body, little = body_of(v)
            self.assertTrue(little)
            value = expected_value(v)
            for w in widths_of(v):
                with self.subTest(name=v["name"], widths=w):
                    self.assertEqual(encode(value, w), body)
                    got = decode(body, little, w)
                    self.assertTrue(kg_equal(got, value), (got, value))
                    # and the extra copy of the bytes decodes the same
                    self.assertTrue(kg_equal(decode(bytearray(body), little, w), value))

    def test_decode(self):
        for v in self.vectors:
            if v["mode"] != "decode":
                continue
            body, little = body_of(v)
            value = expected_value(v)
            for w in widths_of(v):
                with self.subTest(name=v["name"], widths=w):
                    got = decode(body, little, w)
                    self.assertTrue(kg_equal(got, value), (got, value))

    def test_reject(self):
        tested = 0
        for v in self.vectors:
            if v["mode"] != "reject":
                continue
            msg, little, hdr = split(v)
            compressed = len(msg) >= 8 and msg[2] == 1
            if not compressed and not header_consistent(msg):
                continue                      # header-level entry: session tests
            tested += 1
            for w in widths_of(v):
                with self.subTest(name=v["name"], widths=w):
                    with self.assertRaises(ProtocolError):
                        if compressed:
                            decompress(msg)
                        else:
                            decode(msg[8:], little, w)
        self.assertGreater(tested, 20)

    def test_vectors_decode_independent_of_input_type(self):
        for v in self.vectors:
            if v["mode"] == "reject":
                continue
            body, little = body_of(v)
            a = decode(body, little)
            b = decode(memoryview(body), little)
            self.assertTrue(kg_equal(a, b), v["name"])


# ----------------------------------------------------------------------------
# encoder behaviour
# ----------------------------------------------------------------------------

class TestEncode(unittest.TestCase):
    def test_atoms(self):
        self.assertEqual(encode(5), b"\xf9" + struct.pack("<q", 5))
        self.assertEqual(encode(5, 32), b"\xfa" + struct.pack("<i", 5))
        self.assertEqual(encode(2.5), b"\xf7" + struct.pack("<d", 2.5))
        self.assertEqual(encode(2.5, 32), b"\xf8" + struct.pack("<f", 2.5))
        self.assertEqual(encode(KGSym("a")), b"\xf5a\x00")
        self.assertEqual(encode("hi"), b"\x0a\x00\x02\x00\x00\x00hi")
        self.assertEqual(encode(KGChar("x")), b"\xf6x")
        self.assertEqual(encode(KLONG_UNDEFINED), b"\x65\x00")
        self.assertEqual(encode(True), b"\xff\x01")

    def test_numpy_scalars(self):
        self.assertEqual(encode(np.int64(3)), encode(3))
        self.assertEqual(encode(np.int32(3)), b"\xfa" + struct.pack("<i", 3))
        self.assertEqual(encode(np.int16(3)), b"\xfb" + struct.pack("<h", 3))
        self.assertEqual(encode(np.uint8(3)), b"\xfc\x03")
        self.assertEqual(encode(np.float64(1.5)), encode(1.5))
        self.assertEqual(encode(np.float32(1.5)), b"\xf8" + struct.pack("<f", 1.5))
        self.assertEqual(encode(np.bool_(True)), b"\xff\x01")
        self.assertEqual(encode(np.array(7)), encode(7))          # zero-dimensional array is an atom
        self.assertEqual(encode(np.str_("ab")), encode("ab"))

    def test_non_ascii_char_goes_as_char_vector(self):
        self.assertEqual(encode(KGChar("é")), b"\x0a\x00\x02\x00\x00\x00\xc3\xa9")
        self.assertEqual(decode(encode(KGChar("é"))), "é")

    def test_arrays_other_widths(self):
        def hdr(t, n):
            return bytes([t, 0]) + struct.pack("<i", n)
        self.assertEqual(encode(np.array([1, 2], dtype=np.int32)), hdr(6, 2) + struct.pack("<ii", 1, 2))
        self.assertEqual(encode(np.array([1, 2], dtype=np.int16)), hdr(5, 2) + struct.pack("<hh", 1, 2))
        self.assertEqual(encode(np.array([1, 2], dtype=np.uint8)), hdr(4, 2) + b"\x01\x02")
        self.assertEqual(encode(np.array([True, False])), hdr(1, 2) + b"\x01\x00")
        self.assertEqual(encode(np.array([-1, 2], dtype=np.int8)), hdr(5, 2) + struct.pack("<hh", -1, 2))
        self.assertEqual(encode(np.array([1.5], dtype=np.float32)), hdr(8, 1) + struct.pack("<f", 1.5))
        self.assertEqual(encode(np.array([1, 2], dtype=np.uint16)), hdr(6, 2) + struct.pack("<ii", 1, 2))
        self.assertEqual(encode(np.array([1, 2], dtype=np.int64), 32), hdr(6, 2) + struct.pack("<ii", 1, 2))
        self.assertEqual(encode(np.array([1.5], dtype=np.float64), 32), hdr(8, 1) + struct.pack("<f", 1.5))
        # big-endian and strided inputs still produce little-endian bytes
        be = np.array([1, 2, 3], dtype=">i8")
        self.assertEqual(encode(be), encode(np.array([1, 2, 3])))
        strided = np.arange(10)[::3]
        self.assertEqual(encode(strided), encode(np.array([0, 3, 6, 9])))

    def test_range_errors(self):
        with self.assertRaises(IPCRangeError):
            encode(2 ** 63)
        with self.assertRaises(IPCRangeError):
            encode(-2 ** 63 - 1)
        encode(2 ** 63 - 1)
        encode(-2 ** 63)
        with self.assertRaises(IPCRangeError):
            encode(2 ** 31, 32)
        with self.assertRaises(IPCRangeError):
            encode(np.array([2 ** 40]), 32)
        with self.assertRaises(IPCRangeError):
            encode(np.array([2 ** 64 - 1], dtype=np.uint64))
        with self.assertRaises(IPCRangeError):
            encode([2 ** 63, 1])

    def test_type_errors(self):
        for bad in (None, object(), {1, 2}, np.array([1 + 2j]), np.array([(1, 2)], dtype=[("a", "i4"), ("b", "i4")])):
            with self.assertRaises(IPCTypeError, msg=repr(bad)):
                encode(bad)
        with self.assertRaises(IPCTypeError):
            encode(KGSym("a\x00b"))
        with self.assertRaises(IPCTypeError):
            encode("\ud800")
        with self.assertRaises(IPCTypeError) as cm:
            encode([1, object()])
        self.assertIn("object", str(cm.exception))

    def test_lists(self):
        self.assertEqual(encode([]), b"\x00\x00\x00\x00\x00\x00")
        self.assertEqual(encode(np.array([])), b"\x00\x00\x00\x00\x00\x00")
        self.assertEqual(encode((1, 2)), encode(np.array([1, 2])))
        self.assertEqual(encode([KGSym("a"), KGSym("b")]), b"\x0b\x00\x02\x00\x00\x00a\x00b\x00")
        self.assertEqual(encode(np.array(["ab", "c"], dtype=object)), encode(["ab", "c"]))
        self.assertEqual(encode(np.array(["ab", "c"])), encode(["ab", "c"]))     # numpy unicode
        mixed = encode([1, "a"])
        self.assertEqual(mixed[:6], b"\x00\x00\x02\x00\x00\x00")
        rank3 = np.arange(8).reshape(2, 2, 2)
        self.assertTrue(kg_equal(decode(encode(rank3)), rank3))
        self.assertEqual(encode(np.zeros((0, 3))), b"\x00\x00\x00\x00\x00\x00")

    def test_call_envelope_is_always_a_general_list(self):
        syms = [KGSym("f"), KGSym("x")]
        self.assertEqual(encode(syms)[0], 11)          # as data: a symbol vector
        env = encode_call(syms)
        self.assertEqual(env[0], 0)                    # as a call: general list of two symbol atoms
        self.assertEqual(env, b"\x00\x00\x02\x00\x00\x00\xf5f\x00\xf5x\x00")
        back = decode(env)
        self.assertTrue(kg_equal(back, np.array(syms, dtype=object)))
        self.assertEqual(encode_call([]), b"\x00\x00\x00\x00\x00\x00")

    def test_dict(self):
        d = {KGSym("a"): 1, KGSym("b"): 2}
        self.assertEqual(encode(d)[0], 99)
        self.assertEqual(encode({}), b"\x63" + encode([]) + encode([]))
        # insertion order is the wire order
        d2 = {KGSym("b"): 1, KGSym("a"): 2}
        self.assertTrue(encode(d2).index(b"b\x00") < encode(d2).index(b"a\x00"))
        # non-symbol keys, array values
        d3 = {1: np.array([1, 2]), 2: "x"}
        self.assertTrue(kg_equal(decode(encode(d3)), d3))

    def test_functions(self):
        add = KGLambda(lambda x, y: x + y)
        self.assertEqual(decode(encode(add)), KdbLambda("kg", ":dyad"))
        self.assertEqual(decode(encode(KGFn(KGOp("+", 2), [], 2))), KdbLambda("kg", ":dyad"))
        self.assertEqual(decode(encode(KGFn(None, [], 0))), KdbLambda("kg", ":nilad"))
        self.assertEqual(decode(encode(KGFn(None, [], 1))), KdbLambda("kg", ":monad"))
        self.assertEqual(decode(encode(KGFn(None, [], 3))), KdbLambda("kg", ":triad"))
        self.assertEqual(decode(encode(KGFn(None, [], 5))), KdbLambda("kg", ":fn/5"))
        fn = KGFn(None, [], 1)
        self.assertEqual(decode(encode(fn, fn_text=lambda f: "{x+1}")), KdbLambda("kg", "{x+1}"))
        self.assertEqual(decode(encode(fn, fn_text=lambda f: None)), KdbLambda("kg", ":monad"))
        previous = codec._fn_text_hook
        codec.set_function_text_hook(lambda f: "{hooked}")
        try:
            self.assertEqual(decode(encode(fn)), KdbLambda("kg", "{hooked}"))
        finally:
            codec.set_function_text_hook(previous)
        self.assertEqual(decode(encode(fn)), KdbLambda("kg", ":monad"))
        interp_fn = _klong("{x+y}")
        self.assertEqual(decode(encode(interp_fn)).context, "kg")

    def test_placeholders_reencode_exactly(self):
        lam = KdbLambda("d", "{x+y}")
        self.assertEqual(encode(lam), b"\x64d\x00\x0a\x00\x05\x00\x00\x00{x+y}")
        err = KdbError("type")
        self.assertEqual(encode(err), b"\x80type\x00")
        with self.assertRaises(IPCTypeError):
            encode(KdbError("a\x00b"))
        self.assertEqual(str(KdbOpaque(102, b"\x66\x01")), ":kdb-function")
        # a lambda with undecodable text bytes keeps them
        raw = b"\x64\x00\x0a\x00\x02\x00\x00\x00\xff\xfe"
        lam = decode(raw)
        self.assertEqual(encode(lam), raw)

    def test_depth_limit(self):
        v = 1
        for _ in range(MAX_DEPTH):
            v = [v, "x"]                 # mixed, so it stays a general list
        encode(v)
        with self.assertRaises(IPCRangeError):
            encode([v, "x"])
        d = {KGSym("a"): 1}
        for _ in range(MAX_DEPTH + 1):
            d = {KGSym("a"): d}
        with self.assertRaises(IPCRangeError):
            encode(d)
        cyc = []
        cyc.append(cyc)
        with self.assertRaises(IPCRangeError):
            encode(cyc)

    def test_one_copy_buffer(self):
        a = np.arange(1000, dtype=np.int64)
        e = encode(a)
        self.assertEqual(len(e), 6 + 8000)
        self.assertTrue(np.array_equal(np.frombuffer(e, dtype="<i8", offset=6), a))


class TestBytes(unittest.TestCase):
    """Python bytes and bytearray are a kdb+ byte vector (type 4); it decodes to an integer array."""

    def test_encode(self):
        self.assertEqual(bytes(encode(b"")), b"\x04\x00\x00\x00\x00\x00")
        self.assertEqual(bytes(encode(b"\x00\xff")), b"\x04\x00\x02\x00\x00\x00\x00\xff")
        self.assertEqual(bytes(encode(bytearray(b"\x00\xff"))), bytes(encode(b"\x00\xff")))
        self.assertEqual(bytes(encode(np.array([0, 255], dtype=np.uint8))), bytes(encode(b"\x00\xff")))

    def test_decode_vector_and_atom(self):
        v = decode(bytes([4, 0, 3, 0, 0, 0, 0, 1, 255]))
        self.assertEqual(list(v), [0, 1, 255])
        self.assertEqual(decode(bytes([4, 0, 0, 0, 0, 0])).shape, (0,))
        a = decode(bytes([0xfc, 7]))
        self.assertEqual(a, 7)
        self.assertIsInstance(a, int)

    def test_roundtrip_values(self):
        for b in (b"", b"\x00\xff", bytearray(range(256))):
            self.assertEqual(list(decode(bytes(encode(b)))), list(b))

    def test_nested(self):
        got = decode(bytes(encode([1, b"ab", [b"\xff"]])))
        self.assertEqual(got[0], 1)
        self.assertEqual(list(got[1]), [97, 98])
        self.assertEqual(list(got[2][0]), [255])
        d = decode(bytes(encode({KGSym("k"): b"\x01\x02", KGSym("j"): 5})))
        self.assertEqual(list(d[KGSym("k")]), [1, 2])

    def test_large_and_limit(self):
        big = bytes(range(256)) * 4096                      # 1 MiB
        body = bytes(encode(big))
        self.assertEqual(len(body), 6 + len(big))
        self.assertEqual(bytes(decode(body).astype(np.uint8)), big)

    def test_compressed(self):
        data = b"abcd" * 5000
        msg = _msg(bytes(encode(data)))
        c = compress(msg)
        self.assertEqual(c[2], 1)
        self.assertEqual(bytes(decode(decompress(c)[8:]).astype(np.uint8)), data)

    def test_other_kinds_still_a_type_error(self):
        for bad, kind in ((None, "None"), ({1}, "set"), (1j, "complex"), (object(), "object")):
            with self.assertRaises(IPCTypeError) as cm:
                encode(bad)
            self.assertEqual(str(cm.exception), "cannot send None" if bad is None else "cannot send a value of kind %s" % kind)


class TestTemporalEncode(unittest.TestCase):
    """numpy datetime64 is a kdb+ timestamp (12), timedelta64 a timespan (16); bytes computed by hand.

    kdb+ timestamp: int64 nanoseconds since 2000-01-01 (Unix 946684800 s). 2000-01-02 is
    86400e9 = 0x4E94914F0000 ns; 2000-01-01T00:00:01 is 1e9 = 0x3B9ACA00; NaT is INT64_MIN.
    """
    NULL = struct.pack("<q", -2 ** 63)

    def test_timestamp_vector_bytes(self):
        a = np.array(["2000-01-01", "2000-01-02", "NaT"], dtype="datetime64[ns]")
        want = (b"\x0c\x00" + struct.pack("<i", 3) + struct.pack("<q", 0) + struct.pack("<q", 86400 * 10 ** 9) + self.NULL)
        self.assertEqual(bytes(encode(a)), want)
        self.assertEqual(want[6:14], bytes(8))
        self.assertEqual(want[14:22], bytes.fromhex("00004f91944e0000"))

    def test_before_the_epoch_is_negative(self):
        a = np.array(["1999-12-31T23:59:59"], dtype="datetime64[s]")
        self.assertEqual(bytes(encode(a)), b"\x0c\x00" + struct.pack("<i", 1) + struct.pack("<q", -10 ** 9))

    def test_a_unix_time_value(self):
        # 2024-01-15T10:30:00: the value q itself gives for 2024.01.15D10:30:00.000000000
        a = np.array(["2024-01-15T10:30:00"], dtype="datetime64[ns]")
        self.assertEqual(bytes(encode(a)), b"\x0c\x00" + struct.pack("<i", 1) + struct.pack("<q", 758629800000000000))

    def test_every_unit_is_normalised_to_nanoseconds(self):
        want = bytes(encode(np.array(["2000-01-02T00:00:01"], dtype="datetime64[ns]")))
        for unit in ("s", "ms", "us", "ns"):
            with self.subTest(unit=unit):
                self.assertEqual(bytes(encode(np.array(["2000-01-02T00:00:01"], dtype="datetime64[%s]" % unit))), want)
        self.assertEqual(bytes(encode(np.array(["2000-01-02"], dtype="datetime64[D]"))),
                         b"\x0c\x00" + struct.pack("<i", 1) + struct.pack("<q", 86400 * 10 ** 9))
        self.assertEqual(bytes(encode(np.array(["2000-02"], dtype="datetime64[M]"))),
                         b"\x0c\x00" + struct.pack("<i", 1) + struct.pack("<q", 31 * 86400 * 10 ** 9))
        self.assertEqual(bytes(encode(np.array(["2001"], dtype="datetime64[Y]"))),
                         b"\x0c\x00" + struct.pack("<i", 1) + struct.pack("<q", 366 * 86400 * 10 ** 9))
        self.assertEqual(bytes(encode(np.array(["2000-01-01T00:00:00.000001000"], dtype="datetime64[ns]"))),
                         b"\x0c\x00" + struct.pack("<i", 1) + struct.pack("<q", 1000))

    def test_nat_of_every_unit_is_the_null(self):
        for unit in ("s", "D", "ns"):
            a = np.array(["NaT", "2000-01-01"], dtype="datetime64[%s]" % unit)
            self.assertEqual(bytes(encode(a)), b"\x0c\x00" + struct.pack("<i", 2) + self.NULL + bytes(8))

    def test_timespan_vector_bytes(self):
        a = np.array([1, 2], dtype="timedelta64[s]")
        self.assertEqual(bytes(encode(a)), b"\x10\x00" + struct.pack("<i", 2) + struct.pack("<qq", 10 ** 9, 2 * 10 ** 9))
        a = np.array([5, -2 ** 63], dtype="int64").view("timedelta64[ms]")             # the second is NaT
        self.assertEqual(bytes(encode(a)), b"\x10\x00" + struct.pack("<i", 2) + struct.pack("<q", 5 * 10 ** 6) + self.NULL)
        a = np.array([1], dtype="timedelta64[D]")
        self.assertEqual(bytes(encode(a)), b"\x10\x00" + struct.pack("<i", 1) + struct.pack("<q", 86400 * 10 ** 9))

    def test_atoms(self):
        self.assertEqual(bytes(encode(np.datetime64("2000-01-01T00:00:01"))), b"\xf4" + struct.pack("<q", 10 ** 9))
        self.assertEqual(bytes(encode(np.datetime64("NaT"))), b"\xf4" + self.NULL)
        self.assertEqual(bytes(encode(np.timedelta64(5, "ms"))), b"\xf0" + struct.pack("<q", 5 * 10 ** 6))
        self.assertEqual(bytes(encode(np.timedelta64("NaT"))), b"\xf0" + self.NULL)
        self.assertEqual(bytes(encode(np.array(np.datetime64("2000-01-01T00:00:01")))), b"\xf4" + struct.pack("<q", 10 ** 9))

    def test_empty_vectors_keep_their_type(self):
        self.assertEqual(bytes(encode(np.array([], dtype="datetime64[ns]"))), b"\x0c\x00" + struct.pack("<i", 0))
        self.assertEqual(bytes(encode(np.array([], dtype="timedelta64[ns]"))), b"\x10\x00" + struct.pack("<i", 0))

    def test_matrix_is_a_list_of_vectors(self):
        a = np.array([["2000-01-01", "2000-01-02"]], dtype="datetime64[D]")
        row = b"\x0c\x00" + struct.pack("<i", 2) + struct.pack("<qq", 0, 86400 * 10 ** 9)
        self.assertEqual(bytes(encode(a)), b"\x00\x00" + struct.pack("<i", 1) + row)

    def test_a_list_of_datetime_values_is_a_general_list_of_atoms(self):
        v = [np.datetime64("2000-01-01"), np.datetime64("2000-01-02")]
        self.assertEqual(bytes(encode(v)), b"\x00\x00" + struct.pack("<i", 2) + b"\xf4" + struct.pack("<q", 0)
                         + b"\xf4" + struct.pack("<q", 86400 * 10 ** 9))

    def test_column_of_a_table(self):
        t = codec.KdbTable(a=np.array([1, 2]), t=np.array(["2000-01-01", "NaT"], dtype="datetime64[ns]"))
        want = (b"\x62\x00\x63" + b"\x0b\x00" + struct.pack("<i", 2) + b"a\x00t\x00"
                + b"\x00\x00" + struct.pack("<i", 2)
                + b"\x07\x00" + struct.pack("<i", 2) + struct.pack("<qq", 1, 2)
                + b"\x0c\x00" + struct.pack("<i", 2) + bytes(8) + self.NULL)
        self.assertEqual(bytes(encode(t)), want)

    def test_out_of_range_and_calendar_units_are_refused(self):
        with self.assertRaises(IPCRangeError):
            encode(np.array(["9000-01-01"], dtype="datetime64[D]"))       # beyond the int64 nanosecond range
        with self.assertRaises(IPCRangeError):
            encode(np.array(["1600-01-01"], dtype="datetime64[D]"))
        with self.assertRaises(IPCRangeError):
            encode(np.array(["1677-09-22"], dtype="datetime64[D]"))       # inside numpy's ns range, before kdb+'s
        with self.assertRaises(IPCTypeError):
            encode(np.array([1], dtype="timedelta64[M]"))                  # a month has no fixed length

    def test_round_trip_is_the_underlying_integer(self):
        """temporals decode to the underlying integer, so the datetime64 does not come back
        as such. The asymmetry is deliberate; the value is the kdb+ nanosecond count."""
        a = np.array(["2000-01-01", "2024-01-15T10:30:00", "NaT"], dtype="datetime64[ns]")
        back = decode(encode(a))
        self.assertEqual(back.dtype, np.int64)
        self.assertEqual(back.tolist(), [0, 758629800000000000, -2 ** 63])
        np.testing.assert_array_equal(back[:2].view("timedelta64[ns]") + np.datetime64("2000-01-01", "ns"), a[:2])
        t = np.array([1, 2], dtype="timedelta64[s]")
        np.testing.assert_array_equal(decode(encode(t)).view("timedelta64[ns]"), t)
        self.assertEqual(decode(encode(np.datetime64("2000-01-02"))), 86400 * 10 ** 9)


# ----------------------------------------------------------------------------
# decoder behaviour
# ----------------------------------------------------------------------------

class TestDecode(unittest.TestCase):
    def test_atoms_and_vectors_types(self):
        v = decode(b"\xf9" + struct.pack("<q", 7))
        self.assertIs(type(v), int)
        v = decode(b"\xf7" + struct.pack("<d", 7.5))
        self.assertIs(type(v), float)
        v = decode(encode(np.array([1, 2, 3])))
        self.assertEqual(v.dtype, np.int64)
        self.assertTrue(v.flags.writeable)
        v = decode(encode(np.array([1, 2, 3])), widths=32)
        self.assertEqual(v.dtype, np.int32)
        v = decode(encode(np.array([1.5])), widths=32)
        self.assertEqual(v.dtype, np.float32)
        self.assertIsInstance(decode(b"\xf5abc\x00"), KGSym)
        self.assertIsInstance(decode(b"\xf6a"), KGChar)
        self.assertIsInstance(decode(b"\x0b\x00\x01\x00\x00\x00a\x00")[0], KGSym)

    def test_nulls_are_not_translated(self):
        self.assertEqual(decode(b"\xf9" + struct.pack("<q", -2 ** 63)), -2 ** 63)
        self.assertTrue(math.isnan(decode(b"\xf7" + struct.pack("<d", float("nan")))))
        self.assertEqual(decode(b"\xf5\x00"), KGSym(""))            # null symbol

    def test_int32_range_error(self):
        big = b"\xf9" + struct.pack("<q", 2 ** 40)
        self.assertEqual(decode(big), 2 ** 40)
        with self.assertRaises(IPCRangeError):
            decode(big, widths=32)
        with self.assertRaises(IPCRangeError):
            decode(encode(np.array([2 ** 40])), widths=32)
        with self.assertRaises(IPCRangeError):
            decode(b"\xf4" + struct.pack("<q", 2 ** 40), widths=32)   # a timestamp is a long too
        self.assertEqual(decode(b"\xfa" + struct.pack("<i", 5), widths=32), 5)

    def test_general_list_normalisation(self):
        body = b"\x00\x00\x02\x00\x00\x00\xf9" + struct.pack("<q", 1) + b"\xf9" + struct.pack("<q", 2)
        v = decode(body)
        self.assertEqual(v.dtype, np.int64)
        v = decode(encode_call([1.5, 2.5]))
        self.assertEqual(v.dtype, np.float64)
        v = decode(encode_call([1, 2.5]))
        self.assertEqual(v.dtype, object)
        self.assertIs(type(v[0]), int)
        v = decode(encode(np.array([[1, 2], [3, 4]])))
        self.assertEqual(v.shape, (2, 2))
        v = decode(encode(np.array([[1, 2], [3.0, 4]])))
        self.assertEqual(v.dtype, np.float64)
        # rows of different dtype are not stacked
        v = decode(encode_call([np.array([1, 2]), np.array([1.5, 2.5])]))
        self.assertEqual(v.dtype, object)
        v = decode(encode([]))
        self.assertEqual(v.shape, (0,))

    def test_dict_keys(self):
        # list-valued key becomes a tuple, unhashable dict key is a protocol error
        keys = b"\x00\x00\x01\x00\x00\x00" + encode(np.array([1, 2]))
        d = decode(b"\x63" + keys + encode([5]))
        self.assertEqual(d, {(1, 2): 5})
        inner = b"\x63" + encode([KGSym("a")]) + encode([1])
        with self.assertRaises(ProtocolError):
            decode(b"\x63\x00\x00\x01\x00\x00\x00" + inner + encode([1]))

    def test_error_object(self):
        e = decode(b"\x80type\x00")
        self.assertEqual(e, KdbError("type"))

    def test_generic_null_and_opaque(self):
        self.assertIs(decode(b"\x65\x00"), KLONG_UNDEFINED)
        o = decode(b"\x65\x01")
        self.assertEqual(o, KdbOpaque(101, b"\x65\x01"))
        # nested opaque inside a list re-encodes byte for byte
        lst = b"\x00\x00\x02\x00\x00\x00\x66\x07\xf9" + struct.pack("<q", 1)
        v = decode(lst)
        self.assertEqual(encode(v), lst)
        # a projection holding a lambda and a table, big-endian, comes back little-endian
        inner_le = b"\x68" + struct.pack("<i", 2) + b"\x64\x00" + encode("x+y") + encode(np.array([1, 2], dtype=np.int32).astype(np.int64))
        inner_be = b"\x68" + struct.pack(">i", 2) + b"\x64\x00\x0a\x00" + struct.pack(">i", 3) + b"x+y" + \
            b"\x07\x00" + struct.pack(">i", 2) + struct.pack(">qq", 1, 2)
        o = decode(inner_be, little_endian=False)
        self.assertEqual(o.type, 104)
        self.assertEqual(o.raw, inner_le)
        self.assertEqual(encode(o), inner_le)
        self.assertEqual(decode(inner_le), o)

    def test_big_endian_everything(self):
        be = (b"\x00\x00" + struct.pack(">i", 3)               # general list of 3
              + b"\xf9" + struct.pack(">q", 7)
              + b"\x09\x00" + struct.pack(">i", 2) + struct.pack(">dd", 1.5, 2.5)
              + b"\x0b\x00" + struct.pack(">i", 1) + b"ab\x00")
        v = decode(be, little_endian=False)
        self.assertEqual(v[0], 7)
        self.assertTrue(kg_equal(v[1], np.array([1.5, 2.5])))
        self.assertEqual(v[2][0], KGSym("ab"))

    def test_guid_bytes_are_not_swapped(self):
        g = bytes(range(16))
        for little in (True, False):
            self.assertEqual(decode(b"\xfe" + g, little).tolist(), list(range(16)))
        # opaque copy keeps guid bytes untouched when swapping the rest
        proj = b"\x68" + struct.pack(">i", 1) + b"\xfe" + g
        self.assertEqual(decode(proj, False).raw[5:], b"\xfe" + g)

    def test_depth_limit_exact(self):
        v = encode(1)
        for _ in range(MAX_DEPTH):
            v = b"\x00\x00\x01\x00\x00\x00" + v
        decode(v)                                     # 64 nested lists: accepted
        with self.assertRaises(ProtocolError):
            decode(b"\x00\x00\x01\x00\x00\x00" + v)   # 65: rejected
        d = encode(1)
        for _ in range(MAX_DEPTH):
            d = b"\x63" + encode([KGSym("a")]) + b"\x00\x00\x01\x00\x00\x00" + d
        with self.assertRaises(ProtocolError):
            decode(d)

    def test_opaque_depth_limit(self):
        o = b"\x66\x01"
        for _ in range(MAX_DEPTH + 2):
            o = b"\x6a" + o
        with self.assertRaises(ProtocolError):
            decode(o)

    def test_table_shapes(self):
        t = (b"\x62\x00\x63\x0b\x00\x01\x00\x00\x00a\x00" + b"\x00\x00\x01\x00\x00\x00" + encode(np.array([1, 2])))
        self.assertTrue(kg_equal(decode(t), {KGSym("a"): np.array([1, 2])}))
        # a char vector column stays a string
        t = (b"\x62\x00\x63\x0b\x00\x01\x00\x00\x00a\x00" + b"\x00\x00\x01\x00\x00\x00" + encode("xy"))
        self.assertEqual(decode(t), {KGSym("a"): "xy"})

    def test_input_validation(self):
        with self.assertRaises(TypeError):
            decode("text")
        with self.assertRaises(ValueError):
            decode(b"\xf9" + bytes(8), widths=16)


# ----------------------------------------------------------------------------
# compression
# ----------------------------------------------------------------------------

def _msg(body, little=True, typ=0):
    n = 8 + len(body)
    return bytes([1 if little else 0, typ, 0, 0]) + struct.pack("<I" if little else ">I", n) + body


class TestCompress(unittest.TestCase):
    EXAMPLE = bytes.fromhex("0100010025000000be040000000a00b004000061625503ff6203ff6203ff6203ff620103a4")

    def test_reference_example_decompresses(self):
        out = decompress(self.EXAMPLE)
        self.assertEqual(out, _msg(b"\x0a\x00" + struct.pack("<i", 1200) + b"ab" * 600))
        self.assertEqual(decode(out[8:]), "ab" * 600)

    def test_reference_example_compresses_identically(self):
        msg = _msg(b"\x0a\x00" + struct.pack("<i", 1200) + b"ab" * 600)
        self.assertEqual(compress(msg, threshold=0), self.EXAMPLE)

    def test_threshold_and_ratio_rules(self):
        small = _msg(b"\x0a\x00" + struct.pack("<i", 100) + b"a" * 100)
        self.assertEqual(compress(small), small)                       # not over 2000 bytes
        rng = random.Random(5)
        noise = _msg(b"\x0a\x00" + struct.pack("<i", 3000) + bytes(rng.randrange(256) for _ in range(3000)))
        self.assertEqual(compress(noise), noise)                       # would not halve
        big = _msg(b"\x0a\x00" + struct.pack("<i", 3000) + b"a" * 3000)
        c = compress(big)
        self.assertEqual(c[2], 1)
        self.assertLess(len(c), len(big) // 2)
        self.assertEqual(decompress(c), big)

    def test_roundtrip_random_structured(self):
        rng = random.Random(11)
        for k in range(300):
            n = rng.randint(30, 5000)
            parts, size = [], 0
            while size < n:
                if rng.random() < 0.4:
                    p = bytes(rng.randrange(256) for _ in range(rng.randint(1, 20)))
                else:
                    unit = bytes(rng.randrange(256) for _ in range(rng.randint(1, 6)))
                    p = unit * rng.randint(1, 300)
                parts.append(p)
                size += len(p)
            msg = _msg(b"".join(parts)[:n], little=rng.random() < 0.5, typ=rng.randrange(3))
            c = compress(msg, threshold=0)
            if c != msg:
                self.assertEqual(c[2], 1)
                self.assertEqual(decompress(c), msg, k)

    def test_roundtrip_big_endian_header(self):
        msg = _msg(b"\x0a\x00\x00\x00\x0b\xb8" + b"abc" * 1000, little=False)
        c = compress(msg)
        self.assertEqual(c[0], 0)
        self.assertEqual(struct.unpack(">I", c[4:8])[0], len(c))
        self.assertEqual(decompress(c), msg)

    def test_roundtrip_encoded_values(self):
        value = np.arange(5000, dtype=np.int64) % 7
        msg = _msg(encode(value))
        c = compress(msg)
        self.assertEqual(c[2], 1)
        self.assertTrue(kg_equal(decode(decompress(c)[8:]), value))

    def test_limits(self):
        with self.assertRaises(ProtocolError):
            decompress(self.EXAMPLE, max_size=1000)                    # declared 1214 > limit
        # declared size just within the limit passes the check
        self.assertEqual(len(decompress(self.EXAMPLE, max_size=1214)), 1214)

    def test_malformed_streams(self):
        ex = self.EXAMPLE
        bad = [
            ex[:11],                                                    # header cut
            ex[:-1][:4] + struct.pack("<I", len(ex) - 1) + ex[8:-1],    # stream ends early
            ex[:4] + struct.pack("<I", len(ex) + 1) + ex[8:],           # length field wrong
            b"\x02" + ex[1:],                                           # bad byte order flag
            ex[:8] + struct.pack("<I", 7) + ex[12:],                    # size below header
            ex[:8] + struct.pack("<I", 0x80000000) + ex[12:],           # size above int32
            ex[:8] + struct.pack("<I", 10 ** 8) + ex[12:],              # impossible expansion
            ex[:12] + b"\x01\x00\x00" + ex[15:],                        # back-reference to nothing
        ]
        for i, b in enumerate(bad):
            with self.subTest(i=i):
                with self.assertRaises(ProtocolError):
                    decompress(b)

    def test_garbage_streams_only_protocol_error(self):
        rng = random.Random(21)
        for k in range(400):
            n = rng.randint(0, 80)
            body = bytes(rng.randrange(256) for _ in range(n))
            usize = rng.choice([8, 9, 20, 100, 1214, 5000, 1 << 20, 1 << 28, 1 << 31, (1 << 32) - 1])
            msg = bytes([rng.choice([0, 1]), rng.randrange(3), 1, 0])
            order = "<I" if msg[0] == 1 else ">I"
            msg += struct.pack(order, 12 + n) + struct.pack(order, usize) + body
            try:
                out = decompress(msg)
            except ProtocolError:
                continue
            self.assertEqual(len(out), usize)

    def test_mutated_valid_streams(self):
        rng = random.Random(22)
        for k in range(600):
            b = bytearray(self.EXAMPLE)
            for _ in range(rng.randint(1, 3)):
                b[rng.randrange(len(b))] = rng.randrange(256)
            b[4:8] = struct.pack("<I", len(b))
            try:
                decompress(bytes(b))
            except ProtocolError:
                pass


# ----------------------------------------------------------------------------
# properties (seeded random generation; Hypothesis is not installed)
# ----------------------------------------------------------------------------

_SYM_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.é中"
_STR_ALPHABET = _SYM_ALPHABET + " \n\t\"+-*/ü\U0001f600"


def rand_value(rng, depth=0):
    """A random Klong value that survives encode/decode unchanged under kg_equal."""
    kinds = ["int", "float", "str", "char", "sym", "iarr", "farr", "iarr2", "list", "dict", "undef", "empty", "symlist", "strlist"]
    if depth >= 3:
        kinds = kinds[:8] + ["undef", "empty", "symlist", "strlist"]
    k = rng.choice(kinds)
    if k == "int":
        return rng.choice([0, 1, -1, 2 ** 63 - 1, -2 ** 63, rng.randint(-10 ** 6, 10 ** 6), rng.randint(-2 ** 62, 2 ** 62)])
    if k == "float":
        return rng.choice([0.0, -0.0, 1.5, float("inf"), float("-inf"), float("nan"), rng.uniform(-1e9, 1e9), rng.random() * 1e-300])
    if k == "str":
        return "".join(rng.choice(_STR_ALPHABET) for _ in range(rng.randint(0, 12)))
    if k == "char":
        return KGChar(chr(rng.randint(32, 126)))
    if k == "sym":
        return KGSym("".join(rng.choice(_SYM_ALPHABET) for _ in range(rng.randint(0, 6))))
    if k == "iarr":
        return np.array([rng.randint(-2 ** 62, 2 ** 62) for _ in range(rng.randint(1, 8))], dtype=np.int64)
    if k == "farr":
        return np.array([rng.choice([rng.uniform(-1e6, 1e6), float("nan"), float("inf")]) for _ in range(rng.randint(1, 8))])
    if k == "iarr2":
        r, c = rng.randint(1, 4), rng.randint(1, 4)
        return np.array([[rng.randint(-100, 100) for _ in range(c)] for _ in range(r)], dtype=np.int64)
    if k == "undef":
        return KLONG_UNDEFINED
    if k == "empty":
        return rng.choice([[], np.array([]), "", {}])
    if k == "symlist":
        return [KGSym("".join(rng.choice(_SYM_ALPHABET) for _ in range(rng.randint(0, 4)))) for _ in range(rng.randint(1, 5))]
    if k == "strlist":
        return [rand_value(rng, 9) if False else "".join(rng.choice(_STR_ALPHABET) for _ in range(rng.randint(0, 5)))
                for _ in range(rng.randint(1, 5))]
    if k == "list":
        items = [rand_value(rng, depth + 1) for _ in range(rng.randint(1, 5))]
        return np.array(items, dtype=object) if rng.random() < 0.5 else items
    # dict with homogeneous keys
    n = rng.randint(0, 4)
    keyspace = rng.choice(["sym", "int", "str"])
    if keyspace == "sym":
        keys = {KGSym("k%d" % rng.randint(0, 99)) for _ in range(n)}
    elif keyspace == "int":
        keys = {rng.randint(-1000, 1000) for _ in range(n)}
    else:
        keys = {"s%d" % rng.randint(0, 99) for _ in range(n)}
    return {k: rand_value(rng, depth + 1) for k in sorted(keys, key=str)}


class TestProperties(unittest.TestCase):
    def test_roundtrip_random_values(self):
        rng = random.Random(1234)
        for i in range(1500):
            v = rand_value(rng)
            with self.subTest(i=i):
                b = encode(v)
                self.assertTrue(kg_equal(decode(b), v), (v, b.hex()))
                # re-encoding the decoded value is stable
                self.assertEqual(encode(decode(b)), b)

    def test_roundtrip_random_values_32(self):
        rng = random.Random(4321)
        for i in range(500):
            v = rand_value(rng)
            try:
                b = encode(v, 32)
            except IPCRangeError:
                continue
            got = decode(b, widths=32)
            # floats are narrowed in 32-bit mode: compare through float32
            self.assertTrue(_equal_32(got, v), (v, b.hex()))

    def test_roundtrip_numpy_dtypes(self):
        rng = np.random.default_rng(7)
        for dt in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.float32, np.float64, np.bool_):
            for n in (1, 5, 100):
                if np.dtype(dt).kind == "f":
                    a = rng.standard_normal(n).astype(dt)
                elif np.dtype(dt).kind == "b":
                    a = rng.integers(0, 2, n).astype(dt)
                else:
                    info = np.iinfo(dt)
                    a = rng.integers(info.min, info.max, n, dtype=dt, endpoint=True)
                with self.subTest(dtype=dt, n=n):
                    self.assertTrue(kg_equal(decode(encode(a)), a))

    def test_compress_roundtrip_of_encoded_values(self):
        rng = random.Random(99)
        for i in range(60):
            v = np.array([rng.randint(0, 5) for _ in range(rng.randint(300, 4000))], dtype=np.int64)
            msg = _msg(encode(v))
            c = compress(msg, threshold=0)
            self.assertEqual(decompress(c), msg)

    # -- hostile input -----------------------------------------------------

    def _valid_bodies(self):
        rng = random.Random(77)
        out = [encode(rand_value(rng)) for _ in range(150)]
        out += [bytes.fromhex(v["hex"])[8:] for v in load_vectors()
                if v["mode"] != "reject" and bytes.fromhex(v["hex"])[2] == 0]
        return out

    def test_garbage_only_protocol_error(self):
        rng = random.Random(555)
        for i in range(4000):
            n = rng.randint(0, 64)
            data = bytes(rng.randrange(256) for _ in range(n))
            for little in (True, False):
                try:
                    decode(data, little)
                except ProtocolError:
                    pass

    def test_garbage_with_plausible_type_bytes(self):
        rng = random.Random(556)
        types = list(range(0, 20)) + [98, 99, 100, 101, 102, 104, 106, 127, 0xF9, 0xF5, 0xFE, 0x80, 0xF6]
        for i in range(4000):
            n = rng.randint(1, 48)
            data = bytes([rng.choice(types)]) + bytes(rng.choice([0, 1, 2, 3, 0x7f, 0xff, rng.randrange(256)]) for _ in range(n))
            for w in (64, 32):
                try:
                    decode(data, rng.random() < 0.7, w)
                except (ProtocolError, IPCRangeError):
                    pass

    def test_mutations_only_protocol_error(self):
        rng = random.Random(888)
        bodies = self._valid_bodies()
        for i in range(8000):
            b = bytearray(rng.choice(bodies))
            for _ in range(rng.randint(1, 4)):
                op = rng.random()
                if op < 0.5 and b:
                    b[rng.randrange(len(b))] = rng.randrange(256)
                elif op < 0.65 and b:
                    del b[rng.randrange(len(b)):]
                elif op < 0.8:
                    b.insert(rng.randrange(len(b) + 1), rng.randrange(256))
                elif op < 0.9 and len(b) > 5:
                    pos = rng.randrange(len(b) - 4)
                    b[pos:pos + 4] = struct.pack("<i", rng.choice([0, -1, 1, 255, 0x7fffffff, 1 << 24, rng.randrange(-5, 5000)]))
                else:
                    b += bytes(rng.randrange(256) for _ in range(rng.randint(1, 8)))
            try:
                decode(bytes(b), rng.random() < 0.8, rng.choice([64, 32]))
            except (ProtocolError, IPCRangeError):
                pass

    def test_hostile_counts_do_not_allocate(self):
        """A short message claiming huge counts is rejected before any big allocation."""
        hostile = []
        for t in list(range(1, 20)):
            hostile.append(bytes([t, 0]) + struct.pack("<i", 0x7fffffff) + b"\x00" * 16)
            hostile.append(bytes([t, 0]) + struct.pack("<i", 50_000_000) + b"\x00" * 16)
        hostile.append(b"\x00\x00" + struct.pack("<i", 0x7fffffff) + b"\x00" * 16)
        hostile.append(b"\x00\x00" + struct.pack("<i", 100_000_000) + b"\x00" * 64)
        hostile.append(b"\x68" + struct.pack("<i", 0x7fffffff))
        hostile.append(b"\x69" + struct.pack("<i", 100_000_000) + b"\x00" * 64)
        hostile.append(b"\x62\x00\x63\x0b\x00" + struct.pack("<i", 0x7fffffff) + b"a\x00")
        hostile.append(b"\x63\x0b\x00" + struct.pack("<i", 0x7fffffff) + b"a\x00")
        hostile.append(b"\x64\x00\x0a\x00" + struct.pack("<i", 0x7fffffff))
        tracemalloc.start()
        try:
            for h in hostile:
                tracemalloc.clear_traces()
                with self.assertRaises(ProtocolError, msg=h[:8].hex()):
                    decode(h)
                _, peak = tracemalloc.get_traced_memory()
                self.assertLess(peak, 1 << 20, h[:8].hex())
                for little in (False,):
                    h2 = h
                    tracemalloc.clear_traces()
                    try:
                        decode(h2, little)
                    except ProtocolError:
                        pass
                    _, peak = tracemalloc.get_traced_memory()
                    self.assertLess(peak, 1 << 20)
        finally:
            tracemalloc.stop()

    def test_decode_memory_proportional_to_input(self):
        """Mutated inputs never use more than a small multiple of their own size."""
        rng = random.Random(31337)
        bodies = [encode(np.arange(200, dtype=np.int64)), encode([1, "a", KGSym("x")]),
                  encode({KGSym("a"): np.arange(50)}), encode(["ab", "cd", "ef"])]
        tracemalloc.start()
        try:
            for i in range(1500):
                b = bytearray(rng.choice(bodies))
                pos = rng.randrange(len(b))
                b[pos] = rng.randrange(256)
                if rng.random() < 0.5 and len(b) > 6:
                    pos = rng.randrange(len(b) - 3)
                    b[pos:pos + 4] = struct.pack("<i", rng.choice([-1, 0x7fffffff, 1 << 28, 1 << 30]))
                tracemalloc.clear_traces()
                try:
                    decode(bytes(b))
                except ProtocolError:
                    pass
                _, peak = tracemalloc.get_traced_memory()
                self.assertLess(peak, 64 * 1024 + 64 * len(b), (peak, bytes(b).hex()))
        finally:
            tracemalloc.stop()


def _equal_32(got, v):
    """Equality after the narrowing a 32-bit build applies to floats."""
    seq = (list, tuple, np.ndarray)
    if isinstance(v, np.ndarray) and v.dtype.kind == "f":
        v = v.astype(np.float32).astype(np.float64)
    elif isinstance(v, float):
        with np.errstate(over="ignore"):
            v = float(np.float32(v))
    elif isinstance(v, seq) and not (isinstance(v, np.ndarray) and v.dtype != object):
        if not isinstance(got, seq) or len(got) != len(v):
            return False
        return all(_equal_32(g, x) for g, x in zip(got, v))
    elif isinstance(v, dict):
        return isinstance(got, dict) and len(got) == len(v) and all(
            k in got and _equal_32(got[k], x) for k, x in v.items())
    return kg_equal(got, v)


if __name__ == "__main__":
    unittest.main()


# These tests check the codec's own table form; klongpy.db, when another test
# has loaded it, makes decoded tables arrive as its Table instead.
_saved_table_factory = None


def setUpModule():
    global _saved_table_factory
    from klongpy.ipc import codec as _codec
    _saved_table_factory = _codec._table_factory
    _codec.set_table_factory(None)


def tearDownModule():
    from klongpy.ipc import codec as _codec
    _codec.set_table_factory(_saved_table_factory)
