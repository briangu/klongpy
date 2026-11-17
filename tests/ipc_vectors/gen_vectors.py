"""Generator for tests/ipc_vectors/vectors.json (not a test; run by hand).

    python tests/ipc_vectors/gen_vectors.py [output.json]

Builds every message byte by byte with struct, independent of klongpy.ipc.codec
(so the vectors are not self-fulfilling), and asserts the four KX-published
strings and the other worked examples of docs/ipc_capabilities.md along the way.
"""
import os
import json, struct, sys

def hdr(body, typ=0, le=True, comp=0):
    total = 8 + len(body)
    return bytes([1 if le else 0, typ, comp, 0]) + struct.pack('<I' if le else '>I', total) + body

def E(le): return '<' if le else '>'
def atom(t, fmt, v, le=True): return bytes([256 - t]) + struct.pack(E(le) + fmt, v)
def vec(t, fmt, vals, le=True, attr=0):
    return bytes([t, attr]) + struct.pack(E(le) + 'i', len(vals)) + b''.join(struct.pack(E(le) + fmt, v) for v in vals)
def sym(s): return bytes([256 - 11]) + s.encode() + b'\0'
def symvec(ss, le=True, attr=0): return bytes([11, attr]) + struct.pack(E(le)+'i', len(ss)) + b''.join(s.encode() + b'\0' for s in ss)
def charvec(b, le=True):
    if isinstance(b, str): b = b.encode()
    return bytes([10, 0]) + struct.pack(E(le)+'i', len(b)) + b
def gl(items, le=True): return bytes([0, 0]) + struct.pack(E(le)+'i', len(items)) + b''.join(items)
def dct(k, v, t=99): return bytes([t]) + k + v
def table(names, cols, le=True, attr=0):
    return bytes([98, attr]) + dct(symvec(names, le), gl(cols, le))
def lam(ctx, text, le=True): return b'\x64' + ctx.encode() + b'\0' + charvec(text, le)
NULL = b'\x65\x00'
def longa(v, le=True): return atom(7, 'q', v, le)
def lv(vals, le=True, attr=0): return vec(7, 'q', vals, le, attr)
def fv(vals, le=True): return vec(9, 'd', vals, le)

V = []
def add(name, klong, body, mode, widths='64', typ=0, source='derived', expect=None, le=True, comp=0, error=None, whole=None):
    e = {"name": name, "klong": klong}
    e["hex"] = (whole if whole is not None else hdr(body, typ, le, comp)).hex()
    e["mode"] = mode; e["widths"] = widths; e["source"] = source
    if typ: e["type"] = typ
    if expect is not None: e["expect"] = expect
    if error: e["error"] = error
    V.append(e)

# --- KX published (ref 8, SER page) ---------------------------------------
kx_int = bytes.fromhex("010000000d000000fa01000000")
assert hdr(atom(6, 'i', 1)) == kx_int
add("kx-int-atom", "1", None, "roundtrip", "32", source="kx", whole=kx_int)
kx_ivec = bytes.fromhex("01000000120000000600010000000100 0000".replace(" ", ""))
assert hdr(vec(6, 'i', [1])) == kx_ivec
add("kx-int-vector", "[1]", None, "roundtrip", "32", source="kx", whole=kx_ivec)
kx_dict = bytes.fromhex("0100000021000000630b0002000000610062000600020000000200000003000000")
assert hdr(dct(symvec(["a", "b"]), vec(6, 'i', [2, 3]))) == kx_dict
add("kx-dict-sym-int", ":{[:a 2][:b 3]}", None, "roundtrip", "32", source="kx", whole=kx_dict)
kx_lam = bytes.fromhex("010000001500000064000a00050000007b782b797d")
assert hdr(lam("", "{x+y}")) == kx_lam
add("kx-lambda-root", "{x+y} (q lambda, root context)", None, "roundtrip", "any", source="kx", whole=kx_lam,
    expect={"lambda": ["", "{x+y}"]})

# --- roundtrip ---------------------------------------
r81 = bytes.fromhex("0101000011000000" "0a0003000000312b31")
assert hdr(charvec("1+1"), 1) == r81
add("string-1+1-sync", '"1+1"', charvec("1+1"), "roundtrip", "any", typ=1)
r82 = bytes.fromhex("0102000011000000f90200000000000000")
assert hdr(longa(2), 2) == r82
add("response-long-2", "2", longa(2), "roundtrip", "64", typ=2)
r83 = bytes.fromhex("0100000026000000070003000000010000000000000002000000000000000300000000000000")
assert hdr(lv([1, 2, 3])) == r83
add("long-vector", "[1 2 3]", lv([1, 2, 3]), "roundtrip", "64")
r84 = bytes.fromhex("010000001e000000090002000000000000000000f83f0000000000000440")
assert hdr(fv([1.5, 2.5])) == r84
add("float-vector", "[1.5 2.5]", fv([1.5, 2.5]), "roundtrip", "64")
r85 = bytes.fromhex("010000000d000000f561626300")
assert hdr(sym("abc")) == r85
add("symbol-atom", ":abc", sym("abc"), "roundtrip", "any")
r86 = bytes.fromhex("0100000026000000000004000000f90100000000000000f5616200f678f70000000000000440")
body86 = gl([longa(1), sym("ab"), b'\xf6x', atom(9, 'd', 2.5)])
assert hdr(body86) == r86
add("mixed-list", "[1 :ab 0cx 2.5]", body86, "roundtrip", "64")
r87 = bytes.fromhex("0100000029000000630b00020000006100620007000200000001000000000000000200000000000000")
assert hdr(dct(symvec(["a", "b"]), lv([1, 2]))) == r87
add("dict-sym-long", ":{[:a 1][:b 2]}", dct(symvec(["a", "b"]), lv([1, 2])), "roundtrip", "64")
r89 = bytes.fromhex("010200000e000000807479706500")
assert hdr(b'\x80type\0', 2) == r89
add("error-type", "error object 'type'", b'\x80type\0', "roundtrip", "any", typ=2, expect={"error": "type"})
r810 = bytes.fromhex("010000000a0000006500")
assert hdr(NULL) == r810
add("generic-null", ":undefined (generic null)", NULL, "roundtrip", "any", expect={"undefined": True})
r811 = bytes.fromhex("0101000025000000000003000000f561646400f90100000000000000f90200000000000000")
b811 = gl([sym("add"), longa(1), longa(2)])
assert hdr(b811, 1) == r811
add("call-symbol-fn", "[:add 1 2]", b811, "roundtrip", "64", typ=1)
r811b = bytes.fromhex("01010000290000000000030000000a0003000000616464f90100000000000000f90200000000000000")
b811b = gl([charvec("add"), longa(1), longa(2)])
assert hdr(b811b, 1) == r811b
add("call-string-fn", '["add" 1 2]', b811b, "roundtrip", "64", typ=1)
add("long-null", "-9223372036854775808", longa(-2**63), "roundtrip", "64", expect={"int": -2**63})
assert hdr(longa(-2**63)).hex() == "0100000011000000f90000000000000080"
add("empty-list", "[]", gl([]), "roundtrip", "any")
assert hdr(gl([])).hex() == "010000000e000000000000000000"
add("empty-string", '""', charvec(""), "roundtrip", "any")
add("char-atom", "0cx", b'\xf6x', "roundtrip", "any")
add("string-utf8", '"héllo"', charvec("héllo"), "roundtrip", "any")
add("float-atom", "1.5", atom(9, 'd', 1.5), "roundtrip", "64")
add("float-nan", "NaN", atom(9, 'd', float('nan')), "roundtrip", "64", expect={"float": "nan"})
add("rank2-array", "[[1 2][3 4]]", gl([lv([1, 2]), lv([3, 4])]), "roundtrip", "64")
add("ragged-list", "[[1 2][3]]", gl([lv([1, 2]), lv([3])]), "roundtrip", "64")
add("symbol-vector", "[:a :b :c]", symvec(["a", "b", "c"]), "roundtrip", "any")
add("list-of-strings", '["ab" "c"]', gl([charvec("ab"), charvec("c")]), "roundtrip", "any")
add("dict-empty", ":{}", dct(gl([]), gl([])), "roundtrip", "any")
add("dict-mixed-values", ':{[:a 1][:b "x"]}', dct(symvec(["a", "b"]), gl([longa(1), charvec("x")])), "roundtrip", "64")
add("dict-vector-values", ":{[:a [1 2]][:b [3.5 4.5]]}",
    dct(symvec(["a", "b"]), gl([lv([1, 2]), fv([3.5, 4.5])])), "roundtrip", "64")
# 32-bit builds
add("int-vector-32", "[1 2 3]", vec(6, 'i', [1, 2, 3]), "roundtrip", "32")
add("real-vector-32", "[1.5 2.5]", vec(8, 'f', [1.5, 2.5]), "roundtrip", "32")
add("real-atom-32", "1.5", atom(8, 'f', 1.5), "roundtrip", "32")
# lambdas and opaque functions
add("lambda-kg", "kg lambda (display form)", lam("kg", ":dyad"), "roundtrip", "any", expect={"lambda": ["kg", ":dyad"]})
add("lambda-context-d", "q lambda in context d", lam("d", "{x+y}"), "roundtrip", "any", expect={"lambda": ["d", "{x+y}"]})
op = b'\x66\x01'
add("opaque-operator", ":kdb-function (102)", op, "roundtrip", "any", expect={"opaque": [102, op.hex()]})
proj = b'\x68' + struct.pack('<i', 2) + op + longa(1)
add("opaque-projection", ":kdb-function (104 with arguments)", proj, "roundtrip", "any", expect={"opaque": [104, proj.hex()]})
adv = b'\x6a' + op
add("opaque-adverb", ":kdb-function (106 over an operator)", adv, "roundtrip", "any", expect={"opaque": [106, adv.hex()]})
unary = b'\x65\x03'
add("opaque-unary-primitive", ":kdb-function (101, nonzero index)", unary, "roundtrip", "any", expect={"opaque": [101, unary.hex()]})

# --- decode only -----------------------------------------------------------
add("be-long-vector", "[1 2 3]", lv([1, 2, 3], le=False), "decode", "64", le=False)
add("be-float-vector", "[1.5 2.5]", fv([1.5, 2.5], le=False), "decode", "64", le=False)
add("be-dict", ":{[:a 1][:b 2]}", dct(symvec(["a", "b"], False), lv([1, 2], False)), "decode", "64", le=False)
add("be-int-atom", "1", atom(6, 'i', 1, False), "decode", "64", le=False)
add("be-mixed-list", "[1 :ab 0cx 2.5]", gl([longa(1, False), sym("ab"), b'\xf6x', atom(9, 'd', 2.5, False)], False), "decode", "64", le=False)
add("be-lambda", "q lambda {x+y}", lam("", "{x+y}", False), "decode", "any", le=False, expect={"lambda": ["", "{x+y}"]})
bproj = b'\x68' + struct.pack('>i', 2) + op + longa(1, False)
add("be-opaque-projection", ":kdb-function, held little-endian", bproj, "decode", "any", le=False, expect={"opaque": [104, proj.hex()]})
comp = bytes.fromhex("0100010025000000be040000000a00b004000061625503ff6203ff6203ff6203ff620103a4")
add("compressed-char-vector", '1200 chars "abab..."', None, "decode", "any", whole=comp, comp=1,
    expect={"str": "ab" * 600})
add("attribute-ignored", "[1 2 3]", lv([1, 2, 3], attr=1), "decode", "64")
tbl = table(["a", "b"], [lv([1, 2]), fv([3.5, 4.5])])
r88 = bytes.fromhex("01000000470000006200630b000200000061006200000002000000070002000000010000000000000002000000000000000900020000000000000000000c400000000000001240")
assert hdr(tbl) == r88, (hdr(tbl).hex())
add("table", ":{[:a [1 2]][:b [3.5 4.5]]}", tbl, "decode", "64")
add("table-sorted-attribute", ":{[:a [1 2]][:b [3.5 4.5]]}", table(["a", "b"], [lv([1, 2], attr=3), fv([3.5, 4.5])], attr=1), "decode", "64")
add("table-symbol-and-string-columns", "table with a symbol column and a string column", table(["s", "t"], [symvec(["x", "y"]), gl([charvec("ab"), charvec("c")])]), "decode", "64",
    expect={"dict": [[{"sym": "s"}, {"array": [{"sym": "x"}, {"sym": "y"}], "dtype": "object"}],
                     [{"sym": "t"}, {"array": [{"str": "ab"}, {"str": "c"}], "dtype": "object"}]]})
keyed = dct(table(["k"], [lv([1, 2])]), table(["v"], [lv([3, 4])]))
add("keyed-table", "[:{[:k [1 2]]} :{[:v [3 4]]}] (keys table, values table)", keyed, "decode", "64",
    expect={"array": [{"dict": [[{"sym": "k"}, {"array": [1, 2], "dtype": "int64"}]]},
                      {"dict": [[{"sym": "v"}, {"array": [3, 4], "dtype": "int64"}]]}], "dtype": "object"})
add("sorted-dict", ":{[:a 1][:b 2]}", dct(symvec(["a", "b"], attr=1), lv([1, 2]), t=127), "decode", "64")
add("table-empty", ":{}", table([], []), "decode", "64")
# temporals: underlying numbers
add("timestamp-atom", "1", atom(12, 'q', 1), "decode", "64")
add("timestamp-vector", "[1 2]", vec(12, 'q', [1, 2]), "decode", "64")
add("timespan-atom", "86400000000000", atom(16, 'q', 86400000000000), "decode", "64")
add("month-atom", "12", atom(13, 'i', 12), "decode", "64")
add("date-atom", "9000", atom(14, 'i', 9000), "decode", "64")
add("date-vector", "[0 1 -1]", vec(14, 'i', [0, 1, -1]), "decode", "64")
add("datetime-atom", "0.5", atom(15, 'd', 0.5), "decode", "64")
add("minute-atom", "90", atom(17, 'i', 90), "decode", "64")
add("second-atom", "3661", atom(18, 'i', 3661), "decode", "64")
add("time-atom", "45296789", atom(19, 'i', 45296789), "decode", "64")
add("time-vector", "[0 1000]", vec(19, 'i', [0, 1000]), "decode", "64")
# other integer / float widths, decoded in the 64-bit build
add("bool-vector", "[1 0]", bytes([1, 0]) + struct.pack('<i', 2) + bytes([1, 0]), "decode", "64")
assert hdr(bytes([1, 0]) + struct.pack('<i', 2) + bytes([1, 0])).hex() == "01000000100000000100020000000100"
add("bool-atom", "1", b'\xff\x01', "decode", "64")
add("byte-atom", "255", b'\xfc\xff', "decode", "64")
add("byte-vector", "[0 1 255]", vec(4, 'B', [0, 1, 255]), "decode", "64")
add("short-atom", "-2", atom(5, 'h', -2), "decode", "64")
add("short-vector", "[1 -2 3]", vec(5, 'h', [1, -2, 3]), "decode", "64")
add("int-atom-64", "1", atom(6, 'i', 1), "decode", "64")
add("int-vector-64", "[1 2 3]", vec(6, 'i', [1, 2, 3]), "decode", "64")
add("int-null-vector", "[-2147483648]", vec(6, 'i', [-2**31]), "decode", "64")
add("real-atom-64", "1.5", atom(8, 'f', 1.5), "decode", "64")
add("real-vector-64", "[1.5 2.5]", vec(8, 'f', [1.5, 2.5]), "decode", "64")
add("float-null-q", "NaN (bit pattern fff8...)", bytes.fromhex("f7000000000000f8ff"), "decode", "64", expect={"float": "nan"})
add("float-infinities", "[inf -inf]", fv([float('inf'), float('-inf')]), "decode", "64", expect={"array": ["inf", "-inf"], "dtype": "float64"})
guid = bytes(range(16))
assert hdr(b'\xfe' + guid).hex() == "0100000019000000fe000102030405060708090a0b0c0d0e0f"
add("guid-atom", "[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]", b'\xfe' + guid, "decode", "64")
add("guid-vector", "two guids as a 2x16 array", bytes([2, 0]) + struct.pack('<i', 2) + guid + bytes(range(16, 32)), "decode", "64",
    expect={"array": [list(range(16)), list(range(16, 32))], "dtype": "int64"})
add("char-atom-high-byte", "U+FFFD (invalid UTF-8)", b'\xf6\xff', "decode", "any", expect={"char": "�"})
add("string-invalid-utf8", "invalid UTF-8 replaced", charvec(b'\xffa'), "decode", "any", expect={"str": "�a"})
assert hdr(atom(12, 'q', 1)).hex() == "0100000011000000f40100000000000000"

# --- reject ----------------------------------------------------------------
def rej(name, why, body=None, whole=None, **kw):
    add(name, why, body, "reject", "any", error="system", whole=whole, **kw)

rej("reject-unused-type-3", "type 3 does not exist", bytes([3, 0]) + struct.pack('<i', 0))
rej("reject-atom-type-minus-3", "atom type -3", b'\xfd\x00')
rej("reject-enum-vector-20", "enumerations are never sent", bytes([20, 0]) + struct.pack('<i', 0))
rej("reject-type-112", "dynamic load: layout unconfirmed", b'\x70\x00')
rej("reject-type-77-anymap", "mapped list type", bytes([77, 0]) + struct.pack('<i', 0))
rej("reject-truncated-vector", "long vector of 3 with two elements", bytes([7, 0]) + struct.pack('<i', 3) + struct.pack('<qq', 1, 2))
rej("reject-truncated-atom", "long atom with four bytes", b'\xf9\x01\x00\x00\x00')
rej("reject-empty-body", "no object at all", b'')
rej("reject-count-too-big", "long vector claiming 2^31-1 elements", bytes([7, 0]) + struct.pack('<i', 0x7fffffff) + b'\x00' * 8)
rej("reject-count-negative", "long vector with count -1", bytes([7, 0]) + struct.pack('<i', -1))
rej("reject-list-count-too-big", "general list claiming 1e9 items", bytes([0, 0]) + struct.pack('<i', 10**9))
rej("reject-char-vector-count", "char vector longer than the message", bytes([10, 0]) + struct.pack('<i', 100) + b'abc')
rej("reject-symbol-vector-count", "symbol vector of 5 with one symbol", bytes([11, 0]) + struct.pack('<i', 5) + b'a\0')
rej("reject-trailing-bytes", "one byte after the object", longa(1) + b'\x00')
rej("reject-symbol-unterminated", "symbol atom without NUL", b'\xf5abc')
rej("reject-error-unterminated", "error text without NUL", b'\x80type')
nest = longa(1)
for _ in range(65): nest = gl([nest])
rej("reject-nesting-65", "65 nested general lists", nest)
rej("reject-lambda-text-not-chars", "lambda whose text is a long atom", b'\x64\x00' + longa(1))
rej("reject-lambda-context-unterminated", "lambda context without NUL", b'\x64kg')
rej("reject-dict-length-mismatch", "2 keys, 1 value", dct(symvec(["a", "b"]), lv([1])))
rej("reject-dict-atom-values", "dict whose values are an atom", dct(symvec(["a"]), longa(1)))
rej("reject-dict-atom-keys", "dict whose keys are an atom", dct(sym("a"), lv([1])))
rej("reject-table-column-lengths", "table columns of different length", table(["a", "b"], [lv([1, 2]), lv([1])]))
rej("reject-table-names-not-symbols", "table names are a long vector", bytes([98, 0]) + dct(lv([1]), gl([lv([1])])))
rej("reject-table-without-dict", "table tag followed by a long atom", bytes([98, 0]) + longa(1))
rej("reject-projection-count", "projection claiming 2^31-1 items", b'\x68' + struct.pack('<i', 0x7fffffff))
rej("reject-guid-truncated", "guid atom with 15 bytes", b'\xfe' + bytes(15))
# header-level (session only; the codec harness skips these)
add("reject-header-oversize", "length field above max_message", None, "reject", "any", error="system", whole=bytes.fromhex("01000000ffffff7f"))
add("reject-header-length-below-8", "length field 7", None, "reject", "any", error="system", whole=bytes.fromhex("0100000007000000"))
add("reject-header-negative-length", "length field -1", None, "reject", "any", error="system", whole=bytes.fromhex("01000000ffffffff"))
add("reject-header-bad-byte-order", "byte 0 is 2", None, "reject", "any", error="system", whole=bytes.fromhex("0200000008000000"))
add("reject-header-bad-message-type", "message type 3", None, "reject", "any", error="system", whole=bytes.fromhex("010300000a0000006500"))
add("reject-header-bad-compression-flag", "compression flag 2", None, "reject", "any", error="system", whole=bytes.fromhex("010002000a0000006500"))
add("reject-header-reserved-byte", "byte 3 set (large-message extension, capability 5 only)", None, "reject", "any", error="system", whole=bytes.fromhex("010000010a0000006500"))
# compressed
def chdr(usize, stream):
    return bytes.fromhex("01000100") + struct.pack('<I', 12 + len(stream)) + struct.pack('<I', usize) + stream
add("reject-compressed-bomb", "declares 256 MiB from a 15-byte stream", None, "reject", "any", error="system",
    whole=chdr(0x10000000, bytes.fromhex("00 41 41".replace(" ", ""))))
add("reject-compressed-bad-backref", "back-reference before any data was indexed", None, "reject", "any", error="system",
    whole=chdr(20, bytes.fromhex("010000")))
add("reject-compressed-truncated", "stream ends before the declared size", None, "reject", "any", error="system",
    whole=comp[:-3][:4] + struct.pack('<I', len(comp) - 3) + comp[8:-3])
add("reject-compressed-usize-below-header", "declared size 4", None, "reject", "any", error="system",
    whole=chdr(4, b"\x00"))

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vectors.json')
json.dump(V, open(out, 'w'), indent=1)
print(len(V), "vectors")
