"""
Signatures of .extern and .entry (klongpy/static/signature.py
step 1): the grammar, positioned errors for malformed text, names, and the
check of values against types.
"""
import unittest

import numpy as np

from klongpy.static import Signature, SignatureError, Ty, c_identifier, parse_signature, parse_type
from klongpy.static.signature import Mismatch, check_entry_name, check_symbol, conform
from klongpy.types import KGChar, KGSym

GOOD = [
    # text, params, result, canonical text
    ("int -> float", [Ty("int")], Ty("float"), "int -> float"),
    ("float[64] -> float", [Ty("float", (64,))], Ty("float"), "float[64] -> float"),
    ("int float -> void", [Ty("int"), Ty("float")], Ty("void"), "int float -> void"),
    ("-> int", [], Ty("int"), "-> int"),
    ("->int", [], Ty("int"), "-> int"),
    ("  int->float  ", [Ty("int")], Ty("float"), "int -> float"),
    ("float[16,241] int[3] -> float[241]", [Ty("float", (16, 241)), Ty("int", (3,))], Ty("float", (241,)),
     "float[16,241] int[3] -> float[241]"),
    ("float[ 2 , 3 ] -> int[2,3,4,5]", [Ty("float", (2, 3))], Ty("int", (2, 3, 4, 5)), "float[2,3] -> int[2,3,4,5]"),
    ("str int -> void", [Ty("str"), Ty("int")], Ty("void"), "str int -> void"),
    ("int\tint -> int", [Ty("int"), Ty("int")], Ty("int"), "int int -> int"),
]

# text, the part of the text the error points at, a word of the message
BAD = [
    ("int float", "int float", "'->'"),
    ("", "", "'->'"),
    ("int => float", "int => float", "'->'"),
    ("int -> float -> int", "->", "exactly one"),
    ("int ->", "->", "result type"),
    ("int -> flot", "flot", "unknown type 'flot'"),
    ("Int -> int", "Int", "unknown type 'Int'"),
    ("int -> str", "str", "str cannot be a result"),
    ("void -> int", "void", "result type only"),
    ("int void -> int", "void", "result type only"),
    ("int[] -> int", "[]", "dimensions"),
    ("int[0] -> int", "[0]", "positive"),
    ("int[-1] -> int", "[-1]", "dimensions"),
    ("int[N] -> int", "[N]", "dimensions"),
    ("int[2,] -> int", "[2,]", "dimensions"),
    ("int[1,2,3,4,5] -> int", "[1,2,3,4,5]", "at most 4"),
    ("str[4] -> int", "[4]", "cannot have dimensions"),
    ("int -> void[2]", "[2]", "cannot have dimensions"),
    ("int, float -> int", "int,", "not a type"),
    ("int[3 -> int", "int[3", "not a type"),
    ("int -> float int", "int", "after the result type"),
    ("3 -> int", "3", "expected a type"),
    ("int -> 7", "7", "expected a type"),
]


class TestParse(unittest.TestCase):
    def test_well_formed(self):
        for text, params, result, canon in GOOD:
            with self.subTest(text=text):
                sig = parse_signature(text)
                self.assertIsInstance(sig, Signature)
                self.assertEqual(list(sig.params), params)
                self.assertEqual(sig.result, result)
                self.assertEqual(sig.arity, len(params))
                self.assertEqual(str(sig), canon)
                self.assertEqual(parse_signature(canon), sig, "the canonical text parses to the same signature")

    def test_malformed_text_is_a_positioned_error(self):
        for text, at, word in BAD:
            with self.subTest(text=text):
                with self.assertRaises(SignatureError) as cm:
                    parse_signature(text)
                e = cm.exception
                self.assertIsInstance(e, ValueError)
                self.assertIn(word, e.message)
                self.assertEqual(text[e.span[0]:e.span[1]], at)

    def test_str_is_for_extern_parameters_only(self):
        self.assertEqual(parse_signature("str -> int").params, (Ty("str"),))
        with self.assertRaises(SignatureError) as cm:
            parse_signature("str -> int", allow_str=False)
        self.assertIn("externs only", cm.exception.message)

    def test_not_a_string(self):
        for v in (3, None, ["int"]):
            with self.assertRaises(SignatureError):
                parse_signature(v)

    def test_parse_type(self):
        self.assertEqual(parse_type("float[16,241]"), Ty("float", (16, 241)))
        self.assertEqual(parse_type(" int "), Ty("int"))
        for bad in ("", "int float", "int[", "floats"):
            with self.assertRaises(SignatureError):
                parse_type(bad)

    def test_ty(self):
        t = Ty("float", (16, 241))
        self.assertEqual((t.rank, t.size, t.is_array, str(t)), (2, 3856, True, "float[16,241]"))
        self.assertEqual((Ty("int").rank, Ty("int").size, Ty("int").is_array, str(Ty("int"))), (0, 1, False, "int"))
        self.assertEqual(len({Ty("int"), Ty("int", ())}), 1)


class TestNames(unittest.TestCase):
    def test_c_identifier(self):
        self.assertEqual(c_identifier("read.adc"), "read_adc")
        self.assertEqual(c_identifier("a.b.c"), "a_b_c")
        self.assertEqual(c_identifier("step"), "step")

    def test_extern_names(self):
        for ok in ("read_adc", "f", "spi_xfer2", "INIT", "_x", "x"):
            self.assertIsNone(check_symbol(ok), ok)
        for bad, word in (("kg_trap", "kg_"), ("int", "reserved"), ("while", "reserved"), ("main", "reserved"),
                          ("memcpy", "reserved"), ("read.adc", "no dot"), ("2x", "C identifier"), (".p", "no dot"),
                          ("", "C identifier"), ("café", "C identifier"), (3, "C symbol must be a string")):
            with self.subTest(name=bad):
                self.assertIn(word, check_symbol(bad))

    def test_entry_names(self):
        for ok in ("step", "run", "int", "a.b"):       # prog_int is a fine C name
            self.assertIsNone(check_entry_name(ok), ok)
        for bad in ("init", "t", "size", "storage.t", "y", "a_b", ""):
            with self.subTest(name=bad):
                self.assertIsNotNone(check_entry_name(bad))


class TestConform(unittest.TestCase):
    def test_scalars(self):
        self.assertEqual(conform(Ty("int"), 3, "v"), 3)
        self.assertIs(type(conform(Ty("int"), np.int64(3), "v")), int)
        self.assertIs(type(conform(Ty("int"), True, "v")), int)
        self.assertIs(type(conform(Ty("float"), 3, "v")), float)        # an int converts to float
        self.assertIs(type(conform(Ty("float"), np.float32(1.5), "v")), float)
        self.assertEqual(conform(Ty("int"), np.array(7), "v"), 7)       # a 0-d array is a scalar
        self.assertIsNone(conform(Ty("void"), "anything", "v"))
        self.assertEqual(conform(Ty("str"), "abc", "v"), "abc")

    def test_arrays(self):
        a = conform(Ty("float", (2, 2)), [[1, 2], [3, 4]], "v")
        self.assertEqual((a.dtype, a.shape), (np.float64, (2, 2)))
        a = conform(Ty("int", (3,)), np.array([1, 2, 3], dtype=np.int32), "v")
        self.assertEqual(a.dtype, np.int64)

    def test_mismatches(self):
        cases = [
            (Ty("int"), 1.5, "type", "must be int, got a float"),
            (Ty("int"), [1, 2], "type", "must be int, got"),
            (Ty("float"), "s", "type", "must be float, got a string"),
            (Ty("float"), None, "type", "got nothing"),
            (Ty("str"), 3, "type", "must be str, got an integer"),
            (Ty("str"), KGSym("a"), "type", "must be str, got a symbol"),
            (Ty("str"), KGChar("a"), "type", "must be str"),
            (Ty("int", (3,)), [1.5, 2, 3], "type", "must be int[3], got float[3]"),
            (Ty("float", (3,)), [1, 2], "shape", "must be float[3], got int[2]"),
            (Ty("float", (2, 2)), [1, 2, 3, 4], "shape", "must be float[2,2], got int[4]"),
            (Ty("float", (3,)), 1.0, "type", "must be float[3], got a float"),
            (Ty("float", (2,)), [[1], [2, 3]], "type", "must be float[2]"),
            (Ty("float", (2,)), np.array(["a", "b"]), "type", "must be float[2]"),
        ]
        for ty, value, kind, text in cases:
            with self.subTest(ty=str(ty), value=repr(value)):
                with self.assertRaises(Mismatch) as cm:
                    conform(ty, value, "the value")
                self.assertEqual(cm.exception.kind, kind)
                self.assertIn(text, cm.exception.message)
                self.assertTrue(cm.exception.message.startswith("the value must be"))


if __name__ == "__main__":
    unittest.main()
