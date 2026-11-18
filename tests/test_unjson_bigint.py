"""``.unjson``: a list that holds an integer beyond 64 bits keeps every integer exact, on numpy and torch."""
import importlib.util
import unittest

from klongpy import KlongInterpreter

BACKENDS = ["numpy"] + (["torch"] if importlib.util.find_spec("torch") is not None else [])
BIG = 9223372036854775808


class TestUnjsonBigInt(unittest.TestCase):
    def interp(self, backend):
        return KlongInterpreter() if backend == "numpy" else KlongInterpreter(backend="torch", device="cpu")

    def test_list_with_a_float(self):
        for b in BACKENDS:
            with self.subTest(backend=b):
                r = self.interp(b)('.unjson("[9223372036854775808,0.5]")')
                self.assertEqual([int(r[0]), float(r[1])], [BIG, 0.5])
                self.assertIs(type(r[0]), int)

    def test_several_integers_and_negative(self):
        for b in BACKENDS:
            with self.subTest(backend=b):
                r = self.interp(b)('.unjson("[1,18446744073709551616,-9223372036854775809,2.5]")')
                self.assertEqual(list(r), [1, 2**64, -2**63 - 1, 2.5])

    def test_alone_and_in_a_list_of_integers(self):
        for b in BACKENDS:
            with self.subTest(backend=b):
                k = self.interp(b)
                self.assertEqual(k('.unjson("9223372036854775808")'), BIG)
                self.assertEqual(list(k('.unjson("[9223372036854775808,1]")')), [BIG, 1])

    def test_nested(self):
        for b in BACKENDS:
            with self.subTest(backend=b):
                r = self.interp(b)('.unjson("[[1,2],[9223372036854775808,0.5]]")')
                self.assertEqual([int(v) for v in r[0]], [1, 2])
                self.assertEqual([int(r[1][0]), float(r[1][1])], [BIG, 0.5])

    def test_json_round_trip(self):
        for b in BACKENDS:
            with self.subTest(backend=b):
                k = self.interp(b)
                self.assertEqual(k('.json(.unjson("[9223372036854775808,0.5]"))'), "[9223372036854775808,0.5]")

    def test_ordinary_lists_unchanged(self):
        for b in BACKENDS:
            with self.subTest(backend=b):
                k = self.interp(b)
                self.assertEqual(k('.json(.unjson("[1,2.5,[3,4]]"))'), "[1,2.5,[3,4]]")
                self.assertEqual(k('.json(.unjson("[1,2,3]"))'), "[1,2,3]")


if __name__ == "__main__":
    unittest.main()
