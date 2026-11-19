"""Whole-array paths for indexing by a vector, indexing strings, joining an atom, array power and
grading strings. Each case pins the result the element-by-element code gave, so a fast path
cannot change a value, a type, a shape or an error."""
import unittest

from klongpy import KlongInterpreter
from klongpy.types import KlongException
from tests.backend_compat import is_torch_backend


def py(x):
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [py(e) for e in x]
    return x


class _Base(unittest.TestCase):
    def kg(self, src):
        return KlongInterpreter()(src)

    def same(self, src, expected):
        r = py(self.kg(src))
        self.assertEqual(r, expected, src)
        return r

    def fails(self, src):
        with self.assertRaises(KlongException, msg=src):
            self.kg(src)


class TestIndexByVector(_Base):
    def test_values(self):
        self.same('[10 20 30 40]@[2 0 -1 3]', [30, 10, 40, 40])
        self.same('[10 20 30 40]@[0 1 2 3]', [10, 20, 30, 40])
        self.same('[1.5 2.5 3.5]@[2 2 0]', [3.5, 3.5, 1.5])
        self.same('[1 2 3]@[-1 -2 -3]', [3, 2, 1])
        self.same('[2 3 4]@(!3)', [2, 3, 4])
        self.same('[1 2 3]@(0<[1 0 1])', [2, 1, 2])
        self.same('[1 2 3]@[0 1 2]@[2 1]', [3, 2])

    def test_rows_of_a_matrix(self):
        self.same('[[1 2][3 4][5 6]]@[2 0]', [[5, 6], [1, 2]])
        self.same('[[1 2][3 4]]@[-1]', [[3, 4]])

    def test_out_of_range_is_an_error(self):
        for s in ('[10 20 30 40]@[4]', '[10 20 30 40]@[-5]', '[1 2 3]@[1 2 3 0 0]',
                  '[1 2 3]@[0 1 100000000000000000000]', '[]@[0]', '[[1 2][3 4]]@[2]'):
            self.fails(s)

    def test_float_and_non_integer_indices_are_errors(self):
        for s in ('[10 20 30 40]@[1.0 2.0]', '[10 20 30 40]@[1.5]', '[1 2 3]@[0.5 1]',
                  '[1 2 3]@[1 "a"]'):
            self.fails(s)

    def test_scalar_and_empty_index(self):
        self.same('[10 20 30 40]@2', 30)
        self.same('[10 20 30 40]@-1', 40)
        self.fails('[10 20 30 40]@4')
        self.same('[10 20 30 40]@[]', [])
        self.same('[]@[]', [])

    def test_nested_index_follows_the_shape_of_the_index(self):
        self.same('[10 20 30 40]@[[0 1][2 3]]', [[10, 20], [30, 40]])
        self.same('[10 20 30 40]@[[0 1][2]]', [[10, 20], [30]])
        self.same('[10 20 30 40]@[0 [1 2]]', [10, [20, 30]])
        self.same('[[1 2][3 4]]@[[0 0][1 1]]', [[[1, 2], [1, 2]], [[3, 4], [3, 4]]])
        self.same('[[1 2][3 4]]@[[-1 0]]', [[[3, 4], [1, 2]]])
        self.same('[1 2 3]@[[0][1]]', [[1], [2]])

    def test_mixed_and_object_lists(self):
        self.same('[1 "a" 2.5 [1 2]]@[3 0 1]', [[1, 2], 1, "a"])
        self.same('[[1 2][3 4 5]]@[1 0]', [[3, 4, 5], [1, 2]])
        self.same('[100000000000000000000 5 7]@[0 1]', [100000000000000000000, 5])
        self.same('[100000000000000000000 5 7]@0', 100000000000000000000)

    def test_list_of_strings(self):
        self.assertEqual(py(self.kg('["ab" "cd" "ef"]@[2 0 0 -2]')), ["ef", "ab", "ab", "cd"])
        self.fails('["ab" "cd"]@[2]')

    def test_other_targets_unchanged(self):
        self.same('{y+x*x}@[2 3]', 7)
        self.assertEqual(py(self.kg(':{["a" 1]["b" 2]}@["a" "b"]')), [1, 2])
        self.same('[1 2 3]@"a"', [1, 2, 3])
        self.fails('3@[0]')

    def test_sorting_by_the_grade(self):
        self.same('r::[3 1 2 1];r@<r', [1, 1, 2, 3])
        self.same('r::[3 1 2];r@>r', [3, 2, 1])

    def test_large_vector(self):
        self.assertEqual(self.kg('+/(!100000)@|!100000'), 4999950000)


class TestIndexString(_Base):
    def test_characters_and_substrings(self):
        self.assertEqual(self.kg('"hello"@[1 4 0]'), "eoh")
        self.assertEqual(self.kg('"hello"@1'), "e")
        self.assertEqual(self.kg('"hello"@-1'), "o")
        self.assertEqual(self.kg('"hello"@[]'), "")
        self.assertEqual(self.kg('""@[]'), "")

    def test_result_type(self):
        from klongpy.core import KGChar
        self.assertIsInstance(self.kg('"hello"@1'), KGChar)
        r = self.kg('"hello"@[1 4]')
        self.assertIsInstance(r, str)
        self.assertNotIsInstance(r, KGChar)

    def test_errors(self):
        for s in ('"hello"@5', '"hello"@[0 9]', '""@0', '"hello"@[-6]', '"hello"@[1 [2 3]]',
                  '"abc"@[0.0 1.0]', '"abc"@[0 []]'):
            self.fails(s)
        self.assertEqual([str(q) for q in self.kg('"abc"@[[0 1][2]]')], ["ab", "c"])    # a list of index lists indexes item by item

    def test_whole_number_float_and_non_integer_index(self):
        self.assertEqual(self.kg('"hello"@1.0'), "hello")
        self.assertEqual(self.kg('"abc"@"a"'), "abc")

    def test_code_points(self):
        s = '"héllo\U0001F600éx"'
        self.assertEqual(self.kg(s + '@[1 5 6 7 8]'), "é\U0001F600éx")
        self.assertEqual(self.kg(s + '@5'), "\U0001F600")
        self.assertEqual(self.kg(s + '@6'), "e")
        self.assertEqual(self.kg(s + '@7'), "́")
        self.assertEqual(self.kg(s + '@-1'), "x")
        self.assertEqual(self.kg('#' + s), 9)

    def test_nested_index_gives_strings(self):
        self.assertEqual(py(self.kg('"hello"@[[0 1][2 3]]')), ["he", "ll"])
        self.assertEqual(py(self.kg('"abc"@[[0][1]]')), ["a", "b"])
        self.assertEqual(py(self.kg('"abc"@[[0 1][1 2]]')), ["ab", "bc"])
        self.fails('"abc"@[[0 3][1 2]]')

    def test_long_non_ascii_string_by_element(self):
        k = KlongInterpreter()
        k('s::5000#"aé"')
        self.assertEqual(py(k("{s@x}'[0 1 4998 4999]")), ["a", "é", "a", "é"])
        self.assertEqual(k('#s@!5000'), 5000)


class TestJoinAtom(_Base):
    def test_values_and_types(self):
        self.same('[1 2 3],4', [1, 2, 3, 4])
        self.same('[1 2 3],4.5', [1, 2, 3, 4.5])
        self.same('[1.5 2],3', [1.5, 2, 3])
        self.same('1,[2 3]', [1, 2, 3])
        self.same('0.5,[1 2]', [0.5, 1, 2])
        self.same('[1 2],-1', [1, 2, -1])
        self.same("a::[];{a::a,x;x}'!5;a", [0, 1, 2, 3, 4])

    def test_kinds_that_take_the_general_path(self):
        self.same('[1 2],9223372036854775807', [1, 2, 9223372036854775807])
        self.same('[1 2],100000000000000000000', [1, 2, 100000000000000000000])
        self.same('[1 2],"a"', [1, 2, "a"])
        self.same('[[1 2][3 4]],5', [[1, 2], [3, 4], 5])
        self.same('[],5', [5])
        self.same('5,[]', [5])

    def test_int_vector_stays_int(self):
        r = self.kg('[1 2 3],4')
        self.assertIn(str(r.dtype), ("int64", "torch.int64"))


class TestArrayPower(_Base):
    def test_small_integer_exponent(self):
        self.same('[1 2 3]^2', [1, 4, 9])
        self.same('[1 2 3]^0', [1, 1, 1])
        self.same('[1 2 3]^1', [1, 2, 3])
        self.same('[1 2 3]^3', [1, 8, 27])
        self.same('[[1 2][3 4]]^2', [[1, 4], [9, 16]])
        self.same('2^[1 2 3]', [2, 4, 8])
        self.same('[2 3]^[2 3]', [4, 27])
        self.same('(-8)^[1 2]', [-8, 64])

    def test_integer_arrays_wrap_in_64_bits_in_every_arrangement(self):
        self.same('[3037000500 2]^2', [-9223372036709301616, 4])
        self.same('[2 3]^64', [0, 8733086111712066817])
        self.same('[2 3]^[64 64]', [0, 8733086111712066817])
        self.same('2^[62 63 64]', [4611686018427387904, -9223372036854775808, 0])
        self.same('3^[40 41 42]', [-6289078614652622815, -420491770248316829, -1261475310744950487])
        self.same('10^[19]', [-8446744073709551616])
        self.same('[10]^19', [-8446744073709551616])
        self.same('2^64', 18446744073709551616)  # two scalars are exact

    def test_a_float_gives_floats(self):
        self.same('[2.0 3]^2', [4.0, 9.0])
        self.same('[1 2 3]^2.0', [1.0, 4.0, 9.0])
        self.same('[1.5 2.5]^2', [2.25, 6.25])
        self.same('1.5^[1 2]', [1.5, 2.25])
        self.same('[1.0 2.0]^2', [1.0, 4.0])
        self.same('2^[1 2.5]', [2.0, 5.656854249492381] if not is_torch_backend() else [2.0, 5.656854152679443])
        r = py(self.kg('[2.0 3]^64'))
        self.assertTrue(all(isinstance(x, float) for x in r))
        self.assertAlmostEqual(r[0], 2.0 ** 64, delta=2.0 ** 64 * 1e-6)
        self.assertEqual(py(self.kg('[2.0 3]*2')), [4.0, 6.0])

    def test_negative_integer_exponent_gives_floats_in_every_arrangement(self):
        self.assertEqual(self.kg('2^(0-1)'), 0.5)
        r = py(self.kg('[2 4]^(0-1)'))
        self.assertEqual(r, [0.5, 0.25])
        self.assertEqual(py(self.kg('2^[0 -1]')), [1.0, 0.5])
        self.assertEqual(py(self.kg('[2 4]^[1 -2]')), [2.0, 0.0625])
        self.same('[0 2]^(0-1)', [float('inf'), 0.5])

    def test_result_is_an_integer_array(self):
        r = self.kg('(1+!10)^2')
        self.assertIn(str(r.dtype), ("int64", "torch.int64"))


class TestGradeStrings(_Base):
    def test_string_grade(self):
        self.same('<"foobar"', [4, 3, 0, 1, 2, 5])
        self.same('>"foobar"', [5, 2, 1, 0, 3, 4])
        self.same('<"hello, world"', [6, 5, 11, 1, 0, 2, 3, 10, 4, 8, 9, 7])
        self.assertEqual(self.kg('"foobar"@<"foobar"'), "abfoor")
        self.assertEqual(self.kg('"foobar"@>"foobar"'), "roofba")

    def test_string_grade_code_points(self):
        self.same('<"béa\U0001F600z"', [2, 0, 4, 1, 3])
        self.same('>"béa\U0001F600z"', [3, 1, 4, 0, 2])

    def test_empty_and_single(self):
        self.same('<""', [])
        self.same('<"a"', [0])

    def test_list_of_strings(self):
        self.same('<["pear" "apple" "fig" "apple"]', [1, 3, 2, 0])
        self.same('>["pear" "apple" "fig" "apple"]', [0, 2, 3, 1])
        self.assertEqual(py(self.kg('w::["pear" "apple" "fig"];w@<w')), ["apple", "fig", "pear"])

    def test_other_lists_unchanged(self):
        self.same('<[[3 1][1 2][0 5]]', [1, 0, 2])
        self.same('>[[1] [2] [3]]', [2, 1, 0])
        self.same('<[3 1 2]', [1, 2, 0])


if __name__ == "__main__":
    unittest.main()
