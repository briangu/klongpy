"""Ticket 015: operator edge cases (reshape, reverse, each-left/right, dict join,
over, shape, atomic length mismatch, indexing, division by zero, format, enumerate)."""
import subprocess
import sys
import unittest
from pathlib import Path

from klongpy import KlongInterpreter
from klongpy.core import KLONG_UNDEFINED
from klongpy.types import KlongException
from tests.backend_compat import dtype_kind, is_torch_backend, plain

ROOT = Path(__file__).resolve().parent.parent


def py(x):
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [py(e) for e in x]
    return x


class Ticket015(unittest.TestCase):

    def kg(self, src):
        return KlongInterpreter()(src)

    def assert_klong_error(self, src):
        with self.assertRaises(KlongException, msg=src):
            self.kg(src)

    # 1, 2 reshape
    def test_reshape_empty_source(self):
        self.assert_klong_error('[2 2]:^[]')
        self.assert_klong_error('[2 2]:^""')
        self.assert_klong_error('5:^[]')

    def test_reshape_empty_source_zero_size_ok(self):
        self.assertEqual(py(self.kg('[0 2]:^[]')), [])

    def test_reshape_negative_not_minus_one(self):
        self.assert_klong_error('[-2 3]:^[1 2 3 4 5 6]')
        self.assert_klong_error('[-3 2]:^[1 2 3 4 5 6]')

    def test_reshape_minus_one_still_works(self):
        self.assertEqual(py(self.kg('[-1 2]:^[1 2 3 4 5 6]')), [[1, 2], [3, 4], [5, 6]])
        self.assertEqual(py(self.kg('[2 -1]:^!10')), [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

    # 3
    def test_reverse_atom(self):
        self.assertEqual(self.kg('|5'), 5)
        self.assertEqual(self.kg('|"abc"'), "cba")
        self.assertEqual(py(self.kg('|[]')), [])
        self.assertEqual(py(self.kg('|[1 2 3]')), [3, 2, 1])

    # 4
    def test_each_left_atom(self):
        self.assertEqual(py(self.kg('1,:\\5')), [1, 5])
        self.assertEqual(py(self.kg('1,:\\[1 2]')), [[1, 1], [1, 2]])

    def test_each_right_atom(self):
        self.assertEqual(py(self.kg('1,:/5')), [5, 1])
        self.assertEqual(py(self.kg('1,:/[2 3]')), [[2, 1], [3, 1]])

    # 5
    def test_join_dict_bad_pair(self):
        for src in ['d:::{[1 0]};d,5', 'd:::{[1 0]};d,[1]', 'd:::{[1 0]};d,[[1 2]]', 'd:::{[1 0]};d,[1 2 3]']:
            self.assert_klong_error(src)
        # ticket 063: a dictionary on the right is a merge
        self.assertEqual(self.kg('d:::{[1 0]};d,:{[3 4]}'), {1: 0, 3: 4})

    def test_join_dict_good_pair(self):
        r = self.kg('d:::{[1 0]};d,[5 6]')
        self.assertEqual(r[5], 6)
        self.assertEqual(py(self.kg('[1 2],:{[1 0]}')), {1: 2})

    # 6
    def test_over_min_max_string(self):
        self.assertEqual(self.kg('&/"abc"'), "a")
        self.assertEqual(self.kg('|/"cab"'), "c")

    def test_over_plus_empty(self):
        self.assertEqual(py(self.kg('+/[]')), [])
        self.assertEqual(py(self.kg('&/[]')), [])

    # 7
    def test_shape_ragged(self):
        self.assertEqual(py(self.kg('^[[1] [2 3]]')), [2])
        self.assertEqual(py(self.kg('^[[[1]] [[2] [3]]]')), [2])
        self.assertEqual(py(self.kg('^["abc" "de"]')), [2])

    def test_shape_regular_unchanged(self):
        self.assertEqual(py(self.kg('^[[1 2] [3 4]]')), [2, 2])
        self.assertEqual(py(self.kg('^[["ab" "cd"] ["ef" "gh"]]')), [2, 2, 2])
        self.assertEqual(py(self.kg('^["abcd" "efgh"]')), [2, 4])
        self.assertEqual(py(self.kg('^"hello"')), [5])
        self.assertEqual(py(self.kg('^[1 2 3]')), [3])
        self.assertEqual(self.kg('^5'), 0)
        self.assertEqual(py(self.kg('^[1 [2]]')), [2])

    # 8
    def test_atomic_dyad_ragged_length_mismatch(self):
        self.assert_klong_error('[[1] [2 3]]+[1 2 3]')
        self.assert_klong_error('[1 2 3]+[1 2]')

    def test_atomic_dyad_ragged_length_mismatch_python_O(self):
        code = (
            "import sys\n"
            "from klongpy import KlongInterpreter\n"
            "from klongpy.types import KlongException\n"
            "try:\n"
            "    KlongInterpreter()('[[1] [2 3]]+[1 2 3]')\n"
            "except KlongException:\n"
            "    sys.exit(0)\n"
            "sys.exit(1)\n"
        )
        r = subprocess.run([sys.executable, "-O", "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    # negative indices (owner decision)
    def test_index_negative_wraps(self):
        self.assertEqual(self.kg('[1 2 3]@-1'), 3)
        self.assertEqual(self.kg('[1 2 3]@-3'), 1)
        self.assertEqual(self.kg('"abc"@-1'), "c")
        self.assertEqual(py(self.kg('[1 2 3]@[0 -1]')), [1, 3])

    def test_index_out_of_range_raises(self):
        for src in ['[1 2 3]@-4', '[1 2 3]@3', '"abc"@3', '"abc"@-4', '[1 2 3]@[0 3]', '[1 2 3]@[-4]']:
            self.assert_klong_error(src)

    def test_index_dict_unaffected(self):
        self.assertEqual(self.kg('d:::{[5 6]};d@5'), 6)

    # division by zero (owner decision)
    def test_divide_by_zero_scalar_array_consistent(self):
        self.assertIs(self.kg('1%0'), KLONG_UNDEFINED)
        self.assertIs(self.kg('0%0'), KLONG_UNDEFINED)
        if is_torch_backend():
            # Ticket 015 decision: a tensor cannot hold :undefined, so the torch
            # backend keeps the IEEE values (and the tensor stays in the autograd
            # graph). Atoms are :undefined on every backend (asserted above).
            inf = float('inf')
            self.assertEqual(plain(self.kg('[1 2]%0')), [inf, inf])
            r = plain(self.kg('[1 0]%[1 0]'))
            self.assertEqual(r[0], 1.0)
            self.assertTrue(r[1] != r[1])  # 0%0 is nan
            self.assertEqual(plain(self.kg('5%[1 0]')), [5.0, inf])
            return
        r = self.kg('[1 2]%0')
        self.assertEqual(len(r), 2)
        self.assertTrue(all(x is KLONG_UNDEFINED for x in r))
        r = self.kg('[1 0]%[1 0]')
        self.assertEqual(r[0], 1.0)
        self.assertIs(r[1], KLONG_UNDEFINED)
        r = self.kg('5%[1 0]')
        self.assertEqual(r[0], 5.0)
        self.assertIs(r[1], KLONG_UNDEFINED)

    def test_divide_without_zero_is_float_array(self):
        r = self.kg('[1 2]%[1 2]')
        self.assertEqual(dtype_kind(r), "f")
        self.assertEqual(py(r), [1.0, 1.0])
        self.assertEqual(py(self.kg('[[1 2] [3 4]]%2')), [[0.5, 1.0], [1.5, 2.0]])

    def test_divide_nested_zero(self):
        r = self.kg('[[1 2] [3 4]]%[[1 0] [0 4]]')
        if is_torch_backend():
            # torch keeps the IEEE values in the tensor (ticket 015 decision)
            inf = float('inf')
            self.assertEqual(plain(r), [[1.0, inf], [inf, 1.0]])
            return
        self.assertEqual(r[0][0], 1.0)
        self.assertIs(r[0][1], KLONG_UNDEFINED)
        self.assertIs(r[1][0], KLONG_UNDEFINED)
        self.assertEqual(r[1][1], 1.0)

    def test_modulo_zero(self):
        self.assertIs(self.kg('7:%0'), KLONG_UNDEFINED)
        self.assertIs(self.kg('7!0'), KLONG_UNDEFINED)
        if is_torch_backend():
            # torch keeps the IEEE values in the tensor (ticket 015 decision):
            # a remainder by zero is nan, an integer division by zero is inf,
            # and the list becomes a float tensor
            r = plain(self.kg('[7 8]!0'))
            self.assertTrue(all(x != x for x in r))
            self.assertEqual(plain(self.kg('[7 8]:%[2 0]')), [3.0, float('inf')])
            self.assertEqual(plain(self.kg('[7 8]:%[2 3]')), [3, 2])  # no zero: still integers
            self.assertEqual(dtype_kind(self.kg('[7 8]:%[2 3]')), "i")
            return
        r = self.kg('[7 8]!0')
        self.assertTrue(all(x is KLONG_UNDEFINED for x in r))
        r = self.kg('[7 8]:%[2 0]')
        self.assertEqual(r[0], 3)
        self.assertIs(r[1], KLONG_UNDEFINED)

    def test_modulo_and_intdiv_unchanged(self):
        self.assertEqual(self.kg('7!5'), 2)
        self.assertEqual(self.kg('(-7)!5'), -2)
        self.assertEqual(self.kg('10:%8'), 1)

    # format
    def test_format_list_arg(self):
        # "$" is atomic (docstring): a list is formatted element by element, no RecursionError
        self.assertEqual(py(self.kg('$[1 2 3]')), ["1", "2", "3"])
        self.assertEqual(py(self.kg('$[]')), [])
        self.assertEqual(py(self.kg('$["a" 1]')), ["a", "1"])

    def test_format_float_width(self):
        self.assert_klong_error('1.5$3')
        self.assert_klong_error('1.0$"12"')

    def test_format_float_width_with_real(self):
        self.assertEqual(self.kg('5.3$123.45'), "  123.450")

    def test_format_negative_width(self):
        self.assertEqual(self.kg('(-5)$"ab"'), "   ab")

    # enumerate
    def test_enumerate_dict(self):
        self.assert_klong_error('!:{[1 2] [3 4]}')
        self.assertEqual(py(self.kg('!3')), [0, 1, 2])

    # was an expected failure until ticket 063
    def test_nested_dict_literal(self):
        r = self.kg(':{[1 :{[2 3]}]}')
        self.assertIsInstance(r[1], dict)


if __name__ == "__main__":
    unittest.main()
