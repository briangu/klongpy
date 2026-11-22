import os
import unittest
from unittest.mock import patch

import numpy as np

from klongpy import KlongInterpreter
from klongpy.backends import get_backend
from klongpy.errors import KlongShapeError, KlongTypeError
from klongpy.static.fixed import FixedArray, Spec
from tests.custom_backend import TestArray


_TEST_BACKEND = os.environ.get('_KLONGPY_TEST_BACKEND', 'numpy')


@unittest.skipUnless(_TEST_BACKEND == 'test_backend', "Requires test_backend")
class TestBackendAbstraction(unittest.TestCase):
    def test_backend_registration(self):
        backend = get_backend('test_backend')
        self.assertEqual(backend.name, 'test_backend')

    def test_np_wrapper_returns_test_array(self):
        backend = get_backend('test_backend')
        arr = backend.np.asarray([1, 2, 3])
        self.assertIsInstance(arr, TestArray)

    def test_klong_list_literal_uses_backend_array(self):
        klong = KlongInterpreter(backend='test_backend')
        result = klong('[1 2 3]')
        self.assertIsInstance(result, TestArray)

    def test_klong_dyad_add_preserves_backend_array(self):
        klong = KlongInterpreter(backend='test_backend')
        result = klong('[1 2 3]+[4 5 6]')
        self.assertIsInstance(result, TestArray)
        np.testing.assert_array_equal(result, [5, 7, 9])

    def test_klong_float_arithmetic_uses_provider_operations(self):
        klong = KlongInterpreter(backend='test_backend')
        for operation, expression, expected in (
            ('add', '[1.5 2.5]+2', [3.5, 4.5]),
            ('subtract', '[1.5 2.5]-[0.5 0.5]', [1, 2]),
            ('multiply', '3*[1.5 2.5]', [4.5, 7.5]),
        ):
            with self.subTest(operation=operation):
                with patch.object(klong.backend.np, operation, wraps=getattr(klong.backend.np, operation)) as dyad:
                    result = klong(expression)
                dyad.assert_called_once()
                self.assertIsInstance(result, TestArray)
                np.testing.assert_array_equal(result, expected)

    def test_klong_dyad_reports_shape_error(self):
        klong = KlongInterpreter(backend='test_backend')
        with self.assertRaises(KlongShapeError):
            klong('[1.5 2.5]+[1.0]')

    def test_caught_backend_error_does_not_escape_in_place_of_a_native_error(self):
        klong = KlongInterpreter(backend='test_backend')
        caught_shape = '(.try({[1.0 2.0]+[3.0]};[])@1)@"kind"'
        invalid_try = '.try(1;,2)'
        with self.assertRaises(KlongTypeError):
            klong(f'{caught_shape};{invalid_try}')
        self.assertEqual(klong(caught_shape), 'shape')
        with self.assertRaises(KlongTypeError):
            klong(invalid_try)
        self.assertEqual(klong(caught_shape), 'shape')
        with self.assertRaises(KlongTypeError):
            KlongInterpreter(backend='test_backend')(invalid_try)

    def test_typed_extern_respects_provider_gradient_policy(self):
        klong = KlongInterpreter(backend='test_backend')
        seen = []
        klong.externs["total"] = lambda a: seen.append(a) or float(a.sum())
        klong('total::.extern("total";"float[3] -> float"); values::[1.0 2.0 3.0]')
        tracked = klong["values"]
        self.assertIsInstance(tracked, TestArray)
        with patch.object(klong.backend, "has_gradient", side_effect=lambda a: a is tracked):
            with self.assertRaisesRegex(KlongTypeError, "requires gradients"):
                klong("total(values)")
        self.assertEqual(seen, [])
        with patch.object(klong.backend, "to_numpy", side_effect=lambda a: a):
            self.assertEqual(klong("total(values)"), 6.0)
        self.assertIs(type(seen[0]), np.ndarray)

    def test_provider_matrix_destructuring_preserves_rows(self):
        klong = KlongInterpreter(backend='test_backend')
        matrix = TestArray([[1, 2], [3, 4]], dtype=np.float32)
        klong["values"] = matrix
        klong("[a b]::values")
        self.assertIs(klong["values"], matrix)
        for name, expected in (("a", [1, 2]), ("b", [3, 4])):
            row = klong[name]
            self.assertIsInstance(row, TestArray)
            self.assertEqual(row.dtype, np.dtype("float32"))
            self.assertTrue(np.shares_memory(row, matrix))
            np.testing.assert_array_equal(row, expected)

    def test_nonprovider_and_temporal_subclasses_stay_opaque(self):
        klong = KlongInterpreter(backend='test_backend')
        samples = (
            FixedArray([1, 2], Spec(16, 8)),
            np.ma.array([1, 2], mask=[False, True]),
            TestArray(["2000-01-01", "NaT"], dtype="datetime64[D]"),
        )
        for array in samples:
            with self.subTest(dtype=array.dtype, array_type=type(array)):
                klong["values"] = array
                self.assertIs(klong["values"], array)
                with self.assertRaises(KlongTypeError):
                    klong("[a b]::values")

    def test_mixed_list_keeps_foreign_arrays(self):
        klong = KlongInterpreter(backend='test_backend')
        samples = [
            FixedArray([1, 2], Spec(16, 8)),
            np.ma.array([1, 2], mask=[False, True]),
            np.array(["2000-01-01", "NaT"], dtype="datetime64[ns]"),
            np.array([1, 2], dtype="uint8"),
        ]
        klong["values"] = samples
        mixed = klong["values"]
        self.assertEqual(mixed.shape, (len(samples),))
        for actual, expected in zip(mixed, samples):
            self.assertIs(actual, expected)
