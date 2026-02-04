import os
import unittest

from klongpy import KlongInterpreter
from klongpy.backends import get_backend
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
