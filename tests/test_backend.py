import importlib
import importlib.util
import unittest

import numpy

from klongpy.backend import is_supported_type, is_jagged_array, np, UnsupportedDtypeError
from klongpy.backends import get_backend

_TORCH_SPEC = importlib.util.find_spec("torch")
torch = importlib.import_module("torch") if _TORCH_SPEC else None
TORCH_AVAILABLE = torch is not None

# numpy 2.x moved VisibleDeprecationWarning to numpy.exceptions
from numpy.exceptions import VisibleDeprecationWarning as NumpyVisibleDeprecationWarning


class TestIsSupportedType(unittest.TestCase):
    def test_string_not_supported(self):
        self.assertFalse(is_supported_type("hello"))
        self.assertFalse(is_supported_type(""))

    def test_jagged_array_not_supported(self):
        self.assertFalse(is_supported_type([[1, 2], [3]]))
        self.assertFalse(is_supported_type([[1], [2, 3, 4]]))

    def test_regular_list_supported(self):
        # Flat lists are supported
        self.assertTrue(is_supported_type([[1, 2], [3, 4]]))
        # Uniform nested lists are supported
        self.assertTrue(is_supported_type([[1], [2]]))

    def test_numpy_array_supported(self):
        self.assertTrue(is_supported_type(numpy.array([1, 2, 3])))
        self.assertTrue(is_supported_type(numpy.array([[1, 2], [3, 4]])))

    def test_numbers_supported(self):
        self.assertTrue(is_supported_type(42))
        self.assertTrue(is_supported_type(3.14))

    def test_empty_list_supported(self):
        self.assertTrue(is_supported_type([]))

    def test_flat_list_supported(self):
        self.assertTrue(is_supported_type([1, 2, 3]))

    def test_none_supported(self):
        self.assertTrue(is_supported_type(None))

    def test_dict_supported(self):
        self.assertTrue(is_supported_type({"a": 1}))


class TestIsJaggedArray(unittest.TestCase):
    def test_jagged_list(self):
        self.assertTrue(is_jagged_array([[1, 2], [3]]))
        self.assertTrue(is_jagged_array([[1], [2, 3, 4]]))
        self.assertTrue(is_jagged_array([[], [1]]))

    def test_regular_list_not_jagged(self):
        self.assertFalse(is_jagged_array([[1, 2], [3, 4]]))
        self.assertFalse(is_jagged_array([[1], [2]]))
        self.assertFalse(is_jagged_array([[], []]))

    def test_non_list_not_jagged(self):
        self.assertFalse(is_jagged_array("not a list"))
        self.assertFalse(is_jagged_array(42))
        self.assertFalse(is_jagged_array(numpy.array([1, 2, 3])))

    def test_empty_list_not_jagged(self):
        self.assertFalse(is_jagged_array([]))

    def test_single_element_list(self):
        self.assertFalse(is_jagged_array([[1, 2, 3]]))

    def test_flat_list_not_jagged(self):
        self.assertFalse(is_jagged_array([1, 2, 3]))

    def test_nested_empty_lists(self):
        self.assertFalse(is_jagged_array([[], [], []]))

    def test_deeply_nested_same_length(self):
        self.assertFalse(is_jagged_array([[1, 2], [3, 4], [5, 6]]))


class TestNumpyBackend(unittest.TestCase):
    def test_isarray_with_ndarray(self):
        arr = numpy.array([1, 2, 3])
        self.assertTrue(np.isarray(arr))

    def test_isarray_with_list(self):
        self.assertFalse(np.isarray([1, 2, 3]))

    def test_isarray_with_scalar(self):
        self.assertFalse(np.isarray(42))
        self.assertFalse(np.isarray(3.14))

    def test_isarray_with_string(self):
        self.assertFalse(np.isarray("hello"))

    def test_isarray_with_none(self):
        self.assertFalse(np.isarray(None))

    def test_isarray_with_dict(self):
        self.assertFalse(np.isarray({"a": 1}))


class TestTorchBackend(unittest.TestCase):
    """Tests specific to PyTorch backend behavior."""

    @classmethod
    def setUpClass(cls):
        """Get torch backend for testing."""
        if TORCH_AVAILABLE:
            cls.torch_backend = get_backend('torch')
            cls.np = cls.torch_backend.np
        else:
            cls.torch_backend = None
            cls.np = None

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_object_dtype_raises_error(self):
        """Test that object dtype raises UnsupportedDtypeError."""
        with self.assertRaises(UnsupportedDtypeError):
            self.np.asarray([1, "string", 3], dtype=object)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_numeric_array_works(self):
        """Test that numeric arrays work in torch mode."""
        arr = self.np.asarray([1, 2, 3])
        self.assertEqual(len(arr), 3)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_isarray_with_tensor(self):
        """Test that isarray detects torch tensors."""
        tensor = torch.tensor([1, 2, 3])
        self.assertTrue(self.np.isarray(tensor))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_device_selection(self):
        """Test that a device is selected."""
        self.assertIsNotNone(self.np.device)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_zeros(self):
        """Test zeros creation."""
        arr = self.np.zeros((3, 3))
        self.assertEqual(arr.shape, (3, 3))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_ones(self):
        """Test ones creation."""
        arr = self.np.ones((2, 2))
        self.assertEqual(arr.shape, (2, 2))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_arange(self):
        """Test arange creation."""
        arr = self.np.arange(5)
        self.assertEqual(len(arr), 5)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_concatenate(self):
        """Test concatenate operation."""
        a = self.np.asarray([1, 2, 3])
        b = self.np.asarray([4, 5, 6])
        result = self.np.concatenate([a, b])
        self.assertEqual(len(result), 6)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_hstack(self):
        """Test hstack operation."""
        a = self.np.asarray([1, 2, 3])
        b = self.np.asarray([4, 5, 6])
        result = self.np.hstack([a, b])
        self.assertEqual(len(result), 6)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_vstack(self):
        """Test vstack operation."""
        a = self.np.asarray([[1, 2, 3]])
        b = self.np.asarray([[4, 5, 6]])
        result = self.np.vstack([a, b])
        self.assertEqual(result.shape, (2, 3))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_stack(self):
        """Test stack operation."""
        a = self.np.asarray([1, 2, 3])
        b = self.np.asarray([4, 5, 6])
        result = self.np.stack([a, b])
        self.assertEqual(result.shape, (2, 3))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_copy(self):
        """Test copy operation."""
        a = self.np.asarray([1, 2, 3])
        b = self.np.copy(a)
        self.assertEqual(len(b), 3)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_isclose(self):
        """Test isclose operation."""
        result = self.np.isclose(1.0, 1.0 + 1e-9)
        self.assertTrue(result)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_array_equal(self):
        """Test array_equal operation."""
        a = self.np.asarray([1, 2, 3])
        b = self.np.asarray([1, 2, 3])
        self.assertTrue(self.np.array_equal(a, b))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_add(self):
        """Test add operation."""
        a = self.np.asarray([1, 2, 3])
        b = self.np.asarray([4, 5, 6])
        result = self.np.add(a, b)
        self.assertEqual(len(result), 3)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_subtract(self):
        """Test subtract operation."""
        a = self.np.asarray([4, 5, 6])
        b = self.np.asarray([1, 2, 3])
        result = self.np.subtract(a, b)
        self.assertEqual(len(result), 3)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_multiply(self):
        """Test multiply operation."""
        a = self.np.asarray([1, 2, 3])
        b = self.np.asarray([4, 5, 6])
        result = self.np.multiply(a, b)
        self.assertEqual(len(result), 3)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_divide(self):
        """Test divide operation."""
        a = self.np.asarray([4.0, 6.0, 8.0])
        b = self.np.asarray([2.0, 2.0, 2.0])
        result = self.np.divide(a, b)
        self.assertEqual(len(result), 3)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_inf_property(self):
        """Test inf property."""
        self.assertEqual(self.np.inf, float('inf'))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_seterr(self):
        """Test seterr (should be no-op)."""
        self.np.seterr(divide='ignore')  # Should not raise

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_ndarray_property(self):
        """Test ndarray property returns Tensor class."""
        self.assertEqual(self.np.ndarray, torch.Tensor)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_integer_property(self):
        """Test integer property."""
        self.assertEqual(self.np.integer, numpy.integer)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_floating_property(self):
        """Test floating property."""
        self.assertEqual(self.np.floating, numpy.floating)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_asarray_with_object_dtype_kind(self):
        """Test asarray rejects dtype with kind 'O'."""
        class FakeDtype:
            kind = 'O'
        with self.assertRaises(UnsupportedDtypeError):
            self.np.asarray([1, 2, 3], dtype=FakeDtype())

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_asarray_with_numpy_object_array(self):
        """Test asarray rejects numpy object arrays."""
        obj_arr = numpy.array([1, "a", 2], dtype=object)
        with self.assertRaises(UnsupportedDtypeError):
            self.np.asarray(obj_arr)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_array_method(self):
        """Test array method."""
        arr = self.np.array([1, 2, 3])
        self.assertEqual(len(arr), 3)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_take(self):
        """Test take operation."""
        a = self.np.asarray([1, 2, 3, 4, 5])
        indices = self.np.asarray([0, 2, 4])
        result = self.np.take(a, indices)
        self.assertEqual(len(result), 3)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_take_with_axis(self):
        """Test take operation with axis."""
        a = self.np.asarray([[1, 2], [3, 4], [5, 6]])
        indices = self.np.asarray([0, 2])
        result = self.np.take(a, indices, axis=0)
        self.assertEqual(result.shape, (2, 2))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_getattr_torch_function(self):
        """Test __getattr__ delegates to torch."""
        # sin is a torch function
        result = self.np.sin(self.np.asarray([0.0]))
        self.assertEqual(len(result), 1)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_getattr_numpy_fallback(self):
        """Test __getattr__ falls back to numpy for missing torch attrs."""
        # VisibleDeprecationWarning is a numpy attribute (moved to numpy.exceptions in 2.x)
        self.assertEqual(self.np.VisibleDeprecationWarning, NumpyVisibleDeprecationWarning)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_getattr_missing(self):
        """Test __getattr__ raises AttributeError for missing attributes."""
        with self.assertRaises(AttributeError):
            _ = self.np.nonexistent_attribute_xyz123


class TestUnsupportedDtypeError(unittest.TestCase):
    def test_error_is_exception(self):
        """Test that UnsupportedDtypeError is a proper exception."""
        self.assertTrue(issubclass(UnsupportedDtypeError, Exception))

    def test_error_message(self):
        """Test that error message is preserved."""
        msg = "Test error message"
        err = UnsupportedDtypeError(msg)
        self.assertEqual(str(err), msg)

    def test_error_can_be_raised(self):
        """Test that the error can be raised and caught."""
        with self.assertRaises(UnsupportedDtypeError):
            raise UnsupportedDtypeError("test")

    def test_error_inheritance(self):
        """Test that error can be caught as generic Exception."""
        with self.assertRaises(Exception):
            raise UnsupportedDtypeError("test")


class TestCoreWithTorch(unittest.TestCase):
    """Tests for core.py torch-related code."""

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_kg_asarray_string_returns_char_array(self):
        """Test that kg_asarray converts strings to char arrays in torch mode."""
        backend = get_backend('torch')
        result = backend.kg_asarray("hello")
        # Strings are converted to numpy char arrays for Klong compatibility
        self.assertEqual(len(result), 5)
        self.assertEqual(list(result), ['h', 'e', 'l', 'l', 'o'])

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_str_to_chr_arr_fails(self):
        """Test that str_to_chr_arr fails in torch mode."""
        backend = get_backend('torch')
        with self.assertRaises(UnsupportedDtypeError):
            backend.str_to_chr_arr("hello")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_kg_asarray_jagged_returns_object_array(self):
        """Test that kg_asarray falls back to object array for jagged arrays."""
        backend = get_backend('torch')
        result = backend.kg_asarray([[1, 2], [3]])
        # Jagged arrays fall back to numpy object arrays
        self.assertEqual(result.dtype, object)
        self.assertEqual(len(result), 2)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_kg_asarray_numeric_works(self):
        """Test that kg_asarray works for numeric arrays in torch mode."""
        backend = get_backend('torch')
        result = backend.kg_asarray([1, 2, 3])
        self.assertEqual(len(result), 3)

    def test_kg_asarray_string_works_numpy(self):
        """Test that kg_asarray works for strings in numpy mode."""
        backend = get_backend('numpy')
        result = backend.kg_asarray("hi")
        self.assertEqual(len(result), 2)

    def test_kg_asarray_jagged_works_numpy(self):
        """Test that kg_asarray works for jagged arrays in numpy mode."""
        backend = get_backend('numpy')
        result = backend.kg_asarray([[1, 2], [3]])
        self.assertEqual(len(result), 2)

    def test_kg_asarray_object_dtype_numpy(self):
        """Test that kg_asarray handles object dtype in numpy mode."""
        backend = get_backend('numpy')
        result = backend.kg_asarray([1, "a", 2])
        self.assertEqual(len(result), 3)


if __name__ == '__main__':
    unittest.main()
