import unittest
import numpy

from klongpy.backend import is_supported_type, is_jagged_array, np, use_torch, TorchUnsupportedDtypeError


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

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_object_dtype_raises_error(self):
        """Test that object dtype raises TorchUnsupportedDtypeError."""
        with self.assertRaises(TorchUnsupportedDtypeError):
            np.asarray([1, "string", 3], dtype=object)

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_numeric_array_works(self):
        """Test that numeric arrays work in torch mode."""
        arr = np.asarray([1, 2, 3])
        self.assertEqual(len(arr), 3)

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_isarray_with_tensor(self):
        """Test that isarray detects torch tensors."""
        import torch
        tensor = torch.tensor([1, 2, 3])
        self.assertTrue(np.isarray(tensor))

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_device_selection(self):
        """Test that a device is selected."""
        self.assertIsNotNone(np.device)

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_zeros(self):
        """Test zeros creation."""
        arr = np.zeros((3, 3))
        self.assertEqual(arr.shape, (3, 3))

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_ones(self):
        """Test ones creation."""
        arr = np.ones((2, 2))
        self.assertEqual(arr.shape, (2, 2))

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_arange(self):
        """Test arange creation."""
        arr = np.arange(5)
        self.assertEqual(len(arr), 5)

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_concatenate(self):
        """Test concatenate operation."""
        a = np.asarray([1, 2, 3])
        b = np.asarray([4, 5, 6])
        result = np.concatenate([a, b])
        self.assertEqual(len(result), 6)

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_hstack(self):
        """Test hstack operation."""
        a = np.asarray([1, 2, 3])
        b = np.asarray([4, 5, 6])
        result = np.hstack([a, b])
        self.assertEqual(len(result), 6)

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_vstack(self):
        """Test vstack operation."""
        a = np.asarray([[1, 2, 3]])
        b = np.asarray([[4, 5, 6]])
        result = np.vstack([a, b])
        self.assertEqual(result.shape, (2, 3))

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_stack(self):
        """Test stack operation."""
        a = np.asarray([1, 2, 3])
        b = np.asarray([4, 5, 6])
        result = np.stack([a, b])
        self.assertEqual(result.shape, (2, 3))

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_copy(self):
        """Test copy operation."""
        a = np.asarray([1, 2, 3])
        b = np.copy(a)
        self.assertEqual(len(b), 3)

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_isclose(self):
        """Test isclose operation."""
        result = np.isclose(1.0, 1.0 + 1e-9)
        self.assertTrue(result)

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_array_equal(self):
        """Test array_equal operation."""
        a = np.asarray([1, 2, 3])
        b = np.asarray([1, 2, 3])
        self.assertTrue(np.array_equal(a, b))

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_add(self):
        """Test add operation."""
        a = np.asarray([1, 2, 3])
        b = np.asarray([4, 5, 6])
        result = np.add(a, b)
        self.assertEqual(len(result), 3)

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_subtract(self):
        """Test subtract operation."""
        a = np.asarray([4, 5, 6])
        b = np.asarray([1, 2, 3])
        result = np.subtract(a, b)
        self.assertEqual(len(result), 3)

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_multiply(self):
        """Test multiply operation."""
        a = np.asarray([1, 2, 3])
        b = np.asarray([4, 5, 6])
        result = np.multiply(a, b)
        self.assertEqual(len(result), 3)

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_divide(self):
        """Test divide operation."""
        a = np.asarray([4.0, 6.0, 8.0])
        b = np.asarray([2.0, 2.0, 2.0])
        result = np.divide(a, b)
        self.assertEqual(len(result), 3)

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_inf_property(self):
        """Test inf property."""
        self.assertEqual(np.inf, float('inf'))

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_seterr(self):
        """Test seterr (should be no-op)."""
        np.seterr(divide='ignore')  # Should not raise

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_ndarray_property(self):
        """Test ndarray property returns Tensor class."""
        import torch
        self.assertEqual(np.ndarray, torch.Tensor)

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_integer_property(self):
        """Test integer property."""
        self.assertEqual(np.integer, numpy.integer)

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_floating_property(self):
        """Test floating property."""
        self.assertEqual(np.floating, numpy.floating)

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_asarray_with_object_dtype_kind(self):
        """Test asarray rejects dtype with kind 'O'."""
        class FakeDtype:
            kind = 'O'
        with self.assertRaises(TorchUnsupportedDtypeError):
            np.asarray([1, 2, 3], dtype=FakeDtype())

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_asarray_with_numpy_object_array(self):
        """Test asarray rejects numpy object arrays."""
        obj_arr = numpy.array([1, "a", 2], dtype=object)
        with self.assertRaises(TorchUnsupportedDtypeError):
            np.asarray(obj_arr)

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_array_method(self):
        """Test array method."""
        arr = np.array([1, 2, 3])
        self.assertEqual(len(arr), 3)

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_take(self):
        """Test take operation."""
        a = np.asarray([1, 2, 3, 4, 5])
        indices = np.asarray([0, 2, 4])
        result = np.take(a, indices)
        self.assertEqual(len(result), 3)

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_take_with_axis(self):
        """Test take operation with axis."""
        a = np.asarray([[1, 2], [3, 4], [5, 6]])
        indices = np.asarray([0, 2])
        result = np.take(a, indices, axis=0)
        self.assertEqual(result.shape, (2, 2))

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_getattr_torch_function(self):
        """Test __getattr__ delegates to torch."""
        # sin is a torch function
        result = np.sin(np.asarray([0.0]))
        self.assertEqual(len(result), 1)

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_getattr_numpy_fallback(self):
        """Test __getattr__ falls back to numpy for missing torch attrs."""
        # VisibleDeprecationWarning is a numpy attribute
        self.assertEqual(np.VisibleDeprecationWarning, numpy.VisibleDeprecationWarning)

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_getattr_missing(self):
        """Test __getattr__ raises AttributeError for missing attributes."""
        with self.assertRaises(AttributeError):
            _ = np.nonexistent_attribute_xyz123


class TestTorchUnsupportedDtypeError(unittest.TestCase):
    def test_error_is_exception(self):
        """Test that TorchUnsupportedDtypeError is a proper exception."""
        self.assertTrue(issubclass(TorchUnsupportedDtypeError, Exception))

    def test_error_message(self):
        """Test that error message is preserved."""
        msg = "Test error message"
        err = TorchUnsupportedDtypeError(msg)
        self.assertEqual(str(err), msg)

    def test_error_can_be_raised(self):
        """Test that the error can be raised and caught."""
        with self.assertRaises(TorchUnsupportedDtypeError):
            raise TorchUnsupportedDtypeError("test")

    def test_error_inheritance(self):
        """Test that error can be caught as generic Exception."""
        with self.assertRaises(Exception):
            raise TorchUnsupportedDtypeError("test")


class TestCoreWithTorch(unittest.TestCase):
    """Tests for core.py torch-related code."""

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_kg_asarray_string_returns_char_array(self):
        """Test that kg_asarray converts strings to char arrays in torch mode."""
        from klongpy.core import kg_asarray
        result = kg_asarray("hello")
        # Strings are converted to numpy char arrays for Klong compatibility
        self.assertEqual(len(result), 5)
        self.assertEqual(list(result), ['h', 'e', 'l', 'l', 'o'])

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_str_to_chr_arr_fails(self):
        """Test that str_to_chr_arr fails in torch mode."""
        from klongpy.core import str_to_chr_arr
        with self.assertRaises(TorchUnsupportedDtypeError):
            str_to_chr_arr("hello")

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_kg_asarray_jagged_returns_object_array(self):
        """Test that kg_asarray falls back to object array for jagged arrays."""
        import numpy as np
        from klongpy.core import kg_asarray
        result = kg_asarray([[1, 2], [3]])
        # Jagged arrays fall back to numpy object arrays
        self.assertEqual(result.dtype, object)
        self.assertEqual(len(result), 2)

    @unittest.skipUnless(use_torch, "PyTorch backend not enabled")
    def test_kg_asarray_numeric_works(self):
        """Test that kg_asarray works for numeric arrays in torch mode."""
        from klongpy.core import kg_asarray
        result = kg_asarray([1, 2, 3])
        self.assertEqual(len(result), 3)

    @unittest.skipUnless(not use_torch, "NumPy backend required")
    def test_kg_asarray_string_works_numpy(self):
        """Test that kg_asarray works for strings in numpy mode."""
        from klongpy.core import kg_asarray
        result = kg_asarray("hi")
        self.assertEqual(len(result), 2)

    @unittest.skipUnless(not use_torch, "NumPy backend required")
    def test_kg_asarray_jagged_works_numpy(self):
        """Test that kg_asarray works for jagged arrays in numpy mode."""
        from klongpy.core import kg_asarray
        result = kg_asarray([[1, 2], [3]])
        self.assertEqual(len(result), 2)

    @unittest.skipUnless(not use_torch, "NumPy backend required")
    def test_kg_asarray_object_dtype_numpy(self):
        """Test that kg_asarray handles object dtype in numpy mode."""
        from klongpy.core import kg_asarray
        result = kg_asarray([1, "a", 2])
        self.assertEqual(len(result), 3)


if __name__ == '__main__':
    unittest.main()
