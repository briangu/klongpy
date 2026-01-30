"""
Tests for PyTorch backend functionality.

These tests require PyTorch to be installed and USE_TORCH=1 to be set.
Run with: USE_TORCH=1 python -m unittest tests.test_torch_backend

This file tests the torch-specific code paths that cannot be tested
in the main test files due to the backend being selected at import time.
"""
import unittest
import os
import sys

# Check if we're running in torch mode
USE_TORCH = os.environ.get('USE_TORCH') == '1'

if USE_TORCH:
    try:
        import torch
        import numpy as np
        from klongpy import KlongInterpreter
        from klongpy.backend import np as backend_np, use_torch, TorchUnsupportedDtypeError
        from klongpy.autograd import torch_autograd, autograd_of_fn
        from klongpy.core import kg_asarray, str_to_chr_arr, KGLambda, KGSym
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = False


@unittest.skipUnless(USE_TORCH and TORCH_AVAILABLE, "Requires USE_TORCH=1 and torch installed")
class TestTorchAutogradFunction(unittest.TestCase):
    """Tests for the torch_autograd function."""

    def test_with_tensor_input(self):
        """Test torch_autograd with torch.Tensor input."""
        x = torch.tensor([1.0, 2.0, 3.0])
        result = torch_autograd(lambda t: (t**2).sum(), x)
        expected = torch.tensor([2.0, 4.0, 6.0])
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_with_numpy_input(self):
        """Test torch_autograd with numpy array input."""
        x = np.array([1.0, 2.0, 3.0])
        result = torch_autograd(lambda t: (t**2).sum(), x)
        expected = np.array([2.0, 4.0, 6.0])
        self.assertTrue(np.allclose(result.cpu().numpy(), expected, atol=1e-5))

    def test_with_list_input(self):
        """Test torch_autograd with list input."""
        x = [1.0, 2.0, 3.0]
        result = torch_autograd(lambda t: (t**2).sum(), x)
        expected = np.array([2.0, 4.0, 6.0])
        self.assertTrue(np.allclose(result.cpu().numpy(), expected, atol=1e-5))

    def test_non_scalar_output_raises(self):
        """Test that non-scalar output raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError) as ctx:
            torch_autograd(lambda t: t**2, x)  # Returns vector, not scalar
        self.assertIn("scalar", str(ctx.exception))

    def test_scalar_input(self):
        """Test torch_autograd with scalar input."""
        x = 3.0
        result = torch_autograd(lambda t: t**2, x)
        self.assertTrue(np.isclose(result.item(), 6.0, atol=1e-5))

    def test_complex_function(self):
        """Test with more complex function."""
        x = torch.tensor([1.0, 2.0])
        # f(x) = x1^2 + x2^2 + x1*x2
        # df/dx1 = 2*x1 + x2 = 2*1 + 2 = 4
        # df/dx2 = 2*x2 + x1 = 2*2 + 1 = 5
        result = torch_autograd(lambda t: t[0]**2 + t[1]**2 + t[0]*t[1], x)
        expected = torch.tensor([4.0, 5.0])
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))


@unittest.skipUnless(USE_TORCH and TORCH_AVAILABLE, "Requires USE_TORCH=1 and torch installed")
class TestTorchBackendOperations(unittest.TestCase):
    """Tests for TorchBackend class operations."""

    def test_asarray_numeric(self):
        """Test asarray with numeric data."""
        arr = backend_np.asarray([1, 2, 3])
        self.assertEqual(len(arr), 3)
        self.assertIsInstance(arr, torch.Tensor)

    def test_asarray_rejects_object_dtype(self):
        """Test asarray rejects object dtype."""
        with self.assertRaises(TorchUnsupportedDtypeError):
            backend_np.asarray([1, 2, 3], dtype=object)

    def test_asarray_rejects_numpy_object_array(self):
        """Test asarray rejects numpy object arrays."""
        obj_arr = np.array([1, "a", 2], dtype=object)
        with self.assertRaises(TorchUnsupportedDtypeError):
            backend_np.asarray(obj_arr)

    def test_asarray_with_tensor(self):
        """Test asarray with existing tensor on same device returns it."""
        from klongpy.backend import get_default_backend
        # Create tensor on the same device as backend
        backend = get_default_backend()
        device = backend.device
        t = torch.tensor([1, 2, 3], device=device)
        result = backend_np.asarray(t)
        self.assertIs(result, t)

    def test_zeros(self):
        """Test zeros creation."""
        arr = backend_np.zeros((3, 3))
        self.assertEqual(arr.shape, (3, 3))
        self.assertTrue(torch.all(arr == 0))

    def test_ones(self):
        """Test ones creation."""
        arr = backend_np.ones((2, 2))
        self.assertEqual(arr.shape, (2, 2))
        self.assertTrue(torch.all(arr == 1))

    def test_arange(self):
        """Test arange creation."""
        arr = backend_np.arange(5)
        self.assertEqual(len(arr), 5)

    def test_concatenate(self):
        """Test concatenate operation."""
        a = backend_np.asarray([1, 2, 3])
        b = backend_np.asarray([4, 5, 6])
        result = backend_np.concatenate([a, b])
        self.assertEqual(len(result), 6)

    def test_hstack(self):
        """Test hstack operation."""
        a = backend_np.asarray([1, 2, 3])
        b = backend_np.asarray([4, 5, 6])
        result = backend_np.hstack([a, b])
        self.assertEqual(len(result), 6)

    def test_vstack(self):
        """Test vstack operation."""
        a = backend_np.asarray([[1, 2, 3]])
        b = backend_np.asarray([[4, 5, 6]])
        result = backend_np.vstack([a, b])
        self.assertEqual(result.shape, (2, 3))

    def test_stack(self):
        """Test stack operation."""
        a = backend_np.asarray([1, 2, 3])
        b = backend_np.asarray([4, 5, 6])
        result = backend_np.stack([a, b])
        self.assertEqual(result.shape, (2, 3))

    def test_copy(self):
        """Test copy operation."""
        a = backend_np.asarray([1, 2, 3])
        b = backend_np.copy(a)
        self.assertEqual(len(b), 3)
        # Verify it's a copy, not the same tensor
        b[0] = 999
        self.assertNotEqual(a[0].item(), 999)

    def test_isclose(self):
        """Test isclose operation."""
        a = backend_np.asarray(1.0)
        b = backend_np.asarray(1.0 + 1e-9)
        result = backend_np.isclose(a, b)
        self.assertTrue(result)

    def test_array_equal(self):
        """Test array_equal operation."""
        a = backend_np.asarray([1, 2, 3])
        b = backend_np.asarray([1, 2, 3])
        self.assertTrue(backend_np.array_equal(a, b))

    def test_math_operations(self):
        """Test basic math operations."""
        a = backend_np.asarray([1.0, 2.0, 3.0])
        b = backend_np.asarray([4.0, 5.0, 6.0])

        add_result = backend_np.add(a, b)
        expected = torch.tensor([5.0, 7.0, 9.0], device=add_result.device)
        self.assertTrue(torch.allclose(add_result, expected))

        sub_result = backend_np.subtract(b, a)
        expected = torch.tensor([3.0, 3.0, 3.0], device=sub_result.device)
        self.assertTrue(torch.allclose(sub_result, expected))

        mul_result = backend_np.multiply(a, b)
        expected = torch.tensor([4.0, 10.0, 18.0], device=mul_result.device)
        self.assertTrue(torch.allclose(mul_result, expected))

        div_result = backend_np.divide(b, a)
        expected = torch.tensor([4.0, 2.5, 2.0], device=div_result.device)
        self.assertTrue(torch.allclose(div_result, expected))

    def test_take(self):
        """Test take operation."""
        a = backend_np.asarray([1, 2, 3, 4, 5])
        indices = backend_np.asarray([0, 2, 4])
        result = backend_np.take(a, indices)
        self.assertEqual(len(result), 3)

    def test_take_with_axis(self):
        """Test take operation with axis."""
        a = backend_np.asarray([[1, 2], [3, 4], [5, 6]])
        indices = backend_np.asarray([0, 2])
        result = backend_np.take(a, indices, axis=0)
        self.assertEqual(result.shape, (2, 2))

    def test_isarray(self):
        """Test isarray detects tensors and numpy arrays."""
        tensor = torch.tensor([1, 2, 3])
        numpy_arr = np.array([1, 2, 3])
        self.assertTrue(backend_np.isarray(tensor))
        self.assertTrue(backend_np.isarray(numpy_arr))
        self.assertFalse(backend_np.isarray([1, 2, 3]))

    def test_getattr_delegates_to_torch(self):
        """Test that __getattr__ delegates to torch."""
        result = backend_np.sin(backend_np.asarray([0.0]))
        self.assertEqual(len(result), 1)

    def test_getattr_falls_back_to_numpy(self):
        """Test that __getattr__ falls back to numpy."""
        self.assertEqual(backend_np.VisibleDeprecationWarning, np.VisibleDeprecationWarning)

    def test_getattr_raises_for_missing(self):
        """Test that __getattr__ raises AttributeError for missing attrs."""
        with self.assertRaises(AttributeError):
            _ = backend_np.nonexistent_attribute_xyz123

    def test_properties(self):
        """Test backend properties."""
        self.assertEqual(backend_np.inf, float('inf'))
        self.assertEqual(backend_np.ndarray, torch.Tensor)
        self.assertEqual(backend_np.integer, np.integer)
        self.assertEqual(backend_np.floating, np.floating)

    def test_seterr_noop(self):
        """Test seterr is a no-op."""
        backend_np.seterr(divide='ignore')  # Should not raise

    def test_device_selected(self):
        """Test that a device is selected."""
        self.assertIsNotNone(backend_np.device)


@unittest.skipUnless(USE_TORCH and TORCH_AVAILABLE, "Requires USE_TORCH=1 and torch installed")
class TestTorchCoreIntegration(unittest.TestCase):
    """Tests for core.py torch integration."""

    def test_kg_asarray_string_returns_char_array(self):
        """Test that kg_asarray converts strings to char arrays."""
        result = kg_asarray("hello")
        # Strings are converted to numpy char arrays for Klong compatibility
        self.assertEqual(len(result), 5)
        self.assertEqual(list(result), ['h', 'e', 'l', 'l', 'o'])

    def test_str_to_chr_arr_fails(self):
        """Test that str_to_chr_arr fails."""
        with self.assertRaises(TorchUnsupportedDtypeError):
            str_to_chr_arr("hello")

    def test_kg_asarray_jagged_returns_object_array(self):
        """Test that kg_asarray falls back to numpy object array for jagged arrays."""
        import numpy as np
        result = kg_asarray([[1, 2], [3]])
        # Jagged arrays fall back to numpy object arrays
        self.assertEqual(result.dtype, object)
        self.assertEqual(len(result), 2)

    def test_kg_asarray_numeric_works(self):
        """Test that kg_asarray works for numeric data."""
        result = kg_asarray([1, 2, 3])
        self.assertEqual(len(result), 3)


@unittest.skipUnless(USE_TORCH and TORCH_AVAILABLE, "Requires USE_TORCH=1 and torch installed")
class TestTorchAutogradOperator(unittest.TestCase):
    """Tests for :> operator with torch backend."""

    def test_scalar_autograd(self):
        """Test :> operator with scalar input."""
        klong = KlongInterpreter()
        r = klong('{x*x}:>3.0')
        # Should use torch autograd for exact gradient
        self.assertTrue(np.isclose(float(r), 6.0, atol=1e-5))

    def test_autograd_of_fn_with_kglambda(self):
        """Test autograd_of_fn with KGLambda."""
        klong = KlongInterpreter()
        fn = KGLambda(lambda x: torch.sum(x**2))
        result = autograd_of_fn(klong, fn, np.array([1.0, 2.0, 3.0]))
        expected = np.array([2.0, 4.0, 6.0])
        self.assertTrue(np.allclose(result.cpu().numpy(), expected, atol=1e-5))

    def test_autograd_of_fn_with_callable(self):
        """Test autograd_of_fn with plain callable."""
        klong = KlongInterpreter()
        result = autograd_of_fn(klong, lambda x: torch.sum(x**2), np.array([1.0, 2.0, 3.0]))
        expected = np.array([2.0, 4.0, 6.0])
        self.assertTrue(np.allclose(result.cpu().numpy(), expected, atol=1e-5))


if __name__ == '__main__':
    if not USE_TORCH:
        print("=" * 70)
        print("WARNING: USE_TORCH=1 not set. Torch-specific tests will be skipped.")
        print("Run with: USE_TORCH=1 python -m unittest tests.test_torch_backend")
        print("=" * 70)
    unittest.main()
