import unittest

import numpy as np
from klongpy.backend import np as backend_np, use_torch, get_default_backend, kg_asarray, array_size

from klongpy.core import KGChar, KGSym
from tests.backend_compat import requires_strings, requires_object_dtype


# Use backend's isarray for torch compatibility
def isarray(x):
    return backend_np.isarray(x)

# Get the default backend for tests
_backend = get_default_backend()


class TestKGAsArray(unittest.TestCase):

    def test_kg_asarray_scalar_int(self):
        x = 42
        arr = kg_asarray(x)
        assert isarray(arr)
        # Check for integer dtype (works for both numpy and torch)
        dtype_str = str(arr.dtype).lower()
        assert 'int' in dtype_str, f"Expected int dtype, got {arr.dtype}"
        assert arr.shape == ()
        assert arr.item() == 42

    def test_kg_asarray_scalar_float(self):
        x = 3.14
        arr = kg_asarray(x)
        assert isarray(arr)
        dtype_str = str(arr.dtype).lower()
        assert 'float' in dtype_str, f"Expected float dtype, got {arr.dtype}"
        # Use tolerance-based comparison for float32 precision
        assert abs(arr.item() - 3.14) < 1e-5, f"Expected 3.14, got {arr.item()}"

    def test_kg_asarray_empty_list(self):
        x = []
        arr = kg_asarray(x)
        assert isarray(arr)
        dtype_str = str(arr.dtype).lower()
        assert 'float' in dtype_str, f"Expected float dtype, got {arr.dtype}"
        assert array_size(arr) == 0

    @requires_strings
    def test_kg_asarray_string(self):
        s = "hello"
        arr = kg_asarray(s)
        assert isarray(arr)
        assert arr.dtype == object  # assuming KGChar is object dtype
        assert "".join(arr) == "hello"

    def test_kg_asarray_list_of_ints(self):
        x = [1, 2, 3]
        arr = kg_asarray(x)
        assert isarray(arr)
        dtype_str = str(arr.dtype).lower()
        assert 'int' in dtype_str, f"Expected int dtype, got {arr.dtype}"
        # Convert to numpy for comparison
        if hasattr(arr, 'cpu'):
            arr = arr.cpu().numpy()
        assert np.array_equal(arr, [1, 2, 3])

    def test_kg_asarray_nested_list_uniform(self):
        x = [[1, 2], [3, 4]]
        arr = kg_asarray(x)
        assert isarray(arr)
        assert arr.shape == (2,2)
        dtype_str = str(arr.dtype).lower()
        assert 'int' in dtype_str, f"Expected int dtype, got {arr.dtype}"
        # Convert to numpy for comparison
        if hasattr(arr, 'cpu'):
            arr = arr.cpu().numpy()
        assert np.array_equal(arr, [[1,2],[3,4]])

    @unittest.skip("what is the expected behavior for this case?")
    def test_kg_asarray_nested_list_heterogeneous(self):
        # should embedded strings be expanded to individual characters?
        x = [[1, 2], "abc", [3.14, None]]
        arr = kg_asarray(x)
        assert isarray(arr)
        # Because of heterogeneous data, dtype should be object.
        assert arr.dtype == object
        # Check that sub-elements are arrays
        assert isarray(arr[0])
        assert isarray(arr[1])
        assert isarray(arr[2])

    def test_kg_asarray_already_array(self):
        x = np.array([1, 2, 3])
        arr = kg_asarray(x)
        # For torch backend, it converts to tensor, for numpy it should return as-is
        if use_torch:
            # Check values are the same
            if hasattr(arr, 'cpu'):
                arr_np = arr.cpu().numpy()
            else:
                arr_np = arr
            assert np.array_equal(arr_np, x)
        else:
            # Should return as-is because already suitable dtype
            assert arr is x

    @requires_object_dtype
    def test_kg_asarray_jagged_list(self):
        x = [[1, 2, 3], [4, 5], [6]]
        arr = kg_asarray(x)
        assert isarray(arr)
        # Jagged => object dtype
        assert arr.dtype == object
        # Each element should be an array
        assert all(isarray(e) for e in arr)
        assert np.array_equal(arr[0], [1, 2, 3])
        assert np.array_equal(arr[1], [4, 5])
        assert np.array_equal(arr[2], [6])


    def benchmark_kg_asarray(self):
        import timeit
        x = [[1, 2], [3, 4]]
        print(timeit.timeit(lambda: kg_asarray(x), number=100_000))


if __name__ == "__main__":
    # run the benchmark
    TestKGAsArray().benchmark_kg_asarray()
