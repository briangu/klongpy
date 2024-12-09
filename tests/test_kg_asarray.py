import unittest

import numpy as np

from klongpy.core import kg_asarray, KGChar, KGSym


class TestKGAsArray(unittest.TestCase):

    def test_kg_asarray_scalar_int(self):
        x = 42
        arr = kg_asarray(x)
        assert np.isarray(arr)
        assert arr.dtype == int
        assert arr.shape == ()
        assert arr.item() == 42

    def test_kg_asarray_scalar_float(self):
        x = 3.14
        arr = kg_asarray(x)
        assert np.isarray(arr)
        assert arr.dtype == float
        assert arr.item() == 3.14

    def test_kg_asarray_empty_list(self):
        x = []
        arr = kg_asarray(x)
        assert np.isarray(arr)
        assert arr.dtype == float  # default dtype for empty list
        assert arr.size == 0

    def test_kg_asarray_string(self):
        s = "hello"
        arr = kg_asarray(s)
        assert np.isarray(arr)
        assert arr.dtype == object  # assuming KGChar is object dtype
        assert "".join(arr) == "hello"

    def test_kg_asarray_list_of_ints(self):
        x = [1, 2, 3]
        arr = kg_asarray(x)
        assert np.isarray(arr)
        assert arr.dtype == int
        assert np.array_equal(arr, [1, 2, 3])

    def test_kg_asarray_nested_list_uniform(self):
        x = [[1, 2], [3, 4]]
        arr = kg_asarray(x)
        assert np.isarray(arr)
        assert arr.shape == (2,2)
        assert arr.dtype == int
        assert np.array_equal(arr, [[1,2],[3,4]])

    @unittest.skip("what is the expected behavior for this case?")
    def test_kg_asarray_nested_list_heterogeneous(self):
        # should embedded strings be expanded to individual characters?
        x = [[1, 2], "abc", [3.14, None]]
        arr = kg_asarray(x)
        assert np.isarray(arr)
        # Because of heterogeneous data, dtype should be object.
        assert arr.dtype == object
        # Check that sub-elements are arrays
        assert np.isarray(arr[0])
        assert np.isarray(arr[1])
        assert np.isarray(arr[2])

    def test_kg_asarray_already_array(self):
        x = np.array([1, 2, 3])
        arr = kg_asarray(x)
        # Should return as-is because already suitable dtype
        assert arr is x

    def test_kg_asarray_jagged_list(self):
        x = [[1, 2, 3], [4, 5], [6]]
        arr = kg_asarray(x)
        assert np.isarray(arr)
        # Jagged => object dtype
        assert arr.dtype == object
        # Each element should be an array
        assert all(np.isarray(e) for e in arr)
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
