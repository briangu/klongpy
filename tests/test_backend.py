import unittest

from klongpy.backend import is_supported_type, is_jagged_array, np


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
        self.assertTrue(is_supported_type(np.array([1, 2, 3])))
        self.assertTrue(is_supported_type(np.array([[1, 2], [3, 4]])))

    def test_numbers_supported(self):
        self.assertTrue(is_supported_type(42))
        self.assertTrue(is_supported_type(3.14))

    def test_empty_list_supported(self):
        self.assertTrue(is_supported_type([]))


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
        self.assertFalse(is_jagged_array(np.array([1, 2, 3])))

    def test_empty_list_not_jagged(self):
        self.assertFalse(is_jagged_array([]))

    def test_single_element_list(self):
        self.assertFalse(is_jagged_array([[1, 2, 3]]))


class TestNumpyBackend(unittest.TestCase):
    def test_isarray_with_ndarray(self):
        arr = np.array([1, 2, 3])
        self.assertTrue(np.isarray(arr))

    def test_isarray_with_list(self):
        self.assertFalse(np.isarray([1, 2, 3]))

    def test_isarray_with_scalar(self):
        self.assertFalse(np.isarray(42))
        self.assertFalse(np.isarray(3.14))

    def test_isarray_with_string(self):
        self.assertFalse(np.isarray("hello"))


if __name__ == '__main__':
    unittest.main()
