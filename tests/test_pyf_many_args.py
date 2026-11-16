"""A Python function imported with .py or .pyf receives every argument written."""
import sys
import types
import unittest

import numpy as np

from klongpy import KlongInterpreter
from klongpy.errors import KlongPythonError

LEVELS = (0, 1, 2)


def _module():
    m = types.ModuleType("pyf_many_args_mod")
    exec("def opt(a, b, c, d=1, e=2, f=3):\n    return [a, b, c, d, e, f]\n"
         "def opt2(a, b=1, c=2, d=3, e=4):\n    return [a, b, c, d, e]\n"
         "def five(a, b, c, d, e):\n    return [a, b, c, d, e]\n", m.__dict__)
    return m


class TestPyfManyArguments(unittest.TestCase):
    def setUp(self):
        sys.modules["pyf_many_args_mod"] = _module()

    def tearDown(self):
        del sys.modules["pyf_many_args_mod"]

    def klong(self, level=0):
        k = KlongInterpreter(strict=level)
        k('.pyf("numpy";"linspace")')
        k('.pyf("pyf_many_args_mod";["five" "opt" "opt2"])')
        return k

    def test_fourth_argument_of_linspace_reaches_python(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = self.klong(level)
                np.testing.assert_allclose(k("linspace(0;1;4;0)"), [0.0, 0.25, 0.5, 0.75])
                np.testing.assert_allclose(k("linspace(0;1;4;1)"), [0.0, 1 / 3, 2 / 3, 1.0])
                np.testing.assert_allclose(k("linspace(0;1;4)"), [0.0, 1 / 3, 2 / 3, 1.0])

    def test_more_than_three_required_parameters_keep_the_packed_form(self):
        # a design decision, pinned by tests/test_error_wrapping.py: such a function
        # takes (list; dict); it is not a function of the written arguments
        k = self.klong()
        self.assertEqual(list(k("five([1 2 3 4 5];:{})")), [1, 2, 3, 4, 5])

    def test_many_optional_parameters(self):
        k = self.klong()
        self.assertEqual(list(k("opt(1;2;3;4;5;6)")), [1, 2, 3, 4, 5, 6])
        self.assertEqual(list(k("opt(1;2;3;4)")), [1, 2, 3, 4, 2, 3])

    def test_called_through_a_variable_and_apply(self):
        k = self.klong()
        self.assertEqual(list(k("g::opt2;g(1;2;3;4;5)")), [1, 2, 3, 4, 5])
        self.assertEqual([int(v) for v in k("opt2@[1 2 3 4 5]")], [1, 2, 3, 4, 5])

    def test_too_many_is_the_python_error(self):
        k = self.klong()
        with self.assertRaises(KlongPythonError) as c:
            k("opt2(1;2;3;4;5;6)")
        self.assertIn("6 were given", str(c.exception))
        with self.assertRaises(KlongPythonError) as c:
            k("opt(1;2;3;4;5;6;7)")
        self.assertIn("7 were given", str(c.exception))

    def test_too_few_is_the_python_error(self):
        k = self.klong()
        with self.assertRaises(KlongPythonError) as c:
            k("linspace(0)")
        self.assertIn("missing", str(c.exception))


if __name__ == "__main__":
    unittest.main()
