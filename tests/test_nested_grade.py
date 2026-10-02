"""Grade nested arrays consistently across backends."""
import numpy as np
import pytest

from klongpy import KlongInterpreter


@pytest.mark.parametrize('expr,expected', [
    ('<[[0 1] [0 2]]', [0, 1]),
    ('>[[0 1] [0 2]]', [1, 0]),
    ('<[[0 3] [0 1] [0 2]]', [1, 2, 0]),
    ('>[[0 3] [0 1] [0 2]]', [0, 2, 1]),
    ('<[[0 1] [0 1] [0 2]]', [0, 1, 2]),
    ('>[[0 1] [0 1] [0 2]]', [2, 1, 0]),
    ('<[[[0 1] [0 2]] [[0 3] [0 4]]]', [0, 1]),
    ('>[[[0 1] [0 2]] [[0 3] [0 4]]]', [1, 0]),
    ('<[[] []]', [0, 1]),
    ('>[[] []]', [1, 0]),
    ('<[3 1 2]', [1, 2, 0]),
    ('>[3 1 2]', [0, 2, 1]),
])
def test_nested_grade(klong, expr, expected):
    np.testing.assert_array_equal(klong._backend.to_numpy(klong(expr)), expected)


@pytest.mark.parametrize('op,expected', [
    ('<', [[0, 1], [0, 2], [0, 3]]),
    ('>', [[0, 3], [0, 2], [0, 1]]),
])
def test_sort_rows_with_grade(klong, op, expected):
    result = klong(f'a::[[0 2] [0 1] [0 3]];a@{op}a')
    np.testing.assert_array_equal(klong._backend.to_numpy(result), expected)


def test_grade_tensor_tracking_gradients(klong_torch):
    backend = klong_torch._backend
    a = backend.create_grad_tensor([[0., 2.], [0., 1.]])
    klong_torch['a'] = a
    np.testing.assert_array_equal(klong_torch('<a'), [1, 0])
    np.testing.assert_array_equal(klong_torch('>a'), [0, 1])
    assert backend.has_gradient(a)
    np.testing.assert_array_equal(backend.to_numpy(a), [[0., 2.], [0., 1.]])


def test_string_grade_unchanged():
    klong = KlongInterpreter(backend='numpy')
    np.testing.assert_array_equal(klong('<"foobar"'), [4, 3, 0, 1, 2, 5])
    np.testing.assert_array_equal(klong('>"foobar"'), [5, 2, 1, 0, 3, 4])
