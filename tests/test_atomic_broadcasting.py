"""Atomic dyads pair nested lists from the outermost dimension (discussion #82)."""
import operator

import numpy as np
import pytest

from klongpy import KlongInterpreter
from klongpy.compiler import compile_expr


OPS = [
    ('+', operator.add), ('-', operator.sub), ('*', operator.mul),
    ('%', operator.truediv), (':%', lambda a, b: np.trunc(a / b).astype(int)),
    ('!', np.fmod), ('^', operator.pow), ('&', np.minimum), ('|', np.maximum),
    ('=', operator.eq), ('<', operator.lt), ('>', operator.gt),
]


def test_discussion_82(klong):
    result = klong('[1 2 3]+[3 3]:^[1 2 3 4 5 6 7 8 9]')
    np.testing.assert_array_equal(klong._backend.to_numpy(result),
                                  [[2, 3, 4], [6, 7, 8], [10, 11, 12]])


@pytest.mark.parametrize('op,fn', OPS)
@pytest.mark.parametrize('reverse', [False, True])
def test_atomic_vector_matrix(klong, op, fn, reverse):
    vector = np.array([2, 3])
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    a, b = (matrix, vector) if reverse else (vector, matrix)
    aa, bb = (matrix, vector[:, None]) if reverse else (vector[:, None], matrix)
    # Literal operands force the interpreted dyad path.
    left, right = ('[[1 2 3] [4 5 6]]', '[2 3]') if reverse else ('[2 3]', '[[1 2 3] [4 5 6]]')
    result = klong(f'{left}{op}{right}')
    np.testing.assert_allclose(klong._backend.to_numpy(result), fn(aa, bb))
    # Variable operands exercise compiled expressions where supported.
    klong['a'] = klong._backend.kg_asarray(a)
    klong['b'] = klong._backend.kg_asarray(b)
    result = klong(f'a{op}b')
    np.testing.assert_allclose(klong._backend.to_numpy(result), fn(aa, bb))


def test_compiled_expression_rechecks_shapes(klong):
    klong['a'] = klong._backend.kg_asarray([10, 20])
    klong['b'] = klong._backend.kg_asarray([1, 2])
    ast = klong.prog('a+b')[1][0]
    compiled = compile_expr(ast, klong)
    assert compiled is not None
    fn, syms = compiled
    np.testing.assert_array_equal(klong._backend.to_numpy(fn(*[klong[s] for s in syms])), [11, 22])
    klong['b'] = klong._backend.kg_asarray([[1, 2, 3], [4, 5, 6]])
    expected = [[11, 12, 13], [24, 25, 26]]
    np.testing.assert_array_equal(klong._backend.to_numpy(fn(*[klong[s] for s in syms])), expected)
    np.testing.assert_array_equal(klong._backend.to_numpy(klong('a+b')), expected)


@pytest.mark.parametrize('expr,expected', [
    ('[10 20]+[2 2 3]:^!12', np.arange(12).reshape(2, 2, 3) + np.array([10, 20])[:, None, None]),
    ('[[10 20] [30 40]]+[2 2 3]:^!12', np.arange(12).reshape(2, 2, 3) + np.array([[10, 20], [30, 40]])[:, :, None]),
    ('10+[[1 2 3] [4 5 6]]', [[11, 12, 13], [14, 15, 16]]),
    ('[10 20]+[1 2]', [11, 22]),
    ('[[1 2] [3 4]]+[[10 20] [30 40]]', [[11, 22], [33, 44]]),
])
def test_atomic_shapes(klong, expr, expected):
    np.testing.assert_array_equal(klong._backend.to_numpy(klong(expr)), expected)


@pytest.mark.parametrize('a,b', [
    ([1, 2, 3], [[1, 2, 3], [4, 5, 6]]),
    ([1], [2, 3]),
    ([[1], [2]], [[3, 4], [5, 6]]),
])
def test_atomic_shape_mismatch(klong, a, b):
    klong['a'] = klong._backend.kg_asarray(a)
    klong['b'] = klong._backend.kg_asarray(b)
    with pytest.raises(ValueError, match='shape'):
        klong('a+b')


def test_ragged_pairing():
    klong = KlongInterpreter(backend='numpy')
    result = klong('[10 20]+[[1 2] [3 4 5]]')
    np.testing.assert_array_equal(result[0], [11, 12])
    np.testing.assert_array_equal(result[1], [23, 24, 25])


def test_numpy_interop_keeps_native_broadcasting():
    klong = KlongInterpreter(backend='numpy')
    vector = np.array([1, 2, 3])
    matrix = np.arange(1, 10).reshape(3, 3)
    klong['a'], klong['b'] = vector, matrix
    np.testing.assert_array_equal(klong('a+b'), [[2, 3, 4], [6, 7, 8], [10, 11, 12]])
    klong['nativeadd'] = lambda x, y: np.add(x, y)
    native = [[2, 4, 6], [5, 7, 9], [8, 10, 12]]
    np.testing.assert_array_equal(klong('nativeadd(a;b)'), native)
    np.testing.assert_array_equal(klong.np.add(vector, matrix), native)
    np.testing.assert_array_equal(vector, [1, 2, 3])
    np.testing.assert_array_equal(matrix, np.arange(1, 10).reshape(3, 3))


def test_atomic_torch_gradients(klong_torch):
    backend = klong_torch._backend
    matrix = backend.kg_asarray([[1., 2., 3.], [4., 5., 6.]])

    def loss(x):
        klong_torch['a'] = x
        klong_torch['b'] = matrix
        return klong_torch('+/+/a*b')

    gradient = backend.compute_autograd(loss, [2., 3.])
    np.testing.assert_array_equal(backend.to_numpy(gradient), [6., 15.])


def test_python_list_operands(klong):
    vector = [10, 20]
    matrix = [[1, 2, 3], [4, 5, 6]]
    klong['a'], klong['b'] = vector, matrix
    np.testing.assert_array_equal(klong._backend.to_numpy(klong('a+b')),
                                  [[11, 12, 13], [24, 25, 26]])
    np.testing.assert_array_equal(klong._backend.to_numpy(klong('b-a')),
                                  [[-9, -8, -7], [-16, -15, -14]])
    assert vector == [10, 20]
    assert matrix == [[1, 2, 3], [4, 5, 6]]


@pytest.mark.parametrize('dtype', [int, float, object])
@pytest.mark.parametrize('op,fn', [('+', operator.add), ('-', operator.sub), ('=', operator.eq)])
@pytest.mark.parametrize('reverse', [False, True])
def test_zero_dimensional_atom_with_ragged_array(klong, dtype, op, fn, reverse):
    if not klong._backend.supports_object_dtype():
        pytest.skip('Ragged arrays require object dtype')
    scalar = klong._backend.kg_asarray(np.array(10, dtype=dtype))
    klong['s'] = scalar
    klong['r'] = klong._backend.kg_asarray([[1, 2], [3, 4, 5]])
    literal = '[[1 2] [3 4 5]]'
    for expr in ((f'{literal}{op}s', f'r{op}s') if reverse else
                 (f's{op}{literal}', f's{op}r')):
        result = klong(expr)
        assert len(result) == 2
        for actual, row in zip(result, ([1, 2], [3, 4, 5])):
            expected = fn(np.asarray(row), 10) if reverse else fn(10, np.asarray(row))
            np.testing.assert_array_equal(actual, expected)
    assert scalar.ndim == 0
    assert scalar.item() == 10


@pytest.mark.parametrize('reverse', [False, True])
def test_zero_dimensional_torch_atom_keeps_gradients(klong_torch, reverse):
    backend = klong_torch._backend
    matrix = backend.kg_asarray([[1., 2., 3.], [4., 5., 6.]])

    def loss(x):
        assert x.ndim == 0
        klong_torch['a'] = x
        klong_torch['b'] = matrix
        return klong_torch('+/+/b*a' if reverse else '+/+/a*b')

    gradient = backend.compute_autograd(loss, np.array(10.))
    np.testing.assert_array_equal(backend.to_numpy(gradient), 21.)


def test_zero_dimensional_object_atom(klong):
    if not klong._backend.supports_object_dtype():
        pytest.skip('Object scalars require object dtype')
    klong['a'] = klong._backend.kg_asarray(np.array(10, dtype=object))
    klong['b'] = klong._backend.kg_asarray(np.array(2))
    np.testing.assert_array_equal(klong._backend.to_numpy(klong('a+b')), 12)
    np.testing.assert_array_equal(klong._backend.to_numpy(klong('b-a')), -8)
