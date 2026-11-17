"""
ASCII names for the gradient operators.

`.grad(f;x)` is the ASCII form of `x∇f` (numeric gradient, function first like
`.jacobian(f;x)`). `.jacobian(f;x)` and `f:>x` already are the ASCII forms of
`x∂f` and of autograd. `˙` stays an internal identity.

Run on the torch backend with `pytest tests/test_ascii_gradient_names.py --backend torch`.
"""
import numpy as np
import pytest

from conftest import to_numpy, to_scalar, TORCH_AVAILABLE
from klongpy import KlongInterpreter

# the numeric gradient of a float32 tensor is only good to about 2%
ATOL = {'numpy': 1e-3, 'torch': 2e-2, 'test_backend': 1e-3}


def atol(backend):
    return ATOL.get(backend, 1e-3)


class TestGradFunction:
    def test_scalar_point_equals_nabla(self, klong, backend):
        a = to_scalar(klong('.grad({x^2};3)'))
        b = to_scalar(klong('3∇{x^2}'))
        assert np.isclose(a, 6.0, atol=atol(backend))
        assert np.isclose(a, b, atol=atol(backend))

    def test_float_point(self, klong, backend):
        a = to_scalar(klong('.grad({x^3};2.0)'))
        assert np.isclose(a, 12.0, atol=atol(backend) * 10)

    def test_vector_point_equals_nabla(self, klong, backend):
        a = to_numpy(klong('.grad({+/x^2};[1 2 3])'))
        b = to_numpy(klong('[1 2 3]∇{+/x^2}'))
        np.testing.assert_allclose(a, [2, 4, 6], atol=atol(backend))
        np.testing.assert_allclose(a, b, atol=atol(backend))

    def test_negative_and_zero_points(self, klong, backend):
        a = to_numpy(klong('.grad({+/x^2};[-1.0 0.0 2.0])'))
        np.testing.assert_allclose(a, [-2, 0, 4], atol=atol(backend))

    def test_klong_function_name(self, klong, backend):
        klong('sq::{x^2}')
        a = to_scalar(klong('.grad(sq;4)'))
        b = to_scalar(klong('4∇sq'))
        assert np.isclose(a, 8.0, atol=atol(backend))
        assert np.isclose(a, b, atol=atol(backend))

    def test_python_callable(self, klong, backend):
        klong['f'] = lambda x: x * x  # a Python function is called with its parameter named x
        a = to_scalar(klong('.grad(f;3)'))
        assert np.isclose(a, 6.0, atol=atol(backend))

    def test_point_from_a_variable_is_taken_by_value(self, klong, backend):
        # the symbol mode of x∇f (perturb the global x) is not part of .grad
        klong('x::[1 2 3]')
        klong('f::{+/x^2}')
        a = to_numpy(klong('.grad(f;x)'))
        np.testing.assert_allclose(a, [2, 4, 6], atol=atol(backend))
        np.testing.assert_allclose(to_numpy(klong('x')), [1, 2, 3])

    def test_symbol_point_is_rejected(self, klong, backend):
        from klongpy.errors import KlongTypeError
        with pytest.raises(KlongTypeError):
            klong('.grad({x^2};:x)')


class TestAsciiTwins:
    def test_jacobian_function(self, klong, backend):
        a = to_numpy(klong('.jacobian({x^2};3)'))
        np.testing.assert_allclose(a, [[6.0]], atol=atol(backend))

    def test_jacobian_matches_glyph(self, klong, backend):
        klong('f::{x^2}')
        a = to_numpy(klong('.jacobian(f;[1 2])'))
        b = to_numpy(klong('[1 2]∂f'))
        np.testing.assert_allclose(a, b, atol=atol(backend))

    def test_autograd_colon_greater(self, klong, backend):
        a = to_scalar(klong('{x^2}:>3.0'))
        assert np.isclose(a, 6.0, atol=atol(backend))

    def test_track_is_an_identity_without_ascii_twin(self, klong, backend):
        # ˙ stays in the operator table, does nothing, and gets no .track
        np.testing.assert_allclose(to_numpy(klong('˙[1 2]')), [1, 2])
        with pytest.raises(Exception):
            klong('.track([1 2])')


def test_grad_is_registered_like_the_other_autograd_functions():
    from klongpy.sys_fn_autograd import create_system_functions_autograd
    registry = create_system_functions_autograd()
    assert '.grad' in registry and '.jacobian' in registry


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_grad_numpy_and_torch_agree_with_nabla():
    """.grad on the torch backend equals x∇f there and the numpy result."""
    ref = KlongInterpreter(backend='numpy')
    tch = KlongInterpreter(backend='torch')
    for src_grad, src_nabla in (('.grad({x^2};3)', '3∇{x^2}'),
                                ('.grad({+/x^2};[1 2 3])', '[1 2 3]∇{+/x^2}')):
        g_np = to_numpy(ref(src_grad))
        g_t = to_numpy(tch(src_grad))
        n_t = to_numpy(tch(src_nabla))
        np.testing.assert_allclose(g_t, n_t, atol=2e-2)
        np.testing.assert_allclose(g_t, g_np, atol=2e-2)
