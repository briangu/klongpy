"""A projection with one open argument is a function of one argument for every gradient operator."""
import pytest

from conftest import to_scalar


@pytest.fixture
def proj(klong):
    klong("f(a;b)::{(a*a)+b}; p::f(;3); q::f(2;); g(a;b;c)::{a*b*c}; r::g(2;;3)")
    return klong


@pytest.mark.parametrize("expr,want", [
    ("p:>2", 4.0), ("2∇p", 4.0), (".grad(p;2)", 4.0),
    ("q:>5", 1.0), ("r:>4", 6.0),
])
def test_gradient_of_a_projection(proj, expr, want):
    assert to_scalar(proj(expr)) == pytest.approx(want, abs=1e-2)


@pytest.mark.parametrize("expr", ["2∂p", ".jacobian(p;2)"])
def test_jacobian_of_a_projection(proj, expr):
    assert to_scalar(proj(expr)) == pytest.approx(4.0, abs=1e-2)
