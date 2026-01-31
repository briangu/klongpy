"""
Autograd system functions for KlongPy.

Provides .jacobian() for Jacobian matrix computation.

For optimizers (SGD, Adam, etc.), see examples/autograd/optimizers.py
which can be copied to your project and customized.
"""
import sys


def eval_sys_jacobian(klong, x, y):
    """

        .jacobian(x;y)                                          [Jacobian]

        Compute Jacobian matrix of function x at point y.

        For f: R^n -> R^m, returns m x n matrix where J[i,j] = df_i/dx_j.

        Examples:
            f::{[x@0^2 x@1^2]}
            .jacobian(f;[1 2])   -->  [[2 0] [0 4]]

    """
    from .autograd import jacobian_of_fn
    return jacobian_of_fn(klong, x, y)


def create_system_functions_autograd():
    """Create registry of autograd system functions."""
    def _get_name(s):
        i = s.index('.')
        return s[i:i+s[i:].index('(')]

    registry = {}
    m = sys.modules[__name__]
    for x in filter(lambda n: n.startswith("eval_sys_"), dir(m)):
        fn = getattr(m, x)
        registry[_get_name(fn.__doc__)] = fn

    return registry
