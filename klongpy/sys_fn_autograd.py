"""
Autograd system functions for KlongPy.

Provides .grad() for the numeric gradient (the ASCII form of the nabla operator).
Provides .jacobian() for Jacobian matrix computation.
Provides .bkx() for graph export (torch only; .export belongs to the module system).

For optimizers (SGD, Adam), see klongpy/optimizers.py: .pyf("klongpy.optimizers";"SGDOptimizer").
"""
import sys

from .autograd import jacobian_of_fn, _invoke_fn, eval_dyad_grad

from .errors import KlongTypeError, needs_torch_backend as _needs_torch
from .types import KGSym


def eval_sys_grad(klong, x, y):
    """

        .grad(x;y)                                                  [Grad]

        Compute the numeric gradient of function x at point y, using finite
        differences. This is the ASCII form of the nabla operator: .grad(f;y)
        equals y∇f. The function comes first, as in .jacobian(f;y).

        The point y is taken by value: a number or a list of numbers. (The
        operator y∇f also accepts a symbol and then perturbs that variable;
        .grad does not.) For automatic differentiation use f:>y.

        Examples:
            .grad({x^2};3)               -->  6.0 (approximately)
            .grad({+/x^2};[1 2 3])       -->  [2 4 6] (approximately)

    """
    if isinstance(y, KGSym):
        raise KlongTypeError(".grad: the point must be a number or a list of numbers, not a symbol")
    return eval_dyad_grad(klong, y, x)


def eval_sys_jacobian(klong, x, y):
    """

        .jacobian(x;y)                                          [Jacobian]

        Compute Jacobian matrix of function x at point y.

        For f: R^n -> R^m, returns m x n matrix where J[i,j] = df_i/dx_j.

        Examples:
            f::{[x@0^2 x@1^2]}
            .jacobian(f;[1 2])   -->  [[2 0] [0 4]]

    """
    return jacobian_of_fn(klong, x, y)


def eval_sys_backend_export(klong, x, y, z):
    """

        .bkx(x;y;z)                                          [Backend-Export]

        Export a function's computation graph to a file for inspection.
        Requires the torch backend (kgpy --backend torch, or KlongInterpreter(backend="torch")).

        Arguments:
            x           - Function to export
            y           - Example input for tracing the computation graph
            z           - Path to save the graph (.pt2 file)

        Returns:
            Dictionary with:
                "compiled_fn" - The exported module, callable on tensors
                "export_path" - Path where graph was saved
                "graph"       - String representation of computation graph

        Examples:
            f::{x^2}
            info::.bkx(f;3.0;"model.pt2")
            .p(info@"graph")              :" Print computation graph

        Notes:
            - Only supported with PyTorch backend
            - The exported graph can be loaded with torch.export.load()
            - The function is run once on fake tensors and every tensor operation it executes is recorded (torch.export, non-strict), on both engines
            - Compile the exported program with torch: torch.compile(torch.export.load(path).module())

    """
    fn, example_input, output_path = x, y, z

    backend = klong._backend
    if not backend.supports_autograd():
        raise _needs_torch(".bkx()")

    # Wrap the Klong function for torch
    def wrapped_fn(v):
        return _invoke_fn(klong, fn, [v])

    return backend.export_function(wrapped_fn, example_input, output_path)


def eval_sys_gradcheck(klong, x, y):
    """

        .gradcheck(x;y)                                         [Gradcheck]

        Verify that autograd gradients match numeric gradients.
        Uses torch.autograd.gradcheck for rigorous verification.
        Requires the torch backend (kgpy --backend torch, or KlongInterpreter(backend="torch")).

        Arguments:
            x       - Function to check (should return a scalar)
            y       - Input value or list of inputs to check

        Returns:
            1 if gradients are correct
            Raises error if gradients don't match

        Examples:
            f::{x^2}
            .gradcheck(f;3.0)            :" Returns 1

            g::{(x@0^2)+(x@1^2)}
            .gradcheck(g;[1.0 2.0])      :" Returns 1

        Notes:
            - Only supported with PyTorch backend
            - Uses double precision (float64) when available (CPU/CUDA)
            - Falls back to float32 with relaxed tolerances on MPS
            - Useful for verifying custom gradient implementations

    """
    return klong._backend.klong_gradcheck(klong, x, y)


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
