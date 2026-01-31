"""
Autograd system functions for KlongPy.

Provides .jacobian() for Jacobian matrix computation.
Provides .compile() for function compilation and graph export (torch only).

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


def eval_sys_compile(klong, x, y):
    """

        .compile(x;y)                                           [Compile]

        Compile a function for optimized execution using torch.compile.
        Requires PyTorch backend (USE_TORCH=1).

        Arguments:
            x           - Function to compile
            y           - Example input for tracing the computation graph

        Returns:
            Compiled function (faster execution)

        Examples:
            f::{x^2}
            cf::.compile(f;3.0)          :" Returns compiled function
            cf(5.0)                       :" 25.0 (optimized)

        Notes:
            - Only supported with PyTorch backend
            - Raises error on NumPy backend
            - See .export() for saving graphs to files

    """
    from .autograd import _invoke_fn

    fn, example_input = x, y

    backend = klong._backend
    if not backend.supports_autograd():
        raise RuntimeError(
            ".compile() requires PyTorch backend. "
            "Run with USE_TORCH=1 environment variable."
        )

    # Wrap the Klong function for torch
    def wrapped_fn(v):
        return _invoke_fn(klong, fn, [v])

    return backend.compile_function(wrapped_fn, example_input, None)


def eval_sys_export(klong, x, y, z):
    """

        .export(x;y;z)                                          [Export]

        Export a function's computation graph to a file for inspection.
        Requires PyTorch backend (USE_TORCH=1).

        Arguments:
            x           - Function to export
            y           - Example input for tracing the computation graph
            z           - Path to save the graph (.pt2 file)

        Returns:
            Dictionary with:
                "compiled_fn" - The compiled function
                "export_path" - Path where graph was saved
                "graph"       - String representation of computation graph

        Examples:
            f::{x^2}
            info::.export(f;3.0;"model.pt2")
            .p(info@"graph")              :" Print computation graph

        Notes:
            - Only supported with PyTorch backend
            - The exported graph can be loaded with torch.export.load()
            - Use .compile() for just compiling without export

    """
    from .autograd import _invoke_fn

    fn, example_input, output_path = x, y, z

    backend = klong._backend
    if not backend.supports_autograd():
        raise RuntimeError(
            ".export() requires PyTorch backend. "
            "Run with USE_TORCH=1 environment variable."
        )

    # Wrap the Klong function for torch
    def wrapped_fn(v):
        return _invoke_fn(klong, fn, [v])

    return backend.compile_function(wrapped_fn, example_input, output_path)


def eval_sys_gradcheck(klong, x, y):
    """

        .gradcheck(x;y)                                         [Gradcheck]

        Verify that autograd gradients match numeric gradients.
        Uses torch.autograd.gradcheck for rigorous verification.
        Requires PyTorch backend (USE_TORCH=1).

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
    from .autograd import _invoke_fn
    import torch

    fn, inputs = x, y

    backend = klong._backend
    if not backend.supports_autograd():
        raise RuntimeError(
            ".gradcheck() requires PyTorch backend. "
            "Run with USE_TORCH=1 environment variable."
        )

    # Determine dtype based on device support
    device = backend.device
    use_float32 = device.type == 'mps'  # MPS doesn't support float64
    dtype = torch.float32 if use_float32 else torch.float64

    # Wrap the Klong function
    def wrapped_fn(v):
        result = _invoke_fn(klong, fn, [v])
        # Ensure result is a scalar tensor for gradcheck
        if isinstance(result, torch.Tensor) and result.numel() > 1:
            result = result.sum()
        return result

    # Convert inputs to tensor on CPU for gradcheck (avoids MPS float64 issues)
    if isinstance(inputs, (list, tuple)) and not isinstance(inputs[0], torch.Tensor):
        inputs = torch.tensor(inputs, dtype=dtype, device='cpu', requires_grad=True)
    elif not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor([inputs], dtype=dtype, device='cpu', requires_grad=True)
    else:
        inputs = inputs.to(dtype=dtype, device='cpu').requires_grad_(True)

    # Run gradcheck with adjusted tolerances for float32
    if use_float32:
        result = backend.gradcheck(wrapped_fn, (inputs,), eps=1e-4, atol=1e-3, rtol=1e-2)
    else:
        result = backend.gradcheck(wrapped_fn, (inputs,))
    return 1 if result else 0


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
