"""
Autograd system functions for KlongPy.

Provides .jacobian() for Jacobian matrix computation.
Provides .compile() for function compilation and graph export (torch only).

For optimizers (SGD, Adam, etc.), see examples/autograd/optimizers.py
which can be copied to your project and customized.
"""
import sys

from .autograd import jacobian_of_fn, _invoke_fn


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

    compiled_fn = backend.compile_function(wrapped_fn, example_input, None)

    # Wrap with Klong-convention parameter name so KGLambda introspection works
    import torch
    def klong_compiled(x):
        if not isinstance(x, torch.Tensor):
            x = backend.create_grad_tensor(x)
        return compiled_fn(x)

    return klong_compiled


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


def eval_sys_compilex(klong, x, y, z):
    """

        .compilex(x;y;z)                                        [Compile-Extended]

        Compile a function with extended options for mode and backend.
        Requires PyTorch backend (USE_TORCH=1).

        Arguments:
            x           - Function to compile
            y           - Example input for tracing the computation graph
            z           - Options dictionary with compile settings

        Options (z):
            "mode"      - Compilation mode:
                          "default"         - Balanced (default)
                          "reduce-overhead" - Faster compile, less optimization
                          "max-autotune"    - Slower compile, best runtime
            "backend"   - Compilation backend:
                          "inductor"   - Default with C++/Triton codegen
                          "eager"      - No compilation (debugging)
                          "cudagraphs" - CUDA graphs (GPU only)
            "fullgraph" - 1 to require full graph compilation
            "dynamic"   - 1 for dynamic shapes, 0 for static

        Mode Comparison:
            | Mode            | Compile | Runtime | Use Case          |
            |-----------------|---------|---------|-------------------|
            | default         | Medium  | Good    | General use       |
            | reduce-overhead | Fast    | OK      | Development       |
            | max-autotune    | Slow    | Best    | Production        |

        Returns:
            Compiled function

        Examples:
            f::{x^2}

            :" Fast compilation for development
            cf::.compilex(f;3.0;:{["mode" "reduce-overhead"]})

            :" Maximum optimization for production
            cf::.compilex(f;3.0;:{["mode" "max-autotune"]})

            :" Debug mode (no compilation)
            cf::.compilex(f;3.0;:{["backend" "eager"]})

        Notes:
            - Only supported with PyTorch backend
            - Requires C++ compiler for inductor backend
            - Use .cmodes() to see all available options

    """
    fn, example_input, options = x, y, z

    backend = klong._backend
    if not backend.supports_autograd():
        raise RuntimeError(
            ".compilex() requires PyTorch backend. "
            "Run with USE_TORCH=1 environment variable."
        )

    # Extract options from dictionary
    mode = options.get("mode", "default") if isinstance(options, dict) else "default"
    compile_backend = options.get("backend", "inductor") if isinstance(options, dict) else "inductor"
    fullgraph = bool(options.get("fullgraph", 0)) if isinstance(options, dict) else False
    dynamic = None
    if isinstance(options, dict) and "dynamic" in options:
        dynamic = bool(options["dynamic"])

    # Wrap the Klong function for torch
    def wrapped_fn(v):
        return _invoke_fn(klong, fn, [v])

    compiled_fn = backend.compile_function(
        wrapped_fn, example_input, None,
        mode=mode, backend=compile_backend, fullgraph=fullgraph, dynamic=dynamic
    )

    # Wrap with Klong-convention parameter name so KGLambda introspection works
    import torch
    def klong_compiled(x):
        if not isinstance(x, torch.Tensor):
            x = backend.create_grad_tensor(x)
        return compiled_fn(x)

    return klong_compiled


def eval_sys_cmodes(klong):
    """

        .cmodes()                                               [Compile-Modes]

        Get information about available torch.compile modes and backends.
        Requires PyTorch backend (USE_TORCH=1).

        Returns:
            Dictionary with:
                "modes"           - Available compilation modes
                "backends"        - Available compilation backends
                "recommendations" - Suggested settings for common use cases

        Examples:
            info::.cmodes()
            .p(info@"modes")          :" Print available modes
            .p(info@"recommendations") :" Print recommended settings

        Mode Comparison:
            | Mode            | Compile Time | Runtime Speed | Best For     |
            |-----------------|--------------|---------------|--------------|
            | default         | Medium       | Good          | General use  |
            | reduce-overhead | Fast         | Moderate      | Development  |
            | max-autotune    | Slow         | Best          | Production   |

        Backend Comparison:
            | Backend    | Description                              |
            |------------|------------------------------------------|
            | inductor   | Default - C++/Triton code generation     |
            | eager      | No compilation - for debugging           |
            | cudagraphs | CUDA graphs - reduces GPU launch overhead|

    """
    backend = klong._backend
    if not backend.supports_autograd():
        raise RuntimeError(
            ".cmodes() requires PyTorch backend. "
            "Run with USE_TORCH=1 environment variable."
        )

    return backend.get_compile_modes()


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
