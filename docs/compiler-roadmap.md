# Expression Compiler Roadmap

## Current State

The expression compiler has two optimization layers:

1. **Compiler** (`compiler.py` → `backend.compile_expr_ir`): catches top-level expressions and compiles them to backend-specific Python functions. Handles arithmetic, comparisons, reduce/scan on arrays. Produces backend-neutral IR, each backend generates its own code.

2. **Adverb ufuncs** (`adverbs.py`): when the compiler can't handle an expression (e.g., reduce/scan inside a lambda), the interpreter falls through to adverb functions that manually dispatch to numpy ufuncs (`np.add.reduce`, `np.add.accumulate`, etc.). Without this layer, `functools.reduce` on 100K elements is ~5000x slower.

The compiler only sees "leaf" expressions — the final expression the interpreter evaluates. It cannot look inside lambdas or function calls because by the time the compiler runs, those are opaque Python objects, not ASTs.

## Phase 2: Lambda Inlining

Extend the compiler to handle lambdas by compiling their body expressions.

When the compiler encounters a lambda like `{+/x*y}`, it can:
- Inspect the lambda's AST body
- Compile it as a function with parameters mapped from the lambda's arguments
- Return a compiled callable that replaces the interpreter's lambda evaluation

This would make the adverb ufunc optimizations in `adverbs.py` redundant for the common case, since `{+/x}(a)` would compile to `np.sum(a)` directly.

Key challenges:
- Lambda ASTs need to be accessible at compile time (they're stored in `KGFn` objects)
- Variable scoping: lambda parameters vs. outer context variables
- Nested lambdas and closures

## Phase 3: Function Call Tracing

Extend the compiler to trace through function calls and compile the compute graph.

Similar to how `torch.compile` and JAX's `jit` work: trace the execution of a function, record the operations, compile the resulting graph. The interpreter becomes the fallback for operations that escape the trace (I/O, control flow, Python interop).

At this point, most of monads.py/dyads.py/adverbs.py would be pushed into backend-generated code for the compiled path. The interpreter modules would remain as the "slow path" for:
- Expressions with side effects (I/O, assignments)
- Dynamic typing that can't be resolved at compile time
- Python interop calls
- Control flow (conditionals, loops)

## Priority

Phase 2 (lambda inlining) is the high-value next step. It would:
- Eliminate the adverb ufunc layer for most use cases
- Let users write `f::{+/x*y}` and get compiled performance
- Keep the same frontend/backend split architecture
