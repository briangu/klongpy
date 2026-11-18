# Expression Compiler Notes

This page is a note for contributors about the expression compiler inside the KlongPy interpreter (`klongpy/compiler.py`). It is not about compiling Klong to C; for that, see [Compiling Klong to C](embedded.md). The first section says what the compiler does, and the sections after it are plans: none of that work exists.

## What exists

KlongPy has two separate compilers. They share no code and do different jobs.

### The expression compiler (`klongpy/compiler.py`)

It speeds up arithmetic and reductions on numbers and arrays while a program runs.

- The interpreter asks the compiler about each operator node and each adverb-chain node it evaluates. The result, including "not compilable", is cached on the node, so the work is done once per node.
- The compiler turns a node into a small backend-neutral tree (the IR) and the active backend (`compile_expr_ir` in `klongpy/backends/numpy_backend.py` and `torch_backend.py`) generates a Python function from it.
- What the IR covers: integer and float literals, variables that hold a number or an array, the dyads `+ - * % ^`, the comparisons `> < =`, monadic `-`, and reduce (`/`) and scan (`\`) with `+ * | &`.
- Anything else (strings, lists of mixed values, other operators, calls to functions, indexing, assignment) is left to the interpreter. A node that contains such a part is not compiled as a whole.
- A compiled function is valid only for the kinds of value its variables had when it was compiled (int, float, array of a dtype kind). If a variable is later rebound to another kind, or an integer beyond 64 bits, the interpreter evaluates the node instead. A scalar division by zero is also re-evaluated by the interpreter, which returns `:undefined`.
- It does run inside function bodies. For `f::{+/x*y}` the node `+/x*y` is compiled with `x` and `y` as its variables, on the first call, and reused on later calls while their kinds stay the same. What is not compiled is the function as a unit: the interpreter still evaluates the body node by node.

When a reduce or scan is not compiled (an operator outside the table above, or operands that are not plain variables), the interpreter uses the adverb implementations in `klongpy/adverbs.py`, which call the backend's ufunc `reduce` and `accumulate` for the common operators.

### The static compiler (`klongpy/static/`)

It compiles a restricted subset of Klong, the static profile, to C ahead of time. It is described in [Compiling Klong to C](embedded.md) and [The static profile](static-profile.md). It checks a program (`kgpy --check-static`), infers types and shapes, and emits C code with a runtime header. It is not used by the expression compiler and does not speed up a normal KlongPy run.

## Planned: compile a function as a unit

Not available. Compile the whole body of a function, not node by node, so that `f::{+/x*y}` becomes one compiled call and temporaries between operators can be avoided. Open questions: how parameters and outer variables map into the compiled function, nested functions and closures, and how to keep the guard on operand kinds cheap.

## Planned: trace function calls

Not available. Record the operations a function performs and compile the resulting graph, as `torch.compile` and JAX's `jit` do, with the interpreter as the fallback for anything that escapes the trace (I/O, assignments, Python calls, conditionals that depend on values).

## Related

- [Performance](performance.md) has measurements.
- [Roadmap](roadmap.md) lists what is planned for the project as a whole.
