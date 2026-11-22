# Compiler Notes

This page is for contributors and for anyone who asks whether KlongPy compiles functions to speed them up. It says what compiles today and lists plans that do not exist yet.

## What exists

KlongPy has one compiler, the static compiler. Every other program is interpreted.

### The static compiler

It compiles a restricted subset of Klong, the [static profile](static-profile.md), ahead of time. It is part of the Rust engine, so `kg` and `kgpy` both have it:

* `check` decides whether a program is inside the profile, without running it.
* `compile c` writes freestanding C99 with no heap and no recursion. See [Compiling Klong to C](embedded.md).
* `fpga source` writes HLS C++ for an FPGA kernel. See [Klong functions as FPGA hardware](hardware.md).

The compiler infers every type and shape before it emits code, so a program that is outside the profile is rejected at compile time. It does not speed up a normal interpreted run.

### Recording a function for torch

On the torch backend, `.export(fn;input;path)` runs a function once and records the tensor operations it executes with `torch.export`. It does not read the function's source, and the result is a torch program, not Klong. See [PyTorch Backend and Autograd](torch_backend.md#graph-export-and-gradient-checks).

## Planned: compile a function as a unit

Not available. Compile the whole body of a function so that `f::{+/x*y}` becomes one compiled call and temporaries between operators can be avoided. Open questions: how parameters and outer variables map into the compiled function, nested functions and closures, and how to keep a guard on operand kinds cheap.

## Planned: trace function calls

Not available. Record the operations a function performs and compile the resulting graph, as `torch.compile` and JAX's `jit` do, with the interpreter as the fallback for anything that escapes the trace (I/O, assignments, Python calls, conditionals that depend on values). `.export` records a trace for torch; compiling it is left to `torch.compile`.

## Related

* [Performance](performance.md) has measurements.
* [Roadmap](roadmap.md) lists what is planned for the project as a whole.
