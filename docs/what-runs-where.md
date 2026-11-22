# What Runs Where

This page says which part of KlongPy does the work for each command, backend and target. Read it when you are choosing between `kg` and `kgpy`, or between interpreting a program and compiling it.

There are three parts: the engine, the code that does the array arithmetic, and compiled output.

## The engine

The engine is the interpreter, written in Rust. It parses a Klong program and runs it. The same engine is in the `kg` command, in the `kgpy` command and in the firmware for microcontroller boards, so a program means the same thing in all three.

## Who does the array arithmetic

| You run | Array arithmetic is done by | Hardware |
|---|---|---|
| `kg` | The engine's own vectors | CPU |
| `kg --device mps` | Native Metal kernels for supported numeric operations | Apple GPU |
| `kgpy` | The engine's own vectors for flat numeric vectors, and NumPy for arrays of two or more dimensions and for arrays that Python code passes in | CPU |
| `kgpy --backend torch` | PyTorch tensors | CPU, a CUDA GPU, or an Apple GPU through MPS |
| Firmware on a board | The engine's own vectors | Microcontroller |

* Desktop `kg` includes portable column tables and an embedded DuckDB engine. Inspect the installed feature list with `kg --capabilities`.
* `kgpy` adds Python callables, NumPy and PyTorch arrays, pandas and Arrow interchange, and Python network libraries. It shares the native table and SQL primitives.
* The Python host provides differentiation through PyTorch. [Python backends and autograd](torch_backend.md)

A function that needs Python reports that it needs `kgpy` when a program calls it under `kg`.

The three tools that read modules, the interpreter, `lint` and `check`/`compile`, share one implementation of what a module is (a file that assigns `export`) and of where `.l` looks for a name (the directory of the calling file, `KLONGPATH`, the bundled library): it lives in the engine's core, so a board's firmware and the desktop tools answer the same way. The interpreter runs the file, the linter reads its interface to check qualified names, and the compiler reads the interface and the `.extern` declarations without running them.

### Native Metal

On a macOS build that advertises `metal` in `kg --capabilities`, use `kg --device mps` to select the default Metal device. Use `mps:INDEX` to select an enumerated device by zero-based index. The selected device must support the Metal kernels. CPU execution is the default.

Metal accelerates numeric array addition, subtraction, multiplication, and `+/` along the first dimension. The kernels support scalar broadcasting and numeric arrays with matching leading dimensions. Integer arithmetic wraps at the build's integer width. GPU floating-point arithmetic uses float32, and parallel reductions can round differently from CPU reductions. Supported operations can pass buffers directly to each other. A CPU operation that needs their values reads them back.

The `:>` operator uses reverse-mode differentiation for unary functions whose tracked arithmetic uses these operations and produces one output element. Nested gradients and operations that would discard gradient tracking raise errors.

## Compiled output

The compiler takes a function written in the [static profile](static-profile.md), a subset of Klong, and produces source code for another target. The engine is not present when that code runs.

| Output | Runs on | Page |
|---|---|---|
| C source | Anything with a C compiler: a desktop, a Raspberry Pi Pico, an ESP32 | [Compiling Klong to C](embedded.md) |
| FPGA source | An FPGA card, after the vendor's tools build it into a circuit | [Klong functions as FPGA hardware](hardware.md) |

Both `kg` and `kgpy` can check a program against the profile and produce C and FPGA source. Building the circuit and exchanging data with the card are done by `kgpy`.

On an FPGA no interpreter and no backend is running. The function has become hardware. The interpreter's part is to give the reference answer on the desktop, which the hardware's answer is compared with.

## Choosing

| You want | Use |
|---|---|
| A script, a command-line tool, lint or compile in a build | `kg` |
| Tables and SQL from a Klong script | `kg` or `kgpy` |
| NumPy or pandas data, Python libraries, a web service | `kgpy` |
| Native Apple GPU arithmetic | `kg --device mps` |
| Python device tensors and gradients | `kgpy --backend torch` |
| The interpreter on a microcontroller | The [native runtime](native/index.md) firmware |
| One function as fast as the target allows, on a microcontroller or an FPGA | The compiler |
