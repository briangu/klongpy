# What Runs Where

This page says which part of KlongPy does the work for each command, backend and target. Read it when you are choosing between `kg` and `kgpy`, or between interpreting a program and compiling it.

There are three parts: the engine, the code that does the array arithmetic, and compiled output.

## The engine

The engine is the interpreter, written in Rust. It parses a Klong program and runs it. The same engine is in the `kg` command, in the `kgpy` command and in the firmware for microcontroller boards, so a program means the same thing in all three.

## Who does the array arithmetic

| You run | Array arithmetic is done by | Hardware |
|---|---|---|
| `kg` | The engine's own vectors | CPU |
| `kgpy` | The engine's own vectors for flat numeric vectors, and NumPy for arrays of two or more dimensions and for arrays that Python code passes in | CPU |
| `kgpy --backend torch` | PyTorch tensors | CPU, a CUDA GPU, or an Apple GPU through MPS |
| Firmware on a board | The engine's own vectors | Microcontroller |

* `kg` is a single program with no Python in it. It starts in a few milliseconds.
* `kgpy` is the same engine hosted in Python. It adds what Python provides: NumPy and PyTorch arrays, [calling Python](python_integration.md), tables and SQL, the [web server](web_server.md) and [websockets](websockets.md).
* [Automatic differentiation](torch_backend.md) needs the torch backend, because PyTorch does the differentiation.
* MPS is not a separate backend. It is the torch backend with the device set to the Apple GPU.

A function that needs Python reports that it needs `kgpy` when a program calls it under `kg`.

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
| NumPy or pandas data, Python libraries, a database, a web service | `kgpy` |
| A GPU, or gradients | `kgpy --backend torch` |
| The interpreter on a microcontroller | The [native runtime](native/index.md) firmware |
| One function as fast as the target allows, on a microcontroller or an FPGA | The compiler |
