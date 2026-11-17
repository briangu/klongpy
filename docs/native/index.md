# Native Runtime

This section describes the native Klong runtime: a second implementation of the Klong language, written in Rust, that runs on microcontrollers and on a desktop without Python. It is for people who want to run Klong on a small board, or who want to know how that implementation relates to KlongPy and how far it has been tested. This page says what it is, how it relates to KlongPy, what exists today, and what has and has not been proven.

The source is in the `native/` directory of the KlongPy repository. The runtime is new, and it has been run on one real board.

## What it is for

KlongPy is Python, numpy and optionally torch. That is the right tool on a computer, and the wrong one on a microcontroller with a few hundred kilobytes of RAM. The native runtime is the same language with none of that underneath:

* **The language.** The same parser, the same operators, adverbs, strings, dictionaries, modules and error kinds, the same strict levels, and the twelve library files of `klongpy/lib`. A program that avoids the differences below gives the same answer.
* **A small program.** The interpreter is `no_std` Rust with one dependency (`libm`). The linked image for a Cortex-M or RISC-V board with the serial console and the library is between 235 and 330 KB of flash, and it needs 64 KiB of heap for small programs. See [Building and Running](building.md#sizes-and-speed).
* **A console on a serial line.** A REPL with line editing over a UART, Ctrl-C for a program that does not stop, and a command mode for uploading files and running programs from a desktop, so that you can develop on a board without a debugger. See [Serial Console and Upload](console.md).
* **Nodes that talk.** The kdb+ IPC protocol, as KlongPy speaks it, timers, and a management library for reaching several nodes from one. See [IPC and Timers](ipc.md) and [Management Node](management-node.md).

It is a different thing from [compiling Klong to C](../embedded.md). That path compiles a small, fixed-size subset ahead of time to C with no interpreter at all. The native runtime is an interpreter that runs the whole language, on the device, and you can type into it.

A first look, with the program built as described in [Building and Running](building.md):

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ klong -e '+/{x*x}'"'"'!10'
285
$ klong -e '.l("nstat");.p(sd([1 2 3 4]))'
1.1180339889579018
"1.1180339889579018"
```

## How it relates to KlongPy

It is the **same language**, and that is checked and not only intended. The repository has a conformance suite, `conformance/cases`, of 1294 cases at the time of writing for the default build. Each case is a program, the printed value or the error kind it must produce, and optionally the console output. KlongPy runs the suite and the native runtime runs the same files, through a small command-line protocol, with no Python. KlongPy is the reference: a case that KlongPy fails is either a KlongPy bug, which goes to a pending list, or a wrong case. The 169 golden parse trees of KlongPy's own tests are matched byte for byte as well.

These are the differences. All of them follow from having no Python and no numpy, or from running on a small device.

| | KlongPy | Native runtime |
|---|---|---|
| Python interop | `.py`, `.pyf`, `.pya`, `.pyc`, `.bkf` | **none**; calling one is `name error: undefined function` |
| Other desktop functions | `.sys`, `.web`, `.ws`, `.db`, `.bkd`, `.bkp` and others | not available, the same `name` error |
| numpy and torch | the array backend | **none**; the arrays are the runtime's own, and the autograd operators (`∇`, `∂`, `:>`) give `name error: operator not available on this build` |
| Integers | 64 bits, wrapping; a long literal or a long operand is exact for `+ - * < > \| &` | **wrap at the build width** (64 bits by default, 32 with the feature `int32`); a literal that does not fit is a `syntax` error |
| Floats | 64 bits | 64 bits by default, 32 with the feature `float32` |
| Strings | Unicode code points | Unicode code points, as KlongPy: `#"héllo"` is 5 and indexing counts code points. The runtime stores UTF-8; a string is always valid text |
| Data nesting | a list nested 300 levels works | at most 128 levels of nested lists, by default (a `range` error beyond) |
| Names and messages | | a name is at most 255 bytes and an error message at most 300 bytes |
| Recursion | a `range` error at a fixed depth | also a `range` error, at a depth that the build and the native stack set (see [Building and Running](building.md#memory-and-stack)); never a crash |
| `.try` and `.err` | not there | **native only**: catch an error as a value, and raise one |
| `.timer` with a named callback | looked up again at each tick | the function value given is kept |
| IPC | `.srv .cli .clid .clic .clit .async` | the same functions and the same protocol; see [IPC and Timers](ipc.md) for the differences |
| Memory | the Python heap | a heap with a limit: going over it is a `memory` error and the interpreter stays usable |

Two more things are worth knowing. The REPL of the desktop `klong` program is simpler than `kgpy`: each line is a program, and a definition has to fit on one line (the serial console of a board does have continuation lines and history). And the numbers are the same only while they fit: a computation that wraps at 32 bits on a narrow build gives KlongPy's 64-bit answer on the desktop. The [KlongPy differences page](../differences.md) describes the differences of KlongPy from classic Klong, which apply to both.

## What exists today

| Piece | What it is | Page |
|---|---|---|
| `klong` | the runtime as a desktop program: REPL, `-e`, files, `--conformance` | [Building and Running](building.md) |
| `klong-sim` | a simulated board: the real console over a pipe or a pseudo-terminal | [Serial Console and Upload](console.md) |
| `tools/kgupload.py` | upload, run and reset over a serial port, from Python's standard library alone | [Serial Console and Upload](console.md) |
| `klong-ffi` | the board runtime as a static library with a C interface, `klong.h` | [Porting to a Board](porting.md) |
| the ESP32-P4 firmware | the console on the UART of one real board | [ESP32-P4](esp32-p4.md) |
| IPC, timers | kdb+ IPC and `.timer`, on the desktop over TCP | [IPC and Timers](ipc.md) |
| `mgmt` | the management library, `klongpy/lib/mgmt.kg` | [Management Node](management-node.md) |

## Status: what has and has not been proven

"Proven" here means that it was run, with the result stated. Dates are 2026-11-16. An empty cell means no.

| Area | Result | Where it ran |
|---|---|---|
| The language, default build (64-bit) | 1293 of 1293 conformance cases; 169 of 169 parse trees | desktop |
| The language, 32-bit build (`int32`, `float32`) | 1160 of 1160 cases that the build declares | desktop |
| The embedded library | all twelve files load, at strict levels 0, 1 and 2 | desktop |
| The serial console | 1293 of 1293 cases through `^A run` | simulated boards |
| The serial console | 1266 of 1266 cases over the serial port, in 1877 seconds | **an ESP32-P4** |
| REPL, line editing, Ctrl-C on a runaway loop, the embedded library, uploads | work | **an ESP32-P4** |
| Speed | `fib(24)` 4.2 s on the board, against about 35 ms on an Apple-silicon Mac; **the cause is not investigated** | an ESP32-P4 and the Mac |
| Firmware image | 692 KB | an ESP32-P4 |
| IPC with KlongPy, both directions, every mapped type | passes | desktop, localhost TCP |
| IPC with a q implementation (PeachQ 0.92.20261115), both directions | passes when the tests are enabled; not run again for this documentation | desktop, localhost TCP |
| IPC against a KX kdb+ process | not tried | |
| IPC on a device | **not run**; the ESP32-P4 firmware has no IPC | |
| IPC over a UART or a lossy link | tested in memory with seeded faults; **never on a wire** | in-process simulation |
| Timers | work | desktop |
| The management node | works with three native node processes on localhost, and in an in-process test over simulated lossy and serial links | desktop |
| The management node on boards | **not run** | |
| Flash storage on a board | **not done**; the file system is RAM and is lost at reset | |
| Arm Cortex-M (`thumbv7em-none-eabihf`) | the image links; it has no start-up code and was **never booted** | |
| RISC-V without an ESP chip (`riscv32imac`) | the same | |
| An ESP32-C3 under QEMU | the firmware's glue booted and ran expressions, uploads, timers and `.x`; a runaway loop ended in a watchdog timeout, taken to be a fault of the emulator and not investigated | QEMU |
| Fuzzing | seeded fuzzers of the parser, evaluator, IPC decoder and console, no crash or hang found after fixes; not a coverage-guided fuzzer | desktop |
| Memory safety under out-of-memory | with every large allocation failing in turn, every failure is a `memory` error, no abort | desktop |

Known issues on the board:

* After a reset the first line can fail with `input is not valid UTF-8`, because of a stray byte. Send the line again. (The firmware source flushes the receive buffer at start-up now; not yet tried on the board.)
* Opening the serial port resets the board, and the RAM file system is lost at every reset.
* `fib(24)` is far slower on the board than the clock would suggest, and nobody has found out why.

## Where to go next

* To build it and try it on your computer: [Building and Running](building.md).
* To see the board console without a board: [Serial Console and Upload](console.md).
* To use the one real board: [ESP32-P4](esp32-p4.md).
* To connect nodes: [IPC and Timers](ipc.md), then [Management Node](management-node.md).
* To put it on another chip: [Porting to a Board](porting.md).
