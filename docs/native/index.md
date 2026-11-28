# Native Runtime

KlongPy's Rust interpreter runs in the standalone `kg` desktop command, in the Python-hosted `kgpy` command, and in microcontroller firmware. This section covers `kg` and the firmware, including what each target can run and what has been tested on hardware.

The source is in the `native/` directory of the KlongPy repository. It passes the conformance suite on a desktop and has run it on one ESP32-P4 board. Two Raspberry Pi Picos have run it with IPC between them. The [status table](#status-what-has-and-has-not-been-proven) below says what has run where.

`kg --version` and the board banner print the release version. Python packages and Rust binaries can spell the prerelease suffix differently.

## What it is for

The firmware uses the same engine as the desktop commands. A board can select the features that fit its RAM and flash budget.

* **The language.** The same parser, the same operators, adverbs, strings, dictionaries, modules and error kinds, the same strict levels, and the thirteen standard library files in `klongpy/lib` (builds with IPC also carry `mgmt`). A program that avoids the differences below gives the same answer.
* **A small program.** The interpreter is `no_std` Rust. See the [size measurements](building.md#sizes-and-speed) for different feature sets. The [RC1 binary tested on the RP2040 Picos](pico-bench.md#rc1-on-two-rp2040-boards) was 809,252 bytes with the lane, heap diagnostics and compiled examples.
* **A console on a serial line.** A REPL with line editing over a UART, Ctrl-C for a program that does not stop, and a command mode for uploading files and running programs from a desktop, so that you can develop on a board without a debugger. See [Serial Console and Upload](console.md).
* **Nodes that talk.** The kdb+ IPC protocol, as KlongPy speaks it, timers, and a management library for reaching several nodes from one. See [IPC and Timers](ipc.md) and [Management Node](management-node.md).

It is a different thing from [compiling Klong to C](../embedded.md). That path compiles a small, fixed-size subset ahead of time to C with no interpreter at all. The native runtime is an interpreter that runs the whole language, on the device, and you can type into it.

A first look, with the program built as described in [Building and Running](building.md):

<!-- check-docs: requires-file native/target/release/kg; path native/target/release -->
```console
$ kg -e '+/{x*x}'"'"'!10'
285
$ kg -e '.l("nstat");.p(nstat.sd([1 2 3 4]))'
1.1180339889579018
"1.1180339889579018"
```

## How it relates to KlongPy

The conformance suite has 2227 cases for desktop `kg`. Each case pins the printed value or the error kind, and can also check console output. Both desktop commands use the same cases. The runner selects cases based on the features each host or target declares. Golden tests also compare 170 parse trees byte for byte. [Conformance suite](../conformance.md)

These are the differences. All of them follow from having no Python and no numpy, or from running on a small device.

| | KlongPy | Native runtime |
|---|---|---|
| Python interop | `.py`, `.pyf`, `.pya`, `.pyc`, `.bkf` | **none**; calling one is a `name` error that says where it is: `.py needs kgpy (the Python-hosted command)` |
| Tables and SQL | Built-in `.table` and `.db`, plus pandas and Arrow interchange | Desktop `kg` has the same table and SQL primitives. Firmware can enable in-memory tables. |
| Other desktop functions | `.sys`, `.rls`, `.web`, `.ws`, FPGA card bindings, `.bkd` and `.bkp` | These functions require `kgpy`. |
| Arrays and gradients | NumPy arrays, or PyTorch tensors with autograd | Native vectors on CPU. A macOS build with Metal supports device arrays and `:>` for a subset of operations. |
| Integers | 64 bits, wrapping; a long literal or a long operand is exact for `+ - * < > \| &` | 64 bits, wrapping (32 with the feature `int32`); a long literal or a long operand is exact for `+ - * < > = \| &`, and a scalar `^` of two integers is exact, as in KlongPy (the cargo feature `bigint`, on for the desktop binary, off for boards: there a literal that does not fit is a `syntax` error) |
| Floats | 64 bits | 64 bits by default, 32 with the feature `float32` |
| Strings | Unicode code points | Unicode code points, as KlongPy: `#"héllo"` is 5 and indexing counts code points. The runtime stores UTF-8; a string is always valid text |
| Data nesting | a list nested 300 levels works | at most 128 levels of nested lists, by default (a `range` error beyond) |
| Names and messages | | a name is at most 255 bytes and an error message at most 300 bytes |
| Recursion | a `range` error at a fixed depth | also a `range` error, at a depth that the build and the native stack set (see [Building and Running](building.md#memory-and-stack)); never a crash |
| `.try`, `.err`, `.seed`, `.now` | the same functions, from the same runtime | catch an error as a dictionary and raise one (`.try`, `.err`); make `.rn()` reproducible (`.seed`); seconds since 1970 (`.now`, on a platform with a clock). See [the management node page](management-node.md#try-and-err) and the notes below |
| `.timer` with a named callback | looked up again at each tick | the function value given is kept |
| IPC | `.srv .cli .clid .clic .clit .async` | the same functions and the same protocol; see [IPC and Timers](ipc.md) for the differences |
| Memory | the Python heap | a heap with a limit: going over it is a `memory` error and the interpreter stays usable |

Two more things are worth knowing. The REPL of the desktop `kg` program is simpler than `kgpy`: each line is a program, and a definition has to fit on one line (the serial console of a board does have continuation lines and history). And the numbers are the same only while they fit: a computation that wraps at 32 bits on a narrow build gives KlongPy's 64-bit answer on the desktop. The [KlongPy differences page](../differences.md) describes the differences of KlongPy from classic Klong, which apply to both.

## Random numbers and the clock

`.rn()` is a pseudo-random generator that is the same on every target. Its state is one 64-bit word, and each draw is a step of xorshift64\* (Vigna, 2014) with the multiplier `0x2545F4914F6CDD1D`. A 64-bit float is the top 53 bits of the output over 2^53, and a 32-bit float (the feature `float32`) is the top 24 bits over 2^24, the 64-bit value cut to 24 bits. `.seed(n)` takes an integer, sets the state to one step of splitmix64 applied to `n` (a negative `n` as its 64-bit two's complement), and returns `n`: after `.seed(42)` the first three draws are `0.1941059175341826`, `0.5626318272656207` and `0.4861061377100522` on the desktop, in compiled C and on a board with 64-bit floats. Without `.seed` the state starts from whatever entropy the platform has.

`.now()` is the time of day as a float: seconds since 1970-01-01 00:00:00 UTC. A desktop has the clock. A board has it only if its port knows the date, and otherwise `.now()` is a `system` error that says the platform has no wall clock; `.pc()` measures time since the program started and works everywhere. With 32-bit floats the value has a resolution of about two minutes, which is all a 24-bit float holds at that size.

## What exists today

| Piece | What it is | Page |
|---|---|---|
| `kg` | desktop REPL, expressions, files, JSON capability discovery, tables, embedded DuckDB and optional Metal | [Building and Running](building.md) |
| `klong-sim` | a simulated board: the real console over a pipe or a pseudo-terminal | [Serial Console and Upload](console.md) |
| `tools/kgupload.py` | upload, run and reset over a serial port, from Python's standard library alone | [Serial Console and Upload](console.md) |
| `klong-ffi` | the board runtime as a static library with a C interface, `klong.h` | [Porting to a Board](porting.md) |
| the ESP32-P4 firmware | the console on the UART of a real board | [ESP32-P4](esp32-p4.md) |
| IPC, timers | kdb+ IPC and `.timer`, on the desktop over TCP | [IPC and Timers](ipc.md) |
| `mgmt` | the management library, `klongpy/lib/mgmt.kg` | [Management Node](management-node.md) |

## Status: what has and has not been proven

This table is the status of every hardware target. Each row says what ran and its result. An empty cell means no. Conditions: dates are 2026-11-16 unless a row says another; a desktop is macOS or Linux.

| Hardware | What has run |
|---|---|
| A desktop (macOS and Linux) | the whole conformance suite, the 32-bit build, IPC over TCP, timers, the management node |
| An ESP32-P4 development board (chip revision v3.2; see [ESP32-P4](esp32-p4.md)) | the console conformance suite over the serial port, the REPL, uploads, the embedded library |
| Two Raspberry Pi Picos ([two Picos calling each other](pico-bench.md)) | IPC between the nodes over a UART link, and management-library calls, through a serial switch of 16-byte frames that is not part of this repository (its frame format is in `native/boards/pico-bench/README.md` and `klong-sim` models it) |
| An Alveo U250 FPGA | experimental kernels built from the static profile of KlongPy (not the native runtime; see [Klong functions as FPGA hardware](../hardware.md)) |
| Anything else | nothing |

| Area | Result | Where it ran |
|---|---|---|
| The language, default build (64-bit) | 2227 of 2227 conformance cases (2026-11-27); 170 of 170 parse trees | desktop |
| The language, 32-bit build (`int32`, `float32`) | 1408 of 1408 cases that the build declares (2026-11-18) | desktop |
| The embedded library | all thirteen files load, at strict levels 0, 1 and 2 | desktop |
| The serial console | 1720 of 1720 cases through `^A run` (2026-11-27) | simulated boards |
| The serial console | 1266 of 1266 cases over the serial port, in 1877 seconds (the suite had 1266 cases when it was run; see [Conformance](../conformance.md#why-the-counts-differ)) | **an ESP32-P4** |
| REPL, line editing, Ctrl-C on a runaway loop, the embedded library, uploads | work | **an ESP32-P4** |
| Speed | `fib(24)` 4.2 s on the board, against about 35 ms on a desktop (Apple M2 Ultra); **the cause is not investigated** | an ESP32-P4 and the desktop |
| Firmware image | 691,328 bytes; this is the image that was flashed and measured, and the ESP32-P4 figures are for it | an ESP32-P4 |
| IPC with KlongPy, both directions, every mapped type | passes | desktop, localhost TCP |
| IPC with PeachQ 0.92.20261115 (a kdb+-compatible q implementation), both directions | passes when the tests are enabled | desktop, localhost TCP |
| IPC against a KX kdb+ process | not tried | |
| C compiled from Klong, on a device | RC1 moving average and PID agreed with the board interpreter. FIR agreed with a flash reference (2026-11-20). The ESP32-P4 diagnostic image has been linked and awaits its device check. [On a Pico](../embedded.md#on-a-pico) | **two RP2040 Picos** |
| IPC on a device | RC1 passed synchronous and asynchronous calls, echo and a management-library call over the UART lane and switch (2026-11-20). The ESP32-P4 firmware has no IPC. | **two RP2040 Picos** |
| Direct USB CDC console on the Pico | deferred in the RC1 run. The prompt, upload and Ctrl-T checks are separate from the tested UART lane console. | **two RP2040 Picos** |
| IPC over a UART or a lossy link | tested in memory with seeded faults; on a wire only the UART link between two Picos | in-process simulation, two Picos |
| Timers | work | desktop |
| The management node | works with three native node processes on localhost, and in an in-process test over simulated lossy and serial links | desktop |
| The management node on boards | on RC1, the management library loaded and relayed calls over the shared UART lane switch (2026-11-20). Dedicated UART links to multiple nodes and physical cable-loss recovery remain untested. | **two RP2040 Picos** |
| Flash storage on a board | **not done**; the file system is RAM and is lost at reset | |
| Arm Cortex-M4F and M7 (`thumbv7em-none-eabihf`) | the image links; it has no start-up code and was **never booted**. The Picos are Cortex-M0+ (`thumbv6m-none-eabi`) and did boot | |
| RISC-V without an ESP chip (`riscv32imac-unknown-none-elf`) | the same | |
| An ESP32-C3 under QEMU | the firmware's glue booted and ran expressions, uploads, timers and `.x`; a runaway loop ended in a watchdog timeout (see the known limits) | QEMU |
| Fuzzing | seeded fuzzers of the parser, evaluator, IPC decoder and console, no crash or hang found after fixes; not a coverage-guided fuzzer | desktop |
| Memory safety under out-of-memory | with every large allocation failing in turn, every failure is a `memory` error, no abort | desktop |

Known limits on the board:

* After a reset the first line can fail with `input is not valid UTF-8`, because of a stray byte. Send the line again.
* Opening the serial port resets the board, and the RAM file system is lost at every reset.
* `fib(24)` is far slower on the board than the clock would suggest. The cause is not known.
* A runaway loop under the ESP32-C3 emulator (QEMU) ends in a watchdog timeout. The cause is not investigated.

## Where to go next

* To build it and try it on your computer: [Building and Running](building.md).
* To see the board console without a board: [Serial Console and Upload](console.md).
* To use the ESP32-P4: [ESP32-P4](esp32-p4.md). For the Picos: [two Picos calling each other](pico-bench.md).
* To connect nodes: [IPC and Timers](ipc.md), then [Management Node](management-node.md).
* To put it on another chip: [Porting to a Board](porting.md).
