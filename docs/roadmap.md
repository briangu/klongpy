# Roadmap

This page lists what version 2 delivers, what is experimental, the limits you meet today and what is planned. Nothing under "Planned" exists, and none of it has a date.

## Delivered in version 2

* [Start Here](getting-started.md), [what's new in v2](whats-new-v2.md) and [migrating to v2](migrating-to-v2.md)
* One interpreter, written in Rust, behind the `kgpy` and `kg` commands ([What Runs Where](what-runs-where.md))
* [Language Additions](language-additions.md), including [modules](language-additions.md#modules); [strict mode](strict.md) and [lint](lint.md)
* [Errors with positions and kinds](errors.md) and the [REPL](repl.md)
* [Compiling Klong to C](embedded.md) for the [static profile](static-profile.md), a subset of the language. It has run with desktop C compilers and, for three examples, on two Raspberry Pi Picos
* [The conformance suite](conformance.md)
* The interpreter for boards and its tools ([Native Runtime](native/index.md)): [building](native/building.md), the [serial console](native/console.md), [IPC and timers](native/ipc.md), the [management node](native/management-node.md) and [porting to a board](native/porting.md)
* [kdb+ IPC](ipc_capabilities.md) and [timers](timer.md)
* [Contributing](contributing.md)

## Experimental

* [FPGA kernels](hardware.md): Klong functions compiled to hardware. They have run on one Alveo U250 in two images.
* [The interpreter on boards](native/index.md): it passes the conformance suite on a desktop and has run the console suite on one ESP32-P4 board ([ESP32-P4](native/esp32-p4.md)). Two Raspberry Pi Picos have run it with IPC between them ([two Picos calling each other](native/pico-bench.md)). The [status table](native/index.md#status-what-has-and-has-not-been-proven) has the detail.

## Known limits

### Language

Known bugs, where KlongPy's answer is wrong and the right one is known, are listed once, in [Differences from Klong](differences.md#known-bugs). The same page lists the limits of integer power on lists.

Known limit: a Klong function that calls itself is bounded by the stack of the thread that runs it. Under `kgpy`, `f::{:[x;f(x-1);0]};f(6000)` runs and `f(8000)` gives `range error: expression nested too deeply`. Under `kg` the bound is 512 nested evaluations, so `f(254)` runs and `f(255)` gives that error; see the section on the interpreter on boards below.

### PyTorch backend

Details are in [PyTorch Backend and Autograd](torch_backend.md).

* Floats are float32 on every device, so results differ from the numpy backend in the last digits.
* A tensor cannot hold `:undefined`, so an `:undefined` inside a list differs from the numpy backend.
* MPS has no float64.
* `.gradcheck` needs the CPU device and raises on MPS.
* `.bkx` does not complete for some Klong functions, those that branch on a tensor's value; check the `export_error` entry of its result.

### The interpreter on boards and under `kg`

* **Recursion depth on small stacks.** The interpreter bounds the depth of nested evaluation and the stack it uses, and reports a range error when either is exceeded. `kg` allows 512 nested evaluations (`--max-depth` changes it): `f::{:[x;f(x-1);0]};f(240)` runs and `f(255)` gives a range error. A board with a small stack gets a smaller depth: the Pico firmware, with a 112 KiB stack, reaches 105 levels ([two Picos](native/pico-bench.md#what-works)).
* **Heap needed to load large libraries.** A load reads one statement at a time and keeps functions as their text. On the 64-bit host the management library (`.l("mgmt")`) needs a heap of about 43 KiB to load; `.l("nstat")` peaks at about 322 KB. On a Pico about 10 KB stay in use once the library has been used ([two Picos calling each other](native/pico-bench.md#what-works)).
* **IPC on a device has run only on two Picos.** IPC between nodes and management-library calls have run on two Raspberry Pi Picos. The ESP32-P4 has run the console conformance suite only (the firmware has no IPC). The host build is tested against an open-source q implementation (PeachQ).
* **One flaky Rust test.** `data_nesting_is_bounded` in `native/klong-core/tests/full.rs` fails in about one run in ten on a desktop. The cause is not known.
* The default build of `kg` does not declare the conformance tags `bigint`, `timers`, `ipc` or `desktop`, so the suite does not test those areas for it.

### Interop

* IPC has been tested against PeachQ, not against kdb+. The tests that use PeachQ are opt-in (see [Contributing](contributing.md#tests-that-skip-without-a-tool)).

## Planned

None of the following exists.

* **A WebAssembly build of the emitted C.** The C emitter produces freestanding C99; compiling that to WebAssembly is not built. The WebAssembly import module name of an extern is reserved for it (see [Compiling Klong to C](embedded.md)).
* **Testing against real kdb+.** The IPC code would be run against a kdb+ server and client, which has not been done.
* **CI for the Rust tiers.** The GitHub workflows run the Python tests, a linter and the docs build. They do not run the Rust workspace tests or `conformance/run.py` against the `kg` binary. See [Contributing](contributing.md#what-ci-runs).
