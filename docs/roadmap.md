# Roadmap

This page lists what version 2 delivers, what is experimental, the limits you meet today and what is planned. Nothing under "Planned" exists, and none of it has a date.

## Delivered in version 2

* [Start Here](getting-started.md), [what's new in v2](whats-new-v2.md) and [migrating to v2](migrating-to-v2.md)
* [Language Additions](language-additions.md), [strict mode](strict.md) and [lint](lint.md)
* [Errors with positions and kinds](errors.md) and the [REPL](repl.md)
* [Compiling Klong to C](embedded.md) for the [static profile](static-profile.md), a subset of the language. It has run with desktop C compilers and has not been run on a microcontroller
* [The conformance suite](conformance.md)
* The [native runtime](native/index.md): [building](native/building.md), the [serial console](native/console.md), [IPC and timers](native/ipc.md), the [management node](native/management-node.md) and [porting to a board](native/porting.md)
* [kdb+ IPC](ipc_capabilities.md) and [timers](timer.md)
* [Contributing](contributing.md)

## Experimental

* [FPGA kernels](hardware.md): Klong functions compiled to hardware. They have run on one Alveo U250 in two images.
* [The native runtime on boards](native/index.md): it passes the conformance suite on a desktop and has run it on one ESP32-P4 board ([ESP32-P4](native/esp32-p4.md)). Two Raspberry Pi Picos have run it with IPC between them ([two Picos calling each other](native/pico-bench.md)). The [status table](native/index.md#status-what-has-and-has-not-been-proven) has the detail.

## Known limits

### Language

Known bugs, where KlongPy's answer is wrong and the right one is known, are listed once, in [Differences from Klong](differences.md#known-bugs). The same page lists the limits of integer power on lists.

Known limit: a Klong function that calls itself is bounded by Python's recursion limit under KlongPy. With the NumPy backend, `f::{:[x;f(x-1);0]};f(160)` runs and `f(165)` gives `range error: expression nested too deeply`.

### PyTorch backend

Details are in [PyTorch Backend and Autograd](torch_backend.md).

* Floats are float32 on every device, so results differ from the numpy backend in the last digits.
* A tensor cannot hold `:undefined`, so an `:undefined` inside a list differs from the numpy backend.
* MPS has no float64.
* `.gradcheck` needs the CPU device and raises on MPS.
* A name that already holds a compiled function should not be assigned a second compiled function in the same session; use a new name.
* The export of a compiled function (`.export`) does not complete for some Klong functions; check the `export_error` entry of its result.

### Native runtime

* **Recursion depth on small stacks.** The native runtime bounds the depth of nested evaluation and the stack it uses, and reports a range error when either is exceeded. The host binary allows 512 nested evaluations: `f::{:[x;f(x-1);0]};f(240)` runs and `f(255)` gives a range error. A board with a small stack gets a smaller depth: the Pico firmware, with a 112 KiB stack, reaches 105 levels ([two Picos](native/pico-bench.md#what-works)).
* **Heap needed to load large libraries.** A load reads one statement at a time and keeps functions as their text. On the 64-bit host the management library (`.l("mgmt")`) needs a heap of about 43 KiB to load; `.l("nstat")` peaks at about 322 KB. On a Pico about 10 KB stay in use once the library has been used ([two Picos calling each other](native/pico-bench.md#what-works)).
* **IPC on a device has run only on two Picos.** IPC between nodes and management-library calls have run on two Raspberry Pi Picos. The ESP32-P4 has run the console conformance suite only (the firmware has no IPC). The host build is tested against an open-source q implementation (PeachQ).
* **One flaky Rust test.** `data_nesting_is_bounded` in `native/klong-core/tests/full.rs` fails in about one run in ten on a desktop. The cause is not known.
* The default native build does not declare the conformance tags `bigint`, `timers`, `ipc` or `desktop`, so the suite does not test those areas for it.

### Interop

* IPC has been tested against PeachQ, not against kdb+. The tests that use PeachQ are opt-in (see [Contributing](contributing.md#tests-that-skip-without-a-tool)).

## Planned

None of the following exists.

* **A WebAssembly build of the emitted C.** The C emitter produces freestanding C99; compiling that to WebAssembly is not built. The WebAssembly import module name of an extern is reserved for it (see [Compiling Klong to C](embedded.md)).
* **The native core as an engine under KlongPy.** KlongPy would call the Rust core for evaluation where it can. Today the two implementations are separate and agree only on what the conformance suite covers.
* **Testing against real kdb+.** The IPC code would be run against a kdb+ server and client, which has not been done.
* **CI for the native and conformance tiers.** The GitHub workflows run the Python tests, a linter and the docs build. They do not build the Rust workspace or run `conformance/run.py` against the native binary. See [Contributing](contributing.md#what-ci-runs).
