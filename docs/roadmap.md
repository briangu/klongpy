# Roadmap

This page lists what v2 delivered, the limits you will meet today, and what is planned but does not exist. Nothing in "Planned" is available, and none of it has a date.

## Done in v2

* [Getting started](getting-started.md), [what's new in v2](whats-new-v2.md) and [migrating to v2](migrating-to-v2.md)
* [Friendlier syntax](language-additions.md), [strict mode](strict.md) and [lint](lint.md)
* [Errors with positions and kinds](errors.md) and the [REPL](repl.md)
* [Compiling a static subset of Klong to C](embedded.md) and the [static profile](static-profile.md)
* [The conformance suite](conformance.md)
* The [native runtime](native/index.md): [building](native/building.md), the [serial console](native/console.md), the [ESP32-P4 port](native/esp32-p4.md), [IPC and timers](native/ipc.md), the [management node](native/management-node.md) and [porting to a board](native/porting.md)
* [kdb+ IPC](ipc_capabilities.md) and [timers](timer.md)
* [Contributing](contributing.md)

## Known limits

### Language

The cases below are KlongPy answers that are wrong and whose correct answer is known. Each is a case in `conformance/pending/known-differences.cases`; see [Differences from Klong](differences.md#known-bugs) for the examples.

* `&[]` is a type error where the result should be `[]`.
* `[:a :b]?:b` gives `[]` where it should give `[1]`.
* `*"abc"`, `*|"hello"` and the first item of `,\"abc"` are one-character strings where they should be chars.
* `[1 2 3]@1.0` returns the whole list where it should be a type error.
* `%"a"` and `&"a"` are range errors where they should be type errors.

A Klong function that calls itself is limited by Python's recursion limit under KlongPy. On 2026-11-16 with the numpy backend, `f::{:[x;f(x-1);0]};f(160)` ran and `f(165)` gave `range error: expression nested too deeply`.

### PyTorch backend

Details are in [PyTorch Backend](torch_backend.md).

* Floats are float32 on every device, so results differ from the numpy backend in the last digits.
* A tensor cannot hold `:undefined`, so an `:undefined` inside a list differs from the numpy backend.
* MPS has no float64.
* `.gradcheck` needs the CPU device and raises on MPS.
* A name that already holds a compiled function should not be assigned a second compiled function in the same session; use a new name.
* The export of a compiled function (`.export`) did not complete for a Klong function in the test run; check the `export_error` entry of its result.

### Native runtime

* **Recursion depth on small stacks.** The native runtime bounds the depth of nested evaluation and the stack it uses, and reports a range error when either is exceeded. The host binary has a default depth limit of 512 nested evaluations. On 2026-11-16, `f::{:[x;f(x-1);0]};f(240)` ran and `f(255)` gave a range error. A board with a small stack budget gets a smaller depth. The depth a real board allows has not been measured.
* **Heap needed to load large libraries.** The management library (`.l("mgmt")`) needs a heap of at least 270 KiB on the 64-bit host build: with `--heap-limit 269K` it fails with a memory error and with `270K` it loads. A board whose Klong heap is smaller cannot load it. The figure for a 32-bit build has not been measured.
* **IPC on a device has not run on hardware.** IPC and timers are tested against a simulated board and against the host binary, and the host build is tested against an open-source q implementation (PeachQ). The only hardware result is the conformance suite on an ESP32-P4 (1266 of 1266 over the serial port, 2026-11-16). IPC over a real link on a board has not been run.
* **One flaky Rust test.** `data_nesting_is_bounded` in `native/klong-core/tests/full.rs` has failed about one run in ten on the development machine. In six runs on 2026-11-16 it passed every time, and the cause is not known.
* The default native build does not declare the conformance tags `bigint`, `timers`, `ipc` or `desktop`, so the suite does not test those areas for it.

### Interop

* IPC has been tested against PeachQ, not against kdb+. The tests that use PeachQ are opt-in (see [Contributing](contributing.md#tests-that-skip-without-a-tool)).

## Planned

None of the following is available.

* **Hardware kernels on FPGA from static Klong.** Compile a function in the [static profile](static-profile.md) to a hardware kernel and call it from Klong. No code for it exists in the repository.
* **A WebAssembly build of the emitted C.** The C emitter produces freestanding C99; compiling that to WebAssembly is not built. The WebAssembly import module name of an extern is reserved for it (see [Compiling Klong to C](embedded.md)).
* **The native core as an engine under KlongPy.** KlongPy would call the Rust core for evaluation where it can. Today the two implementations are separate and agree only on what the conformance suite covers.
* **Testing against real kdb+.** The IPC code would be run against a kdb+ server and client, which has not been done.
* **CI for the native and conformance tiers.** The GitHub workflows run the Python tests, a linter and the docs build. They do not build the Rust workspace or run `conformance/run.py` against the native binary. See [Contributing](contributing.md#what-ci-runs).
