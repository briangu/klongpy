# Porting to a Board

This page is a summary of what it takes to run the native Klong runtime on a new board: what a port has to supply, the C interface that lets a vendor SDK application drive the runtime, and how much flash and RAM to plan for. It is for someone who has a microcontroller and wants to know what the work is before reading the header that defines the C interface, `native/klong-ffi/include/klong.h`. `native/PORTING.md` in the repository has the notes for a port in Rust.

Two kinds of board have run the runtime: the [ESP32-P4](esp32-p4.md) and [two Raspberry Pi Picos](pico-bench.md). The images for `thumbv7em-none-eabihf` and `riscv32imac-unknown-none-elf` link but have never been booted. The Picos are Cortex-M0+ (`thumbv6m-none-eabi`). The [status table](index.md#status-what-has-and-has-not-been-proven) says what has run where, and the last section here lists what only a new board can show.

## What a port is

The runtime does not know what chip it runs on. A port gives it a few services, and the runtime supplies everything else: the interpreter, the system functions, the [serial console](console.md) with its line editor and upload protocol, timers, and (where the board has a transport) [IPC](ipc.md).

| Piece | Crate | Does a port touch it? |
|---|---|---|
| `klong-core` | the language, `no_std` | no |
| `klong-rt` | the `Platform` and `FileSystem` traits, system functions, timers, IPC, heap accounting | implements its traits |
| `klong-console` | the serial console: prompt, line editor, `^A` commands | no |
| `klong-board` | the skeleton of a port: `Uart`, `Clock`, `System`, `BoardPlatform`, `run`, an allocator, a linkable image | copies or uses it |
| `klong-ffi` | the same runtime as a static library with a C interface | links it |
| `klong-sim` | a board that is not there: the console over memory, a pipe or a pseudo-terminal | no (it is the test bench) |
| `tools/kgupload.py` | the desktop side of upload and run | no |

There are two ways in. If the board's firmware is written in Rust, implement the traits of `klong-board` and call `klong_board::run`. If the firmware is a vendor SDK application in C (ESP-IDF, the Pico SDK), link `libklong_ffi.a` and fill a configuration structure with callbacks. The ESP32-P4 port does the second; its glue is a single file of about 200 lines, `native/boards/esp32p4/main/main.c`.

## What a port supplies

**A serial port** (`Uart`, or the `read` and `write` callbacks).

* `read` returns the bytes that have already arrived, and 0 when there are none. **It must not wait.** The runtime polls it from the main loop and, during an evaluation, every 4096 steps, a few milliseconds at most. That poll is how Ctrl-C stops a running program, so no interrupt handler has to know about Ctrl-C.
* `write` sends all the bytes; it may wait for the transmitter.
* The receive buffer must hold what arrives between two polls. At 115200 baud that is 12 bytes per millisecond. The runtime itself keeps at most 4096 bytes of typed-ahead input during an evaluation and drops the rest.
* During a flash write a board may be unable to receive. Either give the UART a buffer for the longest write, or let the sender pace itself (`kgupload.py --chunk` and `--chunk-delay`), or use `hex` uploads, where the sender never runs more than `--window` records ahead of the acknowledgements.
* The console is never used for IPC. IPC needs a transport of its own.

**A clock** (`Clock`, `now_ms`): monotonic milliseconds or microseconds. The upload timeout (5 seconds) and the timers use it, and a clock that stands still means a stalled transfer never times out. `wall_clock` (seconds since 1970) is optional.

**System services** (`System`, the optional callbacks):

| Callback | Does | If absent |
|---|---|---|
| name | the board's name for the banner, `.ver()` and `^A ver` | required, at most 31 bytes |
| `idle` | called every turn of the main loop and every 4096 steps of an evaluation: feed the watchdog, yield to the RTOS; must be cheap | nothing |
| `wait(max_ms)` | sleep until input may have arrived or until the deadline; returning early is always right | the loop runs busy |
| `reset(code)` | `.x(code)`: restart the chip | `.x` only stops the program |
| `entropy` | 64 random bits to seed `.rn()` | the clock is used |
| `fatal` | the runtime cannot go on (a Rust panic, which is a bug); the message is not terminated | the runtime loops forever |

`reset` matters more than it looks. The conformance runner uses `.x(0)` to get a fresh interpreter for each case, so without a real reset the cases share one interpreter and some fail.

**Storage** (optional): a `klong_rt::FileSystem` with `open read write seek close remove rename list`. The upload protocol needs `Mode::Write` to create or empty a file, `Mode::Update` to keep the content and allow writing anywhere, a `seek` past the end followed by `write` to leave a gap that reads as zero (Intel HEX records arrive in any order), and a `rename` that replaces an existing file (uploads go to `PATH.tmp` first). `write` may return fewer bytes than given, but not 0. `klong_rt::memfs::MemFs`, a RAM file system, is the reference for the contract and what the C interface provides. Without storage, `put` and `hex` answer `err system`, `run` works, and `.l` loads only the built-in library. LittleFS or FAT is the port author's choice; none is done.

**Memory settings.** Two numbers and a limit:

* `stack_bytes`: the native stack the interpreter may use, measured from the frame that calls `klong_run`. Deep recursion beyond it is a `range` error, not a stack overflow.
* the heap: a pair of `alloc` and `free` callbacks (or an allocator over a region), a `heap_limit`, and a `heap_reserve` of 8 to 16 KiB held back so that running out is a `memory` error at the prompt. The runtime counts the bytes it asks for, so the limit has to leave room for the allocator's own overhead (three quarters of the free PSRAM on the ESP32-P4).
* The interrupt flag and the allocator's counters are static variables: they must be in internal RAM. On Xtensa, atomics in PSRAM do not work.

## The C interface

The interface is the header `native/klong-ffi/include/klong.h`, which has `KLONG_ABI_VERSION` 1. Only integers and pointers cross it, never a float or double, so the header means the same under a soft-float and a hard-float C ABI. Everything runs on **one** task: nothing in it is reentrant or thread-safe, and every call and every callback happens on that task.

| Function | Does |
|---|---|
| `int32_t klong_init(const klong_config *config)` | build the runtime, run `boot.kg` if the file system has one, print the banner `Klong VERSION (NAME, int64 float64)` and the first prompt. The config is copied. Returns `KLONG_OK` (0), `KLONG_ERR_ARG` (-1: null config, wrong ABI version or size, a required callback missing) or `KLONG_ERR_STATE` (-2: already initialised) |
| `int32_t klong_step(void)` | one turn of the main loop: console input, then timers. 1 when something was done, 0 when there was nothing to do (the caller may sleep), -1 before `klong_init` |
| `void klong_run(void)` | the main loop: `klong_step`, and the `wait` callback when there was nothing to do. Returns only if `klong_init` was not called |
| `void klong_heap(size_t *used, size_t *peak, size_t *limit)` | heap figures in bytes; any pointer may be NULL |
| `void klong_set_diag(size_t (*diag)(uint32_t mode, int64_t *out, size_t cap))` | a diagnostic of the heap, in a library built with the cargo feature `heapdiag` (the Pico bench): the board's function fills the integers of a mode and the system function `.heap(mode)` returns them; the runtime also says what each allocation is for in `volatile uint8_t KLONG_ALLOC_TAG`. See [the Pico page](pico-bench.md#memory) |

The configuration structure, `klong_config`:

| Field | Meaning |
|---|---|
| `abi_version`, `size` | `KLONG_ABI_VERSION` and `sizeof(klong_config)`; both are checked |
| `ctx` | passed back as the first argument of every callback |
| `alloc(ctx, size, align)`, `free(ctx, ptr, size, align)` | the heap; `align` is a power of two, 8 for 64-bit numbers, so a 4-byte aligned `malloc` is not enough |
| `heap_limit`, `heap_reserve` | the limit in bytes (0 for none) and the reserve |
| `heap_overhead`, `heap_round` | what the allocator keeps per block (its header) and the grain it gives blocks out in, so that what the runtime counts is what the arena holds (0, 0: the sizes asked for) |
| `grow(ctx, ptr, old_size, new_size, align)` | optional: resize a block where it is (1 if done, 0 if not, and then nothing has changed). A vector that grows by doubling then needs the room it grows by and not the old block and the new at once; without it the runtime allocates, copies and frees |
| `stack_bytes` | as above; 32 KiB is the least that is useful |
| `write(ctx, bytes, len)`, `read(ctx, buf, cap)` | the console, raw bytes both ways: no echo, no line buffering, no CR LF translation by the driver |
| `now_ms(ctx)` | monotonic milliseconds |
| `idle`, `wait`, `reset`, `entropy`, `fatal` | the optional services above |
| `name` | the board's name, at most 31 bytes, copied |
| `flags` | `KLONG_FLAG_CRLF` (send CR LF for LF, as a terminal on a UART expects), `KLONG_FLAG_DUMB` (no echo, no editing, no history), `KLONG_FLAG_NO_FS` (no RAM file system: `put` and `hex` fail, `run` works) |

A sketch of the glue, from the ESP32-P4 port (not a complete program; the callbacks are the parts you write):

```c
klong_config cfg = {
    .abi_version = KLONG_ABI_VERSION, .size = sizeof cfg,
    .alloc = k_alloc, .free = k_free, .heap_limit = heap_bytes, .heap_reserve = 16 * 1024,
    .stack_bytes = task_stack - 8 * 1024,
    .write = k_write, .read = k_read, .now_ms = k_now_ms,
    .idle = k_idle, .wait = k_wait, .reset = k_reset,
    .name = "myboard", .flags = KLONG_FLAG_CRLF,
};
if (klong_init(&cfg) == KLONG_OK) klong_run();
```

Two more functions are in the library built with the feature `cfun` (the Pico bench): `klong_cfun_register(name, arity, fn, ctx, out_cap)` defines a system function that runs a C function, and `klong_extern_register(symbol, fn, ctx, out_cap)` provides the C function a program declares with `adc::.extern("read_adc";"int -> float")`. The extern is keyed by its C symbol and is not a global: a program reaches it only through the function `.extern` returned, which checks the arguments and the result against the signature. It may be registered before or after the program declares the symbol.

What a module is and where `.l` looks for a name are decided in one place, `native/klong-core/src/module.rs` (`scan`, `interface`, `search`), which is `no_std` and is part of the core every board links. The runtime (`klong-rt`'s `modules.rs`) calls it with the platform's file system and the library of the image; the linter and the checker call the same functions with the disk.

The C interface has the features `files`, `lib` and `timers`. IPC needs a transport from the platform. The C interface has one, built with `klong-ffi --features lane`: the functions `klong_lane_init`, `klong_lane_pump`, `klong_lane_node` and `klong_lane_stats_get` of `klong.h`. A lane is a UART link that carries 16-byte frames, to a switch or a peer node. Over it the library provides IPC (`.cli("node:2")`, `.srv("node:")`) and a console channel. The Pico firmware is a C application that uses it ([two Picos](pico-bench.md)). The ESP32-P4 firmware does not, and has no IPC. The C interface has no other transport, such as TCP. A port in Rust can provide any transport (`Platform::transports()`) and then calls `poll_events` from its loop.

## Memory budgets

**Flash.** The image of `klong-board`, linked with the size profile (`opt-level = "z"`, LTO, `panic = "abort"`), for the two bare-metal targets that link. Measured on 2026-11-16 (bytes; see [Building and Running](building.md#sizes-and-speed) for how):

| Target | Features | `.text` | `.rodata` | flash |
|---|---|---|---|---|
| `thumbv7em-none-eabihf` | `files`, `lib` | 176,788 | 72,656 | 249,464 |
| | `files`, `lib`, `int32`, `float32` | 162,564 | 72,440 | 235,024 |
| | no `files`, no `lib` | 167,188 | 29,676 | 196,884 |
| `riscv32imac-unknown-none-elf` | `files`, `lib` | 218,990 | 109,092 | 328,102 |
| | `files`, `lib`, `int32`, `float32` | 205,518 | 107,984 | 313,522 |
| | no `files`, no `lib` | 207,056 | 63,664 | 270,740 |

These include the console. Static RAM (`.data` and `.bss`) is 48 bytes beside the heap. The Klong library is about 42 KiB of `.rodata` (the `lib` feature). IPC adds roughly 42 KB on `thumbv7em` and 59 KB on `riscv32imac`, timers 1.7 and 2.5 KB, the `mgmt` library 12.6 KB, and `.try` and `.err` 1.7 and 2.3 KB (older figures). The ESP32-P4 image is 691,328 bytes flashed; it is built with the speed profile and includes ESP-IDF.

The images of the board skeleton, linked with the `min-size` profile (`cargo +stable build -p klong-board --profile min-size`), measured on 2026-11-18. The first column is the console and the embedded library (`files`, `lib`, `ipc`, `timers`), the second the same with `int32,float32`, the third no console and no default features (no files, no library, no IPC):

| Target | Console | Console, 32-bit | No console, bare |
|---|---|---|---|
| `thumbv7em-none-eabihf` | 315,876 | 299,996 | 236,964 |
| `riscv32imac-unknown-none-elf` | 418,192 | 401,072 | 326,510 |
| `thumbv7em-none-eabihf` before the module system (`.l` of a module, `export`, `a.b`) | 300,948 | 285,032 | 226,916 |

The module system costs 12.5 KB on the console image and 7.7 KB on the bare one (the image without files or library has no `.l`: the loader is not built there, which saves 1.4 KB). By part, on `thumbv7em` with the console: the loader of the runtime 4.0 KB, the module scopes and their release in the core 2.1 KB, qualified names `a.b` 1.1 KB, the lookup and assignment rules 1.4 KB, the parser 0.6 KB, the library files written as modules 1.5 KB of text, the rest in what the other changes pull in. `.try`, `.err` (the error as a dictionary), `.seed`, `.now` and reading a qualified name on a host value add 2.4 KB to the console image, 2.3 KB to the 32-bit one and 2.4 KB to the bare one (1.0 KB net of the loader that the bare image no longer has).

**RAM.**

* The console keeps 432 bytes of state (32-bit) inside the board's main structure. On the heap it takes at most `max_line` (1024) for the line, `history_bytes` (2048) for 32 lines of history, `max_entry` (8192) while a multi-line entry is open, 4096 for typed-ahead input, and `max_run` (64 KiB) only while a program sent with `run` is held. `put` and `hex` stream to the file system through a 128-byte buffer. The limits are fields of `ConsoleConfig`; a board with a small heap lowers `max_run` and `max_entry`.
* The interpreter needs about 12 KiB after start-up (about 15 KiB on the 64-bit host in the transcripts of [Building and Running](building.md)). `1+1` needs 1.5 KiB more, a dictionary of 100 entries 26 KiB, a load reads one statement at a time and keeps functions as their text, so `.l("csv")` peaks at about 29 KB, `.l("math")` at about 139 KB, `.l("mgmt")` at about 40 KB and `.l("nstat")` at about 322 KB (see [Building and Running](building.md)). Parsed source costs 20 to 30 bytes of heap per byte of source.
* Practical floors: 64 KiB of heap runs expressions and small programs (the whole conformance suite but one case); 256 KiB runs the whole suite; about 350 KiB covers the largest library file, `nstat`.
* **Stack.** 32 KiB or more for the interpreter task. A level of Klong recursion costs about 0.85 KiB of native stack on a 32-bit target (864 bytes measured on a Pico). At the default of keeping a quarter of the budget back, 16 KiB gives about 14 levels, 64 KiB about 57 and 256 KiB about 228. Deeper recursion ends as a `range` error. On the simulated board (`klong-sim --stack-budget`) the console conformance suite passes all but one case with a 64 KiB stack and all but two with 32 KiB; the misses recurse deeply. On the ESP32-P4 the budget is 120 KiB; the Pico firmware has a 112 KiB stack, tells Klong it may use 108 KiB (4 KiB stay for the frames above), keeps 16 KiB of that and reaches 105 levels.

## Bring-up, in short

The steps:

1. Build the core for the target with `cargo build -p klong-board --lib --profile min-size --target TRIPLE` (stock Rust for Arm and RISC-V; the Espressif fork for Xtensa).
2. Write and test the UART alone: echo bytes, send 4 KiB in one burst and count them back. Write the clock and check it against a stopwatch for a minute.
3. Start with no file system and a static heap, call `klong_run`, and expect the banner and `klong> `.
4. Check `+/!100` (4950), editing and history, and that `f::{f(x+1)};f(0)` is a `range` error, `!100000000` a `memory` error, and `{1}{x+1}:~0` runs until Ctrl-C and then says `interrupt`. The prompt must survive each.
5. Let the runaway loop run for longer than the watchdog period before Ctrl-C.
6. Add storage and test `kgupload.py PORT upload` and a 1 MB binary; pull the plug during an upload and check that the old file is still there. Test a stalled transfer: `^A put f 100 0` and send nothing gives `err system: transfer timed out` after 5 seconds.
7. Check that `boot.kg` runs before the banner.
8. Run the conformance suite through the console and record the count, the time and `heap_peak` from `ver`: `python native/scripts/console_conformance.py --port PORT --baud 115200`. See [Serial Console and Upload](console.md#the-conformance-suite-through-the-console).

If the board resets when the serial port is opened (DTR and RTS on ESP32 development boards), the upload tool's first `ver` waits for it to boot; raise `--timeout` if the boot is slow.

## What only a board can show

These were not and could not be checked on a host:

* that the image boots: start-up code, linker script, vector table, clock setup, and a `-march` that matches the silicon. The images for `thumbv7em` and `riscv32imac` have none of these, and are not bootable (the Pico firmware supplies them for `thumbv6m`);
* UART behaviour at speed: overruns during evaluation and during flash writes, the highest reliable baud rate, whether flow control or pacing is needed;
* the real stack use per recursion level on the target, and so the right `stack_bytes`;
* the allocator on real RAM: PSRAM mapping, fragmentation over days, and whether the heap limit leaves the SDK what it needs;
* the flash file system against the `FileSystem` contract: seek past the end, rename over a file, power loss during an upload;
* watchdog and RTOS behaviour of `idle` and `wait`, soft-float `f64` speed, timings;
* reset through `.x`, and whether opening the serial port resets the board.

The ESP32-P4 has answered the first group: the image boots, the console works, Ctrl-C stops a runaway loop, uploads work and the conformance suite passes over the wire. The flash file system and power loss are untested anywhere, because that board's storage is RAM.

The known weak spots of the protocol are listed on the [console page](console.md#weak-spots).
