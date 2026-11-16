# Porting the native Klong runtime to a board

State: written from the host side of ticket 104. No board has run this yet. Everything below
was proven on a simulated board (`klong-sim`) and by linking for `thumbv7em-none-eabihf` and
`riscv32imac-unknown-none-elf`; the section "What only a board can show" lists what the first
real port has to check. Design: `tickets/design/native-runtime.md` sections 6 and 8; chip
facts: `tickets/design/ref-platforms.md`.

## The pieces

| Crate | What it is | A port touches it? |
|---|---|---|
| `klong-core` | the language (`no_std`) | no |
| `klong-rt` | `Platform` and `FileSystem` traits, system functions, timers and IPC, heap accounting | implements its traits |
| `klong-console` | the serial console: prompt, line editor, `^A` commands (`put`, `hex`, `run`, `ver`) | no |
| `klong-board` | skeleton of a port: `Uart`, `Clock`, `System`, `BoardPlatform`, `run`, an allocator, a linkable image | copies or uses it |
| `klong-sim` | a board that is not there: the same console over memory queues, a pipe or a pseudo-terminal | no (it is the test bench) |
| `tools/kgupload.py` | the desktop side: upload, run, version, reset | no |

A port is a `Platform`. The quickest way to one is `klong-board`:

```rust
let platform = BoardPlatform::new(uart, clock, system, Some(fs));   // or None::<NoFs>
klong_board::run(platform, BoardConfig { stack_bytes, mem: Some(mem), mem_pressed: Some(pressed), ..Default::default() })
```

`native/klong-board/src/bin/image.rs` is this, complete, for a bare-metal target (it links;
it has no start-up code).

## What to implement

**`Uart`** (the console; never IPC)
- `read(&mut self, buf) -> usize`: bytes that have arrived, 0 when none. **Must not wait.**
  The runtime polls it from the main loop and, during an evaluation, every 4096 steps (a few
  milliseconds at most); that poll is how Ctrl-C stops a running program, so no interrupt
  handler has to know about Ctrl-C.
- `write(&mut self, bytes)`: send all of them; may wait for the transmitter.
- The receive buffer (interrupt-driven ring, DMA, or the SDK's driver) must hold what
  arrives between two polls: at 115200 baud that is 12 bytes per millisecond. The runtime
  itself keeps at most 4096 bytes of typed-ahead input during an evaluation and drops the rest.
- During a flash write the board may be deaf (on ESP32 chips external RAM is unreachable and
  interrupts may be held off). `put` data arrives while earlier data is being written, in
  128-byte pieces; either give the UART a receive buffer for the longest write, or have
  senders pace themselves (`kgupload.py --chunk N --chunk-delay MS`), or use `hex`, where the
  sender never runs more than `--window` records ahead of the board's acknowledgements.

**`Clock`**
- `now_us()`: monotonic microseconds. `TickClock` builds it from a periodic interrupt
  (`tick()` from the handler, load and store only, 32-bit atomics).
- `wall_clock()`: seconds since 1970 if the board knows, else `None`.
- The upload timeout (5 s) and timers use this clock; a clock that stands still means a
  stalled transfer never times out.

**`System`**
- `name()`: for the banner, `.ver()` and `^A ver`.
- `idle()`: called every turn of the main loop and every 4096 steps of an evaluation. Feed
  the watchdog, yield to the RTOS. Must be cheap.
- `wait(deadline_us)`: sleep until input may be ready or the deadline; returning early is
  always right; the default returns at once (a busy loop).
- `entropy()`: seed for `.rn()` from a hardware generator, if any.
- `reset(code)`: `.x(code)`. Reset the chip. The conformance runner relies on it to get a
  fresh interpreter per case (see below); if it returns, `.x` only stops the program.

**Storage**: any `klong_rt::FileSystem` (`open read write seek close remove rename list`).
- `klong_rt::memfs::MemFs` is a RAM file system that satisfies the contract and is the
  reference for it (`klong-rt/tests/rt.rs`).
- The upload protocol needs: `Mode::Write` creates or empties; `Mode::Update` keeps the
  content and allows writing anywhere; `seek` beyond the end followed by `write` leaves a gap
  that reads as zero (Intel HEX records arrive in any order); `rename` replaces an existing
  file (uploads go to `<path>.tmp` first); `remove`.
- `write` may return fewer bytes than given; 0 is treated as an error.
- Without storage pass `None::<NoFs>`: `put` and `hex` answer `err system`, `run` works,
  `.l` loads only the built-in library.
- LittleFS or FAT is the port author's choice (ticket 104, open questions).

**Memory** (the two numbers in `BoardConfig`)
- `stack_bytes`: the native stack of the task that calls `run`. `Config::for_stack` turns it
  into a recursion limit checked against the real stack pointer, so deep recursion is a
  `range` error, not a crash. Measured: about 0.8 KiB per level of Klong recursion on a
  32-bit target (16 KiB: about 14 levels; 64 KiB: about 60; 256 KiB: about 250).
  The whole conformance suite through the console passes 1265 of 1266 with a 64 KiB stack and
  1264 with 32 KiB; the misses are cases that recurse deeply and end as `range` errors.
- The global allocator: wrap whatever allocates (`klong_board::heap::RegionHeap` over a
  region, the vendor's `malloc`, `esp-alloc`, TLSF) in `klong_rt::heap::Counting`, call
  `set_limit(bytes, reserve)`, and pass `mem` (`used`, `peak`, `limit`) and `mem_pressed`
  (`pressed`). With these, exhausting the heap is a `memory` error at the prompt. Reserve
  8 to 16 KiB.
- The interrupt flag and the `Counting` and `TickClock` counters are statics: they must be in
  internal RAM (on Xtensa, atomics in PSRAM do not work; `ref-platforms.md` section 1).

## Budget

Flash, linked image of `klong-board` (`sh native/scripts/measure.sh`, section "klong-board-image";
`opt-level = "z"`, LTO, `panic = "abort"`; i64 and f64 unless said; bytes; measured 2026-11-16, the
same numbers as `docs/native/building.md` and `docs/native/porting.md`):

| Target | Features | .text | .rodata | flash | of which console and main loop |
|---|---|---|---|---|---|
| thumbv7em-none-eabihf | files, lib | 176788 | 72656 | 249464 | 20564 |
| | files, lib, int32, float32 | 162564 | 72440 | 235024 | |
| | none | 167188 | 29676 | 196884 | 21436 |
| riscv32imac-unknown-none-elf | files, lib | 218990 | 109092 | 328102 | 26850 |
| | files, lib, int32, float32 | 205518 | 107984 | 313522 | |
| | none | 207056 | 63664 | 270740 | 27984 |

The console column is the difference to the same image running one fixed program instead.
Static RAM is 48 bytes (`.data` + `.bss`) beside the heap. The Klong library is 42 KiB of
`.rodata` (feature `lib`). These images have no IPC and no timers: IPC adds roughly 42 KB on
thumbv7em and 59 KB on riscv32imac, timers 1.7 and 2.5 KB, the `mgmt` library 12.7 KB, and
`.try` and `.err` 1.7 and 2.3 KB (older figures, not re-measured). The ESP32-P4 image is
692 KB (speed profile, with ESP-IDF).

RAM:
- Console state: 432 bytes (32-bit) inside the board's main structure, plus on the heap at
  most `max_line` (1024) for the line, `history_bytes` (2048) for 32 lines of history,
  `max_entry` (8192) while a multi-line entry is open, 4096 of typed-ahead input, and
  `max_run` (64 KiB) only while a program sent with `run` is held. `put` and `hex` stream to
  the file system through a 128-byte buffer. All limits are fields of `ConsoleConfig`; a
  board with a small heap lowers `max_run` and `max_entry`.
- Interpreter (ticket 103): about 12 KiB after start-up; `1+1` needs 1.5 KiB more; a
  dictionary of 100 entries 26 KiB; `.l("csv")` 50 KiB; `.l("math")` 495 KiB; `.l("nstat")`
  1.04 MiB. Parsed source costs 20 to 30 bytes of heap per byte of source.
- IPC, per connection (`native/scripts/ipc_bench.py`, host, 2026-11-16): 2149 bytes of heap per
  idle connection on a server and 2303 on a client (handle included), plus one read buffer shared
  by all connections (64 KiB on a 64-bit host, 1 KiB on a 32-bit target) and the message being
  received; 66 KB left over after 100 connections closed.
- Practical floors: 64 KiB of heap runs expressions and small programs (the conformance
  suite: all but 1 case at 64 KiB in ticket 103's run); 256 KiB runs the whole suite; 1 MiB
  is needed for the larger library files.
- Stack: 32 KiB or more for the interpreter task (above).

## Bring-up checklist

1. Build the core for the target: `cargo build -p klong-board --lib --profile min-size --target <triple>`
   (stock Rust for Arm and RISC-V; Espressif's fork for Xtensa).
2. Write the `Uart`. Test it alone: echo bytes; send 4 KiB in one burst and count them back.
3. Write the `Clock`; check it against a stopwatch for a minute.
4. Start with `None::<NoFs>` and `RegionHeap` over a static array (as `image.rs` does), call
   `klong_board::run`. Expect the banner `Klong <version> (<name>, int64 float64)` and
   `klong> `.
5. `+/!100` gives `4950`. Type a line, use backspace and the arrow keys, recall it with up.
6. `kgupload.py PORT ver` answers. If opening the port resets the board (DTR and RTS on
   ESP32 development boards), the tool's first `ver` waits for it; raise `--timeout` if the
   boot takes long.
7. Limits: `f::{f(x+1)};f(0)` is a `range` error; `!100000000` is a `memory` error; the
   prompt survives both. `{1}{x+1}:~0` runs until Ctrl-C and then says `interrupt`.
8. Watchdog: let the runaway loop run for longer than the watchdog period before Ctrl-C.
9. Storage: `kgupload.py PORT upload file.kg`, then `.l("file.kg")`; a 1 MB binary
   (`upload` sends it as Intel HEX; the board's answer carries the CRC-32 of what it reads
   back). Pull the plug during an upload: the old file must still be there afterwards.
10. A stalled transfer: `^A put f 100 0`, send nothing; `err system: transfer timed out`
    after 5 s and the prompt returns.
11. `boot.kg` on the file system runs before the banner.
12. The conformance suite (below). Record the count, the time, and `heap_peak` from `ver`.

## The conformance suite over serial

```
.venv/bin/python native/scripts/console_conformance.py --port /dev/ttyUSB0 --baud 115200
.venv/bin/python native/scripts/console_conformance.py --port PORT --narrow        # an int32,float32 build
```

Per case the runner sends `.x(0)` (the board resets and prints its banner: a fresh
interpreter), `.strict(N)` when the case has a strict tag, and the program with `^A run`; it
compares the value, the console output and the error kind with the case. Without `--port` it
does the same against simulated boards (1270 of 1270 on the default build; add
`-- --stack-budget 64K --heap-limit 1M --crlf` to make the simulation as small as a board).
Needs: `System::reset` that really resets (or the cases share one interpreter and some fail),
a banner within `--timeout`, and `max_run` of at least the largest case (under 4 KiB).
The `sysio` cases write files; on a board without storage, drop the tag (`--tags`).

## The console, as built

- Prompt `klong> `; continuation `     > ` while brackets, a string or a comment are open
  (the rule of KlongPy's `check_complete`). CR, LF or CR LF end a line. Ctrl-C abandons the
  entry or stops the evaluation; Ctrl-D is ignored. `ConsoleConfig::dumb` turns echo, editing
  and history off.
- The byte 0x01 as the first byte of an empty entry starts a command; replies are single
  lines starting with 0x01.

```
^A ver                              ^A ok ver=0.0.1 int=64 float=64 features=... platform=... fs=1 heap_free=... max_line=... max_run=... max_file=...
^A put <path> <length> <crc32-hex>  ^A go    <length> raw bytes    ^A ok <length> <crc32> [warn syntax: ... | unchecked]
^A run <length> <crc32-hex>         ^A go    <length> raw bytes    program output, value, ^A ok value <n>  |  ^A ok  |  ^A err <kind>: <message>
^A hex <path>                       ^A go    records, each ^A ack <n> or ^A nak <n> <reason>; end-of-file record: ^A ok <length> <crc32>
any failure                         ^A err <kind>: <message>
```

- The sender waits for `go` before sending data.
- CRC-32 is zlib's. A header must end with LF or CR LF when data follows.
- Paths have no blanks. Limits: `max_file` for `put` and `hex`, `max_run` for `run`.
- Known weak spots: the command line itself has no check sum (a damaged length makes the
  board read too few or too many raw bytes; the CRC then fails, but surplus bytes of a
  text file are read as input lines); a program that prints the byte 0x01 confuses a tool
  reading its output.

## What only a board can show

- That the image boots: start-up code, linker script, vector table, clock setup, the
  `-march` matching the silicon (ticket 104 lists the unconfirmed items per board).
- UART behaviour at speed: overruns during evaluation and during flash writes, the highest
  reliable baud rate, whether flow control or pacing is needed for `put`.
- Real stack use per recursion level on the target (measured on aarch64 and wasm32 only),
  hence the right `stack_bytes`.
- The allocator on real RAM (PSRAM mapping, fragmentation over days), and that the heap limit
  leaves the SDK what it needs.
- The flash file system against the `FileSystem` contract (seek past the end, rename over a
  file, power loss during upload).
- Watchdog and RTOS behaviour of `idle` and `wait`; soft-float `f64` speed; timings.
- Reset through `.x`, and whether opening the serial port resets the board.
