# Porting the native Klong runtime to a board, in Rust

The guide to porting is `docs/native/porting.md`: what a port supplies, the C interface, the memory budgets, the bring-up steps and what only a board can show. This file has what that page lacks for a port written in Rust. The status of every board is in the table of `docs/native/index.md`.

## The pieces

| Crate | What it is | A port touches it? |
|---|---|---|
| `klong-core` | the language (`no_std`) | no |
| `klong-rt` | `Platform` and `FileSystem` traits, system functions, timers and IPC, heap accounting | implements its traits |
| `klong-console` | the serial console | no |
| `klong-board` | skeleton of a port: `Uart`, `Clock`, `System`, `BoardPlatform`, `run`, an allocator, a linkable image | copies or uses it |
| `klong-ffi` | the same runtime as a static library with a C interface | links it (for C firmware) |
| `klong-sim` | a board that is not there: the same console over memory queues, a pipe or a pseudo-terminal | no (it is the test bench) |

A port is a `Platform`. The quickest way to one is `klong-board`:

```rust
let platform = BoardPlatform::new(uart, clock, system, Some(fs));   // or None::<NoFs>
klong_board::run(platform, BoardConfig { stack_bytes, mem: Some(mem), mem_pressed: Some(pressed), ..Default::default() })
```

`native/klong-board/src/bin/image.rs` is this, complete, for a bare-metal target. It links and has no start-up code.

## The traits

**`Uart`** is the console and never IPC. `read(&mut self, buf) -> usize` returns the bytes that have arrived, 0 when none, and must not wait. `write(&mut self, bytes)` sends all of them and may wait.

**`Clock`**: `now_us()` is monotonic microseconds. `TickClock` builds it from a periodic interrupt, where `tick()` from the handler uses only loads and stores of 32-bit atomics. `wall_clock()` returns seconds since 1970 or `None`.

**`System`**: `name()`, `idle()`, `wait(deadline_us)`, `entropy()` and `reset(code)`. Returning early from `wait` is always right; the default returns at once.

**Storage** is any `klong_rt::FileSystem`. `klong_rt::memfs::MemFs` is a RAM file system that satisfies the contract and is its reference (`klong-rt/tests/rt.rs`). Without storage pass `None::<NoFs>`.

**Memory.** Wrap whatever allocates in `klong_rt::heap::Counting` (over `klong_board::heap::RegionHeap`, the vendor's `malloc`, `esp-alloc` or TLSF), call `set_limit(bytes, reserve)`, and pass `mem` (`used`, `peak`, `limit`) and `mem_pressed` (`pressed`) in `BoardConfig`. Exhausting the heap is then a `memory` error at the prompt. The interrupt flag and the `Counting` and `TickClock` counters are statics and must be in internal RAM.

## A network that carries messages

`klong-rt::node` is a `Transport` (`node:N`) over a lane of 16-byte frames, `klong-dgram` is its `no_std` datagram layer, and `klong-ffi --features lane` exposes both to a C shell through the `klong_lane_*` functions of `klong.h`. `boards/pico-bench/` is that shell for the RP2040 (`thumbv6m-none-eabi`, which has no compare-and-swap, divide or FPU; the runtime needs none of them).

A port wires the lane's console with `klong_console::Console::lane_begin` and `lane_end` around each turn of its main loop, as `klong_ffi` does, and calls `NodeNet::console_local_input` when somebody types on the local console. The results on two Picos and the design are in `docs/native/pico-bench.md`.
