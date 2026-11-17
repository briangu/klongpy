# Klong on the 4-Pico bench

Status: **built, not yet run on hardware.** Nothing here has been flashed to a Pico or seen a
lane. Everything below the C shell (`main.c`) was run on the host and in the simulator
(`klong-sim`, with a model of the bench's switch); `main.c` and the image were only compiled
and linked. The design and the simulator results are in `docs/native/pico-bench.md`.

A Raspberry Pi Pico (RP2040) as one node of the bench: it learns its node number from its
lane, answers the supervisor's PING and sends STATUS, carries messages of up to 1024 bytes
over the lane's 16-byte frames, runs Klong's kdb+ IPC between nodes (`.cli("node:1")`,
`.srv("node:")`, the management library) over those messages, and has the Klong console on
USB CDC and, as a proposal, on datagram channel 1 of the lane.

## Build

    native/boards/pico-bench/build.sh

needs stable Rust with `thumbv6m-none-eabi` (`rustup +stable target add thumbv6m-none-eabi`),
CMake and Ninja, the Pico SDK 2.x (default `~/scm/pico-sdk`, or set `PICO_SDK_PATH`) and the
Arm GNU toolchain (default `~/.local/arm-gnu-toolchain-15.3.rel2-darwin-arm64-arm-none-eabi`,
or set `ARM_BIN` to its `bin`). It

1. builds `libklong_ffi.a` for `thumbv6m-none-eabi` with `--features lane` (release, LTO,
   `panic=abort`, `opt-level` 3, or `KLONG_OPT=z` for the smaller image) into
   `native/target/pico-bench/`;
2. configures and builds the application out of tree in `native/boards/pico-bench/build/`.

The SDK and the toolchain are only read. Nothing is flashed and no port is opened. Output:
`build/klong_pico_bench.uf2` (and `.elf`, `.bin`).

Measured on 2026-11-16 (SDK 2.3.1, GCC 15.3.1): `.bin` 608,276 bytes (29% of the 2 MB flash;
the `.uf2` is 1,217,024 bytes because of its 256-byte blocks); with `KLONG_OPT=z` 419,100
bytes. Static RAM: `.data` 9.2 KB, `.bss` 56.8 KB (of which the interpreter's 48 KB stack and
the 4 KB lane ring), 195 KB left for the heap of which Klong may use 144 KB.

## What it does with the hardware

| | |
|---|---|
| Lane | uart0, **GP16 TX, GP17 RX**, 115200 8N1, no flow control, FIFOs on, RX pull-up on GP17. Nothing else uses these pins; the SDK's stdio over UART is switched off. |
| Receive | interrupt into a 4096-byte ring (1.4 s of a saturated lane); counters for ring overruns, FIFO overruns, line errors |
| Transmit | the 32-byte FIFO takes what fits; the runtime offers the rest again; no interrupt |
| USB | CDC console (pico_stdio_usb), raw bytes, nothing is dropped except when no host is attached |
| Not touched | RUN, SWD, every other GPIO, the second core, flash (the image has no flash file system: uploads live in RAM and are lost at reset) |
| Node number | learned: the destination of the first unicast frame (to ports 0..15) that arrives; broadcast and supervisor-addressed frames do not name a port. Silent until then. |
| Frames it understands | PING (answered with PONG, the bench node's own format: type 2, source = this node, destination = the sender, same sequence, opaque byte 16), datagram frames (APP_DATA, sub-types 0xD0..0xD3). It sends STATUS (type 3, to the supervisor, once a second, `payload[1]` = 1 alive, `payload[2..3]` uptime seconds). Every other valid frame is ignored silently. |
| Does not do | the bench OS's app loader (sub-types 0xA0..0xB1) and echo service (0xE1/0xE2): this image replaces that OS |
| Reset | `.x(0)` restarts the chip through the watchdog |
| Memory | interpreter stack 48 KB static (the stack pointer is moved there at start; Klong is given 42 KB), heap: newlib `malloc`, limit 144 KB, 16 KB held back for errors; lane buffers (7.8 KB for 6 reassembly slots, plus up to 16 KB of stream buffers per open connection) come from that heap |

Ctrl-T (0x14) on the USB console prints the counters instead of reaching Klong: frames,
datagram outcomes, UART overruns, heap use.

## The console

* **USB CDC**: open the Pico's USB port with a terminal at any speed. The banner
  `Klong 0.0.1 (pico-bench, int64 float64)` is printed at start, before a host can be there,
  so press Enter for the prompt `klong> `. `native/tools/kgupload.py PORT ver` works; `put` and `run` work
  (the file system is in RAM, in the Klong heap, and is lost at reset).
* **Lane (a proposal for the bench owner, not part of the bench protocol)**: datagram channel
  **1** carries the same console byte stream.
  * Any message on channel 1 to the node attaches its sender as the console's peer (the last
    sender wins) and its bytes are typed into the console. Messages are text; a line ends with
    `\r`.
  * The console's output goes to USB and, while a peer is attached, to that peer in messages
    of up to 240 bytes on channel 1, without ACK, best effort, in order. 2 KB are buffered;
    more is dropped (`console_dropped` in the counters).
  * A message of the single byte `0x04` detaches the peer.
  * `tools/lane_frames.py send 0 '1+2\r'` prints the frames a supervisor puts on its lane to
    type `1+2` on node 0.

## Hardware test plan, two Picos

For the bench owner (flashing, the supervisor and the restore procedure are theirs: all four
Picos are held in reset while one is flashed, about 30 s). Each step says what to do and what
success looks like. Expected times are the simulator's (`docs/native/pico-bench.md`); a Pico is
slower than the model, so allow twice as much.

0. **Before anything**: `cd native && cargo test --release --workspace` passes, and
   `./build.sh` made `build/klong_pico_bench.uf2`. Decide which two Picos: say nodes 0 and 1.
   The other two keep their firmware or stay in reset; this image ignores their frames and
   they ignore ours (their echo service answers only sub-type 0xE1).
1. **Flash** `build/klong_pico_bench.uf2` to the two Picos through the supervisor
   (`POST /bmc/flash?target=N`) or, for a Pico on the bench with a cable, BOOTSEL.
2. **Numbered and alive.** Within about 2 s the supervisor's `status` shows the nodes online.
   Success: its PONG and STATUS counters for node 0 and 1 rise by one per second, as with the
   original firmware. (If a node never comes online: it is not receiving PINGs, or its reply
   is dropped; with a cable press Ctrl-T on its console: `node -1` means no frame was
   understood, a rising `bad crc` or `noise bytes skipped` means the lane is noisy, `ring
   overruns` that the interrupt is starved.)
3. **Console on USB** (with a cable on one Pico): Enter gives `klong> `; `1+2` gives `3`;
   `.ver()` lists `ipc` and `timers`; `{1}{x+1}:~0` runs until Ctrl-C. Ctrl-T prints
   `lane: node 0 ...`.
4. **Console on the lane** (no cable): the supervisor-side tool puts the frames of
   `1+2\r` on its lane (destination node 0, channel 1). Success: frames from node 0 to 16
   (datagram START, DATA, END, channel 1) whose message is `1+2\r\n3\r\nklong> `, within a
   second. Decode them with `tools/lane_frames.py decode DUMP.hex`. Send `\x04` to detach.
5. **IPC between the Picos**, typed on each node's console (USB, or the lane console):
   * node 1: `sq::{x*x};.srv("node:")` answers `1`
   * node 0: `h::.cli("node:1");h(:sq,,7)` answers `49` (the first call opens the connection:
     about 30 ms in the model); `t::.pc();h(:sq,,8);.pc()-t` gives the round trip, expected
     about 0.03 (seconds).
   * a message over 1024 bytes: `big::!600;(h(:idf,,big))~big` after `idf::{x}` on node 1
     answers `1` after about 4 s (4.8 KB each way at about 2.6 KB/s).
   * `h(:nosuch,,1)` is a `name` error that names `node:1` in its hint.
6. **The management library across the nodes**: node 0: `.l("mgmt");mgmt.add("n1";"node:1");
   mgmt.all([:sq]@0;[7])` answers `:{["n1" 49]}`; `mgmt.start(5)` then, 12 s later,
   `{(x@1)?"state"}'mgmt.status()` is `["up"]` while the supervisor's pings go by.
7. **A reset in the middle**: node 0 starts `h(:idf,,big)` with a 6 KB `big`; the supervisor
   does `reset 1` meanwhile. Success: the call ends within about 2 s with a `system` error
   (`connection lost (connection reset by the peer)` or `call timed out`), node 0 keeps
   running; when node 1 is numbered again (about 1 s) and `.srv` is typed there again,
   `h::.cli("node:1");h(:sq,,3)` on node 0 works. Node 1 has lost its functions.
8. **Fan-in**, needs a third Pico (or the supervisor sending): three nodes answer a large
   result to one node at once; the router's 256-frame queue may overflow and the answers
   still arrive, later. Success: correct values, no crash; Ctrl-T on the receiving node shows
   `resent` and `broken` counts above zero is fine.
9. **Record**: round-trip time, the throughput of step 5's big call, `ring overruns`,
   `fifo overruns`, `line errors`, `bad crc` (all expected 0 on a clean lane), and the heap
   peak from Ctrl-T or `.mem()`.
10. **Restore**: the bench owner flashes the original `bench_pico_node.uf2` back through the
    supervisor and runs their own checks. This image keeps nothing in flash.

Success from the supervisor's side in one line: the nodes stay online (PONG and STATUS every
second) throughout steps 4 to 8, and the datagram frames of step 4 and the IPC traffic of
steps 5 to 7 are the only APP_DATA traffic with sub-types 0xD0..0xD3.

## Not verified without hardware

* That the image boots: the stack switch (`msr msp` in `main.c`), the interrupt, the USB CDC
  and stdio set-up, the clock; all compiled and linked only.
* That 48 KB of stack is right: the interpreter's use per level of Klong recursion on a
  Cortex-M0+ was not measured (about 0.8 KB on 64-bit hosts); `f::{f(x+1)};f(0)` must give a
  `range` error, not a crash.
* UART behaviour at speed: the ring and the FIFOs at 720 frames/s while the interpreter
  evaluates; transmit when the FIFO is full.
* The speed of soft-float `f64` and 64-bit arithmetic on the M0+, hence the real latency of a
  call (the simulator charges 0.2 ms per 4096 evaluation steps).
* The supervisor's behaviour with this image: that PONG and STATUS in the format above are what
  it expects (they follow `software/worker/runtime.h` of the bench sources).
