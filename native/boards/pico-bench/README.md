# Klong on the 4-Pico bench

Status: **built, not yet run on hardware.** Nothing here has been flashed to a Pico or seen a
lane. Everything below the C shell (`main.c`) was run on the host and in the simulator
(`klong-sim`, with a model of the bench's switch); `main.c` and the image were only compiled
and linked. The design and the simulator results are in `docs/native/pico-bench.md`.

A Raspberry Pi Pico (RP2040) as one node of the bench: it learns its node number from its
lane, answers the supervisor's PING and sends STATUS, carries messages of up to 1024 bytes
over the lane's 16-byte frames, runs Klong's kdb+ IPC between nodes (`.cli("node:1")`,
`.srv("node:")`, the management library) over those messages, has the Klong console on USB
CDC and, as a proposal, on datagram channel 1 of the lane, and, in the image as built, runs a
**boot self-test** and reports it to the supervisor. One image is flashed to every Pico.

The lane runs at **4.5 Mbaud** (a frame is 36 us, 28,125 frames/s, 112 KB/s of data). The
build also works at 115200 (`LANE_BAUD=115200 ./build.sh`).

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
`build/klong_pico_bench.uf2` (and `.elf`, `.bin`). Switches, all environment variables of
`build.sh`: `KLONG_SELFTEST=0` leaves the boot self-test out, `LANE_BAUD=115200` builds for the
slow lane, `KLONG_OPT=z` makes the smaller image.

Measured on 2026-11-16 (SDK 2.3.1, GCC 15.3.1): `.bin` 618,300 bytes (30% of the 2 MB flash;
the `.uf2` is 1,236,992 bytes because of its 256-byte blocks); with `KLONG_OPT=z` about 430,000.
Static RAM: `.data` 9.3 KB, `.bss` 93.7 KB (of which the interpreter's 48 KB stack, the 32 KB
receive ring and the 8 KB transmit ring), 159 KB left for the heap, of which Klong may use 112 KB.

## What it does with the hardware

| | |
|---|---|
| Lane | uart0, **GP16 TX, GP17 RX**, `uart_init(uart0, 4500000)` (4.505 Mbaud at the 125 MHz `clk_peri`, which is kept; the router tolerates 3%), 8N1, no flow control, FIFOs on, RX pull-up on GP17. Nothing else uses these pins; the SDK's stdio over UART is switched off. |
| Receive | the UART interrupt (highest priority; FIFO level 8 bytes plus the receive timeout) moves bytes into a **32 KB ring**: 2048 frames, 73 ms of a lane that never pauses. The 32-byte FIFO alone fills in 71 us. Counters: ring overruns, FIFO overruns, line errors. |
| Transmit | the runtime copies frames into an **8 KB ring** and returns; the transmit interrupt (FIFO half empty) feeds the wire. At most 12 ms of wire time (5.4 KB at 4.5 Mbaud, 128 bytes at 115200) may wait in it; the runtime offers the rest at its next turn, so the interpreter never waits for the wire and an ACK is not stuck behind a full window. A full ring is counted (`tx ring full`). |
| USB | CDC console (pico_stdio_usb), raw bytes, nothing is dropped except when no host is attached |
| Not touched | RUN, SWD, every other GPIO, the second core, flash (the image has no flash file system: uploads live in RAM and are lost at reset) |
| Node number | learned: the destination of the first unicast frame (to ports 0..15) that arrives; broadcast and supervisor-addressed frames do not name a port. Silent until then. |
| Frames it understands | PING, answered with PONG (type 2, source = this node, destination = the sender, **the same sequence, length 0**, opaque byte 16); datagram frames (APP_DATA, sub-types 0xD0..0xD3). It sends STATUS (type 3, to 16, **once a second**, length 4: `payload[0]` load 0..100 (always 0), `payload[1]` flags, `payload[2..3]` uptime seconds u16 LE, monotonic, saturating at 65535). Every other valid frame is ignored silently, in particular the deployer's APP_DATA sub-types 0xA0..0xB1 and 0xE1/0xE2 (tested with floods of them between datagram frames). |
| Does not do | the bench OS's app loader (sub-types 0xA0..0xB1) and echo service (0xE1/0xE2): this image replaces that OS |
| Reset | `.x(0)` restarts the chip through the watchdog |
| Memory | interpreter stack 48 KB static (the stack pointer is moved there at start; Klong is given 42 KB), heap: newlib `malloc`, limit 112 KB, 16 KB held back for errors; lane buffers (7.8 KB for 6 reassembly slots, plus up to 16 KB of stream buffers per open connection) come from that heap; the 112 KB limit leaves 47 KB for allocator overhead |

Ctrl-T (0x14) on the USB console prints the counters instead of reaching Klong: frames,
datagram outcomes, UART overruns (`rx ring overruns` is the one that says the main loop was away
too long), heap use, the self-test verdict.

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

## The boot self-test

Built in by default; one image for every Pico. After boot, every node serves
(`.srv("node:")`, with two functions that count the calls they answer, named for the node:
`stping<N>` and `stecho<N>`). A node that is not number 0 then plays client against the nodes
**below** it, nearest first: it tries `.cli("node:K")` for K = N-1, ..., 0 and the first that
answers is its peer (a node that runs other firmware does not answer; that costs the handshake
timeout, 3 s at 4.5 Mbaud, 10 s at 115200). So of two flashed Picos the lower one serves and the
higher one connects, and neither needs to know the other's number. Against the peer the client:

1. `sync`: a synchronous call (`h(:stping<K>,,7)` gives 7);
2. `async`: an asynchronous call, then a synchronous one, then waits for the asynchronous answer;
3. `echo`: 500 integers (4 KB) there and back, compared;
4. closes that connection, then `mgmt`: the management library calls the peer (`.l("mgmt")`; skipped if the library is not in the image).

The clock is the board's (`time_us_64`). The verdict goes to address 16, datagram channel 1, as
text, first at the end of the test and then every 3 seconds, and as the flags byte of STATUS:

| State | STATUS flags | Report text (channel 1, to 16) |
|---|---|---|
| alive, no self-test (an image built with `KLONG_SELFTEST=0`) | `0x01` | none |
| serving, looking for a peer, or running | **`0x21`** | `klong selftest serving node=N calls=C` or `klong selftest running node=N calls=C` |
| passed | **`0x41`** | `klong selftest pass 94ms node=N peer=K sync=1ms echo=79ms calls=C` |
| failed | **`0x81`** | `klong selftest fail STEP REASON node=N peer=K`, STEP one of `sync async echo mgmt`, REASON the error's kind and message |

The number after `pass` is the time of the four steps in milliseconds. `calls` is how many
calls this node answered as a server. A node that is not number 0 and finds nobody below it
reports `serving` (it may be the lowest board with the image, so that is not a failure) and
looks again every 10 seconds; a failed test is run again after 15 seconds; a pass stays. The
bench's own OS uses STATUS flag bits 1 and 2 for its apps, so these values (bits 5 to 7) cannot
be mistaken for it. The report's channel is also the console's, so a supervisor that reads
channel 1 sees these lines among any console text.

## Hardware test plan, two Picos

For the bench owner (flashing, the supervisor and the restore procedure are theirs: all four
Picos are held in reset while one is flashed, about 80 to 90 s for this 618 KB image).
Times are from the simulator's model at 4.5 Mbaud, which charges 10 us of processor per frame
received and 5 us per frame sent (estimates for a Cortex-M0+ at 125 MHz); a real Pico may be
several times slower, so the waits below are generous.

0. **Before anything**: `cd native && cargo test --release --workspace` passes, and
   `./build.sh` made `build/klong_pico_bench.uf2` (check: its `.bin` is 618,300 bytes; the
   lane is 4.5 Mbaud and the self-test is on). Decide which two Picos, say nodes 0 and 1 (any two:
   with nodes 1 and 3, node 3 tries 2, waits 3 s, and uses 1). The other two keep the bench
   firmware or stay in reset; this image ignores their frames and they ignore ours.
1. **Flash the same file** to the two Picos through the supervisor (`POST /bmc/flash?target=N`,
   twice). Each flash holds all four Picos in reset; when the second is done they all start.
2. **Numbered and alive, within 3 s of the last flash finishing.** The supervisor's PING reaches
   each node within 1 s of boot; it answers with PONG, and STATUS follows once a second.
   Success: both nodes online, STATUS flags `0x21`.
3. **The self-test, within 10 s** (the model says 1.2 s after boot at 4.5 Mbaud: 1 s of waiting
   for the lower node to listen, 0.1 s of calls). Success, on the supervisor:
   * the higher node's STATUS flags change from `0x21` to **`0x41`** (the next STATUS after the
     test, so at most 1 s later), and a channel-1 datagram from it reads
     `klong selftest pass <ms>ms node=1 peer=0 ...` (the model: 94 ms; accept up to 2000 ms);
   * the lower node stays at `0x21` and reports `klong selftest serving node=0 calls=5` (5 calls
     answered: sync, async, sync, echo, mgmt), repeated every 3 s.
4. **How long before it is a failure**: if after **30 s** no node has left `0x21`, or a node's flags
   are `0x81`, the test failed. Read the failing step in the `fail` text (`sync` and `async`:
   the connection or the call; `echo`: a 4 KB message did not survive; `mgmt`: the library).
   `0x81` repeats the test every 15 s, so a transient fault turns into `0x41` by itself.
   Still `0x21` with `serving` on both after 30 s: the higher node did not find the lower one
   (it was not numbered, or its frames did not arrive); with a cable, Ctrl-T on both shows
   `node` numbers, `frames rx/tx`, `bad crc`, `rx ring overruns`.
5. **Then, with a cable on one Pico** (USB CDC): Enter gives `klong> `; `1+2` gives `3`;
   `h::.cli("node:0");t::.pc();h(:stping0,,5);.pc()-t` (on the higher node, after the self-test
   closed its connection) gives the round trip in seconds (the model: about 0.001);
   `big::!6000;(h(:stecho0,,big))~big` gives `1` (48 KB each way in about 0.5 s in the model).
   `{1}{x+1}:~0` runs until Ctrl-C while the supervisor keeps seeing PONGs (the interrupt serves
   the lane); Ctrl-T afterwards must show `rx ring overruns 0`.
6. **The console over the lane** (no cable; needs a supervisor-side sender, see
   `tools/lane_frames.py send 0 '1+2\r'` for the frames): channel 1 to node 0 gets
   `1+2\r\n3\r\nklong> ` back.
7. **A reset in the middle**: from the cable, start a long call to the peer; the supervisor does
   `reset N` on the peer. Success: the call ends within about 1 s with a `system` error and the
   node keeps running; when the peer is numbered again (about 1 s) and the self-test runs again
   there, it serves again.
8. **Record**: the self-test's `pass` line, the round trip, `rx ring overruns`, `fifo overruns`,
   `line errors`, `bad crc`, `tx ring full` (all expected 0 on a clean lane), the heap peak.
9. **Restore**: the bench owner flashes the original `bench_pico_node.uf2` back through the
   supervisor and runs their own checks. This image keeps nothing in flash.

## Not verified without hardware

* That the image boots: the stack switch (`msr msp` in `main.c`), the interrupt, the USB CDC
  and stdio set-up, the clock; all compiled and linked only.
* That 48 KB of stack is right: the interpreter's use per level of Klong recursion on a
  Cortex-M0+ was not measured (about 0.8 KB on 64-bit hosts); `f::{f(x+1)};f(0)` must give a
  `range` error, not a crash.
* UART behaviour at 28,125 frames/s: the interrupt keeping up (about 7% of the processor when
  saturated, an estimate), the FIFO never overrunning (71 us to fill), the 32 KB ring while the
  interpreter evaluates, the 12 ms transmit limit, the value of clk_peri and the real baud rate
  (Ctrl-T prints it).
* The speed of soft-float `f64` and 64-bit arithmetic on the M0+, and the processor time per
  frame (the simulator charges 10 us received, 5 us sent, and 0.2 ms per 4096 evaluation steps:
  estimates), hence the real latency of a call and whether the endpoints keep up with the lane.
* The supervisor's behaviour with this image: that PONG and STATUS in the format above are what
  it expects (they follow `software/worker/runtime.h` of the bench sources).
