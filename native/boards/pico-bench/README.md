# Klong on Raspberry Pi Picos

Firmware that runs the native Klong interpreter on a Raspberry Pi Pico (RP2040) as one node of a small network, with IPC between nodes. This README is for someone with two Picos who wants to build the image and flash it. Results, memory design and known limits are in `docs/native/pico-bench.md`; the status of every board is in the table of `docs/native/index.md`.

Each node has one lane, a UART link on GP16 (TX) and GP17 (RX) of `uart0`. The lane carries 16-byte frames, so a node needs a switch or a peer that speaks them. The published two-Pico result used a serial switch of 16-byte frames that is not part of this repository; the frame format is below and `klong-sim` models the switch. Without that hardware you can build and flash the firmware, run the same runtime and self-test in the simulator, and drive a node from any program that writes the frames; you cannot reproduce the measured two-board numbers. `docs/native/pico-bench.md` says what to use instead.

```text
[0..1] A5 5A  [2] 01  [3] type  [4] source  [5] destination  [6..7] sequence, u16 little-endian
[8] sub-type  [9] payload length 0..4  [10..13] payload, unused bytes 0
[14..15] CRC-16/CCITT-FALSE over bytes 2..13, little-endian
```

The types are PING 1, PONG 2, STATUS 3 and APP_DATA 9. Address 16 is the supervisor, the peer that numbers the nodes and reads their STATUS. The node learns its number from the lane, answers PING, sends STATUS once a second and carries messages of up to 1024 bytes. Above that it runs the kdb+ IPC (`.cli("node:1")`, `.srv("node:")`) and the management library. The two RP2040 boards use the same image.

## Requirements

* Stable Rust with the `thumbv6m-none-eabi` target: `rustup +stable target add thumbv6m-none-eabi`.
* CMake and Ninja.
* The Pico SDK 2.x. Set `PICO_SDK_PATH` to it.
* The Arm GNU toolchain (`arm-none-eabi-gcc`). Set `ARM_BIN` to its `bin` directory.
* The default image includes compiled Klong examples. Their generator uses standard-library Python and the desktop `kg` binary (by default `native/target/release/kg`). It runs each example with standalone `kg` and checks the results pinned in its `.expect` file. Set `KLONG` or `KLONG_PYTHON` to select an existing installation. Use `KLONG_C_EXAMPLES=0` to build the interpreter firmware without those examples.

## Build

    native/boards/pico-bench/build.sh

The script builds `libklong_ffi.a` for `thumbv6m-none-eabi` with `--features lane` into `native/target/pico-bench/`, then builds the application out of tree in `native/boards/pico-bench/build/`. The SDK and the toolchain are only read. Nothing is flashed and no port is opened. The output is `build/klong_pico_bench.uf2`, with `.elf` and `.bin`.

| Environment variable | Effect |
|---|---|
| `KLONG_SELFTEST=0` | leaves the boot self-test out |
| `LANE_BAUD=115200` | builds for a slow lane; the default is 4,500,000 |
| `KLONG_OPT=z` | optimizes for code size |
| `KLONG_STACK_KB=96` | reserves 96 KiB of stack, leaving the remaining RAM after static allocations for the heap. The default is 112 KiB. |
| `KLONG_HEAP_DIAG=0` | leaves the `.heap` diagnostic out |

`build.sh` prints the RAM map from the linker's symbols and runs `tools/check_layout.py`, which fails the build when the layout is one the firmware cannot run.

## Flash

Hold BOOTSEL while connecting the RP2040 Pico over USB and copy `build/klong_pico_bench.uf2` to the drive that appears. Do the same for the second Pico with the same file. The Klong filesystem lives in RAM, so uploaded programs are lost at reset. The November 20 RC1 physical check used SWD for flashing and the lane for console commands. [RC1 run results](../../../docs/native/pico-bench.md#rc1-on-two-rp2040-boards)

## What it does with the hardware

| | |
|---|---|
| Lane | `uart0`, GP16 TX, GP17 RX, 8N1, no flow control, FIFOs on, RX pull-up on GP17. The SDK's stdio over UART is switched off. |
| Receive | the UART interrupt moves bytes into a 16 KB ring (1024 frames, 37 ms of a lane that never pauses). Counters: ring overruns, FIFO overruns, line errors. |
| Transmit | the runtime copies frames into an 8 KB ring and returns; the transmit interrupt feeds the wire. At most 12 ms of wire time may wait in it. |
| USB | a CDC console (`pico_stdio_usb`), raw bytes |
| Not touched | RUN, SWD, every other GPIO, the second core, flash |
| Node number | the destination of the first unicast frame that arrives. The node is silent until then. |
| Frames | answers PING with PONG; sends STATUS once a second to address 16 (the supervisor); takes datagram frames (APP_DATA sub-types `0xD0` to `0xD3`); ignores every other valid frame |
| Reset | `.x(0)` restarts the chip through the watchdog |
| Safety net | a HardFault handler, a watchdog of 8 s fed in the idle path, and a canary at the bottom of the stack. The reason for a restart appears in the self-test report as `boot=hardfault`, `boot=stack`, `boot=fatal`, `boot=watchdog` or `boot=reset`. |

Ctrl-T (0x14) on the USB console prints the counters instead of passing it to Klong: frames, datagram outcomes, UART overruns, heap use and the self-test verdict. `rx ring overruns` says the main loop was away too long.

## The console

* **USB.** The firmware enables a CDC console. To check it, open the Pico's USB port in a terminal, press Enter for the `klong> ` prompt and try `1+2`. The upload client provides `native/tools/kgupload.py PORT ver`, `put` and `run`. The RC1 physical run deferred the direct USB prompt and Ctrl-T counter checks. It used the lane for console operations.
* **The lane.** Datagram channel 1 is a console. Input is a datagram with the ACK flag that ends at a line end; output is text sent to the node that last sent input. Channel 2 carries control (`done ID STATUS`, `int`, `hello`) and channel 3 the self-test report. `native/tools/klong_bench.py` is a client; `tools/lane_frames.py send 0 '1+2\n' --ack` prints the frames of an input for any other tool to put on a lane.

## The boot self-test

Every node serves. A node that is not number 0 connects to the nearest node below it that answers, so of two Picos the lower serves and the higher is the client. The client times four steps with the board's clock:

1. `sync`, a synchronous call;
2. `async`, an asynchronous call, then a synchronous one, then the asynchronous answer;
3. `echo`, 500 integers (4 KB) there and back, compared;
4. `mgmt`, the management library calling the peer, on builds that include the library.

The verdict goes to address 16 on channel 3 as text, every 3 seconds, and as the flags byte of STATUS:

| State | STATUS flags | Report text |
|---|---|---|
| alive, no self-test | `0x11` | none |
| serving, looking for a peer, or running | `0x31` | `klong selftest serving node=N calls=C` or `klong selftest running ...` |
| passed | `0x51` | `klong selftest pass 361ms node=N peer=K sync=1ms echo=79ms calls=C async=Ams load=Lms add=Dms call=Mms mgmt=Gms` |
| failed | `0x91` | `klong selftest fail STEP REASON node=N peer=K`, with STEP one of `sync async echo mgmt` |

A node that finds nobody below it reports `serving` and looks again every 10 seconds. A failed test runs again after 15 seconds, and a pass stays.

## Trying it

Without hardware, `cargo test --release --workspace` in `native/` and `python -m unittest tests.test_native_pico_pair` run the pair in the simulator and in a model of the firmware's allocator.

With two flashed RP2040 Picos on a lane that reaches a numbering source, check in this order:

1. Check that both nodes have numbers and return a live version from `.ver()?"version"` on the lane. The supervisor can retain an older startup greeting after a reflash.
2. With the default boot self-test and no extra user requests, look for the higher node's `0x51` pass report and the lower node's `0x31` serving report with five calls. A `0x91` report includes the failed step. If the nodes stay at `0x31`, check numbering and peer availability.
3. From the lane console on each node, open `.cli` to the other node and evaluate `6*7`. Check for 42. Exercise `.l("mgmt")`, `mgmt.add` and `mgmt.call` on the serving node.
4. Test recovery on the lane. Interrupt `{1}{x+1}:~0` with Ctrl-C and then evaluate `3+4`. For recursion, try `f::{:[x;1+f(x-1);0]};f(105)`, sample larger inputs until the guard reports a `range` error, then check that another expression works. Record the firmware and the tested inputs with the result.
5. For a separate direct USB check, try the CDC prompt and upload commands above. Use Ctrl-T to read the UART counters after the lane tests.

Record the `pass` line, the round trip and any collected counters, including `rx ring overruns`, `fifo overruns`, `line errors`, `bad crc` and `tx ring full`. The counters help diagnose transport faults. The RC1 run and earlier measurements are in [Two Raspberry Pi Picos calling each other](../../../docs/native/pico-bench.md).
