# Two Raspberry Pi Picos calling each other

Two Raspberry Pi Pico boards (RP2040, 264 KB of RAM) run the native Klong interpreter and call each other over IPC. This page is for someone who wants to know what a runtime of this size can do on a network of such boards, or who wants to build the firmware and try it. For what has been run where, see the [status table](index.md#status-what-has-and-has-not-been-proven).

## The setup

Each Pico has one lane: a UART link, GP16 (TX) and GP17 (RX) of `uart0`, 8N1, no flow control, at 4.5 Mbaud. The two lanes are joined through a frame switch that forwards fixed 16-byte frames, 4 bytes of payload each, and a supervisor board numbers the nodes and reads their status. The lane carries 112 KB/s of data at best.

The switch and the supervisor, which provides a web console, are part of a test bench that is not published, so the two-Pico result used a serial frame switch that you cannot build from this repository. The firmware, the simulator and the host tools are in the repository. To reproduce the setup you do not need them:

* The `node` transport needs any byte link between two UARTs and something that numbers the nodes. A node learns its number from the destination of the first frame it receives.
* The simulator in `native/klong-sim` models the switch, with a clock for each lane, and runs the same runtime and the same self-test. `klong-bench` runs it from a script.
* `native/tools/klong_bench.py` is a console client for the nodes. `native/tools/bench_bridge_sim.py` serves the same interface in front of the simulator, so the client runs without any hardware.
* `tests/test_native_pico_pair.py` runs two nodes of the firmware's own allocator behind a model of the switch.

IPC over the lane is the ordinary kdb+ protocol. A reliability layer (handshake, sequence numbers, go-back-N retransmission) sits between it and the datagrams that carry it, as for any serial link. An address is `node:N`: `.cli("node:2")` connects to node 2 and `.srv("node:")` accepts connections from any node. See [IPC and Timers](ipc.md) for the functions.

Two nodes in the simulator, one serving and one calling. The time is virtual and includes the connection and 60 ms in which the nodes learn their numbers:

<!-- check-docs: requires-file native/target/release/klong-bench; path native/target/release -->
```console
$ klong-bench 2 --baud 4500000 <<'EOF'
> 1> sq::{x*x};.srv("node:")
> 0> h::.cli("node:1");h(:sq,,7)
> time
> EOF
1: 1
0: 49
t = 0.066 s
```

## Build the firmware

You need stable Rust with the `thumbv6m-none-eabi` target, CMake, Ninja, the Pico SDK 2.x and the Arm GNU toolchain. Then:

    native/boards/pico-bench/build.sh

The result is `build/klong_pico_bench.uf2` in that directory. One image runs on every Pico. `native/boards/pico-bench/README.md` lists the switches (`KLONG_SELFTEST=0`, `LANE_BAUD=115200`, `KLONG_OPT=z`, `KLONG_STACK_KB`) and the pins.

At boot every node serves. A node that is not number 0 connects to the nearest node below it that answers and runs a self-test: a call, an asynchronous call, a 4 KB echo, and the management library calling the peer. The verdict is reported to the supervisor as text and as status flags.

## What works

| Measure | Result |
|---|---|
| Image | about 650 KB of flash (430 KB with `KLONG_OPT=z`) |
| Boot self-test, both boards started together | 0.52 s |
| Largest integer vector on the node that ran the self-test | 5,500 |
| Largest integer vector on the serving node, management library loaded | 5,500 |
| Loading the management library (`.l("mgmt")`) | 0.16 s; about 10 KB stay in use once it has been used |
| A call to the other node through the library, first | 26 ms |
| The same call, repeated in one program | 11 ms (ten calls in 84 ms) |
| A plain `.cli` call, no library | 3.4 ms |
| A 4 KB echo | 94 ms (87 KB/s of the lane's 112) |
| `fib(18)`, 8,361 calls | 0.39 s |
| Recursion depth | 105 levels; the 106th is a `range` error and the node goes on |

Conditions: two boards, 2026-11-17, through the console client, one firmware image (Pico SDK 2.3.1, Arm GNU 15.3.1); the plain `.cli` call was measured on an earlier build. Each vector size is the largest of the sizes tried, in steps of 300 to 500 elements.

## Memory

The RP2040's RAM is one address range. The firmware gives the top 112 KB to the interpreter's stack, 16 KB to the receive ring, 8 KB to the transmit ring and 16 KB to the evaluator's hot functions, which run from RAM. The rest, about 98 KB, is the heap. A level of recursion costs about 870 bytes of stack.

The runtime manages the heap itself. It keeps one free list, sorted by address, and puts small blocks at the top and vectors and parse trees at the bottom, so a large vector finds free space whole. A request that cannot be met is a `memory` error and not a crash.

Functions from a loaded file are kept as text. Calling one parses it again from that text and keeps the tree in a cache of 24 entries. A program that calls many functions pays a parse for each, which is why the first call through the library is slower than the repeated ones.

`.heap(mode)` reads the heap back. It is a diagnostic in the build by default; `KLONG_HEAP_DIAG=0 ./build.sh` leaves it out.

| Call | Returns |
|---|---|
| `.heap(0)` | 32 integers: pool size, bytes in use, the peak, free bytes, the largest free chunk, counts of free chunks and requests refused, and the addresses of the pool and of the stack |
| `.heap(1)`, `.heap(2)`, ... | pages of runs of the pool in address order: offset, size, free or used, kinds of the blocks |
| `.heap(100)`, ... | pages of live blocks of 512 bytes or more: offset, size, kind |
| `.heap(200)` | the cached small blocks, by size class |
| `.heap(300)`, `.heap(301)` | restart the high-water mark; read the heap near it |

`.mem()` gives `[in use, peak, limit]` as the runtime counts them.

## Known limits

* About a fifth of the free memory cannot be used for one large vector. Tables that grow during a load leave holes in the heap.
* A node is deaf for about 160 ms while it looks for a peer at start. It asks each node below it in turn and waits 80 ms for each, and then again every 10 s while it has none.
* After the self-test drops the management library, its names are integers, not undefined, until `.l("mgmt")` runs again. Calling `mgmt.add` in between is a `type error`.
* Recursion stops at about 105 levels.
* The receive ring under sustained traffic in both directions at once has not been measured on the boards. `rx ring overruns` in the console's Ctrl-T output counts losses.
* Floating-point arithmetic is in software on the RP2040 and has been measured only through integer code (`fib`).
* The USB console has not been tested.

## Run the host tools

Everything below runs without a Pico:

    cd native && cargo build --release          # klong-bench, klong-sim
    python native/tools/bench_bridge_sim.py --port 18751 --nodes 2 &
    python native/tools/klong_bench.py --host 127.0.0.1:18751 status
    python native/tools/klong_bench.py --host 127.0.0.1:18751 eval --node 1 'sq::{x*x};sq(12)'

`tests/test_native_pico_pair.py`, `tests/test_native_bench_cli.py` and `cargo test --release -p klong-sim` run the pair, the client and the simulator.
