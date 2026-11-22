# Two Raspberry Pi Picos calling each other

Two Raspberry Pi Pico boards (RP2040, 264 KiB of RAM) ran the native Klong interpreter and called each other over IPC. For other targets, see the [board status table](index.md#status-what-has-and-has-not-been-proven).

## The setup

Each Pico has one UART lane on GP16 (TX) and GP17 (RX) of `uart0`, configured at 4.5 Mbaud, 8N1, with no flow control. The two lanes are joined through a switch that forwards fixed 16-byte frames, each carrying up to 4 bytes of payload. A supervisor board numbers the nodes and reads their status. The framing permits at most 112.5 kB/s of payload before acknowledgements and higher-level protocol overhead.

The two-Pico result used a serial switch of 16-byte frames that is not part of this repository. Its frame format is in `native/boards/pico-bench/README.md` and `klong-sim` models it. The firmware, the simulator and the host tools are in the repository. Without the switch you can run the same runtime and self-test in the simulator and drive a node from your own frame source, but you cannot reproduce the measured two-board numbers. To set up a link of your own:

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

The result is `build/klong_pico_bench.uf2` in that directory. The two RP2040 boards used the same image. `native/boards/pico-bench/README.md` lists the build settings and the pins.

At boot every node serves. A node that is not number 0 connects to the nearest node below it that answers and runs a self-test: a call, an asynchronous call, a 4 KB echo, and the management library calling the peer. The verdict is reported to the supervisor as text and as status flags.

## RC1 on two RP2040 boards

The November 20, 2026 run used a sealed KlongPy 2.0.0rc1 build (source archive SHA-256 beginning `547dd7b9`). Both boards booted UF2 image SHA-256 `2e89418263b298b8644725a0f238ead6d8c182778f824dfa97bf8abe29bc09f0` and returned `2.0.0-rc.1` from `.ver()` over the lane. The raw firmware binary was 809,252 bytes.

| Check | RC1 observation |
|---|---|
| Paired boot | Node 2 served five calls. Node 3 passed in 615 ms, including a 95 ms echo of 500 integers. A controlled reboot passed in 616 ms. |
| IPC | A plain remote evaluation returned 42 in both directions, with about 5 ms per call after the connection opened. Three management calls returned 2 and took 30, 15 and 14 ms. |
| Compiled C | Moving average and PID agreed with the interpreter on both boards. FIR agreed with all 241 reference values in flash. See [On a Pico](../embedded.md#on-a-pico) for the workloads and timings. |
| Interrupt and recovery | Each node reported an interrupt on the lane and then evaluated `3+4` to 7. |
| Recursion and recovery | For the tested recursive function, both boards returned 113 at depth argument 113 and reported a `range` error at 114. Both evaluated `1+2` after a deeper call failed. |
| Largest integer vector after reset | After 16 seconds idle, `#!5951` on the self-test client (node 3) and `#!6826` on the serving node (node 2) succeeded. The next size failed on each. |
| Largest integer vector with management loaded | On node 2, after one successful management call and two seconds idle, `#!5060` succeeded and `#!5061` returned a `memory` error. |
| Largest integer vector after mixed use | After compiled examples, recursion and IPC, node 2 reached 3,386 and node 3 reached 4,473. |

The physical run exercised the UART lane console. Direct Pico USB CDC prompt and Ctrl-T counter checks were deferred, and this run provides no direct USB upload result. The current checkout's stack-budget fix and boot self-test memory cleanup passed host regressions and built together for RP2040. The follow-up image has not been flashed.

<a id="what-works"></a>

## Earlier measurements

| Measure | Result |
|---|---|
| Image | About 650 KB for the November 17 measurements (430 KB with `KLONG_OPT=z`). The November 18 image with compiled examples, string functions and static-profile declarations used 734 KB of flash. |
| Boot self-test, both boards started together | 0.52 s |
| Largest integer vector on the node that ran the self-test | 5,500 |
| Largest integer vector on the serving node, management library loaded | 5,500 |
| Loading the management library (`.l("mgmt")`) | 0.16 s; about 10 KB stay in use once it has been used |
| A call to the other node through the library, first | 26 ms |
| The same call, repeated in one program | 11 ms for one call; ten calls in one request take 84 ms |
| A plain `.cli` call, no library | 3.4 ms |
| A 4 KB echo | 94 ms |
| `fib(18)`, 8,361 calls | 0.39 s |
| Recursion depth | 105 levels; the 106th is a `range` error and the node goes on |
| Compiled C on both boards (`.c.check()`, November 18) | Moving average and PID were about 100 times faster than their interpreted versions. FIR was checked against reference values from a desktop. See [On a Pico](../embedded.md#on-a-pico). |

Conditions: two boards, 2026-11-17, through the console client, one firmware image (Pico SDK 2.3.1, Arm GNU 15.3.1); the plain `.cli` call was measured on an earlier firmware image. Each vector size is the largest of the sizes tried, in steps of 300 to 500 elements.

## Memory

The default stack reservation is 112 KiB. The receive and transmit rings use 16 KiB and 8 KiB respectively, and the evaluator's hot functions also run from RAM. The RC1 linker map showed 56,344 bytes for static data and code in RAM, 99,304 bytes between the static area and the stack, and 114,688 bytes for the stack. The runtime's allocation limit reported by `.mem()` was 93,160 bytes.

The allocator tends to place small blocks near the top and vectors and parse trees near the bottom. A large vector needs a contiguous block and enough headroom under the runtime's allocation limit. After the controlled management call on RC1, node 2 had 50,368 bytes free in its pool, with a largest free block of 40,480 bytes. The limit also reserves room for the evaluator to report an allocation error.

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

* Available contiguous memory depends on what the node has loaded and run. Use `.heap(0)` and `.mem()` to inspect the current pool and the runtime's counted usage.
* The boot self-test clears its `mgmt` binding. Run `.l("mgmt")` before using `mgmt.add` from the console.
* The stack budget depends on the build and the call path. The recursion results above describe the tested function.
* Sustained bidirectional lane traffic still needs measurement. The USB console's Ctrl-T report includes `rx ring overruns` for diagnosing receive losses.
* The RP2040 performs floating-point arithmetic in software. The compiled workloads above use single precision, and this interpreter build uses double precision.

## Run the host tools

Everything below runs without a Pico:

    cd native && cargo build --release          # klong-bench, klong-sim
    python native/tools/bench_bridge_sim.py --port 18751 --nodes 2 &
    python native/tools/klong_bench.py --host 127.0.0.1:18751 status
    python native/tools/klong_bench.py --host 127.0.0.1:18751 eval --node 1 'sq::{x*x};sq(12)'

`tests/test_native_pico_pair.py`, `tests/test_native_bench_cli.py` and `cargo test --release -p klong-sim` run the pair, the client and the simulator.
