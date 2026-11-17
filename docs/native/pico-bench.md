# Native IPC on the 4-Pico Bench

**Status: built, proven in the simulator, and run on two Picos of the bench (2026-11-16).** The boot self-test passes between them, calls cross the switch, and the console answers through the supervisor's bridge. The figures labelled *measured* are from those two boards; the simulator's figures are *virtual time* (the lanes' baud rate exactly, the processor's work fitted to the measurements: see [below](#in-the-simulator)); figures labelled *predicted* are for an image that has not been flashed. What is still open is listed [at the end](#what-the-hardware-has-shown-and-what-is-open).

This page is for someone who wants native Klong nodes to talk to each other, `.cli` to `.srv`, over a packet switch that carries nothing but 16-byte frames. It is the first network of the native runtime that is neither a byte stream nor a serial line, and it shows how little the IPC code had to change to take it.

## The bench

Four Raspberry Pi Picos (RP2040: two Cortex-M0+ cores at 125 MHz, 264 KB of RAM, 2 MB of flash) each have one UART lane to a packet switch (an FPGA router): GP16 TX and GP17 RX of uart0, 8N1, no flow control, **4.5 Mbaud** now (it was built and first tested for 115200, which still works: every figure below is given at both rates). A supervisor (a QT Py ESP32-S3) has a lane too, port 16. The switch forwards **fixed 16-byte frames**: a frame carries 4 bytes of payload. At 115200 baud a lane moves 720 frames a second and **2.9 KB/s of data at best**; at 4.5 Mbaud a frame is 36 us, 28,125 frames a second, **112 KB/s** of data, and a 1024-byte message takes 9.3 ms. It is store-and-forward and best effort: each output port holds 256 frames in a FIFO, and a frame that finds it full is lost. Frames from one port to another stay in order. There is no flow control and no retransmission. The supervisor sends a PING to every node about once a second and expects a PONG and a STATUS, or it marks the node offline. All wired ports have 256-frame queues.

```text
 node 0 ─┐                                        ┌─ node 2
 node 1 ─┼── lane ── [ switch: 16-byte frames ] ──┼─ lane ── ...
 supervisor ┘            256-frame FIFO per port        node 3
```

## Layers

| Layer | Where | What it gives |
|---|---|---|
| frames | `klong-dgram`, `frame` | 16-byte frames in a byte stream: a sliding window that resyncs on `A5 5A 01` and a good CRC-16 |
| datagrams | `klong-dgram`, `dgram` | messages of 1 to 1024 bytes in START, DATA, END and ACK frames; per-source reassembly; duplicate suppression; resends |
| lane | `klong-dgram`, `lane` | a node on its lane: learns its number, answers PING, sends STATUS |
| adapter | `klong-rt`, `node` | the `node` transport: `.cli("node:2")`, `.srv("node:")`, a console on channel 1 |
| reliability | `klong-ipc`, `rel` | the layer every non-stream link already uses: handshake, fragmentation, go-back-N retransmission, reset |
| IPC | `klong-ipc` / `klong-rt` | the kdb+ protocol, unchanged |

`klong-dgram` is `no_std`, allocates nothing and has no `unsafe`; its buffers are const generics.

### Frames and datagrams

The frame is `A5 5A 01`, type, source, destination, sequence (u16 LE), sub-type, payload length 0 to 4, four payload bytes with the unused ones zero, and CRC-16/CCITT-FALSE (check value `0x29B1` of `123456789`) of bytes 2 to 13. A frame is valid only if the type is one the switch knows, the addresses are ones it routes, and the padding is zero. The parser keeps a 16-byte window and, when a frame fails, drops one byte, not sixteen, so a frame that starts inside a torn one is still found.

A datagram is an APP_DATA frame sequence with sub-types `0xD0` to `0xD3`:

| Frame | Sub-type | Sequence | Payload |
|---|---|---|---|
| START | `0xD0` | message id (u16 per sender, +1 per message, wraps) | length u16 LE (1 to 1024), flags (bit 0: ACK wanted), channel |
| DATA | `0xD1` | fragment index | 1 to 4 bytes of the message, 4 for all but the last |
| END | `0xD2` | message id | CRC-32 of the whole message (zlib; `0xCBF43926` for `123456789`) |
| ACK | `0xD3` | message id | 0 delivered (or duplicate), 1 broken, 2 refused |

The sender sends START, the DATA frames in order, END, back to back. 1024 bytes are 258 frames, 4128 wire bytes, 0.36 s at 115200 and 9.3 ms at 4.5 Mbaud. With ACK asked for it waits 300 ms plus 1.5 ms per frame for the ACK and resends the whole message with the *same* id, three times at most. The receiver keeps one reassembly state per source (17 possible: nodes 0 to 15 and the supervisor), abandons a message when a START arrives in the middle of it, marks it broken on a fragment out of order or past the length, delivers it at END only if the id, the fragment count and the CRC-32 agree, remembers the id of the last message it delivered from each source and acknowledges a repeat with 0 without delivering it again. Channels are a byte in START: 0 for IPC.

**RAM** is bounded by the type: `Lane<MAX, SLOTS>` is `(SLOTS + 1) * MAX` bytes of buffers plus about 650 bytes (measured: 19,064 bytes for 1024 and 17 slots, 7,792 for 1024 and 6, 1,904 for 256 and 4). A reassembly buffer is handed to a source when it starts a message and taken back when the message ends, so `SLOTS` is how many sources may be in the middle of a message at once; one more is refused (ACK 2, or silence without ACK). The Pico image uses 6.

Tests (`native/klong-dgram/tests`): the check values; the frame layout and every way a frame is invalid; the parser against noise, a torn frame, a lost byte and sync bytes inside a payload; each rule of "Receiving"; the sender's frames, its timeout (300 + 387 ms for 1024 bytes), its three resends and the ACK codes; messages of 1, 3, 4, 5, 1023 and 1024 bytes and 1025 refused; id wrap-around; and, with a seeded generator, a network of lanes with random noise between frames, dropped, duplicated and corrupted frames, START in the middle of a message, three senders into one port with the queue overflowing, and 300 runs of arbitrarily mutated streams that may deliver only what was sent and never twice in a row. A frame sequence printed by the supervisor-side tool `native/boards/pico-bench/tools/lane_frames.py` is delivered by the Rust code as the tool says.

One departure from the datagram spec's wording on PING: the layer itself ignores it, as specified, but the *lane* around it answers PING with PONG and sends STATUS once a second, because a node that does not is marked offline (`KLONG_LANE_NO_DUTIES` turns that off).

## IPC over datagrams

Native IPC runs over links. A link is a byte stream (TCP) or a packet link, which the runtime puts a reliability layer on (`rel`: 1 header byte, SYN handshake, 6-bit sequence numbers, go-back-N with a window of 16, RST, keepalive). A serial line is a packet link. So is the switch, once something cuts the lane into packets: that is the whole adapter (`klong-rt/src/node.rs`, about 530 lines).

* **Address.** `.cli("node:2")` opens a connection to node 2. `"node:"` is the name of the transport and `2` its address, the form `"name:address"` that [IPC and Timers](ipc.md#the-functions) already gives other transports. `node:16` is the supervisor, `node:n2` is the same as `node:2`. `.srv("node:")` accepts connections from any node. A node's own number is what the first PING told it; until then `.cli` says `address unreachable`.
* **Handshake.** The first packet the client sends is the reliability layer's SYN. The server accepts a connection when the first packet of one arrives, answers with its own SYN, and the kdb+ handshake (the login string and the capability byte) runs over the stream as on TCP. A connection to a node that does not listen is answered with a reset and fails at once: `closed the connection during the handshake`, the same words as a refused login. A user name and password in the address are not carried (`node:1:u:p` is a bad address), so `.srv.auth` sees empty ones.
* **Messages larger than 1024 bytes.** An IPC message goes into the reliability layer's byte stream, which cuts it into packets of 126 bytes. One packet is one datagram of 128 bytes (2 header bytes of the adapter, 1 of the reliability layer, 125 of IPC), 34 frames, 47 ms on the lane. So a message of any size crosses the link, and the kdb+ bytes inside are exactly the bytes TCP would carry. Nothing in the datagram layer's limit shows through.
* **Timeouts, per lane rate** (`NodeConfig::for_baud`, asked for through a new, optional `PacketLink::tuning()`; no other transport changes). At 115200 baud: first retransmission 500 ms and doubling, 6 retries (31 s), 10 s for the handshake. At 4.5 Mbaud the lane is 39 times faster and what limits is how long a peer's main loop may be away (an evaluation between two `idle` calls: tens of milliseconds): first retransmission 100 ms, 7 retries (25 s), 3 s for the handshake. The call timeout is the usual 30 s, and `.clit` changes it. The acknowledgement delay of the reliability layer is a quarter of the first timeout.
* **Both ways at once.** Two bytes in front of every packet name the sender's end (client or server) and the connection's *epoch*, a 15-bit number the client chooses when it connects. So two nodes can be client and server of each other at the same time, and a late packet of a connection that is gone cannot be taken for one of the new connection. A node has at most one connection of each kind to a given peer: a second `.cli("node:1")` while one is open is `address in use`.
* **A peer that resets.** The new instance has no connection. A packet for a connection it does not have is answered with a reset, and the caller sees `connection lost (connection reset by the peer)`, the pending call fails, and the connection is closed for good. A peer that restarts and connects again has a new epoch, and the server replaces the old connection with the new one. A peer that is simply gone is found by the retry budget (about 30 s) or the call timeout.
* **The console.** Datagram channel 1 carries the Klong console's byte stream between a node and the node or the supervisor that last wrote to it. This is a proposal, not part of the bench protocol; the firmware's `README.md` has it.

### Why this and not the alternatives

* **Datagram ACKs instead of the reliability layer.** The datagram layer has its own ACK and resend. Using it per IPC chunk would make the sender wait for an ACK before the next datagram (stop and wait), which by estimate (not measured) costs a quarter of the throughput, and it still would not give the handshake, the ordering across datagrams or a reset. So the adapter sends without ACK and lets `rel` recover.
* **An adapter header with its own fragmentation.** Rejected: it would repeat what `rel` does for any packet link, and be a third copy of the logic beside the serial link's.
* **Refusing messages over 1024 bytes.** Rejected: the management library moves source files and results of any size, and `.cli` users do not expect a limit that TCP has not.
* **A larger packet.** Measured with the simulator at 115200 baud: with 0.5% of frames lost, three 2.4 KB echoes take 13 s with packets of 254 bytes and 12 s with 126 (500 ms first timeout), 10 s with 62; with 2% lost, 104 s, 69 s and 36 s; without loss 5.3 s, 5.55 s and 6.0 s. At 4.5 Mbaud (48 KB moved, 0.2% of frames lost, first timeout 100 ms): 7.2 s with 126-byte packets, 8.8 s with 254, 10 s with 510 and 16 s with 1022, and 440 to 470 ms for every size without loss. A lost frame costs a whole packet and a timeout, so the small packet wins whenever frames are lost and costs almost nothing when they are not; 126 is the choice for both rates. The bench only loses frames when a queue overflows, so a larger packet may be right later: `NodeConfig::link_mtu`.
* **Datagrams per connection.** A connection does not own any datagram state: the lane has one sender (one message at a time, a 1 KB buffer) and one reassembly pool. `PacketLink::send` answers "busy" while the previous message is still leaving, which paces the connections to the lane's speed.

## In the simulator

`klong-sim` has a model of the switch (`switch.rs`): ports for N nodes and the supervisor, a byte timeline per lane at 115200 baud (a frame takes 1.389 ms each way), store and forward, a 256-frame FIFO per output port that drops what it cannot hold, the supervisor's PING every second, and optional random loss. `bench.rs` puts N boards (`klong-rt` with the `node` transport) behind it in one thread, on one virtual clock; while one board waits, the others run. `klong-bench` is that as a program with a tiny script language (a line `K> CODE` runs CODE on node K), and what follows is run by the documentation checker. `--baud N` sets the lane rate (default 115200).

The baud rate is a parameter of the switch model and of the nodes' link tuning. At 4.5 Mbaud the endpoints matter, so the model also charges them. **Fitted to the two boards on 2026-11-16 (`bench::FIT_*`, kept by the test `latency_fitted_to_the_hardware`, which asserts each within 15%):**

| What the model charges | Value | Why |
|---|---|---|
| a call of a Klong function | 46 us (a slow tick of 5.9 ms per 128 calls) | measured: `fib(18)`, 8361 calls, 0.385 s (the first image: 94 us) |
| a 16-byte frame received, sent | 10 us, 5 us | estimates from the cycle counts (the interrupt moving 16 bytes about 250 cycles, the parser 300, the table CRC-16 150, the checks and the datagram layer 500, about 1200 cycles at 125 MHz; encoding with its CRC and queueing for the 5); they hide behind the 36 us a frame takes on the wire, and the measurements did not contradict them |
| an IPC message sent, or received whole and handled (`Platform::message_cpu`) | 880 us | fitted: see the paragraph below |
| the receive and transmit rings | 16 KB and 8 KB, the 12 ms transmit limit | the firmware's; a byte that arrives while the receive ring is full is lost and counted |

The fit is where the simulator was wrong by a factor of 2 for a small call. The first fit charged 100 us a frame and 50 us a frame sent and still gave 1.9 ms where the first image took 3.74 ms; the second image took 3.34 ms with an evaluator twice as fast, so the call is not evaluation. What the boards show (measured, 2026-11-16, through the bridge, nodes 2 and 3): a small call `h("1+1")` costs 3.36 ms and `h(:f,,7)` (no parse on the server) 3.43 ms; the same call with 40 and with 400 bytes more costs 3.94 and 8.14 ms, 11.9 us a byte against 8.9 on the wire; 40 calls issued with `.async` and waited for took 1.73 ms each; a 4 KB echo took 92 ms; `.E("1+1")` locally takes 0.345 ms. The wire and the frames explain 0.70 ms of the 3.36. The loop of a waiting board (a sleep of 200 us) adds a few hundred microseconds. The rest, about 2.6 ms, is **processor time of the IPC messages, four in a call** (request sent and received, answer sent and received): the pipelined cost of 1.73 ms a call puts it at 0.65 to 0.8 ms for each one on a board. That the cost is per message and not per frame or per packet is why a 4 KB echo, 33 packets each way, is not affected: bulk traffic is wire-bound. It is a measured total, not a decomposition: nothing on the board separates the kdb+ codec from the session and from `serve`, and a profile of an M0+ would. In the model the figure is 880 us because it has **one clock for all the boards**, so two processors cannot work at once and some of the charge passes while a frame is on the wire. Two things follow that the model does not reproduce, and the tests do not claim: pipelined calls take 2.8 ms each in the model against the board's 1.73 (the model adds the two processors), and a byte on a bulk transfer costs 10.0 us in the model against 11.9 on the boards (a 4 KB echo: 82 ms against 92: the sender's loop between two packets, about 0.2 ms each, which none of the knobs reproduces). The model's `fib(18)` is 0.383 s and its self-test 116 ms (the board's: 0.385 s, 125 ms).

Two nodes: node 1 serves, node 0 connects and calls. The time is the virtual time at the end, with the connection and the first call included:

<!-- check-docs: requires-file native/target/release/klong-bench; path native/target/release -->
```console
$ klong-bench 2 <<'EOF'
> 1> sq::{x*x};.srv("node:")
> 0> h::.cli("node:1");h(:sq,,7)
> time
> EOF
1: 1
0: 49
t = 0.122 s
```

(The 0.122 s include 60 ms during which the nodes learn their numbers from the supervisor's first PING.) At 4.5 Mbaud (`--baud 4500000`) the same connection and call take 66 ms in all, 60 of them waiting for the supervisor's first PING to number the nodes, and 48 KB (6000 integers) go to the peer in 0.47 s of the virtual clock, 102 KB/s:

<!-- check-docs: requires-file native/target/release/klong-bench; path native/target/release -->
```console
$ klong-bench 2 --baud 4500000 <<'EOF'
> 1> sq::{x*x};count::{#x};.srv("node:")
> 0> h::.cli("node:1");h(:sq,,7)
> time
> 0> big::!6000;t::.pc();r::h(:count,,big);r
> 0> _1000*.pc()-t
> EOF
1: 1
0: 49
t = 0.066 s
0: 6000
0: 473
```

The management library across four nodes, with node 0 as the manager:

<!-- check-docs: requires-file native/target/release/klong-bench; path native/target/release -->
```console
$ klong-bench 4 <<'EOF'
> 1> sq::{x*x};.srv("node:")
> 2> sq::{x*x};.srv("node:")
> 3> sq::{x*x};.srv("node:")
> 0> .l("mgmt");mgmt.add("n1";"node:1");mgmt.add("n2";"node:2");mgmt.add("n3";"node:3");mgmt.names()
> 0> mgmt.all([:sq]@0;[7])
> time
> EOF
1: 1
2: 1
3: 1
0: ["n1" "n2" "n3"]
0: :{["n1" 49] ["n2" 49] ["n3" 49]}
t = 1.131 s
```

A message far over 1024 bytes: 600 integers are about 4.8 KB on the wire, and cross each way in 3.7 s:

<!-- check-docs: requires-file native/target/release/klong-bench; path native/target/release -->
```console
$ klong-bench 2 <<'EOF'
> 1> idf::{x};.srv("node:")
> 0> h::.cli("node:1");big::!600;r::h(:idf,,big);(r~big),#r
> time
> EOF
1: 1
0: [1 600]
t = 3.759 s
```

What the errors look like: a node that does not listen, a second connection to the same node, the node's own number and a bad address:

<!-- check-docs: requires-file native/target/release/klong-bench; path native/target/release -->
```console
$ klong-bench 2 <<'EOF'
> 0> .cli("node:1")
> 1> .srv("node:")
> 0> h::.cli("node:1");.cli("node:1")
> 0> .cli("node:0")
> 0> .cli("node:x")
> EOF
0: error: system: node:1 closed the connection during the handshake (bad credentials, or it does not speak the kdb+ protocol)
1: 1
0: error: system: cannot connect to node:1: address in use
0: error: system: cannot connect to node:0: bad address
0: error: system: cannot connect to node:x: bad address
```

A node that loses power: the connection is dead for good, the node has no functions when it is back, and a new connection works once it serves again:

<!-- check-docs: requires-file native/target/release/klong-bench; path native/target/release -->
```console
$ klong-bench 2 <<'EOF'
> 1> sq::{x*x};.srv("node:")
> 0> h::.cli("node:1");h(:sq,,3)
> reset 1
> spin 2
> 1> sq::{x*x};.srv("node:")
> 0> h(:sq,,4)
> 0> h::.cli("node:1");h(:sq,,4)
> EOF
1: 1
0: 9
1: 1
0: error: system: connection lost (connection reset by the peer): node:1
0: 16
```

Three nodes answer 9.6 KB each to the manager at the same moment. Their three lanes into its one port overflow the 256-frame queue, 260 frames are lost, the layers above resend them, and the answers are all right:

<!-- check-docs: requires-file native/target/release/klong-bench; path native/target/release -->
```console
$ klong-bench 4 <<'EOF'
> 1> blob::!1200;.srv("node:")
> 2> blob::!1200;.srv("node:")
> 3> blob::!1200;.srv("node:")
> 0> .l("mgmt");mgmt.add("n1";"node:1");mgmt.add("n2";"node:2");mgmt.add("n3";"node:3");0
> time
> 0> x::mgmt.all("blob";[]);{(x@1)~!1200}'x
> time
> stats
> EOF
1: 1
2: 1
3: 1
0: 0
t = 0.917 s
0: [1 1 1]
t = 14.294 s
frames in 11129, out 10658, dropped by overflow 260, injected loss 0, max queue 256
```

(Those 10,661 frames of 16 bytes are 170 KB on the lanes for 29 KB of results: the three answers share one lane into the manager, 10 s at best, and the overflow and the resends cost the rest.)

### The numbers

From `native/klong-sim/tests/lane_ipc.rs` and `selftest.rs` (run with `cargo test --release -p klong-sim --test lane_ipc --test selftest -- --nocapture --test-threads=1`; every test runs at both rates), virtual time, a clean switch unless said:

| What | 115200 baud | 4.5 Mbaud |
|---|---|---|
| Connect (SYN, SYN, login, answer) | 27 ms | 1 ms |
| A small call, node to node | 33.7 ms | 3.5 ms (fitted to the measured 3.36 ms: 0.7 ms of wire and frames, the rest the processor time of four IPC messages) |
| A call that carries 12 KB one way | 4.58 s: 2.6 KB/s | 0.12 s: 102 KB/s |
| The same echoed back (twice the bytes, one after the other) | 9.13 s | 0.24 s |
| Ceiling of the lane | 2.88 KB/s | 112.5 KB/s; the layers' headers and framings cost 10% |
| `mgmt.all` of a small call over 3 nodes | 212 ms | 24 ms |
| 2 KB of source to each of 3 nodes with `mgmt.all(".E";...)` | 2.60 s | 0.08 s |
| 4 nodes, 12 connections (each pair both ways), every call | 1.3 s | 0.1 s |
| 5 s with the supervisor's pings going by | 6 PONGs and 5 STATUS from every node, a call every second unaffected, nothing skipped | the same |
| Fan-in of 3 senders into one port, 256-frame queue | 3 x 9.6 KB: 12.6 s, 90 frames lost to overflow (queue 256), answers right | 3 x 48 KB: 1.5 s, no overflow (the queue reached 29), answers right |
| The same with the queue cut to 16 frames | not run | 223 frames lost, 1.56 s, answers right |
| 3 echoes of 2.4 KB with 0 / 0.2% / 0.5% of frames lost | 5.56 s / 7.1 s / 12.4 s | 0.15 s / 0.40 s / 0.78 s |
| A node that resets part-way into a 6 KB call | the call fails after 1.57 s with `connection reset by the peer` | after 0.03 s |
| Self-test: the verdict reaches the supervisor, after boot (a node waits 1 s, and 0.6 s more for each node number below 4) | 6.4 s | 2.9 s (the test itself: the board's measured 125 ms, the model's 116 ms) |
| The same with a node that runs other firmware between the two (one probe of 0.4 s more) | 6.2 s | 2.7 s |

Over TCP the same small call costs microseconds; here it costs two datagrams and four messages' worth of processor. At 115200 the throughput is the lane's. At 4.5 Mbaud it is within 10% of the lane's too in the model, and the boards agree for bulk traffic (measured: a 4 KB echo, 8.2 KB on the lane, in 92 ms, 89 KB/s of the 112 KB/s). At full rate a Pico receives a frame every 36 us and has the model's 10 us for it, so about a quarter of the processor goes to the lane when it is saturated; if the true cost is more than 3 times the estimate the receive ring fills and the layers above resend (a saturating test and the `rx ring overruns` counter have not been run on the boards).

### What the simulator cannot show

* Callbacks (a server function calling back into the client that waits for it, `.cli.h`) are not exercised here: the simulated boards share one thread, and "two runtimes on one thread cannot call each other" (see [IPC and Timers](ipc.md#limits)). They work over TCP between processes and nothing in the adapter treats them differently.
* Anything that costs processor time that was not measured: the model charges a call of a Klong function (46 us, measured with `fib`), the frames and the IPC messages, and nothing for a vector operation or a parse. A Pico running soft-float `f64` is slower than the model on numeric work.
* Anything about the UART, the interrupts, the USB and the supervisor's own implementation.

## The firmware

`native/boards/pico-bench/` is a Pico SDK application of about 300 lines of C around `klong-ffi` built with `--features lane` for `thumbv6m-none-eabi`. Its `README.md` has the build, the pins, the console, the self-test and the test plan for two Picos. Facts, measured on 2026-11-16 (`native/boards/pico-bench/build.sh`, Pico SDK 2.3.1, Arm GNU 15.3.1, Rust stable):

| | |
|---|---|
| Flash | 655,276 bytes (`.bin`; 31% of 2 MB); about 430,000 with `KLONG_OPT=z`. `.uf2`: 1,310,720 bytes (measured, 2026-11-17, the image of the eighth round) |
| The runtime alone, `thumbv6m` linked as a probe, `opt-level=z`, with the library, timers and IPC | 292,724 bytes; the console without IPC 258,500 (see [Porting to a Board](porting.md) for `thumbv7em`) |
| RAM | the whole of the 264 KB, split as [below](#ram-stack-and-heap) |
| Lane buffers (in the heap) | 7.8 KB of reassembly and send buffers for 6 sources, 2 KB console output, up to 16 KB of stream buffers per open connection (what is in use after boot: 21.9 KB on a serving node, measured with `.mem()` on the previous image) |

The Cortex-M0+ has no compare-and-swap, no hardware divide and no FPU. The runtime needed no change for the first: its only atomics are loads and stores, which `thumbv6m` has, and nothing uses `Arc`. 64-bit division and all floating point come from the compiler's software routines.

### RAM, stack and heap

The RP2040 has 264 KB: 256 KB in four banks and 8 KB in two scratch banks, all one address range, `0x20000000` to `0x20042000`. The image of the ninth round divides it so, from the bottom (`build.sh` prints this from the linker's symbols and `tools/check_layout.py` fails the build when it is not a layout the firmware can run: sections inside RAM and not overlapping, none above `end`, the stack ending at the top of RAM above the heap, the heap not smaller than the one the host test models, a margin above `klong_run`):

| | Bytes | What |
|---|---|---|
| vector table | 192 | |
| `.data` | 26,408 | 16,188 the evaluator's hot functions (below), about 0.8 KB of veneers to reach flash from them, about 3 KB the unwinder that the Rust objects pull in, about 2 KB USB and `flash_do_cmd`, 1 KB newlib's malloc state, the rest impure data, mutexes and helper routines |
| `.bss` | 31,160 | 16,384 the receive ring (1024 frames, 37 ms of a saturated lane), 8,192 the transmit ring, 2,700 the heap diagnostic (below; `KLONG_HEAP_DIAG=0` leaves it out), 3,884 the rest |
| heap | 97,888 | from the end of `.bss` to the bottom of the stack: 4,096 bytes for newlib's malloc (the SDK's own users), then the runtime's pool, 93,792 bytes, of `klong_heap.h` |
| stack | 114,688 (112 KB) | from the top of RAM down: the interpreter's, which starts in the two scratch banks (the SDK's stacks are there too, and nothing else uses them: core 1 is not started); its first 32 bytes are the canary |

That is 57,760 + 97,888 + 114,688 = 270,336. (The image of the eighth round had `.bss` 28,368 and a heap of 100,784; the diagnostic costs 2.9 KB of static memory, which is 360 elements of the largest vector.)

**What happened on the boards.** The image of the sixth round (SHA-256 `c5a33b93`) ran node 2, the lower, normally and killed node 3, the client of the self-test: it was numbered, ran, said `serving` from 10 to 20 s (not `pass` a second after boot as the first image did), made its four calls and went silent about 25 s after boot, for good, with the supervisor seeing it offline; reset, it did the same. Reproduced on the host (`tests/test_native_pico_pair.py`, two nodes of the real runtime behind a switch, the firmware's own allocator over a model of newlib's malloc), the cause is two faults and a late start.

1. **The silence is the SDK's `panic("Out of memory")`.** The image links `pico_malloc`, which wraps `malloc` so that a NULL is a panic: it prints on a USB that nobody is attached to and then spins with the interrupts on, so the node answers nothing, not even the PING. The linked image has the wrapper (`__wrap_malloc` and the string "Out of memory" are in its map). The firmware's allocator was written for a malloc that returns NULL, and gives it the whole arena, so the first request it cannot meet is a panic. It met one in the management step of the self-test: `.l("mgmt")` peaks at 67.7 KB in the runtime's own count (measured, previous image, node 3), and the arena of that image was 75,284 bytes. What the runtime counts is the bytes it asks for; the arena holds also malloc's header on each block (8 bytes; about 850 live blocks at that peak on the host, 7 KB), the rounding of every block to 8, the free blocks kept for reuse (up to 8 KB) and the holes first-fit leaves. The runtime was within its limit and the arena was not: the limit was 4% under the arena, and the 4% were 3 KB. (The first image survived the same peak in a heap of 108 KB.) The host run with the old policy stops answering with `PANIC Out of memory` for every arena from 200 to 290 KB (the host's values are twice the board's size, so its arenas are larger: the proportions are the board's) and passes from 300 KB, where there is room; with the fixed policy it never stops, at any size from 60 KB up (test `the_image_that_failed_on_the_boards_goes_quiet_in_the_management_step` and `the_fixed_policy_never_goes_quiet_whatever_the_arena`).
2. **The late start is a race that the first image won by chance.** Each node probes the nodes below it for a peer, nearest first, and waits 3 s for each that does not answer. Node 2 is the lowest Klong node: below it, nodes 1 and 0 run the bench's other firmware and are mute, so node 2 spends 6 s of its first attempt in `.cli` calls that no one answers, and **a node that is blocked in `.cli` serves nobody**: the hello that node 3 sends it at the same time is not answered, and node 3 goes on to probe nodes 1 and 0 (6 s more), finds no peer, says `serving`, and asks again after 10 s. The first image passed at 1 s because node 3 happened to ask before node 2 began to probe; with the numbers the supervisor gave the nodes that time they asked together. That is a property of the code, not of the image, and it reproduces on the host: the pass came at 23 s.

The other suspects were examined and nothing was found in them (the boards could not be asked: what follows is the host and the image's map). (a) The stack in the scratch banks: the SDK places core 0's stack in scratch Y and core 1's in scratch X only by `__StackTop` and the `.stack_dummy` markers, which the layout check shows inside the interpreter's stack; nothing else lives there (no `.scratch_x` or `.scratch_y` section in the map), the SDK's stack guards are not enabled, and the stack is contiguous with the main RAM (one address range, and the stack limit check of the runtime kept it 17 KB short of the bottom). (b) `_sbrk` returns what it should (the host test uses the same function under a model of malloc, and the layout check puts its limit at the stack bottom); the heap could not reach the stack. (c) The self-test freeing its variables: the library's definitions were set to 0 after the test and the pools were released at the end of the outermost program, and the run on the host shows the console and the library working afterwards. (d) The free lists: they hold blocks malloc gave and give them back to malloc, and the host test fills freed blocks with `0xDD` and new ones with `0xAA` (a use after free reads garbage: none was found). (e) The stack budget told to Klong was 133 KB of 137: smaller than the stack. But a **calibration was wrong**: the lane's initialisation overrides the part of the budget kept below the deepest level with an eighth, not the 16 KB the board asked for (`klong_config.stack_keep` was ignored by it), so the limit was 119 KB, not 117, and, more importantly, the measurement of 90 levels in the first image was at a limit of 80,640 bytes, not 69,120: **a level costs about 864 bytes (the frames on the path add up to 864: read from the disassembly; measured 90 levels to 95 at 80,640), not 745**. The 159 levels predicted were 136. Both are corrected: the lane honours `stack_keep`, and the predictions below use 870.

**What was done.**
* **The heap is the firmware's own** (the ninth round; the seventh and eighth rounds took memory from newlib's `__real_malloc`): the first 4 KB of the heap are newlib's, for the SDK's own `malloc` users through the `_sbrk` of `main.c`; the rest is the runtime's pool, one address-ordered free list with coalescing in `klong_heap.h`, which the host tests run unchanged. A NULL is an answer: a vector that does not fit is a `memory` error; a request that cannot fail and does (a frame) ends in `k_fatal`, which records it and restarts the chip. [The heap](#the-heap-what-the-boards-showed-and-the-allocator-that-followed-the-ninth-round) has the rest.
* **The count is physical.** `klong_config.heap_round` (8) and `heap_overhead` (0): the runtime counts each block as the pool holds it (its size rounded up to 8; the pool's blocks have no header), `realloc` of a block that cannot be resized in place counts the new block while the old one is still there, and the limit is the pool less 2 KB (or 1/64) for holes; a vector may take it up to 6 KB short of that, and the small requests that cannot fail have the 6 KB.
* **The stack**: 112 KB, so that the heap can hold the management library (its counted peak, 67.7 KB plus 10 KB of headers and rounding, against a large-block limit of 83.3 KB). The first idea, 137 KB, left a limit of 58.5 KB, and the management step would have failed, gracefully now, in any case.
* **The self-test**: the first probe of a node waits 400 ms, not 3 s (a Klong node answers in 5 ms), a node starts 600 ms after the one above it (so that node 3 is done, in 125 ms, before node 2 probes), and a node that finds no peer asks again after 1.5 s plus a second per node number, four times, then every 10 s. On the host a pass comes 3.4 s after boot (numbering about 0.3 s, then 1 s, then 0.6 s for node 3).
* **A safety net.** A HardFault handler (`isr_hardfault`) stores the program counter and link register of the fault in the watchdog's scratch registers and restarts the chip; the watchdog is enabled (8 s) and fed in the idle path (`k_idle`: every turn of the main loop and every 128 calls); the idle path also looks at a canary of 32 bytes at the bottom of the stack (that the heap's end and an overrun recursion would reach first) and restarts the node with the reason `stack`; `k_fatal` records the first 8 characters of the message. At the next boot the reason becomes text, in the self-test's reports to address 16 on channel 3, and so in the supervisor's `selftest` line: `klong selftest serving node=3 calls=0 boot=hardfault pc=10002d04 lr=10002d01`, `boot=stack`, `boot=fatal ...`, `boot=watchdog` (a hang), `boot=reset` (`.x`), nothing after a power-on (`watchdog_caused_reboot`, `watchdog_enable_caused_reboot` and the scratch registers 0 to 3, which survive a watchdog reset). The text is in every report until the next restart. Tested on the host: a node started with a reason reports it (the fault handler itself can only be read, in the disassembly of the image: it picks the stack by the bit of the link register, calls `fault_c` through a literal with the Thumb bit set, and the vector table points at it).

**The split, and why.** (1) A level of Klong recursion costs 864 bytes of stack. (2) 110 levels need 110 x 870 + 1,100 = 96.8 KB, plus 16 KB kept below the deepest level (what the lane's pump needs there, about 3 KB by the frames of its path, and the walk over nested data, which the nesting limit of 21 levels is made from), plus 4 KB above `klong_run` and for the interrupts: about 117 KB. The image takes **112 KB**, which gives 107 levels (predicted), because the heap must hold the management library. (3) The heap is then 98.4 KB, a vector of about 7,900 integers. The alternatives, by `KLONG_STACK_KB` (*predicted*, from 870 bytes a level, an assumed 20 KB counted in use at idle, and the limits above; a build with another stack does not have the host test's arena and its check is off):

| `KLONG_STACK_KB` | Levels of `f::{:[x;1+f(x-1);0]}` | Heap | Largest `#!n` | Management library |
|---|---|---|---|---|
| 80 | 69 | 130 KB | 11,900 | fits |
| 96 | 88 | 114 KB | 9,900 | fits |
| 100 | 93 | 110 KB | 9,400 | fits |
| **112** (this image) | 107 | 98.4 KB | 7,900 | fits, with about 5 to 13 KB to spare |
| 120 | 116 | 90.4 KB | 6,900 | tight |
| 137 | 136 | 73.4 KB | 4,800 | fails (memory error) |

The code in RAM: 57 functions of `klong-core`, 16,188 bytes (`eval_call_in` 2,136, `num_op` 1,652, `atom_op` 1,536, `eval_node` 1,416, `resolve` 1,316, `run_fn` 1,300, `atomic` 1,116, `eval_dyad` 1,080, `push_frame` 876, `eval_seq` 740, `lookup` 700, `node_value` 520, `new_frame` 416, `pop_frame` 396, `eval_call` 260, `params_of` 84), placed there by the feature `ram-hot` (a `link_section` that the SDK's linker script puts in `.data`). From flash the code runs through the 16 KB instruction cache at a quarter of the clock (31 MHz) or, since the second image, half of it (62.5 MHz: `PICO_FLASH_CLKDIV`); a miss of 8 bytes costs about 60 cycles at the first (the figure in `CMakeLists.txt`) and, by estimate, about 30 at the second. Measured: `fib(18)` 0.79 s on the first image, 0.385 s on the second, which has the hot functions in RAM and the faster flash clock together; the share of each was not measured. **Kept in RAM, whole**: the 16 KB is 6% of the RAM, it is what the evaluator runs through at every call, and the hot set is about the size of the cache. A smaller set would be the control path (11.2 KB) with the three operator functions left to the cache (4.3 KB): it frees 5 KB, which is 640 elements of vector or 6 levels of recursion, and is *predicted* to cost under 10% of `fib`; it is not measured, and the way to measure it is two images that differ in the three `link_section` lines of `ops.rs`.

### The heap: what the boards showed and the allocator that followed (the ninth round)

**What the boards showed** (image `a232666c`, the eighth round's, nodes 2 and 3 through the bridge, 2026-11-17; every figure here was read with `klong_bench.py eval`). `.mem()` is `[in use, peak, limit]` and the limit for a vector is `limit - 6,144` less what is in use, so a heap with no fragmentation would allow `(limit - 6,144 - in use) / 8` elements (less about 150 for the evaluation of `#!n` itself):

| | in use at idle | limit | allowed by the count (less 150) | the largest `#!n` | lost |
|---|---|---|---|---|---|
| node 2, the server | 22,232 | 89,443 | 7,633 (7,483) | 6,421 (6,468 fails) | 1,060 elements, 8.5 KB, 14% |
| node 3, the client, after its self-test | 33,872 | 89,443 | 6,178 (6,028) | 4,359 (4,406 fails) | 1,670 elements, 13.4 KB, 28% |

*Both are counted memory and fragmentation, and the second is the larger.* The 11.6 KB by which the client's use is above the server's is **retention**: it explains 1,455 elements of the 2,062 between the two largest vectors (the remainder, 607, is fragmentation that the client has more of). The rest of each deficit is **fragmentation**, not an arena smaller than believed: the evidence, all through the console and all with `.mem()` unchanged to a hundred bytes:
* on node 3, `v1::!4359` (34.9 KB) and `v2::!1500` (12 KB) and `v3::!100` were held at once (counted 82,992 of a limit of 83,299 for large blocks): 46.9 KB of large blocks fit, in two places, while one of 35.2 KB (4,406) did not: the free memory was in at least two pieces, with something live between them;
* five vectors of 8.8 KB (1,100 elements) were held together, 44 KB; only the count stopped the sixth;
* after those allocations, and with 1.3 KB more in use, the same node made `#!4406`, `#!4600`, `#!5000` and `#!5500` (it had refused 4,406); node 2 went from 6,421 to at least 7,400 (7,000, 7,100, 7,200, 7,250, 7,300 and 7,400 all worked, 7,500 and 7,800 did not) with its in use going from 22,232 to 22,304. A retention does not go away by being asked for memory; a block that the allocator was holding does. The likeliest holder, from the host model below, is the 8 KB of free small blocks that the firmware kept for reuse: they sit wherever the program that freed them had them, including where a library's peak had pushed the region of small blocks, and were given back only when a large request failed (the board's own heap could not be read: that is what `.heap` is for);
* the client steps of the self-test, run by hand on node 2 against node 3 (`.cli`, a call, `.async`, a 4 KB echo, `.clic`, the names set to 0), leave 1.1 KB (23,864 to 24,992): not the connection, not the async callback, not `big`;
* `.l("mgmt")` fails on node 3 (35.2 KB in use + the library's 52 KB above what is in use is over the 83.3 KB that a large request may reach: **counted memory**), and also on node 2 (22.3 KB in use), where the count allows it: **physical**; a failed load leaves its symbols behind (1.5 KB, 22,304 to 23,864): the names a library interns are never freed.

**The host model, and what it cannot show.** `tests/test_native_pico_pair.py` runs two nodes of the real runtime behind a Python switch with the firmware's allocator source over an arena. The model of the eighth round (newlib's malloc, its size-class lists, the region at the top) with an arena where the library's peak is the same share (about 80%) as on the board shows the direction of the boards and not the size: at 400 KB the client's largest vector is 33,788 against the server's 36,424 (93%; the board 68%, from the counts above) and its counted use is 11.4 KB more; at 300 KB 20,409 against 24,022 (85%); `.l("mgmt")` is refused after the self-test, and none of it changes with later activity. With the arena of the 1.2 MB of the eighth round it showed 99.5%. The 64-bit objects are twice the Pico's, the library's peak is several times the board's, and the model of malloc is the nano allocator's rules, not its code: so every assertion of the test is a ratio against what the counted use allows, and the sizes are the boards'. The test `test_the_model_of_the_previous_allocator_shows_the_direction_of_what_the_boards_showed` keeps that model so that the harness is known to see the problem.

**Found in the model with the diagnostic** (`.heap(1..)`, below, on the host pair): after the self-test the client's free span ended at 342 KB and the server's at 366 KB of 409: between were 80 small live blocks of 24 to 64 bytes (the names the library interned and a few tables, each in a 64-byte cell with the parse's temporaries that had shared it, and free holes of 50 to 800 bytes between them) over the 24 KB that the library's parse had reached. Two causes: **the small blocks of a parse were among the blocks that live for ever**, so what the library left (names, the tables that grew while it loaded) was scattered over everything the peak had touched; and **the cache of freed small blocks hands a block back at whatever address it had**, so a name made after a parse sat where a token had been. Neither is newlib's. With the first rule alone the client's largest vector was 85% of the server's, with the second alone 94% (the plain pool: 93%); with both, 96.9% of the server's and 99.8% of what its counted use allows (the server 99.9%): what is left of the difference is the client's retention, counted.

**What the allocator is now** (`klong_heap.h`, the firmware and the host tests the one source):
* one pool, an address-ordered list of free chunks with coalescing on every free, no regions, no break, no malloc; the first 4 KB of the heap stay newlib's for the SDK (`_sbrk` in `main.c` refuses beyond them), so that the SDK's `pico_malloc` has what it has always had;
* blocks of up to 8 KB come from the highest chunk that fits, from its top end; larger blocks, and everything a parse makes (the runtime tags it: `htag::TREE`), from the lowest, from its bottom end; so a vector has the free span between the two whole;
* the small blocks kept for reuse (4 KB, down from 8) never go to the names and the tables of globals, and are given back to the pool whenever the node is idle (`k_wait`, at most every 20 ms) and whenever a request fails; the limit does not subtract them;
* `grow`, a new callback of `klong_config`: a block is resized where it is when the chunk after it is free, so a vector that grows by appending needs its growth and not the old block and the new at once; a small block that is shrunk is moved instead if there is room higher up (the tables of symbols and globals, which `release_caches` shrinks at the end of every program, then go back to the top);
* a library is loaded from the text that `.l` read, without a second copy of it for error messages (`Interp::load_owned`: 12.6 KB less at the peak of `mgmt`'s load), and the tables indexed by symbol give back at the end of a program the room they grew beyond their symbols;
* the limit is the pool less 2 KB (1/64 above 128 KB); a large request keeps 6 KB of it back.

**`.heap(mode)`**, a diagnostic: `klong-ffi`'s cargo feature `heapdiag` (on in the Pico bench build; `KLONG_HEAP_DIAG=0 ./build.sh` leaves it out and nothing of it remains, the tags and the tables included) gives `.heap` a vector of integers for each mode; one call is under 600 bytes of output, so one datagram. Offsets are from the base of the pool (field 0). The runtime tells the allocator what each allocation is for, `KLONG_ALLOC_TAG`: 0 other, 1 vector, 2 string, 3 symbols, 4 globals, 5 parse, 6 IPC, 7 console (best effort: what is allocated inside a scope has that scope's kind).

| mode | returns |
|---|---|
| `.heap(0)` | the summary, 32 integers, in this order: 0 base (address), 1 pool bytes, 2 taken from the pool (live blocks and the cache), 3 the most taken, 4 cached bytes, 5 times the cache was given back, 6 bytes given back, 7 free bytes, 8 the largest free chunk, 9 free chunks, 10 and 11 offset and size of the lowest free chunk, 12 and 13 of the highest, 14 live blocks of 512 bytes or more, 15 their bytes, 16 more than the table (96 entries) holds, 17 and 18 requests the pool could not serve (small, large), 19 blocks resized in place, 20 asked to and could not, **21 in use (`.mem()`), 22 peak, 23 limit, 24 reserve, 25 bytes taken from newlib's 4 KB, 26 newlib's arena, 27 and 28 its used and free, 29 free stack now, 30 `end` (address), 31 bottom of the stack (address)** |
| `.heap(1)`, `.heap(2)`, ... | pages of 20 runs of the pool in address order, until `[]`: `[offset size kind kinds]` four integers a run, kind 0 free, 1 used, and for a used run the bits (1 << tag) of the kinds of the blocks that started in it |
| `.heap(100)`, `.heap(101)`, ... | pages of 24 of the live blocks of 512 bytes or more, by address: `[offset size tag]` |
| `.heap(200)` | the cached blocks of each size class, 8 to 128 bytes, 16 integers |

**What would show a failure.** A largest vector below what the counted use allows, with the largest free chunk (field 8) at the same value: something live is in the middle of the pool: the runs just above the first big free run say where (their kinds say what, the table of blocks gives the large ones). A largest vector at what the count allows: retention, and field 21 against the other node's says how much. A request refused with a free chunk large enough: the limit (fields 17 and 18 stay 0 for a refusal by count).

*Predicted for the image of the ninth round, to be confirmed by flashing* (nodes 2 and 3, after the self-test; the host's own ratios, 99.7% to 99.9% of what the count allows, applied to the boards' counts, with the library's retention as measured): pool 93,792, limit 91,744, so a vector may reach 85,600 counted bytes. In use at idle: node 2 about 21.5 KB (20.5 to 22.5), node 3 about 31.5 KB (29 to 34: the symbols stay). The largest `#!n`: **node 2 about 7,750 (7,500 to 7,900)**, **node 3 about 6,500 (6,100 to 6,800)**, node 3 in either case within 1% of `(85,600 - in use) / 8 - 150`. `.l("mgmt")` on node 3 **loads** (it needs about 40 KB above what is in use now that the second copy of its text is gone, 52 KB if that saving is not real: 85,600 - 31,500 = 54 KB); with it loaded about 4,300 (3,900 to 4,700); dropped again, the largest vector within 1% of before it. `fib(18)` 0.384 s (the allocator is not on its path: 60 allocations in all), recursion 105 levels, the self-test 125 to 150 ms.

### The memory of the client of the self-test (the eighth round; its region and its host model were replaced in the ninth)

Measured on the boards after the seventh round's image (2026-11-17): the server (node 2) had 22,976 bytes in use and made vectors up to 7,390 elements (fails at 7,429); the client of the self-test (node 3), straight after its pass, had 35,152 in use and made 4,000 (fails at 5,000); one `h::.cli("node:2");h("1+1")` and `.clic(h)` gave back 8 KB; after more activity 4,312, unchanged by setting its globals to 0, and `.l("mgmt")` then failed with a memory error. Found on the host (`tests/test_native_pico_pair.py`: the client and the server behind a switch, with `pico_node.c` dumping the model of malloc, block by block with the stack of each allocation, on a signal):

1. **A closed connection stayed until the next one.** The table of handles (`Ipc::handles`) held a strong reference to each connection, and was pruned of the entries of dead handles only when the next handle was made: after `.clic(h)` and `h::0` the connection, with its buffers (an echo's 4 KB), was kept until the next `.cli`. This is the 8 KB that one connect and close recovered. The entry now holds the connection weakly (the handle's closure holds it for as long as the handle lives).
2. **The permanent blocks of a library were in the middle of malloc's arena.** The table of symbols and of globals grow by doubling while a library loads, and the blocks they end in (4 to 5 KB) and the small blocks of the symbols stay when the library is dropped; with newlib's first fit they were in the middle of the heap, where the library's own blocks had been, so that the free memory was in pieces and no large block was contiguous. On the host the largest free block of the client was 77 KB smaller than the server's after the self-test, with most of its memory free. (Symbols are never freed: that part of the difference stays, about 3 KB on the board.) The seventh round's firmware gave newlib every block; now **blocks of up to 8 KB are not newlib's**: they live in a region at the top of the arena, which grows down and shrinks back when its lowest block is freed, in an address-ordered list with no header (the runtime tells `free` the size) that fits a request into the highest hole that holds it. The vectors, the strings and the parse trees of over 8 KB, which come and go, have the bottom of the arena to themselves: what a program leaves behind is at the top and the bottom is contiguous. When the region cannot grow (a vector made the break of malloc high, then went), small blocks take what malloc has free below it, and give it back there (the address says which it was). The free lists of the size classes are given back, and the request tried once more, when an allocation fails before an out-of-memory error is reported.
3. **The count was heavier than the arena's use.** With the region there is no header on nearly any block, and counting 8 bytes more for each (the seventh round) cost about 3 KB of every program's memory. Blocks are now counted rounded up to 8.

On the host (arena of 1.2 MB: the management library takes a megabyte of it on the 64-bit host and 45 KB on the board, so the numbers that mean something are proportions; `test_the_client_of_the_selftest_has_the_memory_of_the_server_and_the_library_can_be_loaded_and_dropped`): after its self-test the client makes 99.5% of the server's largest vector (133,739 against 134,375 elements; it was 93% with the region and the leak not yet repaired, and the dump of the heap before showed the whole of the difference to be the two causes above); with the management library loaded 87% (116,942: the library's own memory), dropped again 99.5% (133,690), loaded and dropped a second time the same, and `.l("mgmt")` loads each time; a connection closed leaves the same memory after the first cycle as after the second (within 400 bytes).

**Why node 2 came in at 7,390 and not 7,500.** The limit for a vector is the limit of the runtime (89,455 bytes: the arena less 8 KB for the free blocks that are kept and 3% for holes) less the reserve of 6 KB, 83,311 bytes, less what is counted in use, so `(83,311 - 22,976) / 8 = 7,541` elements, less the 1.2 KB that evaluating `#!n` itself takes (the console's line, the result) = 7,390. The 7,500 to 8,200 predicted assumed 20 KB in use at idle (the seventh round's prediction removed the pools of the evaluator and forgot the 8 bytes a block that its own count added: 22,976 measured, about 3 KB of it headers). With the headers gone from the count:

*Predicted for the image of the eighth round, to be confirmed by flashing* (the figures of the boards, the first of each pair the server and the second the client straight after its self-test): in use at idle about 19.5 KB (19 to 21) and about 22 KB (21 to 24); the largest `#!n` **about 7,800 (7,600 to 8,000)** on the server and **about 7,400 (7,100 to 7,700)** on the client (the 3 KB of its symbols and a fraction of a percent of fragmentation: on the host 99.5%); after loading `mgmt` and dropping it again the same as before loading, and the library loads again (it needs about 52 KB counted above what is in use: it loads with room to spare below 31 KB in use); **with `mgmt` loaded about 4,500 (4,000 to 5,000)** (the library holds about 18 KB, its symbols and the dictionary of nodes). The management library is loadable on a client with about 60 KB left. Recursion 105 to 109 levels (measured 105 works and 109 fails on the seventh image), `fib(18)` about 0.382 s, the self-test about 128 ms begun 3 s after boot; the blocks that are not on the free lists of the size classes cost a walk of the region's list (about 10 us a block with 100 holes), which the load of the management library has to do a few thousand times: the self-test may take up to 30 ms longer (predicted 130 to 160 ms).

### Receiving and sending at 4.5 Mbaud

A byte arrives every 2.2 us, a frame every 36 us, and the 32-byte hardware FIFO fills in 71 us, so the lane is read by interrupt, never by polling. The choice is between parsing frames in the interrupt into a frame queue and moving raw bytes into a ring and parsing in the main loop. The firmware **moves raw bytes into a 16 KB ring** (1024 frames, 37 ms of a lane that never pauses; it was 32 KB in the first design and is 16 KB in the image that runs on the boards) because the parser, the CRC and the datagram layer are Rust code with state that the main loop also touches, putting them at interrupt level would cost about 25% of the processor at saturation in the interrupt and need locking, and the ring is cheap (6% of the RAM). The assumption behind 37 ms: the main loop is away longer than that only inside one primitive of the interpreter that does not call `idle` (a sort or a sum over a million elements), since the runtime reads the ring at every turn of its loop, every 4096 evaluation steps and inside every IPC call. What does not fit is dropped and counted (`rx ring overruns` in `klong_lane_stats`, in Ctrl-T on USB and in the simulator's `rx_ring_overruns`), along with FIFO overruns and line errors; the layers above recover by retransmission, as they do for any lost frame. The interrupt runs at the highest priority with a FIFO level of 8 bytes, so it has 53 us before the FIFO overflows, and the receive-timeout interrupt takes the tail of a burst. Per call the adapter reads at most 8 KB (512 frames) so that sending is not starved while a full ring drains.

Transmitting never blocks the interpreter: the runtime copies a frame into an 8 KB ring and returns, and the transmit interrupt (FIFO half empty) feeds the wire. At most 12 ms of wire time (5.4 KB at 4.5 Mbaud, 128 bytes at 115200) may wait in the ring: a deeper queue only delays what has to go out behind it, an ACK behind a full window (measured in the simulator: with a 8 KB limit at 115200 three echoes took 8.3 s instead of 5.55 s), and a shallower one lets the lane idle while the main loop is away. If the main loop is away for more than 12 ms the wire idles; nothing breaks. `PacketLink::send` answers "busy" while a datagram's frames are still being queued, which paces each connection to the lane.

## Interactive use

A Klong console on any node, from a web page in the supervisor (the bridge, built in the bench's own repository) or from a workstation with `native/tools/klong_bench.py`. **Status: the node side and the CLI are proven in the simulator and against a model of the bridge, and they have run on two Picos through the supervisor's real bridge (2026-11-16: `status` with the decoded flags, `eval`, interrupts, the measurements on this page).**

How it works. A node's console takes its input from datagram **channel 1** and its output goes the same way; **channel 2** is control; **channel 3** is the self-test report, so it stays out of the console.

| What | Format |
|---|---|
| Input | a channel-1 datagram of at most 1024 bytes that ends at a line end, with the **ACK flag**; the sender becomes the console's peer. Its lines are evaluated in order by the same console that serves USB, with no echo and no prompt. A datagram without the ACK flag is not input (it is another console's output, and is dropped, so two nodes typing at each other do not loop) |
| Output | channel 1, text, no ACK flag, to the peer that last sent input, in datagrams of at most 1024 bytes cut at a line end; results and error messages as the console prints them. With no peer (nobody has typed, or somebody typed on the USB console since) it is dropped and counted: a timer that prints on an idle node loses its output |
| `done ID STATUS` | channel 2, after all the lines of input datagram ID have been evaluated and their output sent: `ok`, `error` (some line raised one; its text went to channel 1) or `interrupted` |
| `int` | channel 2, from the peer: ends the running evaluation by the serial console's Ctrl-C rule (a 0x03 the console reads through the same lane pump that runs every 4096 steps, so it is seen while the interpreter is busy) and discards what was received before it, the rest of the datagram and the datagrams behind it, each of which is told `done ID interrupted`. Ignored when idle |
| `hello` | channel 2: the reply is `klong VERSION node=N TARGET int64 float64` |
| STATUS | flag bit `0x10` (not a cube node): `0x11` alive, `0x31` self-test serving or running, `0x51` pass, `0x91` fail. Load 100 while input is being evaluated, 0 when idle |

A **second sender** is refused, not queued: while a node's console is busy for one sender, a datagram with the ACK flag on channel 1 from another address is answered with ACK status 2 and goes nowhere (and may be sent again); the owner may send as many as it likes, which queue like typed-ahead input, and the console is free for others when its last datagram is done. A node typing into another node's console, when it is free, takes the output away from whoever typed before, since the peer is the address that last sent input: **during a page session only the supervisor should type at the nodes.**

The upload commands of the serial console (`^A put`, `^A run`) work on channel 1: a datagram is a chunk of the command's stream, and the console answers with its ordinary `\x01go` and `\x01ok` lines. They are binary-safe to the console (a datagram is CRC-checked and its bytes reach the console unchanged, even 0x00 and 0x03 outside an evaluation) but the **1024-byte limit and the bridge's newline rule** limit what can be sent through the bridge: the bridge adds a newline to input that has none, so a chunk must end at a newline, which means text files in chunks of at most 1023 bytes plus the newline, no binary files, no line over 1023 bytes. Through a lane directly (not the bridge) any bytes in any chunk sizes work.

`klong_bench.py` (standard library only, Python 3.9+):

    klong_bench.py --host bench.local status
    klong_bench.py --host bench.local repl --node 1
    klong_bench.py --host bench.local eval --node 1 'sq::{x*x};sq(12)'
    klong_bench.py --host bench.local run  --node 1 program.kg

`--host` defaults to the environment variable `KLONG_BENCH_HOST`. The CLI is a client of the bridge like the page: it names itself (`id` and `name` on every call), claims control with its first input (or asks for it with `--take`), keeps it alive by polling, gives it up when it ends, leaves out the echo of its own lines (the bridge's transcript has `klong> LINE` for every line typed), and prints "not delivered" when the bridge says the input was refused or never acknowledged. Ctrl-C in the REPL interrupts the node; a second one stops waiting.

`native/tools/bench_bridge_sim.py` is a stand-in for the bridge in front of the simulator, with the same routes, errors and control rules, for the CLI's tests and as an executable reference. A session against it (the port is free for the example; the virtual bench runs at four times real time):

<!-- check-docs: requires-file native/target/release/klong-bench; path native/target/release; link native -->
```console
$ python native/tools/bench_bridge_sim.py --port 18751 --nodes 2 --speed 4 > bridge.log 2>&1 & echo $! > bridge.pid; for i in $(seq 100); do grep -q "simulator pid" bridge.log && break; sleep 0.1; done
$ for i in $(seq 200); do python native/tools/klong_bench.py --host 127.0.0.1:18751 status 2>/dev/null | grep -q "yes" && break; sleep 0.1; done
$ python native/tools/klong_bench.py --host 127.0.0.1:18751 eval --node 1 'sq::{x*x};sq(12)'
144
$ printf 'n::40\nn+2\n1+"a"\n:q\n' | python native/tools/klong_bench.py --host 127.0.0.1:18751 repl --node 1 | sed -n 2,7p
klong[1]> 40
klong[1]> 42
klong[1]> 1:1: type error: cannot apply + (Plus) to integer and string
  1+"a"
  ^^^^^
klong[1]>
$ kill $(cat bridge.pid)
```

(Ctrl-C in `repl`, or an interrupt sent to `eval` while it runs `{1}{x+1}:~0`, ends the loop with `interrupt error: interrupted`; `tests/test_native_bench_cli.py` does that, with a signal, and checks the load of 100 in the node's STATUS while it runs.)

## The boot self-test

For the first hardware test the image runs a fixed self-test at boot (`klong-rt/src/selftest.rs`, the same Rust code that the simulator runs in `klong-sim/tests/selftest.rs`), selected at build time (`KLONG_SELFTEST=0` leaves it out), one image for every board. Every node serves. A node that is not number 0 connects to the nearest node below it that answers (`.cli("node:K")` for K = N-1 down to 0), so of two flashed boards the lower serves and the higher is the client, and neither needs the other's number in advance. The client runs four steps and times them with the board clock: a synchronous call; an asynchronous call, then a synchronous one, then the asynchronous answer; a 4 KB echo (500 integers) compared; and, on a new connection, the management library calling the peer. The result goes to address 16 on datagram **channel 3** as text, every 3 seconds, and as the flags byte of STATUS (bit `0x10` says it is not a cube node):

| State | STATUS flags | Report text |
|---|---|---|
| alive, no self-test | `0x11` | none |
| serving, looking for a peer, or running | `0x31` | `klong selftest serving node=N calls=C` or `klong selftest running node=N calls=C` |
| passed | `0x51` | `klong selftest pass 94ms node=N peer=K sync=1ms echo=79ms calls=C` |
| failed | `0x91` | `klong selftest fail STEP REASON node=N peer=K` |

(`pass` is followed by the time of the four steps in milliseconds; STEP is `sync`, `async`, `echo` or `mgmt` and REASON the error's kind and message.) A node that finds nobody below it is the lowest board with the image as far as it can tell, so it reports `serving` and looks again every 10 seconds, never `fail`; a failure is run again after 15 seconds, a pass stays. The flag values use bits 4 to 7, which the bench's own OS does not (it uses bits 1 and 2 for its apps).

In the simulator, behind the switch model and with a simulated supervisor that reads the channel-3 datagrams and the STATUS flags, at both rates: two boards with the same program pass (flags `0x51` on the client, `0x31` and `calls=5` on the server, the report repeating); a muted node in between (other firmware) is skipped after a handshake timeout and the next one is the peer; a lone board serves and never fails; a peer with a broken echo makes the client report `fail echo type: cannot apply + (Plus) to integer and string` with flags `0x91` and, once it is fixed, `0x51`; and a flood of the deployer's sub-types `0xA0..0xB1` and `0xE1/0xE2` to both nodes during the test changes nothing (the datagram layer's tests do the same between and inside messages).

One thing the self-test found, and what it turned out to be: with a function `f` defined on both nodes, `h(:f,,5)` and `.async(h(:f);cb)` run the *local* `f`. That is the language, not the adapter and not a native-only bug: a symbol that names a function you have defined is that function (KlongPy evaluates `:f` to the function as well), and calling a peer with a function sends its source for the peer to apply (`f(inc,1)` in [IPC Capabilities](../ipc_capabilities.md)). To call the peer's own `f` when you also have one, use the list `h([:f 5])` or the string `h("f(5)")`, and `.async(h;cb)` with the same requests. The self-test names its functions for their node (`stping0`, `stecho0`) so that it never meets the case; `tests/test_native_ipc.py` pins the behaviour down with two processes.

`klong-ffi` with `--features lane` adds `klong_lane_init`, `klong_lane_pump`, `klong_lane_node` and `klong_lane_stats_get` to `klong.h`. `tests/lane.c` (run by `cargo test -p klong-ffi`) drives them from a C program that plays the supervisor: noise, a broadcast PING, a PING that numbers the node, the PONG and the STATUS a second later, ignored frames, a console line answered on channel 1, and, in a second run, the self-test of node 0 (`serving`, STATUS flags `0x31`).

## Open questions for the bench owner

* **Does the supervisor accept this node's PONG and STATUS?** They now follow the stated rules (PONG with the PING's sequence and length 0; STATUS to 16 every second, length 4, load, flags, uptime LE) and `software/worker/runtime.h`; they have not run against the supervisor.
* **The self-test's report on channel 3** and the flag values `0x31`, `0x51`, `0x91` were agreed with the bench's own session; they are constants.
* **Who sends datagrams from the supervisor?** The QT Py firmware has no datagram sender; the console over the lane needs one (`tools/lane_frames.py` shows the frames). The self-test needs only the supervisor to read.
* **The processor's cost per frame** is still an estimate (10 us received, 5 sent), and hidden by the wire in every measurement so far (a 4 KB echo at 89 KB/s). If a Pico needs several times that, the lane's 112 KB/s cannot be used and the receive ring is what absorbs the difference; the test is the `rx ring overruns` counter (Ctrl-T on USB) under a bulk transfer in both directions at once.
* **Packet size** (126 bytes) and the first timeouts (100 ms at 4.5 Mbaud) are tuned in the simulator for random loss; the bench's losses come from queue overflow, in bursts.
* **Keepalive** is off, so a node that vanishes without a reset is found by the 30 s call timeout or the retry budget.
* **Flashing 641 KB takes 80 to 90 s per Pico**; the image could be 430 KB with `KLONG_OPT=z`.

## What the hardware has shown, and what is open

Shown on two Picos, 2026-11-16 (measured): the image boots (the supervisor sees the node alive; the Picos have no USB cable on the bench, so the USB console is not among what was shown); the stack switch is right and the evaluator's depth limit is honoured, with a `range` error at the limit and the node running on; the UART interrupt and the 16 KB ring keep up with a 4 KB echo at 4.5 Mbaud (89 KB/s) and the self-test passes (`klong selftest pass 125ms node=3 peer=2 sync=5ms echo=92ms`); the supervisor accepts the node's STATUS and the flags `0x31` and `0x51` (`status` decodes them); calls cost 3.36 ms and `fib(18)` 0.385 s; the echo and prompt handling of the console and the reconnect work.

Open: the receive ring under a sustained bulk transfer in both directions (the overrun counter); the real speed of 64-bit and floating-point arithmetic beyond `fib` (integers only); the per-message processor cost split into its parts; the A/B of the hot set in RAM; and everything in *predicted* above until an image with the new heap and stack is flashed. `native/boards/pico-bench/README.md` has the test plan for two Picos.
