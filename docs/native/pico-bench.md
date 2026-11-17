# Native IPC on the 4-Pico Bench

**Status: designed, built and proven in the simulator; not yet run on hardware.** No Pico has been flashed with this firmware and no lane has carried one of its frames. Every number below that has a unit of time is *virtual time* from the simulator, which charges the lanes' baud rate exactly and the processor's work with stated estimates. The firmware image builds and links. What that leaves unconfirmed is listed [at the end](#what-only-hardware-can-show).

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

The baud rate is a parameter of the switch model and of the nodes' link tuning. At 4.5 Mbaud the endpoints are the limit, so the model also charges them: **10 us of processor per 16-byte frame received and 5 us per frame sent, 0.2 ms per 4096 evaluation steps. These are estimates, not measurements.** The 10 us: the interrupt moving 16 bytes (about 250 cycles), the frame parser (about 300) and the table CRC-16 (about 150; the bitwise one took 6 times that, so the table, 512 bytes of flash, is in), the validity checks and the datagram layer (about 500), about 1200 cycles at 125 MHz. The 5 us: encoding with its CRC and queueing. Receive and transmit rings are modelled too (32 KB and 8 KB, with the 12 ms transmit limit), and a byte that arrives while the receive ring is full is lost and counted.

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
t = 0.118 s
```

(The 0.118 s include 60 ms during which the nodes learn their numbers from the supervisor's first PING.) At 4.5 Mbaud (`--baud 4500000`) the same connection and call take 62 ms in all, 60 of them waiting for the supervisor's first PING to number the nodes, and 48 KB (6000 integers) go to the peer in 0.47 s of the virtual clock, 102 KB/s:

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
t = 0.062 s
0: 6000
0: 469
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
t = 1.071 s
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
t = 3.756 s
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
t = 0.872 s
0: [1 1 1]
t = 14.237 s
frames in 11166, out 10661, dropped by overflow 260, injected loss 0, max queue 256
```

(Those 10,661 frames of 16 bytes are 170 KB on the lanes for 29 KB of results: the three answers share one lane into the manager, 10 s at best, and the overflow and the resends cost the rest.)

### The numbers

From `native/klong-sim/tests/lane_ipc.rs` and `selftest.rs` (run with `cargo test --release -p klong-sim --test lane_ipc --test selftest -- --nocapture --test-threads=1`; every test runs at both rates), virtual time, a clean switch unless said:

| What | 115200 baud | 4.5 Mbaud |
|---|---|---|
| Connect (SYN, SYN, login, answer) | 27 ms | 1 ms |
| A small call, node to node | 30.7 ms | 1.0 ms (two datagrams of about 11 frames, 0.4 ms each, and the model's 0.2 ms polling step) |
| A call that carries 12 KB one way | 4.58 s: 2.6 KB/s | 0.12 s: 102 KB/s |
| The same echoed back (twice the bytes, one after the other) | 9.13 s | 0.24 s |
| Ceiling of the lane | 2.88 KB/s | 112.5 KB/s; the layers' headers and framings cost 10% |
| `mgmt.all` of a small call over 3 nodes | 198 ms | 6 ms |
| 2 KB of source to each of 3 nodes with `mgmt.all(".E";...)` | 2.58 s | 0.07 s |
| 4 nodes, 12 connections (each pair both ways), every call | 1.2 s | 0.1 s |
| 5 s with the supervisor's pings going by | 6 PONGs and 5 STATUS from every node, a call every second unaffected, nothing skipped | the same |
| Fan-in of 3 senders into one port, 256-frame queue | 3 x 9.6 KB: 12.5 s, 90 frames lost to overflow (queue 256), answers right | 3 x 48 KB: 1.45 s, no overflow (queue reached 172), answers right |
| The same with the queue cut to 64 frames | not run | 570 frames lost, 1.55 s, answers right |
| 3 echoes of 2.4 KB with 0 / 0.2% / 0.5% of frames lost | 5.55 s / 7.1 s / 12.1 s | 0.14 s / 0.40 s / 0.79 s |
| A node that resets part-way into a 6 KB call | the call fails after 1.57 s with `connection reset by the peer` | after 0.03 s |
| Self-test: the verdict reaches the supervisor, after boot | 4.5 s | 1.1 s |
| The same with a node that runs other firmware between the two (one handshake timeout more) | 14.5 s | 4.1 s |

Over TCP the same small call costs microseconds; here it costs two datagrams. At 115200 the throughput is the lane's. At 4.5 Mbaud it is within 10% of the lane's too, **with the estimated processing costs**; whether a real Pico keeps up is the first thing the hardware has to show. At full rate a Pico receives a frame every 36 us and has the model's 10 us for it, so about a quarter of the processor goes to the lane when it is saturated; if the true cost is more than 3 times the estimate the receive ring fills and the layers above resend.

### What the simulator cannot show

* Callbacks (a server function calling back into the client that waits for it, `.cli.h`) are not exercised here: the simulated boards share one thread, and "two runtimes on one thread cannot call each other" (see [IPC and Timers](ipc.md#limits)). They work over TCP between processes and nothing in the adapter treats them differently.
* The real processors' speed. The model charges 0.2 ms per 4096 evaluation steps, a stand-in; a Pico running soft-float `f64` will be slower, and a call that evaluates something heavy will be dominated by it.
* Anything about the UART, the interrupts, the USB and the supervisor's own implementation.

## The firmware

`native/boards/pico-bench/` is a Pico SDK application of about 300 lines of C around `klong-ffi` built with `--features lane` for `thumbv6m-none-eabi`. Its `README.md` has the build, the pins, the console, the self-test and the test plan for two Picos. Facts, measured on 2026-11-16 (`native/boards/pico-bench/build.sh`, Pico SDK 2.3.1, Arm GNU 15.3.1, Rust stable):

| | |
|---|---|
| Flash | 623,172 bytes (`.bin`; 30% of 2 MB); about 430,000 with `KLONG_OPT=z`. `.uf2`: 1,246,720 bytes |
| The runtime alone, `thumbv6m` linked as a probe, `opt-level=z`, with the library, timers and IPC | 292,724 bytes; the console without IPC 258,500 (see [Porting to a Board](porting.md) for `thumbv7em`) |
| Static RAM | `.data` 9.3 KB, `.bss` 93.7 KB: the 48 KB interpreter stack, the 32 KB receive ring, the 8 KB transmit ring |
| Heap | 112 KB limit (159 KB available), 16 KB held back |
| Stack | 48 KB static, the runtime told 42 KB (unmeasured on this core) |
| Lane buffers (in the heap) | 7.8 KB of reassembly and send buffers for 6 sources, 2 KB console output, up to 16 KB of stream buffers per open connection |

The Cortex-M0+ has no compare-and-swap, no hardware divide and no FPU. The runtime needed no change for the first: its only atomics are loads and stores, which `thumbv6m` has, and nothing uses `Arc`. 64-bit division and all floating point come from the compiler's software routines.

### Receiving and sending at 4.5 Mbaud

A byte arrives every 2.2 us, a frame every 36 us, and the 32-byte hardware FIFO fills in 71 us, so the lane is read by interrupt, never by polling. The choice is between parsing frames in the interrupt into a frame queue and moving raw bytes into a ring and parsing in the main loop. The firmware **moves raw bytes into a 32 KB ring** (2048 frames, 73 ms of a lane that never pauses) because the parser, the CRC and the datagram layer are Rust code with state that the main loop also touches, putting them at interrupt level would cost about 25% of the processor at saturation in the interrupt and need locking, and the ring is cheap (12% of the RAM). The assumption behind 73 ms: the main loop is away longer than that only inside one primitive of the interpreter that does not call `idle` (a sort or a sum over a million elements), since the runtime reads the ring at every turn of its loop, every 4096 evaluation steps and inside every IPC call. What does not fit is dropped and counted (`rx ring overruns` in `klong_lane_stats`, in Ctrl-T on USB and in the simulator's `rx_ring_overruns`), along with FIFO overruns and line errors; the layers above recover by retransmission, as they do for any lost frame. The interrupt runs at the highest priority with a FIFO level of 8 bytes, so it has 53 us before the FIFO overflows, and the receive-timeout interrupt takes the tail of a burst. Per call the adapter reads at most 8 KB (512 frames) so that sending is not starved while a full ring drains.

Transmitting never blocks the interpreter: the runtime copies a frame into an 8 KB ring and returns, and the transmit interrupt (FIFO half empty) feeds the wire. At most 12 ms of wire time (5.4 KB at 4.5 Mbaud, 128 bytes at 115200) may wait in the ring: a deeper queue only delays what has to go out behind it, an ACK behind a full window (measured in the simulator: with a 8 KB limit at 115200 three echoes took 8.3 s instead of 5.55 s), and a shallower one lets the lane idle while the main loop is away. If the main loop is away for more than 12 ms the wire idles; nothing breaks. `PacketLink::send` answers "busy" while a datagram's frames are still being queued, which paces each connection to the lane.

## Interactive use

A Klong console on any node, from a web page in the supervisor (the bridge, built in the bench's own repository) or from a workstation with `native/tools/klong_bench.py`. **Status: the node side and the CLI are proven in the simulator and against a model of the bridge; no Pico has run Klong yet.**

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
* **The processor's cost per frame** is an estimate (10 us received, 5 sent). If a Pico needs several times that, the lane's 112 KB/s cannot be used and the receive ring is what absorbs the difference; a first measurement is the `rx ring overruns` counter under a bulk transfer.
* **Packet size** (126 bytes) and the first timeouts (100 ms at 4.5 Mbaud) are tuned in the simulator for random loss; the bench's losses come from queue overflow, in bursts.
* **Keepalive** is off, so a node that vanishes without a reset is found by the 30 s call timeout or the retry budget.
* **Flashing 618 KB takes 80 to 90 s per Pico**; the image could be 430 KB with `KLONG_OPT=z`.

## What only hardware can show

That the image boots and the USB console comes up; that the 48 KB stack and the stack switch are right; the UART at 28,125 frames/s (the interrupt keeping up, the 32 KB ring while the interpreter evaluates, the real baud rate at 125 MHz); the real speed of 64-bit and floating-point arithmetic on the M0+ and the real processor time per frame; the supervisor's view of the node; and every figure above that depends on the processor. `native/boards/pico-bench/README.md` has the test plan for two Picos.
