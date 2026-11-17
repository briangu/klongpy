# Native IPC on the 4-Pico Bench

**Status: designed, built and proven in the simulator; not yet run on hardware.** No Pico has been flashed with this firmware and no lane has carried one of its frames. Every number below that has a unit of time is *virtual time* from the simulator, which charges the lanes' 115200 baud exactly and the processor's work only roughly. The firmware image builds and links. What that leaves unconfirmed is listed [at the end](#what-only-hardware-can-show).

This page is for someone who wants native Klong nodes to talk to each other, `.cli` to `.srv`, over a packet switch that carries nothing but 16-byte frames. It is the first network of the native runtime that is neither a byte stream nor a serial line, and it shows how little the IPC code had to change to take it.

## The bench

Four Raspberry Pi Picos (RP2040: two Cortex-M0+ cores at 125 MHz, 264 KB of RAM, 2 MB of flash) each have one UART lane to a packet switch (an FPGA router): GP16 TX and GP17 RX of uart0, 115200 baud 8N1, no flow control. A supervisor (a QT Py ESP32-S3) has a lane too, port 16. The switch forwards **fixed 16-byte frames**: a frame carries 4 bytes of payload, so a lane moves about 720 frames a second and **2.9 KB/s of data at best**. It is store-and-forward and best effort: each output port holds 256 frames in a FIFO, and a frame that finds it full is lost. Frames from one port to another stay in order. There is no flow control and no retransmission. The supervisor sends a PING to every node about once a second and expects a PONG and a STATUS, or it marks the node offline.

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

The sender sends START, the DATA frames in order, END, back to back. 1024 bytes are 258 frames, 4128 wire bytes, 0.36 s. With ACK asked for it waits 300 ms plus 1.5 ms per frame for the ACK and resends the whole message with the *same* id, three times at most. The receiver keeps one reassembly state per source (17 possible: nodes 0 to 15 and the supervisor), abandons a message when a START arrives in the middle of it, marks it broken on a fragment out of order or past the length, delivers it at END only if the id, the fragment count and the CRC-32 agree, remembers the id of the last message it delivered from each source and acknowledges a repeat with 0 without delivering it again. Channels are a byte in START: 0 for IPC.

**RAM** is bounded by the type: `Lane<MAX, SLOTS>` is `(SLOTS + 1) * MAX` bytes of buffers plus about 650 bytes (measured: 19,064 bytes for 1024 and 17 slots, 7,792 for 1024 and 6, 1,904 for 256 and 4). A reassembly buffer is handed to a source when it starts a message and taken back when the message ends, so `SLOTS` is how many sources may be in the middle of a message at once; one more is refused (ACK 2, or silence without ACK). The Pico image uses 6.

Tests (`native/klong-dgram/tests`): the check values; the frame layout and every way a frame is invalid; the parser against noise, a torn frame, a lost byte and sync bytes inside a payload; each rule of "Receiving"; the sender's frames, its timeout (300 + 387 ms for 1024 bytes), its three resends and the ACK codes; messages of 1, 3, 4, 5, 1023 and 1024 bytes and 1025 refused; id wrap-around; and, with a seeded generator, a network of lanes with random noise between frames, dropped, duplicated and corrupted frames, START in the middle of a message, three senders into one port with the queue overflowing, and 300 runs of arbitrarily mutated streams that may deliver only what was sent and never twice in a row. A frame sequence printed by the supervisor-side tool `native/boards/pico-bench/tools/lane_frames.py` is delivered by the Rust code as the tool says.

One departure from the datagram spec's wording on PING: the layer itself ignores it, as specified, but the *lane* around it answers PING with PONG and sends STATUS once a second, because a node that does not is marked offline (`KLONG_LANE_NO_DUTIES` turns that off).

## IPC over datagrams

Native IPC runs over links. A link is a byte stream (TCP) or a packet link, which the runtime puts a reliability layer on (`rel`: 1 header byte, SYN handshake, 6-bit sequence numbers, go-back-N with a window of 16, RST, keepalive). A serial line is a packet link. So is the switch, once something cuts the lane into packets: that is the whole adapter (`klong-rt/src/node.rs`, about 530 lines).

* **Address.** `.cli("node:2")` opens a connection to node 2. `"node:"` is the name of the transport and `2` its address, the form `"name:address"` that [IPC and Timers](ipc.md#the-functions) already gives other transports. `node:16` is the supervisor, `node:n2` is the same as `node:2`. `.srv("node:")` accepts connections from any node. A node's own number is what the first PING told it; until then `.cli` says `address unreachable`.
* **Handshake.** The first packet the client sends is the reliability layer's SYN. The server accepts a connection when the first packet of one arrives, answers with its own SYN, and the kdb+ handshake (the login string and the capability byte) runs over the stream as on TCP. A connection to a node that does not listen is answered with a reset and fails at once: `closed the connection during the handshake`, the same words as a refused login. A user name and password in the address are not carried (`node:1:u:p` is a bad address), so `.srv.auth` sees empty ones.
* **Messages larger than 1024 bytes.** An IPC message goes into the reliability layer's byte stream, which cuts it into packets of 126 bytes. One packet is one datagram of 128 bytes (2 header bytes of the adapter, 1 of the reliability layer, 125 of IPC), 34 frames, 47 ms on the lane. So a message of any size crosses the link, and the kdb+ bytes inside are exactly the bytes TCP would carry. Nothing in the datagram layer's limit shows through.
* **Timeouts.** Scaled to the link: first retransmission 500 ms and doubling, 6 retries, 10 s for the handshake (the defaults are 8 retries and 5 s for TCP-like links). The call timeout is the usual 30 s, and `.clit` changes it. The adapter asks for these through a new, optional `PacketLink::tuning()`; no other transport changes.
* **Both ways at once.** Two bytes in front of every packet name the sender's end (client or server) and the connection's *epoch*, a 15-bit number the client chooses when it connects. So two nodes can be client and server of each other at the same time, and a late packet of a connection that is gone cannot be taken for one of the new connection. A node has at most one connection of each kind to a given peer: a second `.cli("node:1")` while one is open is `address in use`.
* **A peer that resets.** The new instance has no connection. A packet for a connection it does not have is answered with a reset, and the caller sees `connection lost (connection reset by the peer)`, the pending call fails, and the connection is closed for good. A peer that restarts and connects again has a new epoch, and the server replaces the old connection with the new one. A peer that is simply gone is found by the retry budget (about 30 s) or the call timeout.
* **The console.** Datagram channel 1 carries the Klong console's byte stream between a node and the node or the supervisor that last wrote to it. This is a proposal, not part of the bench protocol; the firmware's `README.md` has it.

### Why this and not the alternatives

* **Datagram ACKs instead of the reliability layer.** The datagram layer has its own ACK and resend. Using it per IPC chunk would make the sender wait for an ACK before the next datagram (stop and wait), which by estimate (not measured) costs a quarter of the throughput, and it still would not give the handshake, the ordering across datagrams or a reset. So the adapter sends without ACK and lets `rel` recover.
* **An adapter header with its own fragmentation.** Rejected: it would repeat what `rel` does for any packet link, and be a third copy of the logic beside the serial link's.
* **Refusing messages over 1024 bytes.** Rejected: the management library moves source files and results of any size, and `.cli` users do not expect a limit that TCP has not.
* **One 1024-byte datagram per packet.** Measured with the simulator (below): with 0.5% of frames lost, three 2.4 KB echoes take 13 s with packets of 254 bytes and 12 s with 126 (500 ms first timeout), 10 s with 62; with 2% lost, 104 s, 69 s and 36 s; without loss 5.3 s, 5.55 s and 6.0 s. A lost frame costs a whole packet and a timeout, and 126 loses 4% of the speed to halve that cost. The bench only loses frames when a queue overflows, so a larger packet may be right later; it is `NodeConfig::link_mtu`.
* **Datagrams per connection.** A connection does not own any datagram state: the lane has one sender (one message at a time, a 1 KB buffer) and one reassembly pool. `PacketLink::send` answers "busy" while the previous message is still leaving, which paces the connections to the lane's speed.

## In the simulator

`klong-sim` has a model of the switch (`switch.rs`): ports for N nodes and the supervisor, a byte timeline per lane at 115200 baud (a frame takes 1.389 ms each way), store and forward, a 256-frame FIFO per output port that drops what it cannot hold, the supervisor's PING every second, and optional random loss. `bench.rs` puts N boards (`klong-rt` with the `node` transport) behind it in one thread, on one virtual clock; while one board waits, the others run. `klong-bench` is that as a program with a tiny script language (a line `K> CODE` runs CODE on node K), and what follows is run by the documentation checker.

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

(The 0.118 s include 60 ms during which the nodes learn their numbers from the supervisor's first PING.) The management library across four nodes, with node 0 as the manager:

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
t = 1.073 s
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

From `native/klong-sim/tests/lane_ipc.rs` (run with `cargo test --release -p klong-sim --test lane_ipc -- --nocapture --test-threads=1`), virtual time, a clean switch unless said:

| What | Result |
|---|---|
| Connect (SYN, SYN, login, answer) | 27 ms |
| A small call, node to node | 30.7 ms (median and worst of 10) |
| A call that carries 830 bytes / 4 KB / 12 KB one way | 0.33 s / 1.55 s / 4.58 s: 2.5 / 2.6 / 2.6 KB/s |
| The same echoed back (so twice the bytes, one after the other) | 0.64 s / 3.07 s / 9.13 s |
| Ceiling | 2.88 KB/s (4 bytes per 16-byte frame, 720 frames/s); the layers' headers and the two framings cost 10% |
| `mgmt.all` of a small call over 3 nodes | 198 ms |
| 2 KB of source to each of 3 nodes with `mgmt.all(".E";...)` | 2.58 s |
| 4 nodes, 12 connections (each pair both ways), every call | 1.2 s in all |
| 5 s with the supervisor's pings going by | every node answered 6 PONGs and sent 5 STATUS; a call every second was unaffected; no frame was skipped or sent to an address the switch does not route |
| Fan-in of 3 x 9.6 KB into one port | 12.5 s, 90 frames lost to overflow (the queue reached 256), all answers right; with 3000 ppm random loss on top, still right |
| 3 echoes of 2.4 KB with 0 / 0.2% / 0.5% of frames lost | 5.55 s / 7.1 s / 12.1 s |
| A node reset 1.5 s into a 6 KB call | the call fails after 1.57 s with `connection reset by the peer`; the node, numbered again within a second, serves a new connection |

Over TCP the same small call costs microseconds; here it costs two datagrams of about ten frames each (15 ms each) and the delay of the runtime's polling. The throughput is the lane's.

### What the simulator cannot show

* Callbacks (a server function calling back into the client that waits for it, `.cli.h`) are not exercised here: the simulated boards share one thread, and "two runtimes on one thread cannot call each other" (see [IPC and Timers](ipc.md#limits)). They work over TCP between processes and nothing in the adapter treats them differently.
* The real processors' speed. The model charges 0.2 ms per 4096 evaluation steps, a stand-in; a Pico running soft-float `f64` will be slower, and a call that evaluates something heavy will be dominated by it.
* Anything about the UART, the interrupts, the USB and the supervisor's own implementation.

## The firmware

`native/boards/pico-bench/` is a Pico SDK application of about 240 lines of C around `klong-ffi` built with `--features lane` for `thumbv6m-none-eabi`. Its `README.md` has the build, the pins, the console and the test plan for two Picos. Facts, measured on 2026-11-16 (`native/boards/pico-bench/build.sh`, Pico SDK 2.3.1, Arm GNU 15.3.1, Rust stable):

| | |
|---|---|
| Flash | 608,276 bytes (`.bin`; 29% of 2 MB); 419,100 with `KLONG_OPT=z`. `.uf2`: 1,217,024 bytes |
| The runtime alone, `thumbv6m` linked as a probe, `opt-level=z`, with the library, timers and IPC | 292,724 bytes (`.text` 205,876, `.rodata` 86,840); the console without IPC 258,500. For `thumbv7em`: 249,464 without IPC (see [Porting to a Board](porting.md)) |
| Static RAM | `.data` 9.2 KB, `.bss` 56.8 KB including the 48 KB interpreter stack and the 4 KB lane ring |
| Heap | 144 KB limit (195 KB available), 16 KB held back |
| Stack | 48 KB static, the runtime told 42 KB (about 50 levels of Klong recursion at 0.8 KB each: unmeasured on this core) |
| Lane buffers (in the heap) | 7.8 KB of reassembly and send buffers for 6 sources, 2 KB console output, about 1 to 2.5 KB and up to 16 KB of buffers per connection |

The Cortex-M0+ has no compare-and-swap, no hardware divide and no FPU. The runtime needed no change for the first: its only atomics are loads and stores (`TickClock`, the heap counters, the interrupt flag), which `thumbv6m` has, and nothing uses `Arc`. 64-bit division and all floating point come from the compiler's software routines, as on `thumbv7em-none-eabi`.

`klong-ffi` with `--features lane` adds `klong_lane_init`, `klong_lane_pump`, `klong_lane_node` and `klong_lane_stats_get` to `klong.h`. `tests/lane.c` (run by `cargo test -p klong-ffi --features lane`) drives them from a C program that plays the supervisor: noise, a broadcast PING, a PING that numbers the node, the PONG and the STATUS a second later, ignored frames, and a console line sent in datagram frames and answered on channel 1.

## Open questions for the bench owner

* **Does the supervisor accept this node's PONG and STATUS?** They follow `software/worker/runtime.h`, which was read, and not run.
* **Channel 1 for the console, and its text protocol**, are proposals. If the bench has plans for channels or for sub-types `0xD0` to `0xD3`, say so.
* **Who sends datagrams from the supervisor?** The QT Py firmware has no datagram sender. A console line or a test over the lane needs one; `tools/lane_frames.py` shows the frames.
* **Queue depth.** The bench's lean profile keeps 2 frames per port, the statistics profile 256. The datagram and IPC layers assume the 256 of the owner's description: with 2 frames, concurrent senders lose most of what they send. One sender into one port loses nothing.
* **Packet size** (126 bytes) and the 500 ms first timeout are tuned for random loss in the simulator; the bench's losses come from queue overflow, in bursts. Measure on the bench.
* **Keepalive** is off, so a node that vanishes without a reset is found by the 30 s call timeout or the retry budget. A keepalive costs 3 frames per interval per connection.
* **Flash use.** The image keeps nothing in flash; a boot script would need a file system there.

## What only hardware can show

That the image boots and the USB console comes up; that the 48 KB stack and the stack switch are right; the UART at 720 frames/s while the interpreter evaluates (the 4 KB ring is sized for 1.4 s of a saturated lane); the real speed of 64-bit and floating-point arithmetic on the M0+; the supervisor's view of the node; and every figure above that depends on the processor. `native/boards/pico-bench/README.md` has the test plan for two Picos.
