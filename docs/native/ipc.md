# IPC and Timers

This page covers the network and timer functions of the native Klong runtime: `.srv`, `.cli`, `.clid`, `.clic`, `.clit`, `.async`, the `.srv.*` hooks, `.timer` and `.timerc`. It shows a native runtime and KlongPy calling each other in both directions, says how far compatibility with q and kdb+ has been tested, states the trust model, and lists the limits and the differences from KlongPy. It is for people who want a native node to talk to KlongPy or to a q process.

The functions are the same ones, with the same names and the same wire protocol, as in KlongPy, so read [IPC Capabilities](../ipc_capabilities.md) for the value mapping, `.cli` proxies, `.clid`, callbacks and the hooks in full. This page does not repeat that page. It covers what is different on the native side.

!!! warning "Trust model"
    A peer that connects to a native node can run any Klong code on it. With the file functions built in, it can read and write that node's files. There is no encryption and no TLS. A user name and password given to `.cli` travel in clear text, and `.srv.auth` only checks them, so it is a convenience and not a defence. Listen only on an isolated or trusted network, bind to `127.0.0.1` when you can, and never expose the port to an untrusted network. This is the same model as KlongPy's (see the [security model](https://github.com/briangu/klongpy#security-model-trusted-networks) in the README). It is also true of a management node, which controls every node behind it: see [Management Node](management-node.md).

## Where it runs

| Where | State |
|---|---|
| The `klong` program on the desktop, over TCP | built in and tested: with KlongPy in both directions, native to native, and against PeachQ |
| The simulated board `klong-sim` | built in (`.ver()` lists `ipc`), but it has no transport: `.srv` and `.cli` answer `system error: this platform has no tcp transport` |
| An ESP32-P4 | **not built in.** The firmware has no IPC: the C interface has no transports yet. Nothing about IPC has run on a board |
| The nodes of a packet switch that carries 16-byte frames (the Pico bench), `.cli("node:2")` | built in as the `node` transport; proven in the simulator, **not run on hardware**: see [the Pico bench](pico-bench.md) |
| Links over a UART, or a lossy packet radio | the code is there and was tested in memory with seeded faults (see [below](#links-that-are-not-tcp)); it has not been run on any wire |

The `klong` program registers one transport, TCP. A board would supply its own.

## The functions

| Function | Does |
|---|---|
| `.srv(ADDR)` | start listening; returns 1. `ADDR` is a port, or the string `"bind:port"` for one interface |
| `.cli(ADDR)` | connect; returns a function that evaluates a string there, or calls a symbol with arguments |
| `.clid(c)` | the remote context as a dictionary (shares the connection when given one from `.cli`) |
| `.clic(c)` | close a connection; returns 1, or 0 if it was not open |
| `.clit(c;seconds)` | set the call timeout of a connection (default 30; 0 for none) |
| `.async(f;cb)` | call `f`, then `cb` with the result, without waiting |
| `.srv.o`, `.srv.c`, `.srv.e`, `.srv.auth` | hooks you define: a client opened, a client closed, an error nobody waits for, and the login check |
| `.cli.h` | inside a function run for a client, the handle of that client's connection |
| `.timer(name;seconds;fn)`, `.timerc(handle)` | a repeating timer, and stopping it |

Addresses are as in KlongPy:

| Address | Meaning |
|---|---|
| `8888` | port 8888 on localhost (for `.srv`: every interface) |
| `"host:8888"` | port 8888 on `host` |
| `"host:8888:user:password"` | the same, with a user name and password |
| `"tcp:host:8888"` | the same as `"host:8888"`, with the transport named |
| `"name:address"` | any other transport that the platform has, with its own address; `.cli("uart:1@921600")` is how the management example in [Management Node](management-node.md) names a UART. The `klong` program has only `tcp` |

An address that is not one of these is a `type` error (`bad address a:b: expected a port or "host:port"`). A `.srv` on a port that is taken is a `system` error (`cannot listen on 127.0.0.1:8888: address in use`).

## A native server and a KlongPy client

The block starts a native node that serves two functions and prints a line when a client arrives and when it leaves, then runs a small KlongPy client file. `kgpy -l` runs the file and goes on to the REPL; the REPL sees an empty standard input and leaves, and `sed` keeps the first four lines of output, which are the client's. (`kgpy FILE`, without `-l`, did not return in the tests for this page when the file opened a connection, which is why `-l` is used here.)

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ klong -e 'sq::{x*x};total::{+/x};.srv.o::{.p("client connected")};.srv.c::{.p("client left")};.srv("127.0.0.1:8888")' > server.log 2>&1 & echo $! > server.pid; for i in $(seq 600); do [ -s server.log ] && break; sleep 0.1; done
$ cat > client.kg <<'EOF'
> c::.cli("127.0.0.1:8888")
> .p(c(:sq,,7))
> .p(c("1+2"))
> .p(c(:total,,!101))
> f::c(:sq)
> .p(f(12))
> .clic(c)
> EOF
$ kgpy -l client.kg | sed -n 1,4p
49
3
5050
144
$ for i in $(seq 100); do [ $(wc -l < server.log) -ge 3 ] && break; sleep 0.1; done; cat server.log; kill $(cat server.pid)
1
client connected
client left
```

Reading the client: `c(:sq,,7)` calls the symbol `:sq` with the argument list `[7]` (`,7` is the one-element list), `c("1+2")` evaluates a string on the server, and `f::c(:sq)` asks for the remote function itself, a proxy that you call as if it were local. The first line of `server.log` is the `1` that `.srv` returned. The server evaluated each request in its own interpreter, and it printed the two hook lines itself.

One trap, the same in KlongPy: a symbol that names a function **defined on the caller's side** is that function. With `sq` defined on both nodes, `c(:sq,,7)` and `f::c(:sq)` send the caller's own `sq` for the peer to apply (the way `c(inc,1)` sends the source of `inc`), not a call of the peer's. To call the peer's function whatever you have defined, send the list `c([:sq 7])` or the string `c("sq(7)")`, and for `.async` give it the handle, `af::.async(c;cb)`, and call `af([:sq 7])`.

## A KlongPy server and a native client

The other direction. The KlongPy node runs in the background for up to 25 seconds (`sleep 25 |` keeps its REPL from seeing the end of its input, and the file `kgpy.pid` holds its process number so that the last command can stop it), and the native `klong` is the client. `kgpy -s 8888` serves port 8888 and `-l` loads a file first.

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ printf 'sq::{x*x}\nmean::{(+/x)%%#x}\n' > server.kg
$ sleep 25 2>/dev/null | kgpy -s 8888 -l server.kg > kgpy.log 2>&1 & echo $! > kgpy.pid; for i in $(seq 600); do lsof -nP -iTCP:8888 -sTCP:LISTEN >/dev/null 2>&1 && break; sleep 0.1; done
$ klong -e 'c::.cli(8888);.p(c(:sq,,7));.p(c(:mean,,[1 2 3 4]));c("1+2")'
49
2.5
3
$ printf 'c::.cli(8888)\nc(:nosuch,,1)\n' | klong 2>&1 | grep -v '^ *$'
  :monad
  1:1: name error: symbol not found: nosuch
  c(:nosuch,,1)
  ^^^^^^^^^^^^^
  hint: reported by localhost:8888
$ kill $(cat kgpy.pid)
```

An error on the server comes back as an error of the same kind on the caller, and the `hint` line says which peer reported it. Both runtimes do this for every error kind of the language, and for `remote`, the kind of an error that a q peer signalled.

### Calls that do not wait

`.async(f;cb)` sends the request, returns 1, and runs `cb` with the answer when it arrives. Native callbacks run from the event loop, never inside an evaluation that is waiting. With `-e` the program keeps running after the last statement while an answer is still due, so the callback prints:

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ printf 'sq::{x*x}\n' > server.kg
$ sleep 25 2>/dev/null | kgpy -s 8888 -l server.kg > kgpy.log 2>&1 & echo $! > kgpy.pid; for i in $(seq 600); do lsof -nP -iTCP:8888 -sTCP:LISTEN >/dev/null 2>&1 && break; sleep 0.1; done
$ klong -e 'c::.cli(8888);g::.async(c(:sq);{.p("got ",$x)});g(6);.p("sent")'
sent
"sent"
got 36
$ kill $(cat kgpy.pid)
```

The answer to `g(6)` came after `sent`: the call did not wait. An error in an asynchronous call does not call `cb`. It is logged on the caller's error channel as `.async: remote call failed: ...`.

### Timeouts, logins and refused connections

A call that takes longer than the connection's timeout fails with a `system` error, and the connection is no longer usable (open a new one). A server that is busy evaluating moves no bytes meanwhile, so it looks dead to its peers for that long.

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ klong -e 'wait::{[t];t::.pc()+x;{.pc()<t}{x}:~0;x};.srv("127.0.0.1:8888")' > w.log 2>&1 & echo $! > w.pid; for i in $(seq 600); do [ -s w.log ] && break; sleep 0.1; done
$ printf 'c::.cli(8888)\n.clit(c;0.5)\nc(:wait,,2)\n' | klong 2>&1 | grep -v '^ *$'
  :monad
  1
  1:1: system error: call timed out after 0.5s: localhost:8888
  c(:wait,,2)
  ^^^^^^^^^^^
$ klong -e 'c::.cli(8888);.p(c(:wait,,0.2))'
0.2
"0.2"
$ kill $(cat w.pid)
```

A server that defines `.srv.auth` checks the user name and password in the address. A wrong or missing login is refused during the handshake, and the client's error says so:

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ klong -e '.srv.auth::{(x~"ann")&y~"pw"};sq::{x*x};.srv("127.0.0.1:8888")' > a.log 2>&1 & echo $! > a.pid; for i in $(seq 600); do [ -s a.log ] && break; sleep 0.1; done
$ klong -e 'c::.cli("127.0.0.1:8888:ann:pw");c(:sq,,5)'
25
$ klong -e 'c::.cli("127.0.0.1:8888:bob:no");c(:sq,,5)' 2>&1 | sed -n 1p
1:4: system error: 127.0.0.1:8888 closed the connection during the handshake (bad credentials, or it does not speak the kdb+ protocol)
$ kill $(cat a.pid); sleep 1
$ klong -e '.cli(8888)' 2>&1 | sed -n 1p
1:1: system error: cannot connect to localhost:8888: connection refused
```

(The user name and password are sent in clear text; this example is exactly what the trust box warns about.) Connecting does not return at once to a host that does not answer: the TCP transport waits at most 5 seconds in `connect`.

## Timers

`.timer(name;seconds;fn)` starts a repeating timer and returns a handle. `seconds` may be fractional. `fn` is called with no arguments every interval; its result decides what happens: 0 stops the timer, anything else continues. `.timerc(handle)` stops a timer and returns 1, or 0 if it was not running. A callback that fails cancels its timer, and the error is written to the error channel as `timer NAME callback failed; cancelling timer: ...`.

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ klong -e 'n::0;cb::{n::n+1;.p(n);:[n<3;1;0]};.timer("count";0.2;cb)'
:timer
1
2
3
```

As with a server, `klong` stays alive while a timer runs, so the program above ended on its own when the callback returned 0. In the REPL, timers and the IPC connections are served while the REPL waits for a line.

Time on a native node is cooperative. **Nothing fires while an evaluation or a blocked IPC call runs.** A timer that came due meanwhile fires once afterwards and then at its next multiple of the interval, counted from its start. A timer is not a way to run work at an exact moment.

KlongPy and native differ in one respect. KlongPy looks up a named callback again at each tick, so redefining `cb` changes what a running timer calls. The native `.timer` keeps the function value it was given. After `cb::{.p("old");1}` and `.timer("t";0.2;cb)`, a later `cb::{.p("new");0}` changes what a KlongPy timer prints from `old` to `new`, and leaves the native timer printing `old`. The library of the management node avoids the difference by passing a function that calls a name: `{mgmt.tick()}`.

## Events, one task

All of this runs on the one task that evaluates Klong code. A client call sends its request and then loops: move bytes, take the next event of that connection, and serve it if it is a request, until its own answer arrives, its timeout passes or the console sends Ctrl-C. That is why a server function can call back into the client that is waiting for it (the `.cli.h` handle, as in KlongPy), and why calls can nest. The nesting is limited to 8 levels, and the 9th is answered with an error (`nested calls deeper than ...`). Requests that arrive while nothing waits are served from the event loop, one per connection at a time, in arrival order.

Ctrl-C on the console ends a blocked call and closes that connection.

While a call waits, the node does nothing else: no timer fires and other clients wait. This matters for the [management node](management-node.md), which waits for its nodes.

## Compatibility with q and kdb+

The native runtime speaks the kdb+ IPC protocol, as KlongPy does: the handshake, the framing, the type codes, little endian, compression understood (capability 3), compression of what is sent when it pays. A q client library can connect to `.srv` and send a string or call a function by name, and `.cli` can connect to a q process.

What was tested: the codec against 119 shared test vectors (decode, reject, round trip byte for byte), through both the codec and the session. And, with the process tests of `tests/test_native_ipc.py`, a native client with **PeachQ 0.92.20261115** as the server, and PeachQ's `hopen` as a client of a native `.srv`. PeachQ is a third-party implementation of q. The tests run only when the environment variable `KLONG_TEST_PEACHQ` names a PeachQ executable. On this machine on 2026-11-16 no such executable was set, so those 11 tests were skipped, and they have **not been run again for this page**; their result is as recorded when the feature was built. The runtime has not been tested against a KX kdb+ process. What a kdb+ server prints or accepts that PeachQ does not is not known.

## Differences from KlongPy

| | KlongPy | native |
|---|---|---|
| A list that mixes integers and floats | object array | arrives as a float array |
| Dictionary keys from a peer | any | numbers, chars, strings and symbols; a list as a key is a `type` error for that one message, and the connection stays up |
| A symbol longer than 255 bytes | | `range` error for that message |
| Bytes that are not valid UTF-8 in a string | replaced by U+FFFD | replaced by U+FFFD (strings are always valid text) |
| An error signalled by q with a bare name (`type`, `length`) | kind `remote` | kind `remote` |
| `.timer` with a named callback | looked up at every tick | the function value given is kept |
| Python objects, `klongpy.db` tables | sent as tables | not available (no Python, no tables) |
| Connect | blocks, with retries that can take minutes against a host that does not answer | blocks at most 5 seconds |
| Two interpreters in one process | allowed | not on two threads at once (the nesting bound and a deferred-failure flag are per process) |

The error kinds an error takes on the way: each kind of the language arrives as the same kind on the other side, so a `range` error on a native server is a `KlongRangeError` in KlongPy and a `range` error again when it travels on to another native node. The `remote` kind is new in the native runtime, as it is in KlongPy.

## Limits

* **Message size.** The default limit is 256 MiB, which is a figure for a desktop. A device sets it from its heap: decoding takes about 1 to 2 times the message for a vector and, at worst, about 70 bytes of heap per byte of message for a dictionary of bytes. A 1,000,000-element float array was sent each way between KlongPy and native.
* **Nesting.** Values nest at most 64 levels on the wire, calls at most 8.
* **Memory per connection.** About 1 to 2.5 KB of heap per connection, plus one read buffer shared by all connections (64 KiB on a 64-bit host, 1 KiB on a 32-bit target) and the message being received. On this machine on 2026-11-16 `native/scripts/ipc_bench.py` measured 2149 bytes per idle connection on a server and 2303 on a client, and 66 KB left over after 100 connections had closed.
* **Symbols from peers are never freed.** A peer can grow the symbol table; the heap limit bounds it.
* **Speed.** One megabyte of integers echoed over localhost TCP took 1.95 ms per round trip between two native nodes, 1.97 ms from KlongPy to native, 1.42 ms from native to KlongPy and 1.35 ms from KlongPy to KlongPy (the same script, 20 repetitions, the same day). A native node polls its sockets without being told they are ready and sleeps between polls.
* **A connection to an address nobody listens on, in the OS's ephemeral port range.** `.cli("localhost:PORT")` tries the addresses `localhost` has in order, the IPv6 one first. With nothing listening there and PORT among the ports the OS gives to outgoing connections, a socket can be given PORT as its own port and connect to itself (TCP's simultaneous open), and then reads its own greeting as the server's: `bad capability byte in the server's reply`. The host transport refuses a connection whose two ends are the same address, and goes on to the next address. (It showed as a failure of one run in several of the management page of these docs on 2026-11-16, until the checker took its ports from below the ephemeral range.)
* **A long evaluation moves no bytes.** A peer on a packet link sees no acknowledgements meanwhile, and with the keepalive on, an evaluation longer than the keepalive budget looks like a dead link.
* **Callbacks within one process.** Two runtimes on one thread cannot call each other, because each waits for the other. Calls between processes nest as described.

### Links that are not TCP

The IPC code does not do I/O itself. A platform supplies transports, and the runtime puts a reliability layer on any link that is not a reliable stream. For a UART there is a framing with COBS and a CRC-16, and over it a go-back-N layer with a window of 16 packets and a retry budget. These were tested in memory, with seeded loss, duplication and reordering of 1 to 30 percent and with damaged bytes. At 30 percent a stream arrived intact only with a retry budget raised from 8 to 16, and with 100 percent loss the session fails with a timeout after 8 retransmissions. A CRC-16 cannot make "never delivers a corrupted value" absolute: a damaged frame passes with a probability of about 1.5 in 100,000. None of this has run on a real UART, and the `klong` program cannot reach these transports, so the process tests are all TCP. See [Porting to a Board](porting.md) for what a board supplies.

### Links that carry messages, not bytes

A packet link does not have to be a serial line. If the network delivers bounded messages (a radio, a switch, a bus with frames), a platform makes a `PacketLink` over it: `mtu` is the largest message, `send` and `recv` move one message, and the same reliability layer runs on top. An IPC message larger than the network's message is not a problem and not an error: it is part of the byte stream that the layer cuts into packets. The link may also give the layer timeouts that suit it (`PacketLink::tuning`: handshake timeout, retries, keepalive), because the defaults assume a link of some KB/s or more. The one such network so far is the Pico bench's switch, whose only unit is a frame of 4 bytes and whose largest message is 1024 bytes, at 2.9 KB/s on its first 115200 baud lanes and 112 KB/s at 4.5 Mbaud (the simulator's results: a small call in 31 ms and 1.0 ms, bulk data at 2.6 and 102 KB/s): `.cli("node:2")` connects to node 2 over it, with the kdb+ bytes unchanged. It is described, with its limits and its simulator results, in [Native IPC on the 4-Pico Bench](pico-bench.md); it has not run on a real board.
