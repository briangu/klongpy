# Management Node

This page covers the `mgmt` library (`klongpy/lib/mgmt.kg`), which turns one node into a management node: it keeps a registry of other nodes, checks that they are alive, relays calls to one node or to all of them, and stores source files on them. It is for people who run several native Klong nodes and want one place to reach them from. It is the role that a baseboard management controller has in a server, and it is not tied to any board.

!!! warning "Trust model"
    Anyone who can reach the management node can run any code on it **and on every node behind it**, and can read and replace the files of each of them. Nothing is encrypted, and `.srv.auth` sees a user name and a password that travel in clear text. Let the management node listen on an isolated or trusted network only, preferably on `127.0.0.1` or a private link. This is the trust model of [IPC and Timers](ipc.md), made larger: one connection to the management node is a connection to all of them.

!!! note "What has been run"
    Most of this page ran on one computer: the nodes and the management node are `klong` processes that talk over localhost TCP, and a second set of tests runs several runtimes in one thread over simulated lossy and serial links. On hardware, the library has loaded and management-library calls (`mgmt.call`) have run between two Raspberry Pi Picos of the [Pico bench](pico-bench.md). **Not run on a board**: a management board with nodes on UARTs, and a pulled cable. The hooks `node.reset` and `node.power` (below) are a place for a board to plug into, and no board does yet. The [status table](index.md#status-what-has-and-has-not-been-proven) has the one summary.

## The pieces

```
  desktop (KlongPy or klong)  --.cli-->  management node  --.cli-->  node n1
                                         (mgmt library)   --.cli-->  node n2
                                                          --.cli-->  node n3
```

A **node** is any runtime that serves IPC: a native `klong` with `.srv`. A node needs nothing installed. When the management node connects, it defines three names on the node (`mgmt.run`, `mgmt.job` and `mgmt.r`) and nothing else, and it writes nothing to the node's storage unless you ask it to.

The **management node** loads the library with `.l("mgmt")`. The native runtime carries the file in its image when it is built with the `ipc` feature, so no file is needed there. KlongPy finds it on its library path.

Nodes are named by strings and reached by transport address strings, the ones `.cli` takes. The library never looks inside an address, so a node can be at `"127.0.0.1:5001"`, `"tcp:10.0.0.11:5000"` or `"uart:1@921600"` on a platform that has that transport.

## Reference

| Call | Does |
|---|---|
| `mgmt.add(name;address)` | register a node and try to connect to it; a name that is already there gets the new address |
| `mgmt.nodes` | the registry, a dictionary from name to `:{["addr" a] ["h" handle or 0] ["state" s] ["seen" t] ["ver" v] ["wait" w] ["t" n]}` |
| `mgmt.names()` | the names, in the order they were added |
| `mgmt.status()` | name to `:{addr state seen age ver}`: the registry without the handles |
| `mgmt.timeout(name;seconds)` | the call timeout of one node |
| `mgmt.start(seconds)` | check every node now, and then every `seconds` |
| `mgmt.stop()` | stop checking |
| `mgmt.open(name)` | connect a node now (again, if it already has a link) |
| `mgmt.call(name;fn;args)` | call `fn` on a node with the list `args`, and return the result |
| `mgmt.each(names;fn;args)`, `mgmt.all(fn;args)` | the same call on several nodes at once, and on all of them: a dictionary from name to the result or to an error value |
| `mgmt.put(name;path;text)` | write a file on a node; returns the number of characters written |
| `mgmt.boot(name;text)` | write the node's `boot.kg`, which it runs when it starts |
| `mgmt.load(name;path)` | have the node load a file now (`.l(path)` there) |
| `mgmt.reset(name)`, `mgmt.power(name;on)` | call `node.reset(name)` and `node.power(name;on)`, when this board defines them; otherwise `name error: mgmt: this board has no node.reset` |

Settings: `mgmt.ct` is the call timeout a node gets when it is added (30 seconds), `mgmt.pi` is the interval that `mgmt.start` was given, and `mgmt.probe` is the request a check sends (`".ver()"`). The defaults are provisional. In `mgmt.nodes`, `state` is `"up"` or `"down"`, `seen` is the `.pc()` time of the last answer (-1 before the first), `ver` is what `.ver()` gave there, `wait` is 1 while a probe is unanswered, and `t` is the node's call timeout.

`fn` is a symbol (the node's function of that name), a string (source that the node evaluates; give a function literal such as `"{+/x}"`, and with `args` of `[]` the value of the source is the result), or a function (sent as its source). Passing a string like `"+/x"` that is not a function literal does not apply it to `args`.

## A worked example

Three nodes, a management node in front of them, and a desktop client, all on localhost. The commands are run from one shell. The ports are `8888` for the first node and the next ones above it, and `8891` for the management node; the checker of this documentation replaces `8888` with a free port.

Start three nodes. Each defines `sq` and serves one port:

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ for i in 1 2 3; do klong -e "sq::{x*x};.srv(\"127.0.0.1:$((8888+i))\")" > n$i.log 2>&1 & echo $! > n$i.pid; done; for i in $(seq 600); do [ -s n1.log ] && [ -s n2.log ] && [ -s n3.log ] && break; sleep 0.1; done
```

Write the management node's program. It registers the nodes, starts the liveness check every 0.3 seconds, defines one helper, and serves the management port. The ports are the arguments of the program, which a file reads from `.a`:

<!-- check-docs: requires-file native/target/release/klong; path native/target/release; continue -->
```console
$ cat > mgr.kg <<'EOF'
> .l("mgmt")
> mgmt.add("n1";"127.0.0.1:",.a@0)
> mgmt.add("n2";"127.0.0.1:",.a@1)
> mgmt.add("n3";"127.0.0.1:",.a@2)
> mgmt.start(0.3)
> states::{{(mgmt.nodes?x)?"state"}'mgmt.names()}
> .srv("127.0.0.1:",.a@3)
> EOF
$ klong mgr.kg $((8888+1)) $((8888+2)) $((8888+3)) $((8888+10)) > mgr.log 2>&1 & echo $! > mgr.pid; for i in $(seq 600); do lsof -nP -iTCP:$((8888+10)) -sTCP:LISTEN >/dev/null 2>&1 && break; sleep 0.1; done; cat mgr.log
```

The management node prints nothing. `klong` stays alive because a timer runs and a server listens.

Now the desktop. A client connects to the management node with `.cli` and sends it requests: a string to evaluate, or a list whose first element is the symbol of a function. The `mgmt` functions are called through the management node, so the list `[:mgmt.call "n2" :sq [7]]` is a request to call `mgmt.call("n2";:sq;[7])` there. The client is another `klong` here; it could be KlongPy.

<!-- check-docs: requires-file native/target/release/klong; path native/target/release; continue -->
```console
$ cat > desk.kg <<'EOF'
> m::.cli("127.0.0.1:",.a@0)
> .p(m("mgmt.names()"))
> .p(m([:mgmt.call "n2" :sq [7]]))
> .p(m([:mgmt.all :sq [6]]))
> .p(m("states()"))
> EOF
$ klong desk.kg $((8888+10))
[n1 n2 n3]
49
:{[n1 36] [n2 36] [n3 36]}
[up up up]
```

`mgmt.call` relayed the call to `n2` and returned 49. `mgmt.all` sent the request to all three nodes before waiting for any answer, so they computed at the same time, and returned a dictionary of the results. The helper `states` shows the registry entries. The list literal `[:mgmt.call ...]` is how the library itself takes symbols, because symbols in list literals are never looked up as variables. At strict levels 0 and 1 a quoted symbol is the value of a variable of that name when there is one, so a desktop that has its own `sq` and writes `:sq` outside a list literal sends its function and not the name.

### Deploy

`mgmt.put` writes a file on a node, and `mgmt.load` has the node load it. All three nodes run in this directory, so the file shows up here:

<!-- check-docs: requires-file native/target/release/klong; path native/target/release; continue -->
```console
$ cat > deploy.kg <<'EOF'
> m::.cli("127.0.0.1:",.a@0)
> .p(m([:mgmt.put "n2" "cube.kg" "cube::{x*x*x}"]))
> m([:mgmt.load "n2" "cube.kg"])
> .p(m([:mgmt.call "n2" :cube [3]]))
> r::m([:mgmt.all :cube [2]])
> .p(r?"n2")
> .p((r?"n1")?"message")
> EOF
$ klong deploy.kg $((8888+10))
13
27
8
n1: undefined function: cube
$ cat cube.kg
cube::{x*x*x}
```

`mgmt.put` returned the 13 characters it wrote. The call of `cube` on `n2` gave 27. `mgmt.all` gave 8 for `n2` and, for the two nodes that never loaded the file, an error value, which does not stop the others: its message starts with the node's name. `mgmt.boot(name;text)` writes `boot.kg` the same way. A board runs its `boot.kg` when it starts; the host `klong` does not, so to bring a node back with it you start it as `klong boot.kg`.

### Errors

An error on a node is raised on the management node as an error of the same kind, with the name of the node in front of the message, and it travels on to the desktop with that kind:

<!-- check-docs: requires-file native/target/release/klong; path native/target/release; continue -->
```console
$ printf 'm::.cli(%s)\nm([:mgmt.call "n1" "{1+x}" ["a"]])\n' $((8888+10)) | klong 2>&1 | sed -n 2p
  1:1: type error: n1: cannot apply + (Plus) to integer and string
```

A link that fails, and a call that times out, are `system` errors that start with the node's name. Errors of the management node itself start with `mgmt: `. In a fan-out, `mgmt.each` and `mgmt.all` do not raise: a dead node or a slow one becomes an error value in the result, and costs at most its own timeout, because the requests go out to every node first and each node has until its own deadline. The error value is a dictionary:

```
:{["error" kind] ["node" name] ["message" text]}
```

Klong has no error type, so a node whose own answer is such a dictionary cannot be told from a failed one.

### Liveness

Every interval each node is sent `mgmt.probe` without waiting. An answer makes the node `"up"`. A node that has not answered by the next check is `"down"`, and so is one whose link failed or whose call timed out; the check connects it again at the following check. Kill a node and the check notices:

<!-- check-docs: requires-file native/target/release/klong; path native/target/release; continue -->
```console
$ kill $(cat n3.pid); sleep 1.2
$ cat > after.kg <<'EOF'
> m::.cli("127.0.0.1:",.a@0)
> .p(m("states()"))
> r::m([:mgmt.all :sq [6]])
> .p(r?"n1")
> .p(r?"n3")
> EOF
$ klong after.kg $((8888+10))
[up up down]
36
:{[error system] [node n3] [message n3: not connected]}
```

With the check running, a fan-out does not connect a node that has no link. It reports `not connected` at once and leaves connecting to the check. Without that, a node that accepts a connection and then says nothing would hold every fan-out for the handshake time (5 seconds, not settable from Klong). A node that is busy for longer than the interval is reported `"down"` until it answers: an evaluation moves no bytes. A killed process is reported at the next check; a link that goes silent, at the check after the unanswered probe, so up to two intervals. If you want the node back, start it again on the same port; the next check connects it and marks it up.

While `mgmt.call` or a fan-out waits, the management node does nothing else: no check runs and other clients wait. One evaluation at a time.

Last, stop what the example started:

<!-- check-docs: requires-file native/target/release/klong; path native/target/release; continue -->
```console
$ for f in mgr n1 n2; do kill $(cat $f.pid); done
```

## `.try` and `.err`

The library needed two functions that Klong did not have: a way to catch an error and a way to raise one. The native runtime has them, always, and the management library uses them on the management node and on the nodes. KlongPy has neither.

`.try(f;args)` applies the function `f` to the list `args`. The result is `[1 result]` or, if there was an error, `[0 kind message]` with the kind's name as a string. An interrupt (Ctrl-C, `.x`) is not caught. `.err(kind;message)` raises an error of that kind with that message, for any kind except `interrupt`.

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ klong -e 'f::{x+"a"};.try(f;,1)'
[0 "type" "cannot apply + (Plus) to integer and string"]
$ klong -e 'g::{x+1};.try(g;,1)'
[1 2]
$ klong -e '.err("range";"bad index")'
1:1: range error: bad index
  .err("range";"bad index")
  ^^^^^^^^^^^^^^^^^^^^^^^^^
$ klong -e '.err("interrupt";"stop")'
1:1: range error: .err: interrupt is not an error kind
  .err("interrupt";"stop")
  ^^^^^^^^^^^^^^^^^^^^^^^^
```

A function literal with no argument, written where a value is expected, is evaluated on the spot: that is the rule of the language, and KlongPy does the same (`f::{x+1};f({41})` gives 42, because `f` receives 41). So `.try({1+"a"};[0])` raises the error while its arguments are evaluated, before `.try` runs, and `.timer("t";1;{0})` hands `.timer` the number 0. To pass such a function, give it a name or an argument:

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ klong -e 'h::{1+"a"};.try(h;[0])'
[0 "type" "cannot apply + (Plus) to integer and string"]
$ klong -e '.try({x;1+"a"};[0])'
[0 "type" "cannot apply + (Plus) to integer and string"]
```

KlongPy has no `.try` and no `.err`. KlongPy as the management process (next section) therefore works around them:

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ kgpy -e '.err("range";"x")' 2>&1 | head -1
<expr>:1:1: name error: undefined function: .err
```

The two functions cost about 1.7 KB of flash on `thumbv7em-none-eabihf` and 2.3 KB on `riscv32imac-unknown-none-elf`. Whether KlongPy should get them, and under these names, is open.

## KlongPy as the management process

The file loads in KlongPy at every strict level and is tested there (`TestKlongPyAsManagementNode` in `tests/test_native_mgmt.py`), with native nodes behind it. A native client can drive it through `.srv`. What works:

* the registry, and `mgmt.call` with values of every type;
* `mgmt.all` and `mgmt.each` when every link is up (an error on a node is still an error value, because the node catches it);
* `mgmt.put`, `mgmt.boot`, `mgmt.load`;
* the liveness check marking a killed node down.

What differs:

* `mgmt.add` does not connect; the first call does.
* An error of a node keeps its kind but does not carry the node's name.
* A link that fails or a node that is too slow during a fan-out fails the whole call.
* The check does not connect a node again; `mgmt.open(name)` does. The reason is that `.cli` in KlongPy blocks for minutes when nobody listens at the address.
* An error of the management node itself is a `name` error about `mgmt.error`, after the message is printed.
* At strict level 2 KlongPy cannot be asked whether `.try` exists, so set `mgmt.trap::0` by hand after loading the library.

The nodes behind it are native runtimes in every case.

## Limits and what is not done

* **Resources.** A load reads one statement at a time and keeps functions as their text. On the 64-bit host the library loads with `--heap-limit 44K` (it fails with `43K`; measured 2026-11-17), peaks at about 40 KB in all and holds about 18 KB more afterwards (`.mem()`). On a Pico it grew the resident memory by 5 to 6.4 KB ([Pico bench](pico-bench.md)). It is 12.6 KB of text in the image, of which about half is the header that is also in this page. A 32-bit build has not been measured.
* **Defaults.** The probe interval of 5 seconds in the library's example and the call timeout of 30 seconds are provisional. The keepalive of the packet layer and the probe do similar jobs, and one of them may be enough on a real UART.
* **Reconnect.** A node comes back only at a check, so a management node without `mgmt.start` does not reconnect a node by itself; `mgmt.open(name)` does.
* **Hooks.** `node.reset` and `node.power` are looked up by name when called. No board defines them.
* **No Python helper.** There is no `klongpy.mgmt` class; the client side is `.cli` to the management node, as above.
* **Boards.** The reference configuration (a management board with nodes on UARTs), the behaviour with a pulled cable, a node's `boot.kg` run at power-up after `mgmt.boot`, and the heap and flash of a real management board have not been tried. See [ESP32-P4](esp32-p4.md): that firmware has no IPC.
