# Inter-Process Communication (IPC) Capabilities

> **Security:** a peer that can reach this port can run any Klong code in your interpreter, including `.py` and `.sys`. Credentials and data are sent in clear text and there is no encryption. Use it on trusted networks only and bind to localhost where you can. See the [security model](https://github.com/briangu/klongpy#security-model-trusted-networks).

KlongPy interpreters can connect to each other, run code and call functions remotely, read and write remote variables, and receive callbacks. The functions are `.srv` (serve), `.cli` (connect and call), `.clid` (connect and treat the remote context as a dictionary), `.clic` (close), `.clit` (call timeout) and `.async` (call without waiting).

The examples below were run with a second interpreter in the same process, as the tests do. A line starting with `server?>` ran in the serving interpreter and a line starting with `client?>` in the connecting one. The port is 8888 throughout; where a result is an error, the line shows `error:` and the message.

## The protocol

The connection speaks the kdb+ IPC protocol: the same handshake, message framing and type codes, little-endian, with compression understood (capability 3). A q or other kdb+ client library can connect to a KlongPy `.srv`, send a string to evaluate, or call a Klong function by name, and a KlongPy `.cli` can connect to a q process. kdb+ is a trademark of KX. This is an independent implementation written from public documentation and Apache-licensed client code. It is tested against PeachQ, an open-source kdb+-compatible implementation, and not against kdb+ itself (see the [roadmap](roadmap.md)).

What goes across, in each direction:

| Klong value | kdb+ type sent | Klong value received |
|---|---|---|
| integer | long atom (int atom from a 32-bit build) | integer (boolean, byte, short, int and long atoms all arrive as integers) |
| float | float atom (real from a 32-bit build) | float |
| char below U+0080 | char atom | char |
| any other char | char vector of its UTF-8 bytes | string |
| string | char vector, UTF-8 | string |
| symbol | symbol atom | symbol |
| `:undefined` | the generic null `(::)` | `:undefined` |
| 1-D integer or float array | long or float vector | integer or float array |
| list of symbols | symbol vector | list of symbols |
| array of rank 2 or more, mixed or ragged list, list of strings | general list, elements encoded one by one | list; a rectangular numeric one becomes an array |
| dictionary | dictionary (99) | dictionary |
| function | lambda (100), context `kg`, source text | remote function (see below) |
| anything else (a Python object, a connection handle) | not sent: a type error | |

A `klongpy.db` table is sent as a kdb+ table (type 98) of its columns, after any pending inserts are committed. A table received becomes a `Table` when `klongpy.db` is loaded in the receiving process (`.py("klongpy.db")`), and otherwise a dictionary from column symbol to column. The table's index is not carried: the receiver must apply `.index` again. A keyed table from q arrives as a two-element list of the key table and the value table.

Storing a table on a server and reading it back (needs pandas and DuckDB, `pip install "klongpy[db]"`):

<!-- check-docs: requires duckdb pandas -->
```kgpy
server?> .py("klongpy.db")
1
server?> .srv(8888)
1
server?> store:::{}
:{}
server?> put::{store,x,,y;1}
:dyad
server?> get::{store?x}
:monad
client?> .py("klongpy.db")
1
client?> f::.cli(8888)
remote[localhost:8888]:fn
client?> T::.table([["a" [1 2 3]] ["b" [1.5 2.5 3.5]]])
a   b
1 1.5
2 2.5
3 3.5
client?> f(:put,"T",T)
1
client?> R::f(:get,,"T")
a   b
1 1.5
2 2.5
3 3.5
client?> R?"a"
[1 2 3]
client?> .schema(R)
["a" "b"]
server?> #store?"T"
3
```

Also received: timestamps, timespans, dates, times and the other temporal types arrive as their underlying integer (a datetime as its float), a GUID as an array of 16 integers, a kdb+ table as above, and any other q function type as an opaque value that prints as `:kdb-function` and can be sent back unchanged. Nulls and infinities are not translated: a long null arrives as -9223372036854775808 and a float null as NaN. A value with no kdb+ form is a type error on the sending side and nothing is sent.

## Addresses

`.cli` and `.clid` take an integer or a string:

| Argument | Connects to |
|---|---|
| `8888` | port 8888 on localhost |
| `"host:8888"` | port 8888 on `host` |
| `"host:8888:user:password"` | the same, sending a user name and password (the order of kdb+'s `hopen`) |

`"host:8888:user"` sends a user name with no password, and the password may itself contain `:`. A string that does not end in a port number is a type error. `.srv` takes a port, or the string `"bind:port"` to listen on one interface only.

## The .cli() Function

`.cli(x)` returns a function that runs commands on the server `x`. Call it with a string to evaluate Klong source there, or with a symbol and arguments to call a function by name. The arguments are applied as they are: a symbol stays a symbol and a string is data, never run as a program.

```kgpy
server?> .srv(8888)
1
client?> f::.cli(8888)
remote[localhost:8888]:fn
client?> f("1+1")
2
client?> f("avg::{(+/x)%#x}")
:monad
client?> f("avg(!100)")
49.5
client?> f(:avg,,!100)
49.5
client?> data::!100
[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99]
client?> f(:avg,,data)
49.5
```

To start a server from the shell, `kgpy -s 8888` runs the REPL with `.srv(8888)` evaluated first.

Applying a symbol with no arguments returns the remote value. For a function that value is a remote function proxy, which you call as if it were local:

```kgpy
server?> .srv(8888)
1
server?> avg::{(+/x)%#x}
:monad
server?> total::100
100
client?> f::.cli(8888)
remote[localhost:8888]:fn
client?> f(:total)
100
client?> q::f(:avg)
remote[localhost:8888]:fn:avg:monad
client?> q(!100)
49.5
```

A call with no arguments is sent as `` (`name; ::) `` and a niladic remote function runs with none.

## The .clid() Function

The KlongPy context is a dictionary. `.clid(x)` connects the same way as `.cli` and returns a dictionary that sets and gets values in the remote context. Given a connection from `.cli`, it shares that connection.

```kgpy
server?> .srv(8888)
1
client?> f::.cli(8888)
remote[localhost:8888]:fn
client?> d::.clid(f)
remote[localhost:8888]:dict
client?> d,[:foo 2]
remote[localhost:8888]:dict
client?> d?:foo
2
client?> d,[:bar "hello"]
remote[localhost:8888]:dict
client?> d?:bar
"hello"
client?> d,:fn,{x+1}
remote[localhost:8888]:dict
client?> t::d?:fn
remote[localhost:8888]:fn:fn:monad
client?> t(10)
11
server?> foo
2
server?> fn(1)
2
client?> d?:nosuch
error: symbol not found: nosuch
```

A set is the three-item request `("set"; name; value)`. A KlongPy server treats it as an assignment of a global unless the program defines its own `set`. A get of a name the server does not have is a name error.

## How functions cross the wire

A function literal is sent as its source text, and the receiver parses it. The names in its body are the receiver's names, not yours. Here the function `{x+k}` is defined on the client, where `k` is 100, but it runs on the server, where `k` is 7:

```kgpy
server?> .srv(8888)
1
server?> k::7
7
server?> twice::{x@x@y}
:dyad
client?> k::100
100
client?> f::.cli(8888)
remote[localhost:8888]:fn
client?> f(:twice,{x+k},1)
15
client?> inc::{x+k}
:monad
client?> inc(1)
101
client?> f(inc,1)
8
```

`f(:twice,{x+k},1)` calls the server's `twice` with the function and 1, which gives (1+7)+7. `f(inc,1)` sends the source of `inc` as the function to run remotely, and the server applies it to 1.

A projection has no source text. It is sent as an opaque value with the number of arguments it still takes. The receiver can hold it, store it and send it back, but cannot call it:

```kgpy
server?> .srv(8888)
1
server?> show::{x}
:monad
client?> f::.cli(8888)
remote[localhost:8888]:fn
client?> p::{x+y}(1;)
:monad
client?> p(2)
3
client?> f(:show,,p)
:monad
```

A remote function proxy (the result of `f(:name)` or `d?:name`) is bound to its connection and name. Calling it sends the call; closing the connection makes it fail.

## Remote Function Proxies and Enumeration

Proxies can be stored in an array and enumerated like local functions. With three functions defined on the server through `.clid`:

```kgpy
server?> .srv(8888)
1
client?> d::.clid(8888)
remote[localhost:8888]:dict
client?> d,:avg,{(+/x)%#x}
remote[localhost:8888]:dict
client?> d,:sum,{(+/x)}
remote[localhost:8888]:dict
client?> d,:max,{(x@>x)@0}
remote[localhost:8888]:dict
client?> a::d?:avg
remote[localhost:8888]:fn:avg:monad
client?> b::d?:sum
remote[localhost:8888]:fn:sum:monad
client?> c::d?:max
remote[localhost:8888]:fn:max:monad
client?> {x@,!100}'[a b c]
[49.5 4950.0 99.0]
```

Each proxy runs on the server with the range 0 to 99 as its argument, and the results come back as an array.

## Errors

A request that fails on the server is answered with an error whose text is `<wire name>: <message>`. There is no position and no traceback. The client raises the Klong error class that the wire name stands for, with the server's message, and `remote` set to true:

| Wire name | Error class on the client | `kind` |
|---|---|---|
| `parse` | `KlongSyntaxError` | `syntax` |
| `name` | `KlongNameError` | `name` |
| `rank` | `KlongArityError` | `arity` |
| `type` | `KlongTypeError` | `type` |
| `length` | `KlongShapeError` | `shape` |
| `index` | `KlongRangeError` | `range` |
| `os` | `KlongSystemError` | `system` |
| `python` | `KlongPythonError` | `python` |
| `internal` | `KlongInternalError` | `internal` |
| `wsfull` | `KlongMemoryError` | `memory` |
| `stop` | `KlongInterrupt` | `interrupt` |

The raised object is also a `KlongRemoteException`, so one `except` clause catches every error that came from the peer. Its `text` is the text as received, `wire_name` the name before the colon, and `peer` the address. Text that does not start with one of these names, such as the error a q process sends, raises `KlongRemoteError` with kind `remote` and the text as the message. The connection stays open after an error answer.

```kgpy
server?> .srv(8888)
1
client?> f::.cli(8888)
remote[localhost:8888]:fn
client?> f("[1 2 3]@10")
error: index 10 out of range for list of 3
client?> f("1+")
error: missing right operand for +
client?> f("nosuch(1)")
error: undefined function: nosuch
client?> f("1+1")
2
```

In Python the first error is a `KlongRangeError` with `kind == "range"` and `wire_name == "index"`.

Failures of the connection itself are `KlongSystemError` (`KlongIPCException` and its subclasses): the server refused the handshake (raised from `.cli`), the server could not be reached (shown by the first call, after the retries described below), the connection closed or was lost, a protocol error, a call timed out, or an oversize message. Every call waiting on a connection that fails gets the same error.

## Credentials: `.srv.auth`

A server accepts any client unless the program defines `.srv.auth`, a function of the user name and password (both strings) that returns 1 to accept and 0 to refuse. The client sends them in the address. They travel in clear text. A client of the old pickle protocol (see [`KLONGPY_IPC`](#klongpy_ipc-and-the-old-pickle-protocol)) sends no credentials, so a server with `.srv.auth` defined refuses it; the message size limit `ipc_max_message` applies to it as it does to kdb+ clients.

```kgpy
server?> .srv.auth::{(x~"bob")&(y~"secret")}
:dyad
server?> .srv(8888)
1
client?> f::.cli("localhost:8888:bob:secret")
remote[localhost:8888]:fn
client?> f("1+1")
2
client?> .cli("localhost:8888:bob:wrong")
error: remote[localhost:8888] closed the connection during the handshake (bad credentials, or it does not speak the kdb+ protocol)
client?> .cli(8888)
error: remote[localhost:8888] closed the connection during the handshake (bad credentials, or it does not speak the kdb+ protocol)
```

A refused client raises a system error from `.cli`. The server also drops a connection that does not complete the handshake within 5 seconds, and reads at most 256 bytes of it.

When nothing listens on the port (the connection is refused), `.cli` does not fail at once, and it does not raise. It tries up to 5 times and waits after each failed attempt, 5 seconds first and twice as long each time: 5, 10, 20, 40 and 80 seconds. The interpreter is blocked meanwhile (Ctrl-C stops the wait). After the fifth wait, about 155 seconds after the call (155.1 s measured), `.cli` returns the connection function as usual, and the failure shows on the first call through it: a system error, `connection not established`. A wrong password or a server that is not kdb+ is different: that is the handshake error above, raised from `.cli`. The retry numbers are the defaults `max_retries=5` and `retry_delay=5.0` of `HostPortConnectionProvider` in `klongpy/sys_fn_ipc.py`, which `.cli` uses without passing others; there is no Klong-level way to change them.

## Closing, Timeouts and `.clit`

`.clic(x)` closes a connection made by `.cli` or `.clid` and every proxy that shares it. It returns 1 if it closed one and 0 if it was already closed.

A call waits for the answer for 30 seconds by default. A call that gets none raises a system error whose message contains `timed out`, closes the connection, and fails every other call waiting on it with the same error, because the protocol has no request ids that would let the late answer be skipped. Pressing Ctrl-C during a wait does the same.

`.clit(h;seconds)` sets the timeout of a connection (a `.cli` function, a `.clid` dictionary, a proxy, or the handle passed to a server hook). `0` removes the limit. It returns 1, or 0 if the first argument is not a connection. From Python the same setting is the `call_timeout` attribute of the connection object.

```kgpy
server?> .pyf("time";"sleep")
1
server?> .srv(8888)
1
client?> f::.cli(8888)
remote[localhost:8888]:fn
client?> .clit(f;0.3)
1
client?> f("sleep(1)")
error: call timed out after 0.3s: remote[localhost:8888]
client?> f("1+1")
error: connection not established
client?> .clit(1;5)
0
client?> .clic(f)
0
```

## Limits and compression

The largest message a connection accepts is the `.system` key `ipc_max_message`, in bytes, 256 MiB when it is not set. It is read when a connection is made, on both ends, so set it before `.srv` and `.cli`. The length is checked from the message header before any memory is allocated, and a compressed message is checked against its declared uncompressed size. A message over the limit closes the connection. A list or string that declares more elements than the message holds, or nesting deeper than 64 levels, is a protocol error and also closes it.

```kgpy
server?> .system,"ipc_max_message",1000;1
1
server?> echo::{x}
:monad
server?> .srv(8888)
1
client?> f::.cli(8888)
remote[localhost:8888]:fn
client?> #f(:echo,,!50)
50
client?> #f(:echo,,!200)
error: connection closed by peer: remote[localhost:8888]
client?> f::.cli(8888)
remote[localhost:8888]:fn
client?> f("1+1")
2
```

The decoder never runs anything. Decompression is always supported. Sending compressed messages is off by default. From Python, `klong.ipc_compress = True` compresses a message when it is over 2000 bytes, the peer is not on loopback, and the result is under half the size, which is the rule kdb+ uses.

## Calls without waiting: `.async`

`.async(f;cb)` returns a function that calls `f` and then `cb` with the result, and returns 1 at once. Both must be functions. For a local function the call runs as a task on the Klong loop. When `f` is a connection from `.cli` or a remote function proxy, the request is sent without blocking and `cb` runs on the Klong loop when the answer arrives. An error answer or a lost connection is logged and `cb` is not called. With a connection the wrapper takes one argument, the request, as the connection does.

```kgpy
server?> .srv(8888)
1
server?> sq::{x*x}
:monad
client?> res::0
0
client?> cb::{res::x}
:monad
client?> f::.cli(8888)
remote[localhost:8888]:fn
client?> aq::.async(f(:sq);cb)
async::monad
client?> aq(9)
1
client?> res
81
client?> g::.async(f;cb)
async::monad
client?> g("6*7")
1
client?> res
42
client?> .async(f;1)
error: y must be a function
```

(The `client?>` lines wait a moment between steps so that the callback has run; at a prompt you would see `res` change a moment after the call.)

## Calls in both directions

While a call is waiting for its answer, the connection keeps serving the requests that arrive, so a server function can call back into the client that is waiting for it. Inside a function run for a client, `.cli.h` is the handle of that client's connection:

```kgpy
server?> .srv(8888)
1
server?> callme::{.cli.h(:hello,,x)*10}
:monad
client?> hello::{x+1}
:monad
client?> f::.cli(8888)
remote[localhost:8888]:fn
client?> f(:callme,,5)
60
```

Requests that arrive while no call is waiting are served on the Klong loop in the order they arrived, one at a time. Calls nest up to 8 levels; a request that would go deeper is answered with an `index` error. The supported model is one thread of control per connection. Callbacks that nest through a real kdb+ process are not supported, because kdb+ itself may return nested synchronous answers out of order.

## Server Lifecycle

`.srv(x)` returns 1 when the server is now listening and 0 when this interpreter was already serving. `.srv(0)` stops the server and drops its client connections; it returns 1 if a server was running and 0 if not. Each interpreter has its own server, so several interpreters can serve different ports.

```kgpy
server?> .srv(8888)
1
server?> .srv(8888)
0
server?> .srv(0)
1
server?> .srv(0)
0
```

If the port cannot be bound, `.srv` raises a system error instead of returning. With another process already listening on port 8888 the message looks like this:

```text
cannot listen on 0.0.0.0:8888: error while attempting to bind on address ... [errno 48] address already in use
```

## Server Callbacks

The server has four hooks, assigned as global functions. They run on the Klong loop, and a hook may call the client it is given.

| Hook | Called | Arguments |
|---|---|---|
| `.srv.o` | after a client has completed the handshake | the client's handle |
| `.srv.c` | when a client that connected disconnects | the client's handle |
| `.srv.e` | on an error that nobody is waiting for: an error in a request the client sent without waiting for an answer (an asynchronous message, such as a q client sends), and a connection error | the client's handle and the error text, a string |
| `.srv.auth` | during the handshake, see above | user name and password, strings |

`.srv.c` is not called for a client whose handshake was refused. A client that simply closes its end is a normal disconnect and does not go to `.srv.e`. Errors in requests that a client waits for are returned to that client and do not go to `.srv.e`.

```kgpy
server?> opened::[]
[]
server?> .srv.o::{opened::opened,x}
:monad
server?> .srv.c::{closed::x}
:monad
server?> .srv(8888)
1
client?> f::.cli(8888)
remote[localhost:8888]:fn
client?> g::.cli(8888)
remote[localhost:8888]:fn
server?> #opened
2
```

A hook may call back to the client as soon as it runs:

```kgpy
server?> back::0
0
server?> .srv.o::{back::x(:hello,,41)}
:monad
server?> .srv(8888)
1
client?> hello::{x+1}
:monad
client?> f::.cli(8888)
remote[localhost:8888]:fn
server?> back
42
```

## Building a pub-sub example

Using the server callbacks, it is easy to set up a pub-sub example where a client connects and subscribes to a server. Periodically the server calls the `update` function on the client with new data. The complete programs are in `examples/ipc/`, and the tests run them over the kdb+ protocol.

Server:

```
:"broadcast fake stock data to all subscribed clients"

:" Map of clients handles to their subscribed tickers "
clients:::{}

:" Called by clients to subscribe to ticker updates "
subscribe::{.d("subscribing client: ");.p(x);clients,.cli.h,,(clients?.cli.h),,x;.p(clients)}

:" Periodically called to broadcast updates to all subscribed clients "
send::{.d("sending to client");.p(x);x(:update,,{x,.rn()*50}'y)}
broadcast::{.p("sending messages to clients");{send(x@0;x@1)}'clients}
cb::{:[(#clients)>0;broadcast();.p("no clients to broadcast to")];1}
th::.timer("ticker";1;cb)

:" Setup the IPC server and callbacks "
.srv(8888)
.srv.o::{.d("client connected: ");.p(x);clients,x,,[]}
.srv.c::{.d("client disconnected: ");.p(x);x_clients;.d("clients left: ");.p(#clients)}
.srv.e::{.d("error: ");.p(x);.p(y)}
```

Client

```
:"Connect to the broadcast server"

.p("connecting to server on port 8888")

cli::.cli(8888)
.p(cli)

:" Called by server when there is a subscription update."
update::{.d("subscription update: ");.p(x)}

cli(:subscribe,,["MSFT" "GOOG" "AAPL"])
```

Running these is easy:

```bash
$ kgpy examples/ipc/srv_pubsub.kg
no clients to broadcast to
no clients to broadcast to
...
```

Once the client runs, the server begins to send updates to it:

```bash
$ kgpy examples/ipc/cli_pubsub.kg
connecting to server on port 8888
remote[localhost:8888]:fn
subscription update: [MSFT 16.310530573710896 GOOG 27.199690444331594 AAPL 35.81725374157503]
subscription update: [MSFT 43.28567690091258 GOOG 32.06719233158067 AAPL 47.306031721530864]
```

## `KLONGPY_IPC` and the old pickle protocol

Before this version, KlongPy sent Python pickles, which only another KlongPy could read and which let any peer run arbitrary Python in the receiving process. That protocol is deprecated and will be removed. It is still there for a transition, selected by the environment variable `KLONGPY_IPC`:

| Value | `.srv` accepts | `.cli` and `.clid` speak |
|---|---|---|
| `auto` (the default for `.srv`) | both, telling them apart by the first bytes | |
| `kdb` (the default for `.cli`) | only the kdb+ protocol | the kdb+ protocol |
| `legacy` | only the pickle protocol | the pickle protocol |

The first use of the pickle protocol in a process prints a `FutureWarning`: "the pickle IPC protocol is deprecated and will be removed". Upgrade the servers first, then the clients. A server on `auto` that sees an old client works, with the warning, unless it has `.srv.auth` defined (the old protocol has no credentials, so the client is refused); `KLONGPY_IPC=kdb` always refuses it. A value other than these three is a system error. The unpickling that the old protocol does is why it must only be used on a trusted network.

Two `kgpy` processes, with the server on the default mode and the client on `legacy`:

```bash
$ kgpy -s 8888
$ KLONGPY_IPC=legacy kgpy
?> f::.cli(8888)
... FutureWarning: the pickle IPC protocol is deprecated and will be removed ...
remote[localhost:8888]:fn
?> f("1+1")
2
```

With the server on `KLONGPY_IPC=kdb` the same client connects and its first call fails with `connection lost`. With the server on `legacy` and a default client, `.cli` raises a system error from the handshake.

## Known gaps

- There is no function to send a true asynchronous message (kdb+'s `neg[h]` form) from Klong. A KlongPy server does accept asynchronous messages from other clients.
- Calling through a connection from a function that was itself called by a q process (nested callbacks with kdb+) is not supported.
- There is no TLS. Anyone on the path can read the credentials and data.
- On the wire a function is its source text and the receiver parses it, so a function that was never given a source text (a projection) arrives opaque, and a function of the sender that uses names the receiver lacks fails on the receiver.

## See also

- [IPC and timers in the native runtime](native/ipc.md): IPC on the Rust runtime.
- [Management node](native/management-node.md): managing native nodes.
- [Timers](timer.md), [Web server](web_server.md), [WebSockets](websockets.md) and the [roadmap](roadmap.md) (testing against a real kdb+ is planned, not done).
