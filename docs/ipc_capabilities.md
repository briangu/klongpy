# Inter-Process Communication (IPC) Capabilities

> **Security:** a peer that can reach this port can run any Klong code in your interpreter, including `.py` and `.sys`. Credentials and data are sent in clear text and there is no encryption. Use it on trusted networks only and bind to localhost where you can. See the [security model](https://github.com/briangu/klongpy#security-model-trusted-networks).

KlongPy interpreters, and q processes, can connect to each other, run code and call functions remotely, read and write remote variables, and receive callbacks. This page is for someone who runs a KlongPy server or client and needs the details: an example, the protocol and its security settings, the types that cross the wire, and each function.

The functions are `.srv` (serve), `.cli` (connect and call), `.clid` (connect and treat the remote context as a dictionary), `.clic` (close), `.clit` (call timeout) and `.async` (call without waiting).

## Example

Run the server and the client as two processes, for example in two terminals, running `kgpy` in each. A `server?>` line below is typed in the serving terminal and a `client?>` line in the other; the port is 8888 throughout.

```kgpy
server?> avg(v)::{(+/v)%#v}
:monad
server?> .srv(8888)
1
client?> f::.cli(8888)
remote[localhost:8888]:fn
client?> f("1+1")
2
client?> f(:avg,,!100)
49.5
```

`.cli` returns a function. A string argument is Klong source for the server to run. A symbol is the name of a server function, followed by its arguments.  `kgpy -s 8888` starts the REPL with `.srv(8888)` already evaluated, so a call to `.srv(8888)` there returns 0.

## Protocol and security

- **Protocol.** The connection speaks the kdb+ IPC protocol: the same handshake, message framing and type codes, little-endian, with compression understood (capability 3). A q or other kdb+ client can connect to `.srv`, and `.cli` can connect to a q process.
- **Independence.** kdb+ is a trademark of KX. This is an independent implementation written from public documentation and Apache-licensed client code. It is tested against [PeachQ](https://www.timestored.com/peachq/), an open-source kdb+-compatible implementation, and not against kdb+ itself (see the [roadmap](roadmap.md)).
- **Credentials.** A server accepts any client unless the program defines `.srv.auth`; see [Credentials](#credentials-srvauth).
- **Message size.** `ipc_max_message` limits the size of a message; see [Limits and compression](#limits-and-compression).
- **One protocol.** KlongPy 0.7 sent Python pickles, which let any peer run Python in the receiving process. That protocol was removed in 2.0, so nothing received is unpickled, and a 0.7 peer cannot connect; see [0.7 peers](#07-peers).
- **No encryption.** There is no TLS, and a peer that can reach the port can run Klong code, including `.py` and `.sys`.

## Types across the wire

What goes across in each direction:

| Klong value | kdb+ type sent | Klong value received |
|---|---|---|
| integer within 64 bits | long atom (int atom from a 32-bit build) | integer (boolean, byte, short, int and long atoms all arrive as integers) |
| float | float atom (real from a 32-bit build) | float |
| char below U+0080 | char atom | char |
| any other char | char vector of its UTF-8 bytes | string |
| string | char vector, UTF-8 | string |
| symbol | symbol atom | symbol |
| `:undefined` | the generic null `(::)` | `:undefined` |
| 1-D integer or float array | long or float vector | integer or float array |
| Python `bytes` or `bytearray` (also a `uint8` array) | byte vector (type 4) | integer array (a byte atom arrives as an integer) |
| list of symbols | symbol vector | list of symbols |
| array of rank 2 or more, mixed or ragged list, list of strings | general list, elements encoded one by one | list; a rectangular numeric one becomes an array |
| dictionary | dictionary (99) | dictionary |
| function | lambda (100), context `kg`, source text | remote function (see below) |
| integer beyond 64 bits (`2^64`) | not sent: `range error: integer does not fit 64 bits` (0.7 sent it by pickle) | |
| anything else (a Python object such as a set, complex number or `None`, a connection handle) | not sent: a type error that names the kind | |

A `klongpy.db` table is sent as a kdb+ table (type 98) of its columns, after any pending inserts are committed. A table received becomes a `Table` when `klongpy.db` is loaded in the receiving process (`.py("klongpy.db")`), and otherwise a dictionary from column symbol to column. The table's index is not carried: the receiver must apply `.index` again. A keyed table from q arrives as a two-element list of the key table and the value table.

Storing a table on a server and reading it back. This needs the `db` extra, `pip install "klongpy[db]"`. `,,` in `f(:get,,"T")` makes the one argument a list of one; with two arguments, as in `f(:put,"T",T)`, list them separately.

<!-- check-docs: requires duckdb pandas -->
```kgpy
server?> .py("klongpy.db")
1
server?> .srv(8888)
1
server?> store:::{}
:{}
server?> put(k;v)::{store,:{;k;v};1}
:dyad
server?> get(k)::{store@k}
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
client?> R@"a"
[1 2 3]
client?> .schema(R)
["a" "b"]
server?> #store@"T"
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
client?> f("avg(v)::{(+/v)%#v}")
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

To start a server from the shell, `kgpy -s 8888` starts the REPL with `.srv(8888)` already evaluated, so a call to `.srv(8888)` there returns 0.

Applying a symbol with no arguments returns the remote value. For a function that value is a remote function proxy, which you call as if it were local:

```kgpy
server?> .srv(8888)
1
server?> avg(v)::{(+/v)%#v}
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
server?> twice(g;v)::{g@g@v}
:dyad
client?> k::100
100
client?> f::.cli(8888)
remote[localhost:8888]:fn
client?> f(:twice,{x+k},1)
15
client?> inc(n)::{n+k}
:monad
client?> inc(1)
101
client?> f(inc,1)
8
```

`f(:twice,{x+k},1)` calls the server's `twice` with the function and 1, which gives (1+7)+7. `f(inc,1)` sends the source of `inc` as the function to run remotely, and the server applies it to 1.

Every function with a source travels as text: a function of `x`, `y` and `z`, a function with named parameters (any number, as `fn(a;b)::{a+b}`: the text is the definition, `fn` is only a placeholder, and the receiver keeps the function and defines nothing), a function with a locals list or a head destructuring, and one that calls itself by `.f`. The receiver rebuilds a function with the same parameter names and arity. A KlongPy peer and a native peer do this in both directions, as an argument and as a returned value.

A projection has no source text, and the format has no way to carry its bound arguments. It is sent as an opaque value with the number of arguments it still takes, not as an error. The receiver can hold it, store it and send it back, but cannot call it:

```kgpy
server?> .srv(8888)
1
server?> show(v)::{v}
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

Each proxy runs on the server with `!100` as its argument, and the results come back as an array.

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

A server accepts any client unless the program defines `.srv.auth`, a function of the user name and password (both strings) that returns 1 to accept and 0 to refuse. The client sends them in the address. They travel in clear text.

```kgpy
server?> .srv.auth(user;pw)::{(user~"bob")&(pw~"secret")}
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

## Connecting to a port where nothing listens

When the connection is refused, `.cli` blocks and retries up to 5 times, waiting 5, 10, 20, 40 and 80 seconds (about 155 seconds in all; Ctrl-C stops the wait). It then returns the connection function as usual, and the first call through it fails with a system error, `connection not established`.

A wrong password or a server that is not kdb+ is different: that is the handshake error above, raised from `.cli`. The retry numbers are the defaults `max_retries=5` and `retry_delay=5.0` of `HostPortConnectionProvider` in `klongpy/sys_fn_ipc.py`; Klong code cannot change them.

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
server?> echo(v)::{v}
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
server?> sq(n)::{n*n}
:monad
client?> res::0
0
client?> cb(r)::{res::r}
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
server?> callme(n)::{.cli.h(:hello,,n)*10}
:monad
client?> hello(n)::{n+1}
:monad
client?> f::.cli(8888)
remote[localhost:8888]:fn
client?> f(:callme,,5)
60
```

Requests that arrive while no call is waiting are served on the Klong loop in the order they arrived, one at a time. Calls nest up to 8 levels; a request that would go deeper is answered with an `index` error. The supported model is one thread of control per connection. Callbacks that nest through a real kdb+ process are not supported, because kdb+ itself may return nested synchronous answers out of order.

## Calling your own server

A synchronous call waits for the answer on the thread that makes it. A file, a timer callback or a hook runs on the interpreter's event loop thread, which is also the thread that serves requests from the connections of the same interpreter. A synchronous call from there to a server in the same process, `c::.cli(PORT); c("1+1")` against its own `.srv(PORT)`, or a server hook that calls the handle of a client in the same process, could never be answered. It is a system error at once, naming the cause. Send the request without waiting instead, with [`.async`](system-functions.md#async):

```text
a::.async(c;{.p(x)})
a("1+1")
```

The result is printed by the callback when it arrives. A call between two processes is not affected, and neither is a call between two interpreters in one Python process. The call timeout still applies to every other call (`.clit`, 30 seconds by default).

## Server Lifecycle

`.srv(x)` returns 1 when the server is now listening and 0 when this interpreter was already serving. `.srv(0)` stops the server and drops its client connections; it returns 1 if a server was running and 0 if not. Each interpreter has its own server, so several interpreters can serve different ports. A file run stays up while the server listens (see [The kgpy command](repl.md#options)).

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
server?> .srv.o(h)::{opened::opened,h}
:monad
server?> .srv.c(h)::{closed::h}
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
server?> .srv.o(h)::{back::h(:hello,,41)}
:monad
server?> .srv(8888)
1
client?> hello(n)::{n+1}
:monad
client?> f::.cli(8888)
remote[localhost:8888]:fn
server?> back
42
```

## Building a pub-sub example

A pub-sub pair uses the server callbacks: a client connects and subscribes, and the server periodically calls the `update` function on the client with new data. The complete programs are in `examples/ipc/`, and the tests run them over the kdb+ protocol.

Server:

```
:"broadcast fake stock data to all subscribed clients"

:" Map of client handles to their subscribed tickers "
clients:::{}

:" Called by clients to subscribe to ticker updates "
subscribe(tickers)::{.pf("subscribing client {}: {}";[;.cli.h;tickers]);clients,.cli.h,,(clients@.cli.h),tickers;.p(clients)}

:" Periodically called to send each subscribed client a random price per ticker "
send(c;tickers)::{.pf("sending to client {}";c);c(:update,,{[;x;.rn()*50]}'tickers)}
broadcast()::{.p("sending messages to clients");{[c tickers]::x;send(c;tickers)}'clients}
cb::{:[(#clients)>0;broadcast();.p("no clients to broadcast to")];1}
th::.timer("ticker";1;cb)

:" Setup the IPC server and callbacks "
.srv(8888)
.srv.o::{.pf("client connected: {}";x);clients,x,,[]}
.srv.c::{.pf("client disconnected: {}";x);x_clients;.pf("clients left: {}";#clients)}
.srv.e::{.pf("error: {} {}";[;x;y])}
```

Client:

```
:"Connect to the broadcast server"

.p("connecting to server on port 8888")

cli::.cli(8888)
.p(cli)

:" Called by server when there is a subscription update."
update(prices)::{.pf("subscription update: {}";,prices)}

.comment("----")
cli(:subscribe,,["AM.*"])
----
cli(:subscribe,,["AM.MSFT" "AM.GOOG" "AM.AAPL"])
```

Run the server, then the client:

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
subscription update: [[AM.MSFT 7.167480584416302] [AM.GOOG 5.384861539663505] [AM.AAPL 31.692928854915532]]
subscription update: [[AM.MSFT 41.134210064419605] [AM.GOOG 42.0242208432845] [AM.AAPL 24.240419848719025]]
```

## 0.7 peers

KlongPy 0.7 spoke a different protocol, which sent Python pickles. It was removed in 2.0, so a 0.7 process and a 2.0 process cannot exchange IPC messages in either direction. Upgrade both ends to 2.0 or later; see [Migrating to v2](migrating-to-v2.md#ipc-speaks-the-kdb-protocol).

- A 0.7 client connecting to a 2.0 server is closed at once, before anything is unpickled, and the server logs that the client speaks the pickle protocol of 0.7. The client sees `connection lost`.
- A 2.0 client connecting to a 0.7 server waits for the handshake. The 0.7 server answers nothing, so `.cli` raises a system error after 5 seconds: `handshake timed out after 5.0s`, with a note that a 0.7 server does not speak the kdb+ protocol.

The environment variable `KLONGPY_IPC` of the transition releases no longer exists; it is ignored like any unknown variable.

## Known gaps

- There is no function to send a true asynchronous message (kdb+'s `neg[h]` form) from Klong. A KlongPy server does accept asynchronous messages from other clients.
- Calling through a connection from a function that was itself called by a q process (nested callbacks with kdb+) is not supported.
- There is no TLS. Anyone on the path can read the credentials and data.
- On the wire a function is its source text and the receiver parses it, so a function that has no source text (a projection) arrives opaque, and a function of the sender that uses names the receiver lacks fails on the receiver.

## See also

- [IPC and timers in the native runtime](native/ipc.md): IPC on the Rust runtime.
- [Management node](native/management-node.md): managing native nodes.
- [Timers](timer.md), [Web server](web_server.md), [WebSockets](websockets.md) and the [roadmap](roadmap.md) (testing against a real kdb+ is planned, not done).
