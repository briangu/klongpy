# Inter-Process Communication (IPC) Capabilities


KlongPy has powerful Inter-Process Communication (IPC) features that enable it to connect and interact with remote KlongPy instances. This includes executing commands, retrieving or storing data, and even defining functions remotely. These new capabilities are available via two new functions: .cli() and .clid().

## The .cli() Function

The .cli() function creates an IPC client. You can pass it either an integer (interpreted as a port on "localhost:<port>"), a string (interpreted as a host address "<host>:<port>"), or a remote dictionary (which shares the network connection and returns a remote function).

Use .cli() to evaluate commands on a remote KlongPy server, define functions, perform calculations, or retrieve values. You can also pass it a symbol to retrieve a value or a function from the remote server.

Start the IPC server:

```bash
$ kgpy -s 8888
Welcome to KlongPy REPL v0.4.0
author: Brian Guarraci
repo  : https://github.com/briangu/klongpy
crtl-d or ]q to quit

Running IPC server at 8888

?>
```

In a different terminal:

```bash
$ kgpy

?> f::.cli(8888)
remote[localhost:8888]:fn
?> f("avg::{(+/x)%#x}")
:monad
?> f("avg(!100)")
49.5
?> :" Call a remote function and pass a local value (!100) "
?> data::!100
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
 96 97 98 99]
?> f(:avg,,data)
49.5
```

Using remote function proxies, you can reference a remotely defined function and call it as if it were local:

```
?> q::f(:avg)
remote[localhost:8888]:avg:monad
?> q(!100)
49.5 
```

## The .clid() Function

As seen in Python interop examples, the KlongPy context is effectively a dictionary.  The .clid() function creates an IPC client that treats the remote KlongPy context as a dictionary, allowing you to set/get values on the remote instance.  Combined with the remote function capabilities, the remote dictionary makes it easy to interact with remote KlongPy instances.

Here are some examples:

```
?> :" Open a remote dictionary using the same connection as f "
?> d::.clid(f)
remote[localhost:8888]:dict
?> :" Add key/value pair :foo -> 2 to remote context "
?> d,[:foo 2]
?> :" Get the value for :foo key from the remote context "
?> d?:foo
2
?> d,[:bar "hello"]
?> d?:bar
hello
?> :" Assign a remote function to :fn "
?> d,:fn,{x+1}
?> t::d?:fn
remote[localhost:8888]:fn:monad
?> t(10)
11
```

These powerful capabilities allow for more effective use of distributed computing resources. Please be aware of potential security issues, as you are allowing a remote server to execute potentially arbitrary commands from your client. Always secure your connections and validate your commands to avoid potential attacks.

## Remote Function Proxies and Enumeration

Another powerful feature of KlongPy's IPC capabilities is the use of remote function proxies. These function proxies behave as if they were local functions, but are actually executed on a remote server. You can easily create these function proxies using .cli() or .clid(), and then use them as you would any other function.

One of the most powerful aspects of these remote function proxies is that they can be stored in an array and then enumerated. When you do this, KlongPy will execute each function in turn with the specified parameters.

For example, suppose you have created three remote function proxies:

```
?> d::.clid(8888)
?> d,:avg,{(+/x)%#x}
?> d,:sum,{(+/x)}
?> d,:max,{(x@>x)@0}
?> a = d?:avg
?> b = d?:sum
?> c = d?:max
```

You can then call each of these functions with the same parameter by using enumeration:

```
?> {x@,!100}'[a b c]
[  49.5 4950.    99. ]
```

In this example, KlongPy will execute each function with the range 0-99 as a parameter, and then store the results in the results array. The :avg function will calculate the average of the numbers, the :sum function will add them up, and the :max function will return the largest number in the range.

This makes it easy to perform multiple operations on the same data set, or to compare the results of different functions. It's another way that KlongPy's IPC capabilities can enhance your data analysis and distributed computing tasks.

## Closing Remote Function Proxies

Closing remote connections is done with the .clic() command.  Once it is closed, all proxies that shared that connection are now disconnected as well.

```
?> f::.cli(8888)
?> .clic(f)
1
```

## Async function calls

KlongPy supports async function calls.  While it works for local functions, its primarily for remote functions.

To indicate a function call should be async, the .async() function wraps the function and the supplied callback is called when complete.

Calling an async function results in 1, indicating it was executed.

```
?> fn::f(:avg)
remote[localhost:8888]:avg:monad
?> cb::{.d("remote done: ");.p(x)}
:monad
?> afn::.async(fn;cb)
async:monad
?> afn(!100)
1
?> remote done: 49.5
```

Note, the result of .async() is a function, so it's possible to reuse these.

## Synchronization

While the IPC server I/O is async, the KlongPy interpreter is single-threaded.  All remote operations are synchronous to make it easy to use remote operations as part of a normal workflow.  Of course, when calling over to another KlongPy instance, you have no idea what state that instance is in, but within the calling instance operations will be sequential.

## Server Callbacks

The KlongPy IPC server has 3 connection related callbacks that can be assigned to pre-defined symbols:

### Client connection open: `.srv.o`

Called when a new client connection is established.  The argument passed is the remote connection handle (fn) to the connecting client.  Note, handler functions should not call back to the client when called as it will produce a deadlock - the client is in the process of connecting to the server and not servicing requests.

```
.src.o::{.d("client has connected: ");.p(x)}
```

### Client connection close: `.srv.c`

Called when a client disconnects or drops the connection due to an error.  The passed argument is the client handle similar to `.srv.o`.

```
.src.e::{.d("client has disconnected: ");.p(x)}
```

### Client conncetion error: `.srv.e`

Called when there is a client error condition.  Arguments are the client handle and the exception that caused the error.

```
.src.e::{.d("client has had an error: ");.d(x);.d(" ");.p(y)}
```

## Building a pub-sub example

Using the server callbacks, it's easy to setup a pub-sub example where a client connects and then subscribes to a server. Periodically the server will call the update method on the client with new data.

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

One we run the client, the server will begin to send updates to the client:

```bash
$ kgpy examples/ipc/cli_pubsub.kg
connecting to server on port 8888
remote[localhost:8888]:fn
subscription update: [MSFT 16.310530573710896 GOOG 27.199690444331594 AAPL 35.81725374157503]
subscription update: [MSFT 43.28567690091258 GOOG 32.06719233158067 AAPL 47.306031721530864]
```
