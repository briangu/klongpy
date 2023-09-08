
![Unit Tests](https://github.com/briangu/klongpy/workflows/Unit%20Tests/badge.svg)

# KlongPy

KlongPy is a vectorized Python port of the [Klong](https://t3x.org/klong) [array language](https://en.wikipedia.org/wiki/Array_programming) and emphasizes Python interop making it easy to integrate Python's rich ecosystem while getting the succinctness of Klong.  

Using [CuPy](https://github.com/cupy/cupy), you have the flexibility of both CPU and GPU backends.

Leveraging [NumPy](https://numpy.org/), an [Iverson Ghost](https://analyzethedatanotthedrivel.org/2018/03/31/NumPy-another-iverson-ghost/) that traces its roots back to APL, as its runtime target the runtime may target either GPU ([CuPy](https://github.com/cupy/cupy)) or CPU backends.

The project builds upon the work of [Nils M Holm](https://t3x.org), the creator of the Klong language, who has written a comprehensive [Klong Book](https://t3x.org/klong/book.html) for anyone interested in diving deeper. In short, if you're a data scientist, researcher, or just a programming language enthusiast, KlongPy may just be the next thing you want to check out.

# Overview

KlongPy is both an Array Language runtime and a set of powerful tools for building high performance data analysis and distributed computing applications.  Some of the features include: 

* [__Array Programming__](https://en.wikipedia.org/wiki/Array_programming): Based on [Klong](https://t3x.org/klong), a concise, expressive, and easy-to-understand array programming language. Its simple syntax and rich feature set make it an excellent tool for data scientists and engineers.
* [__Speed__](#performance): Designed for high-speed vectorized computing, enabling you to process large data sets quickly and efficiently on either CPU or GPU.
* [__Fast Columnar Database__](#fast-columnar-database): Includes integration with [DuckDb](http://duckdb.org), a super fast in-process columnar store that can operate directly on NumPy arrays with zero-copy.
* [__Inter-Process Communication (IPC)__](#inter-process-communication-ipc-capabilities): Includes built-in support for IPC, enabling easy communication between different processes and systems. Ticker plants and similar pipelines are easy to build.
* [__Table and Key-Value Store__](#table-and-key-value-stores): Includes a simple file-backed key value store that can be used to store database tables or raw key/value pairs.
* [__Python Integration__](#python-integration): Seamlessly compatible with Python and modules, allowing you to leverage existing Python libraries and frameworks.
* [__Web server__](#web-server): Includes a web server, making it easy to build sites backed by KlongPy capabilities.
* [__Timers__](#timer): Includes periodic timer facility to periodically perform tasks.


# Examples

Consider this simple Klong expression that computes an array's average: `(+/a)%#a`. Decoded, it means "sum of 'a' divided by the length of 'a'", as read from right to left.

Below, we define the function 'avg' and apply it to the array of 1 million integers (as defined by !1000000)

Let's try this in the KlongPy REPL:

```Bash
$ rlwrap kgpy

Welcome to KlongPy REPL v0.3.78
author: Brian Guarraci
repo  : https://github.com/briangu/klongpy
crtl-d or ]q to quit

?> avg::{(+/x)%#x}
:monad
?> avg(!1000000)
499999.5
```

Now let's time it (first, right it once, then 100 times):

```
?> ]T avg(!1000000)
total: 0.0032962500117719173 per: 0.0032962500117719173
?> ]T:100 avg(!1000000)
total: 0.10882879211567342 per: 0.0010882879211567343
```

We can also import Python modules to use directly in Klong language.  

Here we import the standard Python math library and redefine avg to use 'fsum':

```
?> .py("math")
1
?> favg::{fsum(x)%#x}
:monad
?> favg(!1000000)
499999.5
```

Notice that using fsum is slower than using Klong '+/'.  This is because the '+/' operation is vectorized while fsum is not.

```
?> ]T favg(!1000000)
total: 0.050078875152394176 per: 0.050078875152394176
?> ]T:100 favg(!1000000)
total: 2.93945804098621 per: 0.029394580409862103
```

To use KlongPy within Python, here's a basic outline:

```python
from klongpy import KlongInterpreter

# instantiate the KlongPy interpeter
klong = KlongInterpreter()

# define average function in Klong (Note the '+/' (sum over) uses np.add.reduce under the hood)
klong('avg::{(+/x)%#x}')

# create a billion random uniform values [0,1)
data = np.random.rand(10**9)

# reference the 'avg' function in Klong interpeter and call it directly from Python.
r = klong['avg'](data)

print(f"avg={np.round(r,6)}")
```

And let's run a performance comparison between CPU and GPU backends:

```python
import time
from klongpy.backend import np
from klongpy import KlongInterpreter

klong = KlongInterpreter()
klong('avg::{(+/x)%#x}')

data = np.random.rand(10**9)

start = time.perf_counter_ns()
r = klong['avg'](data)
stop = time.perf_counter_ns()

print(f"avg={np.round(r,6)} in {round((stop - start) / (10**9), 6)} seconds")
```

Run (CPU)

    $ python3 tests/perf_avg.py
    avg=0.5 in 0.16936 seconds

Run (GPU)

    $ USE_GPU=1 python3 tests/perf_avg.py
    avg=0.500015 in 0.027818 seconds

# Python integration

Seamlessly blending Klong and Python is the cornerstone of KlongPy, enabling you to utilize each language where it shines brightest. For instance, you can integrate KlongPy into your ML/Pandas workflows, or deploy it as a powerhouse driving your website backend.

The charm of KlongPy lies in its dictionary-like interpreter that hosts the current KlongPy state, making it incredibly simple to extend KlongPy with custom functions and shuttle data in and out of the interpreter.

Imagine your data processed elsewhere, just set it into KlongPy and watch as the Klong language works its magic, accessing and manipulating your data with effortless ease. Even more, Python lambdas or functions can directly be exposed as Klong functions, adding an array of powerful tools to your Klong arsenal.

KlongPy indeed is a force multiplier, amplifying the power of your data operations.

## Function example

Call a Python function from Klong:

```python
from klongpy import KlongInterpreter
klong = KlongInterpreter()
klong['f'] = lambda x, y, z: x*1000 + y - z
r = klong('f(3; 10; 20)')
assert r == 2990
```

and vice versa, you can call a Klong function from Python:

```python
from klongpy import KlongInterpreter
klong = KlongInterpreter()
klong("f::{(x*1000) + y - z}")
r = klong['f'](3, 10, 20)
assert r == 2990
```

## Data example

Since the Klong interpreter context is dictionary-like, you can store values there for access in Klong:

```python
data = np.arange(10*9)
klong['data'] = data
r = klong('1+data')
assert r == 1 + data
```

Variables may be directly retrieved from KlongPy context:

```python
klong('Q::1+data')
Q = klong['Q']
print(Q)
```

## Python library access

Python functions, including lambdas, can be easily added to support common operations.  

In order to be consistent with Klong language, the paramters of Python functions may have at most three paramters and they must be x, y, and z.

```Python
from datetime import datetime
from klongpy import KlongInterpreter
klong = KlongInterpreter()
klong['strptime'] = lambda x: datetime.strptime(x, "%d %B, %Y")
klong("""
    a::strptime("21 June, 2018")
    .p(a)
    d:::{};d,"timestamp",a
    .p(d)
""")
```

prints the following dictionary to the console:

```
2018-06-21 00:00:00
{'timestamp': datetime.datetime(2018, 6, 21, 0, 0)}
```

You can go one step further and call back into Python from Klong with the result:

```Python
from datetime import datetime
from klongpy import KlongInterpreter
klong = KlongInterpreter()
klong['strptime'] = lambda x: datetime.strptime(x, "%d %B, %Y")
klong['myprint'] = lambda x: print(f"called from KlongPy: {x}")
klong("""
    a::strptime("21 June, 2018")
    myprint(a)
    d:::{};d,"timestamp",a
    myprint(d)
""")
```
outputs
```
called from KlongPy: 2018-06-21 00:00:00
called from KlongPy: {'timestamp': datetime.datetime(2018, 6, 21, 0, 0)}
```

## Loading Python Modules directly into KlongPy

KlongPy has the powerful ability to load Python modules directly. This can be extremely useful when you want to utilize the functionality offered by various Python libraries, and seamlessly integrate them into your KlongPy programs.  

Here is an example of how you can load a Python module into KlongPy:

```bash
$ rlwrap kgpy

Welcome to KlongPy REPL v0.3.76
author: Brian Guarraci
repo  : https://github.com/briangu/klongpy
crtl-d or ]q to quit

?> .py("math")
1
?> sqrt(64)
8.0
?> fsum(!100)
4950.0
```

In order to keep consistency with Klong 3-parameter function rules, KlongPy will attempt to remap loaded functions to use the x,y and z convention.  For example, in the Python math module, fsum is defined as fsum(seq), so KlongPy remaps this to fsum(x) so that it works within the runtime.


## Loading Custom Python Modules

Custom modules can be written for KlongPy in the same way as any Python module, the main
difference is that they don't need to be installed (e.g. via pip).

Simply create a directory with a __init__.py and appropriate files, as in:


```Python
# __init__.py
from .hello_world import hello
```

```Python
# hello_world.py

def hello():
    return "world!"

def not_exported():
    raise RuntimeError()
```

Now, you can import the module with the .py command and run the "hello" function.

```bash
$ rlwrap kgpy

Welcome to KlongPy REPL v0.3.76
author: Brian Guarraci
repo  : https://github.com/briangu/klongpy
crtl-d or ]q to quit

?> .py("tests/plugins/greetings")
1
?> hello()
world!
```

# Fast Columnar Database 

KlongPy provides a module "klongpy.db" that includes DuckDb integration.  DuckDb can operate directly on NumPy arrays, which allows for zero-copy SQL execution over pre-existing NumPy data.

## Tables

```
?> .py("klongpy.db")
?> t::.table([["a" [1 2 3]] ["b" [2 3 4]]])
a b
1 2
2 3
3 4
?> t,"c",,[3 4 5]
a b c
1 2 3
2 3 4
3 4 5
```

Indexes (one or more columns) can be created on a table.  The current indexes can be seen in the table discription prefix.

```
?> .index(t; ["a"])
['a']
```

When a column is indexed, it appears with an asterisk in the pretty-print format:

```
?> t
a* b
 1 2
 2 3
 3 5
```

Inserting a row with a pre-existing value at an index results in an update:

```
?> .insert(t, [3 5 6])
a* b c
 1 2 3
 2 3 4
 3 5 6
```

Indexes may be reset via .rindex().  True is returned if the index was reset.

```
?> .rindex(t)
1
```

## Database

Databases are created from a map of table names to table instances.  A database instance is a function which accepts SQL and runs it over the underlying tables.  SQL results are NumPy arrays and can be directly used in normal KlongPy operations.

```
?> T::.table(,"a",,[1 2 3])
a
1
2
3
?> db::.db(:{},"T",,T)
:db
?> db("select * from T")
[1 2 3]
```

Since KlongPy uses DuckDb under the hood, you can perform sophisticated SQL over the underlying NumPy arrays.  

For example, it's easy to use JOIN with this setup:

```
d::[]
d::d,,"a",,[1 2 3]
d::d,,"b",,[2 3 4]
T::.table(d)

e::,"c",,[3 4 5]
G::.table(e)

q:::{}
q,"T",,T
q,"G",,G
db::.db(q)
```

We can now issue a JOIN SQL:

```
?> db("select * from T join G on G.c = T.b")
[[2 3 3]
 [3 4 4]]
```

## Pandas DataFrame integration

Tables are backed by Pandas DataFrames, so it's easy to integrate Pandas directly into KlongPy via DuckDb.

```Python
from klongpy import KlongInterpreter
import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40]}
df = pd.DataFrame(data)

klong = KlongInterpreter()
klong['df'] = df
r = klong("""
.py("klongpy.db")
t::.table(df)
db::.db(:{},"people",t)
db("select Age from people")
""")
```

# Table and Key-Value Stores

To support the KlongPy database cababilities, the klongpy.db module includes a key-value store capability that allows for saving and retreiving tables from disk.  There is a more generic key-value store as well as a TableStore.  The TableStore merges tables when writing to disk, while the generic key-value store writes raw serialized data and doesn't consider the contents.

Key-value stores operate as dictionaries, so setting a value updates the contents on disk and reading a value retrieves it.  Similar to Klong dictionaries, if the value does not exist, then the undefined value is returned.

### TableStore

Since KlongPy Tables are backed by Pandas DataFrames, it's convenient to be able to save/load them from disk.  For this we use the .tables() command.  If table is already present on disk, then the set results in the merge of the two DataFrames.

Let's consider that we have a table called 'prices' and we want to store it on disk.

```
?> tbs::.tables("/tmp/tables")
/tmp/tables:tables
?> tbs,"prices",prices
/tmp/tables:tables
```

Similarly, reading values is the same as getting a value from a dict:

```
?> prices::tbs?"prices"
```

### Generic key-value store

A simple key-value store backed by disk is available via the .kvs() command.

```
?> kvs::.kvs("/tmp/kvs")
/tmp/kvs:kvs
?> kvs,"hello",,"world"
/tmp/kvs:kvs
```

Now a file /tmp/kvs/hello exists with a pickled instance of "hello".

Retrieving a value is the same as reading from a dictionary:

```
?> kvs?"hello"
world
```

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

# Web server

KlongPy includes a simple web server module.  It's optional so you need to install the dependencies:

```bash
$ pip3 install klongpy[web]
```

The web server allows you to implement KlongPy functions as GET/POST handlers for registered routes.


Here's a simple example that lets you fetch and update a data array:

```
:" Import the Klongpy web module.  Requires pip3 install klongpy[web] first"
.py("klongpy.web")

:" Array of data to display"
data::[]

:" Return the data for a GET method at /"
index::{x;data}

:" Create the GET route handlers"
get:::{}
get,"/",index

:" Append the query param q value to data"
update::{[p];p::x?"p";.p(p);data::data,p}

:" Create the POST route handlers"
post:::{}
post,"/p",update

:" Start the web server with the GET and POST handlers"
.web(8888;get;post)

.p("curl -X POST -d""p=100"" ""http://localhost:8888/p""")
.p("curl ""http://localhost:8888""")

data
```

Test it out:

```bash
$ curl "http://localhost:8888"
[]
$ curl -X POST -d"p=100" "http://localhost:8888/p"
[100]
$ curl "http://localhost:8888"
[100]
```

# Timer

KlongPy includes periodic timer capabilities:

```
cb::{.p("hello")}
th::.timer("greeting";1;cb)
```

To stop the timer, it can be closed via:

```
.timerc(th)
```

The following example will create a timer which counts to 5 and then 
terminates the timer by return 0 from the callback.

```
counter::0
u::{counter::counter+1;.p(counter);1}
c::{.p("stopping timer");0}
cb::{:[counter<5;u();c()]}
th::.timer("count";1;cb)
```

which displays:

```
1
2
3
4
5
stopping timer
```


# Performance

The Klong language is simple, so the overhead is low.  The bulk of the compute time will likely be spent in NumPy doing actual work.

Here's a contrived rough benchmark to show the magnitude differences between Python, KlongPy (CPU + GPU) and Numpy (CPU).

**Spoiler**: GPU-backed KlongPy is about 790x faster than naive Python and 36x faster than NumPy-backed KlongPy.

### Python

```python
def python_vec(number=100):
    r = timeit.timeit(lambda: [2 * (1 + x) for x in range(10000000)], number=number)
    return r/number
```

### KlongPy

```python
# NumPy and CuPy (CuPy is enabled via USE_GPU=1 environment variable
def klong_vec(number=100):
    klong = KlongInterpreter()
    r = timeit.timeit(lambda: klong("2*1+!10000000"), number=number)
    return r/number
```

### NumPy (explicit usage)

```python
def NumPy_vec(number=100):
    r = timeit.timeit(lambda: np.multiply(np.add(np.arange(10000000), 1), 2), number=number)
    return r/number
```

## Results

### CPU (AMD Ryzen 9 7950x)

    $ python3 tests/perf_vector.py
    Python: 0.369111s
    KlongPy USE_GPU=None: 0.017946s
    Numpy: 0.017896s
    Python / KlongPy => 20.568334
    Numpy / KlongPy => 0.997245

### GPU (Same CPU with NVIDIA GeForce RTX 3090)

    $ USE_GPU=1 python3 tests/perf_vector.py
    Python: 0.364893s
    KlongPy USE_GPU=1: 0.000461s
    NumPy: 0.017053s
    Python / KlongPy => 790.678069
    Numpy / KlongPy => 36.951443


# Installation

### CPU

    $ pip3 install klongpy

### GPU support

    Choose your CuPy prebuilt binary or from source.  Note, the ROCM support for CuPy is experimental and likely will have issues.

    'cupy' => build from source
    'cuda12x' => "cupy-cuda12x"
    'cuda11x' => "cupy-cuda11x"
    'cuda111' => "cupy-cuda111"
    'cuda110' => "cupy-cuda110"
    'cuda102' => "cupy-cuda102"
    'rocm-5-0' => "cupy-rocm-5-0"
    'rocm-4-3' => "cupy-rocm-4-3"

    $ pip3 install klongpy[cupy]

### Everything

    $ pip3 install klongpy[full]

### Develop

    $ git clone https://github.com/briangu/klongpy.git
    $ cd klongpy
    $ python3 setup.py develop


# REPL

KlongPy has a REPL similar to Klong's REPL.

```bash
$ pip3 install klongpy[repl]
$ rlwrap kgpy

Welcome to KlongPy REPL
author: Brian Guarraci
repo  : https://github.com/briangu/klongpy
crtl-c to quit

?> 1+1
2
>? "hello, world!"
hello, world!
?> prime::{&/x!:\2+!_x^1%2}
:monad
?> prime(4)
0
?> prime(251)
1
?> ]T prime(251)
total: 0.0004914579913020134 per: 0.0004914579913020134
```

Read about the [prime example here](https://t3x.org/klong/prime.html).


# Status

KlongPy aims to be a complete implementation of klong.  It currently passes all of the integration tests provided by klong as well as additional suites.

Since CuPy is [not 100% compatible with NumPy](https://docs.cupy.dev/en/stable/user_guide/difference.html), there are currently some gaps in KlongPy between the two backends.  Notably, strings are supported in CuPy arrays so KlongPy GPU support currently is limited to math.

Primary ongoing work includes:

* Additional tools to make KlongPy applications more capable.
* Additional syntax error help
* Actively switch between CuPy and NumPy when incompatibilities are present
    * Work on CuPy kernels is in this branch: _cupy_reduce_kernels

# Differences from Klong

KlongPy is effectively a superset of the Klong language, but has some key differences:

    * Infinite precision: The main difference in this implementation of Klong is the lack of infinite precision.  By using NumPy we are restricted to doubles.
    * Python integration: Most notably, the ".py" command allows direct import of Python modules into the current Klong context.
    * IPC - KlongPy supports IPC between KlongPy processes.

# Related

 * [Klupyter - KlongPy in Jupyter Notebooks](https://github.com/briangu/klupyter)
 * [Advent Of Code '22](https://github.com/briangu/aoc/tree/main/22)


# Running tests

```bash
python3 -m unittest
```

# Unused operators

The following operators are yet to be used:

```
:! :& :, :< :> :?
```

# Acknowledgement

HUGE thanks to Nils M Holm for his work on Klong and providing the foundations for this interesting project.

