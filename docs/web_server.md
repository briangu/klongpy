# Web Server

> **Security:** `.web` listens on all interfaces by default and has no authentication or encryption. Bind to localhost (`.web("127.0.0.1:8080";get;post)`) and use it on trusted networks only. See the [security model](https://github.com/briangu/klongpy#security-model-trusted-networks).

`.web` serves Klong functions as HTTP GET and POST routes, for a small API or a status page. The module is optional; install its dependency with `pip install "klongpy[web]"` and load it with `.py("klongpy.web")`.

## What a handler receives and returns

`.web(address;get;post)` takes two dictionaries from route path to handler: `get` for GET routes and `post` for POST routes. It returns the server handle.

- A handler is a function of one argument. A function of any other arity is skipped, and its route answers 404.
- A GET handler receives a dictionary of the query parameters. A POST handler receives a dictionary of the form fields of the body (`application/x-www-form-urlencoded`). Keys and values are strings, so the value of `?p=100` is `"100"`. A missing key reads as `:undefined`.
- A `HEAD` request to a GET route is answered as the GET, without the body.
- A request to a path that has a route, with a method that route does not have, answers 405 (`Allow` names the methods that are served). A path with no route answers 404.

### The response

A handler returns text, or a dictionary that chooses the status, the content type and the body.

- Text is the body, with status 200 and the type `text/plain; charset=utf-8`. Any other value is converted with Python's `str`, so a list arrives in Python's notation (`['100']`).
- A dictionary may have the keys `"status"` (an integer from 100 to 599, default 200), `"type"` (a content type, default `text/plain`) and `"body"` (text, default empty). Each key is optional. A dictionary with any other key, or a status or type of the wrong kind, is a type error: the request answers 500 and the error is logged.

<!-- check-docs: requires aiohttp -->
```kgpy
?> .py("klongpy.web")
1
?> missing(q)::{:{;"status";404;"type";"application/json";"body";.json(:{;"error";"no such item"})}}
:monad
?> missing(:{})
:{["status" 404] ["type" "application/json"] ["body" "{""error"":""no such item""}"]}
```

A client sees:

```text
$ curl -i http://127.0.0.1:8888/missing
HTTP/1.1 404 Not Found
Content-Type: application/json; charset=utf-8

{"error":"no such item"}
```

### Errors

A handler that fails (a Klong error, or an unusable response) answers 500 with the body `handler failed`. One line goes to stderr with the route and the error, and the server goes on:

```text
web POST /row: handler failed: type error: cannot apply + (Plus) to integer and string
```

The Python traceback follows it under `-v` or `--traceback`. A body that cannot be read answers 400 with `Invalid request`.

### Where a handler runs

KlongPy has two event loops, each on its own thread: the I/O loop, which serves the sockets, and the Klong loop, which runs Klong code. Handlers run on the Klong loop, one at a time, like timers, `.async` callbacks and IPC handlers. A handler can therefore start a timer, make `.cli` and `.async` calls, and stop the server. `.webc(h)` in a handler completes the reply being served and then stops the server; a file run with nothing else alive then ends with status 0.

Known limit: the Klong loop runs one thing at a time, so a handler that takes long delays the other requests. A handler that requests its own server over HTTP waits for itself, so request another process's server instead.

## Example

A server that stores values posted to `/p` and returns them from `/`:

```
.py("klongpy.web")

:" The data, returned by a GET of /"
data::[]
index(query)::{data}
get:::{;"/";index}

:" Append the form field p of a POST to /p, and return the new length"
update(form)::{[p];p::form@"p";data::data,p;#data}
post:::{;"/p";update}

.web("127.0.0.1:8888";get;post)
```

Save it as `web.kg` and run `kgpy web.kg`. The run stays up while the server listens, until Ctrl-C (or SIGTERM) or `.webc` on the handle.

The requests and responses of a run (the headers other than `Content-Type` are left out):

```text
$ curl -i "http://127.0.0.1:8888/"
HTTP/1.1 200 OK
Content-Type: text/plain; charset=utf-8

[]
$ curl -X POST -d "p=100" "http://127.0.0.1:8888/p"
1
$ curl "http://127.0.0.1:8888/"
['100']
$ curl -i "http://127.0.0.1:8888/nope"
HTTP/1.1 404 Not Found
```

The posted value is a string, so `data` holds `'100'`, not the number 100. `.web(8888;get;post)` with a bare port listens on all interfaces; the address string above listens on localhost only.

The same server from the REPL, with a handler that returns a string and a list:

<!-- check-docs: requires aiohttp -->
```kgpy
?> .py("klongpy.web")
1
?> data::!10
[0 1 2 3 4 5 6 7 8 9]
?> index(query)::{"Hello, Klong World! ", data}
:monad
?> get:::{;"/";index}
:{["/" :monad]}
?> post:::{}
:{}
?> h::.web(8888;get;post)
web[0.0.0.0:8888]
```

From another terminal:

```bash
$ curl http://localhost:8888
['Hello, Klong World! ' 0 1 2 3 4 5 6 7 8 9]
```

Stop the server with:

<!-- check-docs: continue -->
```kgpy
?> .webc(h)
1
```

## JSON

`.json` writes a Klong value as JSON text and `.unjson` reads it back (see [JSON](language-additions.md#json)). A handler that answers JSON sets the type:

```kgpy
?> stats(q)::{:{;"type";"application/json";"body";.json(:{;"count";3;"values";[1 2 3]})}}
:monad
?> stats(:{})@"body"
"{""count"":3,""values"":[1,2,3]}"
```

## Requesting a server from Klong

There is no HTTP client function. Use Python's standard library through `.pyf`: `urlopen` returns the response, `.pyc` calls its `read` method, and the bytes are decoded. This reads the JSON answer of another process's server:

```
.pyf("urllib.request";"urlopen")
fetch(url)::{[r b];r::urlopen(url);b::.pyc(r,"read";[];:{});.pyc(b,"decode";[];:{})}
.unjson(fetch("http://127.0.0.1:8888/stats"))
```

`.unjson` returns a dictionary with `count` 3 and `values` `[1 2 3]`. The call waits for the answer and fails with a Python error if the server cannot be reached.

## See also

- [WebSockets](websockets.md) and [IPC capabilities](ipc_capabilities.md).
- [System functions](system-functions.md): `.web`, `.webc`.
