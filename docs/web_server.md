# Web Server

> **Security:** `.web` listens on all interfaces by default and has no authentication or encryption. Bind to localhost (`.web("127.0.0.1:8080";get;post)`) and use it on trusted networks only. See the [security model](https://github.com/briangu/klongpy#security-model-trusted-networks).

`.web` serves Klong functions as HTTP GET and POST routes, for a small API or a status page. The module is optional; install its dependency with `pip3 install "klongpy[web]"` and load it with `.py("klongpy.web")`.

## What a handler receives and returns

`.web(address;get;post)` takes two dictionaries from route path to handler: `get` for GET routes and `post` for POST routes. It returns the server handle.

- A handler is a function of one argument. A function of any other arity is skipped, and its route answers 404.
- A GET handler receives a dictionary of the query parameters. A POST handler receives a dictionary of the form fields of the body. Keys and values are strings, so the value of `?p=100` is `"100"`. A missing key reads as `:undefined`.
- The handler's value becomes the response body, as text, with status 200. A string is sent as it is. Any other value is converted with Python's `str`, so a list arrives in Python's notation (`['100']`).
- A handler that raises an error answers 400 with the body `Invalid request`. A path with no route answers 404.

Known limit: the response is always `text/plain`. To send JSON or another format, build the string in the handler.

## Example

A server that stores values posted to `/p` and returns them from `/`:

```
.py("klongpy.web")

:" The data, returned by a GET of /"
data::[]
index::{x;data}
get:::{}
get,"/",index

:" Append the form field p of a POST to /p, and return the new length"
update::{[p];p::x?"p";data::data,p;#data}
post:::{}
post,"/p",update

.web("127.0.0.1:8888";get;post)
```

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
?> index::{x; "Hello, Klong World! ", data}
:monad
?> get:::{}; get,"/",index
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

## See also

- [WebSockets](websockets.md) and [IPC capabilities](ipc_capabilities.md).
- [System functions](system-functions.md): `.web`, `.webc`.
