# WebSockets

`.ws` connects a Klong program to a WebSocket server, sends it messages and runs a handler for each message that arrives. Use it to read a stream, such as a market data feed, from a program written in Klong. The module is optional: install it with `pip install "klongpy[ws]"` and load it with `.py("klongpy.ws")`.

## Client basics

Define the message handler before connecting:

```klong
.py("klongpy.ws")

:" Message handler: x is the client, y the decoded message"
.ws.m::{.d("recv: "); .p(y)}

:" Connect to a WebSocket URI"
ws::.ws("ws://localhost:8765")

:" Send a message, encoded as JSON"
ws(["ping" 1])

:" Close the connection"
.wsc(ws)
```

`.ws` takes a full WebSocket URI such as `ws://localhost:8765` and returns a function that sends its argument as a message. `.wsc(ws)` closes the connection and returns 1.

## The message handler

`.ws.m` is a dyadic function that every incoming message calls: `x` is the client and `y` is the decoded message. Its return value is ignored; nothing is sent back. If `.ws.m` is not defined, each message is dropped and the warning `symbol not found: :.ws.m` is logged.

Do not write the parameters as `{[c msg]; ...}`. A leading `[c msg]` declares local variables, not parameters, so the function takes no arguments and the call fails with `Klong function called with 2 but expected 0`.

## Messages

Messages are JSON. Klong values are encoded when sent and JSON is decoded when received:

| Sent | The server receives |
|---|---|
| `["ping" 1]` | `["ping", 1]` |
| `1+!3` | `[1, 2, 3]` |

| The server sends | The handler receives |
|---|---|
| `[1,2,3]` | `[1 2 3]` |
| `"pong"` | `pong` (a string; `.p` shows it without quotes) |
| `{"a":1}` | `:{[a 1]}` |
| `["echo",["ping",1]]` | `[echo [ping 1]]` |

Conditions: these pairs were observed against a Python `websockets` server on localhost.

Known limit: the module provides a client only. A WebSocket server is not exposed.

## See also

- [Web server](web_server.md) and [IPC capabilities](ipc_capabilities.md).
- [Roadmap](roadmap.md) for known limits.
