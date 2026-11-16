# WebSockets

KlongPy includes an optional WebSocket client module that lets you connect to WebSocket servers and handle messages from KlongPy code.

Install the WebSocket extras first:

```bash
pip install "klongpy[ws]"
```

## Client Basics

Load the module and define a message handler before connecting:

```klong
.py("klongpy.ws")

:" Message handler (dyad: x=client, y=message)"
.ws.m::{.d("recv: "); .p(y)}

:" Connect to a WebSocket URI"
ws::.ws("ws://localhost:8765")

:" Send a message (encoded as JSON)"
ws(["ping" 1])

:" Close the connection"
.wsc(ws)
```

### Message Encoding

Messages are encoded/decoded as JSON. JSON arrays are decoded back into Klong lists.

### Required Handler

Incoming messages always invoke `.ws.m`. If you don't define it, each message is dropped and a warning `symbol not found: :.ws.m` is logged. The handler must be a dyadic function: `x` is the client and `y` is the decoded message.

```klong
.ws.m::{.p("message: "); .p(y)}
```

Do not write the parameters as `{[c msg]; ...}`. A leading `[c msg]` declares local variables, not parameters, so that function takes no arguments and the call fails with `Klong function called with 2 but expected 0`.

### Tested behaviour

Against a small Python `websockets` server on localhost (run on 2026-11-16): sending `["ping" 1]` and `1+!3` reached the server as the JSON `["ping", 1]` and `[1, 2, 3]`. Replies of `[1,2,3]`, `"pong"` and `{"a":1}` reached the handler as `[1 2 3]`, `pong` and `:{[a 1]}`. `.ws` returned the connection function, and `.wsc(ws)` returned 1. A reply that is a list of mixed or ragged elements, such as `["echo",["ping",1]]`, was not delivered to the handler in that test. Messages are not covered by the checked examples of this page, because the checker does not start a WebSocket server.

### Notes

- The current implementation provides a **client** only (server support is not yet exposed).
- `.ws` expects a full WebSocket URI (for example, `ws://localhost:8765`).

## See also

- [Web server](web_server.md) and [IPC capabilities](ipc_capabilities.md).
- [Roadmap](roadmap.md) for known limits.
