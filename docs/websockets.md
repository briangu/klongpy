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
.ws.m::{[c msg]; .d("recv: "); .p(msg)}

:" Connect to a WebSocket URI"
ws::.ws("ws://localhost:8765")

:" Send a message (encoded as JSON)"
ws(["ping" 1])

:" Close the connection"
.wsc(ws)
```

### Message Encoding

Messages are encoded/decoded as JSON. NumPy arrays are converted to lists on send, and JSON arrays are decoded back into Klong lists.

### Required Handler

Incoming messages always invoke `.ws.m`. If you don’t define it, the client will raise an error on receipt. The handler should be a dyadic function:

```klong
.ws.m::{[c msg]; .p("message: "); .p(msg)}
```

### Notes

- The current implementation provides a **client** only (server support is not yet exposed).
- `.ws` expects a full WebSocket URI (for example, `ws://localhost:8765`).
