"""kdb+ IPC session: a state machine with no I/O (design sections 2, 5, 8, 9).

A :class:`Session` is one end of one connection.  The endpoint layer owns the
socket and the interpreter; it

* feeds received bytes in with :meth:`Session.feed` and gets events back,
* asks the session to frame what it wants to send and writes the returned bytes,
* tells the session when it starts and ends serving an incoming request, so
  that responses can be matched without message ids (design section 5).

Nothing here touches a socket, a clock or an event loop; time is passed in
(``deadline`` / ``check_deadlines(now)``).

Handshake (design section 8, reference section 1)
    client:  ``hello()`` returns ``user:password`` + capability byte + NUL;
             the first byte received is the server's reply (``HandshakeAccepted``),
             EOF before it means refusal (``HandshakeRejected``).
    server:  the first NUL-terminated string (at most 256 bytes) yields a
             ``Handshake`` event; the endpoint decides (``.srv.auth``) and calls
             ``accept()`` (returns the one reply byte) or ``reject()`` (close).
    Capability: we offer 3 and answer ``min(peer, 3)`` to peers offering 1 to 6;
    a peer offering 0 (or anything else) is refused, which avoids the
    unconfirmed capability-0 framing (reference 1.1, 10.1).

Framing: 8-byte header (byte order, message type 0/1/2, compression flag,
reserved byte, int32 total length including the header).  The length is
checked against ``max_message`` before anything is buffered or allocated, and
so is the declared uncompressed size of a compressed message.  Header byte 3
must be zero: a non-zero value would be the large-message extension of
capability 5/6, which is UNCONFIRMED in the reference and never negotiated
(safe default: protocol error).

Response matching without ids (design section 5):

* outstanding sync requests live in *levels*; level 0 holds those issued by
  ordinary code;
* ``begin_service`` (the endpoint starts serving an incoming request) opens a
  deeper level when a blocking call is outstanding; requests issued while
  serving go into the deepest level;
* an incoming response completes the oldest request of the deepest level that
  has any;
* ``end_service`` closes the level; one that still has requests is a protocol
  error and closes the connection;
* the number of nested levels is limited (``max_nesting``, default 8); beyond
  it ``begin_service`` raises :class:`NestingLimit` and the endpoint answers
  with :meth:`refuse` (a range error, wire name ``index``).
"""
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .codec import encode, encode_call, decode, KdbError
from .compress import compress, decompress, MAX_MESSAGE
from .errors import IPCError, ProtocolError, IPCRangeError, IPCStateError

__all__ = [
    "Session", "Pending", "Service", "Handshake", "HandshakeAccepted", "HandshakeRejected",
    "Message", "Failure", "Closed", "NestingLimit",
    "ASYNC", "SYNC", "RESPONSE", "KIND_NAMES",
    "DEFAULT_CAPABILITY", "MAX_NESTING", "HANDSHAKE_LIMIT",
]

ASYNC, SYNC, RESPONSE = 0, 1, 2
KIND_NAMES = {ASYNC: "async", SYNC: "sync", RESPONSE: "response"}

#: capability byte we send and the highest we negotiate (design section 1)
DEFAULT_CAPABILITY = 3
#: a server reads at most this many bytes looking for the handshake's NUL
HANDSHAKE_LIMIT = 256
#: default depth of nested service levels
MAX_NESTING = 8

_INT32_MAX = 0x7FFFFFFF


class NestingLimit(IPCError):
    """An incoming request would nest deeper than ``max_nesting`` (answer with ``refuse``)."""

    kind = "range"


# ----------------------------------------------------------------------------
# events and bookkeeping objects
# ----------------------------------------------------------------------------

@dataclass
class Handshake:
    """Server side: a client's greeting parsed.  Call ``accept()`` or ``reject()``."""
    user: str
    password: str
    capability: int


@dataclass
class HandshakeAccepted:
    """Client side: the server's one-byte reply; ``capability`` is the negotiated value."""
    capability: int


@dataclass
class HandshakeRejected:
    """Client side: the connection ended before the server's reply byte."""


@dataclass(eq=False)
class Pending:
    """One outstanding sync request."""
    tag: Any = None
    blocking: bool = True
    deadline: Optional[float] = None
    level: int = 0
    done: bool = False
    failed: bool = False


@dataclass
class Message:
    """One complete message.

    ``kind`` is ``"async"``, ``"sync"`` or ``"response"``.  For a response,
    ``pending`` is the request it answers.  ``value`` may be a ``KdbError`` (the
    peer's error object).  ``error`` is set, and ``value`` is None, when the
    framing was fine but the value did not fit this build (``IPCRangeError`` on
    a 32-bit build): a problem of this message only, not of the connection.
    """
    kind: str
    value: Any
    pending: Optional[Pending] = None
    error: Optional[IPCError] = None
    size: int = 0


@dataclass
class Failure:
    """Fatal protocol failure.  The session is closed; close the transport and
    fail ``pending`` with a ``system`` error."""
    error: ProtocolError
    pending: List[Pending] = field(default_factory=list)


@dataclass
class Closed:
    """The connection ended (peer EOF, deadline or ``check_deadlines``).
    ``pending`` are the requests that will never be answered."""
    reason: str
    pending: List[Pending] = field(default_factory=list)


@dataclass(eq=False)
class Service:
    """An incoming request being served (see ``Session.begin_service``)."""
    kind: str
    opened_level: bool
    answered: bool = False


_STATE_HANDSHAKE = "handshake"      # waiting for the peer's greeting / reply
_STATE_AUTH = "auth"                # server: greeting parsed, waiting for accept()/reject()
_STATE_READY = "ready"
_STATE_CLOSED = "closed"


def _ascii(b, what):
    try:
        return b.decode("ascii")
    except UnicodeDecodeError:
        raise ProtocolError("%s is not ASCII" % what) from None


class Session:
    """One end of one kdb+ IPC connection."""

    def __init__(self, role, *, user="", password="", capability=DEFAULT_CAPABILITY,
                 max_message=MAX_MESSAGE, widths=64, max_nesting=MAX_NESTING):
        if role not in ("client", "server"):
            raise ValueError("role must be 'client' or 'server'")
        if not 1 <= capability <= 6:
            raise ValueError("capability must be 1..6")
        self.role = role
        self.user = user
        self.password = password
        self.offered = capability
        self.capability = None          # negotiated, known once the handshake is done
        self.max_message = max_message
        self.widths = widths
        self.max_nesting = max_nesting
        self.state = _STATE_HANDSHAKE
        self._buf = bytearray()
        self._off = 0
        self._levels = [deque()]
        self._services: List[Service] = []
        self._hello_sent = False
        self._peer_capability = None

    @classmethod
    def client(cls, user="", password="", **kw):
        return cls("client", user=user, password=password, **kw)

    @classmethod
    def server(cls, **kw):
        return cls("server", **kw)

    # -- introspection -------------------------------------------------------

    @property
    def closed(self):
        return self.state == _STATE_CLOSED

    @property
    def ready(self):
        return self.state == _STATE_READY

    @property
    def nesting(self):
        """Number of open service levels above level 0."""
        return len(self._levels) - 1

    def pending(self):
        """All outstanding requests, deepest level first, oldest first within a level."""
        return [p for lvl in reversed(self._levels) for p in lvl]

    # -- handshake -----------------------------------------------------------

    def hello(self):
        """Client: the bytes to send first, ``user:password`` + capability + NUL."""
        if self.role != "client" or self._hello_sent or self.state != _STATE_HANDSHAKE:
            raise IPCStateError("hello() is the client's first and only send")
        for part in (self.user, self.password):
            if not part.isascii() or "\x00" in part:
                raise ValueError("credentials must be ASCII without NUL")
        self._hello_sent = True
        return ("%s:%s" % (self.user, self.password)).encode("ascii") + bytes((self.offered, 0))

    def accept(self):
        """Server: accept the greeting; returns the single reply byte to send."""
        if self.state != _STATE_AUTH:
            raise IPCStateError("accept() answers a Handshake event")
        self.capability = min(self._peer_capability, self.offered)
        self.state = _STATE_READY
        return bytes((self.capability,))

    def reject(self):
        """Server: refuse the greeting.  kdb+ just closes the connection."""
        if self.state != _STATE_AUTH:
            raise IPCStateError("reject() answers a Handshake event")
        self.close()

    # -- receiving -----------------------------------------------------------

    def feed(self, data=b""):
        """Give the session received bytes; returns the events they complete.

        Call ``feed(b"")`` after ``accept()`` to drain bytes that arrived
        behind the greeting.  After a ``Failure`` the session is closed and
        further input is ignored.
        """
        if self.state == _STATE_CLOSED:
            return []
        if data:
            self._buf += data
        events = []
        while True:
            try:
                ev = self._step()
            except ProtocolError as exc:
                events.append(self._fail(exc))
                break
            if ev is None:
                break
            events.append(ev)
        self._compact()
        return events

    def feed_eof(self):
        """The transport saw end-of-stream."""
        if self.state == _STATE_CLOSED:
            return []
        in_handshake = self.state == _STATE_HANDSHAKE
        partial = len(self._buf) > self._off
        pending = self._drain_pending()
        self.state = _STATE_CLOSED
        if in_handshake and self.role == "client":
            return [HandshakeRejected()]
        return [Closed("truncated" if partial else "eof", pending)]

    def _compact(self):
        if self._off:
            del self._buf[:self._off]
            self._off = 0

    def _fail(self, exc):
        pending = self._drain_pending()
        self.state = _STATE_CLOSED
        self._buf.clear()
        self._off = 0
        return Failure(exc, pending)

    def _step(self):
        if self.state == _STATE_HANDSHAKE:
            return self._step_handshake()
        if self.state == _STATE_READY:
            return self._step_frame()
        return None                     # auth: wait for accept()/reject()

    def _step_handshake(self):
        avail = len(self._buf) - self._off
        if self.role == "client":
            if not self._hello_sent:
                raise ProtocolError("data received before hello was sent")
            if avail < 1:
                return None
            reply = self._buf[self._off]
            self._off += 1
            if reply > 6:
                raise ProtocolError("bad capability byte %d in the server's reply" % reply)
            self.capability = min(reply, self.offered)
            self.state = _STATE_READY
            return HandshakeAccepted(self.capability)
        # server: the greeting ends at the first NUL, within HANDSHAKE_LIMIT bytes
        end = self._buf.find(b"\x00", self._off, self._off + HANDSHAKE_LIMIT)
        if end < 0:
            if avail >= HANDSHAKE_LIMIT:
                raise ProtocolError("no end of greeting within %d bytes" % HANDSHAKE_LIMIT)
            return None
        text = bytes(self._buf[self._off:end])
        self._off = end + 1
        if not text:
            raise ProtocolError("empty greeting")
        cap = text[-1]
        if not 1 <= cap <= 6:
            # includes capability 0: "user:pass" + single NUL parses with a
            # letter in this position, "user:pass" + NUL NUL ends at the first NUL
            raise ProtocolError("unsupported capability byte %d" % cap)
        creds = _ascii(text[:-1], "credentials")
        user, _, password = creds.partition(":")
        self._peer_capability = cap
        self.state = _STATE_AUTH
        return Handshake(user, password, cap)

    def _step_frame(self):
        avail = len(self._buf) - self._off
        if avail < 8:
            return None
        o = self._off
        hdr = self._buf[o:o + 8]
        if hdr[0] > 1:
            raise ProtocolError("bad byte-order flag %d" % hdr[0])
        if hdr[1] > 2:
            raise ProtocolError("bad message type %d" % hdr[1])
        if hdr[2] > 1:
            raise ProtocolError("bad compression flag %d" % hdr[2])
        if hdr[3] != 0:
            raise ProtocolError("reserved header byte set (large messages are not negotiated)")
        little = hdr[0] == 1
        (length,) = struct.unpack_from("<i" if little else ">i", hdr, 4)
        if length < 8 or (hdr[2] == 1 and length < 12):
            raise ProtocolError("message length %d shorter than its header" % length)
        if length > self.max_message:
            raise ProtocolError("message length %d exceeds the limit of %d" % (length, self.max_message))
        if avail < length:
            return None
        frame = bytes(self._buf[o:o + length])
        self._off = o + length
        if hdr[2] == 1:
            frame = decompress(frame, self.max_message)
        kind = frame[1]
        try:
            value = decode(frame, little, self.widths, offset=8)
            err = None
        except IPCRangeError as exc:
            value, err = None, exc
        if kind == RESPONSE:
            pend = self._match_response()
            return Message("response", value, pend, err, length)
        return Message(KIND_NAMES[kind], value, None, err, length)

    def _match_response(self):
        for lvl in reversed(self._levels):
            if lvl:
                p = lvl.popleft()
                p.done = True
                return p
        raise ProtocolError("response without an outstanding request")

    # -- sending -------------------------------------------------------------

    def _require_ready(self):
        if self.state != _STATE_READY:
            raise IPCStateError("session is %s" % self.state)

    def frame(self, kind, body, *, compress_body=False):
        """Wrap an encoded object in a message header (always little-endian)."""
        self._require_ready()
        total = 8 + len(body)
        if total > _INT32_MAX:
            raise IPCRangeError("message of %d bytes exceeds 2 GB" % total)
        msg = b"\x01" + bytes((kind, 0, 0)) + struct.pack("<I", total) + bytes(body)
        if compress_body and self.capability >= 1:
            msg = compress(msg)
        return msg

    def _body(self, value, call):
        return encode_call(value, self.widths) if call else encode(value, self.widths)

    def send_async(self, value, *, call=False, compress_body=False):
        """Frame an async (type 0) message; nothing comes back."""
        return self.frame(ASYNC, self._body(value, call), compress_body=compress_body)

    def send_sync(self, value, *, call=False, blocking=True, tag=None, deadline=None,
                  compress_body=False):
        """Frame a sync (type 1) request and register it as outstanding.

        ``call=True`` encodes ``value`` (a sequence) as a call envelope: always
        a general list.  Returns ``(bytes, Pending)``.
        """
        data = self.frame(SYNC, self._body(value, call), compress_body=compress_body)
        p = Pending(tag, blocking, deadline, level=len(self._levels) - 1)
        self._levels[-1].append(p)
        return data, p

    # -- serving -------------------------------------------------------------

    def begin_service(self, message):
        """The endpoint starts serving ``message`` (a sync or async request).

        Opens a deeper level if a blocking call is outstanding.  Raises
        :class:`NestingLimit` when that would exceed ``max_nesting``; answer
        the request with :meth:`refuse` instead and do not serve it.
        """
        self._require_ready()
        if message.kind not in ("sync", "async"):
            raise IPCStateError("only requests are served")
        blocking = any(p.blocking for lvl in self._levels for p in lvl)
        if blocking and len(self._levels) - 1 >= self.max_nesting:
            raise NestingLimit("nested calls deeper than %d" % self.max_nesting)
        if blocking:
            self._levels.append(deque())
        svc = Service(message.kind, blocking)
        self._services.append(svc)
        return svc

    def end_service(self, service):
        """The endpoint has finished serving ``service``.

        Raises :class:`ProtocolError` (and closes the session) when the level
        being closed still has unanswered requests.
        """
        if not self._services or self._services[-1] is not service:
            raise IPCStateError("services must end in the reverse order they began")
        if service.kind == "sync" and not service.answered:
            raise IPCStateError("sync request ended without a response")
        self._services.pop()
        if service.opened_level:
            lvl = self._levels.pop()
            if lvl:
                exc = ProtocolError("%d request(s) unanswered when their level closed" % len(lvl))
                self._levels.append(lvl)         # so the failure reports them
                ev = self._fail(exc)
                exc.pending = ev.pending
                raise exc

    def _service_check(self, service):
        if self._services[-1:] != [service] or service.kind != "sync" or service.answered:
            raise IPCStateError("not the innermost unanswered sync request")

    def send_response(self, service, value, *, compress_body=False):
        """Frame the response (type 2) to the sync request ``service``."""
        self._require_ready()
        self._service_check(service)
        data = self.frame(RESPONSE, encode(value, self.widths), compress_body=compress_body)
        service.answered = True
        return data

    def send_error(self, service, text):
        """Frame an error response (``80``, text, ``00``) to ``service``."""
        return self.send_response(service, KdbError(text))

    def refuse(self, message):
        """Answer a request that was not served because of the nesting limit.

        Returns the bytes to send: an error response (wire name ``index``, a
        range error) for a sync request, ``b""`` for an async one.
        """
        self._require_ready()
        if message.kind != "sync":
            return b""
        return self.frame(RESPONSE, encode(KdbError(
            "index: nested calls deeper than %d" % self.max_nesting), self.widths))

    # -- ending --------------------------------------------------------------

    def _drain_pending(self):
        out = self.pending()
        for p in out:
            p.failed = True
        self._levels = [deque()]
        self._services = []
        return out

    def close(self):
        """Close the session; returns the pending requests, which now fail."""
        if self.state == _STATE_CLOSED:
            return []
        out = self._drain_pending()
        self.state = _STATE_CLOSED
        self._buf.clear()
        self._off = 0
        return out

    def next_deadline(self):
        """Earliest deadline among outstanding requests, or None."""
        ds = [p.deadline for p in self.pending() if p.deadline is not None]
        return min(ds) if ds else None

    def check_deadlines(self, now):
        """A request without ids cannot be skipped: when any deadline has
        passed the connection is closed and every pending request fails.
        Returns a ``Closed("timeout")`` event, or None."""
        if self.state == _STATE_CLOSED:
            return None
        d = self.next_deadline()
        if d is None or now < d:
            return None
        return Closed("timeout", self.close())
