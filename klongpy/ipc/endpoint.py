"""The IPC endpoint over the kdb+ protocol.

The codec turns values into bytes and the session frames them and matches
responses; this module is the glue that owns a connection's behaviour:

* mapping Klong calls to request messages (``to_wire``) and request messages
  back to evaluation (``evaluate_request``);
* error text on the wire (``wire_error_text``);
* functions on the wire and remote function proxies (``lambda_value``);
* ``KdbConnectionMixin``, the kdb+ half of ``NetworkClient``: handshake, the
  reader loop's event handling, blocking and non-blocking calls, serving.

Threads
    The ioloop thread does all socket I/O and feeds the session.  A request
    message is served on the klong loop thread, except while a thread is
    blocked in a call on that connection: then that thread serves it at once,
    inline, which is what makes a callback from the server during a client's
    call work.  Session state is touched under one
    lock; writes are queued to the ioloop in the order they were framed.
    The supported model is one logical thread of control per connection
    (several threads sharing one handle still work, but a nested request goes
    to the most recent blocked caller).
"""
import asyncio
import logging
import queue
import threading
import time
from collections import deque

import numpy as np

from ..native_ext import native
from ..core import (KGCall, KGFn, KGFnWrapper, KGLambda, KGSym,
                    KLONG_UNDEFINED, get_fn_arity_str, reserved_fn_args, reserved_fn_symbol_map)
from ..errors import (ERROR_KINDS, KlongError, KlongInternalError, KlongRemoteError,
                      KlongSystemError, KlongTypeError, classify)
from .codec import KdbError, KdbLambda, set_function_text_hook
from .errors import (IPCError, IPCStateError, KlongIPCConnectionClosedException,
                     KlongIPCConnectionFailureException, KlongIPCHandshakeException,
                     KlongIPCTimeoutException, ProtocolError)
from .session import (Closed, Failure, Handshake, HandshakeAccepted, HandshakeRejected,
                      Message, NestingLimit, Session)

DEFAULT_CALL_TIMEOUT = 30.0
CLOSE_TIMEOUT = 5.0
#: seconds a peer has to finish the handshake
HANDSHAKE_TIMEOUT = 5.0
READ_SIZE = 65536

_HANDLE_SYM = KGSym(".cli.h")
_SET_SYM = KGSym("set")


# ----------------------------------------------------------------------------
# errors on the wire
# ----------------------------------------------------------------------------

#: wire name -> the KlongError class it stands for (``docs/errors.md``)
WIRE_CLASSES = {c.wire: c for c in ERROR_KINDS.values() if c.wire}
_REMOTE_CLASSES = {}


def _remote_class(kcls):
    """The subclass of ``kcls`` that is also a ``KlongRemoteException`` (made once, importable from here)."""
    c = _REMOTE_CLASSES.get(kcls)
    if c is None:
        name = f"KlongRemote{kcls.__name__[len('Klong'):]}"
        c = type(name, (KlongRemoteException, kcls), {
            "__module__": __name__, "__doc__": f"A {kcls.__name__} reported by the peer.",
            "kind": kcls.kind, "wire": kcls.wire})
        _REMOTE_CLASSES[kcls] = c
        globals()[name] = c
    return c


class KlongRemoteException(KlongRemoteError):
    """The peer answered a request with an error object (kind ``remote``, or the kind it names).

    ``text`` is the whole text as received (``<wire name>: <message>`` from a
    KlongPy peer, anything from another one); ``wire_name`` is the part before
    ``: `` when it is one of ours, else ``None``.  Creating one from text that
    starts with a wire name gives an instance of that name's class too
    (``index`` is a ``KlongRangeError``), with ``kind`` of that class and the
    message without the prefix; any other text keeps ``kind == "remote"``.
    ``hint`` names the peer.
    """

    remote = True

    def __new__(cls, text="", peer=None):
        if cls is KlongRemoteException:
            kcls = WIRE_CLASSES.get(str(text).partition(": ")[0]) if ": " in str(text) else None
            if kcls is not None:
                cls = _remote_class(kcls)
        return super().__new__(cls)

    def __init__(self, text="", peer=None):
        text = str(text)
        head, sep, rest = text.partition(": ")
        wire_name = head if sep and head in WIRE_CLASSES else None
        super().__init__(rest if wire_name else text,
                         hint=None if peer is None else f"reported by {peer}")
        self.text = text
        self.peer = None if peer is None else str(peer)
        self.wire_name = wire_name


def wire_error_text(exc):
    """``<wire name>: <message>`` for an exception raised while serving a request.

    The name is the ``wire`` of the error's class (``kind``): ``index`` for a
    range error, ``length`` for a shape error.  An exception that is not a
    ``KlongError`` is classified as an operator's would be; a bare ``KlongError``
    is a ``type`` error.
    """
    msg = str(exc)
    if isinstance(exc, KlongRemoteException):
        return exc.text.replace("\x00", "")
    if isinstance(exc, IPCError):
        name = ERROR_KINDS[exc.kind].wire
    elif isinstance(exc, KlongError):
        name = exc.wire or "type"
    elif isinstance(exc, KeyError):
        name = "name"
        msg = f"symbol not found: {exc.args[0] if exc.args else ''}"
    elif isinstance(exc, OSError):
        name = "os"
    else:
        kcls = classify(exc)
        name = kcls.wire
        if kcls is KlongInternalError:
            msg = f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
    if not msg:
        msg = type(exc).__name__
    return f"{name}: {msg}".replace("\x00", "")


# ----------------------------------------------------------------------------
# waiting for a response
# ----------------------------------------------------------------------------

class Waiter:
    """A thread blocked on a call.  Quacks like a future for ``_cleanup_pending_responses``.

    The inbox carries ``("response", value)``, ``("failed", exception)`` and
    ``("request", message)``: a request that arrived while the thread was
    blocked and is to be served by it, inline.
    """

    def __init__(self):
        self.inbox = queue.SimpleQueue()
        self._done = False

    def done(self):
        return self._done

    def set_result(self, value):
        self._done = True
        self.inbox.put(("response", value))

    def set_exception(self, exc):
        self._done = True
        self.inbox.put(("failed", exc))


class CallbackWaiter:
    """A non-blocking request: ``finish(value, error)`` runs when the answer (or the failure) arrives."""

    def __init__(self, finish):
        self._finish = finish
        self._done = False

    def done(self):
        return self._done

    def set_result(self, value):
        self._done = True
        self._finish(value, None)

    def set_exception(self, exc):
        self._done = True
        self._finish(None, exc)


# ----------------------------------------------------------------------------
# request values
# ----------------------------------------------------------------------------

class KGRemoteFnRef:
    """A remote function that arrived as its display form: only its arity is known."""

    def __init__(self, arity):
        self.arity = arity

    def __str__(self):
        return get_fn_arity_str(self.arity)


class KGRemoteFnCall:
    """Call the remote function ``sym`` with ``params``."""

    def __init__(self, sym: KGSym, params):
        self.sym = sym
        self.params = params


class KGRemoteDictSetCall:
    """Set the remote global ``key`` to ``value``."""

    def __init__(self, key, value):
        self.key = key
        self.value = value


class KGRemoteDictGetCall:
    """Read the remote global ``key``."""

    def __init__(self, key):
        self.key = key


# ----------------------------------------------------------------------------
# the first bytes of a connection
# ----------------------------------------------------------------------------

def detect_handshake(buf):
    """Tell a kdb+ handshake from a frame of the removed 0.7 pickle protocol.

    A 0.7 client starts with a 16-byte binary message id (a version-4 UUID, whose
    byte 8 is always 0x80 or above) and a 4-byte length.  A kdb+ handshake is ASCII
    ``user:password``, a capability byte from 1 to 6 and a zero.  Scan forward: a
    byte of 0x80 or above before any zero is ``"pickle"``; a zero preceded by a byte
    from 1 to 6, with a ``:`` earlier, is ``"kdb"``; any other zero is ``"pickle"``.
    Returns ``"kdb"``, ``"pickle"``, ``None`` (need more bytes) or ``"bad"`` (256
    bytes with neither: not a handshake of either kind).  Nothing is unpickled.
    """
    for i, b in enumerate(buf):
        if b >= 0x80:
            return "pickle"
        if b == 0:
            if i > 0 and 1 <= buf[i - 1] <= 6 and b":" in bytes(buf[:i - 1]):
                return "kdb"
            return "pickle"
        if i >= 255:
            return "bad"
    return None


# ----------------------------------------------------------------------------
# remote function proxy
# ----------------------------------------------------------------------------

class KGRemoteFnProxy(KGLambda):
    """A function that lives on the other side of ``nc`` and is called by name."""

    def __init__(self, nc, sym: KGSym, arity):
        self.nc = nc
        self.sym = sym
        self.args = reserved_fn_args[:arity]

    def _params(self, ctx):
        return [ctx[reserved_fn_symbol_map[x]] for x in reserved_fn_args[:len(self.args)]]

    def __call__(self, _, ctx):
        return self.nc.call(KGRemoteFnCall(self.sym, self._params(ctx)))

    def call_async(self, params, on_done):
        return self.nc.call_async(KGRemoteFnCall(self.sym, list(params)), on_done)

    def __str__(self):
        return f"{self.nc.__str__()}:{self.sym}{super().__str__()}"


# ----------------------------------------------------------------------------
# values to and from the wire
# ----------------------------------------------------------------------------

def _backend(klong):
    return getattr(klong, "_backend", None)


def _backend_name(klong):
    return getattr(_backend(klong), "name", "numpy")


def to_wire_value(klong, v):
    """Backend arrays (torch tensors) to numpy before encoding (tests/test_torch_regressions.py).

    Only values that are backend arrays, and lists and dictionaries holding
    them, are touched; the numpy backend returns everything unchanged and
    costs nothing.
    """
    be = _backend(klong)
    if be is None or _backend_name(klong) == "numpy":
        return v
    try:
        if be.is_backend_array(v):
            return be.to_numpy(v)
        if isinstance(v, (list, tuple)):
            return [to_wire_value(klong, x) for x in v]
        if isinstance(v, dict):
            return {to_wire_value(klong, key): to_wire_value(klong, x) for key, x in v.items()}
        if isinstance(v, np.ndarray) and v.dtype == object:
            out = np.empty(len(v), dtype=object)
            for i, x in enumerate(v):
                out[i] = to_wire_value(klong, x)
            return out
    except Exception:  # noqa: BLE001 - leave it to the encoder to report a bad value
        return v
    return v


def _from_wire_array(klong, v):
    """Decoded numeric numpy arrays to the backend's array type (tests/test_torch_regressions.py)."""
    if isinstance(v, np.ndarray) and v.dtype.kind in "if" and _backend_name(klong) != "numpy":
        try:
            return _backend(klong).kg_asarray(v)
        except Exception:  # noqa: BLE001
            return v
    return v


def display_arity(text):
    """Arity of a display-form lambda text (``:monad``, ``:fn/5``), else None."""
    return native().function_display_arity(text)


def parse_function(klong, text):
    """The function a source text defines, or None.

    The text must be exactly one function literal, plain (``{x+y}``) or named (``fn(a;b)::{a+b}``): the extension's
    function_form says so without running anything. Only then is it built (evaluated, which only makes the function).
    """
    try:
        form = native().function_form(text)
        if form is None:
            return None
        named = form[0]
        return klong._function_from_text(text, named)
    except Exception:  # noqa: BLE001 - any failure means "not a function we understand"
        return None


def lambda_value(klong, lam, nc=None, sym=None):
    """A ``kg``-context lambda received from the peer as a Klong value.

    With ``nc`` and ``sym`` (the answer to a symbol lookup) it becomes a remote
    function proxy bound to that connection and name.  Otherwise a source text
    becomes a local function and a display form an opaque ``KGRemoteFnRef``.
    A lambda of any other context (a q function) stays as it is.
    """
    if not isinstance(lam, KdbLambda) or not lam.is_klong:
        return lam
    arity = display_arity(lam.text)
    fn = None
    if arity is None:
        fn = parse_function(klong, lam.text)
        if fn is None:
            return lam
        arity = fn.arity
    if nc is not None and sym is not None:
        return KGRemoteFnProxy(nc, sym, arity)
    return fn if fn is not None else KGRemoteFnRef(arity)


def from_wire(klong, v, nc=None, sym=None):
    if isinstance(v, KdbLambda):
        return lambda_value(klong, v, nc, sym)
    return _from_wire_array(klong, v)


def to_wire(msg):
    """A call as ``Client.call`` receives it to ``(value, is_call, is_lookup)`` for the session.

    * ``KGRemoteFnCall(sym, params)``  -> general list ``(sym; params...)``; with no params
      ``(sym; ::)``, q's way to call a niladic function (a one-item list is data, see ``call_items``)
    * ``KGRemoteDictSetCall(k, v)``    -> general list ``("set"; k; v)``
    * ``KGRemoteDictGetCall(k)``, a symbol -> symbol atom (lookup)
    * anything else (string to evaluate, data)     -> itself
    """
    if isinstance(msg, KGRemoteFnCall):
        return [msg.sym, *(msg.params if len(msg.params) else [KLONG_UNDEFINED])], True, False
    if isinstance(msg, KGRemoteDictSetCall):
        key = msg.key if isinstance(msg.key, KGSym) else KGSym(msg.key)
        return ["set", key, msg.value], True, False
    if isinstance(msg, KGRemoteDictGetCall):
        return (msg.key if isinstance(msg.key, KGSym) else KGSym(msg.key)), False, True
    return msg, False, isinstance(msg, KGSym)


# ----------------------------------------------------------------------------
# serving a request
# ----------------------------------------------------------------------------

def _is_text(v):
    return type(v) is str


def call_items(value):
    """The items of a call envelope, or None when ``value`` is plain data.

    A general list, or a symbol vector, whose first item is a symbol or a char
    vector, with at least one more item (the decoder cannot tell a one-item
    symbol vector from a one-item general list; q calls a niladic function
    as ``(`f; ::)``, and so do we); or whose first item is a ``kg`` lambda.
    """
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.dtype != object:
            return None
    elif not isinstance(value, (list, tuple)):
        return None
    if len(value) == 0:
        return None
    first = value[0]
    if isinstance(first, KdbLambda):
        return list(value) if first.is_klong else None
    if (isinstance(first, KGSym) or _is_text(first)) and len(value) >= 2:
        return list(value)
    return None


def _arity_of(target):
    if isinstance(target, KGLambda):
        return target.get_arity()
    if isinstance(target, KGFn):
        return target.arity
    return None


def _lookup(klong, sym):
    """The value a symbol names for a peer: a global, or for a dotted name (mgmt.call, which the module system reads as mgmt@"call")
    the entry of the dictionary the first part names. KeyError: symbol not found."""
    try:
        return klong[sym]
    except KeyError:
        name = str(sym)
        parts = name.split(".")
        if name.startswith(".") or len(parts) < 2 or not all(parts):
            raise
        value = klong[parts[0]]
        for key in parts[1:]:
            if not isinstance(value, dict) or key not in value:
                raise KeyError(name) from None
            value = value[key]
        return value


def _apply(klong, callee, args):
    args = [lambda_value(klong, a) for a in args]
    if isinstance(callee, KdbLambda):
        fn = parse_function(klong, callee.text)
        if fn is None:
            raise KlongTypeError(f"cannot run function text from the peer: {callee.text}")
        target = fn
    elif isinstance(callee, KGSym):
        target = _lookup(klong, callee)          # KeyError: symbol not found
        if isinstance(target, KGFnWrapper):
            target = target.fn
        fn = callee if callee in klong else target
    else:
        target = fn = klong(callee)
    if not (isinstance(target, (KGFn, KGLambda)) or callable(target)):
        raise KlongTypeError(f"not callable: {callee}")
    if len(args) == 1 and args[0] is KLONG_UNDEFINED and _arity_of(target) == 0:
        args = []                                # q's (`f; ::) calls f with no arguments
    return klong.apply(fn, args)


def evaluate_request(klong, value):
    """Do what a request message asks; may raise."""
    if isinstance(value, KGSym):
        return _lookup(klong, value)
    if _is_text(value):
        return klong(value)
    items = call_items(value)
    if items is None:
        return value
    callee, args = items[0], items[1:]
    if (len(items) == 3 and isinstance(callee, (str, KGSym)) and str(callee) == "set"       # KGSym("set") != "set"
            and isinstance(args[0], KGSym) and not _SET_SYM in klong):
        klong[args[0]] = lambda_value(klong, args[1])
        return KLONG_UNDEFINED
    return _apply(klong, callee, args)


def response_value(klong, r):
    """What goes into the response for the result ``r`` of a request."""
    if r is None:
        return KLONG_UNDEFINED
    if isinstance(r, KGFnWrapper):
        r = r.fn
    if isinstance(r, KGRemoteFnRef):         # a function that arrived as its display form goes back the same way
        return KdbLambda("kg", str(r))
    return to_wire_value(klong, r)


# ----------------------------------------------------------------------------
# the connection
# ----------------------------------------------------------------------------

def _is_loopback(host):
    host = str(host)
    return host in ("localhost", "::1") or host.startswith("127.")


class KdbConnectionMixin:
    """The protocol half of ``NetworkClient``.

    Needs from the host class: ``ioloop``, ``klongloop``, ``klong``,
    ``conn_provider``, ``reader``, ``writer``, ``session``, ``role``, ``user``,
    ``password``, ``max_message``, ``authenticate``, ``on_error``,
    ``pending_responses``, ``call_timeout``, ``is_open()``,
    ``_cleanup_pending_responses`` and the locks created in its ``__init__``.
    """

    # -- small helpers -------------------------------------------------------

    def _failure(self, msg):
        return KlongIPCConnectionFailureException(msg)

    def _make_proxy(self, sym, arity):
        return KGRemoteFnProxy(self, sym, arity)

    def _send(self, data):
        """Queue ``data`` for the socket; call with ``self._lock`` held to keep frame order."""
        writer = self.writer
        if writer is None:
            raise self._failure(f"connection lost: {str(self.conn_provider)}")

        def write():
            if not writer.is_closing():
                writer.write(data)
        try:
            self.ioloop.call_soon_threadsafe(write)
        except RuntimeError:
            raise self._failure(f"connection lost: {str(self.conn_provider)}") from None

    def _on_klongloop(self, fn):
        try:
            self.klongloop.call_soon_threadsafe(fn)
        except RuntimeError as e:
            logging.info(f"IPC: cannot run work, the klong loop is closed: {e}")

    def _peer_host(self):
        return getattr(self.conn_provider, "host", "")

    def _abort(self, err):
        """Close the connection and fail everything pending on it with ``err``."""
        writer = self.writer
        with self._lock:
            self._cleanup_pending_responses(err)
        if writer is not None:
            try:
                self.ioloop.call_soon_threadsafe(writer.close)
            except RuntimeError:
                pass

    # -- handshake (ioloop) ---------------------------------------------------

    async def _read_some(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise KlongIPCHandshakeException(
                f"handshake timed out after {HANDSHAKE_TIMEOUT}s: {str(self.conn_provider)} "
            "(a peer that does not speak the kdb+ protocol, such as a KlongPy 0.7 server, never answers)")
        try:
            return await asyncio.wait_for(self.reader.read(READ_SIZE), remaining)
        except asyncio.TimeoutError:
            raise KlongIPCHandshakeException(
                f"handshake timed out after {HANDSHAKE_TIMEOUT}s: {str(self.conn_provider)} "
            "(a peer that does not speak the kdb+ protocol, such as a KlongPy 0.7 server, never answers)") from None
        except (OSError, ConnectionError):
            raise KlongIPCHandshakeException(
                f"connection lost during the handshake: {str(self.conn_provider)}") from None

    async def _handshake(self):
        """Run the handshake for this connection's role.  Leaves ``self.session`` ready and
        any events that arrived behind the greeting in ``self._early_events``."""
        deadline = time.monotonic() + HANDSHAKE_TIMEOUT
        where = str(self.conn_provider)
        if self.role == "client":
            sess = Session.client(self.user, self.password, max_message=self.max_message)
            try:
                hello = sess.hello()
            except ValueError as e:
                raise KlongIPCHandshakeException(f"cannot connect to {where}: {e}") from None
            with self._lock:
                self.session = sess
            self.writer.write(hello)
            while True:
                data = await self._read_some(deadline)
                with self._lock:
                    events = sess.feed(data) if data else sess.feed_eof()
                for i, ev in enumerate(events):
                    if isinstance(ev, HandshakeAccepted):
                        self._early_events = events[i + 1:]
                        return
                    if isinstance(ev, HandshakeRejected):
                        raise KlongIPCHandshakeException(
                            f"{where} closed the connection during the handshake "
                            "(bad credentials, or it does not speak the kdb+ protocol)")
                    if isinstance(ev, Failure):
                        raise KlongIPCHandshakeException(f"bad handshake reply from {where}: {ev.error}")
        # server
        sess = Session.server(max_message=self.max_message)
        with self._lock:
            self.session = sess
        events = sess.feed(self._initial)
        self._initial = b""
        hs = None
        while hs is None:
            for ev in events:
                if isinstance(ev, Handshake):
                    hs = ev
                    break
                if isinstance(ev, (Failure, Closed)):
                    raise KlongIPCHandshakeException(f"bad handshake from {where}")
            if hs is None:
                data = await self._read_some(deadline)
                with self._lock:
                    events = sess.feed(data) if data else sess.feed_eof()
        ok = True
        if self.authenticate is not None:
            ok = await self.authenticate(self, hs.user, hs.password)
        if not ok:
            with self._lock:
                sess.reject()
            raise KlongIPCHandshakeException(f"access denied for {hs.user!r} from {where}")
        with self._lock:
            reply = sess.accept()
            self.writer.write(reply)
            self._early_events = sess.feed(b"")

    # -- reader loop (ioloop) ------------------------------------------------

    async def _listen_kdb(self):
        try:
            data = await self.reader.read(READ_SIZE)
        except (OSError, ConnectionError) as e:
            logging.info(f"Connection error {e}")
            raise self._failure(f"connection lost: {str(self.conn_provider)}") from None
        with self._lock:
            sess = self.session
            events = [] if sess is None else (sess.feed(data) if data else sess.feed_eof())
        if not data and not events:
            raise KlongIPCConnectionClosedException(f"connection closed: {str(self.conn_provider)}")
        self._on_events(events)

    def _on_events(self, events):
        for ev in events:
            if isinstance(ev, Message):
                if ev.kind == "response":
                    self._on_response(ev)
                else:
                    self._dispatch_request(ev)
            elif isinstance(ev, Failure):
                raise self._failure(f"protocol error from {str(self.conn_provider)}: {ev.error}")
            elif isinstance(ev, Closed):
                if ev.reason == "eof":
                    raise KlongIPCConnectionClosedException(f"connection closed by peer: {str(self.conn_provider)}")
                raise self._failure(f"connection lost ({ev.reason}): {str(self.conn_provider)}")

    def _on_response(self, ev):
        with self._lock:
            waiter = self.pending_responses.pop(ev.pending, None)
        if waiter is None or waiter.done():
            return
        if ev.error is not None:
            waiter.set_exception(ERROR_KINDS[ev.error.kind](str(ev.error)))   # a response that did not decode
        else:
            waiter.set_result(ev.value)

    def _dispatch_request(self, msg):
        """Hand an incoming request to the innermost blocked caller, else to the klong loop."""
        schedule = False
        with self._lock:
            if self._waiters:
                self._waiters[-1].inbox.put(("request", msg))
                return
            self._serve_queue.append(msg)
            if not self._serve_scheduled:
                self._serve_scheduled = schedule = True
        if schedule:
            self._on_klongloop(self._drain)

    def _redispatch(self, msgs):
        for m in msgs:
            self._dispatch_request(m)

    def _drain(self):
        """Serve queued requests, strictly in arrival order (klong loop thread)."""
        while True:
            with self._lock:
                if not self._serve_queue or self.session is None:
                    self._serve_scheduled = False
                    self._serve_queue.clear()
                    return
                msg = self._serve_queue.popleft()
            self._serve(msg)

    # -- serving -------------------------------------------------------------

    def _serve(self, msg):
        """Serve one request message on the current thread and answer it."""
        sync = msg.kind == "sync"
        with self._lock:
            sess = self.session
            if sess is None or sess.closed:
                return
            try:
                svc = sess.begin_service(msg)
            except NestingLimit:
                if sync:
                    try:
                        self._send(sess.refuse(msg))
                    except KlongIPCConnectionFailureException:
                        pass
                return
        klong = self.klong
        ctx = klong
        had = True
        try:
            prev = ctx[_HANDLE_SYM]
        except KeyError:
            had, prev = False, None
        ctx[_HANDLE_SYM] = self
        error = None
        result = None
        try:
            try:
                if msg.error is not None:
                    raise msg.error
                result = response_value(klong, evaluate_request(klong, msg.value))
            except Exception as e:  # noqa: BLE001 - every failure becomes an error response
                error = wire_error_text(e)
                logging.info(f"IPC: request failed: {error}")
            if sync:
                self._respond(sess, svc, result, error)
            elif error is not None:
                self._report_error(error)
        finally:
            if had:
                ctx[_HANDLE_SYM] = prev
            else:
                try:
                    del ctx[_HANDLE_SYM]
                except KeyError:
                    pass
            with self._lock:
                try:
                    sess.end_service(svc)
                except ProtocolError as e:
                    self._abort(self._failure(f"protocol error from {str(self.conn_provider)}: {e}"))
                except IPCStateError:
                    pass                      # the connection went away while serving

    def _respond(self, sess, svc, result, error):
        with self._lock:
            try:
                if error is not None:
                    data = sess.send_error(svc, error)
                else:
                    try:
                        data = sess.send_response(svc, result, compress_body=self._compress_ok())
                    except IPCError as e:         # the result has no kdb+ form
                        data = sess.send_error(svc, wire_error_text(e))
                self._send(data)
            except (IPCStateError, KlongIPCConnectionFailureException):
                logging.info("IPC: connection closed before the response could be sent")

    def _compress_ok(self):
        return bool(getattr(self.klong, "ipc_compress", False)) and not _is_loopback(self._peer_host())

    def _report_error(self, text):
        """Errors of a request nobody waits for go to the ``.srv.e`` hook and the log."""
        logging.warning(f"IPC: error serving a request from {str(self.conn_provider)}: {text}")
        if self.on_error is not None:
            try:
                asyncio.run_coroutine_threadsafe(self.on_error(self, text), self.ioloop)
            except RuntimeError:
                pass

    # -- calling -------------------------------------------------------------

    def _register(self, msg, waiter, blocking, timeout):
        """Frame ``msg`` as a sync request, register ``waiter`` and send.  Returns the Pending, lookup symbol."""
        value, as_call, lookup = to_wire(msg)
        lookup_sym = value if lookup else None
        value = to_wire_value(self.klong, value) if not as_call else [to_wire_value(self.klong, v) for v in value]
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            sess = self.session
            if sess is None or sess.closed:
                raise self._failure(f"connection lost: {str(self.conn_provider)}")
            try:
                data, pend = sess.send_sync(value, call=as_call, blocking=blocking, tag=waiter,
                                            deadline=deadline, compress_body=self._compress_ok())
            except IPCError as e:
                raise ERROR_KINDS[e.kind](str(e)) from None
            self.pending_responses[pend] = waiter
            if blocking:
                self._waiters.append(waiter)
            try:
                self._send(data)
            except Exception:
                self.pending_responses.pop(pend, None)
                if blocking:
                    self._waiters.remove(waiter)
                raise
        return pend, lookup_sym

    def _result(self, raw, lookup_sym):
        if isinstance(raw, KdbError):
            raise KlongRemoteException(raw.text, self.conn_provider)
        return from_wire(self.klong, raw, self, lookup_sym)

    def _call_kdb(self, msg, timeout):
        waiter = Waiter()
        pend, lookup_sym = self._register(msg, waiter, True, timeout)
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            raw = self._wait(waiter, deadline, timeout)
        finally:
            with self._lock:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                self.pending_responses.pop(pend, None)
            # a request that reached this thread as it was leaving belongs to someone else
            late = []
            while True:
                try:
                    kind, payload = waiter.inbox.get_nowait()
                except queue.Empty:
                    break
                if kind == "request":
                    late.append(payload)
            self._redispatch(late)
        return self._result(raw, lookup_sym)

    def _wait(self, waiter, deadline, timeout):
        """Block for the response, serving requests that arrive meanwhile."""
        try:
            while True:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    self._timed_out(timeout)
                try:
                    kind, payload = waiter.inbox.get(timeout=remaining)
                except queue.Empty:
                    self._timed_out(timeout)
                if kind == "response":
                    return payload
                if kind == "failed":
                    raise payload
                self._serve(payload)
        except KeyboardInterrupt:
            self._abort(self._failure(f"interrupted while waiting: {str(self.conn_provider)}"))
            raise

    def _timed_out(self, timeout):
        err = KlongIPCTimeoutException(f"call timed out after {timeout}s: {str(self.conn_provider)}")
        self._abort(err)
        raise err

    def call_async(self, msg, on_done, timeout=...):
        """Send ``msg`` as a sync request without blocking; ``on_done(result, error)`` runs on the klong loop.

        Nothing waits for the answer, so the connection's deadline is enforced by a timer on the ioloop.
        """
        if not self.is_open():
            raise KlongSystemError("connection not established")
        if timeout is ...:
            timeout = self.call_timeout
        lookup_holder = []

        def finish(raw, err):                      # ioloop thread (or the failing thread)
            def run():                             # klong loop
                if err is not None:
                    on_done(None, err)
                    return
                try:
                    result = self._result(raw, lookup_holder[0] if lookup_holder else None)
                except Exception as e:  # noqa: BLE001
                    on_done(None, e)
                    return
                on_done(result, None)
            self._on_klongloop(run)

        waiter = CallbackWaiter(finish)
        _, lookup_sym = self._register(msg, waiter, False, timeout)
        lookup_holder.append(lookup_sym)
        if timeout is not None:
            try:
                self.ioloop.call_soon_threadsafe(
                    lambda: self.ioloop.call_later(timeout, self._check_deadlines))
            except RuntimeError:
                pass

    def _check_deadlines(self):
        """ioloop timer: a request past its deadline closes the connection."""
        with self._lock:
            sess = self.session
            if sess is None or sess.closed:
                return
            ev = sess.check_deadlines(time.monotonic())
            nxt = None if ev is not None else sess.next_deadline()
        if ev is not None:
            err = self._failure(f"call timed out after {self.call_timeout}s: {str(self.conn_provider)}")
            with self._lock:
                self._cleanup_pending_responses(err)
            if self.writer is not None:
                self.writer.close()
        elif nxt is not None:
            self.ioloop.call_later(max(0.0, nxt - time.monotonic()), self._check_deadlines)


# ----------------------------------------------------------------------------
# the server side of the handshake: .srv.auth
# ----------------------------------------------------------------------------

def auth_accepts(result):
    """``.srv.auth`` returns 1 to accept, 0 to refuse."""
    try:
        return bool(int(result))
    except (TypeError, ValueError):
        return False


def new_connection_state(obj):
    """The per-connection state ``KdbConnectionMixin`` expects, set on ``obj``."""
    obj._lock = threading.RLock()
    obj._waiters = []
    obj._serve_queue = deque()
    obj._serve_scheduled = False
    obj._early_events = []
    obj.session = None


# ----------------------------------------------------------------------------
# function source text
# ----------------------------------------------------------------------------

def function_source_text(fn):
    """Source text of a function literal as the engine recorded it, or None.

    A function of x, y and z travels as its literal, ``{x+y}``.  A function with
    named parameters travels as the definition that made it, ``fn(a;b)::{a+b}``
    (the name ``fn`` is a placeholder: the receiver parses the text and keeps
    only the function, it defines nothing).  Locals lists, head destructurings
    and ``.f`` are part of the literal's text.  A projection, an operator or a
    function with no recorded source goes out in its display form.  The receiver
    parses the text, so names in the body are the receiver's.  The wire form is the extension's (function_wire_text).
    """
    if getattr(fn, 'args', None) is not None:
        return None
    text = getattr(fn, 'source_text', None)     # the native engine keeps the literal's text itself
    if text is None:
        return None
    params = getattr(getattr(fn, 'a', None), 'params', None)
    names = None if params is None else [str.__str__(p).split('`')[0] for p in params]
    return native().function_wire_text(text, names)


set_function_text_hook(function_source_text)
