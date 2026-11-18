"""The legacy pickle IPC protocol (deprecated).

Until the kdb+ wire protocol, KlongPy peers exchanged ``16-byte message id +
4-byte big-endian length + pickle``.  That format is kept here, unchanged, for
one transition release so that old servers and old clients still work:

* ``.srv`` serves a legacy client when it recognises one (see
  :func:`detect_protocol`), or when ``KLONGPY_IPC=legacy``;
* ``.cli`` speaks it when ``KLONGPY_IPC=legacy`` (for talking to an old server).

Unpickling runs arbitrary code from the peer.  That is the reason this path is
deprecated and why it prints one warning per process.

The value classes below are pickled by module path, so their ``__module__`` is
pinned to ``klongpy.sys_fn_ipc``, where they have always lived (and where they
are still importable from).
"""
import asyncio
import concurrent.futures
import logging
import pickle
import struct
import uuid
import warnings
from asyncio import StreamReader, StreamWriter
from asyncio.exceptions import IncompleteReadError

from ..core import (KGFn, KGFnWrapper, KGLambda, KGSym, KlongException,
                    get_fn_arity_str, reserved_fn_args, reserved_fn_symbols)
from ..errors import KlongSystemError, KlongTypeError
from .errors import KlongIPCTimeoutException

LEGACY_MODULE = "klongpy.sys_fn_ipc"


def _pin(cls):
    cls.__module__ = LEGACY_MODULE
    return cls


@_pin
class KGRemoteCloseConnection():
    pass


@_pin
class KGRemoteFnRef:
    def __init__(self, arity):
        self.arity = arity

    def __str__(self):
        return get_fn_arity_str(self.arity)


@_pin
class KGRemoteFnCall:
    def __init__(self, sym: KGSym, params):
        self.sym = sym
        self.params = params


@_pin
class KGRemoteDictSetCall:
    def __init__(self, key, value):
        self.key = key
        self.value = value


@_pin
class KGRemoteDictGetCall:
    def __init__(self, key):
        self.key = key


class KGRemoteCloseConnectionException(KlongException):
    pass


# ----------------------------------------------------------------------------
# protocol selection
# ----------------------------------------------------------------------------

IPC_MODES = ("kdb", "legacy", "auto")


def ipc_mode(default):
    """The protocol mode from ``KLONGPY_IPC`` (read at each use), else ``default``."""
    import os
    v = os.environ.get("KLONGPY_IPC", "").strip().lower()
    if not v:
        return default
    if v not in IPC_MODES:
        raise KlongSystemError(f"KLONGPY_IPC must be one of {', '.join(IPC_MODES)}, not {v!r}")
    return v


def detect_protocol(buf):
    """Tell a kdb+ handshake from a legacy frame without consuming anything.

    Scan forward; a byte of 0x80 or above before any zero
    means legacy (byte 8 of a version-4 UUID always is); a zero preceded by a
    byte from 1 to 6, with a ``:`` earlier, means kdb+; any other zero means
    legacy.  Returns ``"kdb"``, ``"legacy"``, ``None`` (need more bytes) or
    ``"bad"`` (256 bytes without either: neither protocol).
    """
    for i, b in enumerate(buf):
        if b >= 0x80:
            return "legacy"
        if b == 0:
            if i > 0 and 1 <= buf[i - 1] <= 6 and b":" in bytes(buf[:i - 1]):
                return "kdb"
            return "legacy"
        if i >= 255:
            return "bad"
    return None


_warned = False

LEGACY_WARNING = ("the pickle IPC protocol is deprecated and will be removed: it lets any peer run "
                  "arbitrary Python code. Upgrade the peer to speak the kdb+ protocol "
                  "(KLONGPY_IPC=kdb), or set KLONGPY_IPC=kdb to refuse legacy peers.")


def warn_legacy_once():
    """Print the deprecation warning, once per process."""
    global _warned
    if _warned:
        return
    _warned = True
    warnings.warn(LEGACY_WARNING, FutureWarning, stacklevel=2)


class PrefixedReader:
    """A reader that serves ``prefix`` first, then the underlying stream.

    The server reads the first bytes of a connection to decide which protocol
    it is; this gives them back to the legacy code, which reads with
    ``readexactly``.
    """

    def __init__(self, prefix, reader):
        self._prefix = bytes(prefix)
        self._reader = reader

    async def readexactly(self, n):
        if not self._prefix:
            return await self._reader.readexactly(n)
        head, self._prefix = self._prefix[:n], self._prefix[n:]
        if len(head) == n:
            return head
        return head + await self._reader.readexactly(n - len(head))

    def at_eof(self):
        return not self._prefix and self._reader.at_eof()

    def __getattr__(self, name):
        return getattr(self._reader, name)


# ----------------------------------------------------------------------------
# pickle framing
# ----------------------------------------------------------------------------

def encode_message(msg_id, msg):
    data = pickle.dumps(msg)
    length_bytes = struct.pack("!I", len(data))
    return msg_id.bytes + length_bytes + data


def decode_message_len(raw_msglen):
    return struct.unpack('!I', raw_msglen)[0]


def decode_message(raw_msg_id, data):
    msg_id = uuid.UUID(bytes=raw_msg_id)
    message_body = pickle.loads(data)
    return msg_id, message_body


async def stream_send_msg(writer: StreamWriter, msg_id, msg):
    writer.write(encode_message(msg_id, msg))
    await writer.drain()


async def stream_recv_msg(reader: StreamReader, max_message=None):
    raw_msg_id = await reader.readexactly(16)
    raw_msglen = await reader.readexactly(4)
    msglen = decode_message_len(raw_msglen)
    if max_message is not None and msglen > max_message:
        # refused before the body is read or unpickled
        raise ConnectionError(f"message of {msglen} bytes is larger than the limit of {max_message}")
    data = await reader.readexactly(msglen)
    return decode_message(raw_msg_id, data)


# ----------------------------------------------------------------------------
# serving a legacy request
# ----------------------------------------------------------------------------

async def execute_server_command(future_loop, result_future, klong, command, nc):
    """

    Execute a command on the klong loop and return the result via the result_future.

    The network connection that initiated the command is pushed onto the context stack as ".cli.h"
    so that it can be used by the command.

    :param future_loop: the loop to run the result_future on
    :param result_future: the future to return the result on
    :param klong: the klong interpreter
    :param command: the command to execute
    :param nc: the network client

    """
    handle_sym = KGSym('.cli.h')
    try:
        klong._context[handle_sym] = nc
        if isinstance(command, KGRemoteFnCall):
            r = klong[command.sym]
            if callable(r):
                if isinstance(r, KGLambda):
                    ctx = {reserved_fn_symbols[i]: command.params[i] for i in range(min(len(reserved_fn_args), len(command.params)))}
                    response = r(klong, ctx)
                else:
                    response = r(*command.params)
            else:
                raise KlongTypeError(f"not callable: {command.sym}")
        elif isinstance(command, KGRemoteDictSetCall):
            klong[command.key] = command.value
            response = None
        elif isinstance(command, KGRemoteDictGetCall):
            response = klong[command.key]
            if isinstance(response, KGFnWrapper):
                response = response.fn
        else:
            response = klong(str(command))
        if isinstance(response, KGFn):
            response = KGRemoteFnRef(response.arity)
        elif isinstance(response, KGLambda):
            # TODO: move to using .arity for KGLambda
            response = KGRemoteFnRef(response.get_arity())
        future_loop.call_soon_threadsafe(result_future.set_result, response)
    except KeyError as e:
        future_loop.call_soon_threadsafe(result_future.set_exception, KlongException(f"symbol not found: {e}"))
    except Exception as e:
        import traceback
        traceback.print_exception(type(e), e, e.__traceback__)
        future_loop.call_soon_threadsafe(result_future.set_exception, KlongException("internal error"))
        logging.error(f"TcpClientHandler::handle_client: Klong error {e}")
    finally:
        try:
            del klong._context[handle_sym]
        except KeyError:
            pass


async def run_command_on_klongloop(klongloop, klong, command, nc):
    result_future = asyncio.Future()
    future_loop = asyncio.get_event_loop()
    assert future_loop != klongloop
    coroutine = execute_server_command(future_loop, result_future, klong, command, nc)
    klongloop.call_soon_threadsafe(asyncio.create_task, coroutine)
    result = await result_future
    return result


# ----------------------------------------------------------------------------
# the connection side of the legacy protocol
# ----------------------------------------------------------------------------

class LegacyConnectionMixin:
    """The pickle protocol half of ``NetworkClient`` (used when ``protocol == "legacy"``).

    Needs from the host class: ``reader``, ``writer``, ``pending_responses``
    (id -> asyncio future), ``klong``, ``klongloop``, ``ioloop``,
    ``conn_provider``, ``call_timeout``, ``is_open()``, ``_cleanup_pending_responses``,
    ``cleanup()``, ``running``, ``_failure(msg)`` (builds the connection-failure exception).
    """

    async def _listen_legacy(self):
        """

        Listen for messages from the remote peer and dispatch them.

        If there is a pending response for a message, the response is returned via the future.
        If the message is the KGRemoteCloseConnection, then the connection is closed.
        Otherwise, the message is executed on the klong loop and the result is sent back to the peer.

        """
        try:
            msg_id, msg = await stream_recv_msg(self.reader, self.max_message)
            if msg_id in self.pending_responses:
                future = self.pending_responses.pop(msg_id)
                future.set_result(msg)
                if isinstance(msg, KGRemoteCloseConnection):
                    logging.info(f"Recieved close connection ack: {str(self.conn_provider)}")
                    raise KGRemoteCloseConnectionException()
            elif isinstance(msg, KGRemoteCloseConnection):
                logging.info(f"Received remote close connection request: {str(self.conn_provider)}")
                await stream_send_msg(self.writer, msg_id, msg)
                raise KGRemoteCloseConnectionException()
            else:
                response = await run_command_on_klongloop(self.klongloop, self.klong, msg, self)
                await stream_send_msg(self.writer, msg_id, response)
        except (OSError, ConnectionResetError, ConnectionRefusedError, IncompleteReadError) as e:
            logging.info(f"Connection error {e}")
            raise self._failure(f"connection lost: {str(self.conn_provider)}")
        except KGRemoteCloseConnectionException as e:
            raise e
        except Exception as e:
            logging.warning(f"unexpected error: {type(e)} {e}")
            raise e

    def _call_legacy(self, msg, timeout):
        """Send a pickled message and wait for the response (``timeout`` already resolved)."""
        msg_id = uuid.uuid4()
        future = self.ioloop.create_future()
        self.pending_responses[msg_id] = future
        is_lookup = isinstance(msg, (KGSym, KGRemoteDictGetCall))

        async def send_message_and_get_result():
            writer = self.writer
            if writer is None:
                self.pending_responses.pop(msg_id, None)
                raise self._failure(f"connection lost: {str(self.conn_provider)}")
            try:
                await stream_send_msg(writer, msg_id, msg)
                return await asyncio.wait_for(future, timeout)
            except asyncio.TimeoutError:
                self.pending_responses.pop(msg_id, None)
                err = KlongIPCTimeoutException(f"call timed out after {timeout}s: {str(self.conn_provider)}")
                self._cleanup_pending_responses(err)
                if writer is self.writer:
                    writer.close()
                raise err from None

        from .endpoint import CLOSE_TIMEOUT  # late: endpoint imports this module
        cf = asyncio.run_coroutine_threadsafe(send_message_and_get_result(), self.ioloop)
        try:
            response = cf.result(None if timeout is None else timeout + CLOSE_TIMEOUT)
        except concurrent.futures.TimeoutError:
            cf.cancel()
            self.pending_responses.pop(msg_id, None)
            raise KlongIPCTimeoutException(f"call timed out after {timeout}s: {str(self.conn_provider)}") from None
        if is_lookup and isinstance(response, KGRemoteFnRef):
            sym = msg.key if isinstance(msg, KGRemoteDictGetCall) else msg
            response = self._make_proxy(sym, response.arity)
        return response

    def _close_legacy(self):
        """Tell the legacy peer to close gracefully, then close (bounded)."""
        from .endpoint import CLOSE_TIMEOUT
        if self.is_open():
            timeout = CLOSE_TIMEOUT if self.call_timeout is None else min(self.call_timeout, CLOSE_TIMEOUT)
            try:
                self._call_legacy(KGRemoteCloseConnection(), timeout)
            except Exception as e:  # noqa: BLE001 - the peer may already be gone
                logging.info(f"close: peer did not acknowledge: {e}")
