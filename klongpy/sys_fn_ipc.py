"""IPC system functions: .cli .clid .clic .clit .srv .async and the .srv.* hooks.

The connection speaks the kdb+ IPC protocol (``klongpy.ipc``): see
``klongpy/ipc/endpoint.py`` for what a request means and how calls are matched
and served.  The pickle protocol of earlier releases is kept for one transition
release in ``klongpy/ipc/legacy.py`` and selected with ``KLONGPY_IPC``:

    KLONGPY_IPC=kdb      speak and accept only the kdb+ protocol
    KLONGPY_IPC=legacy   speak and accept only the legacy pickle protocol
    KLONGPY_IPC=auto     (server default) accept both, telling them apart on accept

``.cli``/``.clid`` default to the kdb+ protocol.
"""
import asyncio
import concurrent.futures
import logging
import pickle  # noqa: F401  (legacy names stay importable from here)
import struct  # noqa: F401
import sys
import threading
import uuid  # noqa: F401
from asyncio import StreamReader, StreamWriter

from klongpy.core import (KGCall, KGFn, KGFnWrapper, KGLambda, KGSym,
                          KlongException, KLONG_UNDEFINED, is_list,  # noqa: F401
                          reserved_fn_args, reserved_fn_symbols, reserved_fn_symbol_map)

from klongpy.errors import KlongRangeError, KlongSystemError, KlongTypeError
from klongpy.ipc.errors import (KlongIPCException, KlongIPCConnectionClosedException,  # noqa: F401
                                KlongIPCConnectionFailureException,
                                KlongIPCCreateConnectionException, KlongIPCHandshakeException,
                                KlongIPCTimeoutException)
from klongpy.ipc.endpoint import (CLOSE_TIMEOUT, DEFAULT_CALL_TIMEOUT, HANDSHAKE_TIMEOUT,  # noqa: F401
                                  KdbConnectionMixin, KGRemoteFnProxy, KlongRemoteException,
                                  auth_accepts, new_connection_state)
from klongpy.ipc import endpoint as _endpoint
from klongpy.ipc.legacy import (KGRemoteCloseConnection, KGRemoteCloseConnectionException,  # noqa: F401  (legacy names)
                                KGRemoteDictGetCall, KGRemoteDictSetCall, KGRemoteFnCall,
                                KGRemoteFnRef, LegacyConnectionMixin, PrefixedReader,
                                decode_message, decode_message_len, detect_protocol,
                                encode_message, execute_server_command, ipc_mode,
                                run_command_on_klongloop, stream_recv_msg, stream_send_msg,
                                warn_legacy_once)


def get_system(klong):
    """Return the interpreter's `.system` dictionary (ioloop, klongloop, closeEvent)."""
    try:
        return klong['.system']
    except KeyError:
        raise KlongSystemError(".system is not defined: this interpreter was not created by create_repl "
                             "(IPC needs .system with ioloop, klongloop and closeEvent)") from None


class ConnectionProvider:
    async def connect(self):
        raise KlongIPCCreateConnectionException()

    async def close(self):
        raise NotImplementedError()


class HostPortConnectionProvider(ConnectionProvider):
    """

    This connection provider is used to create a NetworkClient from a host/port pair.

    """
    def __init__(self, host, port, max_retries=5, retry_delay=5.0):
        self.host = host
        self.port = port
        self.running = True
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.reader = None
        self.writer = None
        self._thread_ident = None

    async def connect(self):
        """"

        Attempt to connect to the remote server.  If the connection fails, retry up to max_retries times.

        """
        self._thread_ident = threading.current_thread().ident
        current_delay = self.retry_delay
        retries = 0
        while self.running and retries < self.max_retries:
            try:
                logging.info(f"connecting to {self.host}:{self.port}")
                self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
                logging.info(f"connected to {self.host}:{self.port}")
                retries = 0
                return self.reader, self.writer
            except (OSError, ConnectionResetError, ConnectionRefusedError):
                if not self.running:
                    break
                retries += 1
                logging.info(f"connection error to {self.host}:{self.port} retries: {retries} delay: {current_delay}")
                await asyncio.sleep(current_delay)
                current_delay *= 2
            except Exception as e:
                logging.warning(f"unexpeced connection error {e} to {self.host}:{self.port}")
                break
        if retries >= self.max_retries:
            logging.info(f"Max retries reached: {self.max_retries} {self.host}:{self.port}")
            raise KlongIPCCreateConnectionException()
        logging.info(f"Stopping client: {self.host}:{self.port}")
        return None, None

    async def close(self):
        """

        Close the connection.  This is called when the client is stopped.

        """
        assert threading.current_thread().ident == self._thread_ident
        if not self.is_open():
            return
        self.running = False
        if self.writer is not None:
            self.writer.close()
            await self.writer.wait_closed()
        self.reader = None
        self.writer = None

    def is_open(self):
        # TODO: are there threading access risks here?
        return self.writer is not None and not self.writer.is_closing()

    def __str__(self):
        return f"remote[{self.host}:{self.port}]"


class ReaderWriterConnectionProvider(ConnectionProvider):
    """

    This connection provider is used to create a NetworkClient from an existing reader/writer pair.

    """
    def __init__(self, reader: StreamReader, writer: StreamWriter, host, port):
        self.reader = reader
        self.writer = writer
        self.host = host
        self.port = port
        self._thread_ident = None

    async def connect(self):
        self._thread_ident = threading.current_thread().ident
        if not self.is_open():
            raise KlongIPCCreateConnectionException()
        return self.reader, self.writer

    async def close(self):
        if threading.current_thread().ident != self._thread_ident:
            raise RuntimeError("close called from different thread")
        if not self.is_open():
            return
        self.writer.close()
        await self.writer.wait_closed()
        self.writer = None
        self.reader = None

    def is_open(self):
        return self.writer is not None and not self.writer.is_closing()

    def __str__(self):
        return f"remote[{self.host}:{self.port}]"


class NetworkClient(LegacyConnectionMixin, KdbConnectionMixin, KGLambda):
    """

    This network client is used to either connect to a remote server or to handle a connection by a remote client.
    The network connection may be used in KlongPy as a remote dictionary or a remote function.

    If the remote client connects to the server, the server will create a NetworkClient so that it can be used
    in KlongPy as a remote dictionary or a remote function.

    Similarly, if a KlongPy client connects to a remote server, the client will create a NetworkClient so that it
    can be used in KlongPy as a remote dictionary or a remote function.

    ``protocol`` is ``"kdb"`` (default) or ``"legacy"`` (pickle, deprecated). ``role`` is ``"client"`` for a
    connection this process opened and ``"server"`` for one it accepted; the server side of a kdb+ connection
    starts from the bytes ``initial`` that were read to recognise the protocol and calls
    ``authenticate(nc, user, password)`` (a coroutine returning a bool) before accepting.

    """
    def __init__(self, ioloop, klongloop, klong, conn_provider, shutdown_event=None, on_connect=None, on_close=None,
                 on_error=None, *, protocol="kdb", role="client", user="", password="", initial=b"",
                 max_message=None, authenticate=None):
        self.ioloop = ioloop
        self.klongloop = klongloop
        self.klong = klong
        self.shutdown_event = shutdown_event
        self.conn_provider = conn_provider
        # outstanding requests: Pending (kdb+) or message id (legacy) -> waiter / future
        self.pending_responses = {}
        self.running = False
        self.run_task = None
        self.on_connect = on_connect
        self.on_close = on_close
        self.on_error = on_error
        self.reader: StreamReader = None
        self.writer: StreamWriter = None
        self._run_exit_event = threading.Event()
        # deadline in seconds for each call; None disables it. Settable per handle.
        self.call_timeout = DEFAULT_CALL_TIMEOUT
        self.protocol = protocol
        self.role = role
        self.user = user
        self.password = password
        self.max_message = max_message if max_message is not None else _endpoint_max_message()
        self.authenticate = authenticate
        self._initial = bytes(initial)
        new_connection_state(self)

        if shutdown_event is not None:
            self.shutdown_event.subscribe(self.close)

    def _cleanup_pending_responses(self, close_exception):
        """

        Cleanup the pending responses and set the exception on the futures.

        From the KlongPy perspective, any outstanding remote calls will fail with the close_exception.

        """
        if close_exception is None:
            close_exception = KlongIPCConnectionClosedException(f"connection closed: {str(self.conn_provider)}")
        with self._lock:
            if self.session is not None:
                self.session.close()
            pending = list(self.pending_responses.values())
            self.pending_responses.clear()
            for future in pending:
                if not future.done():
                    future.set_exception(close_exception)

    def run_client(self):
        """

        Start the network client as initiatiated by the Klong interpreter.

        The network client will connect to the remote server and handle server push-requests by
        running messages in the Klong interpreter.

        When the .clic function is called, the network client is stopped and the connection is closed.

        Raises KlongSystemError when the server refuses the handshake (wrong credentials, not a kdb+ protocol server).

        """
        self.running = True
        self.role = "client"
        connect_event = threading.Event()
        errors = []

        async def _on_connect(client, **kwargs):
            if client.on_connect is not None:
                await client.on_connect(self)
            connect_event.set()

        async def _on_error(client, e):
            errors.append(e)
            if client.on_error is not None:
                await client.on_error(self, e)
            connect_event.set()

        self.ioloop.call_soon_threadsafe(asyncio.create_task, self._run(_on_connect, self.on_close, _on_error))
        connect_event.wait()
        if errors and isinstance(errors[0], KlongIPCHandshakeException):
            self.cleanup()
            raise KlongSystemError(str(errors[0])) from errors[0]
        return self

    def run_server(self):
        """

        Start the network client for a client connected to the server.

        The network client will handle server requests and shutdown when the connection is closed.

        """
        self.running = True
        self.role = "server"
        return self._run(self.on_connect, self.on_close, self.on_error)

    async def _run(self, on_connect, on_close, on_error):
        """

        Get the connection, run the handshake and start listening for messages.

        If a connection drops or an error occurs, the pending responses are
        cleared and exceptions are returned to the KlongPy callers.

        In order to notify KlongPy applications of network activity:

        When the connection is opened, the on_connect callback is called.
        If a connection is closed, the on_close callback is called (only if on_connect was).
        If an error occurs, the on_error callback is called.

        The handlers call back into the KlongPy runtime and notify the application.

        :param on_connect: called when a connection is established
        :param on_close: called when a connection is closed
        :param on_error: called when a connection error occurs

        """
        while self.running:
            close_exception = None
            connected = False
            try:
                self.reader, self.writer = await self.conn_provider.connect()
                if self.reader is None or self.writer is None:
                    break
                if self.protocol == "kdb":
                    await self._handshake()
                if on_connect is not None:
                    try:
                        await on_connect(self)
                    except Exception as e:
                        logging.warning(f"error while running on_connect handler: {e}")
                connected = True
                early, self._early_events = self._early_events, []
                if early:
                    self._on_events(early)
                while self.running:
                    await self._listen()
            except (KlongIPCConnectionFailureException, KlongIPCCreateConnectionException) as e:
                close_exception = e
                if on_error is not None:
                    try:
                        await on_error(self, e)
                    except Exception as e:
                        logging.warning(f"error while running on_error handler: {e}")
                break
            except KlongIPCConnectionClosedException as e:
                logging.info(f"Remote peer closed the connection: {str(self.conn_provider)}")
                self.running = False
                close_exception = e
                break
            except KGRemoteCloseConnectionException as e:
                logging.info(f"Remote client closing connection: {str(self.conn_provider)}")
                self.running = False
                close_exception = e
                break
            except Exception as e:
                close_exception = KlongIPCConnectionFailureException("unknown error")
                logging.warning(f"Unexepected error {e}.")
                if on_error is not None:
                    try:
                        await on_error(self, e)
                    except Exception as e:
                        logging.warning(f"error while running on_error handler: {e}")
                break
            finally:
                writer = self.writer
                self.writer = None
                self.reader = None
                if writer is not None:
                    try:
                        writer.close()
                    except Exception:  # noqa: BLE001 - already gone
                        pass
                self._cleanup_pending_responses(close_exception)
                with self._lock:
                    self.session = None
                    self._serve_queue.clear()
                if connected and on_close is not None:
                    try:
                        await on_close(self)
                    except Exception as e:
                        logging.warning(f"error while running on_close handler: {e}")
        logging.info(f"Stopping client: {str(self.conn_provider)}")
        self._run_exit_event.set()

    async def _listen(self):
        """Read from the connection once and act on what arrives (see KdbConnectionMixin / LegacyConnectionMixin)."""
        if self.protocol == "legacy":
            return await self._listen_legacy()
        return await self._listen_kdb()

    def call(self, msg, timeout=...):
        """

        Send a message to the remote peer and wait for the response.

        ``msg`` is a string (evaluated remotely), a symbol (looked up remotely), a KGRemoteFnCall,
        KGRemoteDictGetCall or KGRemoteDictSetCall, or any value to send as it is.

        The call fails after "timeout" seconds (default: self.call_timeout, None for no limit).
        A timed out call closes the connection and every pending request on it fails with the same error.
        While waiting, requests from the peer on this connection are served by the waiting thread.

        """
        if not self.is_open():
            raise KlongSystemError("connection not established")
        if timeout is ...:
            timeout = self.call_timeout
        if self.protocol == "legacy":
            return self._call_legacy(msg, timeout)
        return self._call_kdb(msg, timeout)

    @staticmethod
    def arg_to_msg(x):
        """The message for the argument of ``handle(x)``: a list that starts with a symbol is a call."""
        if is_list(x) and len(x) > 0 and isinstance(x[0], KGSym):
            return KGRemoteFnCall(x[0], list(x[1:]))
        return x

    def __call__(self, _, ctx):
        """

        Evaluate a remote function call.

        """
        x = ctx[reserved_fn_symbol_map[reserved_fn_args[0]]]
        return self.call(self.arg_to_msg(x))

    def _stop(self):
        """

        Stop the network client and wait (bounded) for the run loop to exit.

        """
        self.running = False
        if self._run_exit_event.wait(CLOSE_TIMEOUT):
            self._run_exit_event.clear()
        else:
            logging.warning(f"timed out waiting for the client to stop: {str(self.conn_provider)}")

    def cleanup(self):
        """

        Cleanup the network client and the underlying connection.

        """
        if self.shutdown_event is not None:
            self.shutdown_event.unsubscribe(self.close)

        if not self.running:
            return

        # run close in the appropriate context task to avoid deadlock
        try:
            loop = asyncio.get_running_loop()
            if loop == self.ioloop:
                self.ioloop.call_soon(asyncio.create_task, self.conn_provider.close())
                self._stop()
            else:
                raise RuntimeError()
        except RuntimeError:
            self.running = False
            # close the connection first: it wakes a run loop that is blocked reading
            try:
                asyncio.run_coroutine_threadsafe(self.conn_provider.close(), self.ioloop).result(CLOSE_TIMEOUT)
            except (concurrent.futures.TimeoutError, OSError, RuntimeError) as e:
                logging.warning(f"error closing connection {str(self.conn_provider)}: {e}")
            self._stop()

    def close(self):
        """

        Close the network client and the underlying connection.

        A kdb+ connection is just closed (the peer sees the end of the stream). On the legacy protocol
        a message is sent first to tell the peer to close gracefully. If the connection is already gone
        or the peer does not answer, the client is closed anyway.

        """
        if not self.running:
            return
        if self.protocol == "legacy":
            self._close_legacy()
        self.cleanup()

    def is_open(self):
        return self.conn_provider.is_open()

    def get_arity(self):
        return 1

    def __str__(self):
        return f"{str(self.conn_provider)}:fn"

    @staticmethod
    def create_from_conn_provider(ioloop, klongloop, klong, conn_provider, shutdown_event=None, on_connect=None, on_close=None, on_error=None, **kwargs):
        """

        Create a network client to connect to a remote server.

        :param ioloop: the asyncio ioloop
        :param klongloop: the klong loop
        :param klong: the klong interpreter
        :param conn_provider: where the connection comes from
        :param kwargs: protocol, role, user, password, initial, max_message, authenticate (see the class)
        :return: a network client

        """
        return NetworkClient(ioloop, klongloop, klong, conn_provider, shutdown_event=shutdown_event, on_connect=on_connect, on_close=on_close, on_error=on_error, **kwargs)

    @staticmethod
    def create_from_host_port(ioloop, klongloop, klong, host, port, shutdown_event=None, on_connect=None, on_close=None, on_error=None, **kwargs):
        """

        Create a network client to connect to a remote server.

        :param ioloop: the asyncio ioloop
        :param klongloop: the klong loop
        :param klong: the klong interpreter
        :param host: the host to connect to
        :param port: the port to connect to
        :return: a network client

        """
        conn_provider = HostPortConnectionProvider(host, port)
        return NetworkClient.create_from_conn_provider(ioloop, klongloop, klong, conn_provider, shutdown_event=shutdown_event, on_connect=on_connect, on_close=on_close, on_error=on_error, **kwargs)

    @staticmethod
    def create_from_addr(ioloop, klongloop, klong, shutdown_event, addr, on_connect=None, on_close=None, on_error=None, **kwargs):
        """

        Create a network client to connect to a remote server.

        :param ioloop: the asyncio ioloop
        :param klongloop: the klong loop
        :param klong: the klong interpreter
        :param addr: the address to connect to: an integer port (on "localhost"), "host:port" or
                     "host:port:user:password" (the order of kdb+'s hopen).

        :return: a network client

        """
        host, port, user, password = parse_address(addr)
        kwargs.setdefault("user", user)
        kwargs.setdefault("password", password)
        return NetworkClient.create_from_host_port(ioloop, klongloop, klong, host, port, shutdown_event=shutdown_event, on_connect=on_connect, on_close=on_close, on_error=on_error, **kwargs)


def _endpoint_max_message():
    from klongpy.ipc.compress import MAX_MESSAGE
    return MAX_MESSAGE


def parse_address(addr):
    """``port``, ``host:port``, ``host:port:user`` or ``host:port:user:password`` to (host, port, user, password)."""
    addr = str(addr)
    parts = addr.split(":", 3)
    if len(parts) == 1:
        host, port, user, password = "localhost", parts[0], "", ""
    elif len(parts) == 2:
        host, port, user, password = parts[0], parts[1], "", ""
    else:
        host, port, user, password = parts[0], parts[1], parts[2], parts[3] if len(parts) == 4 else ""
    try:
        port = int(port)
    except ValueError:
        raise KlongTypeError(f"bad address {addr!r}: expected port, host:port or host:port:user:password") from None
    return host or "localhost", port, user, password


class TcpServerConnectionHandler:
    def __init__(self, ioloop, klongloop, klong, shutdown_event=None, max_message=None):
        self.ioloop = ioloop
        self.klong = klong
        self.klongloop = klongloop
        self.shutdown_event = shutdown_event
        self.max_message = max_message

    def _call_hook(self, name, *args):
        """Run the Klong hook ``name`` on the klong loop, like any request (a hook may call the peer)."""
        def run():
            try:
                fn = self.klong[name]
                if callable(fn):
                    fn(*args)
            except Exception as e:
                logging.warning(f"Server: error while running {name} handler: {e}")
        try:
            self.klongloop.call_soon_threadsafe(run)
        except RuntimeError as e:
            logging.info(f"Server: cannot run {name}, the klong loop is closed: {e}")

    async def _on_connect(self, nc):
        logging.info(f"New connection from {str(nc.conn_provider)}")
        self._call_hook('.srv.o', nc)

    async def _on_close(self, nc):
        logging.info(f"Connection closed from {str(nc.conn_provider)}")
        self._call_hook('.srv.c', nc)

    async def _on_error(self, nc, e):
        logging.info(f"Connection error from {str(nc.conn_provider)}")
        self._call_hook('.srv.e', nc, e if isinstance(e, str) else str(e))

    async def _authenticate(self, nc, user, password):
        """``.srv.auth(user; password)``: 1 accepts, 0 refuses; no hook accepts everyone."""
        try:
            fn = self.klong['.srv.auth']
        except KeyError:
            return True
        if not callable(fn):
            return True
        cf = concurrent.futures.Future()

        def run():
            try:
                cf.set_result(auth_accepts(fn(user, password)))
            except Exception as e:
                logging.warning(f"Server: error while running .srv.auth handler: {e}")
                cf.set_result(False)
        try:
            self.klongloop.call_soon_threadsafe(run)
            return await asyncio.wait_for(asyncio.wrap_future(cf), _endpoint.HANDSHAKE_TIMEOUT)
        except (asyncio.TimeoutError, RuntimeError):
            return False

    async def _detect(self, reader):
        """Read the first bytes of a connection until the protocol is known (bounded by the handshake timeout)."""
        buf = b""
        deadline = asyncio.get_running_loop().time() + _endpoint.HANDSHAKE_TIMEOUT
        while True:
            verdict = detect_protocol(buf)
            if verdict is not None:
                return verdict, buf
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return None, buf
            try:
                chunk = await asyncio.wait_for(reader.read(4096), remaining)
            except (asyncio.TimeoutError, OSError):
                return None, buf
            if not chunk:
                return None, buf
            buf += chunk

    async def handle_client(self, reader: StreamReader, writer: StreamWriter):
        """

        Handle a client connection.  Messages are read from the client and executed on the klong loop.

        The first bytes tell a kdb+ handshake from a legacy pickle frame (KLONGPY_IPC selects which
        are accepted); anything else, or a peer that says nothing for the handshake timeout, is dropped.

        """
        results = writer.get_extra_info('peername')
        if results is None:
            logging.warning("Connection closed before peername could be retrieved")
            return
        host, port = results[0], results[1]
        if host == "::1":
            host = "localhost"
        mode = ipc_mode("auto")
        verdict, prefix = await self._detect(reader)
        if verdict not in ("kdb", "legacy"):
            logging.info(f"Server: no recognisable handshake from {host}:{port}; dropping")
            return
        if verdict == "kdb" and mode == "legacy":
            logging.info(f"Server: refusing a kdb+ client from {host}:{port} (KLONGPY_IPC=legacy)")
            return
        if verdict == "legacy" and mode == "kdb":
            logging.info(f"Server: refusing a legacy pickle client from {host}:{port} (KLONGPY_IPC=kdb)")
            return
        kwargs = {}
        if verdict == "legacy":
            warn_legacy_once()
            reader = PrefixedReader(prefix, reader)
            kwargs["protocol"] = "legacy"
        else:
            kwargs.update(protocol="kdb", initial=prefix, authenticate=self._authenticate,
                          max_message=self.max_message)
        conn_provider = ReaderWriterConnectionProvider(reader, writer, host, port)
        nc = NetworkClient.create_from_conn_provider(self.ioloop, self.klongloop, self.klong, conn_provider, shutdown_event=self.shutdown_event, on_connect=self._on_connect, on_close=self._on_close, on_error=self._on_error, **kwargs)
        try:
            await nc.run_server()
        finally:
            nc.cleanup()


class TcpServerHandler:
    """

    The IPC server of one interpreter. Each interpreter keeps its own handler in its `.system` dictionary.

    All socket work happens on the ioloop. create_server and shutdown_server may be called from any
    other thread and return once the listening socket is bound, or closed.

    """
    def __init__(self):
        self.connection_handler = None
        self.ioloop = None
        self.server = None
        self.shutdown_event = None
        self.connections = []

    def create_server(self, ioloop, klongloop, klong, bind, port, shutdown_event=None, timeout=CLOSE_TIMEOUT * 2, max_message=None):
        """

        Start listening. Returns 1 when the socket is bound, 0 if this handler is already serving.
        Raises KlongSystemError if the socket cannot be bound.

        """
        if self.server is not None:
            return 0
        self.connection_handler = TcpServerConnectionHandler(ioloop, klongloop, klong, shutdown_event=shutdown_event, max_message=max_message)
        self.ioloop = ioloop
        where = f"{bind or '0.0.0.0'}:{port}"
        cf = asyncio.run_coroutine_threadsafe(self.run_server(bind, port), ioloop)
        try:
            cf.result(timeout)
        except concurrent.futures.TimeoutError:
            cf.cancel()
            self._reset()
            raise KlongSystemError(f"cannot listen on {where}: timed out") from None
        except OSError as e:
            self._reset()
            raise KlongSystemError(f"cannot listen on {where}: {e.strerror or e}") from e
        self.shutdown_event = shutdown_event
        if shutdown_event is not None:
            shutdown_event.subscribe(self.shutdown_server)
        return 1

    def _reset(self):
        self.connection_handler = None
        self.server = None
        self.ioloop = None

    async def _close(self, server, connections):
        server.close()
        for writer in connections:
            if not writer.is_closing():
                writer.close()
        try:
            await asyncio.wait_for(server.wait_closed(), CLOSE_TIMEOUT)
        except asyncio.TimeoutError:
            logging.warning("IPC server: timed out waiting for connections to close")

    def shutdown_server(self):
        """

        Stop listening and drop the client connections. Returns 1 if a server was stopped, else 0.

        """
        server, ioloop = self.server, self.ioloop
        if server is None:
            return 0
        connections = list(self.connections)
        self.connections.clear()
        if self.shutdown_event is not None:
            self.shutdown_event.unsubscribe(self.shutdown_server)
            self.shutdown_event = None
        self.server = None
        self.ioloop = None
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is ioloop:
            # cannot block the ioloop on its own work
            ioloop.create_task(self._close(server, connections))
        else:
            try:
                asyncio.run_coroutine_threadsafe(self._close(server, connections), ioloop).result(CLOSE_TIMEOUT * 2)
            except (concurrent.futures.TimeoutError, RuntimeError) as e:
                logging.warning(f"IPC server: error while shutting down: {e}")
        return 1

    async def handle_client(self, reader, writer):
        self.connections.append(writer)
        handler = self.connection_handler

        try:
            if handler is not None:
                await handler.handle_client(reader, writer)
        finally:
            writer.close()
            if writer in self.connections:
                self.connections.remove(writer)

    async def run_server(self, bind, port):
        self.server = await asyncio.start_server(self.handle_client, bind, port, reuse_address=True, start_serving=True)
        addr = self.server.sockets[0].getsockname()
        logging.info(f'Serving IPC on {addr}')




class NetworkClientDictHandle(dict):
    def __init__(self, nc: NetworkClient):
        self.nc = nc

    def __getitem__(self, x):
        return self.get(x)

    def __setitem__(self, x, y):
        return self.set(x, y)

    def __contains__(self, x):
        raise NotImplementedError()

    def get(self, x):
        return self.nc.call(KGRemoteDictGetCall(x))

    def set(self, x, y):
        self.nc.call(KGRemoteDictSetCall(x, y))
        return self

    @property
    def call_timeout(self):
        return self.nc.call_timeout

    @call_timeout.setter
    def call_timeout(self, seconds):
        self.nc.call_timeout = seconds

    def close(self):
        return self.nc.close()

    def is_open(self):
        return self.nc.is_open()

    def __str__(self):
        return f"{str(self.nc.conn_provider)}:dict"


def eval_sys_fn_create_client(klong, x):
    """

        .cli(x)                                      [Create-IPC-client]

        Return a function which evaluates commands on a remote KlongPy server.

        If "x" is an integer, then it is interpreted as a port in "localhost:<port>".
        if "x" is a string, then it is interpreted as a host address "<host>:<port>",
        or "<host>:<port>:<user>:<password>" to send credentials (the order of kdb+'s hopen).

        If "x" is a remote dictionary, the underlying network connection
        is shared and a remote function is returned.

        The connection speaks the kdb+ IPC protocol (KLONGPY_IPC=legacy selects the
        deprecated pickle protocol, for talking to an old server). Credentials are
        sent in clear text. If the server refuses the handshake, an error is raised.

        Calls wait for the answer, 30 seconds at most (see .clit); a call that times
        out closes the connection. While a call waits, requests the server makes on
        this connection (callbacks) are served.

        Connection examples:

                   .cli(8888)            --> remote function to localhost:8888
                   .cli("localhost:8888") --> remote function to localhost:8888
                   .cli("localhost:8888:bob:secret") --> same, as user bob

                   d::.clid(8888)
                   .cli(d)                --> remote function to same connection as d

        Evaluation examples:

                   f::.cli(8888)

            A string is evaluated remotely as Klong source:

                   f("1+1")               --> 2 (computed remotely)
                   f("avg::{(+/x)%#x}")   --> "avg" function is defined remotely
                   f("avg(!100)")         --> 49.5 (computed remotely)

            Remote functions may be evaluated by passing an array with the first element
            being the symbol of the remote function to execute.  The remaining elements
            are supplied as parameters (as they are: symbols stay symbols):

            Example: call :avg with a locally generated !100 range which is passed to the remote server.

                   f(:avg,,!100)          --> 49.5

            Similary:

                   b::!100
                   f(:avg,,b)             --> 49.5

            When a symbol is applied, the remote value is returned.
            For functions, a remote function proxy is returned.

            Example: retrieve a function proxy to :avg and then call it as if it were local function.

                   q::f(:avg)
                   q(!100)                --> 49.5

            Example: retrieve a copy of a remote array.

                   f("b::!100")
                   p::f(:b)               --> "p: now holds a copy of the remote array "b"

            A function is sent as its source text and the receiver parses it, so the
            names in its body are the receiver's. A projection is sent as an opaque value.

            An error on the server is raised here as the error class of its kind, with the
            server's message: a range error there is a range error here. Text from a peer
            that is not a KlongPy server, or a name KlongPy does not know, is a remote error.
            A connection that fails, is refused or times out is a system error.

    """
    x = x.a if isinstance(x,KGCall) else x
    if isinstance(x,NetworkClient):
        return x
    system = get_system(klong)
    ioloop = system['ioloop']
    klongloop = system['klongloop']
    shutdown_event = system.get('closeEvent')
    if isinstance(x, NetworkClientDictHandle):
        return x.nc
    protocol = "legacy" if ipc_mode("kdb") == "legacy" else "kdb"
    if protocol == "legacy":
        warn_legacy_once()
    return NetworkClient.create_from_addr(ioloop, klongloop, klong, shutdown_event, x, protocol=protocol,
                                          max_message=system.get('ipc_max_message')).run_client()


def eval_sys_fn_create_dict_client(klong, x):
    """

        .clid(x)                                [Create-IPC-dict-client]

        Return a dictionary which evaluates set/get operations on a remote KlongPy server.

        If "x" is an integer, then it is interpreted as a port in "localhost:<port>".
        if "x" is a string, then it is interpreted as a host address "<host>:<port>" or
        "<host>:<port>:<user>:<password>".

        If "x" is a remote function (from .cli), the underlying network connection
        is shared and a remote dictionary is returned. The protocol, timeout and errors
        are those of .cli.

        Connection examples:

                   .clid(8888)            --> remote dictionary to localhost:8888
                   .clid("localhost:8888") --> remote dictionary to localhost:8888

                   f::.cli(8888)
                   .clid(f)               --> remote dictionary on the same connection as f

        Evaluation examples:

                   d::.clid(8888)

            Set a remote key/value pair :foo -> 2 on the remote server.

                   d,[:foo 2]             --> sets :foo to 2
                   d,[:bar "hello"]       --> sets :bar to "hello"
                   d,:fn,{x+1}            --> sets :fn to the monad {x+1}

            Get a remote value:

                   d?:foo                 --> 2
                   d?:bar                 --> "hello"
                   d?:fn                  --> remote function proxy to :fn

            To use the remote function proxy:
                   q::d?:fn
                   q(2)                   --> 3 (remotely executed after passing 2)

            A get of a name the server does not have is a name error. A set sends the
            three-item request ("set"; name; value), which a KlongPy server treats as an
            assignment of a global.

    """
    x = x.a if isinstance(x,KGCall) else x
    if isinstance(x,NetworkClientDictHandle):
        return x
    nc = x if isinstance(x,NetworkClient) else eval_sys_fn_create_client(klong, x)
    return NetworkClientDictHandle(nc)


def eval_sys_fn_shutdown_client(x):
    """

        .clic(x)                                      [Close-IPC-client]

        Close a remote dictionary or function opened by .cli or .clid.

        Returns 1 if closed, 0 if already closed.

        When a connection is closed, all remote proxies / functions tied to this connection
        will also close and will fail if called.

    """
    if isinstance(x, KGCall):
        x = x.a
    if isinstance(x, (NetworkClient, NetworkClientDictHandle)) and x.is_open():
        x.close()
        return 1
    return 0


def eval_sys_fn_client_timeout(x, y):
    """

        .clit(x,y)                                    [Set-IPC-call-timeout]

        Set the call timeout of a remote function or dictionary opened by .cli or .clid
        (or handed to a server hook) to "y" seconds. A call that gets no answer in time
        closes the connection and fails, as does every other call waiting on it.
        The default is 30 seconds; 0 removes the limit.

        Returns 1 when set, 0 if "x" is not a connection handle.

        Example:

                   c::.cli(8888)
                   .clit(c;2.5)           --> 1: calls on c now fail after 2.5 seconds

    """
    if isinstance(x, KGCall):
        x = x.a
    if isinstance(x, KGRemoteFnProxy):
        x = x.nc
    if not isinstance(x, (NetworkClient, NetworkClientDictHandle)):
        return 0
    try:
        seconds = float(y)
    except (TypeError, ValueError):
        raise KlongTypeError("y must be a number of seconds") from None
    if seconds != seconds or seconds < 0:
        raise KlongRangeError("y must be a number of seconds, 0 or more")
    x.call_timeout = None if seconds == 0 else seconds
    return 1


def eval_sys_fn_create_ipc_server(klong, x):
    """

        .srv(x)                                       [Start-IPC-server]

        Open a server port to accept IPC connections.

        If "x" is an integer, then it is interpreted as a port in "<all>:<port>".
        if "x" is a string, then it is interpreted as a bind address "<bind>:<port>"

        if "x" is 0, then the server is closed and existing client connections are dropped.

        Returns 1 when the server is listening and 0 if it was already started
        (or, for 0, if there was no server). If the port cannot be bound, an error
        is raised.

        Each interpreter has its own server, so several interpreters may serve
        different ports.

        Clients speak the kdb+ IPC protocol, so q and other kdb+ client libraries can
        connect. A client that speaks the deprecated pickle protocol of earlier KlongPy
        releases is recognised and served too (with a one-time warning), unless
        KLONGPY_IPC=kdb; KLONGPY_IPC=legacy accepts only that protocol.

        Hooks, set as global functions: .srv.o(h) when a client connects, .srv.c(h) when
        it disconnects, .srv.e(h;text) on an error that nobody is waiting for, and
        .srv.auth(user;password), which returns 1 to accept a client and 0 to refuse it
        (with no .srv.auth every client is accepted). The hooks run on the Klong loop.

        A message larger than the .system key "ipc_max_message" (bytes, 256 MiB when not
        set, read when a connection is made) closes the connection. Set it before .srv
        and .cli, as in .system,"ipc_max_message",1000000.

        Security: a bare port listens on all interfaces (0.0.0.0), and a client can
        run any Klong code in this interpreter. KlongPy assumes every peer is
        trusted. Credentials are sent in clear text and there is no encryption.
        To limit access, bind to localhost, as in .srv("127.0.0.1:8888"),
        and firewall the port. Never expose it to an untrusted network or the
        internet.

    """
    x = str(x)
    parts = x.split(":")
    bind = parts[0] if len(parts) > 1 else None
    try:
        port = int(parts[0] if len(parts) == 1 else parts[1])
    except ValueError:
        raise KlongTypeError(f"bad address {x!r}: expected port or bind:port") from None
    system = get_system(klong)
    server = system.get('ipc_server')
    if server is None:
        server = system.setdefault('ipc_server', TcpServerHandler())
    if len(parts) == 1 and port == 0:
        return server.shutdown_server()
    return server.create_server(system['ioloop'], system['klongloop'], klong, bind, port,
                                shutdown_event=system.get('closeEvent'),
                                max_message=system.get('ipc_max_message'))


class KGAsyncCall(KGLambda):
    user = False  # __init__ does not call KGLambda.__init__
    def __init__(self, klongloop, fn, cb, klong):
        self.klongloop = klongloop
        self.klong = klong

        # Wrap callbacks - KGFnWrapper now handles dynamic resolution automatically
        self.fn = KGFnWrapper(klong, fn) if isinstance(fn, KGFn) else fn
        self.cb = KGFnWrapper(klong, cb) if isinstance(cb, KGFn) else cb

        arity = fn.get_arity() if issubclass(type(fn), KGLambda) else fn.arity
        self.args = [reserved_fn_symbol_map[x] for x in reserved_fn_args[:arity]]

    def _is_kdb_remote(self):
        fn = self.fn
        nc = fn.nc if isinstance(fn, KGRemoteFnProxy) else fn
        return isinstance(nc, NetworkClient) and nc.protocol == "kdb"

    def _call_remote(self, params):
        """Send the call without blocking; the callback runs on the klong loop when the answer arrives."""
        def done(result, error):
            if error is not None:
                logging.warning(f".async: remote call failed: {error}")
                return
            if self.cb is not None:
                self.cb(result)

        if isinstance(self.fn, KGRemoteFnProxy):
            self.fn.call_async(params, done)
        else:
            self.fn.call_async(NetworkClient.arg_to_msg(params[0]), done)

    async def acall(self, klong, params):
        # Execute the function based on its type
        if issubclass(type(self.fn), KGLambda):
            ctx = {reserved_fn_symbols[i]:params[i] for i in range(min(len(reserved_fn_args),len(params)))}
            r = self.fn(klong, ctx)
        elif callable(self.fn):
            r = self.fn(*params)
        else:
            # Shouldn't reach here, but handle it
            r = klong.apply(self.fn.a, [*params], self.fn.arity)

        # Invoke callback - KGFnWrapper handles dynamic resolution automatically
        if self.cb is not None:
            self.cb(r)

    def __call__(self, klong, ctx):
        params = [ctx[x] for x in self.args]
        if self._is_kdb_remote():
            self._call_remote(params)
            return 1
        # thread safe: the call may be made from a thread other than the loop's
        cf = asyncio.run_coroutine_threadsafe(self.acall(klong, params), self.klongloop)
        cf.add_done_callback(lambda f: f.cancelled() or f.exception() is None
                             or logging.warning(f".async: call failed: {f.exception()}"))
        return 1

    def __str__(self):
        return f"async:{super().__str__()}"


def eval_sys_fn_create_async_wrapper(klong, x, y):
    """

        .async(x,y)                             [Async-function-wrapper]

        Returns an async functional wrapper for the function "x" and calls "y"
        when completed. The wrapper has the same arity as the wrapped function.
        Both must be functions, else a type error is raised.

        Calling the wrapper returns 1 at once. A local function runs as a task on the
        Klong loop and "y" is called with its result. When "x" is a connection (from
        .cli) or a remote function proxy, the request is sent without blocking
        and "y" runs on the Klong loop when the answer arrives; an error answer or a
        lost connection is logged and "y" is not called. With a connection the
        wrapper takes the one request argument, as the connection does.

    """
    if not issubclass(type(x), (KGFn, KGLambda)):
        raise KlongTypeError("x must be a function")
    if not issubclass(type(y), (KGFn, KGLambda)):
        raise KlongTypeError("y must be a function")
    system = get_system(klong)
    klongloop = system['klongloop']
    # KGAsyncCall will wrap the callbacks automatically
    return KGAsyncCall(klongloop, x, y, klong)


def create_system_functions_ipc():
    def _get_name(s):
        i = s.index(".")
        return s[i : i + s[i:].index("(")]

    registry = {}

    m = sys.modules[__name__]
    for x in filter(lambda n: n.startswith("eval_sys_"), dir(m)):
        fn = getattr(m, x)
        registry[_get_name(fn.__doc__)] = fn

    return registry


def create_system_var_ipc():
    # populate the .srv.* handlers with undefined values
    registry = {
        ".srv.o": KLONG_UNDEFINED,
        ".srv.c": KLONG_UNDEFINED,
        ".srv.e": KLONG_UNDEFINED,
        ".srv.auth": KLONG_UNDEFINED,
    }
    return registry
