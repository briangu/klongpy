import asyncio
import logging
import pickle
import struct
import sys
import threading
import uuid
from asyncio import StreamReader, StreamWriter
from asyncio.exceptions import IncompleteReadError

import numpy as np

from klongpy.core import (KGCall, KGFn, KGFnWrapper, KGLambda, KGSym,
                          KlongException, get_fn_arity_str, is_list,
                          reserved_fn_args, reserved_fn_symbol_map)


class KlongIPCException(Exception):
    pass

class KlongIPCConnectionClosedException(KlongIPCException):
    pass

class KlongIPCConnectionFailureException(KlongIPCException):
    pass

class KlongIPCCreateConnectionException(KlongIPCException):
    pass


class KGRemoteCloseConnection():
    pass

class KGRemoteCloseConnectionException(KlongException):
    pass


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


async def stream_recv_msg(reader: StreamReader):
    raw_msg_id = await reader.readexactly(16)
    raw_msglen = await reader.readexactly(4)
    msglen = decode_message_len(raw_msglen)
    data = await reader.readexactly(msglen)
    return decode_message(raw_msg_id, data)


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
    try:
        handle_sym = KGSym('.cli.h')
        klong._context[handle_sym] = nc
        if isinstance(command, KGRemoteFnCall):
            r = klong[command.sym]
            if callable(r):
                response = r(*command.params)
            else:
                raise KlongException(f"not callable: {command.sym}")
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
        future_loop.call_soon_threadsafe(result_future.set_result, response)
    except KeyError as e:
        future_loop.call_soon_threadsafe(result_future.set_exception, KlongException(f"symbol not found: {e}"))
    except Exception as e:
        import traceback
        traceback.print_exception(type(e), e, e.__traceback__)
        future_loop.call_soon_threadsafe(result_future.set_exception, KlongException("internal error"))
        logging.error(f"TcpClientHandler::handle_client: Klong error {e}")
    finally:
        del klong._context[handle_sym]


async def run_command_on_klongloop(klongloop, klong, command, nc):
    result_future = asyncio.Future()
    future_loop = asyncio.get_event_loop()
    assert future_loop != klongloop
    coroutine = execute_server_command(future_loop, result_future, klong, command, nc)
    klongloop.call_soon_threadsafe(asyncio.create_task, coroutine)
    result = await result_future
    return result


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


class NetworkClient(KGLambda):
    """

    This network client is used to either connect to a remote server or to handle a connection by a remote client.
    The network connection may be used in KlongPy as a remote dictionary or a remote function.

    If the remote client connects to the server, the server will create a NetworkClient so that it can be used
    in KlongPy as a remote dictionary or a remote function.

    Similarly, if a KlongPy client connects to a remote server, the client will create a NetworkClient so that it
    can be used in KlongPy as a remote dictionary or a remote function.

    """
    def __init__(self, ioloop, klongloop, klong, conn_provider, shutdown_event=None, on_connect=None, on_close=None, on_error=None):
        self.ioloop = ioloop
        self.klongloop = klongloop
        self.klong = klong
        self.shutdown_event = shutdown_event
        self.conn_provider = conn_provider
        self.pending_responses = {}
        self.running = False
        self.run_task = None
        self.on_connect = on_connect
        self.on_close = on_close
        self.on_error = on_error
        self.reader: StreamReader = None
        self.writer: StreamWriter = None
        self._run_exit_event = threading.Event()

        if shutdown_event is not None:
            self.shutdown_event.subscribe(self.close)

    def _cleanup_pending_responses(self, close_exception):
        """

        Cleanup the pending responses and set the exception on the futures.

        From the KlongPy perspective, any outstanding remote calls will fail with the close_exception.

        """
        for future in self.pending_responses.values():
            future.set_exception(close_exception)
        self.pending_responses.clear()

    def run_client(self):
        """

        Start the network client as initiatiated by the Klong interpreter.

        The network client will connect to the remote server and handle server push-requests by
        running messages in the Klong interpreter.

        When the .clic function is called, the network client is stopped and the connection is closed.

        """
        self.running = True
        connect_event = threading.Event()
        async def _on_connect(client, **kwargs):
            if client.on_connect is not None:
                await client.on_connect(self)
            connect_event.set()
        async def _on_error(client, e):
            if client.on_error is not None:
                await client.on_error(self, e)
            connect_event.set()

        self.ioloop.call_soon_threadsafe(asyncio.create_task, self._run(_on_connect, self.on_close, _on_error))
        connect_event.wait()
        return self

    def run_server(self):
        """

        Start the network client for a client connected to the server.

        The network client will handle server requests and shutdown when the connection is closed.

        """
        self.running = True
        return self._run(self.on_connect, self.on_close, self.on_error)

    async def _run(self, on_connect, on_close, on_error):
        """

        Get the connection and start listening for messages.

        If a connection drops or an error occurs, the pending responses are
        cleared and exceptions are returned to the KlongPy callers.

        In order to notify KlongPy applications of network activity:

        When the connection is opened, the on_connect callback is called.
        If a connection is closed, the on_close callback is called.
        If an error occurs, the on_error callback is called.

        The handlers call back into the KlongPy runtime and notify the application.

        :param on_connect: called when a connection is established
        :param on_close: called when a connection is closed
        :param on_error: called when a connection error occurs

        """
        while self.running:
            close_exception = None
            try:
                self.reader, self.writer = await self.conn_provider.connect()
                if on_connect is not None:
                    try:
                        await on_connect(self)
                    except Exception as e:
                        logging.warning(f"error while running on_connect handler: {e}")
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
                self.writer = None
                self.reader = None
                self._cleanup_pending_responses(close_exception)
                if on_close is not None:
                    try:
                        await on_close(self)
                    except Exception as e:
                        logging.warning(f"error while running on_close handler: {e}")
        logging.info(f"Stopping client: {str(self.conn_provider)}")
        self._run_exit_event.set()

    async def _listen(self):
        """

        Listen for messages from the remote server and dispatch them.

        If there is a pending response for a message, the response is returned via the future.
        If the message is the KGRemoteCloseConnection, then the connection is closed.
        Otherwise, the message is executed on the klong loop and the result is sent back to the server.

        """
        try:
            msg_id, msg = await stream_recv_msg(self.reader)
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
            # if self.running:
            logging.info(f"Connection error {e}")
            raise KlongIPCConnectionFailureException(f"connection lost: {str(self.conn_provider)}")
        except KGRemoteCloseConnectionException as e:
            raise e
        except Exception as e:
            logging.warning(f"unexpected error: {type(e)} {e}")
            raise e

    def call(self, msg):
        """

        Send a message to the remote server and wait for the response.

        """
        if not self.is_open():
            raise KlongException("connection not established")

        msg_id = uuid.uuid4()
        future = self.ioloop.create_future()
        self.pending_responses[msg_id] = future

        async def send_message_and_get_result():
            await stream_send_msg(self.writer, msg_id, msg)
            return await future

        return asyncio.run_coroutine_threadsafe(send_message_and_get_result(), self.ioloop).result()

    def __call__(self, _, ctx):
        """

        Evaluate a remote function call.

        """
        x = ctx[reserved_fn_symbol_map[reserved_fn_args[0]]]
        try:
            msg = KGRemoteFnCall(x[0], x[1:]) if is_list(x) and len(x) > 0 and isinstance(x[0],KGSym) else x
            response = self.call(msg)
            if isinstance(x,KGSym) and isinstance(response, KGRemoteFnRef):
                response = KGRemoteFnProxy(self.nc, x, response.arity)
            return response
        except Exception as e:
            import traceback
            traceback.print_exception(type(e), e, e.__traceback__)
            raise e

    def _stop(self):
        """

        Stop the network client.
        First send the KGRemoteCloseConnection message to the server to tell it to close the connection.

        """
        self.running = False
        self._run_exit_event.wait()
        self._run_exit_event.clear()

    def cleanup(self):
        """

        Cleanup the network client and the underlying connection.

        """
        if not self.running:
            return

        if self.shutdown_event is not None:
            self.shutdown_event.unsubscribe(self.close)

        # run close in the appropriate context task to avoid deadlock
        try:
            loop = asyncio.get_running_loop()
            if loop == self.ioloop:
                self.ioloop.call_soon(asyncio.create_task, self.conn_provider.close())
                self._stop()
            else:
                raise RuntimeError()
        except RuntimeError:
            self._stop()
            asyncio.run_coroutine_threadsafe(self.conn_provider.close(), self.ioloop).result()

    def close(self):
        """

        Close the network client and the underlying connection.

        Sends a message to the server to tell it to gracefully close the connection.

        """
        if not self.running:
            return
        self.call(KGRemoteCloseConnection())
        self.cleanup()

    def is_open(self):
        return self.conn_provider.is_open()

    def get_arity(self):
        return 1

    def __str__(self):
        return f"{str(self.conn_provider)}:fn"

    @staticmethod
    def create_from_conn_provider(ioloop, klongloop, klong, conn_provider, shutdown_event=None, on_connect=None, on_close=None, on_error=None):
        """

        Create a network client to connect to a remote server.

        :param ioloop: the asyncio ioloop
        :param klongloop: the klong loop
        :param klong: the klong interpreter
        :param host: the host to connect to
        :param port: the port to connect to
        :return: a network client

        """
        return NetworkClient(ioloop, klongloop, klong, conn_provider, shutdown_event=shutdown_event, on_connect=on_connect, on_close=on_close, on_error=on_error)

    @staticmethod
    def create_from_host_port(ioloop, klongloop, klong, host, port, shutdown_event=None, on_connect=None, on_close=None, on_error=None):
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
        return NetworkClient.create_from_conn_provider(ioloop, klongloop, klong, conn_provider, shutdown_event=shutdown_event, on_connect=on_connect, on_close=on_close, on_error=on_error)

    @staticmethod
    def create_from_addr(ioloop, klongloop, klong, shutdown_event, addr, on_connect=None, on_close=None, on_error=None):
        """

        Create a network client to connect to a remote server.

        :param ioloop: the asyncio ioloop
        :param klongloop: the klong loop
        :param klong: the klong interpreter
        :param addr: the address to connect to.  If the address is an integer, it is interpreted as a port in "localhost:<port>".

        :return: a network client

        """
        addr = str(addr)
        parts = addr.split(":")
        host = parts[0] if len(parts) > 1 else "localhost"
        port = int(parts[0] if len(parts) == 1 else parts[1])
        return NetworkClient.create_from_host_port(ioloop, klongloop, klong, host, port, shutdown_event=shutdown_event, on_connect=on_connect, on_close=on_close, on_error=on_error)


class KGRemoteFnRef:
    def __init__(self, arity):
        self.arity = arity

    def __str__(self):
        return get_fn_arity_str(self.arity)


class KGRemoteFnCall:
    def __init__(self, sym: KGSym, params):
        self.sym = sym
        self.params = params


class KGRemoteDictSetCall:
    def __init__(self, key, value):
        self.key = key
        self.value = value


class KGRemoteDictGetCall:
    def __init__(self, key):
        self.key = key


class KGRemoteFnProxy(KGLambda):

    def __init__(self, nc: NetworkClient, sym: KGSym, arity):
        self.nc = nc
        self.sym = sym
        self.args = reserved_fn_args[:arity]

    def __call__(self, _, ctx):
        params = [ctx[reserved_fn_symbol_map[x]] for x in reserved_fn_args[:len(self.args)]]
        return self.nc.call(KGRemoteFnCall(self.sym, params))

    def __str__(self):
        return f"{self.nc.__str__()}:{self.sym}{super().__str__()}"


class TcpServerConnectionHandler:
    def __init__(self, ioloop, klongloop, klong):
        self.ioloop = ioloop
        self.klong = klong
        self.klongloop = klongloop

    async def _on_connect(self, nc):
        logging.info(f"New connection from {str(nc.conn_provider)}")
        fn = self.klong['.srv.o']
        if callable(fn):
            try:
                fn(nc)
            except Exception as e:
                logging.warning(f"Server: error while running on_connect handler: {e}")

    async def _on_close(self, nc):
        logging.info(f"Connection closed from {str(nc.conn_provider)}")
        fn = self.klong['.srv.c']
        if callable(fn):
            try:
                fn(nc)
            except Exception as e:
                logging.warning(f"Server: error while running on_close handler: {e}")

    async def _on_error(self, nc, e):
        logging.info(f"Connection error from {str(nc.conn_provider)}")
        fn = self.klong['.srv.e']
        if callable(fn):
            try:
                fn(nc, e)
            except Exception as e:
                logging.warning(f"Server: error while running on_error handler: {e}")

    async def handle_client(self, reader: StreamReader, writer: StreamWriter):
        """

        Handle a client connection.  Messages are read from the client and executed on the klong loop.

        """
        results = writer.get_extra_info('peername')
        if results is None:
            logging.warning(f"Connection closed before peername could be retrieved")
            return
        host, port = results[0], results[1]
        if host == "::1":
            host = "localhost"
        conn_provider = ReaderWriterConnectionProvider(reader, writer, host, port)
        nc = NetworkClient.create_from_conn_provider(self.ioloop, self.klongloop, self.klong, conn_provider, on_connect=self._on_connect, on_close=self._on_close, on_error=self._on_error)
        try:
            await nc.run_server()
        finally:
            nc.cleanup()


class TcpServerHandler:
    def __init__(self):
        self.connection_handler = None
        self.task = None
        self.server = None
        self.connections = []

    def create_server(self, ioloop, klongloop, klong, bind, port):
        if self.task is not None:
            return 0
        self.connection_handler = TcpServerConnectionHandler(ioloop, klongloop, klong)
        self.task = ioloop.call_soon_threadsafe(asyncio.create_task, self.run_server(bind, port))
        return 1

    def shutdown_server(self):
        if self.task is None:
            return 0
        for writer in self.connections:
            if not writer.is_closing():
                writer.close()
        self.connections.clear()

        self.server.close()
        self.server = None
        self.task.cancel()
        self.task = None
        self.connection_handler = None
        return 1

    async def handle_client(self, reader, writer):
        self.connections.append(writer)

        try:
            await self.connection_handler.handle_client(reader, writer)
        finally:
            writer.close()
            if writer in self.connections:
                self.connections.remove(writer)

    async def run_server(self, bind, port):
        self.server = await asyncio.start_server(self.handle_client, bind, port, reuse_address=True)

        addr = self.server.sockets[0].getsockname()
        logging.info(f'Serving on {addr}')

        async with self.server:
            await self.server.serve_forever()

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
        try:
            response = self.nc.call(KGRemoteDictGetCall(x))
            if isinstance(x,KGSym) and isinstance(response, KGRemoteFnRef):
                response = KGRemoteFnProxy(self.nc, x, response.arity)
            return response
        except Exception as e:
            import traceback
            traceback.print_exception(type(e), e, e.__traceback__)
            raise e

    def set(self, x, y):
        try:
            self.nc.call(KGRemoteDictSetCall(x, y))
            return self
        except Exception as e:
            import traceback
            traceback.print_exception(type(e), e, e.__traceback__)
            raise e

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
        if "x" is a string, then it is interpreted as a host address "<host>:<port>"

        If "x" is a remote dictionary, the underlying network connection
        is shared and a remote function is returned.

        Connection examples:

                   .cli(8888)            --> remote function to localhost:8888
                   .cli("localhost:8888") --> remote function to localhost:8888

                   d::.clid(8888)
                   .cli(d)                --> remote function to same connection as d

        Evaluation examples:

                   f::.cli(8888)

            A string is passed it is evaluated remotely:

                   f("hello")             --> "hello" is evaluated remotely
                   f("avg::{(+/x)%#x}")   --> "avg" function is defined remotely
                   f("avg(!100)")         --> 49.5 (computed remotely)

            Remote functions may be evaluated by passing an array with the first element
            being the symbol of the remote function to execute.  The remaining elements
            are supplied as parameters:

            Example: call :avg with a locally generated !100 range which is passed to the remote server.

                   f(:avg,,!100)          --> 49.5

            Similary:

                   b::!100
                   f(:avg,,b)             --> 49.5

            When a symbol is applied, the remote value is returned.
            For functions, a remote function proxy is returned.

            Example: retrieve a function proxy to :avg and then call it as if it were a local function.

                   q::f(:avg)
                   q(!100)                --> 49.5

            Example: retrieve a copy of a remote array.

                   f("b::!100")
                   p::f(:b)               --> "p: now holds a copy of the remote array "b"

    """
    x = x.a if isinstance(x,KGCall) else x
    if isinstance(x,NetworkClient):
        return x
    system = klong['.system']
    ioloop = system['ioloop']
    klongloop = system['klongloop']
    shutdown_event = system['closeEvent']
    nc = x.nc if isinstance(x,NetworkClientDictHandle) else NetworkClient.create_from_addr(ioloop, klongloop, klong, shutdown_event, x).run_client()
    return nc


def eval_sys_fn_create_dict_client(klong, x):
    """

        .clid(x)                                [Create-IPC-dict-client]

        Return a dictionary which evaluates set/get operations on a remote KlongPy server.

        If "x" is an integer, then it is interpreted as a port in "localhost:<port>".
        if "x" is a string, then it is interpreted as a host address "<host>:<port>"

        If "x" is a remote function, the underlying network connection
        is shared and a remote function is returned.

        Examples:  .cli(8888)             --> remote function to localhost:8888
                   .cli("localhost:8888") --> remote function to localhost:8888
                   .cli(d)                --> remote function to same connection as d

        Connection examples:

                   .cli(8888)            --> remote function to localhost:8888
                   .cli("localhost:8888") --> remote function to localhost:8888

                   f::.cli(8888)
                   .cli(f)                --> remote function to same connection as f

        Evaluation examples:

                   d::.cli(8888)

            Set a remote key/value pair :foo -> 2 on the remote server.

                   d,[:foo 2]             --> sets :foo to 2
                   d,[:bar "hello"]       --> sets :bar to "hello"
                   d,:fn,{x+1}            --> sets :fn to the monad {x+1)

            Get a remote value:

                   d?:foo                 --> 2
                   d?:bar                 --> hello
                   d?:fn                 --> remote function proxy to :avg

            To use the remote function proxy:
                   q::d?:fn
                   q(2)               --> 3 (remotely executed after passing 2)

    """
    x = x.a if isinstance(x,KGCall) else x
    if isinstance(x,NetworkClientDictHandle):
        return x
    system = klong['.system']
    ioloop = system['ioloop']
    klongloop = system['klongloop']
    shutdown_event = system['closeEvent']
    nc = x if isinstance(x,NetworkClient) else NetworkClient.create_from_addr(ioloop, klongloop, klong, shutdown_event, x).run_client()
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


_ipc_tcp_server = TcpServerHandler()


def eval_sys_fn_create_ipc_server(klong, x):
    """

        .srv(x)                                       [Start-IPC-server]

        Open a server port to accept IPC connections.

        If "x" is an integer, then it is interpreted as a port in "<all>:<port>".
        if "x" is a string, then it is interpreted as a bind address "<bind>:<port>"

        if "x" is 0, then the server is closed and existing client connections are dropped.

    """
    global _ipc_tcp_server
    x = str(x)
    parts = x.split(":")
    bind = parts[0] if len(parts) > 1 else None
    port = int(parts[0] if len(parts) == 1 else parts[1])
    if len(parts) == 1 and port == 0:
        return _ipc_tcp_server.shutdown_server()
    system = klong['.system']
    ioloop = system['ioloop']
    klongloop = system['klongloop']
    return _ipc_tcp_server.create_server(ioloop, klongloop, klong, bind, port)


class KGAsyncCall(KGLambda):
    def __init__(self, klongloop, fn, cb):
        self.klongloop = klongloop
        self.cb = cb
        self.fn = fn
        self.args = [reserved_fn_symbol_map[x] for x in reserved_fn_args[:fn.arity]]

    async def acall(self, klong, params):
        r = klong.call(KGCall(self.fn.a, [*params], self.fn.arity))
        self.cb(r)

    def __call__(self, klong, ctx):
        params = [ctx[x] for x in self.args]
        self.klongloop.create_task(self.acall(klong, params))
        return 1

    def __str__(self):
        return f"async:{super().__str__()}"


def eval_sys_fn_create_async_wrapper(klong, x, y):
    """

        .async(x,y)                             [Async-function-wrapper]

        Returns an async functional wrapper for the function "x" and calls "y"
        when completed. The wrapper has the same arity as the wrapped function.

    """
    if not issubclass(type(x),KGFn):
        raise KlongException("x must be a function")
    if not issubclass(type(y),KGFn):
        raise KlongException("y must be a function")
    system = klong['.system']
    klongloop = system['klongloop']
    return KGAsyncCall(klongloop, x, KGFnWrapper(klong, y))


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
    # TODO: use real undefined value instead of np.inf
    registry = {
        ".srv.o": np.inf,
        ".srv.c": np.inf,
        ".srv.e": np.inf,
    }
    return registry


