import asyncio
import json
import logging
import sys
import threading

import numpy as np
import websockets

from klongpy.core import (KGCall, KGLambda, KGSym, KlongException,
                          reserved_fn_args, reserved_fn_symbol_map)


class KlongWSException(Exception):
    pass

class KlongWSConnectionClosedException(KlongWSException):
    pass

class KlongWSConnectionFailureException(KlongWSException):
    pass

class KlongWSCreateConnectionException(KlongWSException):
    pass


class KGRemoteCloseConnection():
    pass

class KGRemoteCloseConnectionException(KlongException):
    pass


class NumpyEncoder(json.JSONEncoder):
    """
    We need to translate NumPy objects into lists as needed:
    https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def encode_message(msg):
    return json.dumps(msg, cls=NumpyEncoder)


def decode_message(data):
    return json.loads(data)


async def execute_server_command(future_loop, result_future, klong, sym, command, nc):
    """
    """
    try:
        klong._context.push({KGSym('.ws.h'): nc})
        r = klong[sym]
        if callable(r):
            response = r(nc, command)
        else:
            raise KlongException(f"not callable: {command.sym}")
        future_loop.call_soon_threadsafe(result_future.set_result, response)
    except KeyError as e:
        future_loop.call_soon_threadsafe(result_future.set_exception, KlongException(f"symbol not found: {e}"))
    except Exception as e:
        import traceback
        traceback.print_exception(type(e), e, e.__traceback__)
        future_loop.call_soon_threadsafe(result_future.set_exception, KlongException("internal error"))
        logging.error(f"execute_server_command Klong error {e}")
        # import traceback
        # traceback.print_exception(type(e), e, e.__traceback__)
    finally:
        klong._context.pop()


async def run_command_on_klongloop(klongloop, klong, sym, command, nc):
    result_future = asyncio.Future()
    future_loop = asyncio.get_event_loop()
    assert future_loop != klongloop
    coroutine = execute_server_command(future_loop, result_future, klong, sym, command, nc)
    klongloop.call_soon_threadsafe(asyncio.create_task, coroutine)
    result = await result_future
    return result


class ConnectionProvider:
    async def connect(self):
        raise KlongWSCreateConnectionException()

    async def close(self):
        raise NotImplementedError()


class ClientConnectionProvider(ConnectionProvider):
    """
    This connection provider is used to create a NetworkClient from a host/port pair.
    """
    def __init__(self, uri):
        self.uri = uri
        self.websocket = None

    async def connect(self):
        """
        Attempt to connect to the remote server.
        """
        try:
            self.websocket = await websockets.connect(uri=self.uri)
            return self.websocket
        except Exception as e:
            logging.warning(f"Unexpected connection error {e} to {self.uri}")
            raise KlongWSCreateConnectionException()

    async def close(self):
        """
        Close the connection.  This is called when the client is stopped.
        """
        if self.websocket is not None:
            await self.websocket.close()
        self.websocket = None

    def is_open(self):
        return self.websocket is not None and not self.websocket.closed

    def __str__(self):
        return f"remote[{self.uri}]"


class ExistingConnectionProvider(ConnectionProvider):
    """
    This connection provider is used to create a NetworkClient from an existing websocket.
    """
    def __init__(self, websocket, uri):
        self.websocket = websocket
        self.uri = uri

    async def connect(self):
        if self.websocket is None:
            raise KlongWSCreateConnectionException()
        return self.websocket

    async def close(self):
        if self.websocket is not None:
            await self.websocket.close()
        self.websocket = None

    def __str__(self):
        return f"remote[{self.uri}]"


class NetworkClient(KGLambda):
    """
    """
    def __init__(self, ioloop, klongloop, klong, conn_provider, shutdown_event=None, on_connect=None, on_close=None, on_error=None, on_message=None):
        self.ioloop = ioloop
        self.klongloop = klongloop
        self.klong = klong
        self.shutdown_event = shutdown_event
        self.conn_provider = conn_provider
        self.running = False
        self.run_task = None
        self.on_connect = on_connect
        self.on_close = on_close
        self.on_error = on_error
        self.on_message = on_message
        self.websocket = None
        self._run_exit_event = threading.Event()

        if shutdown_event is not None:
            self.shutdown_event.subscribe(self.close)

    def run_client(self):
        """
        """
        self.running = True
        connect_event = threading.Event()
        async def _on_connect(client, **kwargs):
            if client.on_connect is not None:
                await client.on_connect(client)
            connect_event.set()
        async def _on_error(client, e):
            if client.on_error is not None:
                await client.on_error(client, e)
            connect_event.set()

        self.run_task = self.ioloop.call_soon_threadsafe(asyncio.create_task, self._run(_on_connect, self.on_close, _on_error, self.on_message))
        connect_event.wait()
        return self

    def run_server(self):
        """
        """
        self.running = True
        return self._run(self.on_connect, self.on_close, self.on_error, self.on_message)

    async def _run(self, on_connect, on_close, on_error, on_message):
        while self.running:
            try:
                self.websocket = await self.conn_provider.connect()
                if on_connect is not None:
                    try:
                        await on_connect(self)
                    except Exception as e:
                        logging.warning(f"error while running on_connect handler: {e}")
                while self.running:
                    await self._listen(on_message)
            except (KlongWSConnectionFailureException, KlongWSCreateConnectionException) as e:
                if on_error is not None:
                    try:
                        await on_error(self, e)
                    except Exception as e:
                        logging.warning(f"error while running on_error handler: {e}")
                break
            except KGRemoteCloseConnectionException:
                logging.info(f"Remote client closing connection: {str(self.conn_provider)}")
                self.running = False
                break
            except Exception as e:
                logging.warning(f"Unexepected error {e}.")
                if on_error is not None:
                    try:
                        await on_error(self, e)
                    except Exception as e:
                        logging.warning(f"error while running on_error handler: {e}")
                break
            finally:
                self.websocket = None
                if on_close is not None:
                    try:
                        await on_close(self)
                    except Exception as e:
                        logging.warning(f"error while running on_close handler: {e}")
        logging.info(f"Stopping client: {str(self.conn_provider)}")
        self._run_exit_event.set()

    async def _listen(self, on_message):
        try:
            msg = await self.websocket.recv()
            msg = decode_message(msg)
            if on_message is not None:
                try:
                    await on_message(self, msg)
                except Exception as e:
                    logging.warning(f"error while running on_message handler: {e}")
            await run_command_on_klongloop(self.klongloop, self.klong, ".ws.m", msg, self)
            # if response is not None and response != np.inf:
            #     await self.websocket.send(encode_message(response))
        except websockets.exceptions.ConnectionClosed:
            logging.info("Connection error")
            raise KlongWSConnectionFailureException()

    def call(self, msg):
        """
        """
        if not self.is_open():
            raise KlongException("connection not established")

        try:
            msg = encode_message(msg)
        except Exception as e:
            print("error encoding message", e, msg)

#        return asyncio.run_coroutine_threadsafe(self.websocket.send(msg), self.ioloop).result()
        self.ioloop.call_soon_threadsafe(asyncio.create_task, self.websocket.send(msg))

    def __call__(self, _, ctx):
        """
        """
        x = ctx[reserved_fn_symbol_map[reserved_fn_args[0]]]
        try:
            return self.call(x)
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
        self.run_task = None

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
            else:
                raise RuntimeError()
            self._stop()
        except RuntimeError:
            asyncio.run_coroutine_threadsafe(self.conn_provider.close(), self.ioloop).result()
            self._stop()


    def close(self):
        """

        Close the network client and the underlying connection.

        Sends a message to the server to tell it to gracefully close the connection.

        """
        if not self.running:
            return
        self.cleanup()

    def is_open(self):
        return self.conn_provider.is_open()

    def get_arity(self):
        return 1

    def __str__(self):
        return f"{str(self.conn_provider)}:fn"

    @staticmethod
    def create_from_conn_provider(ioloop, klongloop, klong, conn_provider, shutdown_event=None, on_connect=None, on_close=None, on_error=None, on_message=None):
        """

        Create a network client to connect to a remote server.

        :param ioloop: the asyncio ioloop
        :param klongloop: the klong loop
        :param klong: the klong interpreter
        :param host: the host to connect to
        :param port: the port to connect to
        :return: a network client

        """
        return NetworkClient(ioloop, klongloop, klong, conn_provider, shutdown_event=shutdown_event, on_connect=on_connect, on_close=on_close, on_error=on_error, on_message=on_message)

    @staticmethod
    def create_from_uri(ioloop, klongloop, klong, uri, shutdown_event=None, on_connect=None, on_close=None, on_error=None, on_message=None):
        """

        Create a network client to connect to a remote server.

        :param ioloop: the asyncio ioloop
        :param klongloop: the klong loop
        :param klong: the klong interpreter
        :param addr: the address to connect to.  If the address is an integer, it is interpreted as a port in "localhost:<port>".

        :return: a network client

        """
        conn_provider = ClientConnectionProvider(uri)
        return NetworkClient.create_from_conn_provider(ioloop, klongloop, klong, conn_provider, shutdown_event=shutdown_event, on_connect=on_connect, on_close=on_close, on_error=on_error, on_message=on_message)


# class WebsocketServerConnectionHandler:
#     def __init__(self, ioloop, klongloop, klong):
#         self.ioloop = ioloop
#         self.klong = klong
#         self.klongloop = klongloop

#     async def _on_connect(self, nc):
#         logging.info(f"New connection from {str(nc.conn_provider)}")
#         fn = self.klong['.srv.o']
#         if callable(fn):
#             try:
#                 fn(nc)
#             except Exception as e:
#                 logging.warning(f"Server: error while running on_connect handler: {e}")

#     async def _on_close(self, nc):
#         logging.info(f"Connection closed from {str(nc.conn_provider)}")
#         fn = self.klong['.srv.c']
#         if callable(fn):
#             try:
#                 fn(nc)
#             except Exception as e:
#                 logging.warning(f"Server: error while running on_close handler: {e}")

#     async def _on_error(self, nc, e):
#         logging.info(f"Connection error from {str(nc.conn_provider)}")
#         fn = self.klong['.srv.e']
#         if callable(fn):
#             try:
#                 fn(nc, e)
#             except Exception as e:
#                 logging.warning(f"Server: error while running on_error handler: {e}")

#     async def handle_client(self, websocket):
#         """

#         Handle a client connection.  Messages are read from the client and executed on the klong loop.

#         """
#         # host, port, _, _ = writer.get_extra_info('peername')
#         # if host == "::1":
#         #     host = "localhost"
#         conn_provider = ExistingConnectionProvider(websocket, "")
#         nc = NetworkClient.create_from_conn_provider(self.ioloop, self.klongloop, self.klong, shutdown_event, conn_provider, on_connect=self._on_connect, on_close=self._on_close, on_error=self._on_error)
#         try:
#             await nc.run_server()
#         finally:
#             nc.cleanup()


# class WebsocketServerHandler:
#     def __init__(self):
#         self.connection_handler = None
#         self.task = None
#         self.server = None
#         self.connections = []

#     def create_server(self, ioloop, klongloop, klong, bind, port):
#         if self.task is not None:
#             return 0
#         self.connection_handler = WebsocketServerConnectionHandler(ioloop, klongloop, klong)
#         self.task = ioloop.call_soon_threadsafe(asyncio.create_task, self.run_server(bind, port))
#         return 1

#     def shutdown_server(self):
#         if self.task is None:
#             return 0
#         for writer in self.connections:
#             if not writer.is_closing():
#                 writer.close()
#         self.connections.clear()

#         self.server.close()
#         self.server = None
#         self.task.cancel()
#         self.task = None
#         self.connection_handler = None
#         return 1

#     async def handle_client(self, reader, writer):
#         self.connections.append(writer)

#         try:
#             await self.connection_handler.handle_client(reader, writer)
#         finally:
#             writer.close()
#             if writer in self.connections:
#                 self.connections.remove(writer)

#     async def run_server(self, bind, port):
#         self.server = await asyncio.start_server(self.handle_client, bind, port, reuse_address=True)

#         addr = self.server.sockets[0].getsockname()
#         logging.info(f'Serving on {addr}')

#         async with self.server:
#             await self.server.serve_forever()


def eval_sys_fn_create_client(klong, x):
    """

        .ws(x)                                 [Create-Websocket-client]

    """
    x = x.a if isinstance(x,KGCall) else x
    if isinstance(x,NetworkClient):
        return x
    system = klong['.system']
    ioloop = system['ioloop']
    klongloop = system['klongloop']
    shutdown_event = system['closeEvent']
    nc = NetworkClient.create_from_uri(ioloop, klongloop, klong, x, shutdown_event=shutdown_event).run_client()
    return nc


def eval_sys_fn_shutdown_client(x):
    """

        .wsc(x)                                 [Close-Websocket-client]

        Close a remote dictionary or function opened by .cli or .clid.

        Returns 1 if closed, 0 if already closed.

        When a connection is closed, all remote proxies / functions tied to this connection
        will also close and will fail if called.

    """
    if isinstance(x, KGCall):
        x = x.a
    if isinstance(x, NetworkClient) and x.is_open():
        x.close()
        return 1
    return 0


# _ws_server = WebsocketServerHandler()


# def eval_sys_fn_create_websocket_server(klong, x):
#     """

#         .wsrv(x)                                       [Start-IPC-server]

#         Open a server port to accept IPC connections.

#         If "x" is an integer, then it is interpreted as a port in "<all>:<port>".
#         if "x" is a string, then it is interpreted as a bind address "<bind>:<port>"

#         if "x" is 0, then the server is closed and existing client connections are dropped.

#     """
#     global _ws_server
#     x = str(x)
#     parts = x.split(":")
#     bind = parts[0] if len(parts) > 1 else None
#     port = int(parts[0] if len(parts) == 1 else parts[1])
#     if len(parts) == 1 and port == 0:
#         return _ws_server.shutdown_server()
#     system = klong['.system']
#     ioloop = system['ioloop']
#     klongloop = system['klongloop']
#     return _ws_server.create_server(ioloop, klongloop, klong, bind, port)


def create_system_functions_websockets():
    def _get_name(s):
        i = s.index(".")
        return s[i : i + s[i:].index("(")]

    registry = {}

    m = sys.modules[__name__]
    for x in filter(lambda n: n.startswith("eval_sys_"), dir(m)):
        fn = getattr(m, x)
        registry[_get_name(fn.__doc__)] = fn

    return registry


def create_system_var_websockets():
    # populate the .srv.* handlers with undefined values
    # TODO: use real undefined value instead of np.inf
    registry = {
        ".srv.o": np.inf,
        ".srv.c": np.inf,
        ".srv.e": np.inf,
    }
    return registry


