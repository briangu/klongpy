import asyncio
import logging
import pickle
import socket
import struct
import sys
import threading
from asyncio import StreamReader, StreamWriter
from typing import Optional

from klongpy.core import (KGCall, KGFn, KGLambda, KGFnWrapper, KGSym, get_fn_arity_str,
                          is_list, reserved_fn_args, reserved_fn_symbol_map)

_main_loop = asyncio.get_event_loop()
_main_tid = threading.current_thread().ident


class NetworkClient:
    def __init__(self, host, port, sock):
        self.host = host
        self.port = port
        self.sock = sock

    def call(self, msg):
        if not self.is_open():
            return f"connection closed to IPC server [{self.host}:{self.port}]"
        socket_send_msg(self.sock, msg)
        return socket_recv_msg(self.sock)

    def close(self):
        logging.info("closing network client: {self.host}:{self.port}")
        self.sock.close()
        self.sock = None

    def is_open(self):
        return self.sock is not None
    
    def __str__(self):
        return f"remote[{self.host}:{self.port}]"

    @staticmethod
    def create(host, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        return NetworkClient(host, port, sock)

    @staticmethod
    def create_from_addr(addr):
        addr = str(addr)
        parts = addr.split(":")
        host = parts[0] if len(parts) > 1 else "localhost"
        port = int(parts[0] if len(parts) == 1 else parts[1])
        return NetworkClient.create(host, port)


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


async def stream_send_msg(writer: StreamWriter, msg):
    data = pickle.dumps(msg)
    writer.write(struct.pack('>I', len(data)) + data)
    await writer.drain()


async def stream_recv_all(reader: StreamReader, n: int) -> Optional[bytes]:
    data = bytearray()
    remaining = n

    while remaining > 0:
        packet = await reader.read(remaining)
        if not packet:
            return None
        data.extend(packet)
        remaining -= len(packet)

    return bytes(data)


async def stream_recv_msg(reader: StreamReader):
    raw_msglen = await stream_recv_all(reader, 4)
    if not raw_msglen:
        logging.error("stream_recv_msg: remote server error => received empty message")
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    data = await stream_recv_all(reader, msglen)
    return pickle.loads(data)


def socket_send_msg(conn: socket.socket, msg):
    data = pickle.dumps(msg)
    conn.sendall(struct.pack('>I', len(data)) + data)


def socket_recv_all(conn: socket.socket, n: int) -> Optional[bytes]:
    data = bytearray()
    remaining = n

    while remaining > 0:
        packet = conn.recv(remaining)
        if not packet:
            return None
        data.extend(packet)
        remaining -= len(packet)

    return bytes(data)


def socket_recv_msg(conn: socket.socket):
    raw_msglen = socket_recv_all(conn, 4)
    if not raw_msglen:
        logging.error("socket_recv_msg: remote server error => received empty message")
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    data = socket_recv_all(conn, msglen)
    return pickle.loads(data)


class TcpClientHandler:
    def __init__(self, klong):
        self.klong = klong

    async def handle_client(self, reader: StreamReader, writer: StreamWriter):
        while True:
            command = await stream_recv_msg(reader)
            if command:
                try:
                    assert threading.current_thread().ident == _main_tid
                    if isinstance(command, KGRemoteFnCall):
                        r = self.klong[command.sym]
                        response = r(*command.params) if callable(r) else f"not callable: {command.sym}"
                    elif isinstance(command, KGRemoteDictSetCall):
                        self.klong[command.key] = command.value
                        response = None
                    elif isinstance(command, KGRemoteDictGetCall):
                        response = self.klong[command.key]
                        if isinstance(response, KGFnWrapper):
                            response = response.fn
                    else:
                        response = self.klong(command)
                    if isinstance(response, KGFn):
                        response = KGRemoteFnRef(response.arity)
                except KeyError as e:
                    response = f"symbol not found: {e}"
                except Exception as e:
                    response = "internal error"
                    logging.error(f"TcpClientHandler::handle_client: Klong error {e}")
                    import traceback
                    traceback.print_exception(type(e), e, e.__traceback__)
                await stream_send_msg(writer, response)
            else:
                break

        writer.close()
        await writer.wait_closed()


class TcpServerHandler:
    def __init__(self):
        self.client_handler = None
        self.task = None
        self.server = None
        self.connections = []

    def create_server(self, loop, klong, bind, port):
        if self.task is not None:
            return 0
        self.client_handler = TcpClientHandler(klong)
        self.task = loop.create_task(self.tcp_producer(bind, port))
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
        self.client_handler = None
        return 1

    async def handle_client(self, reader, writer):
        self.connections.append(writer)

        try:
            await self.client_handler.handle_client(reader, writer)
        finally:
            if not writer.is_closing():
                writer.close()
            await writer.wait_closed()
            if writer in self.connections:
                self.connections.remove(writer)

    async def tcp_producer(self, bind, port):
        self.server = await asyncio.start_server(self.handle_client, bind, port)

        addr = self.server.sockets[0].getsockname()
        logging.info(f'Serving on {addr}')

        async with self.server:
            await self.server.serve_forever()


class NetworkClientHandle(KGLambda):
    def __init__(self, nc: NetworkClient):
        self.nc = nc

    def __call__(self, _, ctx):
        x = ctx[reserved_fn_symbol_map[reserved_fn_args[0]]]
        try:
            msg = KGRemoteFnCall(x[0], x[1:]) if is_list(x) and len(x) > 0 and isinstance(x[0],KGSym) else x
            response = self.nc.call(msg)
            if isinstance(x,KGSym) and isinstance(response, KGRemoteFnRef):
                response = KGRemoteFnProxy(self.nc, x, response.arity)
            return response
        except Exception as e:
            import traceback
            traceback.print_exception(type(e), e, e.__traceback__)
            raise e

    def close(self):
        return self.nc.close()

    def is_open(self):
        return self.nc.is_open()
    
    def get_arity(self):
        return 1

    def __str__(self):
        return f"{self.nc.__str__()}:fn"


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
        return f"{self.nc.__str__()}:dict"


def eval_sys_fn_create_client(x):
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
    if isinstance(x,NetworkClientHandle):
        return x
    nc = x.nc if isinstance(x,NetworkClientDictHandle) else NetworkClient.create_from_addr(x)
    return NetworkClientHandle(nc)


def eval_sys_fn_create_dict_client(x):
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
    nc = x.nc if isinstance(x,NetworkClientHandle) else NetworkClient.create_from_addr(x)
    return NetworkClientDictHandle(nc)


def eval_sys_fn_shutdown_client(x):
    """

        .clic(x)                                      [Close-IPC-client]

        Close a remote dictionary or function opened by .cli or .clid.

        Returns 1 if closed, 0 if already closed.

        When a connection is closed, all remote proxies / functions tied to this connection
        will also close and will fail if called.

    """
    if isinstance(x, KGCall) and issubclass(type(x.a), KGLambda):
        x = x.a
        if isinstance(x, (NetworkClientHandle, NetworkClientDictHandle)) and x.is_open():
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
    global _main_loop
    global _ipc_tcp_server
    x = str(x)
    parts = x.split(":")
    bind = parts[0] if len(parts) > 1 else None
    port = int(parts[0] if len(parts) == 1 else parts[1])
    if len(parts) == 1 and port == 0:
        return _ipc_tcp_server.shutdown_server()
    return _ipc_tcp_server.create_server(_main_loop, klong, bind, port)


class KGAsyncCall(KGLambda):
    def __init__(self, loop, fn, cb):
        self.loop = loop
        self.cb = cb
        self.fn = fn
        self.args = [reserved_fn_symbol_map[x] for x in reserved_fn_args[:fn.arity]]
   
    async def acall(self, klong, params):
        r = klong.call(KGCall(self.fn.a, [*params], self.fn.arity))
        self.cb(r)

    def __call__(self, klong, ctx):
        params = [ctx[x] for x in self.args]
        self.loop.create_task(self.acall(klong, params))
        return 1

    def __str__(self):
        return f"async:{super().__str__()}"


def eval_sys_fn_create_async_wrapper(klong, x, y):
    """

        .async(x,y)                             [Async-function-wrapper]

        Returns an async callable wrapper for the function "x" and calls "y"
        when completed.

    """
    global _main_loop
    if not issubclass(type(x),KGFn):
        return "x must be a function"
    if not issubclass(type(y),KGFn):
        return "y must be a function"
    return KGAsyncCall(_main_loop, x, KGFnWrapper(klong, y))


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
