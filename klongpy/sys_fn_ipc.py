import asyncio
import logging
import socket
import struct
import sys
import threading
from asyncio import StreamReader, StreamWriter
from typing import Optional

import pickle as dill

from klongpy.core import KGCall, KGFn, KGLambda, KGSym, get_fn_arity_str, reserved_fn_args, reserved_fn_symbol_map

_main_loop = asyncio.get_event_loop()
_main_tid = threading.current_thread().ident


class NetworkClient:
    def __init__(self, host, port, sock):
        self.host = host
        self.port = port
        self.sock = sock

    def call(self, msg):
        socket_send_msg(self.sock, msg)
        return socket_recv_msg(self.sock)

    def close(self):
        self.sock.close()
        self.sock = None

    def is_open(self):
        return self.sock is not None
    
    def __str__(self):
        return f":remote_fn[{self.host}:{self.port}]"


class KGRemoteFnRef:
    def __init__(self, arity):
        self.arity = arity

    def __str__(self):
        return get_fn_arity_str(self.arity)


class KGRemoteFnCall:
    def __init__(self, sym: KGSym, params):
        self.sym = sym
        self.params = params


class KGRemoteFnProxy(KGLambda):

    def __init__(self, nc: NetworkClient, sym: KGSym, arity):
        self.nc = nc
        self.sym = sym
        self.args = reserved_fn_args[:arity]
        self.provide_klong = False

    def __call__(self, _, ctx):
        params = [ctx[reserved_fn_symbol_map[x]] for x in reserved_fn_args[:len(self.args)]]
        return self.nc.call(KGRemoteFnCall(self.sym, params))

    def __str__(self):
        return f"remote[{self.nc.host}:{self.nc.port}]{super().__str__()}"


async def stream_send_msg(writer: StreamWriter, msg):
    data = dill.dumps(msg)
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
    return dill.loads(data)


def socket_send_msg(conn: socket.socket, msg: bytes):
    data = dill.dumps(msg)
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
    return dill.loads(data)


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
                        response = self.klong[command.sym](*command.params)
                    else:
                        response = self.klong(command)
                    if isinstance(response, KGFn):
                        response = KGRemoteFnRef(response.arity)
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

    def create_server(self, loop, klong, bind, port):
        if self.task is not None:
            return 0
        self.client_handler = TcpClientHandler(klong)
        self.task = loop.create_task(self.tcp_producer(bind, port))
        return 1

    def shutdown_server(self):
        if self.task is None:
            return 0
        self.task.cancel()
        self.task = None
        self.client_handler = None
        return 1

    async def tcp_producer(self, bind, port):
        server = await asyncio.start_server(self.client_handler.handle_client, bind, port)

        addr = server.sockets[0].getsockname()
        logging.info(f'Serving on {addr}')

        async with server:
            await server.serve_forever()


class NetworkClientHandle:
    def __init__(self, nc: NetworkClient):
        self.nc = nc

    def __call__(self, x):
        try:
            response = self.nc.call(str(x))
            if isinstance(x,KGSym) and isinstance(response, KGRemoteFnRef):
                response = KGRemoteFnProxy(self.nc, x, response.arity)
            return response
        except Exception as e:
            import traceback
            traceback.print_exception(type(e), e, e.__traceback__)
            raise e

    def close(self):
        if self.nc is not None:
            self.nc.close()
            self.nc = None

    def is_open(self):
        return self.nc is not None
    
    def __str__(self):
        return f":cli[{self.nc.host}:{self.nc.port}]"

    @staticmethod
    def create(host, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        return NetworkClientHandle(NetworkClient(host, port, sock))


def eval_sys_fn_create_client(x):
    """

        .cli(x)                                      [Create IPC client]

    """
    x = str(x)
    parts = x.split(":")
    host = parts[0] if len(parts) > 1 else "localhost"
    port = int(parts[0] if len(parts) == 1 else parts[1])
    return NetworkClientHandle.create(host, port)


def eval_sys_fn_shutdown_client(x):
    """

        .clic(x)                                      [Close IPC client]

    """
    if isinstance(x, KGCall) and issubclass(type(x.a), KGLambda):
        x = x.a.fn
        if isinstance(x, NetworkClientHandle) and x.is_open():
            x.close()
            return 1
    return 0


_ipc_tcp_server = TcpServerHandler()


def eval_sys_create_ipc_server(klong, x):
    """

        .srv(x)                                       [Start IPC server]

    """
    global _main_loop
    global _ipc_tcp_server
    x = str(x)
    parts = x.split(":")
    bind = parts[0] if len(parts) > 1 else None
    port = int(parts[0] if len(parts) == 1 else parts[1])
    if len(parts) == 1 and port == 0:
        return eval_sys_shutdown_ipc_server()
    return _ipc_tcp_server.create_server(_main_loop, klong, bind, port)


def eval_sys_shutdown_ipc_server():
    """

        .srvc()                                        [Stop IPC server]

    """
    global _ipc_tcp_server
    return _ipc_tcp_server.shutdown_server()


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
