import asyncio
import logging
import socket
import struct
import sys
import threading
from asyncio import StreamReader, StreamWriter
from typing import Optional

import dill

from klongpy.core import KGCall, KGLambda

_main_loop = asyncio.get_event_loop()
_main_tid = threading.current_thread().ident


def run_klong(klong, cmd):
    assert threading.current_thread().ident == _main_tid
    return klong(cmd)


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
                    response = run_klong(self.klong, command)
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


class NetworkClientHandle():
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

    def __call__(self, x):
        socket_send_msg(self.sock, x)
        return socket_recv_msg(self.sock)

    def close(self):
        self.sock.close()
        self.sock = None

    def is_open(self):
        return self.sock is not None
    
    def __str__(self):
        return ":ipc"


def eval_sys_fn_create_client(x):
    """

        .cli(x)                                      [Create IPC client]

    """
    x = str(x)
    parts = x.split(":")
    host = parts[0] if len(parts) > 1 else "localhost"
    port = int(parts[0] if len(parts) == 1 else parts[1])
    return NetworkClientHandle(host, port)


def eval_sys_fn_shutdown_client(x):
    """

        .clic(x)                                      [Close IPC client]

    """
    if isinstance(x, KGCall) and isinstance(x.a, KGLambda):
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
