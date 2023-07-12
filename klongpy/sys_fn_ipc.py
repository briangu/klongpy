import asyncio
import socket
import sys
import threading
from asyncio import StreamReader, StreamWriter

from klongpy.core import KGCall, KGFn, KGFnWrapper, KGLambda


_main_loop = asyncio.get_event_loop()
_main_tid = threading.current_thread().ident


def run_klong(klong, cmd):
    assert threading.current_thread().ident == _main_tid
    return klong(cmd)


class TcpClientHandler:
    def __init__(self, klong):
        self.klong = klong

    async def handle_client(self, reader: StreamReader, writer: StreamWriter):
        while True:
            data = await reader.readline()
            command = data.decode().strip()
            if command:
                response = run_klong(self.klong, command)
                writer.write((str(response) + "\n").encode())
                await writer.drain()
            else:
                break

        writer.close()
        await writer.wait_closed()


class TcpServerHandler:
    def __init__(self):
        self.client_handler = None
        self.task = None

    def create_server(self, klong, bind, port):
        if self.task is not None:
            return 0
        self.client_handler = TcpClientHandler(klong)
        self.task = _main_loop.create_task(self.tcp_producer(bind, port))
        return 1

    def shutdown_server(self):
        if self.task is None:
            return 0
        self.task.cancel()
        self.task = None
        self.client_handler = None
        return 1

    async def tcp_producer(self, bind, port):
        server = await asyncio.start_server(
            self.client_handler.handle_client, bind, port
        )

        addr = server.sockets[0].getsockname()
        # print(f'Serving on {addr}')

        async with server:
            await server.serve_forever()


class NetworkClientHandle():
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

    def __call__(self, x):
        self.sock.sendall((str(x) + "\n").encode("utf-8"))
        response = self.sock.recv(1024).decode("utf-8")
        return response

    def close(self):
        self.sock.close()
        self.sock = None

    def is_open(self):
        return self.sock is not None
    
    # def __str__(self):
    #     return ":monad"


def eval_sys_fn_create_client(x):
    """

        .cli(x)                                      [Create IPC client]

    """
    print("connecting to: ", x)
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
            print("shutting down client")
            x.close()
            return 1
    return 0


_ipc_tcp_server = TcpServerHandler()


def eval_sys_create_ipc_server(klong, x):
    """

        .srv(x)                                       [Start IPC server]

    """
    x = str(x)
    parts = x.split(":")
    bind = parts[0] if len(parts) > 1 else None
    port = int(parts[0] if len(parts) == 1 else parts[1])
    return _ipc_tcp_server.create_server(klong, bind, port)


def eval_sys_shutdown_ipc_server():
    """

        .srvc()                                        [Stop IPC server]

    """
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
