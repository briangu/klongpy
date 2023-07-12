import asyncio
import socket
import sys
import threading
from asyncio import StreamReader, StreamWriter

from aiohttp import web

from klongpy import KlongInterpreter
from klongpy.core import KGCall, KGFn, KGFnWrapper, KGLambda, reserved_fn_args


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


def get_input():
    return input("?> ")


class ConsoleInputHandler:
    def __init__(self, klong):
        self.klong = klong

    async def input_producer(self):
        loop = asyncio.get_event_loop()
        while True:
            command = await loop.run_in_executor(None, get_input)
            response = run_klong(self.klong, command)
            print(response)


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
        server = await asyncio.start_server(
            self.client_handler.handle_client, bind, port
        )

        addr = server.sockets[0].getsockname()
        # print(f'Serving on {addr}')

        async with server:
            await server.serve_forever()


class NetworkClientHandle:
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

    def __call__(self, x):
        # import timeit

        # def runit():
        #     self.sock.sendall((str(x) + "\n").encode('utf-8'))
        #     self.sock.recv(1024).decode('utf-8')
        # number = 100000
        # t = timeit.timeit(runit, number=number)
        # print(t, number, t/number, int(1/(t/number)))

        print(f"sending: {x}")

        self.sock.sendall((str(x) + "\n").encode("utf-8"))
        response = self.sock.recv(1024).decode("utf-8")
        return response

    def close(self):
        self.sock.close()
        self.sock = None

    def is_open(self):
        return self.sock is not None


# class NetworkServerHandle:
#     def __init__(self,tcp_task):
#         self.tcp_task = tcp_task

#     def __call__(self):
#         return 1


# def shutdown_server(x):
#     if isinstance(x,KGCall) and isinstance(x.a,KGLambda):
#         x = x.a.fn
#         if isinstance(x,NetworkServerHandle) and x.tcp_task is not None:
#             print("shutting down")
#             x.tcp_task.cancel()
#             x.tcp_task = None
#             return 1
#     return 0


def eval_sys_fn_create_client(hostname, port):
    return NetworkClientHandle(hostname, port)


def eval_sys_fn_shutdown_client(x):
    if isinstance(x, KGCall) and isinstance(x.a, KGLambda):
        x = x.a.fn
        if isinstance(x, NetworkClientHandle) and x.is_open():
            print("shutting down client")
            x.close()
            return 1
    return 0


# def create_server(loop, tcp_server_handler, bind, port):
#     tcp_task = loop.create_task(tcp_server_handler.tcp_producer(bind, port))
#     tcp_server_handler.set_task(tcp_task)
#     # return NetworkServerHandle(tcp_task)
#     return 1


class WebServerHandle:
    def __init__(self, runner, task):
        self.runner = runner
        self.task = task

    def __call__(self):
        return 1

    async def shutdown(self):
        self.task.cancel()
        await self.runner.cleanup()
        self.runner = None
        self.task = None


# def run_web_handler(k, fn, request):
#     q = run_klong(k,fn)
#     print("type(q): ", type(q))
#     return KGLambda(q)(request)


def eval_sys_fn_create_web_server(loop, klong, route_to_function_map):
    print(route_to_function_map)
    app = web.Application()
    for route, function in route_to_function_map.items():
        r = run_klong(klong, function)
        arity = r.arity if isinstance(r, KGFn) else 0
        fn = (
            r
            if isinstance(r, KGCall)
            else KGFnWrapper(klong, r)
            if isinstance(r, KGFn)
            else r
        )

        async def _runit(request, fn=fn, arity=arity):
            try:
                if request.method == "GET":
                    parameters = request.rel_url.query
                elif request.method == "POST":
                    parameters = await request.post()
                else:
                    return web.Response(text="Invalid method", status=405)

                fn_params = [parameters[x] for x in reserved_fn_args[:arity]]
                return web.Response(text=str(fn(*fn_params)))
            except KeyError as e:
                return web.Response(text=f"Missing parameter: {e}", status=400)
            except Exception as e:
                return web.Response(text="Invalid request", status=400)

        app.router.add_get(route, _runit)

    runner = web.AppRunner(app)

    async def start_server():
        await runner.setup()
        site = web.TCPSite(runner, "localhost", 8080)
        await site.start()

    server_task = loop.create_task(start_server())
    return WebServerHandle(runner, server_task)


def eval_sys_fn_shutdown_web_server(x):
    if isinstance(x, KGCall) and isinstance(x.a, KGLambda):
        x = x.a.fn
        if isinstance(x, WebServerHandle) and x.runner is not None:
            print("shutting down web server")
            loop.run_until_complete(x.shutdown())
            return 1
    return 0


if __name__ == "__main__":
    klong = KlongInterpreter()

    loop = asyncio.get_event_loop()

    tcp_server_handler = TcpServerHandler()

    klong[".srv"] = lambda klong, x, y, loop=loop: tcp_server_handler.create_server(
        loop, klong, x, y
    )
    klong[".srvc"] = lambda: tcp_server_handler.shutdown_server()
    klong[".cli"] = lambda x, y: eval_sys_fn_create_client(x, y)
    klong[".clic"] = lambda x: eval_sys_fn_shutdown_client(x)
    klong[".web"] = lambda klong, x, loop=loop: eval_sys_fn_create_web_server(
        loop, klong, x
    )
    klong[".webc"] = lambda x: eval_sys_fn_shutdown_web_server(x)
    # klong['.ws'] = lambda klong, x, loop=loop: create_web_server(loop,klong,x)
    # klong['.wsc'] = lambda x: shutdown_web_server(x)

    console_input_handler = ConsoleInputHandler(klong)

    loop.create_task(console_input_handler.input_producer())
    loop.run_forever()


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
