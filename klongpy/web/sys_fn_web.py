import asyncio
import sys
import threading

from aiohttp import web

from klongpy.core import KGCall, KGFn, KGFnWrapper, KGLambda, reserved_fn_args

_main_loop = asyncio.get_event_loop()
_main_tid = threading.current_thread().ident


class WebServerHandle:
    def __init__(self, runner, task):
        self.runner = runner
        self.task = task

    # def __call__(self):
    #     return 1

    async def shutdown(self):
        self.task.cancel()
        await self.runner.cleanup()
        self.runner = None
        self.task = None

    def __str__(self):
        return "web server"


def eval_sys_fn_create_web_server(klong, x, y):
    """
    
            .web(x, y)                                     [Start Web server]

    """
    global _main_loop
    global _main_tid
    route_get_map = y
    app = web.Application()
    assert threading.current_thread().ident == _main_tid
    for route, function in route_get_map.items():
        # r = klong(function)
        r = function
        arity = r.arity if isinstance(r, KGFn) else r.get_arity() if issubclass(type(r), KGLambda) else 0
        if arity != 1:
            print(f"route {route} handler function requires arity 1, got {arity}")
            continue
        fn = r if isinstance(r, KGCall) else KGFnWrapper(klong, r) if isinstance(r, KGFn) else r

        async def _runit(request: web.Request, fn=fn):
            try:
                # print(request, fn)
                assert request.method == "GET"
                parameters = dict(request.rel_url.query)
                # if request.method == "GET":
                #     parameters = request.rel_url.query
                # # elif request.method == "POST":
                # #     parameters = await request.post()
                # else:
                #     return web.Response(text="Invalid method", status=405)

                # fn_params = [parameters[x] for x in reserved_fn_args[:arity]]
                response = fn(parameters)
                # print(response)
                return web.Response(text=str(response))
            # except KeyError as e:
            #     return web.Response(text=f"Missing parameter: {e}", status=400)
            except Exception as e:
                print(e)
                return web.Response(text="Invalid request", status=400)

        app.router.add_get(route, _runit)

    runner = web.AppRunner(app)

    x = str(x)
    parts = x.split(":")
    bind = parts[0] if len(parts) > 1 else None
    port = int(parts[0] if len(parts) == 1 else parts[1])

    print(f"starting web server at: {bind}:{port}")

    async def start_server():
        await runner.setup()
        site = web.TCPSite(runner, bind, port)
        await site.start()

    server_task = _main_loop.create_task(start_server())
    return WebServerHandle(runner, server_task)


def eval_sys_fn_shutdown_web_server(x):
    """
    
            .webc(x)                                      [Stop Web server]
    
    """
    global _main_loop
    global _main_tid
    if isinstance(x, KGCall) and issubclass(type(x.a), KGLambda):
        x = x.a.fn
        if isinstance(x, WebServerHandle) and x.runner is not None:
            print("shutting down web server")
            _main_loop.run_until_complete(x.shutdown())
            return 1
    return 0


def create_system_functions_web():
    def _get_name(s):
        i = s.index(".")
        return s[i : i + s[i:].index("(")]

    registry = {}

    m = sys.modules[__name__]
    for x in filter(lambda n: n.startswith("eval_sys_"), dir(m)):
        fn = getattr(m, x)
        registry[_get_name(fn.__doc__)] = fn

    return registry
