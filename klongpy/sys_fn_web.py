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

    def __call__(self):
        return 1

    async def shutdown(self):
        self.task.cancel()
        await self.runner.cleanup()
        self.runner = None
        self.task = None


def eval_sys_fn_create_web_server(klong, x):
    """
    
            .web(x)                                       [Start Web server]

    """
    global _main_loop
    global _main_tid
    route_to_function_map = x
    app = web.Application()
    assert threading.current_thread().ident == _main_tid
    for route, function in route_to_function_map.items():
        r = klong(function)
        arity = r.arity if isinstance(r, KGFn) else 0
        fn = r if isinstance(r, KGCall) else KGFnWrapper(klong, r) if isinstance(r, KGFn) else r

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
