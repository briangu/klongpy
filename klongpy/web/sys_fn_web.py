import asyncio
import logging
import sys
import threading

from aiohttp import web

from klongpy.core import KGCall, KGFn, KGFnWrapper, KGLambda

_main_loop = asyncio.get_event_loop()
_main_tid = threading.current_thread().ident


class WebServerHandle:
    def __init__(self, bind, port, runner, task):
        self.bind = bind
        self.port = port
        self.runner = runner
        self.task = task

    async def shutdown(self):
        self.task.cancel()
        await self.runner.cleanup()
        self.runner = None
        self.task = None

    def __str__(self):
        return f"web[{self.bind or '0.0.0.0'}:{self.port}]"


def eval_sys_fn_create_web_server(klong, x, y, z):
    """
    
        .web(x, y, z)                                 [Start Web server]

        Start a web server and return its handle.

        The web server is started at the address specifed by "x":

        If "x" is an integer, then it is interpreted as a port in "0.0.0.0:<port>".
        if "x" is a string, then it is interpreted as a bind address "<bind>:<port>"

        GET routes are specified in a dictionary provided by "y".
        POST routes are specified in a dictionary provided by "z".

        Handler callbacks are called with a dictionary of parameters:

        GET handlers are provided a dictionary of query parameters.
        POST handlers are provided a dictionary of post body parameters.

        Example:

            .py("klongpy.web")
            get:::{}
            get,"/",{x;"hello, world!"}
            wh::.web(8080;get;:{})

            You should now be able to run:

            > curl http://localhost:8080/
            hello, world!

    """
    global _main_loop
    global _main_tid
    app = web.Application()

    logging.info("web server start @ ", x)
    logging.info("GET: ", y)
    logging.info("POST: ", z)

    assert threading.current_thread().ident == _main_tid
    
    for route, fn in y.items():
        arity = fn.arity if isinstance(fn, KGFn) else fn.get_arity() if issubclass(type(fn), KGLambda) else 0
        if arity != 1:
            logging.info(f"GET route {route} handler function requires arity 1, got {arity}")
            continue
        fn = fn if isinstance(fn, KGCall) else KGFnWrapper(klong, fn) if isinstance(fn, KGFn) else fn

        async def _get(request: web.Request, fn=fn, route=route):
            try:
                assert request.method == "GET"
                return web.Response(text=str(fn(dict(request.rel_url.query))))
            except Exception as e:
                logging.info(f"failed web request: {route} with error {e}")
                return web.Response(text="Invalid request", status=400)

        logging.info("adding GET route: ", route)

        app.router.add_get(route, _get)

    for route, fn in z.items():
        arity = fn.arity if isinstance(fn, KGFn) else fn.get_arity() if issubclass(type(fn), KGLambda) else 0
        if arity != 1:
            logging.info(f"POST route {route} handler function requires arity 1, got {arity}")
            continue
        fn = fn if isinstance(fn, KGCall) else KGFnWrapper(klong, fn) if isinstance(fn, KGFn) else fn

        async def _post(request: web.Request, fn=fn):
            try:
                assert request.method == "POST"
                parameters = dict(await request.post())
                return web.Response(text=str(fn(parameters)))
            except Exception as e:
                logging.error(e)
                return web.Response(text="Invalid request", status=400)

        logging.info("adding POST route: ", route)

        app.router.add_post(route, _post)

    runner = web.AppRunner(app)

    x = str(x)
    parts = x.split(":")
    bind = parts[0] if len(parts) > 1 else None
    port = int(parts[0] if len(parts) == 1 else parts[1])

    async def start_server():
        await runner.setup()
        site = web.TCPSite(runner, bind, port)
        await site.start()

    server_task = _main_loop.create_task(start_server())
    return WebServerHandle(bind, port, runner, server_task)


def eval_sys_fn_shutdown_web_server(x):
    """
    
            .webc(x)                                      [Stop Web server]
    
            Stop and close the web server referenced by "x".

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
