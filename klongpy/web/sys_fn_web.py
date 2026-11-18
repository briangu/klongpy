import logging
import sys
import asyncio
import concurrent.futures
import socket
import threading

from aiohttp import web

from klongpy.core import KGCall, KGFn, KGFnWrapper, KGLambda
from klongpy.errors import KlongError, KlongTypeError, classify

RESPONSE_KEYS = ("status", "type", "body")
SERVER_ERROR_BODY = "handler failed"
SELF_WAIT_BODY = ("a program cannot wait for its own server from the Klong loop: the request blocks the loop "
                  "that has to answer it. Use .async or call from another process")
SELF_WAIT_GRACE = 1.0   # seconds a request waits for the Klong loop before the self-wait test is made


def _on_klongloop(klong, running, fn, *args, skip=None):
    """
    Run fn(*args) on the Klong loop (the thread timers, .async and IPC callbacks run on) and
    return an awaitable for its result. Handlers run there, not on the io loop that serves the
    sockets, so that what a handler starts (.timer, .async, .cli, .webc) belongs to the loop
    that serves it and a blocking call in a handler does not stall the server's I/O.
    """
    try:
        klongloop = klong['.system']['klongloop']
    except (KeyError, TypeError):
        klongloop = None
    if klongloop is None or klongloop is running:
        return None, fn(*args)
    future = running.create_future()

    def finish(setter, value):
        if not future.done():
            setter(value)

    def run():
        if skip is not None and skip.is_set():
            return          # the request was answered without the handler
        try:
            result = fn(*args)
        except Exception as e:  # noqa: BLE001 - handed back to the request
            running.call_soon_threadsafe(finish, future.set_exception, e)
        else:
            running.call_soon_threadsafe(finish, future.set_result, result)

    klongloop.call_soon_threadsafe(run)
    return future, None


def _same_address(a, b):
    return a is not None and b is not None and tuple(a[:2]) == tuple(b[:2])


def _sockets_of(value):
    """The sockets a Python value stands for: a socket, an HTTP connection (sock), a socket stream (_sock, raw, fp)."""
    found = []
    for v in (value, getattr(value, 'sock', None), getattr(value, '_sock', None),
              getattr(getattr(value, 'raw', None), '_sock', None),
              getattr(getattr(getattr(value, 'fp', None), 'raw', None), '_sock', None)):
        if isinstance(v, socket.socket):
            found.append(v)
    return found


def _waits_for_itself(klong, request):
    """
    True when the thread of the Klong loop holds, in its Python stack, the client end of the connection this
    request arrived on: Klong code is waiting for this very request, which only the Klong loop can answer.

    A slow handler, a client in another process and a client on another thread never match, because the test
    is the identity of the socket. It sees clients that keep their socket in a local variable of the stack:
    urllib, http.client and what is built on them (requests); a client that hides its socket elsewhere is not
    detected.
    """
    try:
        transport = request.transport
        client = transport.get_extra_info('peername')
        server = transport.get_extra_info('sockname')
        if client is None or server is None:
            return False
        tid = getattr(klong['.system']['klongloop'], '_thread_id', None)
        frame = sys._current_frames().get(tid)
        while frame is not None:
            for v in list(frame.f_locals.values()):
                for sock in _sockets_of(v):
                    try:
                        if _same_address(sock.getsockname(), client) and _same_address(sock.getpeername(), server):
                            return True
                    except OSError:
                        continue
            frame = frame.f_back
    except Exception:  # noqa: BLE001 - a failed test means "not proven": keep waiting
        return False
    return False


async def _wait_for_klongloop(klong, request, future, skip):
    """Wait for the handler's result. Returns (True, result), or (False, None) when the Klong loop is waiting for this request."""
    while True:
        done, _ = await asyncio.wait({future}, timeout=SELF_WAIT_GRACE)
        if done:
            return True, future.result()
        if _waits_for_itself(klong, request):
            skip.set()
            future.cancel()
            return False, None


def _log_handler_failure(klong, method, route, exc):
    """One line on stderr: the route and the Klong error (kind and message); the traceback with -v/--traceback."""
    from klongpy.repl import write_stderr
    err = exc if isinstance(exc, KlongError) else classify(exc)(str(exc))
    try:
        traceback = bool(klong['.system'].get('traceback'))
    except (KeyError, AttributeError, TypeError):
        traceback = False
    line = f"web {method} {route}: handler failed: {err.headline()}"
    if traceback:
        import traceback as tb
        line += "\n" + "".join(tb.format_exception(type(exc), exc, exc.__traceback__)).rstrip("\n")
    write_stderr(line)


def _to_text(value):
    return value if isinstance(value, str) else str(value)


def _make_response(value):
    """
    The response for a handler's value: text is the body with status 200; a dictionary may carry
    "status" (an integer from 100 to 599), "type" (the content type) and "body" (text), each
    optional. Any other value is converted with str, as a body. An unknown key, or a value of
    the wrong kind, is a type error.
    """
    if isinstance(value, dict):
        for key in value:
            if key not in RESPONSE_KEYS:
                raise KlongTypeError(f'unknown response key "{key}"', hint='the keys are "status", "type" and "body"')
        status = value.get("status", 200)
        if isinstance(status, bool) or not isinstance(status, (int, float)) or int(status) != status or not 100 <= status <= 599:
            raise KlongTypeError(f"response status must be an integer from 100 to 599, got {status!r}")
        ctype = value.get("type")
        if ctype is not None and not isinstance(ctype, str):
            raise KlongTypeError("response type must be a string")
        body = _to_text(value.get("body", ""))
        kwargs = {"status": int(status), "text": body}
        if ctype:
            kwargs["content_type"] = ctype.split(";", 1)[0].strip()
            if "charset=" in ctype:
                kwargs["charset"] = ctype.split("charset=", 1)[1].split(";", 1)[0].strip()
            else:
                kwargs["charset"] = "utf-8"
        return web.Response(**kwargs)
    return web.Response(text=_to_text(value))


class WebServerHandle:
    def __init__(self, bind, port, runner, task):
        self.bind = bind
        self.port = port
        self.runner = runner
        self.task = task
        self.ioloop = None
        self.shutdown_event = None
        self.registry = None

    def is_listening(self):
        """True from the moment .web returns until the server is stopped or failed to start.

        A run of a file stays up while this holds (klongpy.repl.keeps_alive).
        """
        task = self.task
        if self.runner is None or task is None:
            return False
        if task.done():
            return not task.cancelled() and task.exception() is None
        return True

    async def shutdown(self):
        runner, task = self.runner, self.task
        self.runner = None
        self.task = None
        if self.shutdown_event is not None:
            self.shutdown_event.unsubscribe(self.close)
        if self.registry is not None:
            self.registry.discard(self)
        if task is not None:
            task.cancel()
        if runner is not None:
            await runner.cleanup()

    def close_soon(self):
        """Stop the server without waiting: from a handler or a timer, which the stop must not wait for."""
        if self.runner is None or self.ioloop is None or self.ioloop.is_closed():
            return
        self.ioloop.call_soon_threadsafe(lambda: asyncio.ensure_future(self.shutdown()))

    def close(self):
        """Stop the server from any thread (the process is ending, or .webc)."""
        if self.runner is None or self.ioloop is None or self.ioloop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self.shutdown(), self.ioloop).result(10)
        except Exception:  # noqa: BLE001 - the loop is going away
            pass

    def __str__(self):
        return f"web[{self.bind or '0.0.0.0'}:{self.port}]"


async def _serve(klong, request, fn, route, get_params, is_post=False):
    """Answer one request: read the parameters, run the handler on the Klong loop, build the response."""
    method = "GET" if request.method == "HEAD" else request.method
    try:
        params = dict(await get_params()) if is_post else get_params()
    except Exception as e:  # noqa: BLE001 - a body that cannot be read is the client's fault
        logging.info("web %s %s: bad request: %s", method, route, e)
        return web.Response(text="Invalid request", status=400)
    try:
        skip = threading.Event()
        future, value = _on_klongloop(klong, asyncio.get_running_loop(), fn, params, skip=skip)
        if future is not None:
            answered, value = await _wait_for_klongloop(klong, request, future, skip)
            if not answered:
                from klongpy.repl import write_stderr
                write_stderr(f"web {method} {route}: refused with 503: {SELF_WAIT_BODY}")
                return web.Response(text=SELF_WAIT_BODY, status=503)
        return _make_response(value)
    except Exception as e:  # noqa: BLE001 - any handler failure answers 500
        _log_handler_failure(klong, method, route, e)
    return web.Response(text=SERVER_ERROR_BODY, status=500)


def eval_sys_fn_create_web_server(klong, x, y, z):
    """

        .web(x;y;z)                                   [Start Web server]

        Start a web server and return its handle.

        The web server is started at the address specified by "x":

        If "x" is an integer, then it is interpreted as a port in "0.0.0.0:<port>".
        if "x" is a string, then it is interpreted as a bind address "<bind>:<port>"

        Security: a bare port listens on all interfaces (0.0.0.0), and routes
        run the Klong functions you register with the request parameters.
        KlongPy assumes every peer is trusted. To limit access, bind to
        localhost, as in .web("127.0.0.1:8080";get;post), and firewall the
        port. Never expose it to an untrusted network or the internet.

        GET routes are specified in a dictionary provided by "y".
        POST routes are specified in a dictionary provided by "z".

        Handler callbacks are called with a dictionary of parameters:

        GET handlers are provided a dictionary of query parameters.
        POST handlers are provided a dictionary of post body parameters.

        A handler returns text (status 200) or a dictionary with the optional
        keys "status", "type" and "body". A handler that fails answers 500 and
        logs one line. Handlers run on the Klong loop, so they may start timers
        and make .cli and .async calls. HEAD is answered as GET without a body;
        another method on a route answers 405.

        Example:

            .py("klongpy.web")
            get:::{}
            get,"/",{x;"hello, world!"}
            wh::.web(8080;get;:{})

            You should now be able to run:

            > curl http://localhost:8080/
            hello, world!

    """
    app = web.Application()

    logging.info("web server start @ %s", x)
    logging.info("GET: %s", y)
    logging.info("POST: %s", z)

    for route, fn in y.items():
        arity = fn.arity if isinstance(fn, KGFn) else fn.get_arity() if issubclass(type(fn), KGLambda) else 0
        if arity != 1:
            logging.info(f"GET route {route} handler function requires arity 1, got {arity}")
            continue

        # Wrap function - KGFnWrapper now handles dynamic resolution automatically
        if isinstance(fn, KGCall):
            logging.info(f"GET route {route} handler cannot be a function call")
            continue
        fn_wrapped = KGFnWrapper(klong, fn) if isinstance(fn, KGFn) else fn

        async def _get(request: web.Request, fn=fn_wrapped, route=route):
            return await _serve(klong, request, fn, route, lambda: dict(request.rel_url.query))

        logging.info("adding GET route: %s", route)

        app.router.add_get(route, _get)

    for route, fn in z.items():
        arity = fn.arity if isinstance(fn, KGFn) else fn.get_arity() if issubclass(type(fn), KGLambda) else 0
        if arity != 1:
            logging.info(f"POST route {route} handler function requires arity 1, got {arity}")
            continue

        # Wrap function - KGFnWrapper now handles dynamic resolution automatically
        if isinstance(fn, KGCall):
            logging.info(f"POST route {route} handler cannot be a function call")
            continue
        fn_wrapped = KGFnWrapper(klong, fn) if isinstance(fn, KGFn) else fn

        async def _post(request: web.Request, fn=fn_wrapped, route=route):
            return await _serve(klong, request, fn, route, request.post, is_post=True)

        logging.info("adding POST route: %s", route)

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

    # create the server task in the ioloop thread and capture the task handle
    server_loop = klong['.system']['ioloop']
    task_future = concurrent.futures.Future()

    def _start():
        task = asyncio.create_task(start_server())
        task_future.set_result(task)

    server_loop.call_soon_threadsafe(_start)
    server_task = task_future.result()

    handle = WebServerHandle(bind, port, runner, server_task)
    system = klong['.system']
    handle.ioloop = server_loop
    # a file run stays up while the server listens (klongpy.repl.keeps_alive), and the
    # server is stopped cleanly when the process ends
    handle.registry = system.setdefault('web_servers', set())
    handle.registry.add(handle)
    handle.shutdown_event = system.get('closeEvent')
    if handle.shutdown_event is not None:
        handle.shutdown_event.subscribe(handle.close)
    return handle


def eval_sys_fn_shutdown_web_server(klong, x):
    """

            .webc(x)                                      [Stop Web server]

            Stop and close the web server referenced by "x". Called in a
            handler, it lets the reply being served finish and then stops
            the server.

    """
    if isinstance(x, KGCall) and issubclass(type(x.a), KGLambda):
        x = x.a.fn
    if isinstance(x, WebServerHandle) and x.runner is not None:
        logging.info("shutting down web server")
        if _on_loop_of(klong['.system'].get('klongloop')):
            # called from a handler or a timer (on the Klong loop): waiting for the stop would wait for the
            # request being served, which waits for this call. The reply is completed, then the server stops.
            x.close_soon()
        else:
            asyncio.run_coroutine_threadsafe(x.shutdown(), klong['.system']['ioloop']).result()
        return 1
    return 0


def _on_loop_of(loop):
    try:
        return loop is not None and asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


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
