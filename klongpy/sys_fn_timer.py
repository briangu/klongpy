import asyncio
import sys
import threading

from klongpy.core import KGCall, KGFn, KGFnWrapper

_main_loop = asyncio.get_event_loop()
_main_tid = threading.current_thread().ident


class KGTimerHandler:
    def __init__(self, name, interval):
        self.name = name
        self.interval = interval
        self.delegate = None

    def cancel(self):
        if self.delegate is None:
            return 0
        self.delegate.cancel()
        self.delegate = None
        return 1
    
    def __str__(self):
        return f"timer:{self.name}:{self.interval}"


def _call_periodic(loop: asyncio.BaseEventLoop, name, interval, callback):
    start = loop.time()

    def run(handle, fn=callback):
        global _main_tid
        assert threading.current_thread().ident == _main_tid
        r = fn()
        if r:
            if interval == 0:
                handle.delegate = loop.call_soon(run, handle)
            else:
                handle.delegate = loop.call_later(interval - ((loop.time() - start) % interval), run, handle)
        else:
            handle.cancel()

    periodic = KGTimerHandler(name, interval)
    if interval == 0:
        periodic.delegate = loop.call_soon(run, periodic)
    else:
        periodic.delegate = loop.call_at(start + interval, run, periodic)

    return periodic


def eval_sys_fn_timer(klong, x, y, z):
    """
    
        .timer(x, y, z)                                   [Create-timer]

        Create a timer named by "x" that repeats every "y" seconds and calls function "z".

        An interval value of 0 indicates immediately callback (async).

        The callback function returns 1 to continue, 0 to stop time timer.

        Example:

            cb::{.p("hello")}
            th::.timer("greeting";1;cb)

            To stop the timer, it can be closed via:

            .timerc(th)

            The following example will create a timer which counts to 20 and then 
            terminates the timer by return 0 from the callback.

            counter::0
            u::{counter::counter+1;.p(counter);1}
            c::{.p("stopping timer");0}
            cb::{:[counter<20;u();c()]}
            th::.timer("count";1;cb)

    """
    global _main_loop
    y= int(y)
    if y < 0:
        return "x must be a non-negative integer"
    z = z if isinstance(z, KGCall) else KGFnWrapper(klong, z) if isinstance(z, KGFn) else z
    if not callable(z):
        return "z must be a function"
    return _call_periodic(_main_loop, x, y, z)


def eval_sys_fn_cancel_timer(x):
    """
    
        .timerc(x)                                        [Cancel-timer]

        Cancel a timer created by .timer()

    """
    if not isinstance(x, KGTimerHandler):
        return 0
    return x.cancel()


def create_system_functions_timer():
    def _get_name(s):
        i = s.index(".")
        return s[i : i + s[i:].index("(")]

    registry = {}

    m = sys.modules[__name__]
    for x in filter(lambda n: n.startswith("eval_sys_"), dir(m)):
        fn = getattr(m, x)
        registry[_get_name(fn.__doc__)] = fn

    return registry
