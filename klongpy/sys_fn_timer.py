import asyncio
import logging
import sys

from klongpy.core import KGCall, KGFn, KGFnWrapper, KlongException, KlongRangeError, KlongTypeError


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
        try:
            r = fn()
        except Exception:
            logging.exception("timer %s callback failed; cancelling timer", name)
            handle.cancel()
            return
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
    
        .timer(x;y;z)                                     [Create-timer]

        Create a timer named by "x" that calls the function "z" every "y"
        seconds (fractions allowed) and returns its handle. An interval of 0
        calls "z" at once, on the Klong loop. The result of the callback decides
        what happens next: a false value (0, an empty list or string) stops the
        timer, and any other value runs it again. A named function is looked up
        again on each tick, so a redefinition takes effect. Stop a timer with .timerc.

        Example:

            n::0
            cb::{n::n+1;.p(n);:[n<3;1;0]}
            th::.timer("count";0.2;cb)

    """
    try:
        y = float(y)
    except (TypeError, ValueError):
        raise KlongTypeError("y must be a non-negative number") from None
    if not y >= 0:
        raise KlongRangeError("y must be a non-negative number")

    # Wrap the callback - KGFnWrapper now handles dynamic resolution automatically
    if isinstance(z, KGCall):
        raise KlongTypeError("z must be a function (not a function call)")
    if isinstance(z, KGFn):
        callback = KGFnWrapper(klong, z)
    elif callable(z):
        callback = z
    else:
        raise KlongTypeError("z must be a function")

    system = klong['.system']
    klongloop = system['klongloop']
    return _call_periodic(klongloop, x, y, callback)


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
