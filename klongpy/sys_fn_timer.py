import asyncio
import sys

from klongpy.core import KGCall, KGFn, KGFnWrapper, KlongRangeError, KlongTypeError


class KGTimerHandler:
    def __init__(self, name, interval, loop=None):
        self.name = name
        self.interval = interval
        self.delegate = None
        self.loop = loop
        self.active = True

    def cancel(self):
        """Stop the timer from any thread; 1 if it was running, 0 if it had already stopped."""
        if not self.active:
            return 0
        self.active = False
        delegate, self.delegate = self.delegate, None
        if delegate is not None:
            if _on_loop(self.loop):
                delegate.cancel()
            else:
                self.loop.call_soon_threadsafe(delegate.cancel)
        return 1
    
    def __str__(self):
        return f"timer:{self.name}:{self.interval}"


def _report_failure(klong, name, exc):
    """Say on stderr that the timer's callback failed and why: the rendered error, with the traceback under -v."""
    from klongpy.repl import format_error, use_color, write_stderr
    try:
        traceback = bool(klong['.system'].get('traceback'))
    except (KeyError, AttributeError, TypeError):
        traceback = False
    write_stderr(f"timer {name} callback failed; cancelling timer\n"
                 + format_error(exc, use_color(sys.stderr), traceback))


def _on_loop(loop):
    """True when the caller is running on that event loop's thread (loop is None: yes)."""
    if loop is None:
        return True
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


def _call_periodic(loop: asyncio.BaseEventLoop, name, interval, callback, klong=None):
    start = loop.time()

    def run(handle, fn=callback):
        try:
            r = fn()
        except Exception as e:
            handle.cancel()
            _report_failure(klong, name, e)
            return
        if r:
            if interval == 0:
                handle.delegate = loop.call_soon(run, handle)
            else:
                handle.delegate = loop.call_later(interval - ((loop.time() - start) % interval), run, handle)
        else:
            handle.cancel()

    periodic = KGTimerHandler(name, interval, loop)

    def arm():
        if not periodic.active:
            return
        if interval == 0:
            periodic.delegate = loop.call_soon(run, periodic)
        else:
            periodic.delegate = loop.call_at(start + interval, run, periodic)

    # A loop that is idle in select() only notices what is scheduled from another
    # thread (a web handler, the main thread) through call_soon_threadsafe, which wakes it.
    if _on_loop(loop):
        arm()
    else:
        loop.call_soon_threadsafe(arm)

    return periodic


def eval_sys_fn_timer(klong, x, y, z):
    """
    
        .timer(x;y;z)                                     [Create-timer]

        Create a timer named by "x" that calls the function "z" every "y"
        seconds (fractions allowed) and returns its handle. An interval of 0
        calls "z" at once, on the Klong loop. The result of the callback decides
        what happens next: a false value (0, an empty list or string) stops the
        timer, and any other value runs it again. The function takes no arguments
        and is passed by name (cb::{...}; .timer("t";1;cb)): a function written
        inline in the call is rejected, or run at the call site if it has no
        arguments. A named function is looked up again on each tick, so a
        redefinition takes effect. Stop a timer with .timerc, from any thread.

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
    hint = "name the function first, cb::{...}, and pass cb; an inline function that takes arguments is rejected"
    if isinstance(z, KGCall):
        raise KlongTypeError("z must be a function that takes no arguments", hint=hint)
    if isinstance(z, KGFn):
        if z.arity != 0:
            raise KlongTypeError(f"z must be a function that takes no arguments, not one of {z.arity}", hint=hint)
        callback = KGFnWrapper(klong, z)
    elif callable(z):
        callback = z
    else:
        raise KlongTypeError("z must be a function", hint=hint)

    system = klong['.system']
    klongloop = system['klongloop']
    return _call_periodic(klongloop, x, y, callback, klong)


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
