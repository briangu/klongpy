import asyncio
import contextlib
import sys
import threading
import time
import os
import warnings
import importlib.resources
from typing import Optional

from . import KlongInterpreter
from .utils import CallbackEvent


PENDING_TASK_TIMEOUT = 5.0


class LoopStopper:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._future = loop.create_future()

    def set(self) -> None:
        if self._future.done():
            return
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._future.set_result, None)
        else:
            self._future.set_result(None)

    async def wait(self) -> None:
        await self._future


def start_loop(loop: asyncio.AbstractEventLoop, stop_event: LoopStopper) -> None:
    asyncio.set_event_loop(loop)
    loop.run_until_complete(stop_event.wait())


def setup_async_loop(debug: bool = False, slow_callback_duration: float = 86400.0):
    loop = asyncio.new_event_loop()
    loop.slow_callback_duration = slow_callback_duration
    if debug:
        loop.set_debug(True)
    stop_event = LoopStopper(loop)
    thread = threading.Thread(target=start_loop, args=(loop, stop_event), daemon=True)
    thread.start()
    return loop, thread, stop_event


def cleanup_async_loop(loop: asyncio.AbstractEventLoop, loop_thread: threading.Thread, stop_event: LoopStopper, debug: bool = False, name: Optional[str] = None) -> None:
    if loop.is_closed():
        return

    loop.call_soon_threadsafe(stop_event.set)
    loop_thread.join()

    # The loop thread has exited, so the loop is not running: drive it here until
    # the cancelled tasks have finished (bounded, so a task that ignores
    # cancellation cannot hang the caller).
    pending_tasks = asyncio.all_tasks(loop=loop)
    if len(pending_tasks) > 0:
        if name:
            print(f"WARNING: pending tasks in {name} loop")
        for task in pending_tasks:
            task.cancel()
        loop.run_until_complete(asyncio.wait(pending_tasks, timeout=PENDING_TASK_TIMEOUT))

    if not loop.is_closed():
        loop.close()


def append_pkg_resource_path_KLONGPATH() -> None:
    with importlib.resources.as_file(importlib.resources.files('klongpy')) as pkg_path:
        pkg_lib_path = os.path.join(pkg_path, 'lib')
        pkg_lib_path = str(pkg_lib_path)
        klongpath = os.environ.get('KLONGPATH', '.:lib')
        if pkg_lib_path in klongpath.split(':'):
            return
        os.environ['KLONGPATH'] = f"{klongpath}:{pkg_lib_path}" if klongpath else pkg_lib_path


def create_repl(debug: bool = False, backend: Optional[str] = None, device: Optional[str] = None, strict: int = 0):
    io_loop, io_thread, io_stop = setup_async_loop(debug=debug)
    klong_loop, klong_thread, klong_stop = setup_async_loop(debug=debug)

    append_pkg_resource_path_KLONGPATH()

    klong = KlongInterpreter(backend=backend, device=device, strict=strict)
    shutdown_event = CallbackEvent()
    klong['.system'] = {'ioloop': io_loop, 'klongloop': klong_loop, 'closeEvent': shutdown_event}

    return klong, (io_loop, io_thread, io_stop, klong_loop, klong_thread, klong_stop)


def cleanup_repl(loops, debug: bool = False) -> None:
    io_loop, io_thread, io_stop, klong_loop, klong_thread, klong_stop = loops
    cleanup_async_loop(io_loop, io_thread, io_stop, debug=debug, name='io_loop')
    cleanup_async_loop(klong_loop, klong_thread, klong_stop, debug=debug, name='klong_loop')


# ---------------------------------------------------------------------------
# Reporting errors and warnings to a person (ticket 042).
#
# KlongError.render() does the layout; this part decides colour, where the text
# goes, how a Python exception that is not a KlongError is shown, and how a
# Python warning raised by the parser is shown.

TRACEBACK_HINT = "run with -v for the Python traceback"
_RED = "\x1b[31m"
_YELLOW = "\x1b[33m"
_RESET = "\x1b[0m"


def use_color(stream) -> bool:
    """Colour only on a terminal, and never when NO_COLOR is set to a non-empty value (no-color.org)."""
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def format_error(exc: BaseException, color: bool = False, traceback: bool = False) -> str:
    """
    The text shown for an exception, without a trailing newline.

    A KlongError is its render() (position, source line, caret, hint). Any other
    exception is a bug in KlongPy or in host code: one line,
    `internal error: Type: message`, and how to get the traceback. With
    traceback=True the Python traceback follows, including the __cause__ chain.
    """
    from .errors import KlongError, KlongInterrupt
    if isinstance(exc, KlongInterrupt):
        text = "interrupted"
    elif isinstance(exc, KlongError):
        text = exc.render(color=color)
        if exc.kind == 'internal' and not traceback:
            text += f"\n  hint: {TRACEBACK_HINT}"
    else:
        msg = str(exc)
        head = f"internal error: {type(exc).__name__}" + (f": {msg}" if msg else "")
        if "\n" in head:
            head = head.split("\n", 1)[0]
        text = f"{_RED}{head}{_RESET}" if color else head
        if not traceback:
            text += f"\n  hint: {TRACEBACK_HINT}"
    if traceback:
        import traceback as tb
        text += "\n" + "".join(tb.format_exception(type(exc), exc, exc.__traceback__)).rstrip("\n")
    return text



def _klong_warning_classes():
    """Every Warning subclass whose name starts with Klong (the parser's warning category lives in klongpy/errors.py)."""
    found, stack = [], [Warning]
    while stack:
        for c in stack.pop().__subclasses__():
            stack.append(c)
            if c.__name__.startswith("Klong"):
                found.append(c)
    return found


def format_warning(message, color: bool = False) -> str:
    """
    A Klong warning as text: `[name:]line:col: warning: message`, then the source
    line and carets when the warning carries a span and a source, then an optional
    `hint`. A warning with its own render(color) (KlongSyntaxWarning) is rendered by
    it; for any other class whose name starts with Klong the same layout is built
    from its `span`, `source` and `hint` attributes, when it has them.
    """
    render = getattr(message, 'render', None)
    if callable(render):        # klongpy.errors.KlongSyntaxWarning
        return render(color=color)
    from .errors import KlongError
    view = KlongError(str(message), span=getattr(message, 'span', None),
                      source=getattr(message, 'source', None), hint=getattr(message, 'hint', None))
    view.kind = 'warning'
    lines = view.render().split("\n")
    lines[0] = lines[0].replace("warning error: ", "warning: ", 1)
    if color:
        lines[0] = f"{_YELLOW}{lines[0]}{_RESET}"
    return "\n".join(lines)


def write_stderr(text: str) -> None:
    """Write a line to stderr after flushing stdout, so the two interleave in order when both go to one place."""
    try:
        sys.stdout.flush()
    except Exception:
        pass
    print(text, file=sys.stderr, flush=True)


@contextlib.contextmanager
def klong_warnings():
    """
    Show Klong warnings (any Warning whose class name starts with Klong) in the short
    form on stderr as they happen, every time (not once per location). Other warnings
    are left to Python's own handling.
    """
    with warnings.catch_warnings():
        for cat in _klong_warning_classes():
            warnings.simplefilter("always", cat)
        previous = warnings.showwarning

        def show(message, category, filename, lineno, file=None, line=None):
            if category.__name__.startswith("Klong"):
                write_stderr(format_warning(message, color=use_color(sys.stderr)))
            else:
                previous(message, category, filename, lineno, file, line)

        warnings.showwarning = show
        yield
