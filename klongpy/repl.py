import asyncio
import contextlib
import sys
import threading
import os
import warnings
import importlib.resources
from typing import Optional

from . import KlongInterpreter
from .utils import CallbackEvent

# The names the front end (create_repl here, the kgpy command in cli.py) binds before a program runs, beyond the engine's own system
# names: a program may use them, so the linter must know them (klongpy.tools.builtin_names).
FRONT_END_NAMES = ('.system', '.helpdb', '.os.env', '.os.argv', '.help_topic')


PENDING_TASK_TIMEOUT = 5.0
# how long a task that is still running at exit is given to end by itself before it is reported and cancelled
FINISHING_TASK_GRACE = 1.0


class LoopStopper:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._future = loop.create_future()
        self.task = None        # the task that runs wait(): ends with the loop, not a leftover

    def set(self) -> None:
        if self._future.done():
            return
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._future.set_result, None)
        else:
            self._future.set_result(None)

    async def wait(self) -> None:
        await self._future


class ExitSignal:
    """
    How a callback ends the process. `.x` raises SystemExit; inside a timer, a
    request or a hook that would end the loop thread and leave the process
    waiting, so the loop thread hands the exit status to this object instead and
    whoever owns the process (the command line) acts on it. `handler`, when
    set, is called with the status in place of setting the event.
    """

    def __init__(self) -> None:
        self.event = threading.Event()
        self.code: Optional[int] = None
        self.handler = None

    def request(self, code=0) -> None:
        if self.code is not None:
            return
        if code is None:
            code = 0
        elif not isinstance(code, int):
            code = 1
        self.code = code
        if self.handler is not None:
            self.handler(code)
        self.event.set()


def start_loop(loop: asyncio.AbstractEventLoop, stop_event: LoopStopper, exit_signal: Optional[ExitSignal] = None) -> None:
    asyncio.set_event_loop(loop)
    try:
        stop_event.task = loop.create_task(stop_event.wait())
        loop.run_until_complete(stop_event.task)
    except SystemExit as exc:
        # .x in a callback: the loop is stopped, the owner of the process ends it
        if exit_signal is None:
            raise
        exit_signal.request(exc.code)


def setup_async_loop(debug: bool = False, slow_callback_duration: float = 86400.0, exit_signal: Optional[ExitSignal] = None):
    loop = asyncio.new_event_loop()
    loop.slow_callback_duration = slow_callback_duration
    if debug:
        loop.set_debug(True)
    stop_event = LoopStopper(loop)
    thread = threading.Thread(target=start_loop, args=(loop, stop_event, exit_signal), daemon=True)
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
    leftovers = [t for t in pending_tasks if t is not stop_event.task]
    if leftovers:
        # A task that is already ending (the last steps of a connection that was closed) needs a few more
        # turns of the loop, which stopped under it: let those finish before calling anything pending.
        loop.run_until_complete(asyncio.wait(leftovers, timeout=FINISHING_TASK_GRACE))
        pending_tasks = asyncio.all_tasks(loop=loop)
        leftovers = [t for t in pending_tasks if t is not stop_event.task]
    if len(pending_tasks) > 0:
        if name and leftovers:
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
        klongpath = os.environ.get('KLONGPATH', os.pathsep.join(('.', 'lib')))
        if os.name == 'nt' and os.pathsep in pkg_lib_path:
            pkg_lib_path = f'"{pkg_lib_path}"'
        if (klongpath == pkg_lib_path or pkg_lib_path in klongpath.split(os.pathsep)
                or klongpath.endswith(f"{os.pathsep}{pkg_lib_path}")):
            return
        os.environ['KLONGPATH'] = os.pathsep.join(filter(None, (klongpath, pkg_lib_path)))


def create_repl(debug: bool = False, backend: Optional[str] = None, device: Optional[str] = None, strict: int = 0):
    exit_signal = ExitSignal()
    io_loop, io_thread, io_stop = setup_async_loop(debug=debug, exit_signal=exit_signal)
    klong_loop, klong_thread, klong_stop = setup_async_loop(debug=debug, exit_signal=exit_signal)

    append_pkg_resource_path_KLONGPATH()

    klong = KlongInterpreter(backend=backend, device=device, strict=strict)
    shutdown_event = CallbackEvent()
    klong['.system'] = {'ioloop': io_loop, 'klongloop': klong_loop, 'closeEvent': shutdown_event,
                        'exitSignal': exit_signal}
    # the help database the library files add their entries to (math.kg and
    # the others load help.kg, which reads it): empty until one is loaded
    klong['.helpdb'] = []

    return klong, (io_loop, io_thread, io_stop, klong_loop, klong_thread, klong_stop)


def keeps_alive(system) -> Optional[str]:
    """
    Why a file run must not end yet: "server" while an IPC server listens,
    "connection" while a client connection opened by .cli or .clid is open,
    "web server" while a .web server listens, "websocket" while a .ws
    connection is open; None when nothing is left to wait for. A timer does not keep a file run up:
    it fires while a server or a connection does, or in the REPL.
    """
    server = system.get('ipc_server')
    if server is not None and server.server is not None:
        return "server"
    for nc in list(system.get('ipc_clients', ())):
        if nc.running and nc.is_open():
            return "connection"
    for web in list(system.get('web_servers', ())):
        if web.is_listening():
            return "web server"
    for ws in list(system.get('ws_clients', ())):
        if ws.running and ws.is_open():
            return "websocket"
    return None


def wait_while_alive(system, poll: float = 0.05) -> None:
    """Block until nothing keeps the process up (see keeps_alive) or a callback asked to exit."""
    signal = system['exitSignal']
    while signal.code is None and keeps_alive(system) is not None:
        signal.event.wait(poll)


def cleanup_repl(loops, debug: bool = False) -> None:
    io_loop, io_thread, io_stop, klong_loop, klong_thread, klong_stop = loops
    cleanup_async_loop(io_loop, io_thread, io_stop, debug=debug, name='io_loop')
    cleanup_async_loop(klong_loop, klong_thread, klong_stop, debug=debug, name='klong_loop')


# ---------------------------------------------------------------------------
# Reporting errors and warnings to a person.
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
