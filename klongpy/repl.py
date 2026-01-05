import asyncio
import threading
import time
import os
import importlib.resources
from typing import Optional

from . import KlongInterpreter
from .utils import CallbackEvent


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

    pending_tasks = asyncio.all_tasks(loop=loop)
    if len(pending_tasks) > 0:
        if name:
            print(f"WARNING: pending tasks in {name} loop")
        for task in pending_tasks:
            loop.call_soon_threadsafe(task.cancel)
        while len(asyncio.all_tasks(loop=loop)) > 0:
            time.sleep(0)

    loop.stop()

    if not loop.is_closed():
        loop.close()


def append_pkg_resource_path_KLONGPATH() -> None:
    with importlib.resources.as_file(importlib.resources.files('klongpy')) as pkg_path:
        pkg_lib_path = os.path.join(pkg_path, 'lib')
        klongpath = os.environ.get('KLONGPATH', '.:lib')
        klongpath = f"{klongpath}:{pkg_lib_path}" if klongpath else str(pkg_lib_path)
        os.environ['KLONGPATH'] = klongpath


def create_repl(debug: bool = False):
    io_loop, io_thread, io_stop = setup_async_loop(debug=debug)
    klong_loop, klong_thread, klong_stop = setup_async_loop(debug=debug)

    append_pkg_resource_path_KLONGPATH()

    klong = KlongInterpreter()
    shutdown_event = CallbackEvent()
    klong['.system'] = {'ioloop': io_loop, 'klongloop': klong_loop, 'closeEvent': shutdown_event}

    return klong, (io_loop, io_thread, io_stop, klong_loop, klong_thread, klong_stop)


def cleanup_repl(loops, debug: bool = False) -> None:
    io_loop, io_thread, io_stop, klong_loop, klong_thread, klong_stop = loops
    cleanup_async_loop(io_loop, io_thread, io_stop, debug=debug, name='io_loop')
    cleanup_async_loop(klong_loop, klong_thread, klong_stop, debug=debug, name='klong_loop')
