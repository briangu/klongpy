#!/usr/bin/env python3
"""
KlongPy REPL command-line interface.

See https://t3x.org/klong/klong-ref.txt.html for additional details.
"""

import argparse
import asyncio
import importlib
import importlib.util
import importlib.metadata
import os
import re
import sys
import threading
import timeit

import colorama

from klongpy import KlongInterpreter, __version__, list_backends
from klongpy.core import kg_write
from klongpy.help_index import apropos_text, help_text
from klongpy.sys_fn import eval_sys_load
from klongpy.repl import cleanup_repl, create_repl

_READLINE_SPEC = importlib.util.find_spec("readline")
readline = importlib.import_module("readline") if _READLINE_SPEC else None


def sys_cmd_shell(klong, cmd):
    """

        ]! command                                               [Shell]

        Pass the given command to the Unix shell.

    """
    command = _arg(cmd)
    if not command:
        return "usage: ]! command"
    os.system(command)
    return None


def _arg(cmd):
    """Return the text after the one-letter command name, stripped."""
    return cmd[2:].strip()


def sys_cmd_apropos(klong, cmd):
    """

        ]a topic                                               [Apropos]

        List the help entries whose name or title contains "topic"
        (case-insensitive), e.g. ]a scan.

    """
    return apropos_text(_arg(cmd))


def sys_cmd_help(klong, cmd):
    """

        ]h topic                                                  [Help]

        Show help for an operator, adverb, system function or variable.
        The topic is a glyph (]h /), a name (]h Over, ]h .p) or part of
        a name. ]h alone lists all topics by group; ]hall shows every
        entry; ]hadverbs (also operators, variables, extensions, forms)
        lists one group.

    """
    try:
        return help_text(_arg(cmd))
    except Exception:
        # fall back to the hand-written operator table
        try:
            klong['help']
        except KeyError:
            klong('.l("help.kg")')
        klong['.help_topic'] = _arg(cmd)
        klong('help(.help_topic)')
        return None


def sys_cmd_dir(klong, cmd):
    """

        ]i dir                                               [Inventory]

        List all *.kg files (Klong source programs) in the given
        directory. When no directory is given, it defaults to the first
        element of the KLONGPATH variable. The ]i command depends on a
        Unix shell and the "ls" utility (it does "cd dir; ls *.kg").

    """
    cmd = _arg(cmd)
    dir = cmd if cmd else (os.environ.get("KLONGPATH") or "./").split(":")[0]
    if not os.path.isdir(dir):
        return f"no such directory: {dir}"
    found = False
    for x in sorted(os.listdir(dir)):
        if x.endswith(".kg"):
            print(x)
            found = True
    if not found:
        print(f"no .kg files found in {dir}")
    return None


def sys_cmd_load(klong, cmd):
    """

        ]l file                                                   [Load]

        ]l file is short for .l("file"). The name is used as typed
        (leading and trailing blanks are removed).

    """
    name = _arg(cmd)
    if not name:
        return "usage: ]l file"
    return eval_sys_load(klong, name) or 1


def sys_cmd_exit(klong, cmd):
    """

        ]q                                                        [Exit]

        ]q is short for .x(0). However, end-of-file (control-D on Unix)
        typically also works.

    """
    print("bye!")
    return ExitRequest(0)


_transcript = None
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def transcript_write(text):
    """Append text to the open transcript file, if any."""
    if _transcript is not None:
        _transcript.write(_ANSI.sub("", text) + "\n")
        _transcript.flush()


def sys_cmd_transcript(klong, cmd):
    """
        ]t file                                             [Transcript]

        Start appending user input and computed values to the given file.
        When no file is given, stop transcript. Input will be prefixed
        with a TAB (HT) character in the transcript file.

    """
    global _transcript
    name = _arg(cmd)
    if _transcript is not None:
        _transcript.close()
        _transcript = None
        if not name:
            return "transcript stopped"
    if not name:
        return "no transcript is open"
    try:
        _transcript = open(name, "a", encoding="utf-8")
    except OSError as e:
        return f"cannot open {name}: {e.strerror or e}"
    return f"transcript appended to {name}"


def sys_cmd_timeit(klong, cmd):
    """
        ]T prog
        ]T:N prog

        Time an klong expression for 1 or N iterations using the timeit facility.

        As Klong manual shows, you can perform timing functions in Klong using the .pc() operator.

        timeit::{[t0];t0::.pc();x@[];.pc()-t0}

        and then use it for a nilad:

        timeit({1+1})

        For one iteration, it's possible this Klong timeit is more accurate than the native Python timeit due to overhead.

        Note: Added in KlongPy.

    """
    usage = "usage: ]T expr  or  ]T:N expr"
    rest = cmd[2:].lstrip() if not cmd[2:].startswith(":") else cmd[2:]
    n = 1
    if rest.startswith(":"):
        count, _, rest = rest[1:].partition(" ")
        try:
            n = int(count)
        except ValueError:
            return f"iteration count must be a positive integer, got '{count}'; {usage}"
        if n < 1:
            return f"iteration count must be a positive integer, got '{count}'; {usage}"
    prog = rest.strip()
    if not prog:
        return usage
    r = timeit.timeit(lambda k=klong, p=prog: k(p), number=n)
    return f"total: {r} per: {r/n}"


def create_sys_cmd_functions():
    def _get_name(s):
        s = s.strip()
        x = s.index(']')+1
        return s[x:x+1]

    registry = {}

    m = sys.modules[__name__]
    for x in filter(lambda n: n.startswith("sys_cmd_"), dir(m)):
        fn = getattr(m,x)
        name = _get_name(fn.__doc__)
        registry[name] = fn

    return registry


def run_sys_cmd(klong, sys_cmds, s):
    """Run one ]-command line and return the value to print (or None).

    Never raises for a bad argument: problems come back as a one-line message.
    """
    if len(s) < 2 or s[1].isspace():
        return "system commands:\n" + "\n".join(
            f"  ]{k}  {_title(fn)}" for k, fn in sorted(sys_cmds.items()))
    fn = sys_cmds.get(s[1])
    if fn is None:
        return f"unknown system command: ]{s[1]}"
    try:
        return fn(klong, s)
    except (SystemExit, KeyboardInterrupt, EOFError):
        raise
    except Exception as e:
        return failure(f"Error: {e.args}")


def _title(fn):
    m = re.search(r"\[([^\]]+)\]\s*$", next(
        (ln for ln in (fn.__doc__ or "").splitlines() if ln.strip()), ""))
    return m.group(1) if m else ""


success = lambda input: f"{colorama.Fore.GREEN}{input}"
failure = lambda input: f"{colorama.Fore.RED}{input}"


class ExitRequest:
    def __init__(self, code=0):
        self.code = 0 if code is None else code


async def repl_eval(klong, p, verbose=True):
    try:
        r = klong(p)
        r = r if r is None else success(kg_write(r, klong._backend, display=False))
    except SystemExit as exc:
        return ExitRequest(exc.code)
    except Exception as e:
        r = failure(f"Error: {e.args}")
        if verbose:
            import traceback
            traceback.print_exception(type(e), e, e.__traceback__)

    return r


def _klongpy_version():
    try:
        return importlib.metadata.distribution('klongpy').version
    except importlib.metadata.PackageNotFoundError:
        return __version__


def show_repl_header(backend_name, device_name=None, ipc_addr=None):
    print()
    print(f"{colorama.Fore.GREEN}Welcome to KlongPy REPL v{_klongpy_version()}")
    print(f"{colorama.Fore.GREEN}Author: Brian Guarraci")
    print(f"{colorama.Fore.GREEN}Web: http://klongpy.org")
    device_str = f" ({device_name})" if device_name else ""
    print(f"{colorama.Fore.CYAN}Backend: {backend_name}{device_str}")
    print(f"{colorama.Fore.YELLOW}]h for help; Ctrl-D or ]q to quit")
    print()
    if ipc_addr:
        print(f"{colorama.Fore.RED}Running IPC server at {ipc_addr}")
        print()


def get_input():
    return input("?> ")


def run_in_loop(klong_loop, coro):
    return asyncio.run_coroutine_threadsafe(coro, klong_loop).result()


class ConsoleInputHandler:
    @staticmethod
    async def input_producer(console_loop, klong_loop, klong, verbose=False, exit_state=None):
        sys_cmds = create_sys_cmd_functions()
        try:
            while True:
                try:
                    s = await console_loop.run_in_executor(None, get_input)
                    if len(s) == 0:
                        continue
                    transcript_write("\t" + s)
                    if s.startswith("]"):
                        r = run_sys_cmd(klong, sys_cmds, s)
                    else:
                        r = run_in_loop(klong_loop, repl_eval(klong, s, verbose=verbose))
                    if isinstance(r, ExitRequest):
                        if exit_state is not None:
                            exit_state["code"] = r.code
                        break
                    if r is not None:
                        print(r)
                        transcript_write(str(r))
                except EOFError:
                    print("\rbye!")
                    break
                except KeyboardInterrupt:
                    buf = ""
                    if readline is not None:
                        try:
                            buf = readline.get_line_buffer()
                        except Exception:
                            buf = ""
                    if buf:
                        print()
                        continue
                    print("\rbye!")
                    if exit_state is not None:
                        exit_state["code"] = 130
                    break
                except Exception as e:
                    print(failure(f"Error: {e.args}"))
                    import traceback
                    traceback.print_exception(type(e), e, e.__traceback__)
        finally:
            console_loop.stop()


async def run_in_klong(klong, s):
    try:
        return klong(s)
    except SystemExit as exc:
        return ExitRequest(exc.code)


def run_file(klong_loop, klong, fname, verbose=False):
    # Add script directory to sys.path so .py/.pyf imports resolve sibling modules
    script_dir = os.path.dirname(os.path.abspath(fname))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    with open(fname, "r") as f:
        content = f.read()
    return run_in_loop(klong_loop, run_in_klong(klong, content))


def main():
    """Main entry point for the KlongPy REPL."""
    if '--' in sys.argv:
        index = sys.argv.index('--')
        main_args = sys.argv[:index]
        extras = sys.argv[index+1:]
    else:
        main_args = sys.argv
        extras = []

    parser = argparse.ArgumentParser(
        prog='KlongPy',
        description='KlongPy REPL',
        epilog='For help, go to http://klongpy.org')
    parser.add_argument('-e', '--expr', help='evaluate expression, no interactive mode')
    parser.add_argument('-l', '--load', help='load program from file')
    parser.add_argument('-s', '--server', help='start the IPC server', type=str)
    parser.add_argument('-t', '--test', help='test program from file')
    parser.add_argument('-v', '--verbose', help='enable verbose output', action="store_true")
    parser.add_argument('-d', '--debug', help='enable debug mode', action="store_true")
    parser.add_argument('--backend', help='set array backend', type=str.lower, choices=list_backends())
    parser.add_argument('--device', help='set backend device (torch only)', type=str)
    parser.add_argument('filename', nargs='?', help='filename to be run if no flags are specified')

    args = parser.parse_args(main_args[1:])

    # Default to the numpy backend: fast and predictable for general array work
    # (and it never surprises non-autograd .kg scripts with MPS-tensor issues).
    # The torch backend (GPU + exact autograd via :>) is opt-in: --backend torch.
    if args.backend is None:
        args.backend = 'numpy'

    if args.debug:
        print("args: ", args)

    if args.expr:
        try:
            klong = KlongInterpreter(backend=args.backend, device=args.device)
        except ValueError as exc:
            parser.error(str(exc))
        result = klong(args.expr)
        if result is not None:
            print(kg_write(result, klong._backend, display=False))
        return

    try:
        klong, loops = create_repl(debug=args.debug, backend=args.backend, device=args.device)
    except ValueError as exc:
        parser.error(str(exc))
    io_loop, _, _, klong_loop, _, _ = loops
    shutdown_event = klong['.system']['closeEvent']

    klong['.helpdb'] = []
    klong['.os.env'] = dict(os.environ)
    klong['.os.argv'] = extras if extras else []

    run_repl = False
    exit_code = None

    if args.server:
        r = klong(f".srv({args.server})")
        if r == 0:
            print("Failed to start server")
    elif args.test:
        print(f"Test: {args.test}")
        with open(args.test, "r") as f:
            for x in f.readlines():
                x = x.strip()
                if len(x) == 0 or x.startswith(":"):
                    continue
                print(x)
                klong(x)

    if args.filename:
        if args.verbose:
            print(f"Running: {args.filename}")
        result = run_file(klong_loop, klong, args.filename, verbose=args.verbose)
        if isinstance(result, ExitRequest):
            exit_code = result.code

        def gather_io_tasks(io_loop):
            done_event = threading.Event()
            tasks = asyncio.all_tasks(loop=io_loop)
            async def main_task():
                gathered_task = asyncio.gather(*tasks)
                gathered_task.add_done_callback(lambda x: done_event.set())
                await gathered_task
            if len(tasks) > 1:
                io_loop.call_soon_threadsafe(asyncio.create_task, main_task())
            else:
                done_event.set()
            return done_event

        if exit_code is None:
            gather_io_tasks(io_loop).wait()
    else:
        run_repl = True

    if args.debug:
        print("run_repl: ", run_repl)

    if run_repl:
        console_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(console_loop)
        if args.load:
            if args.verbose:
                print(f"Loading: {args.load}")
            result = run_file(klong_loop, klong, args.load, verbose=args.verbose)
            if isinstance(result, ExitRequest):
                exit_code = result.code
                run_repl = False
        if run_repl:
            exit_state = {"code": None}
            colorama.init(autoreset=True)
            backend_name = klong._backend.name
            device_name = str(klong._backend.device) if hasattr(klong._backend, 'device') else None
            show_repl_header(backend_name, device_name, args.server)
            console_loop.create_task(ConsoleInputHandler.input_producer(console_loop, klong_loop, klong, args.verbose, exit_state))
            try:
                console_loop.run_forever()
            except KeyboardInterrupt:
                exit_state["code"] = 130
                console_loop.stop()
            console_loop.close()
            if exit_state["code"] is not None:
                exit_code = exit_state["code"]

    shutdown_event.trigger()

    cleanup_repl(loops, debug=args.debug)

    if exit_code is not None:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
