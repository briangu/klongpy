#!/usr/bin/env python3
"""
KlongPy REPL command-line interface.

See https://t3x.org/klong/klong-ref.txt.html for additional details.
"""

import argparse
import asyncio
import concurrent.futures
import ctypes
import importlib
import importlib.util
import importlib.metadata
import os
import re
import signal
import sys
import timeit

import colorama

from klongpy import __version__, list_backends
from klongpy.core import KGSym, kg_write
from klongpy.errors import KlongSystemError
from klongpy.help_index import apropos_text, help_text
from klongpy.source import Source
from klongpy.repl import (cleanup_repl, create_repl, format_error, keeps_alive, klong_warnings, use_color,  # noqa: F401
                          wait_while_alive, write_stderr)
from klongpy.repl_input import Completer, History, check_text, history_path

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
        element of the KLONGPATH variable.

    """
    cmd = _arg(cmd)
    dir = cmd if cmd else (os.environ.get("KLONGPATH") or "./").split(os.pathsep)[0]
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
    try:
        return klong.apply(KGSym(".l"), [name]) or 1
    except FileNotFoundError as e:
        return f"cannot load {name}: {e}"


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


def run_sys_cmd(klong, sys_cmds, s, verbose=False):
    """Run one ]-command line and return the value to print (or None).

    Never raises for a bad argument: problems come back as a one-line message,
    an error as a ReplError.
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
        return ReplError(format_error(e, use_color(sys.stderr), verbose))


def _title(fn):
    m = re.search(r"\[([^\]]+)\]\s*$", next(
        (ln for ln in (fn.__doc__ or "").splitlines() if ln.strip()), ""))
    return m.group(1) if m else ""


success = lambda input: f"{colorama.Fore.GREEN}{input}"
failure = lambda input: f"{colorama.Fore.RED}{input}"


class Terminated(BaseException):
    """SIGTERM arrived (raised by the signal handler of a file run)."""


def _on_sigterm(signum, frame):
    raise Terminated()


class ExitRequest:
    def __init__(self, code=0):
        self.code = 0 if code is None else code


class ReplError(str):
    """An error already rendered for the screen. repl_loop prints it to stderr and counts a failed entry."""


async def repl_eval(klong, p, verbose=False, set_it=True):
    """
    Evaluate one REPL entry; the value of a successful non-empty result is also stored in `it`.

    p is the text or a klongpy.source.Source. Returns None, the text to print, an
    ExitRequest, or a ReplError (the rendered error; the Python traceback follows
    it when verbose). Klong warnings raised while the entry runs are written to
    stderr as they happen.
    """
    try:
        with klong_warnings():
            r = klong(p)
        if r is not None and set_it:
            klong['it'] = r
        if r is not None:
            r = kg_write(r, klong._backend, display=False)
            if use_color(sys.stdout):
                r = success(r)
    except SystemExit as exc:
        return ExitRequest(exc.code)
    except KeyboardInterrupt:
        return ReplError("interrupted")
    except Exception as e:
        return ReplError(format_error(e, use_color(sys.stderr), verbose))

    return r


INTERRUPT_GRACE = 2.0


def interrupt_evaluation(future, thread, grace=INTERRUPT_GRACE):
    """
    Ctrl-C arrived in the main thread while `future` is evaluating in `thread`: raise
    KeyboardInterrupt there (repl_eval turns it into "interrupted") and wait for the
    result. Python code stops at its next bytecode; a long numpy call or a sleep
    finishes first, so after `grace` seconds say so and keep waiting; a second
    Ctrl-C then quits.
    """
    if not future.done() and thread is not None and thread.ident is not None:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread.ident), ctypes.py_object(KeyboardInterrupt))
    try:
        return future.result(timeout=grace)
    except concurrent.futures.TimeoutError:
        write_stderr("interrupt sent, waiting for the evaluation to stop (Ctrl-C again to quit)")
        return future.result()


def make_evaluate(klong_loop, klong_thread, verbose=False):
    """The evaluate(klong, text) for repl_loop: run repl_eval on the klong loop; Ctrl-C interrupts that evaluation."""
    def evaluate(klong, text):
        future = asyncio.run_coroutine_threadsafe(repl_eval(klong, text, verbose=verbose), klong_loop)
        try:
            return future.result()
        except KeyboardInterrupt:
            return interrupt_evaluation(future, klong_thread)
    return evaluate


def _klongpy_version():
    # The source is the one truth: an editable install's metadata can be stale.
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


PROMPT = "?> "
CONTINUATION_PROMPT = ".. "


def _prompt_text(text, color=None):
    """The prompt string, coloured and wrapped so readline does not count the escape codes."""
    # libedit moves the escape codes in front of the text (the colour is reset before
    # the text is drawn), so the continuation prompt is plain there.
    if color is None or readline is None or not sys.stdin.isatty() or "libedit" in (readline.__doc__ or ""):
        return text
    return f"\001{color}\002{text}\001{colorama.Style.RESET_ALL}\002"


def get_input(prompt=PROMPT):
    color = colorama.Fore.CYAN if prompt == CONTINUATION_PROMPT else None
    return input(_prompt_text(prompt, color))


def run_in_loop(klong_loop, coro):
    return asyncio.run_coroutine_threadsafe(coro, klong_loop).result()


def read_entry(read_line, history=None, cancel_on_blank=True):
    """
    Read one REPL entry with read_line(prompt) -> str.

    Returns the entry text, or None when there is nothing to evaluate (a blank
    line, Ctrl-C, or a cancelled continuation). A ]-command is returned as the
    single line it is. An incomplete entry (see check_complete) is continued
    on ".. " lines joined by newlines; at a continuation prompt an empty line
    (when cancel_on_blank) or Ctrl-C cancels. EOFError propagates, also in the
    middle of an entry. A multi-line entry becomes one history item.
    """
    mark = history.mark() if history is not None else 0
    try:
        s = read_line(PROMPT)
    except KeyboardInterrupt:
        print()
        return None
    if len(s) == 0:
        return None
    if s.startswith("]"):
        return s
    lines = [s]
    while check_text("\n".join(lines), f"<repl:{len(lines)}>") == "incomplete":
        try:
            nxt = read_line(CONTINUATION_PROMPT)
        except KeyboardInterrupt:
            print()
            nxt = None
        if nxt is None or (cancel_on_blank and nxt == ""):
            if history is not None:
                history.replace_entry(mark, "\n".join(lines))
            return None
        lines.append(nxt)
    text = "\n".join(lines)
    if len(lines) > 1 and history is not None:
        history.replace_entry(mark, text)
    return text


def repl_loop(klong, evaluate, read_line=get_input, sys_cmds=None, history=None,
              exit_state=None, cancel_on_blank=True, verbose=False):
    """
    The read-eval-print loop, free of any terminal: read_line(prompt) supplies
    the input and evaluate(klong, source) the result; source is a
    klongpy.source.Source named <repl:N> (N counts the entries evaluated), whose
    text is the entry, so an error on the second line of a multi-line entry
    reports line 2. Returns the number of entries that failed when the input
    ends (EOFError) or an ExitRequest comes back; exit_state["failures"] holds the
    same count. Values print to stdout; errors (a ReplError, or an exception
    that escapes evaluate) print to stderr. KeyboardInterrupt at a prompt never
    quits; one raised by evaluate propagates to the caller.
    """
    sys_cmds = create_sys_cmd_functions() if sys_cmds is None else sys_cmds
    entry = 0
    failures = 0

    def failed():
        nonlocal failures
        failures += 1
        if exit_state is not None:
            exit_state["failures"] = failures

    while True:
        try:
            s = read_entry(read_line, history, cancel_on_blank)
        except EOFError:
            print("\rbye!")
            return failures
        if s is None:
            continue
        try:
            transcript_write("\t" + s)
            if s.startswith("]"):
                r = run_sys_cmd(klong, sys_cmds, s, verbose)
            else:
                entry += 1
                r = evaluate(klong, Source(f"<repl:{entry}>", s))
            if isinstance(r, ExitRequest):
                if exit_state is not None:
                    exit_state["code"] = r.code
                return failures
            if isinstance(r, ReplError):
                failed()
                write_stderr(r)
                transcript_write(str(r))
            elif r is not None:
                print(r)
                transcript_write(str(r))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            failed()
            write_stderr(format_error(e, use_color(sys.stderr), verbose))


def setup_readline(klong, sys_cmds, environ=None):
    """
    Wire tab completion and the persistent history. Returns a History (inert
    when readline is missing or the history file is disabled or unwritable).
    """
    if readline is None:
        return History(None, None)
    history = History(readline, history_path(environ))
    try:
        readline.set_completer(Completer(klong, sys_cmds))
        readline.set_completer_delims(" \t\n\"'()[]{};,+-*/<>=~!@#%^&|\\?:")
        if "libedit" in (readline.__doc__ or ""):
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")
    except Exception:
        pass
    history.load()
    return history


async def run_in_klong(klong, s):
    try:
        with klong_warnings():
            return klong(s)
    except SystemExit as exc:
        return ExitRequest(exc.code)


def run_file(klong_loop, klong, fname, verbose=False):
    """
    Run a Klong file on the klong loop and return its result (an ExitRequest for .x).
    Errors propagate: a KlongError carries the Source named fname, so it renders
    as `fname:line:col: ...`. A file that cannot be read is a KlongSystemError.
    """
    # Add script directory to sys.path so .py/.pyf imports resolve sibling modules
    script_dir = os.path.dirname(os.path.abspath(fname))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    try:
        with open(fname, "r") as f:
            content = f.read()
    except OSError as e:
        raise KlongSystemError(f"cannot open {fname}: {e.strerror or e}") from e
    return run_in_loop(klong_loop, run_in_klong(klong, Source(fname, content)))


def run_test_file(klong, fname, report):
    """
    Run a test file one line at a time (blank lines and lines starting with ":" are skipped).
    A failing line is reported with report(exception) and does not stop the run.
    Returns the number of failed lines. Each line is evaluated as a Source named fname
    padded with newlines so an error reports its real line number in the file.
    """
    print(f"Test: {fname}")
    try:
        with open(fname, "r") as f:
            lines = f.read().splitlines()
    except OSError as e:
        raise KlongSystemError(f"cannot open {fname}: {e.strerror or e}") from e
    failed = 0
    for n, raw in enumerate(lines, 1):
        x = raw.strip()
        if len(x) == 0 or x.startswith(":"):
            continue
        print(x)
        try:
            with klong_warnings():
                klong(Source(fname, "\n" * (n - 1) + raw))
        except Exception as e:
            report(e)
            failed += 1
    return failed


SUBCOMMANDS = ('lint', 'check', 'compile', 'fpga', 'ast')

_SUBCOMMAND_HELP = """\
subcommands (kgpy SUBCOMMAND --help lists the options of each):
  lint FILE...          check programs for the classic Klong traps
  check FILE...         check programs against the static profile
  compile c FILE...     compile static-profile programs to C99
  fpga source FILE...   (experimental) write the HLS source of FPGA kernels
  fpga build [FILE...]  (experimental) build an Alveo U250 bitstream
  fpga status [FILE...] report the progress of the build
  ast FILE              print the canonical parse tree

A file with the name of a subcommand is run as kgpy ./lint.
For help, go to http://klongpy.org"""


def _leaf_parser(name, description):
    return argparse.ArgumentParser(prog=f'kgpy {name}', description=description)


def _add_int64(p):
    p.add_argument('--int64', action='store_true', help='kg_int is 64 bits wide (the default is 32)')


def _add_files(p, help_text, optional=False):
    p.add_argument('files', metavar='FILE', nargs='*' if optional else '+', help=help_text)


def _lint_parser():
    p = _leaf_parser('lint', 'Check FILEs for the classic Klong traps without running them, then exit '
                             '(1 if any error was found).')
    _add_files(p, 'the programs to check (- is stdin)')
    p.add_argument('--select', metavar='CODES', help='report only these codes (comma-separated)')
    p.add_argument('--ignore', metavar='CODES', help='leave these codes out (comma-separated)')
    p.add_argument('--define', metavar='NAMES',
                   help='names the host defines before the program runs (comma-separated)')
    p.add_argument('-v', '--verbose', action='store_true', help='enable verbose output')
    return p


def _check_parser():
    p = _leaf_parser('check', 'Check that FILEs are inside the static profile (the subset that compiles ahead of '
                              'time) without running them, then exit (1 if any is outside).')
    _add_files(p, 'the programs to check (- is stdin)')
    _add_int64(p)
    what = p.add_mutually_exclusive_group()
    what.add_argument('--streamable', action='store_true',
                      help='also say what keeps a kernel from taking one record per clock cycle')
    what.add_argument('--dump-ir', action='store_true',
                      help='evaluate the compile-time definitions and print the typed IR instead of checking')
    p.add_argument('-v', '--verbose', action='store_true', help='list the functions, globals and declarations of each file')
    return p


def _compile_c_parser():
    p = _leaf_parser('compile c', 'Compile FILEs of the static profile to C99 (<name>.c, <name>.h, klong_rt.h) in '
                                  'the directory given by -o and print the size report.')
    _add_files(p, 'the programs to compile')
    p.add_argument('-o', '--output', metavar='DIR', help='the directory to write to (default: .)')
    _add_int64(p)
    p.add_argument('--float64', action='store_true', help='kg_float is double (the default is float)')
    p.add_argument('--line-directives', action='store_true',
                   help='emit #line so a debugger steps through the Klong source')
    p.add_argument('--prefix', metavar='NAME', help='the C prefix of the public names (default: the file name)')
    p.add_argument('--dump-ir', action='store_true',
                   help='evaluate the compile-time definitions and print the typed IR')
    return p


def _add_fpga_options(p):
    p.add_argument('--clock-mhz', metavar='N', type=int, default=300, help='the kernel clock in MHz (default: 300)')
    p.add_argument('--lanes', metavar='N', type=int, default=None,
                   help='N cores on one input stream; every .param becomes an array of N values and the result '
                        'gains a leading dimension N')
    p.add_argument('--wrapped', action='store_true',
                   help='the stage A kernel (the C of compile c called once per record) instead of the streamable one')
    p.add_argument('--reassociate', action='store_true',
                   help='sum float reductions as a balanced tree (changes the rounding)')


def _fpga_source_parser():
    p = _leaf_parser('fpga source', '(experimental) Write what Vitis HLS needs to make a kernel of each .entry of '
                                    'FILEs of the static profile (C, <name>_<entry>_top.cpp, test bench, tcl, '
                                    'manifest.json) in the directory given by -o. There is no --float64: the card '
                                    'has float32 only.')
    _add_files(p, 'the programs to turn into kernels')
    p.add_argument('-o', '--output', metavar='DIR', help='the directory to write to (default: .)')
    _add_int64(p)
    p.add_argument('--prefix', metavar='NAME', help='the C prefix of the public names (default: the file name)')
    _add_fpga_options(p)
    p.add_argument('--synth', action='store_true',
                   help='run C synthesis on $KLONG_FPGA_BUILD (user@host; unset: here) and print the estimates')
    return p


def _fpga_build_parser():
    p = _leaf_parser('fpga build', '(experimental) Build one Alveo U250 bitstream holding every .entry of FILEs on '
                                 '$KLONG_FPGA_BUILD (user@host), cached by source; it takes hours and runs detached. '
                                 'It generates the HLS source itself (see kgpy fpga source). With no FILE: --wait for the build in progress.')
    _add_files(p, 'the programs to build', optional=True)
    _add_int64(p)
    _add_fpga_options(p)
    p.add_argument('--wait', action='store_true', help='wait until the build finishes and report it')
    p.add_argument('--force', action='store_true', help='build again although the image exists')
    p.add_argument('--emu', action='store_true', help='a hardware emulation image (never loaded on the card)')
    return p


def _fpga_status_parser():
    p = _leaf_parser('fpga status', '(experimental) Report the progress of the build in progress or, with FILEs, of the '
                                  'build of their image on $KLONG_FPGA_BUILD (user@host).')
    _add_files(p, 'the programs whose image to report', optional=True)
    _add_int64(p)
    _add_fpga_options(p)
    p.add_argument('--emu', action='store_true', help='the hardware emulation image')
    return p


def _ast_parser():
    p = _leaf_parser('ast', 'Print the canonical parse tree of FILE and exit.')
    p.add_argument('file', metavar='FILE', help='the program to parse')
    return p


_LEAVES = {
    ('lint',): _lint_parser, ('check',): _check_parser, ('ast',): _ast_parser,
    ('compile', 'c'): _compile_c_parser, ('fpga', 'source'): _fpga_source_parser,
    ('fpga', 'build'): _fpga_build_parser, ('fpga', 'status'): _fpga_status_parser,
}
_GROUPS = {'compile': {'c': 'compile static-profile programs to C99'},
           'fpga': {'source': '(experimental) write the HLS source of FPGA kernels', 'build': '(experimental) build an Alveo U250 bitstream',
                  'status': 'report the progress of the build'}}


def _group_parser(group):
    p = argparse.ArgumentParser(prog=f'kgpy {group}')
    sub = p.add_subparsers(dest='target', metavar='{' + ','.join(_GROUPS[group]) + '}', required=True)
    for name, text in _GROUPS[group].items():
        sub.add_parser(name, help=text)
    return p


def _check_fpga_options(p, args):
    if args.clock_mhz < 1:
        p.error("--clock-mhz must be a positive number of MHz")
    if args.lanes is not None and args.lanes < 1:
        p.error("--lanes must be 1 or more")
    if args.wrapped and (args.lanes is not None or args.reassociate):
        p.error("--lanes and --reassociate belong to the streamable kernel, not to --wrapped")


def subcommand_main(argv):
    """Run `kgpy SUBCOMMAND ...`; argv starts with the subcommand's name. Returns the exit status."""
    key = (argv[0],)
    if argv[0] in _GROUPS:
        if len(argv) < 2 or argv[1] not in _GROUPS[argv[0]]:
            _group_parser(argv[0]).parse_args(argv[1:2])
        key = (argv[0], argv[1])
        argv = argv[2:]
    else:
        argv = argv[1:]
    p = _LEAVES[key]()
    args = p.parse_intermixed_args(argv)
    bits = 64 if getattr(args, 'int64', False) else 32
    from klongpy import tools
    if key == ('ast',):
        return tools.ast_main(args.file)
    if key == ('lint',):
        return tools.lint_main(args.files, args.select, args.ignore, args.verbose, args.define)
    if key == ('check',):
        if args.dump_ir:
            return tools.dump_ir_main(args.files, bits)
        return tools.check_main(args.files, bits, args.verbose, streamable=args.streamable)
    if key == ('compile', 'c'):
        return tools.emit_main(args.files, args.output, bits, 64 if args.float64 else 32, args.line_directives,
                               args.prefix, args.dump_ir)
    if key == ('fpga', 'source'):
        _check_fpga_options(p, args)
        return tools.emit_hls_main(args.files, args.output, bits, args.prefix, args.clock_mhz, args.synth,
                                   wrapped=args.wrapped, lanes=args.lanes, reassociate=args.reassociate)
    _check_fpga_options(p, args)
    from klongpy.hw.build import hw_build_main
    status = key == ('fpga', 'status')
    return hw_build_main(args.files, bits, args.clock_mhz, args.lanes, False if status else args.force, status,
                         False if status else args.wait, args.emu, wrapped=args.wrapped,
                         reassociate=args.reassociate)


def run_parser():
    """The parser of the run form: kgpy [FILE] and the options of 0.7, --strict and --traceback."""
    parser = argparse.ArgumentParser(
        prog='kgpy',
        description='KlongPy REPL: with no FILE, an interactive session; with FILE, runs it (arguments for the '
                    'program follow --).',
        epilog=_SUBCOMMAND_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--version', action='version', version=f'kgpy {_klongpy_version()}',
                        help='print the version and exit')
    parser.add_argument('-e', '--expr', help='evaluate expression, no interactive mode')
    parser.add_argument('-l', '--load', help='load program from file')
    parser.add_argument('-s', '--server', help='start the IPC server', type=str)
    parser.add_argument('-t', '--test', help='test program from file')
    parser.add_argument('-v', '--verbose', help='enable verbose output; errors also print the Python traceback', action="store_true")
    parser.add_argument('--traceback', help='print the Python traceback with an error (what -v does, without the other output)',
                        action="store_true")
    parser.add_argument('-d', '--debug', help='enable debug mode', action="store_true")
    parser.add_argument('--strict', metavar='N', type=int, choices=[0, 1, 2], default=0,
                        help='strict level: 0 (the default), 1 or 2; --strict alone means 1. Written --strict=N')
    parser.add_argument('--backend', help='set array backend (torch needs the torch extra)', type=str.lower,
                        choices=sorted(set(list_backends()) | {'numpy', 'torch'}))
    parser.add_argument('--device', help='set backend device (torch only)', type=str)
    parser.add_argument('filename', nargs='?', help='filename to be run if no flags are specified')
    return parser


def main():
    """Main entry point for the KlongPy REPL."""
    if len(sys.argv) > 1 and sys.argv[1] in SUBCOMMANDS:
        sys.exit(subcommand_main(sys.argv[1:]))
    if '--' in sys.argv:
        index = sys.argv.index('--')
        main_args = sys.argv[:index]
        extras = sys.argv[index+1:]
    else:
        main_args = sys.argv
        extras = []

    parser = run_parser()

    # --strict[=N]: the bare flag means level 1 and never takes the next word
    # (which may be the file name) for its value, except a lone 0, 1 or 2, so
    # that --strict 2 means level 2 instead of level 1 and a file named 2.
    argv = []
    bare_strict = False
    for a in main_args[1:]:
        if bare_strict and a in ('0', '1', '2'):
            argv[-1] = f'--strict={a}'
        else:
            argv.append('--strict=1' if a == '--strict' else a)
        bare_strict = a == '--strict'
    args = parser.parse_args(argv)
    if args.backend == 'torch' and 'torch' not in list_backends():
        parser.error('the torch backend needs PyTorch, which is not installed: pip install "klongpy[torch]"')

    # Default to the numpy backend: fast and predictable for general array work
    # (and it never surprises non-autograd .kg scripts with MPS-tensor issues).
    # The torch backend (GPU + exact autograd via :>) is opt-in: --backend torch.
    if args.backend is None:
        args.backend = 'numpy'

    if args.debug:
        print("args: ", args)

    show_traceback = args.verbose or args.traceback

    def report(e):
        """Print an exception to stderr: rendered if it is a KlongError, one `internal error` line if not."""
        write_stderr(format_error(e, use_color(sys.stderr), show_traceback))

    def interrupted():
        write_stderr("interrupted")
        sys.exit(130)

    def finish_interrupted(text, code):
        """Ctrl-C or SIGTERM while a finished file run waits: say so; the common exit below closes what is open."""
        write_stderr(text)
        return code

    def start_interpreter():
        """
        The interpreter with the system context every run needs: the loops and
        the close event behind .timer, .srv and .cli (.system), the help
        database, and the host's environment and arguments. -e, a file run, -l
        and the REPL all start here.
        """
        try:
            klong, loops = create_repl(debug=args.debug, backend=args.backend, device=args.device, strict=args.strict)
        except ValueError as exc:
            parser.error(str(exc))
        klong['.helpdb'] = []
        klong['.os.env'] = dict(os.environ)
        klong['.os.argv'] = extras if extras else []
        klong['.system']['traceback'] = show_traceback
        return klong, loops

    if args.expr:
        klong, loops = start_interpreter()
        shutdown_event = klong['.system']['closeEvent']
        try:
            try:
                with klong_warnings():
                    result = klong(Source("<expr>", args.expr))
                if result is not None:
                    # writing the result can fail too: report it like an evaluation error
                    print(kg_write(result, klong._backend, display=False))
            except KeyboardInterrupt:
                interrupted()
            except Exception as e:
                report(e)
                sys.exit(1)
        finally:
            # -e evaluates the expression and exits: a timer, a server or a
            # connection it opened is closed with the process
            shutdown_event.trigger()
            cleanup_repl(loops, debug=args.debug)
        return

    klong, loops = start_interpreter()
    io_loop, _, _, klong_loop, klong_thread, _ = loops
    shutdown_event = klong['.system']['closeEvent']

    run_repl = False
    exit_code = None

    try:
        if args.server:
            r = klong(f".srv({args.server})")
            if r == 0:
                print("Failed to start server")
        elif args.test:
            # a test run never starts the REPL; it exits after the last line, 1 if any line failed
            if run_test_file(klong, args.test, report):
                exit_code = 1

        if args.filename:
            if args.verbose:
                print(f"Running: {args.filename}")
            try:
                signal.signal(signal.SIGTERM, _on_sigterm)
            except (ValueError, OSError):   # not the main thread, or no SIGTERM here
                pass
            result = run_file(klong_loop, klong, args.filename, verbose=args.verbose)
            if isinstance(result, ExitRequest):
                exit_code = result.code

            if exit_code is None:
                # the program is over; what it printed must be visible even when the
                # process now stays up for a server, a connection or a timer
                sys.stdout.flush()
                try:
                    wait_while_alive(klong['.system'])
                except KeyboardInterrupt:
                    exit_code = finish_interrupted("interrupted", 130)
                except Terminated:
                    exit_code = finish_interrupted("terminated", 143)
                else:
                    exit_signal = klong['.system']['exitSignal']
                    if exit_signal.code is not None:
                        exit_code = exit_signal.code
        elif not args.test:
            run_repl = True
    except KeyboardInterrupt:
        interrupted()
    except Terminated:
        write_stderr("terminated")
        sys.exit(143)
    except Exception as e:
        report(e)
        exit_code = 1
        run_repl = False

    if args.debug:
        print("run_repl: ", run_repl)

    if run_repl:
        console_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(console_loop)
        if args.load:
            if args.verbose:
                print(f"Loading: {args.load}")
            try:
                result = run_file(klong_loop, klong, args.load, verbose=args.verbose)
            except KeyboardInterrupt:
                interrupted()
            except Exception as e:
                # a failing load is rendered and the process exits
                report(e)
                exit_code = 1
                run_repl = False
            else:
                if isinstance(result, ExitRequest):
                    exit_code = result.code
                    run_repl = False
        if run_repl:
            exit_state = {"code": None, "failures": 0}
            colorama.init(autoreset=True, strip=True if os.environ.get("NO_COLOR") else None)
            backend_name = klong._backend.name
            device_name = str(klong._backend.device) if hasattr(klong._backend, 'device') else None
            show_repl_header(backend_name, device_name, args.server)
            sys_cmds = create_sys_cmd_functions()
            history = setup_readline(klong, sys_cmds)
            evaluate = make_evaluate(klong_loop, klong_thread, verbose=show_traceback)

            def exit_from_callback(code):
                # .x in a timer or an IPC callback while the prompt waits for input: the
                # main thread cannot be woken from its read, so end the process here
                try:
                    history.save()
                    sys.stdout.flush()
                    sys.stderr.flush()
                finally:
                    os._exit(code)
            klong['.system']['exitSignal'].handler = exit_from_callback
            try:
                repl_loop(klong, evaluate, get_input, sys_cmds, history, exit_state,
                          cancel_on_blank=sys.stdin.isatty(), verbose=show_traceback)
            except KeyboardInterrupt:
                print("\rbye!")
                exit_state["code"] = 130
            finally:
                history.save()
            console_loop.close()
            if exit_state["code"] is not None:
                exit_code = exit_state["code"]
            elif exit_state["failures"] and not sys.stdin.isatty():
                # piped input: a failed entry makes the run fail (an interactive session ends 0)
                exit_code = 1

    shutdown_event.trigger()

    cleanup_repl(loops, debug=args.debug)

    if exit_code is not None:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
