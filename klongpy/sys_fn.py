
import errno
import os
import random
import subprocess
import sys
import time

from .core import KGChannel, KGChannelDir, is_empty, kg_read, kg_write, safe_eq


def eval_sys_append_channel(x):
    """

        .ac(x)                                          [Append-Channel]

        See [Output-Channel].

    """
    return KGChannel(open(x, "a"), KGChannelDir.OUTPUT)


def eval_sys_close_channel(x):
    """

        .cc(x)                                           [Close-Channel]

        Close the input or output channel "x", returning []. Closing an
        already closed channel has no effect. A channel will be closed
        automatically when no variable refers to it and it is not the
        current From or To Channel.

    """
    if not x.raw.closed:
        x.raw.close()


def eval_sys_display(klong, x):
    """

        .d(x)                                                  [Display]

        See [Write].

    """
    klong['.sys.cout'].raw.write(kg_write(x, display=True))


def eval_sys_delete_file(x):
    """

        .df(x)                                             [Delete-File]

        Delete the file specified in the string "x". When the file
        cannot be deleted (non-existent, no permission, etc), signal
        an error.

    """
    if os.path.exists(x):
        try:
            os.remove(x)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise RuntimeError("file could not be deleted: {x}")
            raise e
    else:
        raise RuntimeError("file does not exist: {x}")


def eval_sys_evaluate(klong, x):
    """

        .E(x)                                                 [Evaluate]

        Evaluate the Klong program contained in the string "x" and
        return its result. This is a direct interface to the Klong
        system, e.g. .E("a::123");a will yield 123.

    """
    return klong.exec(x)[-1]


def eval_sys_from_channel(klong, x):
    """

        .fc(x)                                            [From-Channel]

        .fc selects input channel "x" for reading and .tc selects
        output channel "x" for writing, i.e. these functions select a
        new From and To Channel, respectively. All input/output will
        be redirected to the given channel. Both function will return
        the channel that was previously in effect.

        When a false value (0,[],"") is passed to these functions, they
        restore the default From or To Channel (.cin or .cout).

    """
    if safe_eq(x,0) or is_empty(x):
        x = klong['.cin']
    if x.channel_dir != KGChannelDir.INPUT:
        raise RuntimeError("output channel cannot be used input")
    o = klong['.sys.cin']
    klong['.sys.cin'] = x
    return o


def eval_sys_flush(x):
    """

        .fl()                                                    [Flush]

        Make sure that all output sent to the To Channel is actually
        written to the associated file or device ("flush" the channel).

    """
    if x.channel_dir != KGChannelDir.OUTPUT:
        raise RuntimeError("input channel cannot be flushed")
    x.raw.flush()


def eval_sys_input_channel(x):
    """

        .ic(x)                                           [Input-Channel]

        Open the file named in the string "x", link it to an input
        channel, and return that channel. Opening a non-existent file
        is an error.

    """
    if os.path.exists(x):
        try:
            return KGChannel(open(x, 'r'), KGChannelDir.INPUT)
        except IOError:
            raise RuntimeError("file could not be opened: {x}")
    else:
        raise RuntimeError("file does not exist: {x}")


def eval_sys_load(klong, x):
    """

        .l(x)                                                     [Load]

        Load the content of the file specified in the string "x" as if
        typed in at the interpreter prompt.

        Klong will try the names "x", and a,".kg", in all directories
        specified in the KLONGPATH environment variable. Directory names
        in KLONGPATH are separated by colons.

        When KLONGPATH is undefined, it defaults to ".:lib".

        A program can be loaded from an absolute or relative path
        (without a prefix from KLONGPATH) by starting "x" with a "/"
        or "." character.

        .l will return the last expression evaluated, i.e. it can be
        used to load the value of a single expression from a file.

    """
    if os.path.exists(x):
        try:
            # TODO: support path defaults as mentioned above
            with open(x, "r") as f:
                return klong.exec(f.read())[-1]
        except IOError:
            raise RuntimeError("file could not be opened: {x}")
    else:
        raise RuntimeError("file does not exist: {x}")


def eval_sys_more_input(klong):
    """

        .mi(x)                                              [More-Input]

        This function returns 1, if the From Channel is not exhausted
        (i.e. no reading beyond the EOF has been attempted on that
        channel). When no more input is available, it returns 0.

        This is a "negative EOF" function.

    """
    x = klong['.sys.cin']
    if x.channel_dir != KGChannelDir.INPUT:
        raise RuntimeError("output channel cannot be used for input")
    return not x.at_eof


def eval_sys_module(klong, x):
    """

        .module(x)                                              [Module]

        Delimit a module. See MODULES documentation.

    """
    if safe_eq(x, 0) or is_empty(x):
        klong.stop_module()
    else:
        klong.start_module(x)


def eval_sys_output_channel(x):
    """

        .oc(x)                                          [Output-Channel]

        Both of these functions open a file named in the string "x",
        link it to an output channel, and return that channel. The
        difference between them is that .oc truncates any existing
        file and .ac appends to it.

    """
    return KGChannel(open(x,"w"), KGChannelDir.OUTPUT)


def eval_sys_process_clock(klong):
    """

        .pc()                                            [Process-Clock]

        Return the process time consumed by the Klong interpreter so
        far. The return value is in seconds with a fractional part
        whose resolution depends on the operating environment and may
        be anywhere between 50Hz and 1MHz.

        The program {[t0];t0::.pc();x@[];.pc()-t0} measures the process
        time consumed by the nilad passed to it.

        This function is not avaliable in the Plan 9 port of Klong.

    """
    return time.time() - klong.process_start_time()


def eval_sys_print(klong, x):
    """

        .p(x)                                                    [Print]

        Pretty-print the object "x" (like Display) and then print a
        newline sequence. .p("") will just print a newline.

    """
    klong['.sys.cout'].raw.write(str(x)+"\n")


def eval_sys_random_number():
    """

        .rn()                                            [Random-Number]

        Return a random number x, such that 0 <= x < 1.

    """
    return random.random()


def eval_sys_read(klong):
    """

        .r()                                                      [Read]

        Read a single data object from the currently selected input port
        and return it. The object being read may be an atom or a list.
        When it is a dictionary or list, the input may span multiple
        lines.

    """
    f = klong['.sys.cin']
    k = f.raw.tell()
    r = f.raw.read()
    if r == '':
        f.at_eof = True
        return None
    else:
        i,a = kg_read(r,i=0,module=klong.current_module())
        f.raw.seek(k+i,0)
        return a


def eval_sys_read_line(klong):
    """

        .rl()                                                [Read-Line]

        Read a line from the From Channel and return it as a string.
        If there is a line separator at the end of the line, it will
        be stripped from the string.

    """
    f = klong['.sys.cin']
    r = f.raw.readline()
    if r == '':
        f.at_eof = True
    return r.rstrip()


def eval_sys_read_string(klong, x):
    """

        .rs(x)                                             [Read-String]

        .rs is like .r, but reads its input from the string "x". It is
        intended for the converting sequentialized compound data objects,
        such as lists, arrays, and dictionaries, back to their internal
        forms.

    """
    return kg_read(x, i=0, module=klong.current_module())[1]


def eval_sys_system(x):
    """

        .sys(a)                                                 [System]

        Pass the command in the string "x" to the operating system for
        execution and return the exit code of the command. On a Unix
        system, the command would be executed as

        sh -c "command"

        and an exit code of zero would indicate success.

    """
    with subprocess.Popen([x], stdout=subprocess.PIPE) as proc:
        return proc.returncode


def eval_sys_to_channel(klong, x):
    """
        .tc(x)                                              [To-Channel]

        See [From-Channel]
    """
    if safe_eq(x,0) or is_empty(x):
        x = klong['.cout']
    if x.channel_dir != KGChannelDir.OUTPUT:
        raise RuntimeError("input channel cannot be a to channel")
    c = klong['.sys.cout']
    klong['.sys.cout'] = x
    return c


def eval_sys_write(klong, x):
    """

        .w(x)                                                    [Write]

        .d and .w both write "x" to the currently selected output port.
        However, .w writes a "readable" representation of the given
        object and .d pretty-prints the object. The "readable" output
        is suitable for reading by .r.

        For most types of object there is no difference. Only strings
        and characters are printed in a different way:

        Object       | .d(Object) | .w(Object)
        ----------------------------------------
        0cx          |  x         | 0cx
        "test"       |  test      | "test"
        "say ""hi""\"|  say "hi"  | "say ""hi""\"

        NOTE: the \" was added above to work in docstring

        For some objects, there is no readable representation, including
        functions, operators, the undefined object, and the "end of file"
        object. A symbolic representation will be printed for those:
        :nilad, :monad, :dyad, :triad, :undefined, :eof.

        None of these functions terminates its output with a newline
        sequence. Use .p (Print) to do so.

    """
    klong['.sys.cout'].raw.write(kg_write(x))


def eval_sys_exit(x):
    """

        .x(x)                                                     [Exit]

        Terminate the Klong interpreter, returning "success" to the
        operating system, if "x" is false (0, [], "") and "failure",
        if "x" is not false.

    """
    if safe_eq(x, 0) or is_empty(x):
        sys.exit(0)
    sys.exit(1)


def create_system_functions():
    def _get_name(s):
        i = s.index('.')
        return s[i:i+s[i:].index('(')]

    registry = {}

    m = sys.modules[__name__]
    for x in filter(lambda n: n.startswith("eval_sys_"), dir(m)):
        fn = getattr(m,x)
        registry[_get_name(fn.__doc__)] = fn

    return registry
