
import errno
import importlib
import importlib.util
import inspect
import os
import subprocess
import sys
import time
from inspect import Parameter

import numpy

from .core import (KGChannel, KGChannelDir, KGLambda, KGSym, KlongException,
                   is_dict, is_empty, is_list, kg_asarray, kg_read, kg_write, np,
                   reserved_fn_args, reserved_fn_symbol_map, safe_eq, safe_inspect)


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
    r = kg_write(x, display=True)
    klong['.sys.cout'].raw.write(r)
    return r


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
                raise RuntimeError(f"file could not be deleted: {x}")
            raise e
    else:
        raise RuntimeError(f"file does not exist: {x}")


def eval_sys_evaluate(klong, x):
    """

        .E(x)                                                 [Evaluate]

        Evaluate the Klong program contained in the string "x" and
        return its result. This is a direct interface to the Klong
        system, e.g. .E("a::123");a will yield 123.

    """
    return klong(x)


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
            raise RuntimeError(f"file could not be opened: {x}")
    else:
        raise FileNotFoundError(f"file does not exist: {x}")


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
    # if not (os.path.isfile(x) or os.path.isfile(os.path.join(x,".kg"))):
    alt_dirs = [x for x in str((os.environ.get('KLONGPATH') or ".:lib")).split(':') if x]
    for ad in alt_dirs:
        adx = os.path.join(ad,x)
        if os.path.isfile(adx):
            x = adx
            break
        if not adx.endswith(".kg"):
            adx = os.path.join(adx + ".kg")
            if os.path.isfile(adx):
                x = adx
                break

    if not os.path.isfile(x):
        raise FileNotFoundError(f"file does not exist: {x}")

    try:
        with open(x, "r") as f:
            # This is a "hack" to keep the load operation in the same context it was called in
            #   The interpeter pushes a context when it calls a function, so here
            #   we pop it, run the load in the original context, and push the context back on
            #   so that the interpreter can pop it's temporary context off as normal.
            ctx = klong._context.pop()
            try:
                r = klong(f.read())
            finally:
                klong._context.push(ctx)
            return r
    except IOError as e:
        print(e)
        raise RuntimeError(f"file could not be opened: {x}")


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
    o = kg_write(x, display=True)
    klong['.sys.cout'].raw.write(o+"\n")
    return o


def import_directory_module(x):
    """
    Import a module from a directory path.
    """
    x = os.path.realpath(x)
    if x.endswith("__init__.py"):
        x = os.path.dirname(x)
    if os.path.isfile(os.path.join(x, "__init__.py")):
        pardir = os.path.dirname(x)
        sys.path.insert(0, pardir)
        module_name = os.path.basename(os.path.normpath(x))
        try:
            return importlib.import_module(module_name)
        finally:
            sys.path.pop(0)
    else:
        raise FileNotFoundError(f"Not a valid Python module (missing __init__.py): {x}")


def import_file_module(x):
    """
    Import a module from a file path.
    """
    location = os.path.abspath(x)
    module_name = os.path.splitext(os.path.basename(x))[0]
    spec = importlib.util.spec_from_file_location(module_name, location=location)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = location
    module.__package__ = None
    module.__name__ = module_name
    spec.loader.exec_module(module)
    sys.modules[module.__name__] = module
    sys.path.insert(0, os.path.dirname(x))
    return module


def import_module_from_sys(x):
    """
    Import a module already present in sys.modules.
    """
    spec = importlib.util.find_spec(x)
    if spec is not None:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise RuntimeError(f"module could not be imported: {x}")


def _handle_import(item):
    """
    Handles the import of a single item into KlongPy.
    """
    if not callable(item):
        return item

    try:
        if isinstance(item, numpy.ufunc):
            n_args = item.nin
            if n_args <= len(reserved_fn_args):
                item = KGLambda(item, args=reserved_fn_args[:n_args])
        else:
            args = safe_inspect(item, follow_wrapped=True)
            if 'args' in args:
                item = KGLambda(item, args=None, wildcard=True)
                n_args = 3
            else:
                args = [k for k,v in args.items() if (v.kind == Parameter.POSITIONAL_OR_KEYWORD and v.default == Parameter.empty) or (v.kind == Parameter.POSITIONAL_ONLY)]
                n_args = len(args)
                # if there are kwargs, then .pyc() must be used to call this function to override them
                if 'klong' in args:
                    n_args -= 1
                    assert n_args <= len(reserved_fn_args)
                    item = KGLambda(item, args=reserved_fn_args[:n_args], provide_klong=True)
                elif n_args <= len(reserved_fn_args):
                    item = KGLambda(item, args=reserved_fn_args[:n_args])
    except Exception:
        if hasattr(item, "__class__") and hasattr(item.__class__, '__module__') and item.__class__.__module__ == "builtins":
            # LOOK AWAY. You didn't see this.
            # example: datetime(year, month, day[, hour[, minute[, second[, microsecond[,tzinfo]]]]])
            # be sure to count the args before the first optional args starting with "["
            # if there are kwargs, then .pyc() must be used to call this function to override them
            signature_line = next((x.strip() for x in item.__doc__.split("\n") if x.strip()), None)
            if signature_line:
                args = signature_line.split("[")[0].split("(")[1].split(",")
                n_args = len(args)
                if n_args <= len(reserved_fn_args):
                    item = KGLambda(item, args=reserved_fn_args[:n_args])
            else:
                raise
        else:
            raise

    if n_args > len(reserved_fn_args):
        # TODO: this should be logged
        # print(f".py: {name} - too many paramters: use .pyc() to call this function")
        return lambda x,y: item(*x,**y)
    return item


def _import_module(klong, x, from_set=None):
    """
    Import a python module in to the current context.  All methods exposed in the module
    are loaded into the current context space and callable from within KlongPy.

    If the string "x" refers to a directory, KlongPy will append __init__.py
    to determine if the directory is a Python module and attempt to load the module.

    If the string "x" ends with a .py, then KlongPy will attempt to load this file
    as a Python module.

    from_set is a set of symbols to import from the module.
    If from_set is None, then all methods are imported.

    """
    try:
        if os.path.isdir(x) or x.endswith("__init__.py"):
            module = import_directory_module(x)
        elif os.path.isfile(x):
            module = import_file_module(x)
        elif x in sys.modules:
            module = sys.modules[x]
        else:
            module = import_module_from_sys(x)

        export_items = module.__dict__.get("klongpy_exports") or module.__dict__
        ffn = (lambda p: p[0] in from_set) if from_set is not None else (lambda p: not p[0].startswith("__"))

        ctx = klong._context.pop()
        try:
            for name, item in filter(ffn, export_items.items()):
                try:
                    klong[name] = _handle_import(item)
                except Exception as e:
                    # TODO: this should be logged
                    print(f"failed to import function: {name}", e)
        finally:
            klong._context.push(ctx)

        return 1
    except Exception as e:
        print(f".py: {e}")
        raise RuntimeError(f"failed to load module: {x}")


def eval_sys_python(klong, x):
    """

        .py(x)                                                    [Python]

        Import a python module in to the current context.  All methods exposed in the module
        are loaded into the current context space and callable from within KlongPy.

        If the string "x" refers to a directory, KlongPy will append __init__.py
        to determine if the directory is a Python module and attempt to load the module.

        If the string "x" ends with a .py, then KlongPy will attempt to load this file
        as a Python module.

        Example:

            Consider a python module called Greetings that exports the "hello" method such as:

                def hello():
                    print("Hello, World!")

            The following will load the "greetings" module from the Python modules in the same
            way that 'import hello' would do.  Then we can call 'hello'

                .py("greetings")
                hello() --> "Hello, World!"

            Additionally, a full path to a module can be specified.  This can be useful for
            when creating custom modules that don't need to be installed into Python modules.

                .py("<path>/<to>/greetings")

    """
    if not isinstance(x,str):
        raise RuntimeError("module name must be a string")
    return _import_module(klong, x, from_set=None)


def eval_sys_python_call(klong, x, y, z):
    """

        .pyc(x, y, z)                                      [Python-Call]

        Call a python function or class function with args and kwargs.

        This is a utility function to allow for Klong programs to call Python classes,
        or functions, with positional and keyword arguments.

        x may be a list with the first element reference the object symbol and the second reference the function name to call.
        x may also be a symbol that references the function name to call.

        y is either a list of or a single positional arguments to pass to the function.
        z is a dictionary of keyword arguments to pass to the function.

        Example:

            Python objects may be directly called:

                .pyc(obj;[1 2 3];:{"a":1,"b":2,"c":3})

            or a method on the object may be called:

                .pyc(obj,"method";[1 2 3];:{"a":1,"b":2,"c":3})

            if the python object name is snake case, then the quote is not needed:

            Here, parse_date is a function in the iso8601 module:

                .pyf("iso8601";"parse_date")

            and it can be called via:

                .pyc("parse_date";,"2020-01-01";:{})

    """
    if is_list(x):
        if not isinstance(x[0],object):
            raise KlongException("x must be a list of [object;function]")
        if not isinstance(x[1],str):
            raise KlongException("function name must be a string")
        klazz = klong[x[0]] if isinstance(x[0],KGSym) else x[0]
        if not hasattr(klazz, x[1]):
            raise KlongException(f"function {x[1]} not found")
        f = getattr(klazz, x[1])
        if not callable(f):
            return f
    else:
        f = klong[KGSym(x)] if isinstance(x,str) else x
        if not callable(f):
            return f
    if not is_list(y):
        y = [y]
    if not is_dict(z):
        raise KlongException("z must be a dictionary")
    if isinstance(f, KGLambda):
        ctx = {reserved_fn_symbol_map[k]:v for k,v in zip(reserved_fn_args[:f.get_arity()], y)}
        r = f.call_with_kwargs(klong, ctx, kwargs=z)
    else:
        r = f(*y, **z)
    # TODO: hack to convert generators into lists - until we can handle generators in Klong
    # determine if the result is a generator and if it is then convert it to a list
    if inspect.isgenerator(r):
        r = list(r)
    return r


def eval_sys_python_attribute(klong, x, y):
    """

        .pya(x, y)                                    [Python-Attribute]

    """
    if not isinstance(x,object):
        raise KlongException("x must be a Python object")
    if not isinstance(y,str):
        raise KlongException("attribute name must be a string")
    klazz = klong[x] if isinstance(x,KGSym) else x
    if not hasattr(klazz, y):
        raise KlongException(f"attribute {y} not found")
    return _handle_import(getattr(klazz, y))


def eval_sys_python_from(klong, x, y):
    """

        .pyf(x, y)                                         [Python-From]

        Import selected sub-modules from a python module in to the current context.

        [See .py() for detials on importing python modules]

        Example:

            To import a single sub-module:

            .pyf("numpy"; "sqrt")
            sqrt(4) --> 2.0

            Multiple sub-modules can be imported at once:

            .pyf("numpy"; ["sqrt";"sin"])
            sqrt(4) --> 2.0
            sin(1) --> 0.8414709848078965

    """
    if not isinstance(x,str):
        raise RuntimeError("module name must be a string")
    if isinstance(y,str):
        y = [y]
    if not (is_list(y) and all(map(lambda p: isinstance(p,str), y))) or isinstance(y,str):
        raise RuntimeError("from list entry must be a string")
    return _import_module(klong, x, from_set=set(y))


def eval_sys_random_number():
    """

        .rn()                                            [Random-Number]

        Return a random number x, such that 0 <= x < 1.

    """
    return np.random.random()


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


def eval_sys_read_lines(klong):
    """

        .rls()                                              [Read-Lines]

        Read all lines from the From Channel and return it as an array.
        If there is a line separator at the end of the line, it will
        be stripped from the string.

    """
    f = klong['.sys.cin']
    r = f.raw.readlines()
    f.at_eof = True
    return kg_asarray(r)


def eval_sys_read_string(klong, x):
    """

        .rs(x)                                             [Read-String]

        .rs is like .r, but reads its input from the string "x". It is
        intended for the converting sequentialized compound data objects,
        such as lists, arrays, and dictionaries, back to their internal
        forms.

    """
    return kg_read(x, i=0, module=klong.current_module(), read_neg=True)[1]


def eval_sys_system(x):
    """

        .sys(x)                                                 [System]

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
    r = kg_write(x)
    klong['.sys.cout'].raw.write(r)
    return x


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
