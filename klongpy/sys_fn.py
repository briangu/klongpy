
import errno
import importlib
import importlib.abc
import importlib.util
import inspect
import os
import re
import subprocess
import sys
import time
from inspect import Parameter

import numpy

from .errors import (KlongNameError, KlongRangeError, KlongSystemError, KlongTypeError, compat,
                     python_error)
from .lexer import Source
from .core import (KGChannel, KGChannelDir, KGLambda, KGSym, KlongException,
                   bknp, is_dict, is_empty, is_list, kg_read_array, kg_write,
                   reserved_fn_args, reserved_fn_symbol_map, safe_eq, safe_inspect)


def _to_display_value(x, backend):
    """Convert backend tensors to numpy for cleaner display."""
    # Convert backend arrays (tensors) to numpy
    if backend.is_backend_array(x):
        return backend.to_numpy(x)
    # Handle numpy arrays with tensors inside (object arrays)
    if isinstance(x, numpy.ndarray) and x.dtype == object:
        return numpy.array([_to_display_value(item, backend) for item in x], dtype=object)
    # Handle lists with tensors
    if isinstance(x, list):
        return [_to_display_value(item, backend) for item in x]
    return x


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

        Display the object "x". Backend arrays are converted to numpy for cleaner output.
        Use .bkd() for raw backend-specific display.

    """
    x = _to_display_value(x, klong._backend)
    r = kg_write(x, klong._backend, display=True)
    klong['.sys.cout'].raw.write(r)
    return r


def eval_sys_backend_display(klong, x):
    """

        .bkd(x)                                        [Backend-Display]

        Display the object "x" in raw backend format (tensors shown as-is).

    """
    r = kg_write(x, klong._backend, display=True)
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
                raise compat(KlongSystemError, RuntimeError)(f"file could not be deleted: {x}")
            raise e
    else:
        raise compat(KlongSystemError, RuntimeError)(f"file does not exist: {x}")


def eval_sys_evaluate(klong, x):
    """

        .E(x)                                                 [Evaluate]

        Evaluate the Klong program contained in the string "x" and
        return its result. This is a direct interface to the Klong
        system, e.g. .E("a::123");a will yield 123.

        Evaluation happens at GLOBAL scope, so top-level definitions
        (`name::...`) persist as globals, even when .E is called from
        inside a function (e.g. runtime code generation). Without this,
        new vars would land in the (transient) calling frame and be lost.

    """
    kc = klong._context
    # Temporarily drop pushed local frames so global assignment (which
    # targets _context[0]) lands in the real global frame, then restore.
    stashed = []
    while len(kc._context) > kc._min_ctx_count:
        stashed.append(kc._context.popleft())
    try:
        return klong(x)
    finally:
        for d in reversed(stashed):
            kc._context.appendleft(d)


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
        raise compat(KlongTypeError, RuntimeError)("output channel cannot be used input")
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
        raise compat(KlongTypeError, RuntimeError)("input channel cannot be flushed")
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
            raise compat(KlongSystemError, RuntimeError)(f"file could not be opened: {x}")
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
                r = klong(Source(x, f.read()))
            finally:
                klong._context.push(ctx)
            return r
    except IOError as e:
        if isinstance(e, KlongException):
            # an error of the program that was loaded: it has its own position
            raise
        print(e)
        raise compat(KlongSystemError, RuntimeError)(f"file could not be opened: {x}") from e


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
        raise compat(KlongTypeError, RuntimeError)("output channel cannot be used for input")
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
        newline sequence. Backend arrays are converted to numpy for cleaner output.
        Use .bkp() for raw backend-specific print.

    """
    x = _to_display_value(x, klong._backend)
    o = kg_write(x, klong._backend, display=True)
    klong['.sys.cout'].raw.write(o+"\n")
    return o


_FMT_SPEC_RE = re.compile(r"-?\d+(\.\d+)?")


def _fmt_error(template, msg, cls=KlongRangeError):
    return cls(f".fmt: {msg} in template {template!r}")


def _fmt_parse(template):
    """Split a template into literal strings and (index, spec) tuples; index is None for {}."""
    parts = []
    lit = []
    i, n = 0, len(template)
    while i < n:
        c = template[i]
        if c == "{":
            if template[i+1:i+2] == "{":
                lit.append("{")
                i += 2
                continue
            j = template.find("}", i)
            if j < 0:
                raise _fmt_error(template, "unmatched '{'")
            body = template[i+1:j]
            idx_s, colon, spec = body.partition(":")
            if idx_s != "" and not idx_s.isdigit():
                raise _fmt_error(template, f"bad placeholder '{{{body}}}'")
            if colon and not _FMT_SPEC_RE.fullmatch(spec):
                raise _fmt_error(template, f"bad format '{spec}' in placeholder '{{{body}}}'")
            if lit:
                parts.append("".join(lit))
                lit = []
            parts.append((int(idx_s) if idx_s else None, spec if colon else None))
            i = j + 1
        elif c == "}":
            if template[i+1:i+2] == "}":
                lit.append("}")
                i += 2
                continue
            raise _fmt_error(template, "unmatched '}'")
        else:
            lit.append(c)
            i += 1
    if lit:
        parts.append("".join(lit))
    return parts


def _fmt_render(klong, template, args):
    if not isinstance(template, str):
        raise KlongTypeError(f".fmt: template must be a string, got {type(template).__name__}")
    from .dyads import _e_dyad_format2
    backend = klong._backend
    args = _to_display_value(args, backend)
    args = list(args) if (is_list(args) and not isinstance(args, str)) else [args]
    parts = _fmt_parse(template)
    holes = [p for p in parts if not isinstance(p, str)]
    auto = [h for h in holes if h[0] is None]
    if auto and len(auto) != len(holes):
        raise _fmt_error(template, "cannot mix {} and {N} placeholders")
    if auto and len(auto) != len(args):
        raise _fmt_error(template, f"{len(auto)} placeholder(s) but {len(args)} argument(s)")
    out = []
    nxt = 0
    for p in parts:
        if isinstance(p, str):
            out.append(p)
            continue
        idx, spec = p
        if idx is None:
            idx, nxt = nxt, nxt + 1
        if idx >= len(args):
            raise _fmt_error(template, f"index {idx} out of range for {len(args)} argument(s)")
        v = args[idx]
        if spec is None:
            out.append(kg_write(v, backend, display=True))
        elif "." in spec:
            if isinstance(v, bool) or not isinstance(v, (int, float, numpy.integer, numpy.floating)):
                raise _fmt_error(template, f"format '{spec}' needs a number, got {kg_write(v, backend, display=True)!r}", KlongTypeError)
            out.append(_e_dyad_format2(float(spec), float(v), backend))
        else:
            out.append(_e_dyad_format2(int(spec), kg_write(v, backend, display=True), backend))
    return "".join(out)


def eval_sys_format(klong, x, y):
    """

        .fmt(x;y)                                               [Format]

        Return the template string "x" with its placeholders replaced by
        the arguments in "y". If "y" is a list (not a string), its elements
        are the arguments; otherwise "y" is the single argument. To pass one
        list as the only argument, wrap it: .fmt("{}";,[1 2 3]).

        Placeholders: "{}" takes the next argument, "{0}", "{1}" take by
        index (the two styles may not be mixed), "{{" and "}}" are literal
        braces. A format may follow a colon, as in "{:8}", "{0:-8}" and
        "{:5.2}", and is applied like Format2 (n$value): a positive width
        pads on the right, a negative width on the left, and a fractional
        format such as 5.2 formats a number (an integer is converted to a
        real first). Arguments are rendered as Display does.

        Examples:   .fmt("Spot: {}";3.5)       -->  "Spot: 3.5"
                    .fmt("{1} {0}";[;1;2])     -->  "2 1"
                    .fmt("{:5.2}";3)           -->  "    3.00"

    """
    return _fmt_render(klong, x, y)


def eval_sys_print_format(klong, x, y):
    """

        .pf(x;y)                                          [Print-Format]

        Like Format (.fmt), but also write the result and a newline
        sequence to the current output channel. Return the string.

        Example:   .pf("a={}";1)   writes  a=1

    """
    r = _fmt_render(klong, x, y)
    klong['.sys.cout'].raw.write(r + "\n")
    return r


def eval_sys_backend_print(klong, x):
    """

        .bkp(x)                                          [Backend-Print]

        Pretty-print the object "x" in raw backend format (tensors shown as-is)
        and then print a newline sequence.

    """
    o = kg_write(x, klong._backend, display=True)
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


class _SiblingFinder(importlib.abc.MetaPathFinder):
    """A sys.meta_path finder that resolves imports to a specific directory.

    Installed temporarily during import_file_module's exec_module call so
    that `import sibling` inside the loaded file finds the correct sibling
    in the same directory.
    """
    def __init__(self, module_dir):
        self.module_dir = module_dir

    def find_spec(self, fullname, path, target=None):
        if '.' in fullname:
            return None
        # Don't shadow modules that are already loaded — only resolve
        # genuinely new imports from the sibling directory.
        if fullname in sys.modules:
            return None
        candidate = os.path.join(self.module_dir, fullname + '.py')
        if os.path.isfile(candidate):
            return importlib.util.spec_from_file_location(fullname, location=candidate)
        return None


# path -> already-imported module, so repeated .pyf/.py of the same file
# does not re-execute it (see import_file_module).
_FILE_MODULE_CACHE = {}


def import_file_module(x):
    """
    Import a module from a file path.

    Adds the module's directory to sys.path (so spawned subprocesses and
    later imports can find siblings) and installs a temporary meta-path
    finder so that sibling imports resolve during exec_module even if a
    same-named module already exists in sys.modules.
    """
    location = os.path.abspath(x)
    # Cache by path so re-importing the same file (e.g. .pyf(f;"a") then
    # .pyf(f;"b")) returns the SAME module instead of re-executing it — which
    # would re-run module-level side effects (e.g. re-register prometheus metrics).
    cached = _FILE_MODULE_CACHE.get(location)
    if cached is not None:
        return cached
    module_name = os.path.splitext(os.path.basename(x))[0]
    module_dir = os.path.dirname(location)
    spec = importlib.util.spec_from_file_location(module_name, location=location)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = location
    module.__package__ = None
    module.__name__ = module_name
    # Add directory to sys.path so spawned subprocesses can resolve siblings.
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    # Temporary finder ensures sibling imports resolve even if a same-named
    # module is already cached in sys.modules from a different directory.
    finder = _SiblingFinder(module_dir)
    sys.meta_path.insert(0, finder)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.meta_path.remove(finder)
    sys.modules[module.__name__] = module
    _FILE_MODULE_CACHE[location] = module
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
    raise compat(KlongSystemError, RuntimeError)(f"module could not be imported: {x}")


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
            sig_args = safe_inspect(item, follow_wrapped=True)
            if 'args' in sig_args:
                item = KGLambda(item, args=None, wildcard=True)
                n_args = 3
            else:
                # Get required args (no default)
                required_args = [k for k,v in sig_args.items() if (v.kind == Parameter.POSITIONAL_OR_KEYWORD and v.default == Parameter.empty) or (v.kind == Parameter.POSITIONAL_ONLY)]
                # Get optional args (have default)
                optional_args = [k for k,v in sig_args.items() if v.kind == Parameter.POSITIONAL_OR_KEYWORD and v.default != Parameter.empty]
                # Use required args count, but if there are optional args and no required args,
                # use wildcard mode so the function can accept 0-3 args
                if not required_args and optional_args:
                    item = KGLambda(item, args=None, wildcard=True)
                    n_args = 3
                else:
                    args = required_args
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
        def call_with_lists(x, y):
            return item(*x, **y)
        call_with_lists._klong_user = True
        return call_with_lists
    if isinstance(item, KGLambda):
        # user Python: an exception it raises is a KlongPythonError
        item.user = True
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

            # For from_set imports, also check for lazy-loaded attributes not in __dict__
            # (e.g., numpy.random in numpy 2.x)
            if from_set is not None:
                for name in from_set:
                    if name not in export_items and hasattr(module, name):
                        try:
                            item = getattr(module, name)
                            klong[name] = _handle_import(item)
                        except Exception as e:
                            print(f"failed to import function: {name}", e)
        finally:
            klong._context.push(ctx)

        return 1
    except Exception as e:
        print(f".py: {e}")
        raise compat(KlongSystemError, RuntimeError)(f"failed to load module: {x}")


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
        raise compat(KlongTypeError, RuntimeError)("module name must be a string")
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
            raise KlongTypeError("x must be a list of [object;function]")
        if not isinstance(x[1],str):
            raise KlongTypeError("function name must be a string")
        klazz = klong[x[0]] if isinstance(x[0],KGSym) else x[0]
        if not hasattr(klazz, x[1]):
            raise KlongNameError(f"function {x[1]} not found")
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
        raise KlongTypeError("z must be a dictionary")
    try:
        if isinstance(f, KGLambda) and f._wildcard:
            # a wildcard function takes whatever it is given (its arity is 0:
            # building the frame from the arity would drop every argument)
            r = f.fn(klong, *y, **z) if f._provide_klong else f.fn(*y, **z)
        elif isinstance(f, KGLambda):
            ctx = {reserved_fn_symbol_map[k]:v for k,v in zip(reserved_fn_args[:f.get_arity()], y)}
            r = f.call_with_kwargs(klong, ctx, kwargs=z)
        else:
            r = f(*y, **z)
    except KlongException:
        raise
    except Exception as e:
        raise python_error(e) from e
    # TODO: hack to convert generators into lists - until we can handle generators in Klong
    # determine if the result is a generator and if it is then convert it to a list
    if inspect.isgenerator(r):
        r = list(r)
    return r


def eval_sys_python_attribute(klong, x, y):
    """

        .pya(x, y)                                    [Python-Attribute]

        Return the attribute named "y" of the Python object "x".

    """
    if not isinstance(x,object):
        raise KlongTypeError("x must be a Python object")
    if not isinstance(y,str):
        raise KlongTypeError("attribute name must be a string")
    klazz = klong[x] if isinstance(x,KGSym) else x
    if not hasattr(klazz, y):
        raise KlongNameError(f"attribute {y} not found")
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
        raise compat(KlongTypeError, RuntimeError)("module name must be a string")
    if isinstance(y,str):
        y = [y]
    if not (is_list(y) and all(map(lambda p: isinstance(p,str), y))) or isinstance(y,str):
        raise compat(KlongTypeError, RuntimeError)("from list entry must be a string")
    return _import_module(klong, x, from_set=set(y))


def eval_sys_backend_fn(klong, x):
    """

        .bkf(x)                                           [Backend-Function]

        Import functions from the current backend's array module.
        This is similar to .pyf() but uses backend-aware functions that
        work with both numpy and torch backends.

        When using the torch backend, these functions preserve gradient
        tracking for autograd.

        Example:

            .bkf("exp")
            exp(1.0) --> 2.718...

            .bkf(["exp";"sin";"cos"])
            sin(1.0) --> 0.841...

        Common functions available: exp, sin, cos, tan, tanh, sqrt, abs,
        log, log10, floor, ceil, round

    """
    if isinstance(x, str):
        x = [x]
    if not (is_list(x) and all(map(lambda p: isinstance(p, str), x))):
        raise compat(KlongTypeError, RuntimeError)("function name(s) must be a string or list of strings")

    backend = klong._backend
    ctx = klong._context.pop()
    try:
        for fn_name in x:
            if hasattr(backend.np, fn_name):
                fn = getattr(backend.np, fn_name)
                klong[fn_name] = _handle_import(fn)
            else:
                raise compat(KlongNameError, RuntimeError)(f"Backend does not have function: {fn_name}")
    finally:
        klong._context.push(ctx)
    return None


def eval_sys_random_number():
    """

        .rn()                                            [Random-Number]

        Return a random number x, such that 0 <= x < 1.

    """
    return bknp.random.random()


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
        i,a = kg_read_array(r, 0, klong._backend, module=klong.current_module())
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
    return klong._backend.kg_asarray(r)


def eval_sys_read_string(klong, x):
    """

        .rs(x)                                             [Read-String]

        .rs is like .r, but reads its input from the string "x". It is
        intended for the converting sequentialized compound data objects,
        such as lists, arrays, and dictionaries, back to their internal
        forms.

    """
    _, a = kg_read_array(x, 0, klong._backend, module=klong.current_module(), read_neg=True)
    return a


def eval_sys_system(x):
    """

        .sys(x)                                                 [System]

        Pass the command in the string "x" to the operating system for
        execution and return the exit code of the command. On a Unix
        system, the command would be executed as

        sh -c "command"

        and an exit code of zero would indicate success. The command's
        stdout and stderr are inherited from the interpreter process. On
        Windows the command is run through cmd.

    """
    return subprocess.run(x, shell=True).returncode


def eval_sys_to_channel(klong, x):
    """
        .tc(x)                                              [To-Channel]

        See [From-Channel]
    """
    if safe_eq(x,0) or is_empty(x):
        x = klong['.cout']
    if x.channel_dir != KGChannelDir.OUTPUT:
        raise compat(KlongTypeError, RuntimeError)("input channel cannot be a to channel")
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
    r = kg_write(x, klong._backend)
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


def eval_sys_strict(klong, x=None):
    """

        .strict()                                                 [Strict]
        .strict(x)

        Set the strict level and return it. .strict() sets level 1;
        "x" may be 0, 1 or 2. Level 0 is the default: see .unsafe().

        Level 1: malformed input that level 0 only warns about (input
        left after a statement, an unterminated string or list, a
        missing operand) is a syntax error, and so is an assignment
        whose left side is not a name. Inside a function, assigning to
        a name that is not a declared local or an existing variable is
        an error, and so is reading a name that is not defined.

        Level 2: in addition, reading an undefined name is an error
        everywhere, as is reading a local variable before it has been
        assigned and passing more arguments than a function takes; a
        quoted symbol (:a) is always the symbol, never the value of a
        variable of that name.

        The rule for :: is the same at every level: when a variable of
        that name exists in any scope, it is assigned there. Otherwise,
        at top level the variable is created; inside a function the
        name becomes local to the call (level 0) or the assignment is
        an error (level 1 and 2). To change a variable from a function,
        define it at top level first. To create one from inside a
        function, use .E.

        Text is checked for syntax at the level in force when it is
        read, and code runs under the level in force when it runs.

        Example:
            .strict()
            counter::0
            increment::{counter::counter+1}  :" OK - counter exists"
            f::{newvar::42}
            f()  :" ERROR - undefined variable: newvar"
            g::{[t];t::42;t}
            g()  :" OK - t is declared local"

    """
    level = 1 if x is None else x
    if isinstance(level, bool) or not isinstance(level, (int, numpy.integer)):
        raise KlongTypeError("the strict level must be 0, 1 or 2")
    if level not in (0, 1, 2):
        raise KlongRangeError("the strict level must be 0, 1 or 2")
    klong.strict = int(level)
    return klong.strict


def eval_sys_unsafe(klong):
    """

        .unsafe()                                                 [Unsafe]

        Set strict level 0, the default, and return 0. See .strict().

        At level 0, assigning to a name that does not exist inside a
        function makes the name local to that call. It does not create
        a variable outside the function; use .E for that.

        Example:
            .unsafe()
            f::{newvar::42}
            f()     :" 42; newvar is local to the call"
            newvar  :" still undefined"

    """
    klong.strict = 0
    return 0


def create_system_functions():
    def _get_name(s):
        i = s.index('.')
        return s[i:i+s[i:].index('(')]

    registry = {}

    m = sys.modules[__name__]
    for x in filter(lambda n: n.startswith("eval_sys_"), dir(m)):
        fn = getattr(m,x)
        registry[_get_name(fn.__doc__)] = fn

    # .strict takes zero or one argument (ticket 050): a wildcard lambda is
    # handed whichever of x, y, z the call supplied.
    registry['.strict'] = KGLambda(eval_sys_strict, wildcard=True)

    return registry
