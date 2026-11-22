
import importlib
import importlib.abc
import importlib.util
import inspect
import os
import subprocess
import sys
from inspect import Parameter

import numpy

from .errors import (KlongNameError, KlongSystemError, KlongTypeError, compat,
                     python_error)
from .core import KGChannel, KGChannelDir  # noqa: F401 (historical channel exports)
from .core import (KGLambda, KGSym, KlongException,
                   is_dict, is_list, kg_write,
                   reserved_fn_args, reserved_fn_symbol_map, safe_inspect)


def eval_sys_backend_display(klong, x):
    """

        .bkd(x)                                        [Backend-Display]

        Display the object "x" in raw backend format (tensors shown as-is).

    """
    r = kg_write(x, klong._backend, display=True)
    klong._k.call(".d", r)      # the engine's own output channel
    return r


def eval_sys_backend_print(klong, x):
    """

        .bkp(x)                                          [Backend-Print]

        Pretty-print the object "x" in raw backend format (tensors shown as-is)
        and then print a newline sequence.

    """
    o = kg_write(x, klong._backend, display=True)
    klong._k.call(".p", o)
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
            # safe_inspect uses this sentinel for builtins without a signature.
            if sig_args == {"args": []} or any(p.kind == Parameter.VAR_POSITIONAL for p in sig_args.values()):
                item = KGLambda(item, args=None, wildcard=True)
                n_args = 3
            else:
                # Get required args (no default)
                required_args = [k for k,v in sig_args.items() if (v.kind == Parameter.POSITIONAL_OR_KEYWORD and v.default == Parameter.empty) or (v.kind == Parameter.POSITIONAL_ONLY)]
                # Get optional args (have default)
                optional_args = [k for k,v in sig_args.items() if v.kind == Parameter.POSITIONAL_OR_KEYWORD and v.default != Parameter.empty]
                # A function with optional positional parameters takes
                # whichever of x, y and z the call supplies (wildcard): with
                # the count of the required ones only, linspace(0;1;5) lost
                # its third argument (also for .pyc).
                # The wildcard function is handed every argument the call
                # writes, however many (linspace(0;1;4;0) passes the fourth).
                if optional_args and 'klong' not in required_args and len(required_args) <= len(reserved_fn_args):
                    item = KGLambda(item, args=None, wildcard=True)
                    # displayed by the arguments it needs (:monad for fft(a, n=None))
                    item.args = [reserved_fn_symbol_map[a] for a in reserved_fn_args[:len(required_args)]]
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


# the optional modules of the package and the extra that provides what they import
_EXTRA_MODULES = {"klongpy.db": "db", "klongpy.web": "web", "klongpy.ws": "ws"}


def _module_extra(name, error):
    """The extra to install when importing the KlongPy module name failed for want of a package, else None."""
    extra = _EXTRA_MODULES.get(name)
    missing = getattr(error, "name", None)
    if extra and missing and not missing.startswith("klongpy"):
        return extra
    return None


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

        return 1
    except ImportError as e:
        extra = _module_extra(x, e)
        if extra:
            raise compat(KlongSystemError, RuntimeError)(
                f"{x} needs the {extra} extra, which is not installed (no module named {e.name}): "
                f'pip install "klongpy[{extra}]"') from None
        print(f".py: {e}")
        raise compat(KlongSystemError, RuntimeError)(f"failed to load module: {x}")
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

            Consider a Python package called greetings that exports the "hello" function:

                def hello():
                    print("Hello, World!")

            This loads the package, as 'import greetings' does, and then calls 'hello',
            which prints Hello, World!

                .py("greetings")
                hello()

            Additionally, a full path to a module can be specified.  This can be useful for
            when creating custom modules that don't need to be installed into Python modules.

                .py("<path>/<to>/greetings")

    """
    if not isinstance(x,str):
        raise compat(KlongTypeError, RuntimeError)("module name must be a string")
    return _import_module(klong, x, from_set=None)


def eval_sys_python_call(klong, x, y, z):
    """

        .pyc(x;y;z)                                        [Python-Call]

        Call a python function or class function with args and kwargs.

        This is a utility function to allow for Klong programs to call Python classes,
        or functions, with positional and keyword arguments.

        x may be a list: the object symbol, then the name of the method to call.
        x may also be a symbol that references the function name to call.

        y is a list of positional arguments, or a single positional argument.
        z is a dictionary of keyword arguments to pass to the function.

        Example:

            Python objects may be directly called:

                .pyc(obj;[1 2 3];:{["a" 1] ["b" 2] ["c" 3]})

            or a method on the object may be called:

                .pyc(obj,"method";[1 2 3];:{["a" 1] ["b" 2] ["c" 3]})

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

        .pya(x;y)                                     [Python-Attribute]

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

        .pyf(x;y)                                          [Python-From]

        Import the named functions of a Python module into the current context.

        [See .py() for details on importing python modules]

        Example:

            To import one function:

            .pyf("numpy"; "sqrt")
            sqrt(4) --> 2.0

            To import several:

            .pyf("numpy"; ["sqrt" "sin"])
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

            .bkf(["exp" "sin" "cos"])
            sin(1.0) --> 0.841...

        Common functions available: exp, sin, cos, tan, tanh, sqrt, abs,
        log, log10, floor, ceil, round

    """
    if isinstance(x, str):
        x = [x]
    if not (is_list(x) and all(map(lambda p: isinstance(p, str), x))):
        raise compat(KlongTypeError, RuntimeError)("function name(s) must be a string or list of strings")

    backend = klong._backend
    for fn_name in x:
        if hasattr(backend.np, fn_name):
            fn = getattr(backend.np, fn_name)
            klong[fn_name] = _handle_import(fn)
        else:
            raise compat(KlongNameError, RuntimeError)(f"Backend does not have function: {fn_name}")
    return None


def eval_sys_read_lines(klong):
    """

        .rls()                                              [Read-Lines]

        Read all lines from the From Channel and return it as an array.
        If there is a line separator at the end of the line, it will
        be stripped from the string.

    """
    k = klong._k
    r = []
    while k.call(".mi"):
        r.append(k.call(".rl"))     # the engine's input channel; .rl strips the line separator
    if r:
        r.pop()                     # .mi stays true until a read meets the end: the last read gave nothing
    return klong._backend.kg_asarray(r)


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
