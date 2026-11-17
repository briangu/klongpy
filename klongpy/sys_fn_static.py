"""
.extern and .entry: declaring the functions a compiled program takes from its
host and the functions the host calls (docs/static-profile.md).

Under KlongPy the two declarations make a static program runnable as it
stands, with Python functions standing in for what the target provides:

    klong = KlongInterpreter(strict=2)
    klong.externs["read.adc"] = lambda channel: 0.3      # the stand-in
    klong('.extern("read.adc";"int -> float")')           # binds the name read.adc
    klong('read.adc(0)')                                  # 0.3

``klong.externs`` is an ExternTable: a dict from extern name to a Python
callable, which also records what the program declared (``declared``,
``entries``). The stand-in is looked up at each call, so it can be bound
before or after the ``.extern`` line runs and replaced at any time. Arguments
and result are checked against the signature on every call, with the same
signature parser the static checker uses (klongpy/static/signature.py), so a
program that passes a float where the target expects an int fails on the host
too.
"""
import sys

from .errors import (KlongArityError, KlongError, KlongNameError, KlongShapeError, KlongSystemError, KlongTypeError,
                     python_error)
from .static.signature import (EntryDecl, ExternDecl, Mismatch, SignatureError, Ty, check_entry_name, check_extern_name,
                               conform, fixed_spec, is_fixed, parse_signature, parse_type)
from .types import KGBody, KGCall, KGFn, KGLambda, KGSym, KLONG_UNDEFINED, fn_metadata


class ParamDecl:
    """``.param(name; type[; default])``: a global that is an input register of the compiled program."""
    __slots__ = ("name", "ty", "default", "span", "source")

    def __init__(self, name, ty, default=0, span=None, source=None):
        self.name = name
        self.ty = ty
        self.default = default
        self.span = span
        self.source = source

    def __repr__(self):
        return f"<param {self.name}: {self.ty}>"


class ExternTable(dict):
    """
    ``klong.externs``: extern name -> Python stand-in.

        declared   {name: ExternDecl}   what .extern has declared (name, signature, module)
        entries    [EntryDecl]          what .entry has declared, in order
        params     {name: ParamDecl}    what .param has declared (tunable registers)

    A stand-in is called with one positional argument per parameter of the
    signature: a Python int for ``int``, a float for ``float``, a str for
    ``str``, a numpy int64 or float64 array of the declared shape for an
    array. It returns a value of the result type (anything for ``void``).
    """
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.declared = {}
        self.entries = []
        self.params = {}        # name -> ParamDecl, what .param has declared

    def entry(self, name, signature=None):
        """The EntryDecl of name (with that signature text, when a function has several); KeyError if none."""
        for e in self.entries:
            if e.name == name and (signature is None or str(e.signature) == str(parse_signature(signature))):
                return e
        raise KeyError(name)


def extern_table(klong):
    """The ExternTable of an interpreter (klong.externs)."""
    table = getattr(klong, "externs", None)
    if not isinstance(table, ExternTable):
        table = ExternTable(table or {})
        klong.externs = table
    return table


def _plural(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _signature(which, text, allow_str):
    try:
        return parse_signature(text, allow_str=allow_str)
    except SignatureError as e:
        where = f" (at column {e.span[0] + 1} of \"{text}\")" if isinstance(text, str) and text else ""
        raise KlongTypeError(f"{which} signature: {e.message}{where}",
                             hint="a signature is \"params -> result\", for example \"int float[64] -> float\"") from None


def _strings(which, args, names):
    for a, what in zip(args, names):
        if type(a) is not str:
            raise KlongTypeError(f"{which}: the {what} must be a string")


def call_extern(klong, decl, args):
    """Call the stand-in of an extern with argument values, checking both ways against the signature."""
    sig = decl.signature
    name = decl.name
    if len(args) != sig.arity:
        raise KlongArityError(f"{name} is declared \"{sig}\" and takes {_plural(sig.arity, 'argument')}, got {len(args)}")
    fn = extern_table(klong).get(name)
    if fn is None:
        raise KlongSystemError(f"extern {name} has no stand-in on the host",
                               hint=f"bind a Python function before calling it: klong.externs[\"{name}\"] = ...")
    values = []
    for i, (ty, a) in enumerate(zip(sig.params, args)):
        try:
            values.append(conform(ty, a, f"argument {i + 1} of {name}"))
        except Mismatch as m:
            raise (KlongShapeError if m.kind == "shape" else KlongTypeError)(f"{m.message} (declared \"{sig}\")") from None
    try:
        r = fn(*values)
    except KlongError:
        raise
    except Exception as e:
        raise python_error(e) from e
    try:
        r = conform(sig.result, r, f"the result of the stand-in of {name}")
    except Mismatch as m:
        raise (KlongShapeError if m.kind == "shape" else KlongTypeError)(f"{m.message} (declared \"{sig}\")") from None
    if r is None:
        return KLONG_UNDEFINED          # void: there is no value
    if sig.result.shape:
        return klong._backend.kg_asarray(r)
    return r


def _bind(decl):
    """The Klong value an extern name is bound to: calling it calls the stand-in."""
    n = decl.signature.arity
    if n <= 3:
        # x, y, z: a wildcard lambda is handed whichever of them the call supplied
        def call(klong, *args):
            return call_extern(klong, decl, args)
        fn = KGLambda(call, wildcard=True)
        fn.name = decl.name
        return fn
    # more than three parameters: a function with named parameters
    # whose body reads them from its own frame
    params = tuple(KGSym(f"arg{i + 1}") for i in range(n))

    def call(klong):
        ctx = klong._context
        return call_extern(klong, decl, [ctx[p] for p in params])
    inner = KGLambda(call)
    inner.name = decl.name
    return KGFn(KGBody([KGCall(inner, [], 0)], (), None, params), None, n)


def _define_global(klong, sym, value):
    """
    Define a global from inside a system function. The call has pushed a
    frame, and at strict level 1 and above a new name cannot be created
    from inside a function, so the frames above the global one are set
    aside for the assignment (as .E does).
    """
    kc = klong._context
    stashed = []
    while len(kc._context) > kc._min_ctx_count + 1:
        stashed.append(kc._context.popleft())
    try:
        klong[sym] = value
    finally:
        for d in reversed(stashed):
            kc._context.appendleft(d)


def eval_sys_extern(klong, *args):
    """

        .extern(x;y)                                             [Extern]
        .extern(x;y;z)

        Declare a function the host provides. "x" is its name, a
        string; "y" is its signature, a string of the form
        "params -> result"; "z" is the module it is imported from
        when the program is compiled to wasm ("env" when omitted).

        A type is int, float, an array of them with fixed dimensions
        (int[8], float[16,241]) or, for a parameter, str. The result
        may also be void. Parameters are separated by spaces and may
        be absent: "-> int" is a function of no arguments.

        Under KlongPy the name is bound to the Python function that
        the host program registered as its stand-in,
        klong.externs["name"]. Arguments and result are checked
        against the signature on every call; a call without a
        stand-in is an error. A compiled program calls the C function
        of that name, with "." written as "_" (read.adc is read_adc).

        Returns the signature in its canonical form.

        Examples:   .extern("read.adc";"int -> float")
                    .extern("write.pwm";"int float -> void")
                    .extern("fill";"float[64] -> float[64]";"dsp")
                    read.adc(0)

    """
    if len(args) not in (2, 3):
        raise KlongArityError(f".extern takes 2 or 3 arguments, got {len(args)}")
    _strings(".extern", args, ("name", "signature", "module"))
    name, text = args[0], args[1]
    module = args[2] if len(args) == 3 else "env"
    sig = _signature(".extern", text, True)
    bad = check_extern_name(name)
    if bad:
        raise KlongTypeError(f".extern: {bad}")
    table = extern_table(klong)
    old = table.declared.get(name)
    sym = KGSym(name)
    if old is not None:
        if old.signature != sig or old.module != module:
            raise KlongTypeError(f".extern: {name} is already declared \"{old.signature}\" (module {old.module})")
        return str(sig)
    try:
        klong._context[sym]
    except KeyError:
        pass
    else:
        raise KlongNameError(f".extern: {name} is already defined in the program; an extern cannot replace it")
    table.declared[name] = ExternDecl(name, sig, module)
    _define_global(klong, sym, _bind(table.declared[name]))
    return str(sig)


def eval_sys_entry(klong, x, y):
    """

        .entry(x;y)                                               [Entry]

        Mark the function named "x" (a string) as an entry point of
        a compiled program and fix the types of its arguments and
        result with the signature "y" (see .extern; str is not
        allowed). The function must already be defined and must take
        as many arguments as the signature has parameters. A function
        may have several entries with different shapes.

        Under KlongPy this only checks the declaration and records it
        in klong.externs.entries; the function is called as usual.

        Returns the signature in its canonical form.

        Example:   step::{x*2}
                   .entry("step";"float -> float")

    """
    _strings(".entry", (x, y), ("name", "signature"))
    sig = _signature(".entry", y, False)
    bad = check_entry_name(x)
    if bad:
        raise KlongTypeError(f".entry: {bad}")
    try:
        fn = klong._context[KGSym(x)]
    except KeyError:
        raise KlongNameError(f".entry: the function {x} is not defined",
                             hint="define the function before the .entry line") from None
    if not isinstance(fn, KGFn) or isinstance(fn, KGCall) or fn.args is not None or isinstance(fn.a, KGLambda):
        raise KlongTypeError(f".entry: {x} is not a function defined in the program",
                             hint="an entry is a function written name::{...} or name(a;b)::{...}")
    body = fn.a
    if type(body) is KGBody and body.params is not None:
        arity = len(body.params)
    else:
        arity = fn_metadata(fn).max_arg
        if arity is None:
            arity = fn.arity
    if arity != sig.arity:
        raise KlongArityError(f".entry: {x} takes {_plural(arity, 'argument')} but the signature \"{sig}\" "
                              f"has {_plural(sig.arity, 'parameter')}")
    table = extern_table(klong)
    if not any(e.name == x and e.signature == sig for e in table.entries):
        table.entries.append(EntryDecl(x, sig))
    return str(sig)


def _scalar_ty(which, text):
    try:
        ty = parse_type(text)
    except SignatureError as e:
        raise KlongTypeError(f"{which}: {e.message}", hint='a type is int, float or fixed<W,I>, for example "fixed<32,16>"') from None
    return ty


def _fixed_value(klong_value, spec_elem, what):
    """A number, a list or a FixedArray as a FixedArray of the type spec_elem (truncating toward minus infinity)."""
    import numpy as np
    from .static import fixed as fx
    spec = fx.spec_of(spec_elem)
    v = klong_value
    if isinstance(v, fx.FixedArray):
        return fx.FixedArray(fx.convert(v.raw, v.spec, spec), spec)
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    a = np.asarray(v)
    if a.dtype.kind in "iub":
        return fx.FixedArray(fx.from_int(a, spec), spec)
    if a.dtype.kind == "f":
        return fx.FixedArray(fx.from_float(a, spec), spec)
    raise KlongTypeError(f"{what} must be a number or an array of numbers, got {type(klong_value).__name__}")


def eval_sys_fixed(klong, x, y):
    """

        .fixed(x;y)                                               [Fixed]

        Convert the number or array "y" to the fixed-point type named by
        the string "x", for example "fixed<32,16>" (32 bits, 16 of them
        integer bits: the value of the raw integer r is r*2^-16). An int
        converts exactly, a float and a wider fixed-point value are
        truncated toward minus infinity, and the result wraps to the
        width of the type.

        This is how a fixed-point constant is written. Under KlongPy it
        returns a fixed-point value that the static profile understands:
        programs that use fixed-point numbers run through
        klongpy.static.call_entry, not as plain Klong.

        Examples:   .fixed("fixed<32,16>";0.5)
                    .fixed("fixed<64,32>";price)
                    .fixed("fixed<8,4>";[1 2 3])

    """
    _strings(".fixed", (x,), ("type",))
    ty = _scalar_ty(".fixed", x)
    if not is_fixed(ty.elem) or ty.shape:
        raise KlongTypeError(".fixed: the type must be fixed<W,I>, for example \"fixed<32,16>\"")
    return _fixed_value(y, ty.elem, ".fixed: the value")


def eval_sys_float(klong, x):
    """

        .float(x)                                                 [Float]

        Convert the fixed-point number or array "x" (or an int) to a
        float: the value raw*2^-(W-I). The result is exact when the type
        has at most 53 bits; compiled code rounds it to float32.

        Examples:   .float(.fixed("fixed<32,16>";0.5))  -->  0.5

    """
    import numpy as np
    from .static import fixed as fx
    if isinstance(x, fx.FixedArray):
        r = x.to_float()
        return float(r) if r.ndim == 0 else r
    a = np.asarray(x)
    if a.dtype.kind in "iubf":
        r = a.astype(np.float64)
        return float(r) if r.ndim == 0 else r
    raise KlongTypeError(".float: the argument must be a fixed-point number or an array of numbers")


def _default_of(name, ty, default):
    import numpy as np
    from .static import fixed as fx
    if is_fixed(ty.elem):
        return _fixed_value(default, ty.elem, f".param {name}: the default")
    if isinstance(default, fx.FixedArray):
        raise KlongTypeError(f".param: the default of {name} is fixed-point but the type is {ty}")
    if isinstance(default, (bool, np.bool_)):
        default = int(default)
    if ty.elem == "int":
        if not isinstance(default, (int, np.integer)):
            raise KlongTypeError(f".param: the default of {name} must be an int")
        return int(default)
    if not isinstance(default, (int, float, np.integer, np.floating)):
        raise KlongTypeError(f".param: the default of {name} must be a number")
    return float(default)


def eval_sys_param(klong, *args):
    """

        .param(x;y)                                               [Param]
        .param(x;y;z)

        Declare the global named by the string "x" as a tunable
        parameter of a compiled program: an input register that the host
        writes between runs and that is constant during one. "y" is its
        type, "int", "float" or "fixed<W,I>" (a scalar); "z" is the
        value it starts with (0 when omitted).

        The name is defined as an ordinary global. The compiled program
        may read it and may not assign it. Array shapes and loop bounds
        cannot depend on it: a parameter may bound a window only as "use
        the first p of a fixed maximum". Set it between runs with
        klongpy.static.set_param(klong;name;value) (the hardware binding
        .hw.set does the same on the card).

        Returns the type in its canonical form.

        Examples:   .param("window";"int";4)
                    .param("stop";"fixed<32,16>";0.5)

    """
    import numpy as np
    if len(args) not in (2, 3):
        raise KlongArityError(f".param takes 2 or 3 arguments, got {len(args)}")
    _strings(".param", args[:2], ("name", "type"))
    name = args[0]
    ty = _scalar_ty(".param", args[1])
    if ty.shape or ty.elem not in ("int", "float") and not is_fixed(ty.elem):
        raise KlongTypeError(f".param: {name} must be a scalar int, float or fixed<W,I>, not {ty}")
    bad = check_extern_name(name)
    if bad:
        raise KlongTypeError(f".param: {bad.replace('an extern', 'a parameter')}")
    table = extern_table(klong)
    default = _default_of(name, ty, args[2] if len(args) == 3 else 0)
    old = table.params.get(name)
    if old is not None:
        if old.ty != ty:
            raise KlongTypeError(f".param: {name} is already declared as {old.ty}")
        return str(ty)
    sym = KGSym(name)
    try:
        klong._context[sym]
    except KeyError:
        pass
    else:
        raise KlongNameError(f".param: {name} is already defined in the program",
                             hint="a parameter is defined by .param itself: remove the other definition")
    table.params[name] = ParamDecl(name, ty, default)
    _define_global(klong, sym, default)
    return str(ty)


def create_system_functions_static():
    def _get_name(s):
        i = s.index(".")
        return s[i : i + s[i:].index("(")]

    registry = {}

    m = sys.modules[__name__]
    for x in filter(lambda n: n.startswith("eval_sys_"), dir(m)):
        fn = getattr(m, x)
        registry[_get_name(fn.__doc__)] = fn

    # .extern takes two or three arguments: a wildcard lambda is handed
    # whichever of x, y, z the call supplied (as .strict is)
    registry['.extern'] = KGLambda(eval_sys_extern, wildcard=True)
    registry['.param'] = KGLambda(eval_sys_param, wildcard=True)

    return registry
