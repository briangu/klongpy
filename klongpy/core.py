import copy
import inspect
import weakref
from enum import Enum
import sys
import numpy

from .backend import np, TorchUnsupportedDtypeError, get_default_backend, to_numpy

# python3.11 support
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec


def get_dtype_kind(arr, backend):
    """
    Get the dtype 'kind' character for an array (numpy or torch).

    Returns:
        'O' for object dtype
        'i' for integer types
        'f' for float types
        'u' for unsigned integer
        'b' for boolean
        'c' for complex
    """
    return backend.get_dtype_kind(arr)


class KlongException(Exception):
    pass


class KGSym(str):
    def __repr__(self):
        return f":{super().__str__()}"
    def __eq__(self, o):
        return isinstance(o,KGSym) and self.__str__() == o.__str__()
    def __hash__(self):
        return super().__hash__()


def get_fn_arity_str(arity):
    if arity == 0:
        return ":nilad"
    elif arity == 1:
        return ":monad"
    elif arity == 2:
        return ":dyad"
    return ":triad"


class KGFn:
    def __init__(self, a, args, arity):
        self.a = a
        self.args = args
        self.arity = arity

    def __str__(self):
        return get_fn_arity_str(self.arity)

    def is_op(self):
        return isinstance(self.a,KGOp)

    def is_adverb_chain(self):
        return isinstance(self.a,list) and isinstance(self.a[0],KGAdverb)


class KGFnWrapper:
    """
    Wrapper for KGFn that enables calling from Python with dynamic symbol resolution.

    When a KGFn is stored and later invoked, this wrapper automatically re-resolves
    the symbol to use the current function definition. This matches k4 behavior and
    provides REPL-friendly semantics where function redefinitions take effect immediately.

    Example:
        fn = klong['callback']  # Returns KGFnWrapper
        klong('callback::{new implementation}')
        fn(args)  # Uses the NEW implementation
    """

    def __init__(self, klong, fn, sym=None):
        self.klong = klong
        self.fn = fn
        # Use provided symbol name (cached) or search for it
        self._sym = sym if sym is not None else self._find_symbol(fn)

    def _find_symbol(self, fn):
        """Find which symbol this function is currently bound to"""
        if not isinstance(fn, KGFn) or isinstance(fn, KGCall):
            return None

        # Search the context for this function
        # Skip reserved symbols (x, y, z, .f) which are function parameters, not stored callbacks
        for sym, value in self.klong._context:
            # Skip reserved symbols - use the module constants
            if sym in reserved_fn_symbols or sym == reserved_dot_f_symbol:
                continue
            if value is fn:
                return sym
        return None

    def __call__(self, *args, **kwargs):
        # Try to resolve dynamically first if we have a symbol
        if self._sym is not None:
            try:
                current = self.klong._context[self._sym]
                if isinstance(current, KGFn) and not isinstance(current, KGCall):
                    # Use the current definition
                    if len(args) != current.arity:
                        raise RuntimeError(f"Klong function called with {len(args)} but expected {current.arity}")
                    fn_args = [np.asarray(x) if isinstance(x, list) else x for x in args]
                    return self.klong.call(KGCall(current.a, [*fn_args], current.arity))
            except KeyError:
                # Symbol was deleted, fall through to original function
                pass

        if len(args) != self.fn.arity:
            raise RuntimeError(f"Klong function called with {len(args)} but expected {self.fn.arity}")
        fn_args = [np.asarray(x) if isinstance(x, list) else x for x in args]
        return self.klong.call(KGCall(self.fn.a, [*fn_args], self.fn.arity))


class KGCall(KGFn):
    def __str__(self):
        return self.a.__str__() if issubclass(type(self.a), KGLambda) else super().__str__()


class KGOp:
    def __init__(self, a, arity):
        self.a = a
        self.arity = arity


class KGAdverb:
    def __init__(self, a, arity):
        self.a = a
        self.arity = arity


class KGChar(str):
    pass


class KGCond(list):
    pass


def safe_inspect(fn, follow_wrapped=True):
    try:
        return inspect.signature(fn, follow_wrapped=follow_wrapped).parameters
    except ValueError:
        return {"args":[]}

class KGLambda:
    """
    KGLambda wraps a lambda and make it available to Klong, allowing for direct
    integration of python functions in Klong.

    Introspection is used to infer which parameters should be collected from the
    current context and passed to the lambda. Parameter names must be x,y, or z
    according to the klong convention.  The value for these parameters are
    extracted directly from the currentcontext.

    If a lambda requires access to klong itself, that must be the first parameter.

    Function arity is computed by examining the arguments.

    e.g.

    lambda x,y: x + y
    lambda klong, x: klong(x)

    """
    def __init__(self, fn, args=None, provide_klong=False, wildcard=False):
        self.fn = fn
        params = args or safe_inspect(fn)
        self.args = [reserved_fn_symbol_map[x] for x in reserved_fn_args if x in params]
        self._provide_klong = provide_klong or 'klong' in params
        self._wildcard = wildcard

    def _get_pos_args(self, ctx):
        if self._wildcard:
            pos_args = []
            for sym in reserved_fn_symbols:
                try:
                    pos_args.append(ctx[sym])
                except KeyError:
                    break
        else:
            pos_args = [ctx[x] for x in self.args]
        return pos_args

    def __call__(self, klong, ctx):
        pos_args = self._get_pos_args(ctx)
        return self.fn(klong, *pos_args) if self._provide_klong else self.fn(*pos_args)

    def call_with_kwargs(self, klong, ctx, kwargs):
        pos_args = self._get_pos_args(ctx)
        return self.fn(klong, *pos_args, **kwargs) if self._provide_klong else self.fn(*pos_args, **kwargs)

    def get_arity(self):
        return len(self.args)

    def __str__(self):
        return get_fn_arity_str(self.get_arity())


class KGChannelDir(Enum):
    INPUT=1
    OUTPUT=2


class KGChannel:
    class FileHandler:
        def __init__(self, raw, parent):
            self._raw = raw
            self._ref = weakref.ref(parent, self.close)

        def close(self, *args):
            self._raw.close()

    def __init__(self, raw, channel_dir):
        self.channel_dir = channel_dir
        self.raw = raw
        self._fh = KGChannel.FileHandler(raw, self)
        self.at_eof = False

    def __enter__(self):
        return self

    def __exit__(self, ext_type, exc_value, traceback):
        self._fh.close()


class RangeError(Exception):
    def __init__(self, i):
        self.i = i
        super().__init__()


reserved_fn_args = ['x','y','z']
reserved_fn_symbols = [KGSym(n) for n in reserved_fn_args]
reserved_fn_symbol_map = {n:KGSym(n) for n in reserved_fn_args}
reserved_dot_f_symbol = KGSym('.f')


def is_list(x):
    return isinstance(x,list) or (np.isarray(x) and x.ndim > 0)


def is_iterable(x):
    return is_list(x) or (isinstance(x,str) and not isinstance(x, (KGSym, KGChar)))


def is_empty(a):
    return is_iterable(a) and len(a) == 0


def is_dict(x):
    return isinstance(x, dict)


def to_list(a):
    return a if isinstance(a, list) else a.tolist() if np.isarray(a) else [a]


def is_integer(x, backend):
    if issubclass(type(x), (int, numpy.integer)):
        return True
    # Handle 0-dim numpy arrays
    if isinstance(x, numpy.ndarray) and x.ndim == 0:
        return numpy.issubdtype(x.dtype, numpy.integer)
    # Handle backend-specific scalar integers (e.g., torch tensors)
    return backend.is_scalar_integer(x)


def is_float(x, backend):
    if issubclass(type(x), (float, numpy.floating, int)):
        return True
    # Handle 0-dim numpy arrays
    if isinstance(x, numpy.ndarray) and x.ndim == 0:
        return numpy.issubdtype(x.dtype, numpy.floating)
    # Handle backend-specific scalar floats (e.g., torch tensors)
    return backend.is_scalar_float(x)


def is_number(a, backend):
    if is_float(a, backend) or is_integer(a, backend):
        return True
    # Handle 0-dim numpy arrays
    if isinstance(a, numpy.ndarray) and a.ndim == 0:
        return numpy.issubdtype(a.dtype, numpy.number)
    # Handle 0-dim backend tensors as numbers
    if backend.is_backend_array(a) and hasattr(a, 'ndim') and a.ndim == 0:
        return True
    return False


def str_is_float(b):
    try:
        float(b)
        return True
    except ValueError:
        return False

def in_map(x, v):
    try:
        return x in v
    except Exception:
        return False


def kg_asarray(a, backend):
    """Convert input to array using the backend's kg_asarray method."""
    return backend.kg_asarray(a)


def kg_equal(a, b, backend):
    """Compare two values or arrays for equality, handling nested arrays and tensors."""
    if a is b:
        return True

    # Check for arrays (numpy or backend-specific)
    is_numpy_a = isinstance(a, numpy.ndarray)
    is_numpy_b = isinstance(b, numpy.ndarray)
    is_backend_a = backend.is_backend_array(a)
    is_backend_b = backend.is_backend_array(b)

    na, nb = is_numpy_a or is_backend_a, is_numpy_b or is_backend_b

    # Handle arrays with same dtype
    if na and nb:
        a_dtype = get_dtype_kind(a, backend)
        b_dtype = get_dtype_kind(b, backend)
        if a_dtype == b_dtype and a_dtype != 'O':
            return bool(np.array_equal(a, b))

    na, nb = na or isinstance(a, list), nb or isinstance(b, list)

    if na != nb:
        # One is array/list, the other is not - could be scalar tensor/array vs scalar
        # Handle comparing 0-dim arrays/tensors with scalars
        if is_numpy_a and a.ndim == 0 and not nb:
            return kg_equal(a.item(), b, backend)
        if is_numpy_b and b.ndim == 0 and not na:
            return kg_equal(a, b.item(), backend)
        if is_backend_a and hasattr(a, 'ndim') and a.ndim == 0 and not nb:
            return kg_equal(backend.scalar_to_python(a), b, backend)
        if is_backend_b and hasattr(b, 'ndim') and b.ndim == 0 and not na:
            return kg_equal(a, backend.scalar_to_python(b), backend)
        return False

    if na:
        # Handle 0-dim arrays/tensors - compare as scalars
        a_is_0d = hasattr(a, 'ndim') and a.ndim == 0
        b_is_0d = hasattr(b, 'ndim') and b.ndim == 0
        if a_is_0d or b_is_0d:
            a_val = backend.scalar_to_python(a) if a_is_0d else a
            b_val = backend.scalar_to_python(b) if b_is_0d else b
            return kg_equal(a_val, b_val, backend)
        return len(a) == len(b) and all(kg_equal(x, y, backend) for x, y in zip(a, b))

    if is_number(a, backend) and is_number(b, backend):
        # Convert tensors to Python scalars for comparison
        if backend.is_backend_array(a):
            a = backend.scalar_to_python(a)
        if backend.is_backend_array(b):
            b = backend.scalar_to_python(b)
        result = np.isclose(a, b)
        # np.isclose might return an array/tensor, ensure we return bool
        if hasattr(result, 'item'):
            return bool(result.item())
        return bool(result)

    result = a == b
    # Handle tensor/array result from comparison
    if hasattr(result, 'all'):
        # For arrays, check if all elements are equal
        return bool(result.all())
    if hasattr(result, 'item'):
        return bool(result.item())
    return bool(result)

def has_none(a):
    if isinstance(a,list):
        for q in a:
            if q is None:
                return True
    return False


def cmatch(t, i, c):
    return i < len(t) and t[i] == c


def cmatch2(t, i, a, b):
    return cmatch(t, i, a) and cmatch(t, i+1, b)


def cpeek(t,i):
    return t[i] if i < len(t) else None


def cpeek2(t,i):
    return t[i:i+2] if i < (len(t)-1) else None


class UnexpectedChar(Exception):
    def __init__(self, t, i, c):
        super().__init__(f"t: {t[i-10:i+10]} pos: {i} char: {c}")


class UnexpectedEOF(Exception):
    def __init__(self, t, i):
        self.t = t
        self.i = i
        super().__init__(f"t: {t[i-10:]} pos: {i}")


def cexpect(t, i, c):
    if cmatch(t, i, c):
        return i + 1
    raise UnexpectedChar(t, i, c)


def cexpect2(t, i, a, b):
    if cmatch(t, i, a) and cmatch(t, i+1, b):
        return i + 2
    raise UnexpectedChar(t, i, b)


def safe_eq(a,b):
    return isinstance(a,type(b)) and a == b


def rec_flatten(a):
    if is_list(a) and len(a) > 0:
        return np.concatenate([rec_flatten(x) if is_list(x) else np.array([x]) for x in a]).ravel()
    return a


def rec_fn(a,f):
    _backend = get_default_backend()
    return _backend.kg_asarray([rec_fn(x, f) for x in a]) if is_list(a) else f(a)


def vec_fn(a, f, backend):
    """
    Apply a function `f` to an array `a`, with support for both nested arrays and direct vectorized operation.
    """
    if np.isarray(a) and a.dtype == 'O':
        # For object arrays, process each element and preserve structure
        result = [((vec_fn(x, f, backend)) if is_list(x) else f(x)) for x in a]
        return numpy.asarray(result, dtype=object)
    return f(a)


def vec_fn2(a, b, f):
    """
    Apply function `f` recursively to the elements of `a` and `b`, which can be scalar values, vectors, or nested vectors.

    This function distinguishes 8 cases based on the types and dimensions of `a` and `b`:

    1. vec[A],vec[B]: `f` is applied directly to `a` and `b`.
    2. vec[A],obj_vec[B]: `f` is applied recursively to pairs of elements in `a` and `b`.
    3. vec[A],scalar[B]: `f` is applied directly to `a` and `b`.
    4. obj_vec[A],vec[B]: `f` is applied recursively to pairs of elements in `a` and `b`.
    5. obj_vec[A],scalar[B]: `f` is applied recursively to the elements in `a` and the scalar `b`.
    6. scalar[A],vec[B]: `f` is applied directly to `a` and `b`.
    7. scalar[A],obj_vec[B]: `f` is applied recursively to the scalar `a` and the elements in `b`.
    8. scalar[A],scalar[B]: `f` is applied directly to `a` and `b`.

    Parameters
    ----------
    a, b : numpy.array or any type
        The inputs to `f`. They can be numpy arrays of any data type. If they are arrays, they should have the same shape.
        Non-array inputs can be of any type that `f` can accept.

    f : callable
        A function that takes two arguments and can handle the types and dimensions of `a` and `b`.

    Returns
    -------
    numpy.array or any type
        The result of applying `f` to `a` and `b`, which can be a scalar, a vector, or a nested vector depending on
        the inputs and `f`.

    Notes
    -----
    This function assumes that `f` can handle the types and dimensions of `a` and `b`, and that `a` and `b` have the same
    shape if they are arrays. It does not check these conditions, so unexpected results or errors may occur if they are
    not satisfied.

    """
    _backend = get_default_backend()
    _kg_asarray = _backend.kg_asarray
    if np.isarray(a):
        if a.dtype == 'O':
            if np.isarray(b):
                assert len(a) == len(b)
                return _kg_asarray([vec_fn2(x, y, f) for x,y in zip(a,b)])
            else:
                return _kg_asarray([vec_fn2(x, b, f) for x in a])
        elif np.isarray(b) and b.dtype == 'O':
            assert len(a) == len(b)
            return _kg_asarray([vec_fn2(x, y, f) for x,y in zip(a,b)])
    elif np.isarray(b) and b.dtype == 'O':
        return _kg_asarray([vec_fn2(a, x, f) for x in b])
    return f(a,b)


def is_symbolic(c):
    return isinstance(c, str) and (c.isalpha() or c.isdigit() or c == '.')


def is_char(x):
    # Check for both core and backend KGChar classes
    if isinstance(x, KGChar):
        return True
    # Also check for backend KGChar (in case they're different classes)
    return type(x).__name__ == 'KGChar' and isinstance(x, str)


def is_atom(x):
    """ All objects except for non-empty lists and non-empty strings are atoms. """
    return is_empty(x) if is_iterable(x) else True


def kg_truth(x):
    return x*1


def str_to_chr_arr(s, backend):
    """
    Convert string to character array.

    Parameters
    ----------
    s : str
        The string to convert.
    backend : BackendProvider
        The backend to use.

    Returns
    -------
    array
        Array of KGChar objects.

    Raises
    ------
    UnsupportedDtypeError
        If the backend doesn't support string operations.
    """
    return backend.str_to_char_array(s)


def read_num(t, i=0):
    p = i
    use_float = False
    if t[i] == '-':
        i += 1
    while i < len(t):
        if t[i] == '.':
            use_float = True
        elif t[i] == 'e':
            use_float = True
            if cmatch(t,i+1,'-'):
                i += 2
        elif not t[i].isnumeric():
            break
        i += 1
    return i, float(t[p:i]) if use_float else int(t[p:i])


def read_char(t, i):
    i = cexpect2(t, i, '0', 'c')
    if i >= len(t):
        raise UnexpectedEOF(t, i)
    return i+1, KGChar(t[i])


def read_sym(t, i=0, module=None):
    p = i
    while i < len(t) and is_symbolic(t[i]):
        i += 1
    x = t[p:i]
    return i, reserved_fn_symbol_map.get(x) or KGSym(x if x.startswith('.') or module is None else f"{x}`{module}")


def read_op(t, i=0):
    if cmatch2(t,i,'\\','~') or cmatch2(t,i,'\\','*'):
        return i+2,KGOp(t[i:i+2],arity=0)
    return i+1,KGOp(t[i:i+1],arity=0)


def read_shifted_comment(t, i=0):
    while i < len(t):
        c = t[i]
        if c == '"':
            i += 1
            if not cmatch(t,i,'"'):
                break
        i += 1
    return i


def read_sys_comment(t,i,a):
    """

        .comment(x)                                            [Comment]

        Read and discard lines until the current line starts with the
        string specified in "x". Also discard the line containing the
        end-of-comment marker and return "x".

        Example: .comment("end-of-comment")
                 this will be ignored
                 this, too: *%(*^#)&(#
                 end-of-comment

        NOTE: this is handled in the parsing phase and is not a runtime function

    """
    try:
        j = t[i:].index(a)
        while t[i+j+1:].startswith(a):
            j += 1
        return i + j + len(a)
    except ValueError:
        return RuntimeError("end of comment not found")


def skip_space(t, i=0, ignore_newline=False):
    """
        NOTE: a newline character translates to a semicolon in Klong,
        except in functions, dictionaries, conditional expressions,
        and lists. So
    """
    while i < len(t) and (t[i].isspace() and (ignore_newline or t[i] != '\n')):
        i += 1
    return i


def skip(t, i=0, ignore_newline=False):
    i = skip_space(t,i,ignore_newline=ignore_newline)
    if cmatch2(t, i, ':', '"'):
        i = read_shifted_comment(t, i+2)
        i = skip(t, i)
    return i


def read_list(t, delim, i=0, module=None, level=1):
    """

        # A list is any number of class lexemes (or lists) delimited by
        # square brackets.

        L := '[' (C|L)* ']'

    """
    backend = get_default_backend()
    arr = []
    i = skip(t,i,ignore_newline=True)
    while not cmatch(t,i,delim) and i < len(t):
        # we can knowingly read neg numbers in list context
        i, q = kg_read(t, i, read_neg=True, ignore_newline=True, module=module, list_level=level+1)
        if q is None:
            break
        if safe_eq(q, '['):
            i,q = read_list(t, ']', i=i, module=module, level=level+1)
        arr.append(q)
        i = skip(t,i,ignore_newline=True)
    if cmatch(t,i,delim):
        i += 1
    if level == 1:
        try:
            aa = kg_asarray(arr, backend)
            if get_dtype_kind(aa, backend) not in ['O','i','f']:
                aa = numpy.asarray(arr, dtype=object)
        except TorchUnsupportedDtypeError:
            # Backend can't handle this data - fall back to numpy object array
            # Recursively convert inner lists to arrays, converting tensors to numpy
            def convert_inner(x):
                if isinstance(x, list):
                    try:
                        result = kg_asarray(x, backend)
                        # Convert tensor to numpy for object array compatibility
                        return to_numpy(result)
                    except TorchUnsupportedDtypeError:
                        return numpy.asarray([convert_inner(e) for e in x], dtype=object)
                # Convert any tensors to numpy
                return to_numpy(x)
            aa = numpy.asarray([convert_inner(x) for x in arr], dtype=object)
    else:
        aa = arr
    return i, aa


def read_string(t, i=0):
    """

    ".*"                                                    [String]

    A string is (almost) any sequence of characters enclosed by
    double quote characters. To include a double quote character in
    a string, it has to be duplicated, so the above regex is not
    entirely correct. A comment is a shifted string (see below).
    Examples: ""
                "hello, world"
                "say ""hello""!"

    Note: this comforms to the KG read_string impl.
          perf tests show that the final join is fast for short strings

    """
    r = []
    while i < len(t):
        c = t[i]
        if c == '"':
            i += 1
            if not cmatch(t,i,'"'):
                break
        r.append(c)
        i += 1
    return i,"".join(r)


def read_cond(klong, t, i=0):
    """

        # A conditional expression has two forms: :[e1;e2;e3] means "if
        # e1 is true, evaluate to e2, else evaluate to e3".
        # :[e1;e2:|e3;e4;e5] is short for :[e1;e2:[e3;e4;e5]], i.e. the
        # ":|" acts as an "else-if" operator. There may be any number of
        # ":|" operators in a conditional.

        c := ':[' ( e ';' e ':|' )* e ';' e ';' e ']'

    """
    r = []
    i,n = klong._expr(t, i, ignore_newline=True)
    r.append(n)
    i = cexpect(t, i, ';')
    i,n = klong._expr(t, i, ignore_newline=True)
    r.append(n)
    i = skip(t,i,ignore_newline=True)
    if cmatch2(t,i,':','|'):
        i,n = read_cond(klong,t,i+2)
        r.append(n)
    else:
        i = cexpect(t, i, ';')
        i,n = klong._expr(t, i, ignore_newline=True)
        r.append(n)
        i = skip(t,i,ignore_newline=True)
        i = cexpect(t, i, ']')
    return i, KGCond(r)


def list_to_dict(a):
    return {x[0]:x[1] for x in a}


copy_lambda = KGLambda(lambda x: copy.deepcopy(x))

def kg_read(t, i=0, read_neg=False, ignore_newline=False, module=None, list_level=0):
    """

        # Lexeme classes are the sets of the lexemes specified in the
        # previous section, except for operators.

        C := I   # integer
        | H   # character
        | R   # real number
        | S   # string
        | V   # variable (symbol)
        | Y   # (quoted) symbol

        NOTE: this function mirrors the klong implementation so that sys_read/write
        match klong's as well.  The grammar read here is a superset of C.

        NOTE: a newline character translates to a semicolon in Klong,
        except in functions, dictionaries, conditional expressions,
        and lists. So

        a()
        b()

        is equal to a();b(), but

        [1
         2
         3]

        is equal to [1 2 3] and

        :[x;
          y;
          z]

        is equal to :[x;y;z] and

        f::{.d("hello ");
            .p("world!");
            []}

        is a valid function definition.

    """
    i = skip(t, i, ignore_newline=ignore_newline)
    if i >= len(t):
        return i, None
    a = t[i]
    if a == '\n':
        a = ';' # convert newlines to semicolons
    if a in [';','(',')','{','}',']']:
        return i+1,a
    elif cmatch2(t, i, '0', 'c'):
        return read_char(t, i)
    elif a.isnumeric() or (read_neg and (a == '-' and (i+1) < len(t) and t[i+1].isnumeric())):
        return read_num(t, i)
    elif a == '"':
        return read_string(t, i+1)
    elif a == ':' and (i+1 < len(t)):
        aa = t[i+1]
        if aa.isalpha() or aa == '.':
            return read_sym(t, i=i+1, module=module)
        elif aa.isnumeric() or aa == '"':
            return kg_read(t, i+1, ignore_newline=ignore_newline, module=module)
        elif aa == '{':
            i, d = read_list(t, '}', i=i+2, module=module, level=list_level+1)
            d = list_to_dict(d)
            return i, KGCall(copy_lambda,args=d,arity=0)
        elif aa == '[':
            return i+2,':['
        elif aa == '|':
            return i+2,':|'
        return i+2,KGOp(f":{aa}",arity=0)
    elif safe_eq(a, '['):
        return read_list(t, ']', i=i+1, module=module, level=list_level+1)
    elif is_symbolic(a):
        return read_sym(t, i, module=module)
    return read_op(t,i)


def kg_write_symbol(x, display=False):
    return str(x) if display else f":{x}"


def kg_write_integer(x, display=False):
    return str(int(x))


def kg_write_float(x, display=False):
    return str(x)


def kg_write_char(c, display=False):
    return c if display else f"0c{c}"


def kg_write_string(s, display=False):
    if display:
        return s
    arr = ['"']
    for c in s:
        if c == '"':
            arr.append('"')
        arr.append(c)
    arr.append('"')
    return ''.join(arr)


def kg_write_dict(d, backend, display=False):
    # determine if the object d has overwritten the default __str__ and call it
    # if so, otherwise use the default dict str
    if d.__class__.__name__ != 'dict':
        return str(d)
    return ''.join([':{', ' '.join([kg_write(list(e), backend, display=display) for e in d.items()]), '}'])


def kg_write_list(x, backend, display=False):
    return ''.join(['[', ' '.join([kg_write(q, backend, display=display) for q in x]), ']'])


def kg_write_fn(x, display=False):
    return str(x)


def kg_write_channel(x, display=False):
    if x.channel_dir == KGChannelDir.INPUT:
        return ":inchan.0"
    return f":outchan.{2 if x.raw == sys.stderr else 1}"


def kg_write(a, backend, display=False):
    _backend = backend
    # Convert backend arrays (e.g., torch tensors) to display-friendly format
    a = _backend.to_display(a)
    if isinstance(a,KGSym):
        return kg_write_symbol(a, display=display)
    elif is_integer(a, _backend):
        return kg_write_integer(a,display=display)
    elif is_float(a, _backend):
        return kg_write_float(a,display=display)
    elif isinstance(a,KGChar):
        return kg_write_char(a,display=display)
    elif isinstance(a, str):
        return kg_write_string(a,display=display)
    elif isinstance(a,dict):
        return kg_write_dict(a, _backend, display=display)
    elif is_list(a):
        return kg_write_list(a, _backend, display=display)
    elif isinstance(a,KGFn):
        return kg_write_fn(a,display=display)
    elif isinstance(a,KGChannel):
        return kg_write_channel(a,display=display)
    elif hasattr(a,'__str__'):
        return str(a)
    elif safe_eq(a, np.inf):
        return ":undefined"


def kg_argsort(a, backend, descending=False):
    """

    Return the indices of the sorted array (may be nested) or a string.  Duplicate elements are disambiguated by their position in the array.

    argsort("foobar") => [4 3 0 1 2 5]
                                ^ ^
                            arbitrary ordering resolved by index position

    argsort("foobar",descending=True) => [5 2 1 0 3 4]
                                            ^ ^
                            arbitrary ordering resolved by index position

    """
    if not is_iterable(a) or len(a) == 0:
        return a

    # Fast path: for simple 1D numeric arrays, use native argsort
    if hasattr(a, 'ndim') and a.ndim == 1:
        dtype_kind = get_dtype_kind(a, backend)
        if dtype_kind in ('i', 'f', 'u'):
            return backend.argsort(a, descending=descending)

    # Slow path: nested arrays or strings need element-by-element comparison
    def _e(x):
        return (-np.inf,x) if is_empty(a[x]) else (np.max(a[x]),x) if is_list(a[x]) else (a[x],x)
    return np.asarray(sorted(range(len(a)), key=_e, reverse=descending))


def peek_adverb(t,i=0):
    x = cpeek2(t,i)
    if is_adverb(x):
        return i+2,x
    x = cpeek(t,i)
    if is_adverb(x):
        return i+1,x
    return i,None


def is_adverb(s):
    return s in {
        "'",
        ':\\',
        ":'",
        ':/',
        '/',
        ':~',
        ':*',
        '\\',
        '\\~',
        '\\*'
    }


def get_adverb_arity(s, ctx):
    if s == "'":
        return ctx
    elif s == ':\\':
        return 2
    elif s == ':\'':
        return 2
    elif s == ':/':
        return 2
    elif s == '/':
        return 2
    elif s == ':~':
        return 1
    elif s == ':*':
        return 1
    elif s == '\\':
        return 2
    elif s == '\\~':
        return 1
    elif s == '\\*':
        return 1
    raise RuntimeError(f"unknown adverb: {s}")


def merge_projections(arr):
    """

        A projection is a new function that is created by projecting an
        existing function onto at least one of its arguments, resulting
        in the partial application of the original function.

        The notation of projection is that of function application where
        the arguments onto which the function is being projected are
        omitted. For instance,

        Projection   Equivalent function
        {x-y}(5;)    {5-x}
        {x-y}(;5)    {x-5}

        and, given a ternary function f3:

        Projection   Equivalent function
        f3(1;2;)     {f3(1;2;x)}
        f3(1;;3)     {f3(1;x;3)}
        f3(;2;3)     {f3(x;2;3)}
        f3(;;3)      {f3(x;y;3)}
        f3(;2;)      {f3(x;2;y)}
        f3(1;;)      {f3(1;x;y)}

        The projection of a triad is a dyad or a monad, depending on the
        number of arguments onto which the triad is being projected. The
        projection of a dyad is always a monad. There is no projection
        of a monad or nilad.

        Alternatively, monads and nilads can be considered to be their
        own projections (onto zero arguments), but there is no special
        syntax for this case. Any function that is being projected onto
        all of its arguments is simply the function itself.

        Projections are ordinary functions and can be used in all places
        where a verb is expected. For instance:

        f::{x,y}
        f(;0)'[1 2 3]  -->  [[1 0] [2 0] [3 0]]
        f(0;)'[1 2 3]  -->  [[0 1] [0 2] [0 3]]

        g::{x,y,z}
        1g(;2;)3       -->  [1 2 3]

    """
    if len(arr) == 0:
        return arr
    if len(arr) == 1 or not has_none(arr[0]):
        return arr[0]
    sparse_fa = np.copy(arr[0])
    i = 0
    k = 1
    while i < len(sparse_fa) and k < len(arr):
        fa = arr[k]
        j = 0
        while i < len(sparse_fa) and j < len(fa):
            if sparse_fa[i] is None:
                sparse_fa[i] = fa[j]
                j += 1
                while j < len(fa) and safe_eq(fa[j], None):
                    j += 1
            i += 1
        k += 1
    return sparse_fa


def get_fn_arity(f):
    """
    Examine a function AST and infer arity by looking for x,y and z.
    This arity is needed to populate the KGFn.

    NOTE: TODO: it maybe easier / better to do this at parse time vs late.
    """
    if isinstance(f,KGFn) and isinstance(f.a,KGSym) and not in_map(f.a,reserved_fn_symbols):
       return sum(1 for x in set(f.args) if in_map(x, reserved_fn_symbols) or (x is None))
    def _e(f, level=0):
        if isinstance(f,KGFn):
            x = _e(f.a, level=1)
            if isinstance(f.args,list):
                for q in f.args:
                    x.update(_e(q,level=1))
        elif isinstance(f,list):
            x = set()
            for q in f:
                x.update(_e(q,level=1))
        elif isinstance(f,KGSym):
            x = set([f]) if f in reserved_fn_symbols else set()
        else:
            x = set()
        return x if level else len(x)
    return _e(f)
