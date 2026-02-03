"""
KlongPy parser and lexer functions.

This module contains all the parsing functions for the Klong language:
- Lexeme reading (numbers, strings, symbols, operators)
- List parsing
- Conditional expression parsing
- Comment handling
"""
import copy

from .types import (
    KGSym, KGChar, KGOp, KGCond, KGCall, KGLambda,
    reserved_fn_symbol_map,
    safe_eq, is_symbolic
)


# Character matching utilities

def cmatch(t, i, c):
    return i < len(t) and t[i] == c


def cmatch2(t, i, a, b):
    return cmatch(t, i, a) and cmatch(t, i+1, b)


def cpeek(t, i):
    return t[i] if i < len(t) else None


def cpeek2(t, i):
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


# Comment handling

def read_shifted_comment(t, i=0):
    while i < len(t):
        c = t[i]
        if c == '"':
            i += 1
            if not cmatch(t, i, '"'):
                break
        i += 1
    return i


def read_sys_comment(t, i, a):
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


# Whitespace handling

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
    i = skip_space(t, i, ignore_newline=ignore_newline)
    if cmatch2(t, i, ':', '"'):
        i = read_shifted_comment(t, i+2)
        i = skip(t, i)
    return i


# Lexeme readers

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
            if cmatch(t, i+1, '-'):
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
    if cmatch2(t, i, '\\', '~') or cmatch2(t, i, '\\', '*'):
        return i+2, KGOp(t[i:i+2], arity=0)
    return i+1, KGOp(t[i:i+1], arity=0)


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
            if not cmatch(t, i, '"'):
                break
        r.append(c)
        i += 1
    return i, "".join(r)


# Dictionary helper

def list_to_dict(a):
    return {x[0]:x[1] for x in a}


# Lambda for copy operations (used in dict parsing)
copy_lambda = KGLambda(lambda x: copy.deepcopy(x))


def read_list(t, delim, i=0, module=None):
    """
    Parse a list from string t starting at position i.
    Returns a Python list (caller converts to array if needed).

    L := '[' (C|L)* ']'
    """
    arr = []
    i = skip(t, i, ignore_newline=True)
    while not cmatch(t, i, delim) and i < len(t):
        i, q = kg_read(t, i, read_neg=True, ignore_newline=True, module=module)
        if q is None:
            break
        if safe_eq(q, '['):
            i, q = read_list(t, ']', i=i, module=module)
        arr.append(q)
        i = skip(t, i, ignore_newline=True)
    if cmatch(t, i, delim):
        i += 1
    return i, arr


def kg_read(t, i, read_neg=False, ignore_newline=False, module=None):
    """
    Read a Klong lexeme from string t starting at position i.

    C := I | H | R | S | V | Y
    """
    i = skip(t, i, ignore_newline=ignore_newline)
    if i >= len(t):
        return i, None
    a = t[i]
    if a == '\n':
        a = ';'
    if a in [';', '(', ')', '{', '}', ']']:
        return i+1, a
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
            i, d = read_list(t, '}', i=i+2, module=module)
            d = list_to_dict(d)
            return i, KGCall(copy_lambda, args=d, arity=0)
        elif aa == '[':
            return i+2, ':['
        elif aa == '|':
            return i+2, ':|'
        return i+2, KGOp(f":{aa}", arity=0)
    elif safe_eq(a, '['):
        return read_list(t, ']', i=i+1, module=module)
    elif is_symbolic(a):
        return read_sym(t, i, module=module)
    return read_op(t, i)


def kg_read_array(t, i, backend, **kwargs):
    """
    Read a value and convert lists to arrays using the provided backend.

    This is a helper that wraps kg_read and handles list-to-array conversion,
    centralizing the pattern used by the interpreter and eval_sys.

    Parameters
    ----------
    t : str
        The string to read from.
    i : int
        Starting position in the string.
    backend : BackendProvider
        The backend to use for array conversion.
    **kwargs
        Additional arguments passed to kg_read (read_neg, ignore_newline, module).

    Returns
    -------
    tuple
        (new_position, value) where value is converted to an array if it was a list.
    """
    i, a = kg_read(t, i, **kwargs)
    if isinstance(a, list):
        a = backend.kg_asarray(a)
    return i, a


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
    i, n = klong._expr(t, i, ignore_newline=True)
    r.append(n)
    i = cexpect(t, i, ';')
    i, n = klong._expr(t, i, ignore_newline=True)
    r.append(n)
    i = skip(t, i, ignore_newline=True)
    if cmatch2(t, i, ':', '|'):
        i, n = read_cond(klong, t, i+2)
        r.append(n)
    else:
        i = cexpect(t, i, ';')
        i, n = klong._expr(t, i, ignore_newline=True)
        r.append(n)
        i = skip(t, i, ignore_newline=True)
        i = cexpect(t, i, ']')
    return i, KGCond(r)


# Adverb peeking

def peek_adverb(t, i=0):
    from .types import is_adverb
    x = cpeek2(t, i)
    if is_adverb(x):
        return i+2, x
    x = cpeek(t, i)
    if is_adverb(x):
        return i+1, x
    return i, None
