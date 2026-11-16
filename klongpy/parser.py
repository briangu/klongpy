"""
KlongPy parser functions.

There are two front ends. The default one reads tokens from klongpy/lexer.py;
its readers are the tk_* functions at the end of this module and the grammar
methods of KlongInterpreter (_prog, _expr, _factor, _read_fn_args,
_apply_adverbs). The legacy one works on the source string with a character
cursor: the readers in the first part of this module and the _legacy_* methods
of KlongInterpreter. It is selected with KlongInterpreter(parser="legacy") or
KLONGPY_PARSER=legacy, is deprecated, and is kept for one release so the two
can be compared. The character readers (kg_read, read_list, ...) also remain
the readers of data for .r and .rs.

The token readers reproduce the legacy readers decision for decision; the
comments marked "legacy:" say which legacy behaviour a line stands for.
"""
import copy

from .lexer import ERR_CHAR_EOF
from .errors import KlongSyntaxError, KlongSyntaxWarning, UnexpectedChar, UnexpectedEOF, compat

from .types import (
    KGSym, KGQSym, KGChar, KGOp, KGCond, KGCall, KGFn, KGLambda,
    reserved_fn_symbol_map,
    safe_eq, is_list, is_symbolic, is_adverb
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


# UnexpectedChar and UnexpectedEOF live in klongpy.errors (they are
# KlongSyntaxError subclasses) and are re-exported from here.

def malformed_number(text, legacy_text, span=None, source=None):
    """
    The syntax error for a number that does not read ("1.2.3", "3e"). It is
    also a ValueError, which is what the number readers let through before;
    legacy_text is the message of that ValueError.
    """
    e = compat(KlongSyntaxError, ValueError)(f"malformed number '{text}'", span=span, source=source)
    e.legacy_text = legacy_text
    return e


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
        # also a RuntimeError, which is what this raised before
        e = compat(KlongSyntaxError, RuntimeError)("end of comment not found")
        e.legacy_text = "end of comment not found"
        raise e from None


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
            if cmatch(t, i+1, '-') or cmatch(t, i+1, '+'):
                i += 2
        elif not t[i].isnumeric():
            break
        i += 1
    try:
        return i, float(t[p:i]) if use_float else int(t[p:i])
    except ValueError as ex:
        raise malformed_number(t[p:i], str(ex), (p, i)) from ex


_CHAR_EOF = "expected a character after 0c but found end of input"


def read_char(t, i):
    i = cexpect2(t, i, '0', 'c')
    if i >= len(t):
        raise UnexpectedEOF(t, i, message=_CHAR_EOF)
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


def kg_read(t, i, read_neg=False, ignore_newline=False, module=None, qsym=False):
    """
    Read a Klong lexeme from string t starting at position i.

    C := I | H | R | S | V | Y

    qsym: when True, a quoted symbol (`:foo`) is returned as a KGQSym node
    instead of a plain KGSym. The interpreter asks for this where the lexeme
    is an expression that will be evaluated. Everywhere else (list and
    dictionary literals, reading data) a quoted symbol is a symbol value and
    stays a plain KGSym.
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
            i, s = read_sym(t, i=i+1, module=module)
            return i, (KGQSym(s) if qsym else s)
        elif aa.isnumeric() or aa == '"':
            return kg_read(t, i+1, ignore_newline=ignore_newline, module=module)
        elif aa == '{':
            i, d = read_list(t, '}', i=i+2, module=module)
            d = list_to_dict(d)
            # Parser-built call: args is the dictionary literal handed to
            # copy_lambda as-is (arity 0, nothing is bound or evaluated).
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
        Additional arguments passed to kg_read (read_neg, ignore_newline, module, qsym).

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
    i, n = klong._legacy_expr(t, i, ignore_newline=True)
    r.append(n)
    i = cexpect(t, i, ';')
    i, n = klong._legacy_expr(t, i, ignore_newline=True)
    r.append(n)
    i = skip(t, i, ignore_newline=True)
    if cmatch2(t, i, ':', '|'):
        i, n = read_cond(klong, t, i+2)
        r.append(n)
    else:
        i = cexpect(t, i, ';')
        i, n = klong._legacy_expr(t, i, ignore_newline=True)
        r.append(n)
        i = skip(t, i, ignore_newline=True)
        i = cexpect(t, i, ']')
    return i, KGCond(r)


# Adverb peeking

def peek_adverb(t, i=0):
    x = cpeek2(t, i)
    if is_adverb(x):
        return i+2, x
    x = cpeek(t, i)
    if is_adverb(x):
        return i+1, x
    return i, None


# Evaluated array parsing

class KGExprArray(list):
    """
    Container for evaluated array constructors [;expr1;expr2;...]

    Holds unevaluated expressions that are evaluated at runtime.
    Similar to KGCond, this defers evaluation until the expressions
    are actually needed (e.g., when a function is called).

    span, arg_spans and source are set by the token parser (see KGFn).
    """
    span = None
    arg_spans = None
    source = None


def read_expr_array(klong, t, i=0):
    """
    Parse an evaluated array constructor: [;expr1;expr2;...]

    Each element between semicolons is parsed as a full expression
    and evaluated when the array is constructed.

    Returns a list of expressions that will be evaluated to form the array.
    """
    r = []
    i = skip(t, i, ignore_newline=True)

    while i < len(t) and not cmatch(t, i, ']'):
        i, expr = klong._legacy_expr(t, i, ignore_newline=True)
        if expr is not None:
            r.append(expr)
        i = skip(t, i, ignore_newline=True)
        if cmatch(t, i, ';'):
            i += 1
            i = skip(t, i, ignore_newline=True)
        elif cmatch(t, i, ']'):
            break

    if cmatch(t, i, ']'):
        i += 1

    return i, r


# ---------------------------------------------------------------------------
# Token readers (the default front end). See tickets/design/frontend.md.
#
# Positions. The lexer holds the token list (lx.toks, ending with EOF) and the
# index of the next token (lx.pos). An index stands for the legacy character
# position at the END of the previous token: the whitespace before the token
# at that index has not been skipped. Where the legacy code tested a character
# at such a position (cmatch right after a lexeme) the test here is "this kind
# and tok.ws is False". Where the legacy code skipped first and then tested,
# only the kind is tested.
#
# Newlines. The legacy skip(ignore_newline=True) skips whitespace and newlines
# up to the first comment and then, after the comment, stops at the next
# newline. In token terms: NEWLINE tokens are skipped unless nl_comment is set.
# A NEWLINE that is read (not skipped) is a ';'.
#
# Tokens are indexed as tuples in the hot paths:
# 0 kind, 1 start, 2 end, 3 value, 4 ws, 5 nl_comment, 6 unterminated, 7 message.
# ---------------------------------------------------------------------------

class _Punct(str):
    """Marker for a ';' read as punctuation, as opposed to the string literal ";"."""
    __slots__ = ()


SEMI = _Punct(';')


class _DataFallback(Exception):
    """A list holds text that only the character reader lexes the legacy way (see tk_read_list)."""


class ProgResult(tuple):
    """
    What KlongInterpreter.prog returns: the pair (end offset, statements), as
    always, which also carries

        stmt_spans   one (start, end) span per statement (None on the legacy parser)
        source       the lexer.Source the spans refer to (None on the legacy parser)
        warnings     the KlongSyntaxWarning objects of this parse, in source
                     order (ticket 031). prog() does not issue them; __call__
                     and exec do, each time the text is run.
    """
    stmt_spans = None
    source = None
    warnings = ()


# ---------------------------------------------------------------------------
# Syntax checks (ticket 031). The token parser accepts a number of malformed
# inputs because the legacy parser did. Each such place calls tk_issue: at
# strict level 1 and above that raises a KlongSyntaxError; at level 0 the
# issue is recorded on the lexer, the parse goes on exactly as before, and
# prog() turns the issues into KlongSyntaxWarning objects (tk_warnings).
# ---------------------------------------------------------------------------

ADVERB_GLYPHS = frozenset(("'", '/', '\\', '\\~', '\\*', ":'", ':/', ':\\', ':~', ':*'))

_REST = "the rest of the input is not evaluated"
_TO_END = "it runs to the end of the input"


def tk_issue(lx, code, message, span, hint=None, note=None):
    """
    Something the legacy grammar accepts and should not. note says what
    happens to it at strict level 0; it is part of the warning only.
    """
    if lx.strict:
        raise KlongSyntaxError(message, span=span, hint=hint)
    lx.issues.append((span, code, message, note, hint))


def tk_assign_hint(left):
    """The hint for an assignment whose left side is the expression left, not a name (ticket 050)."""
    if isinstance(left, KGFn) and isinstance(left.a, KGSym) and isinstance(left.args, list):
        return f"to define a function write {left.a}::{{...}} and use x, y and z for its arguments"
    if isinstance(left, KGFn) and left.is_op() and left.a.a == '@':
        return "to change an element use amend, a:=value,index"
    if is_list(left):
        return "assign one name at a time: a::1;b::2"
    return None


def tk_warnings(lx):
    """The issues recorded while parsing, as KlongSyntaxWarning objects in source order."""
    out = []
    seen = set()
    open_seen = False
    for span, code, message, note, hint in sorted(lx.issues, key=lambda q: q[0]):
        if (span, code) in seen:
            continue        # a lexeme the parser read twice (once to look at it)
        seen.add((span, code))
        if code == 'unterminated-list' or code == 'unterminated-dict':
            if open_seen:
                continue    # nested in a list that is itself not closed: one warning
            open_seen = True
        out.append(KlongSyntaxWarning(f"{message}; {note}" if note else message,
                                      span=span, source=lx.source, hint=hint, code=code))
    return tuple(out)


def _lexeme(lx, tok):
    """A token as written, quoted and clipped, for a message."""
    s = lx.text[tok[1]:tok[2]]
    if len(s) > 20 or '\n' in s:
        s = s.split('\n', 1)[0][:20] + '...'
    return f"'{s}'"


def _matching(toks, j):
    """Index of the token that closes the bracket opened at index j, or None."""
    depth = 0
    for i in range(j, len(toks)):
        k = toks[i][0]
        if k in ('LPAREN', 'CALLP', 'LBRACK', 'LBRACE', 'COND', 'DICT'):
            depth += 1
        elif k in ('RPAREN', 'RBRACK', 'RBRACE'):
            depth -= 1
            if depth == 0:
                return i
    return None


def tk_missing_operand(lx, what, span, right=True):
    """An operator, an infix function or an adverb chain with nothing to apply it to."""
    tk_issue(lx, 'missing-operand', f"missing right operand for {what}" if right else f"missing operand for {what}", span)


def tk_stray(lx, tok):
    """A closing bracket (or :|) where an expression was expected."""
    v = tok[3]
    tk_issue(lx, 'stray-closer', f"unexpected '{v}'", (tok[1], tok[2]),
             note=f'it is read as the string "{v}"')


def tk_adverb_error(lx, tok):
    """An adverb glyph where a dyadic operator is expected: at every strict level an error."""
    v = tok[3]
    if tok[4]:
        raise KlongSyntaxError(f"adverb {v} must directly follow its verb", span=(tok[1], tok[2]),
                               hint=f"an adverb follows its verb with no space: remove the space before {v}")
    raise KlongSyntaxError(f"adverb {v} does not follow a verb", span=(tok[1], tok[2]),
                           hint="an adverb directly follows an operator, a name or a function literal")


def tk_index_hint(lx, j):
    """The hint for name[...] (the token at index j is the '['), or None."""
    toks = lx.toks
    tok = toks[j]
    prev = toks[j-1] if j > 0 else None
    if tok[0] != 'LBRACK' or tok[4] or prev is None or prev[0] not in ('NAME', 'QNAME', 'RBRACK', 'RPAREN'):
        return None
    text = lx.text
    m = _matching(toks, j)
    if prev[0] == 'NAME' and m is not None:
        name = text[prev[1]:prev[2]]
        inner = text[tok[2]:toks[m][1]].strip()
        if m == j + 2 and toks[j+1][0] in ('NUM', 'NAME', 'STR', 'CHAR'):
            return f"use {name}@{inner} to index"
        if len(inner) <= 24 and '\n' not in inner:
            return f"use {name}@[{inner}] to index"
    return "use @ to index: a@1 for one element, a@[1 2] for several"


def tk_juxtaposed(lx, j):
    """
    In an argument list, the token at index j starts an expression although no
    ';' followed the previous argument: f(1 2) passes two arguments.
    """
    tok = lx.toks[j]
    tk_issue(lx, 'trailing-input', f"unexpected {_lexeme(lx, tok)} after the end of the argument", (tok[1], tok[2]),
             hint=tk_index_hint(lx, j) or "arguments are separated by ;",
             note="it is passed as one more argument")


def tk_finish(lx):
    """
    Called when the top-level parse has stopped: whatever is left is input
    the legacy parser dropped without a word, and a string or comment may run
    to the end of the input.
    """
    toks = lx.toks
    j = lx.pos
    tok = toks[j]
    k = tok[0]
    if k != 'EOF':
        prev = toks[j-1] if j > 0 else None
        pk = prev[0] if prev is not None else None
        span = (tok[1], tok[2])
        text = lx.text
        if k == 'RPAREN' or k == 'RBRACK' or k == 'RBRACE':
            tk_issue(lx, 'stray-closer', f"unexpected '{tok[3]}': nothing is open that it could close", span,
                     note=_REST if toks[j+1][0] != 'EOF' else "it is ignored")
        else:
            hint = None
            if (k == 'LPAREN' or k == 'CALLP') and (pk == 'NAME' or pk == 'QNAME' or pk == 'RBRACE') and tok[4]:
                m = _matching(toks, j)
                args = text[tok[1]:toks[m][2]] if m is not None else None
                if args is None or len(args) > 24 or '\n' in args:
                    args = '(...)'
                callee = text[prev[1]:prev[2]] if pk != 'RBRACE' else '{...}'
                hint = f"no space may separate a function from its argument list: write {callee}{args}"
            elif k == 'LPAREN' or k == 'CALLP':
                hint = "only a name or a function literal can be followed by an argument list; assign the value to a name first"
            elif k == 'LBRACK' and tk_index_hint(lx, j) is not None:
                hint = tk_index_hint(lx, j)
            elif k in ('NUM', 'STR', 'CHAR', 'NAME', 'QNAME') and pk in ('NUM', 'STR', 'CHAR', 'NAME', 'QNAME'):
                hint = "a list is written in brackets, [1 2 3]; statements are separated by ;"
            tk_issue(lx, 'trailing-input', f"unexpected {_lexeme(lx, tok)} after the end of the expression", span,
                     hint=hint, note=_REST)
    # Only the last token can be a string without its closing quote.
    if len(toks) >= 2:
        t = toks[-2]
        if t[0] == 'STR' and t[6]:
            tk_issue(lx, 'unterminated-string', "unterminated string: the closing \" is missing", (t[1], t[2]), note=_TO_END)
    for c in lx.trivia:
        if c[6]:
            tk_issue(lx, 'unterminated-comment', "unterminated comment: the closing \" is missing", (c[1], c[2]),
                     note=_TO_END)


def tk_unterminated(lx, opener, close):
    """A list or dictionary literal that end of input closed. opener is its opening token."""
    if close == 'RBRACK':
        tk_issue(lx, 'unterminated-list', "unterminated list: the closing ] is missing", (opener[1], opener[2]), note=_TO_END)
    else:
        tk_issue(lx, 'unterminated-dict', "unterminated dictionary: the closing } is missing", (opener[1], opener[2]),
                 note=_TO_END)


def tk_dict(lx, entries, span):
    """
    The dictionary a :{...} literal denotes. Entries that are not [key value]
    pairs made the legacy reader fail with an IndexError or a TypeError; the
    syntax error raised here is still an instance of that class.
    """
    try:
        return list_to_dict(entries)
    except (IndexError, TypeError) as e:
        if isinstance(e, TypeError) and 'unhashable' in str(e):
            msg = "a list cannot be a dictionary key"
        else:
            msg = "each entry of a dictionary literal must be a [key value] pair"
        cls = compat(KlongSyntaxError, IndexError if isinstance(e, IndexError) else TypeError)
        raise cls(msg, span=span, hint="write :{[key1 value1] [key2 value2]}") from e


def tk_raise(lx, tok, neg=False):
    """Raise what the legacy readers raised for the text of an ERROR token."""
    if tok[7] is ERR_CHAR_EOF:
        e = UnexpectedEOF(lx.text, len(lx.text), message=_CHAR_EOF)
        e.span = (tok[1], tok[2])
        raise e
    d = ('-' + tok[3]) if neg else tok[3]
    span = (tok[1] - 1 if neg else tok[1], tok[2])
    try:
        if '.' in d or 'e' in d:
            float(d)
        else:
            int(d)
    except ValueError as ex:
        raise malformed_number(d, str(ex), span) from ex
    raise malformed_number(d, tok[7], span)


def tk_comment_end(lx, st, marker):
    """
    The offset just after the end marker of the .comment("marker") call that
    starts at offset st and ends at the read position (see read_sys_comment).
    """
    try:
        return read_sys_comment(lx.text, lx.offset(), marker)
    except KlongSyntaxError as e:
        e.span = (st, lx.end())
        raise


def tk_peek_adverb(lx):
    """
    legacy: peek_adverb. If an adverb starts exactly at the read position,
    consume it and return its text, else return None. Two-character adverbs
    are tested before one-character ones.
    """
    j = lx.pos
    tok = lx.toks[j]
    if tok[4]:
        return None
    k = tok[0]
    if k == 'OP':
        v = tok[3]
        if v == '@':
            nt = lx.toks[j+1]
            if nt[0] == 'OP' and nt[3] == "'" and not nt[4]:
                lx.pos = j + 2
                return "@'"
            return None
        if v == "'" or v == '/' or v == '\\' or v == '\\~' or v == '\\*':
            lx.pos = j + 1
            return v
    elif k == 'COP':
        v = tok[3]
        if v == ":'" or v == ':/' or v == ':\\' or v == ':~' or v == ':*':
            lx.pos = j + 1
            return v
    return None


def tk_read_list(klong, lx, j, close):
    """
    legacy: read_list. Read the body of a list ('[', close 'RBRACK') or of a
    dictionary literal (':{', close 'RBRACE') in data mode, starting at token
    index j (just after the opener). Returns (index after the closer, list).
    Nothing is consumed: the caller sets lx.pos.

    In data mode every token is a datum: a name is a symbol, ';' is the
    string ";", an operator is a KGOp object, and '-' directly followed by a
    number is a negative number. The list ends at the closer or at end of
    input (an unterminated list is accepted).

    A string or char whose content is "[" is a datum like any other (ticket
    023; the legacy reader started a nested list there).
    """
    toks = lx.toks
    module = klong._module
    arr = []
    append = arr.append
    opener = toks[j-1]
    tok = toks[j]
    k = tok[0]
    while k == 'NEWLINE' and not tok[5]:
        j += 1
        tok = toks[j]
        k = tok[0]
    while k != close and k != 'EOF':
        if k == 'NEWLINE':
            # legacy: the skip before this read stopped at a newline that
            # follows a comment; the read itself then skips that newline and
            # reads whatever comes next as a datum, even the closer, and a
            # second newline-after-comment as ";".
            j += 1
            tok = toks[j]
            k = tok[0]
            while k == 'NEWLINE' and not tok[5]:
                j += 1
                tok = toks[j]
                k = tok[0]
            if k == 'EOF':
                break
        if k == 'NUM' or k == 'STR' or k == 'CHAR':
            append(tok[3])
            j += 1
        elif k == 'NAME' or k == 'QNAME':
            v = tok[3]
            append(reserved_fn_symbol_map.get(v) or KGSym(v if module is None or v[0] == '.' else f"{v}`{module}"))
            j += 1
        elif k == 'OP':
            nt = toks[j+1]
            nk = nt[0]
            if (tok[3] == '-' and not nt[4] and (nk == 'NUM' or nk == 'ERROR' or nk == 'CHAR')
                    and lx.text[nt[1]] != ':'):
                # legacy: read_neg. '-' and the digits after it are one number.
                if nk == 'NUM':
                    append(-nt[3])
                    j += 2
                elif nk == 'ERROR' and nt[7] is not ERR_CHAR_EOF:
                    tk_raise(lx, nt, neg=True)
                else:
                    # legacy: "-0cx" is the number -0 followed by the name cx
                    # (the number reader does not know the 0c prefix). Tokens
                    # cannot express that; the caller re-reads the whole list
                    # with the character reader.
                    raise _DataFallback()
            else:
                append(KGOp(tok[3], 0))
                j += 1
        elif k == 'LBRACK':
            j, q = tk_read_list(klong, lx, j + 1, 'RBRACK')
            append(q)
        elif k == 'DICT':
            j, q = tk_read_list(klong, lx, j + 1, 'RBRACE')
            append(KGCall(copy_lambda, args=tk_dict(lx, q, (tok[1], toks[j-1][2])), arity=0))
        elif k == 'COP' or k == 'CALLP':
            append(KGOp(tok[3], 0))
            j += 1
        elif k == 'ERROR':
            tk_raise(lx, tok)
        elif k == 'NEWLINE':
            append(';')
            j += 1
        else:
            # ; ( ) { } ] :[ :| are the strings they are written as
            append(tok[3])
            j += 1
        tok = toks[j]
        k = tok[0]
        while k == 'NEWLINE' and not tok[5]:
            j += 1
            tok = toks[j]
            k = tok[0]
    if k == close:
        j += 1
    else:
        lx.at_end = True    # ran into end of input; tk_read_data undoes this for a look-ahead
        tk_unterminated(lx, opener, close)
    return j, arr


def tk_read_data(klong, lx, j, consume):
    """
    Read the list or dictionary literal whose opener is the token at index j.
    With consume=True the read position moves past it and the value is
    returned (a Python list, or the dictionary literal node). With
    consume=False nothing moves: the legacy parser read a lexeme to look at it
    and then often left it unread, and a malformed number inside it raised
    all the same.
    """
    tok = lx.toks[j]
    close = 'RBRACK' if tok[0] == 'LBRACK' else 'RBRACE'
    at_end = lx.at_end
    try:
        try:
            e, q = tk_read_list(klong, lx, j + 1, close)
        finally:
            if not consume:
                lx.at_end = at_end
        if consume:
            lx.pos = e
        end = lx.toks[e-1][2]
    except _DataFallback:
        e, q = read_list(lx.text, ']' if close == 'RBRACK' else '}', i=tok[2], module=klong._module)
        if e >= len(lx.text) and lx.text[e-1:e] != (']' if close == 'RBRACK' else '}'):
            tk_unterminated(lx, tok, close)
        if consume:
            lx.pos = j + 1
            lx.seek(e)
            lx.at_end = e >= len(lx.text)
        end = e
    if close == 'RBRACE':
        q = KGCall(copy_lambda, args=tk_dict(lx, q, (tok[1], end)), arity=0)
        if consume:
            q.span = (tok[1], lx.end())
            q.source = lx.source
    return q


def tk_check(klong, lx, j):
    """
    legacy: a kg_read whose result is looked at and not consumed. Raise what
    reading the token at index j as a lexeme raised: a malformed number, 0c at
    end of input, or an error inside a list or dictionary literal.
    """
    tok = lx.toks[j]
    if tok[0] == 'ERROR':
        tk_raise(lx, tok)
    else:
        tk_read_data(klong, lx, j, False)


def tk_read_cond(klong, lx, st):
    """
    legacy: read_cond. The read position is just after ':[' (or ':|'); st is
    the offset of that token.

        c := ':[' ( e ';' e ':|' )* e ';' e ';' e ']'
    """
    toks = lx.toks
    r = []
    sp = []
    n = klong._expr(lx, True)
    r.append(';' if n is SEMI else n)
    sp.append(None if n is None else (lx.start, lx.end()))
    # Whitespace and newlines may come between the condition and its ';'
    # (ticket 031; the legacy reader wanted the ';' adjacent here, and only
    # here).
    j = lx.pos
    tok = toks[j]
    while tok[0] == 'NEWLINE' and not tok[5]:
        j += 1
        tok = toks[j]
    if tok[0] != 'SEMI':
        raise UnexpectedChar(lx.text, lx.offset(), ';')
    lx.pos = j + 1
    n = klong._expr(lx, True)
    r.append(';' if n is SEMI else n)
    sp.append(None if n is None else (lx.start, lx.end()))
    j = lx.pos
    tok = toks[j]
    while tok[0] == 'NEWLINE' and not tok[5]:
        j += 1
        tok = toks[j]
    if tok[0] == 'ELIF':
        lx.pos = j + 1
        n = tk_read_cond(klong, lx, tok[1])
        r.append(n)
        sp.append(n.span)
    else:
        if tok[0] != 'SEMI':
            raise UnexpectedChar(lx.text, tok[1], ';')
        lx.pos = j + 1
        n = klong._expr(lx, True)
        r.append(';' if n is SEMI else n)
        sp.append(None if n is None else (lx.start, lx.end()))
        j = lx.pos
        tok = toks[j]
        while tok[0] == 'NEWLINE' and not tok[5]:
            j += 1
            tok = toks[j]
        if tok[0] != 'RBRACK':
            raise UnexpectedChar(lx.text, tok[1], ']')
        lx.pos = j + 1
    c = KGCond(r)
    c.span = (st, lx.end())
    c.arg_spans = tuple(sp)
    c.source = lx.source
    return c


def tk_read_expr_array(klong, lx, st):
    """
    legacy: read_expr_array. The read position is just after '[;'; st is the
    offset of the '['. Each element between semicolons is a full expression.
    """
    toks = lx.toks
    r = []
    sp = []
    j = lx.pos
    tok = toks[j]
    while tok[0] == 'NEWLINE' and not tok[5]:
        j += 1
        tok = toks[j]
    lx.pos = j
    while True:
        k = tok[0]
        if k == 'EOF':
            lx.at_end = True
            tk_unterminated(lx, ('LBRACK', st, st + 1), 'RBRACK')
            break
        if k == 'RBRACK':
            break
        if k == 'NEWLINE':
            # legacy: the skip stopped at a newline after a comment; the
            # expression reader's own skip then passes it.
            lx.pos += 1
        expr = klong._expr(lx, True)
        if expr is not None:
            r.append(';' if expr is SEMI else expr)
            sp.append((lx.start, lx.end()))
        j = lx.pos
        tok = toks[j]
        while tok[0] == 'NEWLINE' and not tok[5]:
            j += 1
            tok = toks[j]
        if tok[0] == 'SEMI':
            j += 1
            tok = toks[j]
            while tok[0] == 'NEWLINE' and not tok[5]:
                j += 1
                tok = toks[j]
        lx.pos = j
    if tok[0] == 'RBRACK':
        lx.pos += 1
    a = KGExprArray(r)
    a.span = (st, lx.end())
    a.arg_spans = tuple(sp)
    a.source = lx.source
    return a
