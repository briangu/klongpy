"""
KlongPy errors: one hierarchy, source positions, and rendering.

    KlongError                 base class; KlongException is the same class
      KlongSyntaxError         kind "syntax"     (UnexpectedChar, UnexpectedEOF are subclasses)
      KlongNameError           kind "name"
      KlongArityError          kind "arity"
      KlongTypeError           kind "type"
      KlongShapeError          kind "shape"
      KlongRangeError          kind "range"
      KlongSystemError         kind "system"
      KlongPythonError         kind "python"
      KlongInternalError       kind "internal"
      KlongMemoryError         kind "memory"
      KlongInterrupt           kind "interrupt"
      KlongStaticError         kind "static"
      KlongRemoteError         kind "remote"

    KlongSyntaxWarning         a Warning, not an error: what the parser accepts
                               at strict level 0 and rejects at level 1 and above

``kind`` is the contract shared with other implementations (the conformance
suite asserts on it, never on message text); ``wire`` is the name the error
has on the IPC wire.

``str(e)`` is the one-line message. ``e.render()`` is the multi-line form with
the position, the source line and carets under the span. ``e.span`` is a
``(start, end)`` pair of offsets into ``e.source.text`` (a
``klongpy.lexer.Source``); both are None until known. The evaluator fills them
in on the way out (see ``KlongInterpreter.eval``): the innermost node that has
a span wins.

Library exceptions that escape an operator are turned into these classes by
``operator_error`` (the classification table is ``classify``). The class that
is raised then is *also* a subclass of the builtin it replaces (``compat``):
``1+"a"`` raises a ``KlongTypeError`` that ``except TypeError`` still catches.

This module imports nothing from klongpy at import time (klongpy.types imports
it); what it needs from other modules is imported inside the functions, which
only run once an error exists.
"""
import re

__all__ = [
    'KlongError', 'KlongException',
    'KlongSyntaxError', 'KlongNameError', 'KlongArityError', 'KlongTypeError',
    'KlongShapeError', 'KlongRangeError', 'KlongSystemError', 'KlongPythonError',
    'KlongInternalError', 'KlongMemoryError', 'KlongInterrupt', 'KlongStaticError',
    'KlongRemoteError', 'UnexpectedChar', 'UnexpectedEOF', 'RangeError', 'KlongSyntaxWarning',
    'ERROR_KINDS', 'compat', 'classify', 'describe', 'operator_name',
    'operator_error', 'python_error', 'system_error', 'needs_torch_backend',
]

_RED = "\x1b[31m"       # colorama.Fore.RED, the style klongpy/cli.py prints failures in
_RESET = "\x1b[0m"      # colorama.Style.RESET_ALL
_YELLOW = "\x1b[33m"    # warnings
_CLIP = 120             # widest source line shown by render()


def _rebuild(cls, pybase, args, state):
    """Unpickle helper: constructor signatures differ, so rebuild without calling one."""
    if pybase is not None:
        cls = compat(cls, pybase)
    e = cls.__new__(cls)
    e.args = args
    e.__dict__.update(state)
    return e


class KlongError(Exception):
    """
    Base class of every error KlongPy raises on purpose.

        kind      stable short name of the class of error ("type", "range", ...)
        message   one line, no position, no trailing period
        span      (start, end) offsets into source.text, or None
        source    the klongpy.lexer.Source the span refers to, or None
        hint      one optional line of advice
    """
    kind = 'error'
    wire = None
    _compat = None      # (klong class, builtin class) on the classes compat() makes

    def __init__(self, message='', *, span=None, source=None, hint=None):
        # Exception.__init__ by name: a compat class has a builtin exception
        # after this class in its MRO, whose __init__ must not see our arguments.
        Exception.__init__(self, message)
        self.message = message
        self.span = span
        self.source = source
        self.hint = hint

    def __str__(self):
        return self.message

    def __reduce__(self):
        c = type(self)._compat
        cls, pybase = c if c else (type(self), None)
        return (_rebuild, (cls, pybase, self.args, dict(self.__dict__)))

    def position(self):
        """(name, line, column) of the start of the span, 1-based, or None when there is no position."""
        if self.span is None or self.source is None:
            return None
        line, col = self.source.line_col(self.span[0])
        return self.source.name, line, col

    def render(self, color=False):
        """
        The error as text for a person:

            examples/demo.kg:3:6: type error: cannot apply + (Plus) to integer and string
              a::1+"x"
                  ^
              hint: convert with $ first

        Without a position only the "<kind> error: <message>" line (and the
        hint) are produced. A source read at the REPL has no name in the
        first line. color=True colours the first line only.
        """
        return _render(f"{self.kind} error: {self.message}", self.span, self.source, self.hint,
                       _RED if color else None)


def _render(head, span, source, hint, color=None):
    """The layout shared by KlongError.render and KlongSyntaxWarning.render."""
    lines = []
    if span is not None and source is not None:
        name = source.name
        line, col = source.line_col(span[0])
        where = f"{line}:{col}" if _is_repl_name(name) else f"{name}:{line}:{col}"
        head = f"{where}: {head}"
        text = source.line_text(line).replace('\t', ' ')
        start = col - 1
        # the span, clipped to its first line; at least one caret
        width = max(1, min(span[1] - span[0], len(text) - start))
        if len(text) > _CLIP:
            # a window of _CLIP columns around the caret
            lo = max(0, min(start - _CLIP // 2, len(text) - _CLIP))
            text = text[lo:lo + _CLIP]
            start -= lo
            width = max(1, min(width, _CLIP - start))
        lines.append("  " + text)
        lines.append("  " + " " * start + "^" * width)
    if hint:
        lines.append(f"  hint: {hint}")
    if color:
        head = f"{color}{head}{_RESET}"
    return "\n".join([head] + lines)


def _is_repl_name(name):
    return not name or name.startswith('<repl')


class KlongSyntaxWarning(Warning):
    """
    The parser accepted something it should not have: input left
    after a complete statement, a stray closing bracket, an operator without
    its operand, a string, list, dictionary or comment that runs to the end
    of the input. At strict level 0 the program still means what it always
    meant and this warning is issued through the warnings module when the
    program text is run; at strict level 1 and above the same condition is a
    KlongSyntaxError with the same message.

        message   one line, no position (str() of the warning)
        span      (start, end) offsets into source.text
        source    the klongpy.lexer.Source
        hint      one optional line of advice
        code      short stable name of the condition: "trailing-input",
                  "stray-closer", "missing-operand", "unterminated-string",
                  "unterminated-list", "unterminated-dict",
                  "unterminated-comment"

    ``python -W error::klongpy.KlongSyntaxWarning`` turns it into an exception.
    """
    kind = 'warning'

    def __init__(self, message='', *, span=None, source=None, hint=None, code=None):
        Warning.__init__(self, message)
        self.message = message
        self.span = span
        self.source = source
        self.hint = hint
        self.code = code

    def __str__(self):
        return self.message

    def __reduce__(self):
        return (_rebuild_warning, (self.message, self.span, self.source, self.hint, self.code))

    def position(self):
        """(name, line, column) of the start of the span, 1-based, or None when there is no position."""
        if self.span is None or self.source is None:
            return None
        line, col = self.source.line_col(self.span[0])
        return self.source.name, line, col

    def render(self, color=False):
        """
        The warning laid out like an error (KlongError.render), with
        "warning:" where an error has "<kind> error:".
        """
        return _render(f"warning: {self.message}", self.span, self.source, self.hint,
                       _YELLOW if color else None)


def _rebuild_warning(message, span, source, hint, code):
    return KlongSyntaxWarning(message, span=span, source=source, hint=hint, code=code)


class KlongSyntaxError(KlongError):
    """Anything the lexer or the parser rejects."""
    kind = 'syntax'
    wire = 'parse'


class KlongNameError(KlongError):
    """Undefined variable or function; assignment to an undeclared name in strict mode."""
    kind = 'name'
    wire = 'name'


class KlongArityError(KlongError):
    """Too many arguments; wrong argument count to a system function."""
    kind = 'arity'
    wire = 'rank'


class KlongTypeError(KlongError):
    """An operand of the wrong kind."""
    kind = 'type'
    wire = 'type'


class KlongShapeError(KlongError):
    """Lengths or shapes that do not agree."""
    kind = 'shape'
    wire = 'length'


class KlongRangeError(KlongError):
    """Index out of range; a value outside an operator's domain."""
    kind = 'range'
    wire = 'index'


class KlongSystemError(KlongError):
    """A system function failed for an outside reason; the OS error is __cause__."""
    kind = 'system'
    wire = 'os'


def needs_torch_backend(name):
    """The error of a system function that only the torch backend provides."""
    return KlongSystemError(f"{name} requires the torch backend",
                      hint="select it with kgpy --backend torch, or KlongInterpreter(backend='torch')")


class KlongPythonError(KlongError):
    """An exception inside user Python called through .py/.pyf/.pyc; the original is __cause__."""
    kind = 'python'
    wire = 'python'


class KlongInternalError(KlongError):
    """Any other exception escaping an operator: a bug in KlongPy."""
    kind = 'internal'
    wire = 'internal'


class KlongMemoryError(KlongError):
    """Allocation failed or a configured heap limit was reached."""
    kind = 'memory'
    wire = 'wsfull'


class KlongInterrupt(KlongError):
    """Evaluation stopped by Ctrl-C or an interrupt request."""
    kind = 'interrupt'
    wire = 'stop'


class KlongStaticError(KlongError):
    """A program is outside the static compile profile (kgpy compile c, kgpy check)."""
    kind = 'static'


class KlongRemoteError(KlongError):
    """An IPC peer answered with an error whose text is not one of our wire names."""
    kind = 'remote'


KlongException = KlongError     # the public name since before the hierarchy existed
RangeError = KlongRangeError    # the old klongpy.types.RangeError (constructed, never raised)

ERROR_KINDS = {c.kind: c for c in (
    KlongSyntaxError, KlongNameError, KlongArityError, KlongTypeError, KlongShapeError,
    KlongRangeError, KlongSystemError, KlongPythonError, KlongInternalError,
    KlongMemoryError, KlongInterrupt, KlongStaticError, KlongRemoteError)}


# ---------------------------------------------------------------------------
# Parser errors
# ---------------------------------------------------------------------------

_LEXEME = re.compile(r':?[A-Za-z.][A-Za-z0-9.]*|[0-9][0-9.]*|"[^"\n]{0,16}"?|:?.', re.S)


def _found_at(t, i, expected=None):
    """
    What the text holds at offset i, for "expected X but found Y", and its span.

    Whitespace is skipped to name the lexeme after it, unless that lexeme is
    the expected one: then the whitespace itself is what is wrong (the parser
    wanted it adjacent, e.g. the ')' of "f(1 )").
    """
    n = len(t)
    j = i
    while j < n and t[j] != '\n' and t[j].isspace():
        j += 1
    if j >= n:
        return "end of input", (n, n)
    if t[j] == '\n':
        return "end of line", (j, j + 1)
    m = _LEXEME.match(t, j)
    lexeme = m.group()
    if j > i and expected is not None and lexeme[:len(expected)] == expected:
        return "whitespace", (i, j)
    return f"'{lexeme}'", (j, j + len(lexeme))


class UnexpectedChar(KlongSyntaxError):
    """
    The parser expected the character c at offset i of the text t.

    The message says what was expected and what was found; the text the class
    produced before the hierarchy existed is kept as ``legacy_text``.
    """
    def __init__(self, t, i, c, source=None):
        found, span = _found_at(t, i, c)
        super().__init__(f"expected '{c}' but found {found}", span=span, source=source)
        self.t = t
        self.i = i
        self.expected = c
        self.legacy_text = f"t: {t[i-10:i+10]} pos: {i} char: {c}"


class UnexpectedEOF(KlongSyntaxError):
    """The text t ended at offset i where more was needed."""
    def __init__(self, t, i, source=None, message="unexpected end of input"):
        n = len(t)
        super().__init__(message, span=(min(i, n), n), source=source)
        self.t = t
        self.i = i
        self.legacy_text = f"t: {t[i-10:]} pos: {i}"


# ---------------------------------------------------------------------------
# Classes that are both a Klong error and the builtin they replace
# ---------------------------------------------------------------------------

_COMPAT = {}


def _builtin_base(cls):
    """The nearest builtin exception class of cls (cls itself when it is one)."""
    for c in cls.__mro__:
        if c.__module__ == 'builtins' and c not in (Exception, BaseException, object):
            return c
    return None


def compat(kcls, pycls):
    """
    The subclass of the Klong error class kcls that is also a pycls, so that
    code written against the Python exception KlongPy used to let through
    (``except IndexError``) keeps working. It has the name of kcls. The same
    class is returned for the same pair; kcls itself when no such class can
    be made.
    """
    key = (kcls, pycls)
    c = _COMPAT.get(key)
    if c is None:
        if pycls is None or issubclass(kcls, pycls):
            c = kcls
        else:
            try:
                c = type(kcls.__name__, (kcls, pycls), {
                    '__module__': kcls.__module__, '__qualname__': kcls.__qualname__,
                    '__doc__': kcls.__doc__, '_compat': key})
            except TypeError:       # incompatible layouts
                c = kcls
        _COMPAT[key] = c
    return c


# ---------------------------------------------------------------------------
# Describing values and operators
# ---------------------------------------------------------------------------

def describe(value):
    """
    A short description of a value for an error message: integer, float,
    string, char, symbol, dict, function, list of 3, 2x3 matrix, :undefined.
    Called only once an error exists.
    """
    from .types import (KGSym, KGChar, KGFn, KGLambda, KGFnWrapper, KGOp, KGAdverb,
                        KGChannel, KGUndefined)
    if value is None:
        return "nothing"
    if isinstance(value, KGUndefined):
        return ":undefined"
    if isinstance(value, bool):
        return "integer"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return ":undefined" if value != value or value in (float('inf'), float('-inf')) else "float"
    if isinstance(value, KGChar):
        return "char"
    if isinstance(value, KGSym):
        return "symbol"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, (KGFn, KGLambda, KGFnWrapper, KGOp, KGAdverb)):
        return "function"
    if isinstance(value, KGChannel):
        return "channel"
    if isinstance(value, (list, tuple)):
        return f"list of {len(value)}" if len(value) else "empty list"
    shape = getattr(value, 'shape', None)
    if shape is not None and hasattr(value, 'dtype'):
        shape = tuple(shape)
        dtype = str(value.dtype)
        if len(shape) == 0:
            # a numpy scalar or a 0-dim array or tensor
            if dtype == 'object' and hasattr(value, 'item'):
                return describe(value.item())
            if 'int' in dtype or 'bool' in dtype:
                return "integer"
            if 'float' in dtype:
                try:
                    return describe(float(value))
                except (TypeError, ValueError):
                    return "float"
            return dtype
        if len(shape) == 1 or dtype == 'object':
            return f"list of {shape[0]}" if shape[0] else "empty list"
        dims = "x".join(str(d) for d in shape)
        return f"{dims} matrix" if len(shape) == 2 else f"{dims} array"
    if callable(value):
        return "function"
    return type(value).__name__


_HEADER = re.compile(r'^\s+(\S+(?: \S+)?)\s+\[([^\]]+)\]\s*$', re.M)
_OP_NAMES = None


def _operator_names():
    """
    {(glyph, arity): name} for operators and {glyph: {arity: name}} folded in
    for adverbs, read once from the bracketed names in the docstring headers
    of klongpy.dyads, klongpy.monads and klongpy.adverbs ("a+b   [Plus]").
    """
    global _OP_NAMES
    if _OP_NAMES is None:
        from . import adverbs, dyads, monads
        names = {}
        for module, prefix, arity in ((dyads, 'eval_dyad_', 2), (monads, 'eval_monad_', 1),
                                      (adverbs, 'eval_adverb_', 0), (adverbs, 'eval_dyad_adverb_', 0)):
            for attr, fn in vars(module).items():
                if not attr.startswith(prefix) or not getattr(fn, '__doc__', None):
                    continue
                for form, name in _HEADER.findall(fn.__doc__):
                    if arity == 2 and len(form) > 2 and form[0] == 'a' and form[-1] == 'b':
                        key = (form[1:-1], 2)
                    elif arity == 1 and len(form) > 1 and form[-1] == 'a' and ' ' not in form:
                        key = (form[:-1], 1)
                    elif arity == 0 and form.startswith('a f') and form[-1] == 'b':
                        key = (form[3:-1], 'adverb', 2)
                    elif arity == 0 and form[0] == 'f' and form[-1] == 'a' and ' ' not in form:
                        key = (form[1:-1], 'adverb', 1)
                    else:
                        continue
                    names.setdefault(key, name)
        _OP_NAMES = names
    return _OP_NAMES


def operator_name(glyph, arity):
    """The documented name of an operator ("Plus" for ('+', 2)), or None."""
    return _operator_names().get((glyph, arity))


def adverb_name(glyph, arity):
    """The documented name of an adverb ("Over" for ('/', 1)), or None."""
    names = _operator_names()
    return names.get((glyph, 'adverb', arity)) or names.get((glyph, 'adverb', 3 - arity))


# ---------------------------------------------------------------------------
# Turning library exceptions into Klong errors
# ---------------------------------------------------------------------------

_SHAPE_WORDS = ('shape', 'size', 'broadcast', 'inhomogeneous')


def classify(exc):
    """
    The Klong error class for an exception that escaped an operator:

        IndexError                                        KlongRangeError
        TypeError (numpy UFuncTypeError is one),
          a backend UnsupportedDtypeError                 KlongTypeError
        ValueError or RuntimeError whose text mentions
          shape, size, broadcast or "inhomogeneous"       KlongShapeError
        ZeroDivisionError, OverflowError, other ValueError  KlongRangeError
        MemoryError                                       KlongMemoryError
        RecursionError                                    KlongRangeError
        an exception class KlongPy defines (autograd)     KlongShapeError or KlongRangeError
        anything else                                     KlongInternalError

    numpy and torch exceptions are recognised by their builtin base class and
    their text, never by importing the library: torch raises a plain
    RuntimeError for sizes that do not match.
    """
    if isinstance(exc, KlongError):
        return type(exc)
    if isinstance(exc, IndexError):
        return KlongRangeError
    if isinstance(exc, RecursionError):
        return KlongRangeError
    if isinstance(exc, MemoryError):
        return KlongMemoryError
    if isinstance(exc, TypeError) or type(exc).__name__.endswith('UnsupportedDtypeError'):
        return KlongTypeError
    if isinstance(exc, (ValueError, RuntimeError)):
        text = str(exc).lower()
        if any(w in text for w in _SHAPE_WORDS):
            return KlongShapeError
        return KlongRangeError if isinstance(exc, ValueError) else KlongInternalError
    if isinstance(exc, ArithmeticError):
        return KlongRangeError
    if _is_own(exc):
        text = str(exc).lower()
        return KlongShapeError if any(w in text for w in _SHAPE_WORDS) else KlongRangeError
    return KlongInternalError


def _is_own(exc):
    """
    True for an exception class KlongPy defines itself outside the hierarchy
    (the autograd errors, a backend's UnsupportedDtypeError). Such an error
    was raised on purpose with a message of its own, which is kept; the Klong
    error it becomes is still an instance of that class.
    """
    cls = type(exc)
    return (cls.__module__ or '').startswith('klongpy.') and cls.__name__ != 'AtomicShapeMismatch'


def is_passthrough(exc):
    """
    True for an exception operators must not relabel as their own failure:
    one a system function raised (see system_error), or an OSError, which no
    operator raises itself (a TimeoutError from a signal handler, a broken
    connection inside a function the operator called).
    """
    return getattr(exc, '_klong_passthrough', False) or isinstance(exc, OSError)


def _odd_member(value, depth=0):
    """
    For a type error on a list: what the list holds that is not a number
    (":undefined", "a string", "a symbol"), or None. Looks at the first
    thousand members and three levels down.
    """
    if isinstance(value, (str, dict)):
        return None
    dtype = getattr(value, 'dtype', None)
    if dtype is not None:
        if getattr(value, 'ndim', 0) == 0 or getattr(dtype, 'kind', 'f') not in 'OUS':
            return None
    elif not isinstance(value, (list, tuple)):
        return None
    for n, el in enumerate(value):
        if n >= 1000:
            break
        d = describe(el)
        if d == ':undefined' or (isinstance(el, str) and str.__str__(el) == 'undefined'):
            return ':undefined'
        if d.startswith('list of') or d.endswith(('matrix', 'array')):
            if depth < 3:
                d = _odd_member(el, depth + 1)
                if d:
                    return d
        elif d not in ('integer', 'float', 'empty list'):
            return ('an ' if d[0] in 'aeiou' else 'a ') + d
    return None


def operator_error(exc, glyph, name, operands):
    """
    The Klong error for the exception exc that escaped the operator written
    glyph and named name, applied to operands (the values). The caller raises
    it ``from exc``.

        cannot apply + (Plus) to integer and string
    """
    kcls = classify(exc)
    if _is_own(exc):
        cls = compat(kcls, type(exc))
        try:
            return cls(str(exc))
        except Exception:
            return kcls(str(exc))
    if isinstance(exc, RecursionError):
        message = "expression nested too deeply"
    else:
        op = f"{glyph} ({name})" if name else f"{glyph}"
        message = f"cannot apply {op} to {' and '.join(describe(v) for v in operands)}"
        if kcls is KlongTypeError:
            for v in operands:
                odd = _odd_member(v)
                if odd:
                    message += f": the list holds {odd}"
                    break
        elif kcls is KlongMemoryError:
            message += ": out of memory"
        elif kcls is KlongInternalError:
            message += f": {type(exc).__name__}: {exc}"
    err = compat(kcls, _builtin_base(type(exc)))(message)
    if glyph == '_' and len(operands) == 2 and isinstance(operands[0], dict) \
            and not isinstance(operands[1], dict) and kcls is KlongTypeError:
        err.hint = 'to remove a key write the key on the left and the dictionary on the right: "a"_d'
    return err


def python_error(exc):
    """The KlongPythonError for an exception raised by user Python (.py, .pyf, .pyc)."""
    text = str(exc)
    message = f"{type(exc).__name__}: {text}" if text else type(exc).__name__
    cls = compat(KlongPythonError, type(exc))
    if cls is KlongPythonError:
        cls = compat(KlongPythonError, _builtin_base(type(exc)))
    try:
        e = cls(message)
    except Exception:
        e = KlongPythonError(message)
    return e


def system_error(exc):
    """
    What a system function's exception becomes. An OSError (file missing,
    connection refused) becomes a KlongSystemError that is still an instance
    of the same OSError class and carries errno and filename. Any other
    exception is returned as it is, marked so that an operator it passes
    through (f'x, f@x) does not relabel it as its own failure.
    """
    if isinstance(exc, OSError):
        base = _builtin_base(type(exc)) or OSError
        text = str(exc)
        if exc.strerror:
            text = f"{exc.strerror}: {exc.filename}" if exc.filename is not None else exc.strerror
        try:
            e = compat(KlongSystemError, base)(text)
            e.errno = exc.errno
            e.strerror = exc.strerror
            e.filename = exc.filename
        except Exception:
            e = KlongSystemError(text)
        return e
    try:
        exc._klong_passthrough = True
    except Exception:
        pass
    return exc

