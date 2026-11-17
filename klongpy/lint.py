"""
kgpy --lint: a static check for the classic Klong traps.

    lint_source(text, name="<string>") -> [Diagnostic, ...]
    lint_file(path)                    -> [Diagnostic, ...]
    lint_main(files, select, ignore)   -> exit status (the body of ``kgpy --lint``)
    RULES                              code -> Rule(code, severity, title)

The linter reads a program; it never runs it. It works on the token list of
klongpy.lexer and on the parse tree (function metadata of ticket 032 only:
``max_arg``, ``arity``, ``params``). No user code is evaluated, no file is
executed (``.l("file")`` is resolved by reading and parsing that file), no
Python module is imported, no REPL, event loop or thread is started. Only
numpy is needed (as for the interpreter itself).

Output of ``kgpy --lint FILE...``: one line per finding,

    file:line:col: CODE message

then a summary line. Exit status 1 if any E finding is reported, 2 if a file
could not be read, else 0. See docs/lint.md for the rule table.

Rules, by code
--------------

    W001  _ between name characters (my_var); not x_a, a_x (Drop with an argument)
    W002  name used before its definition, or never defined
    W003  declared local never used (and not read by a function it calls)
    W010  leading - followed by a number and a dyad (-1+2); not for * and %
    W011  function uses y without x, or z without y, and is called with too few
    W012  bare name in a [...] list that is a defined variable or local
    W013  :[ in the else position of :[ (the :| chain)
    W014  whitespace between a name or } and ( or an adverb
    W015  newline directly after a comment inside a body or conditional
    W016  : directly before a digit
    W017  ; a bracket or an operator inside a [...] data list
    W018  number immediately followed by a name (10c, 2x)
    W019  assignment to x, y or z (that is not declared a local)
    W020  a two-element list joined to a dictionary literal with ,
    E001  anything the parser rejects at strict level 1
    E002  call with more arguments than the callee takes

The notes in docs/lint.md say where a rule is narrower than the design table
and why. Rules that cannot be decided from the text are reduced to the part
that can: W002 and W012 compare textual order with the order statements run
(top level only, right to left inside a statement), W011 needs a call that
shows the problem, W020 needs a list of two elements written out, and E002
knows only the functions bound once to a literal ``name::{...}``.

Suppression: a comment ``:"lint:ignore W012"`` on a line removes that code
on that line (several codes may follow, separated by spaces or commas).

How the pass is built
---------------------

1. The text is tokenised (``.comment("X")`` bodies skipped as the parser does)
   and the comments are kept as trivia.
2. A bracket scan assigns every token to a frame: top level, function body
   ``{}``, parenthesis, call, evaluated list ``[;...]``, conditional ``:[...]``,
   data list ``[...]`` or dictionary body ``:{...}``. Data frames hold data,
   not code (frontend.md section 5), which is what lets W012 and W017 tell a
   symbol from a use. A function body that starts with ``[a;b];`` declares
   locals.
3. The program is parsed at strict level 0 (tree, warnings) and at strict
   level 1 (errors); both stop at the first problem, so parsing resumes at the
   next top-level statement. The tree is used for the metadata of function
   literals; every other rule works on tokens and frames, so it also runs when
   the parse fails.
4. One pass over the frames builds the definition table (top-level
   ``name::``, locals, parameters, ``.l`` imports) that W002, W012 and E002
   consult.

Cross-file names. ``.l("file")`` with a literal argument adds the names that
file defines at its top level, read the same way (recursively, without
running it). The file is looked up as ``.l`` does (``x`` and ``x.kg`` in each
directory of KLONGPATH, default ``.:lib``), tried relative to the linted
file's directory first, then to the current directory, then in klongpy/lib.
``.py``, ``.pyf``, ``.pya``, ``.E``, a ``.l`` whose argument is not a string
literal and a ``.l`` whose file cannot be found "define unknown names": W002
is switched off for the file. Everything else keeps working.
"""
import os
import re
import sys
from bisect import bisect_right
from dataclasses import dataclass

from .lexer import Lexer, Source

__all__ = ['Diagnostic', 'Rule', 'RULES', 'lint_source', 'lint_file', 'lint_main', 'format_diagnostic',
           'parse_codes', 'parse_names']


# ---------------------------------------------------------------------------
# Rules and diagnostics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    code: str
    severity: str       # "warning" or "error"
    title: str


RULES = {r.code: r for r in (
    Rule('W001', 'warning', "_ between name characters: it is the Drop operator"),
    Rule('W002', 'warning', "name used before its definition, or never defined"),
    Rule('W003', 'warning', "declared local never used"),
    Rule('W010', 'warning', "leading - followed by a number and a dyad negates the whole expression"),
    Rule('W011', 'warning', "function uses y without x, or z without y"),
    Rule('W012', 'warning', "bare name in a list is the symbol, not the value"),
    Rule('W013', 'warning', ":[ in the else position of :[ can be the :| chain"),
    Rule('W014', 'warning', "whitespace between a name or } and ( or an adverb"),
    Rule('W015', 'warning', "newline after a comment inside a body or conditional acts as ;"),
    Rule('W016', 'warning', ": directly before a digit is ignored"),
    Rule('W017', 'warning', "; a bracket or an operator inside a data list is data"),
    Rule('W018', 'warning', "number immediately followed by a name"),
    Rule('W019', 'warning', "assignment to x, y or z"),
    Rule('W020', 'warning', "a two-element list joins into a dictionary with ,"),
    Rule('E001', 'error', "the parser rejects this at strict level 1"),
    Rule('E002', 'error', "call with more arguments than the callee takes"),
)}


@dataclass(frozen=True)
class Diagnostic:
    """
    One finding.

        code      "W001" ... "E002"
        severity  "warning" or "error"
        span      (start, end) offsets into source.text
        message   one line, no position
        hint      one line of advice, or None
        source    the klongpy.lexer.Source the span refers to

    line and col (1-based) are derived from the span.
    """
    code: str
    severity: str
    span: tuple
    message: str
    hint: object
    source: object

    @property
    def line(self):
        return self.source.line_col(self.span[0])[0]

    @property
    def col(self):
        return self.source.line_col(self.span[0])[1]

    def __str__(self):
        return format_diagnostic(self)


def format_diagnostic(d, hint=False):
    """``file:line:col: CODE message`` (and, with hint=True, an indented ``hint:`` line)."""
    line, col = d.source.line_col(d.span[0])
    s = f"{d.source.name}:{line}:{col}: {d.code} {d.message}"
    if hint and d.hint:
        s += f"\n  hint: {d.hint}"
    return s


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

_OPERAND_END = frozenset(('NUM', 'CHAR', 'STR', 'NAME', 'QNAME', 'RPAREN', 'RBRACK', 'RBRACE'))
_ADVERB_OPS = frozenset(("'", "/", "\\", "\\~", "\\*"))
_ADVERB_COPS = frozenset((":\\", ":'", ":/", ":~", ":*"))
_DATA_KINDS = ('data', 'dict')
_ARGS = ('x', 'y', 'z')
_EXTRA_BUILTINS = frozenset(('.comment', '.f', '.helpdb', '.os.env', '.os.argv', '.help_topic', '.system', '.cli.h'))
# calls that bring in names the linter cannot see
_UNKNOWN_DEFINERS = frozenset(('.py', '.pyf', '.pya', '.E', '.hw'))
_NOT_DYAD = frozenset(('*', '%'))   # -1*x is (-1)*x: the sign commutes, nothing to warn about


def _comment_end(text, i, marker):
    j = text.find(marker, i)
    if j < 0 or not marker:
        return -1
    while text.startswith(marker, j + 1):
        j += 1
    return j + len(marker)


def _tokenize(source):
    """The tokens (ending with EOF) and the comments of a source, skipping .comment("X") bodies."""
    lx = Lexer(source)
    out = []
    text = lx.text
    while True:
        tok = lx.next()
        out.append(tok)
        k = tok[0]
        if k == 'EOF':
            return out, sorted(lx.trivia, key=lambda t: t[1])
        if k == 'RPAREN' and not tok[4] and len(out) >= 4:
            a, b, c = out[-4], out[-3], out[-2]
            if (a[0] in ('NAME', 'QNAME') and a[3] == '.comment' and b[0] in ('LPAREN', 'CALLP') and not b[4]
                    and c[0] == 'STR' and not c[6]):
                end = _comment_end(text, tok[2], c[3])
                if end >= 0:
                    lx.seek(end)


_builtins = None


def _builtin_names():
    """The names a fresh interpreter binds (system functions and variables), plus what kgpy adds."""
    global _builtins
    if _builtins is None:
        from .interpreter import KlongInterpreter
        names = set(_EXTRA_BUILTINS)
        for d in KlongInterpreter()._context._context:
            names.update(str(k) for k in d.keys())
        _builtins = frozenset(names)
    return _builtins


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

class _Frame:
    """A bracketed region of the token list (or the whole program)."""
    __slots__ = ('kind', 'open', 'close', 'parent', 'fn', 'locals', 'decl', 'params', 'is_decl',
                 'is_target', 'is_header', 'meta')

    def __init__(self, kind, open_, parent):
        self.kind = kind            # top fn paren call elist edict cond data dict
        self.open = open_           # index of the opening token (-1 for top)
        self.close = None           # index of the closing token, None if never closed
        self.parent = parent
        self.fn = self if kind == 'fn' else (parent.fn if parent is not None else None)
        self.locals = ()            # fn: declared local names
        self.decl = None            # fn: (start, end) offsets of the locals declaration
        self.params = None          # fn: named parameters (name(a;b)::{...}), or None
        self.is_decl = False        # data: this list is a locals declaration
        self.is_target = False      # data: this list is the target of ::
        self.is_header = False      # call: the name(a;b) of a named-parameter definition
        self.meta = None            # fn: the parse-tree node of the literal, when the parse got there


def _scan(toks):
    """Assign every token to a frame. Returns (frames, ctx, open_frame)."""
    n = len(toks)
    top = _Frame('top', -1, None)
    frames = [top]
    ctx = [top] * n
    open_frame = {}
    stack = [top]
    decl_at = {}        # token index of a locals declaration's [ -> fn frame
    params_at = {}      # token index of a { -> parameter names

    def push(kind, i, cur):
        f = _Frame(kind, i, cur)
        frames.append(f)
        stack.append(f)
        open_frame[i] = f
        return f

    def close(kinds, i):
        for p in range(len(stack) - 1, 0, -1):
            if stack[p].kind in kinds:
                f = stack[p]
                del stack[p:]
                f.close = i
                ctx[i] = f
                return f
        return None

    for i in range(n):
        t = toks[i]
        k = t[0]
        cur = stack[-1]
        ctx[i] = cur
        if cur.kind in _DATA_KINDS:
            if k == 'LBRACK':
                f = push('data', i, cur)
                if i in decl_at:
                    f.is_decl = True
            elif k == 'DICT':
                push('dict', i, cur)
            elif k == 'RBRACK' and cur.kind == 'data':
                f = close(('data',), i)
                if f is not None and i + 1 < n and toks[i + 1][0] == 'COP' and toks[i + 1][3] == '::':
                    f.is_target = True
            elif k == 'RBRACE' and cur.kind == 'dict':
                close(('dict',), i)
            continue
        if k == 'LBRACE':
            f = push('fn', i, cur)
            if i in params_at:
                f.params = params_at[i]
            _find_decl(toks, i, f, decl_at)
        elif k == 'RBRACE':
            close(('fn', 'edict'), i)
        elif k == 'LPAREN' or k == 'CALLP':
            prev = toks[i - 1][0] if i else None
            call = k == 'CALLP' or (not t[4] and prev in ('NAME', 'QNAME', 'RBRACE'))
            push('call' if call else 'paren', i, cur)
        elif k == 'RPAREN':
            f = close(('call', 'paren'), i)
            if f is not None and f.kind == 'call' and i + 2 < n and toks[i + 1][0] == 'COP' and toks[i + 1][3] == '::' \
                    and toks[i + 2][0] == 'LBRACE' and f.open > 0 and toks[f.open - 1][0] == 'NAME':
                names = []
                ok = True
                for j in range(f.open + 1, i):
                    kj = toks[j][0]
                    if kj == 'NAME':
                        names.append(toks[j][3])
                    elif kj not in ('SEMI', 'NEWLINE'):
                        ok = False
                        break
                if ok:
                    f.is_header = True
                    params_at[i + 2] = tuple(names)
        elif k == 'COND':
            push('cond', i, cur)
        elif k == 'LBRACK':
            if i + 1 < n and toks[i + 1][0] == 'SEMI' and not toks[i + 1][4]:
                push('elist', i, cur)
            else:
                f = push('data', i, cur)
                if i in decl_at:
                    f.is_decl = True
        elif k == 'RBRACK':
            close(('cond', 'elist'), i)
        elif k == 'DICT':
            # :{;k;v;...} is the evaluated dictionary, as [;a;b] is the evaluated list: its body is code
            if i + 1 < n and toks[i + 1][0] == 'SEMI' and not toks[i + 1][4]:
                push('edict', i, cur)
            else:
                push('dict', i, cur)
    return frames, ctx, open_frame


def _find_decl(toks, i, fn, decl_at):
    """If the function opened at token i starts with a locals declaration [a;b]; set fn.locals and fn.decl."""
    n = len(toks)
    j = i + 1
    while toks[j][0] == 'NEWLINE':
        j += 1
    if toks[j][0] != 'LBRACK':
        return
    names = []
    q = j + 1
    while q < n and toks[q][0] in ('NAME', 'QNAME', 'SEMI', 'NEWLINE'):
        if toks[q][0] != 'SEMI' and toks[q][0] != 'NEWLINE':
            names.append(toks[q][3])
        q += 1
    if q >= n or toks[q][0] != 'RBRACK' or not names:
        return
    r = q + 1
    if toks[r][0] == 'COP' and toks[r][3] == '::':
        # head destructuring, {[a b]::x; ...}: the names are locals (when a statement follows)
        fn.locals = tuple(names)
        fn.decl = (toks[j][1], toks[q][2])
        decl_at[j] = fn
        return
    while toks[r][0] == 'NEWLINE':
        r += 1
    if toks[r][0] != 'SEMI':
        return
    r += 1
    while toks[r][0] == 'NEWLINE':
        r += 1
    if toks[r][0] in ('RBRACE', 'EOF'):
        return
    fn.locals = tuple(names)
    fn.decl = (toks[j][1], toks[q][2])
    decl_at[j] = fn
    # {[t];[a b]::x; ...}: a head destructuring after the locals list declares its names too
    if toks[r][0] == 'LBRACK':
        more = []
        q2 = r + 1
        while q2 < n and toks[q2][0] in ('NAME', 'SEMI', 'NEWLINE'):
            if toks[q2][0] == 'NAME':
                more.append(toks[q2][3])
            q2 += 1
        if q2 + 1 < n and toks[q2][0] == 'RBRACK' and toks[q2 + 1][0] == 'COP' and toks[q2 + 1][3] == '::' and more:
            fn.locals = tuple(names) + tuple(more)
            fn.decl = (toks[j][1], toks[q2][2])


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _tree_children(n):
    from .types import KGAdverb, KGFn
    if isinstance(n, KGAdverb):
        yield n.a
    elif isinstance(n, KGFn):
        yield n.a
        a = n.args
        if isinstance(a, list):
            yield from a
        elif a is not None and not isinstance(a, dict):
            yield a
    elif isinstance(n, list):
        yield from n


def _fn_nodes(stmts):
    """{offset of the { : tree node} for every function literal in the statements (iterative)."""
    from .types import KGFn
    out = {}
    stack = list(stmts)
    seen = set()
    while stack:
        n = stack.pop()
        if n is None or isinstance(n, (str, int, float, bool)) or id(n) in seen:
            continue
        seen.add(id(n))
        if isinstance(n, KGFn) and getattr(n, 'max_arg', None) is not None and n.span is not None:
            out.setdefault(n.span[0], n)
        try:
            stack.extend(_tree_children(n))
        except Exception:
            pass
    return out


def _parse_level(source, strict, toks, ctx):
    """
    Parse a source at a strict level, resuming after each problem at the next
    top-level statement. Returns (statements, [(span, message, hint)]).

    At level 0 the parser warns about what level 1 rejects and goes on, but its
    reading stops at the first trailing input; at level 1 it raises. Either
    way the rest of the file is parsed from the next top-level separator.
    """
    from .interpreter import KlongInterpreter
    klong = KlongInterpreter(strict=strict, backend="numpy")   # parse only: the same result whatever the default backend
    text = source.text
    n = len(text)
    stmts = []
    problems = []
    pos = 0
    guard = 0
    starts = [t[1] for t in toks]
    while pos < n and guard < 10000:
        guard += 1
        stop = None
        try:
            r = klong.prog(source, pos)
        except Exception as e:
            span = getattr(e, 'span', None) or (pos, pos)
            problems.append((tuple(span), str(e) or type(e).__name__, getattr(e, 'hint', None)))
            stop = span[0]
        else:
            stmts.extend(r[1])
            for w in getattr(r, 'warnings', ()):
                problems.append((tuple(w.span), w.message, w.hint))
            end = r[0]
            if r.warnings:
                # the reading may have stopped early: resume after the offending statement
                stop = min([w.span[0] for w in r.warnings] + [end]) if end < n else None
                if end >= n:
                    break
            elif end >= n:
                break
            else:
                stop = end
        if stop is None:
            break
        nxt = _next_boundary(toks, ctx, starts, stop)
        if nxt is None or nxt <= pos:
            break
        pos = nxt
    return stmts, problems


def _next_boundary(toks, ctx, starts, offset):
    """Offset just after the first top-level ; or newline at or after offset, or None."""
    i = bisect_right(starts, offset) - 1
    if i < 0:
        i = 0
    n = len(toks)
    while i < n:
        t = toks[i]
        if t[1] >= offset and ctx[i].kind == 'top' and t[0] in ('SEMI', 'NEWLINE'):
            return t[2]
        i += 1
    return None


# ---------------------------------------------------------------------------
# The analysis of one file
# ---------------------------------------------------------------------------

class _Def:
    __slots__ = ('name', 'idx', 'start', 'end', 'fn', 'top')

    def __init__(self, name, idx, start, end, fn, top):
        self.name = name
        self.idx = idx
        self.start = start  # index of the first token of the defining statement
        self.end = end      # index of the token that ends the defining statement
        self.fn = fn        # the function frame of the right-hand side, if it is a literal
        self.top = top      # defined at top level (not inside a function body)


_MAX_DEPTH = 8


class _Analysis:
    """Tokens, frames, parse result and definition table of one source."""

    def __init__(self, source, base_dir, resolve=True, seen=(), depth=0, full=True):
        self.source = source
        self.text = source.text
        self.base_dir = base_dir
        self.resolve = resolve
        self.seen = seen
        self.depth = depth
        self.toks, self.trivia = _tokenize(source)
        self.frames, self.ctx, self.open_frame = _scan(self.toks)
        self.stmts = []
        self.problems = []      # E001 candidates: (span, message, hint)
        self.parsed = False
        self._parse(full)
        self._definitions()

    # -- parse ------------------------------------------------------------
    def _parse(self, full):
        try:
            stmts, problems = _parse_level(self.source, 0, self.toks, self.ctx)
        except Exception:
            return
        self.stmts = stmts
        self.parsed = True
        nodes = _fn_nodes(stmts)
        for f in self.frames:
            if f.kind == 'fn':
                f.meta = nodes.get(self.toks[f.open][1])
        if full:
            self.problems = list(problems)
            try:
                _, problems1 = _parse_level(self.source, 1, self.toks, self.ctx)
            except Exception:
                problems1 = []
            seen = {p[0][0] for p in self.problems}
            for p in problems1:
                if p[0][0] not in seen:
                    seen.add(p[0][0])
                    self.problems.append(p)

    # -- definitions --------------------------------------------------------
    def stmt_end(self, i):
        """Index of the token that ends the statement containing token i."""
        toks, ctx = self.toks, self.ctx
        f = ctx[i]
        n = len(toks)
        top = f.kind == 'top'
        j = i + 1
        while j < n:
            if ctx[j] is f:
                k = toks[j][0]
                if k == 'SEMI' or (top and k == 'NEWLINE') or k == 'EOF' or j == f.close:
                    return j
            elif f.close is not None and j == f.close:
                return j
            j += 1
        return n - 1

    def stmt_start(self, i):
        """Index of the first token of the statement containing token i."""
        toks, ctx = self.toks, self.ctx
        f = ctx[i]
        top = f.kind == 'top'
        j = i - 1
        while j >= 0:
            if j == f.open:
                return j + 1
            if ctx[j] is f:
                k = toks[j][0]
                if k == 'SEMI' or (top and k == 'NEWLINE'):
                    return j + 1
            j -= 1
        return 0

    def _definitions(self):
        toks, ctx = self.toks, self.ctx
        n = len(toks)
        self.defs = {}              # name -> [_Def]  (top level)
        self.fn_assigned = set()    # names assigned inside function bodies
        self.local_names = set()    # every declared local and named parameter
        self.imports = []           # (name effective index, [names]) for top-level .l
        self.import_anywhere = set()
        self.unknown = False
        self.import_limits = {}
        for f in self.frames:
            if f.kind == 'fn':
                self.local_names.update(f.locals)
                if f.params:
                    self.local_names.update(f.params)
        for i in range(n):
            t = toks[i]
            f = ctx[i]
            if f.kind in _DATA_KINDS:
                continue
            if t[0] != 'NAME':
                continue
            nm = t[3]
            nxt = toks[i + 1] if i + 1 < n else None
            fn_rhs = None
            is_def = False
            if nxt is not None and nxt[0] == 'COP' and nxt[3] == '::':
                is_def = True
                if i + 2 < n and toks[i + 2][0] == 'LBRACE':
                    fn_rhs = self.open_frame.get(i + 2)
            elif nxt is not None and nxt[0] == 'CALLP' or (nxt is not None and nxt[0] == 'LPAREN' and not nxt[4]):
                cf = self.open_frame.get(i + 1)
                if cf is not None and cf.is_header:
                    is_def = True
                    fn_rhs = self.open_frame.get(cf.close + 2)
            if is_def:
                if f.fn is None:
                    self.defs.setdefault(nm, []).append(_Def(nm, i, self.stmt_start(i), self.stmt_end(i), fn_rhs, True))
                else:
                    self.fn_assigned.add(nm)
            elif nm in _UNKNOWN_DEFINERS and nxt is not None and nxt[0] in ('LPAREN', 'CALLP') and not nxt[4]:
                self.unknown = True
            elif nm == '.bkf' and nxt is not None and nxt[0] in ('LPAREN', 'CALLP') and not nxt[4]:
                self._backend_functions(i, f)
            elif nm == '.l' and nxt is not None and nxt[0] in ('LPAREN', 'CALLP') and not nxt[4]:
                self._load(i, f)
            elif nm in ('.extern', '.param') and nxt is not None and nxt[0] in ('LPAREN', 'CALLP') and not nxt[4]:
                self._extern(i, f)     # .param("name";...) defines its name as .extern does
        # destructuring [a b]::... defines its names
        for f in self.frames:
            if f.kind == 'data' and f.is_target and f.close is not None and f.fn is None:
                start, end = self.stmt_start(f.open), self.stmt_end(f.close)
                for j in range(f.open + 1, f.close):
                    if toks[j][0] == 'NAME' and ctx[j] is f:
                        self.defs.setdefault(toks[j][3], []).append(_Def(toks[j][3], j, start, end, None, True))
            elif f.kind == 'data' and f.is_target and f.close is not None:
                for j in range(f.open + 1, f.close):
                    if toks[j][0] == 'NAME' and ctx[j] is f:
                        self.fn_assigned.add(toks[j][3])

    def _backend_functions(self, i, f):
        """.bkf("exp") and .bkf(["exp";"sin"]) bind those names; any other argument binds unknown names."""
        toks = self.toks
        cf = self.open_frame.get(i + 1)
        names = []
        ok = cf is not None and cf.close is not None
        if ok:
            j = i + 2
            while j < cf.close:
                t = toks[j]
                if t[0] == 'STR' and not t[6]:
                    names.append(t[3])
                elif t[0] == 'NEWLINE':
                    pass
                elif t[0] in ('LBRACK', 'RBRACK', 'SEMI'):
                    pass
                else:
                    ok = False
                    break
                j += 1
        if not ok:
            self.unknown = True
            return
        if f.fn is None:
            start, end = self.stmt_start(i), self.stmt_end(i)
            for nm in names:
                self.defs.setdefault(nm, []).append(_Def(nm, i, start, end, None, True))
        else:
            self.import_anywhere.update(names)

    def _extern(self, i, f):
        """
        .extern("name";"signature") binds the name (a static-profile program, see
        klongpy/static/signature.py: the first argument is the name, as written
        in the program text). A first argument that is not a string literal binds
        an unknown name. .entry names a function the program already defines, so
        it adds no name.
        """
        toks = self.toks
        n = len(toks)
        arg = toks[i + 2] if i + 2 < n else None
        nxt = toks[i + 3] if i + 3 < n else None
        if not (arg is not None and arg[0] == 'STR' and not arg[6] and nxt is not None and nxt[0] in ('SEMI', 'RPAREN')):
            self.unknown = True
            return
        nm = arg[3]
        if f.fn is None:
            self.defs.setdefault(nm, []).append(_Def(nm, i, self.stmt_start(i), self.stmt_end(i), None, True))
        else:
            self.import_anywhere.add(nm)

    def _load(self, i, f):
        toks = self.toks
        n = len(toks)
        arg = toks[i + 2] if i + 2 < n else None
        if not (arg is not None and arg[0] == 'STR' and not arg[6] and i + 3 < n and toks[i + 3][0] == 'RPAREN'):
            self.unknown = True
            return
        info = None
        if self.resolve:
            path = _find_load(arg[3], self.base_dir)
            if path is not None:
                info = _load_info(path, self.seen, self.depth + 1)
        if info is None:
            self.unknown = True
            return
        if info.unknown:
            self.unknown = True
        names = info.exported()
        for nm, lim in info.limits().items():
            self.import_limits[nm] = _merge_limit(self.import_limits.get(nm, 0), lim) if nm in self.import_limits else lim
        if f.fn is None:
            self.imports.append((self.stmt_end(i), names))
        else:
            self.import_anywhere.update(names)

    # -- exported view ------------------------------------------------------
    def exported(self):
        """Names this source defines at top level, with what it loads."""
        names = set(self.defs)
        for _, ns in self.imports:
            names.update(ns)
        names.update(self.import_anywhere)
        return names

    def limits(self):
        """name -> how many arguments the function bound to it takes (None: not known to be a function)."""
        out = dict(self.import_limits)
        for nm, ds in self.defs.items():
            lim = 0
            for d in ds:
                x = _fn_limit(d.fn) if d.fn is not None else None
                lim = _merge_limit(lim, x)
            if nm in out:
                lim = _merge_limit(lim, out[nm])
            out[nm] = lim
        return out


def _merge_limit(a, b):
    if a is None or b is None:
        return None
    return max(a, b)


def _fn_limit(fn):
    """The most arguments a function literal takes, or None when the tree did not give its metadata."""
    if fn.params is not None:
        return len(fn.params)
    m = fn.meta
    if m is None:
        return None
    try:
        if m.params is not None:
            return len(m.params)
        return max(m.arity, m.max_arg)
    except Exception:
        return None


def _find_load(x, base_dir):
    """The path ``.l(x)`` would read, or None. See the module docstring for the search order."""
    if not x:
        return None
    paths = [p for p in str(os.environ.get('KLONGPATH') or ".:lib").split(':') if p]
    roots = []
    if base_dir:
        roots.append(base_dir)
    roots.append(os.getcwd())
    pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')
    cands = []
    for ad in paths:
        for root in roots:
            d = ad if os.path.isabs(ad) else os.path.join(root, ad)
            cands.append(os.path.join(d, x))
    cands.append(os.path.join(pkg, x))
    for root in roots:
        cands.append(os.path.join(root, x))
    for c in cands:
        if os.path.isfile(c):
            return c
        if not c.endswith('.kg') and os.path.isfile(c + '.kg'):
            return c + '.kg'
    return None


_load_cache = {}


def _load_info(path, seen, depth):
    real = os.path.realpath(path)
    if real in seen or depth > _MAX_DEPTH:
        return None
    try:
        st = os.stat(real)
        key = (real, st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    hit = _load_cache.get(key)
    if hit is not None:
        return hit
    try:
        with open(real, 'r') as f:
            text = f.read()
    except (OSError, UnicodeDecodeError):
        return None
    info = _Analysis(Source(path, text), os.path.dirname(real), True, seen + (real,), depth, full=False)
    if len(_load_cache) > 64:
        _load_cache.clear()
    _load_cache[key] = info
    return info


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

_IGNORE = re.compile(r'lint:ignore\b([\sA-Za-z0-9,]*)')
_CODE = re.compile(r'[WE]\d{3}')


def _plural(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


class _Linter:
    def __init__(self, an, define=()):
        self.an = an
        self.define = frozenset(define)
        self.toks = an.toks
        self.ctx = an.ctx
        self.text = an.text
        self.source = an.source
        self.found = []
        self.underscored = set()    # token indices inside a W001 chain

    def add(self, code, span, message, hint=None):
        self.found.append(Diagnostic(code, RULES[code].severity, tuple(span), message, hint, self.source))

    def span_of(self, i, j=None):
        return (self.toks[i][1], self.toks[i if j is None else j][2])

    def prev(self, i):
        """Index of the token before i in the same expression: newlines are skipped except at top level."""
        j = i - 1
        if self.ctx[i].kind != 'top':
            while j >= 0 and self.toks[j][0] == 'NEWLINE':
                j -= 1
        return j

    def run(self):
        for rule in (self.e001, self.w001, self.w010, self.w011, self.w012_w017, self.w013, self.w014,
                     self.w015, self.w016, self.w018, self.w019, self.w020, self.w003, self.w002, self.e002):
            try:
                rule()
            except Exception as e:      # a rule must never take the linter down
                import warnings
                warnings.warn(f"klongpy.lint: {rule.__name__} failed on {self.source.name}: {type(e).__name__}: {e}",
                              RuntimeWarning, stacklevel=2)
        return self.found

    # -- E001 -------------------------------------------------------------
    def e001(self):
        for span, message, hint in self.an.problems:
            self.add('E001', span, message, hint)

    # -- W001 -------------------------------------------------------------
    def w001(self):
        toks = self.toks
        n = len(toks)
        i = 0
        while i + 2 < n:
            a, b, c = toks[i], toks[i + 1], toks[i + 2]
            if (a[0] == 'NAME' and b[0] == 'OP' and b[3] == '_' and not b[4] and c[0] == 'NAME' and not c[4]
                    and a[3] not in _ARGS and c[3] not in _ARGS):
                # x_a, a_x: Drop with an argument on one side (x_a is "drop x from a"), not a misspelt name
                j = i + 2
                while (j + 2 < n and toks[j + 1][0] == 'OP' and toks[j + 1][3] == '_' and not toks[j + 1][4]
                       and toks[j + 2][0] == 'NAME' and not toks[j + 2][4] and toks[j + 2][3] not in _ARGS):
                    j += 2
                self.underscored.update(range(i, j + 1))
                name = self.text[a[1]:toks[j][2]]
                parts = " _ ".join(name.split('_'))
                self.add('W001', self.span_of(i, j),
                         f"'{name}' reads as {parts}: _ is the Drop operator, names cannot contain underscores",
                         "rename it (my.var or myVar), or put spaces around _ if you mean Drop")
                i = j + 1
            else:
                i += 1

    # -- W010 -------------------------------------------------------------
    def w010(self):
        toks = self.toks
        n = len(toks)
        for i in range(n - 2):
            t = toks[i]
            if t[0] != 'OP' or t[3] != '-':
                continue
            if self.ctx[i].kind in _DATA_KINDS:
                continue
            num, op = toks[i + 1], toks[i + 2]
            if num[0] != 'NUM' or op[0] != 'OP' or op[3] in _NOT_DYAD or op[3] in _ADVERB_OPS:
                continue
            j = self.prev(i)
            if j >= 0 and toks[j][0] in _OPERAND_END:
                continue        # a dyadic minus
            lit = self.text[t[1]:num[2]]
            self.add('W010', (t[1], op[2]),
                     f"a leading - negates everything to its right: {lit}{op[3]}... is -({lit[1:].strip()}{op[3]}...)",
                     f"write ({lit}){op[3]}... to negate only the number")

    # -- W011 / W003 helpers ------------------------------------------------
    def fn_mentions(self):
        """{fn frame: set of 1..3} of the arguments each function mentions outside nested literals."""
        out = {}
        toks, ctx = self.toks, self.ctx
        for i, t in enumerate(toks):
            if t[0] in ('NAME', 'QNAME') and t[3] in _ARGS:
                f = ctx[i]
                if f.kind in _DATA_KINDS or f.fn is None or t[3] in f.fn.locals:
                    continue
                out.setdefault(f.fn, set()).add(_ARGS.index(t[3]) + 1)
        return out

    def calls(self):
        """{callee name: [(call frame, argument count)]} for every call written name(...)."""
        out = {}
        toks = self.toks
        for f in self.an.frames:
            if f.kind == 'call' and not f.is_header and f.close is not None and f.open > 0 \
                    and toks[f.open - 1][0] == 'NAME':
                out.setdefault(toks[f.open - 1][3], []).append((f, self.count_args(f)))
        return out

    def w011(self):
        """
        A function that uses y without x (or z without y) is a dyad (arity is the
        highest argument used). That is a trap only for a caller that passes fewer
        arguments, which then gets a projection instead of a value. Reported when
        the file shows such a call: the literal called directly ({y}(5)), or the
        function bound to a name that is called with too few arguments. A function
        that is never called short is fine (a callback with a fixed (x;y)
        signature, an adverb operand, a table entry): reporting those would flag
        correct code.
        """
        toks, ctx, an = self.toks, self.ctx, self.an
        calls = None
        for fn, used in self.fn_mentions().items():
            if fn.params:
                continue
            hi = max(used)
            missing = [a for a in range(1, hi) if a not in used]
            if not missing:
                continue
            short = None
            if fn.close is not None and fn.close + 1 < len(toks) and toks[fn.close + 1][0] in ('LPAREN', 'CALLP') \
                    and not toks[fn.close + 1][4]:
                cf = self.an.open_frame.get(fn.close + 1)
                if cf is not None and cf.close is not None:
                    c = self.count_args(cf)
                    if c < hi:
                        short = (cf, c)
            elif fn.open >= 2 and toks[fn.open - 1][0] == 'COP' and toks[fn.open - 1][3] == '::' \
                    and toks[fn.open - 2][0] == 'NAME' and ctx[fn.open - 2].fn is None:
                name = toks[fn.open - 2][3]
                if len(an.defs.get(name, ())) == 1 and name not in an.local_names and name not in an.fn_assigned:
                    if calls is None:
                        calls = self.calls()
                    for cf, c in calls.get(name, ()):
                        if c < hi:
                            short = (cf, c)
                            break
            if short is None:
                continue
            cf, c = short
            line = self.source.line_col(toks[cf.open][1])[0]
            miss = " or ".join(_ARGS[a - 1] for a in missing)
            self.add('W011', self.span_of(fn.open), f"function uses {_ARGS[hi - 1]} but not {miss}: the call at line {line} "
                     f"passes {_plural(c, 'argument')} and gets a projection, not a value",
                     f"its arity is {hi}; pass {miss} too, or use the arguments you pass")

    # -- W003 -------------------------------------------------------------
    def w003(self):
        """
        A declared local that is never mentioned. Klong scopes dynamically, so a
        function can also read its caller's locals ({[a];a::x;g()} with g::{a+1}):
        a local whose name is mentioned as a free name (not declared) anywhere
        outside the function is not reported.
        """
        toks, ctx = self.toks, self.ctx
        free = None
        for fn in self.an.frames:
            if fn.kind != 'fn' or not fn.locals:
                continue
            if free is None:
                free = self.free_mentions()
            end = fn.close if fn.close is not None else len(toks)
            dstart, dend = fn.decl
            used = set()
            for j in range(fn.open + 1, end):
                t = toks[j]
                if t[0] in ('NAME', 'QNAME') and not (dstart <= t[1] < dend):
                    used.add(t[3])
            first = {}
            for j in range(fn.open + 1, end):
                t = toks[j]
                if t[0] == 'NAME' and dstart <= t[1] < dend:
                    first.setdefault(t[3], j)
            for nm in fn.locals:
                if nm in used or nm not in first:
                    continue
                if any(not (fn.open <= j < end) for j in free.get(nm, ())):
                    continue
                self.add('W003', self.span_of(first[nm]), f"local '{nm}' is declared but never used",
                         "remove it from the locals list")

    def free_mentions(self):
        """{name: [token index]} of every mention of a name that no enclosing function declares."""
        out = {}
        toks, ctx = self.toks, self.ctx
        for i, t in enumerate(toks):
            if t[0] != 'NAME':
                continue
            f = ctx[i]
            if f.kind in _DATA_KINDS:
                continue
            nm = t[3]
            fn = f.fn
            while fn is not None:
                if nm in fn.locals or (fn.params and nm in fn.params):
                    break
                fn = fn.parent.fn if fn.parent is not None else None
            else:
                out.setdefault(nm, []).append(i)
        return out

    # -- names visible at a token -----------------------------------------
    def declared_at(self, f):
        """Locals and named parameters of the functions around frame f."""
        names = set()
        fn = f.fn
        while fn is not None:
            names.update(fn.locals)
            if fn.params:
                names.update(fn.params)
            fn = fn.parent.fn if fn.parent is not None else None
        return names

    def first_available(self):
        """name -> index of the first token from which a top-level reference is defined."""
        an = self.an
        first = {}
        for nm, ds in an.defs.items():
            first[nm] = min(d.start for d in ds)
        for end, names in an.imports:
            for nm in names:
                if nm not in first or end < first[nm]:
                    first[nm] = end
        return first

    # -- W012 and W017 ------------------------------------------------------
    def w012_w017(self):
        toks, ctx, an = self.toks, self.ctx, self.an
        first = self.first_available()
        anywhere = set(an.defs) | an.import_anywhere
        for _, names in an.imports:
            anywhere.update(names)
        for f in an.frames:
            if f.kind != 'data' or f.is_decl or f.is_target:
                continue
            end = f.close if f.close is not None else len(toks)
            # an entry of a dictionary literal: bare names are keys
            takes_symbols = (f.parent is not None and f.parent.kind == 'dict') or self.takes_symbols(f)
            bad = []
            declared = None
            for j in range(f.open + 1, end):
                if ctx[j] is not f:
                    continue
                t = toks[j]
                k = t[0]
                if k == 'NAME':
                    nm = t[3]
                    if declared is None:
                        declared = self.declared_at(f.parent)
                    if nm in _ARGS or nm == '.f' or takes_symbols:
                        continue
                    if nm in declared or (first.get(nm, 1 << 60) <= j if f.fn is None else nm in anywhere):
                        self.add('W012', self.span_of(j), f"'{nm}' in a list is the symbol :{nm}, not the value of {nm}",
                                 "use [;a;b] to list values")
                elif k in ('SEMI', 'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'COND', 'ELIF', 'CALLP', 'COP'):
                    bad.append(j)
                elif k == 'OP':
                    if t[3] == '-' and j + 1 < len(toks) and toks[j + 1][0] == 'NUM' and not toks[j + 1][4]:
                        continue
                    if t[3] == '_' and j > 0 and toks[j - 1][0] == 'NAME' and not t[4]:
                        continue    # W001 reports my_var
                    bad.append(j)
            if bad:
                j = bad[0]
                t = toks[j]
                glyph = t[3]
                more = f" (and {_plural(len(bad) - 1, 'more item')})" if len(bad) > 1 else ""
                if t[0] == 'SEMI':
                    msg = f"';' inside a list is the string \";\"{more}"
                elif t[0] in ('OP', 'COP'):
                    msg = f"operator {glyph} inside a list is an operator object, not a computation{more}"
                else:
                    msg = f"'{glyph}' inside a list is the string \"{glyph}\"{more}"
                self.add('W017', self.span_of(j), msg, "use [;a;b] for a list of values; separate data with spaces")

    def takes_symbols(self, f):
        """
        Whether the data list f is where the language wants symbols: the object of
        .pyc([obj "method"];...) (obj is looked up by name), or the variables of the
        autograd forms f:>[a c], [a c]∇f, [a c]∂f.
        """
        toks = self.toks
        p = toks[f.open - 1] if f.open > 0 else None
        if p is not None and ((p[0] == 'COP' and p[3] == ':>') or (p[0] == 'OP' and p[3] in ('∇', '∂'))):
            return True
        if f.close is not None and f.close + 1 < len(toks):
            q = toks[f.close + 1]
            if q[0] == 'OP' and q[3] in ('∇', '∂'):
                return True
        c = f.parent
        if c is not None and c.kind == 'call' and c.open > 0 and toks[c.open - 1][3] == '.pyc':
            j = c.open + 1
            while toks[j][0] == 'NEWLINE':
                j += 1
            return j == f.open
        return False

    # -- W013 -------------------------------------------------------------
    def w013(self):
        toks, ctx = self.toks, self.ctx
        for f in self.an.frames:
            if f.kind != 'cond' or f.close is None:
                continue
            semis = [j for j in range(f.open + 1, f.close) if ctx[j] is f and toks[j][0] == 'SEMI']
            if len(semis) < 2:
                continue
            a = semis[-1] + 1
            while toks[a][0] == 'NEWLINE':
                a += 1
            nested = self.an.open_frame.get(a)
            if toks[a][0] != 'COND' or nested is None or nested.close is None:
                continue
            b = nested.close + 1
            while toks[b][0] == 'NEWLINE':
                b += 1
            if b == f.close:
                self.add('W013', self.span_of(a), ":[ in the else branch of :[ can be written as the :| chain",
                         ":[c1;t1:|c2;t2;e] replaces :[c1;t1;:[c2;t2;e]]")

    # -- W014 -------------------------------------------------------------
    def w014(self):
        toks, ctx = self.toks, self.ctx
        for i in range(1, len(toks)):
            t = toks[i]
            if not t[4]:
                continue
            k = t[0]
            adv = (k == 'OP' and t[3] in _ADVERB_OPS) or (k == 'COP' and t[3] in _ADVERB_COPS)
            if not (adv or k in ('LPAREN', 'CALLP')):
                continue
            f = ctx[i]
            if f.kind in _DATA_KINDS:
                continue
            p = toks[i - 1]
            if p[0] not in ('NAME', 'QNAME', 'RBRACE'):
                continue
            if k in ('LPAREN', 'CALLP') and self.is_infix(i - 1):
                continue        # 1 f (2): f is the infix verb and (2) its right operand
            if k in ('LPAREN', 'CALLP'):
                what = self.text[p[1]:p[2]] if p[0] != 'RBRACE' else '}'
                self.add('W014', self.span_of(i), f"whitespace between {what} and {t[3]}: this is not a call, "
                         f"the parenthesised part is dropped", f"write {what}(...) with no space")
            else:
                what = self.text[p[1]:p[2]] if p[0] != 'RBRACE' else '}'
                self.add('W014', self.span_of(i), f"whitespace between {what} and the adverb {t[3]}: "
                         f"an adverb must follow its verb directly", f"write {what}{t[3]} with no space")

    def is_infix(self, j):
        """Whether the NAME or } at token j stands in the verb position of an infix call (1 f 2)."""
        toks = self.toks
        if toks[j][0] == 'RBRACE':
            f = self.ctx[j]
            if f.kind != 'fn' or f.open < 1:
                return False
            j = f.open
        q = self.prev(j) if self.ctx[j].kind != 'top' else j - 1
        if q < 0:
            return False
        return toks[q][0] in _OPERAND_END

    # -- W015 -------------------------------------------------------------
    def w015(self):
        toks, ctx, tr = self.toks, self.ctx, self.an.trivia
        starts = [c[1] for c in tr]
        for i, t in enumerate(toks):
            if t[0] == 'NEWLINE' and t[5] and ctx[i].kind in ('fn', 'cond', 'call', 'elist'):
                k = bisect_right(starts, t[1]) - 1
                span = (tr[k][1], tr[k][2]) if k >= 0 else (t[1], t[2])
                where = {'fn': 'a function body', 'cond': 'a conditional', 'call': 'an argument list',
                         'elist': 'an evaluated list'}[ctx[i].kind]
                self.add('W015', span, f"newline after a comment acts as ';' inside {where}",
                         "write the ; yourself (at strict level 1 this newline is an ordinary newline)")

    # -- W016 / W018 --------------------------------------------------------
    def w016(self):
        text = self.text
        for t in self.toks:
            if t[0] == 'NUM' and text[t[1]] == ':':
                self.add('W016', (t[1], t[2]), f"'{text[t[1]:t[2]]}' is the number {text[t[1]+1:t[2]]}: the colon is ignored",
                         "remove the colon")

    def w018(self):
        toks = self.toks
        for i in range(len(toks) - 1):
            a, b = toks[i], toks[i + 1]
            if a[0] == 'NUM' and b[0] == 'NAME' and not b[4]:
                s = self.text[a[1]:b[2]]
                self.add('W018', (a[1], b[2]), f"'{s}' is the number {self.text[a[1]:a[2]]} followed by the name {b[3]}",
                         "add a space if the name is an infix function, or fix the number")

    # -- W019 -------------------------------------------------------------
    def w019(self):
        toks, ctx = self.toks, self.ctx
        for i in range(len(toks) - 1):
            t = toks[i]
            if t[0] == 'NAME' and t[3] in _ARGS and toks[i + 1][0] == 'COP' and toks[i + 1][3] == '::' \
                    and ctx[i].kind not in _DATA_KINDS and not ctx[i].is_header \
                    and ctx[i].fn is not None and t[3] not in ctx[i].fn.locals:
                self.add('W019', (t[1], toks[i + 1][2]), f"assignment to {t[3]}, which is an argument of the function",
                         "assign to a declared local instead: {[a];a::...}")

    # -- W020 -------------------------------------------------------------
    def w020(self):
        toks, ctx = self.toks, self.ctx
        an = self.an
        for i in range(1, len(toks) - 1):
            t = toks[i]
            if not (t[0] == 'OP' and t[3] == ',' and toks[i + 1][0] == 'DICT'):
                continue
            if ctx[i].kind in _DATA_KINDS:
                continue
            j = self.prev(i)
            if j < 0:
                continue
            p = toks[j]
            two = False
            what = None
            if p[0] == 'RBRACK':
                lf = ctx[j]
                if lf.kind == 'data' and lf.close == j:
                    two = self.count_items(lf) == 2
                    what = "a two-element list"
            elif p[0] == 'NAME':
                ds = an.defs.get(p[3])
                if ds and all(self.is_two_list(d) for d in ds) and p[3] not in an.fn_assigned:
                    two = True
                    what = f"'{p[3]}', a two-element list,"
            if two:
                self.add('W020', (t[1], toks[i + 1][2]), f"{what} joined to a dictionary with , becomes a dictionary entry",
                         "to append the dictionary to the list use ,, ")

    def is_two_list(self, d):
        toks = self.toks
        # name::[a b]
        j = d.idx + 2
        if j >= len(toks) or toks[j][0] != 'LBRACK':
            return False
        f = self.an.open_frame.get(j)
        return f is not None and f.kind == 'data' and f.close is not None and self.count_items(f) == 2

    def count_items(self, f):
        """Number of elements of a data list (a nested list is one element)."""
        toks, ctx = self.toks, self.ctx
        n = 0
        j = f.open + 1
        end = f.close
        while j < end:
            if ctx[j] is f and toks[j][0] != 'NEWLINE':
                n += 1
                if toks[j][0] == 'OP' and toks[j][3] == '-' and toks[j + 1][0] == 'NUM' and not toks[j + 1][4]:
                    j += 1      # a negative number is one element
            j += 1
        return n

    # -- W002 -------------------------------------------------------------
    def w002(self):
        an = self.an
        if an.unknown:
            return
        toks, ctx = self.toks, self.ctx
        builtins = _builtin_names() | self.define
        first = self.first_available()
        anywhere = set(first) | an.fn_assigned | an.local_names | an.import_anywhere
        n = len(toks)
        for i, t in enumerate(toks):
            if t[0] != 'NAME':
                continue
            f = ctx[i]
            if f.kind in _DATA_KINDS or (f.kind == 'call' and f.is_header):
                continue
            nm = t[3]
            if nm in _ARGS or nm in builtins or i in self.underscored:
                continue
            nxt = toks[i + 1] if i + 1 < n else None
            if nxt is not None and nxt[0] == 'COP' and nxt[3] == '::':
                continue
            if nxt is not None and nxt[0] in ('LPAREN', 'CALLP') and not nxt[4]:
                cf = an.open_frame.get(i + 1)
                if cf is not None and cf.is_header:
                    continue
            if f.fn is None:
                e = first.get(nm)
                if e is not None and e <= i:
                    continue
                if e is not None:
                    line = self.source.line_col(toks[min(e, n - 1)][1])[0]
                    self.add('W002', self.span_of(i), f"'{nm}' is used before its definition (defined at line {line})",
                             f"an undefined name evaluates to its own symbol, :{nm}")
                else:
                    self.add('W002', self.span_of(i), f"'{nm}' is never defined in this file",
                             f"an undefined name evaluates to its own symbol, :{nm}")
            else:
                if nm in anywhere or nm in self.declared_at(f):
                    continue
                self.add('W002', self.span_of(i), f"'{nm}' is never defined in this file",
                         f"an undefined name evaluates to its own symbol, :{nm}")

    # -- E002 -------------------------------------------------------------
    def e002(self):
        an = self.an
        toks, ctx = self.toks, self.ctx
        limits = an.limits()
        local = an.local_names | an.fn_assigned
        for f in an.frames:
            if f.kind != 'call' or f.is_header or f.close is None or f.open < 1:
                continue
            c = toks[f.open - 1]
            limit = None
            name = None
            if c[0] == 'NAME':
                name = c[3]
                if name in local or name in _ARGS:
                    continue
                if name in an.defs or name in an.import_limits:
                    limit = limits.get(name)
            elif c[0] == 'RBRACE':
                fn = ctx[f.open - 1]
                if fn.kind == 'fn' and fn.close == f.open - 1:
                    limit = _fn_limit(fn)
                    name = "the function"
            if limit is None:
                continue
            count = self.count_args(f)
            if count > limit:
                a = f.open - 1 if c[0] == 'NAME' else ctx[f.open - 1].open
                self.add('E002', (toks[a][1], toks[f.close][2]),
                         f"{name} takes {_plural(limit, 'argument')} but is called with {count}: the extra arguments are dropped",
                         "an arity error at strict level 2")

    def count_args(self, f):
        toks, ctx = self.toks, self.ctx
        semis = 0
        other = False
        for j in range(f.open + 1, f.close):
            if ctx[j] is f:
                if toks[j][0] == 'SEMI':
                    semis += 1
                elif toks[j][0] != 'NEWLINE':
                    other = True
            else:
                other = True
        if not semis and not other:
            return 0
        return semis + 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_codes(value):
    """A comma-separated string (or an iterable) of codes as a set of upper-case codes; ValueError on an unknown code."""
    if value is None:
        return None
    if isinstance(value, str):
        value = [v for v in re.split(r'[,\s]+', value) if v]
    out = set()
    for v in value:
        c = v.strip().upper()
        if c not in RULES:
            raise ValueError(f"unknown lint code {v!r}; codes are {', '.join(RULES)}")
        out.add(c)
    return out


def parse_names(value):
    """A comma-separated string (or an iterable) of names as a set of names."""
    if not value:
        return set()
    if isinstance(value, str):
        value = re.split(r'[,\s]+', value)
    return {v for v in value if v}


def _suppressed(an):
    """{line: set of codes} from the lint:ignore comments."""
    out = {}
    for c in an.trivia:
        m = _IGNORE.search(c[3])
        if m:
            codes = set(_CODE.findall(m.group(1).upper()))
            if codes:
                out.setdefault(an.source.line_col(c[1])[0], set()).update(codes)
    return out


def lint_source(text, name="<string>", *, select=None, ignore=None, define=None, base_dir=None, resolve=True):
    """
    Lint a program text. Returns the list of Diagnostic, sorted by position.

        text      the program
        name      its name in the findings; also where ``.l("file")`` is looked up
                  (the directory of name, when that is a file)
        select    only these codes (a set, list or comma-separated string)
        ignore    leave these codes out
        define    names the host defines before the program runs (a set, list or
                  comma-separated string): W002 does not report them. The test
                  files under tests/kgtests use t(...), which runner.kg defines.
        base_dir  directory for ``.l`` lookups, instead of the directory of name
        resolve   False: never read other files; every ``.l`` then counts as one
                  that defines unknown names (W002 off)

    Nothing in the program is evaluated and no other file is run.
    """
    select = parse_codes(select)
    ignore = parse_codes(ignore) or set()
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(name)) if os.path.isfile(name) else os.getcwd()
    an = _Analysis(Source(name, text), base_dir, resolve)
    found = _Linter(an, parse_names(define)).run()
    sup = _suppressed(an)
    out = []
    seen = set()
    for d in found:
        if (select is not None and d.code not in select) or d.code in ignore:
            continue
        if d.code in sup.get(d.line, ()):
            continue
        key = (d.code, d.span, d.message)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    out.sort(key=lambda d: (d.span[0], d.code))
    return out


def lint_file(path, **kw):
    """Lint a file. OSError and UnicodeDecodeError propagate."""
    with open(path, 'r') as f:
        text = f.read()
    return lint_source(text, path, **kw)


def lint_main(files, select=None, ignore=None, verbose=False, define=None, stdout=None, stderr=None):
    """
    The body of ``kgpy --lint FILE...``. Prints the findings and a summary and
    returns the exit status: 1 if any E finding was reported, 2 if a file could
    not be read (and no E finding), else 0. A file named - is read from stdin.
    """
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    try:
        select = parse_codes(select)
        ignore = parse_codes(ignore)
    except ValueError as e:
        print(f"kgpy --lint: {e}", file=err)
        return 2
    errors = warnings_ = 0
    unreadable = 0
    for path in files:
        try:
            if path == '-':
                found = lint_source(sys.stdin.read(), "<stdin>", select=select, ignore=ignore, define=define)
            else:
                found = lint_file(path, select=select, ignore=ignore, define=define)
        except (OSError, UnicodeDecodeError) as e:
            print(f"{path}: cannot read: {getattr(e, 'strerror', None) or e}", file=err)
            unreadable += 1
            continue
        for d in found:
            print(format_diagnostic(d, hint=verbose), file=out)
            if d.severity == 'error':
                errors += 1
            else:
                warnings_ += 1
    n = len(files) - unreadable
    print(f"{_plural(errors, 'error')}, {_plural(warnings_, 'warning')} in {_plural(n, 'file')}", file=out)
    if errors:
        return 1
    return 2 if unreadable else 0
