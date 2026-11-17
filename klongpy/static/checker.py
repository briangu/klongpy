"""
The static profile checker: is a Klong program inside the subset that can be
compiled ahead of time? (tickets/design/static-compile.md sections 1 to 4, 9.)

    check_source(text, name="<string>", **options) -> StaticReport
    check_file(path, **options)                    -> StaticReport
    check_main(files, ...)                         -> exit status (kgpy --check-static)

The checker reads a program; it never runs it. It parses at strict level 2,
sorts the top-level statements into stages, and then walks every run-time
function with abstract values (an element type, a shape, and whether the
value is a compile-time constant) in place of real ones. Every problem is
reported, not only the first; each names its rule (klongpy.static.rules) and
points at the offending expression.

How the pass is built

1. Parse (strict level 2), resuming at the next top-level statement after a
   syntax error. ``.l("file")`` is followed by reading and parsing that file
   under the same rules (found the way kgpy --lint finds it).
2. Classify each top-level statement by the root of its tree:
   ``name::{...}`` and ``name(a;b)::{...}`` are run-time functions; any other
   ``name::expr`` is a compile-time value (a global); ``.extern``, ``.entry``
   and ``.l`` with literal arguments are declarations; anything else is an
   error (toplevel-effect).
3. One pass over the trees of the functions gives the call graph (reachability
   from the entries, recursion) and the set of globals some function assigns.
4. Abstract evaluation from each entry with the argument types of its
   signature. A function is checked once per distinct tuple of argument types
   (the specialisation rule of design section 4). A program without any
   ``.entry`` is checked as a library: every function is a root, with
   arguments of unknown type.

What is known without running the program

The type of a global is known when its definition is a literal (a number, a
list of numbers); the value of any other compile-time expression exists only
after the host has evaluated it, which this pass never does. Such a global
is numeric of unknown type and shape, and a rule that needs a type or shape
is applied only when it is known: the checker reports what is certainly wrong
and never guesses. The driver of ticket 091 evaluates the definitions and
hands the values back through ``globals=``; then everything is known and the
same rules decide.

Forms (a rejected operator, a string, a function used as a value, recursion,
a system function, a non-constant count) never need types and are always
decided.
"""
import os
import re
from bisect import bisect_right

import numpy as np

from ..errors import KlongError
from ..lexer import Source
from ..parser import KGExprArray, KGExprDict
from ..types import (KGAdverb, KGBody, KGCall, KGChar, KGCond, KGDestructure, KGFn, KGLambda, KGOp, KGQSym, KGSym,
                     KLONG_UNDEFINED, reserved_fn_args)
from .diagnostics import StaticDiagnostic
from .optable import lookup
from .rules import RULES
from .signature import (INT_RANGE, MAX_RANK, EntryDecl, ExternDecl, SignatureError, Ty, check_entry_name,
                        check_extern_name, fixed_spec, is_fixed, parse_signature, parse_type)

__all__ = ["check_source", "check_file", "check_main", "StaticReport", "FunctionInfo", "GlobalInfo",
           "ExternDecl", "EntryDecl", "validate_global"]


# ---------------------------------------------------------------------------
# Abstract values
# ---------------------------------------------------------------------------

_NC = type("NotConstant", (), {"__repr__": lambda s: "NC"})()        # not a compile-time constant
_CU = type("ConstantUnknown", (), {"__repr__": lambda s: "CU"})()    # a constant whose value this pass does not know


class AT:
    """
    An abstract value.

        kind    "num"   a number or an array of numbers
                "any"   nothing is known (an error was already reported here)
                "void"  the result of a void extern
        elem    "int", "float", "fixed<W,I>" or None (not known)
        shape   a tuple of positive ints, () for a scalar, or None (not known)
        const   _NC, _CU, or the value (a Python int or float, or a numpy array)
    """
    __slots__ = ("kind", "elem", "shape", "const")

    def __init__(self, kind, elem=None, shape=None, const=_NC):
        self.kind = kind
        self.elem = elem
        self.shape = shape
        self.const = const

    def key(self):
        return (self.kind, self.elem, self.shape)

    def plain(self):
        """The same type, not a constant."""
        return self if self.const is _NC else AT(self.kind, self.elem, self.shape)

    @property
    def known(self):
        return self.kind == "num" and self.elem is not None and self.shape is not None

    def ty(self):
        """The Ty of a fully known value, else None."""
        if self.kind == "void":
            return Ty("void")
        return Ty(self.elem, self.shape) if self.known else None

    def __str__(self):
        if self.kind == "void":
            return "void"
        if self.kind != "num" or (self.elem is None and self.shape is None):
            return "a value of unknown type"
        elem = self.elem or "number"
        if self.shape is None:
            return f"{elem} of unknown shape"
        if self.shape:
            return f"{elem}[{','.join(str(d) for d in self.shape)}]"
        return elem

    __repr__ = __str__


ANY = AT("any")
VOID = AT("void")
UNKNOWN = AT("num")          # a number or array of numbers about which nothing else is known


def _num(elem, shape, const=_NC):
    return AT("num", elem, shape, const)


def _from_ty(ty):
    return VOID if ty.elem == "void" else _num(ty.elem, tuple(ty.shape))


def _promote(a, b):
    if is_fixed(a) or is_fixed(b):
        return a if a == b else None      # the checker reports a fixed-point mismatch where it sees the operands
    if a == "float" or b == "float":
        return "float"
    if a == "int" and b == "int":
        return "int"
    return None


def _fixed_holds(elem, const):
    """Does the int constant (a number or an array) fit the integer part of the fixed type elem? None when it is not a known constant."""
    if const is _NC or const is _CU:
        return None
    w, i = fixed_spec(elem)
    lo, hi = -(1 << (i - 1)), (1 << (i - 1)) - 1
    a = np.asarray(const)
    return bool(a.size == 0 or (int(a.min()) >= lo and int(a.max()) <= hi))


def _is_int_const(at):
    return at.const is not _NC and at.const is not _CU and isinstance(at.const, int)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

class FunctionInfo:
    """
    A run-time function: ``name::{...}`` or ``name(a;b)::{...}`` at top level.

        name, node      the name and the function literal (a KGFn)
        params          tuple of parameter names ("x", "y", "z" or the named ones)
        span, source    where the definition is
        calls           names of the functions and externs its body mentions
        reachable       reached from an entry (always True in a program without entries)
        in_profile      no diagnostic lies in it or in anything it calls
        specializations {tuple of argument Ty or None: result Ty or None}; None where not known
    """
    def __init__(self, name, node, span, source):
        self.name = name
        self.node = node
        self.span = span
        self.source = source
        self.params = _params(node)
        self.calls = ()
        self.reachable = False
        self.in_profile = True
        self.specializations = {}

    @property
    def arity(self):
        return len(self.params)


class GlobalInfo:
    """
    A compile-time value: ``name::expr`` at top level, expr not a function literal.

        ty          its Ty when the checker knows it (a literal, or a value passed in), else None
        assigned    some checked run-time function assigns it: it is state (RAM), not a constant
        resolved    False when the value exists only after the host evaluates the definition
    """
    def __init__(self, name, node, span, source):
        self.name = name
        self.node = node
        self.span = span
        self.source = source
        self.at = AT("num", None, None, _CU)
        self.assigned = False
        self.resolved = False
        self.param = False      # declared with .param: an input register, never a constant, never assigned

    @property
    def ty(self):
        return self.at.ty()


class StaticReport:
    """
    What the checker found.

        ok              no diagnostics
        diagnostics     list of StaticDiagnostic, in source order
        source          the Source of the program
        statements      [(kind, name, node, span, source)] in the order they run, with kind
                        "function", "global", "extern", "entry" or "load" (what the driver
                        of ticket 091 walks)
        functions       {name: FunctionInfo}
        globals         {name: GlobalInfo}
        externs         {name: ExternDecl}
        entries         [EntryDecl]
        host_only       names of functions no entry reaches: they run on the host only
        unresolved      names of globals whose type the checker did not know
    """
    def __init__(self, source):
        self.source = source
        self.diagnostics = []
        self.statements = []
        self.functions = {}
        self.globals = {}
        self.externs = {}
        self.entries = []
        self.params = {}

    @property
    def ok(self):
        return not self.diagnostics

    @property
    def host_only(self):
        return [f.name for f in self.functions.values() if not f.reachable]

    @property
    def unresolved(self):
        return [g.name for g in self.globals.values() if not g.resolved]

    def errors(self):
        """The diagnostics as KlongStaticError objects."""
        return [d.error() for d in self.diagnostics]

    def raise_first(self):
        """Raise the first diagnostic as a KlongStaticError; the error carries all of them as ``diagnostics``."""
        if self.diagnostics:
            e = self.diagnostics[0].error()
            e.diagnostics = list(self.diagnostics)
            raise e
        return self

    def render(self, color=False):
        return "\n".join(d.render(color) for d in self.diagnostics)

    def summary(self):
        fns = [f for f in self.functions.values() if f.reachable]
        parts = [_plural(len(fns), "function"), _plural(len(self.entries), "entry", "entries"),
                 _plural(len(self.externs), "extern"), _plural(len(self.globals), "global")]
        return ", ".join(parts)


def _plural(n, word, many=None):
    return f"{n} {word if n == 1 else (many or word + 's')}"


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------

def _name(s):
    return str.__str__(s)


def _is_chain(n):
    a = n.a
    return type(a) is list and len(a) > 0 and isinstance(a[0], KGAdverb)


def _is_fn_literal(n):
    return (isinstance(n, KGFn) and not isinstance(n, KGCall) and n.args is None
            and not isinstance(n.a, (KGOp, KGLambda)) and not _is_chain(n))


def _params(node):
    """The parameter names of a function literal."""
    if node.params is not None:
        return tuple(_name(p) for p in node.params)
    body = node.a
    if type(body) is KGBody and body.params is not None:
        return tuple(_name(p) for p in body.params)
    n = node.max_arg if node.max_arg is not None else node.arity
    return tuple(reserved_fn_args[:n])


def _locals(node):
    loc = node.locals
    if loc is None and type(node.a) is KGBody:
        loc = node.a.locals
    return tuple(_name(q) for q in (loc or ()))


def _body(node):
    """(statements, spans) of a function literal."""
    a = node.a
    stmts = list(a) if (type(a) is KGBody or type(a) is list) else [a]
    spans = list(node.body_spans or ())
    spans += [None] * (len(stmts) - len(spans))
    return stmts, spans


def _args(n):
    a = n.args
    if a is None:
        return []
    return list(a) if isinstance(a, (list, tuple)) else [a]


def _arg_span(n, i, default):
    sp = n.arg_spans
    if sp is not None and i < len(sp) and sp[i] is not None:
        return tuple(sp[i])
    return default


def _is_plain_str(x):
    return isinstance(x, str) and not isinstance(x, (KGSym, KGChar))


# ---------------------------------------------------------------------------
# Compile-time values
# ---------------------------------------------------------------------------

def validate_global(value, int_bits=32):
    """
    Is value acceptable as an initialised global (design section 2)?

    Returns (ty, rule, message): the Ty and (None, None) when it is; ty None,
    the rule id and a one-line reason when it is not. The driver of ticket 091
    calls this on the result of each compile-time definition; the checker
    calls it on literals.
    """
    lo, hi = INT_RANGE[int_bits]
    v = value
    if hasattr(v, "spec") and hasattr(v, "raw"):          # a FixedArray (klongpy/static/fixed.py)
        if v.size == 0:
            return None, "global-array", "an empty list cannot be a global value: every dimension is a positive integer"
        if v.ndim > MAX_RANK:
            return None, "global-array", f"an array of rank {v.ndim} cannot be a global value: the rank is 1 to {MAX_RANK}"
        return Ty(v.spec.elem, tuple(int(d) for d in v.shape)), None, None
    if getattr(v, "ndim", None) == 0 and hasattr(v, "item"):
        v = v.item()
    if isinstance(v, (bool, np.bool_)):
        v = int(v)
    if isinstance(v, (int, np.integer)):
        if not lo <= int(v) <= hi:
            return None, "int-range", f"{int(v)} is outside the range of a {int_bits}-bit kg_int"
        return Ty("int"), None, None
    if isinstance(v, (float, np.floating)):
        return Ty("float"), None, None
    if isinstance(v, KGChar):
        return None, "global-value", "a character cannot be a global"
    if isinstance(v, KGSym):
        return None, "global-value", f"the symbol :{_name(v)} cannot be a global"
    if isinstance(v, str):
        return None, "global-value", "a string cannot be a global"
    if isinstance(v, dict):
        return None, "global-value", "a dictionary cannot be a global"
    if v is None or v is KLONG_UNDEFINED:
        return None, "global-value", ":undefined cannot be a global"
    if isinstance(v, (KGFn, KGLambda, KGOp)) or callable(v):
        return None, "global-value", "a function cannot be a global"
    if isinstance(v, (list, tuple)):
        try:
            v = np.asarray(v)
        except Exception:
            return None, "global-array", "a ragged list cannot be a global"
    if not isinstance(v, np.ndarray):
        if hasattr(v, "shape") and hasattr(v, "dtype"):
            try:
                v = np.asarray(v.detach().cpu() if hasattr(v, "detach") else v)
            except Exception:
                return None, "global-value", f"a {type(value).__name__} cannot be a global"
        else:
            return None, "global-value", f"a {type(value).__name__} cannot be a global"
    kind = v.dtype.kind
    if kind == "O":
        bad = _first_non_number(v)
        if bad is not None and not isinstance(bad, (np.ndarray, list, tuple)):
            _, rule, msg = validate_global(bad, int_bits)
            if rule == "global-value":
                return None, "global-array", "a list that holds " + msg.replace(" cannot be a global", "") + " cannot be a global"
            if rule == "int-range":
                return None, rule, msg
        return None, "global-array", "a ragged list (rows of different lengths, or numbers mixed with lists) cannot be a global"
    if kind not in "iubf":
        return None, "global-value", "a list of strings cannot be a global"
    if v.size == 0:
        return None, "global-array", "an empty list cannot be a global value: every dimension is a positive integer"
    if v.ndim > MAX_RANK:
        return None, "global-array", f"an array of rank {v.ndim} cannot be a global value: the rank is 1 to {MAX_RANK}"
    if kind in "iub":
        if v.size and (int(v.min()) < lo or int(v.max()) > hi):
            worst = int(v.max()) if int(v.max()) > hi else int(v.min())
            return None, "int-range", f"{worst} is outside the range of a {int_bits}-bit kg_int"
        return Ty("int", tuple(int(d) for d in v.shape)), None, None
    return Ty("float", tuple(int(d) for d in v.shape)), None, None


def _first_non_number(arr):
    """The first element of an object array that is not a number, depth first (a nested array if the nesting is ragged)."""
    for e in arr.flat if isinstance(arr, np.ndarray) else arr:
        if isinstance(e, (bool, int, float, np.integer, np.floating)) and not isinstance(e, str):
            continue
        if isinstance(e, np.ndarray) and e.dtype.kind == "O":
            inner = _first_non_number(e)
            if inner is not None and not isinstance(inner, np.ndarray):
                return inner
        return e
    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse(source):
    """
    Parse a Source at strict level 2 without evaluating anything. Returns
    ([(statement, span)], [(span, message, hint)], names): after a syntax
    error the parse resumes at the next top-level statement, so every error
    is found. names are the names the statements that failed to parse were
    defining (``name::`` at their start), so that a use of one is not
    reported a second time as undefined.
    """
    from ..interpreter import KlongInterpreter
    from ..lint import _next_boundary, _scan, _tokenize
    klong = KlongInterpreter(strict=2, backend="numpy")  # parse only; literals must be numpy arrays whatever the default backend
    text = source.text
    n = len(text)
    toks, _ = _tokenize(source)
    _, ctx, _ = _scan(toks)
    starts = [t[1] for t in toks]
    stmts = []
    problems = []
    broken = set()

    def statement_start(offset):
        """Offset of the first token of the top-level statement that holds offset."""
        i = bisect_right(starts, offset) - 1
        while i > 0 and not (ctx[i - 1].kind == 'top' and toks[i - 1][0] in ('SEMI', 'NEWLINE')):
            i -= 1
        return toks[i][1] if 0 <= i < len(toks) else offset

    def attempt(pos, end):
        """Parse text[pos:end]. Returns None, or (span, message, hint) of the first problem."""
        src = source if end >= n else Source(source.name, text[:end])
        try:
            r = klong.prog(src, pos)
        except KlongError as e:
            span = tuple(e.span) if getattr(e, "span", None) else (pos, min(pos + 1, end))
            return span, str(e) or type(e).__name__, getattr(e, "hint", None)
        except Exception as e:      # the parser's own failure on unusual input: report it, go on
            return (pos, min(pos + 1, end)), f"cannot parse: {type(e).__name__}: {e}", None
        spans = r.stmt_spans or [None] * len(r[1])
        stmts.extend(zip(r[1], spans))
        return None

    pos = 0
    for _ in range(100000):
        if pos >= n:
            break
        problem = attempt(pos, n)
        if problem is None:
            break
        # prog stops at the first problem and returns nothing: read the
        # statements before the bad one on their own, then go on after it
        stop = max(problem[0][0], pos)
        bad = max(statement_start(stop), pos)
        if bad > pos:
            earlier = attempt(pos, bad)
            if earlier is not None:
                problems.append(earlier)
        problems.append(problem)
        m = _DEFINES.match(text, bad)
        if m:
            broken.add(m.group(1))
        nxt = _next_boundary(toks, ctx, starts, stop)
        if nxt is None or nxt <= pos:
            break
        pos = nxt
    return stmts, problems, broken


_MODULE = re.compile(r"`[A-Za-z0-9.]+")
_DEFINES = re.compile(r"[ \t]*([A-Za-z.][A-Za-z0-9.]*)[ \t]*(\([^)\n]*\))?[ \t]*::")


def _find_load(name, base_dir):
    from ..lint import _find_load as find
    return find(name, base_dir)


# ---------------------------------------------------------------------------
# The checker
# ---------------------------------------------------------------------------

class _Var:
    """
    A parameter or local. ty is None until the first assignment of this walk;
    hint is the element type an earlier walk of the same body arrived at
    ("float" when some assignment stores a float), which the first
    assignment adopts: the type of a local is the join of all its assignments.
    """
    __slots__ = ("ty", "set", "hint")

    def __init__(self, ty=None, is_set=False, hint=None):
        self.ty = ty
        self.set = is_set
        self.hint = hint


class _Frame:
    """The variables of one function literal being walked; parent is the literal it is written in."""
    __slots__ = ("vars", "parent")

    def __init__(self, parent=None):
        self.vars = {}
        self.parent = parent


class _Refs:
    __slots__ = ("refs", "assigns")

    def __init__(self):
        self.refs = {}      # free name -> span of its first mention
        self.assigns = {}   # free name assigned -> span


_MAX_PASSES = 4


class _Checker:
    def __init__(self, source, base_dir, int_bits, globals_, resolve, all_functions):
        self.source = source
        self.base_dir = base_dir
        self.int_bits = int_bits
        self.given = dict(globals_ or {})
        self.resolve = resolve
        self.all_functions = all_functions
        self.report = StaticReport(source)
        self.functions = self.report.functions
        self.globals = self.report.globals
        self.externs = self.report.externs
        self.src = source           # the Source diagnostics are attached to while walking
        self.mute = 0
        self.seen = set()
        self.stack = []
        self.memo = {}
        self.reported = set()
        self.loaded = set()
        self.kinds = {}             # top-level name -> "function" | "global" | "extern"
        self.refs = {}              # function name -> _Refs
        self.broken = set()         # names whose definition did not parse

    # -- diagnostics ----------------------------------------------------------
    def diag(self, rule, span, message, hint=None, source=None):
        if self.mute:
            return
        assert rule in RULES, rule
        source = source or self.src
        if span is None:
            span = (0, 0)
        span = (int(span[0]), int(span[1]))
        message = _MODULE.sub("", message)      # a name inside .module(:m) is a`m in the tree
        key = (rule, source.name, span, message)
        if key in self.seen:
            return
        self.seen.add(key)
        if hint is None:
            hint = RULES[rule].hint
        self.report.diagnostics.append(StaticDiagnostic(rule, span, source, message, hint))

    # -- top level ------------------------------------------------------------------
    def run(self):
        real = os.path.realpath(self.source.name) if os.path.isfile(self.source.name) else None
        if real:
            self.loaded.add(real)
        self.load(self.source, self.base_dir, 0)
        self.graph()
        self.roots()
        self.finish()
        return self.report

    def load(self, source, base_dir, depth):
        stmts, problems, broken = _parse(source)
        self.broken |= broken
        for span, message, hint in problems:
            self.diag("strict-syntax", span, message, hint, source)
        saved = self.src
        self.src = source
        try:
            for stmt, span in stmts:
                self.statement(stmt, span, source, base_dir, depth)
        finally:
            self.src = saved

    def statement(self, n, span, source, base_dir, depth):
        if span is None:
            span = getattr(n, "span", None) or (0, 0)
        if isinstance(n, KGFn) and not isinstance(n, KGCall) and isinstance(n.a, KGOp) and n.a.a == "::" \
                and type(n.args) is list and len(n.args) == 2 and n.args[1] is not None:
            left, right = n.args
            lsp = _arg_span(n, 0, span)
            if type(left) is KGDestructure:
                self.define_globals([_name(q) for q in left], right, n, span, lsp, source)
                return
            if isinstance(left, KGSym) and not isinstance(left, KGQSym):
                name = _name(left)
                if name in reserved_fn_args:
                    self.diag("toplevel-effect", lsp, f"{name} is a function argument; it cannot be defined at top level")
                    return
                if _is_fn_literal(right):
                    self.define_function(name, right, span, lsp, source)
                else:
                    self.define_globals([name], right, n, span, lsp, source, single=True)
                return
        if isinstance(n, KGCall) and isinstance(n.a, KGSym) and _name(n.a) in (".extern", ".entry", ".l", ".param"):
            self.declaration(_name(n.a), n, span, source, base_dir, depth)
            return
        what = "this expression"
        if isinstance(n, KGCall) and isinstance(n.a, KGSym):
            what = f"the call of {_name(n.a)}"
        self.diag("toplevel-effect", span, f"{what} is evaluated for its effect at top level, which has no meaning in a compiled program")

    def taken(self, name, kind, span):
        """Record a top-level name; False (and a diagnostic) if it clashes with an earlier definition."""
        old = self.kinds.get(name)
        if old is None or (old == "global" and kind == "global"):
            self.kinds[name] = kind
            return True
        a = {"function": "a function", "global": "a global", "extern": "an extern"}
        self.diag("redefinition", span, f"{name} is already defined as {a[old]}; it cannot be defined again as {a[kind]}")
        return False

    def define_function(self, name, node, span, lsp, source):
        if not self.taken(name, "function", lsp):
            return
        f = FunctionInfo(name, node, span, source)
        self.functions[name] = f
        self.report.statements.append(("function", name, node, span, source))

    def define_globals(self, names, right, node, span, lsp, source, single=False):
        known, value = self.literal(right)
        for i, name in enumerate(names):
            if name in self.report.params:
                self.diag("redefinition", lsp, f"{name} is a parameter (.param); it cannot be defined again")
                continue
            if name in reserved_fn_args or not self.taken(name, "global", lsp):
                continue
            g = GlobalInfo(name, node, span, source)
            self.globals[name] = g
            self.report.statements.append(("global", name, node, span, source))
            if name in self.given:
                self.set_global(g, self.given[name], lsp)
            elif known and single:
                self.set_global(g, value, _arg_span(node, 1, span))
            elif known and isinstance(value, np.ndarray) and value.ndim >= 1 and len(value) == len(names):
                self.set_global(g, value[i], _arg_span(node, 1, span))
        # a::b::3 defines b as well: any name the expression assigns exists once the host has run it
        inner = _Refs()
        self.scan(right, span, set(), inner)
        for name in inner.assigns:
            if name not in self.kinds and name not in reserved_fn_args and not name.startswith("."):
                self.kinds[name] = "global"
                self.globals[name] = GlobalInfo(name, node, span, source)
                self.report.statements.append(("global", name, node, span, source))
                if name in self.given:
                    self.set_global(self.globals[name], self.given[name], lsp)

    def set_global(self, g, value, span):
        if isinstance(value, AT):
            g.at = value
            g.resolved = value.known
            return
        if isinstance(value, Ty):
            g.at = _from_ty(value)
            g.resolved = True
            return
        ty, rule, message = validate_global(value, self.int_bits)
        g.resolved = True
        if rule:
            self.diag(rule, span, f"{g.name}: {message}", source=g.source)
            g.at = ANY
            return
        const = _NC
        if not ty.shape:
            v = value.item() if hasattr(value, "item") else value
            const = _NC if is_fixed(ty.elem) else (int(v) if ty.elem == "int" else float(v))
        elif isinstance(value, np.ndarray) and value.size <= 4096 and not is_fixed(ty.elem):
            const = value
        g.at = _num(ty.elem, ty.shape, const)

    def literal(self, n):
        """(True, value) when the value of a top-level definition can be read off the tree, else (False, None)."""
        if isinstance(n, (KGQSym, KGChar)):
            return True, n
        if isinstance(n, KGSym):
            name = _name(n)
            if self.kinds.get(name) in ("function", "extern"):
                return True, KGFn(None, None, 0)
            g = self.globals.get(name)
            if g is not None and g.resolved:
                return True, g.at
            return False, None
        if isinstance(n, (bool, int, float, str, np.ndarray)):
            return True, n
        if isinstance(n, KGExprDict):
            return True, {}
        if isinstance(n, KGFn):
            if isinstance(n.a, KGLambda):
                return (True, {}) if isinstance(n.args, dict) else (False, None)
            if isinstance(n.a, KGOp) and n.a.a == "-" and type(n.args) is not list \
                    and isinstance(n.args, (int, float)) and not isinstance(n.args, bool):
                return True, -n.args
            if _is_fn_literal(n) or (not isinstance(n, KGCall) and not isinstance(n.a, KGOp) and not _is_chain(n)
                                     and type(n.args) is list and any(q is None for q in n.args)):
                return True, n
        return False, None

    def declaration(self, which, n, span, source, base_dir, depth):
        args = _args(n)
        want = {".extern": (2, 3), ".entry": (2, 2), ".l": (1, 1), ".param": (2, 3)}[which]
        if not want[0] <= len(args) <= want[1]:
            takes = "2 or 3 arguments" if which in (".extern", ".param") else _plural(want[0], "argument")
            self.diag("decl-literal", span, f"{which} takes {takes}, got {len(args)}")
            return
        if which == ".param":
            self.param_declaration(n, args, span, source)
            return
        for i, a in enumerate(args):
            if not _is_plain_str(a):
                self.diag("decl-literal", _arg_span(n, i, span), f"argument {i + 1} of {which} must be a string literal")
                return
        if which == ".l":
            self.load_file(args[0], _arg_span(n, 0, span), span, source, base_dir, depth)
            return
        name, text = args[0], args[1]
        nsp, ssp = _arg_span(n, 0, span), _arg_span(n, 1, span)
        try:
            sig = parse_signature(text, allow_str=(which == ".extern"))
        except SignatureError as e:
            at = ssp
            if source.text[ssp[0]:ssp[1]] == '"' + text + '"':
                # the literal has no "" escapes: point inside it
                start = ssp[0] + 1 + e.span[0]
                at = (start, max(ssp[0] + 1 + e.span[1], start + 1))
            self.diag("signature", at, f"{which} signature: {e.message}")
            return
        if which == ".extern":
            bad = check_extern_name(name)
            if bad:
                self.diag("extern", nsp, bad)
                return
            old = self.externs.get(name)
            module = args[2] if len(args) == 3 else "env"
            if old is not None:
                if old.signature != sig or old.module != module:
                    self.diag("extern", span, f"extern {name} is declared again with a different signature "
                                              f"(first as \"{old.signature}\")")
                return
            if not self.taken(name, "extern", nsp):
                return
            self.externs[name] = ExternDecl(name, sig, module, span, source)
            self.report.statements.append(("extern", name, n, span, source))
            return
        bad = check_entry_name(name)
        if bad:
            self.diag("entry", nsp, bad)
            return
        f = self.functions.get(name)
        if f is None:
            kind = self.kinds.get(name)
            if kind == "global":
                self.diag("entry", nsp, f"{name} is a global, not a function; only a function can be an entry")
            elif kind == "extern":
                self.diag("entry", nsp, f"{name} is an extern; only a function defined in the program can be an entry")
            elif name not in self.broken:
                self.diag("entry", nsp, f"the function {name} is not defined before this .entry",
                          "define the function first; .entry checks the signature against it")
            return
        if f.arity != sig.arity:
            self.diag("entry", ssp, f"{name} takes {_plural(f.arity, 'argument')} but the signature has {_plural(sig.arity, 'parameter')}")
            return
        self.report.entries.append(EntryDecl(name, sig, span, source))
        self.report.statements.append(("entry", name, n, span, source))

    def param_declaration(self, n, args, span, source):
        """``.param("name";"type"[;default])``: a global that is an input register."""
        for i, a in enumerate(args[:2]):
            if not _is_plain_str(a):
                self.diag("decl-literal", _arg_span(n, i, span), f"argument {i + 1} of .param must be a string literal")
                return
        name, text = args[0], args[1]
        nsp, tsp = _arg_span(n, 0, span), _arg_span(n, 1, span)
        try:
            ty = parse_type(text)
        except SignatureError as e:
            at = tsp
            if source.text[tsp[0]:tsp[1]] == '"' + text + '"':
                start = tsp[0] + 1 + e.span[0]
                at = (start, max(tsp[0] + 1 + e.span[1], start + 1))
            self.diag("signature", at, f".param type: {e.message}")
            return
        if ty.shape or ty.elem not in ("int", "float") and not is_fixed(ty.elem):
            self.diag("param", tsp, f"the type of a parameter is a scalar int, float or fixed<W,I>; {ty} is not",
                      "a parameter is one register: pass arrays as entry arguments")
            return
        bad = check_extern_name(name)
        if bad:
            self.diag("param", nsp, bad.replace("an extern", "a parameter"))
            return
        if len(args) == 3:
            d = args[2]
            if isinstance(d, KGFn) and isinstance(d.a, KGOp) and d.a.a == "-" and type(d.args) is not list:
                d = -d.args if isinstance(d.args, (int, float)) else None
            if isinstance(d, bool) or not isinstance(d, (int, float, np.integer, np.floating)):
                self.diag("param", _arg_span(n, 2, span), "the default of .param must be a number literal")
                return
            if ty.elem == "int" and isinstance(d, (float, np.floating)):
                self.diag("param", _arg_span(n, 2, span), f"the default of the int parameter {name} must be an int")
                return
        old = self.report.params.get(name)
        if old is not None:
            if old != ty:
                self.diag("param", span, f"the parameter {name} is declared again as {ty}; it was {old}")
            return
        if not self.taken(name, "global", nsp):
            return
        g = GlobalInfo(name, n, span, source)
        g.at = _num(ty.elem, ())
        g.resolved = True
        g.assigned = True
        g.param = True
        self.globals[name] = g
        self.report.params[name] = ty
        self.report.statements.append(("param", name, n, span, source))

    def load_file(self, name, nsp, span, source, base_dir, depth):
        self.report.statements.append(("load", name, None, span, source))
        if not self.resolve:
            return
        path = _find_load(name, base_dir)
        if path is None:
            self.diag("load", nsp, f"cannot find the file \"{name}\" named by .l",
                      "it is looked up like .l does: in the directory of the program, the KLONGPATH directories, then klongpy/lib")
            return
        real = os.path.realpath(path)
        if real in self.loaded:
            return
        if depth >= 8:
            self.diag("load", nsp, f".l nests too deeply at \"{name}\"")
            return
        self.loaded.add(real)
        try:
            with open(real, "r") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError) as e:
            self.diag("load", nsp, f"cannot read \"{path}\": {getattr(e, 'strerror', None) or e}")
            return
        self.load(Source(path, text), os.path.dirname(real), depth + 1)

    # -- call graph --------------------------------------------------------------------
    def graph(self):
        fns = self.functions
        for f in fns.values():
            r = _Refs()
            bound = set(f.params) | set(_locals(f.node))
            stmts, spans = _body(f.node)
            for s, sp in zip(stmts, spans):
                self.scan(s, sp or f.span, bound, r)
            self.refs[f.name] = r
            f.calls = tuple(k for k in r.refs if k in fns or k in self.externs)
        entries = self.report.entries
        if entries and not self.all_functions:
            todo = [e.name for e in entries]
            while todo:
                f = fns.get(todo.pop())
                if f is None or f.reachable:
                    continue
                f.reachable = True
                todo.extend(k for k in f.calls if k in fns)
        else:
            for f in fns.values():
                f.reachable = True
        for f in fns.values():
            if not f.reachable:
                continue
            r = self.refs[f.name]
            for name in r.assigns:
                g = self.globals.get(name)
                if g is not None:
                    g.assigned = True
                    if g.param:
                        self.diag("param", r.assigns[name], f"{name} is a parameter: a compiled function reads it and cannot assign it",
                                  "a parameter is written by the host between runs; keep the running value in another global",
                                  source=f.source)
            for callee in f.calls:
                if callee not in fns:
                    continue
                path = self.path(callee, f.name)
                if path is None:
                    continue
                if callee == f.name:
                    msg = f"{f.name} calls itself"
                else:
                    msg = f"{f.name} calls {callee}, which leads back to {f.name} ({' -> '.join([f.name] + path)})"
                self.diag("recursion", r.refs[callee], msg, source=f.source)

    def path(self, start, goal):
        """The names from start to goal along calls, or None."""
        prev = {start: None}
        todo = [start]
        while todo:
            cur = todo.pop(0)
            if cur == goal:
                out = []
                while cur is not None:
                    out.append(cur)
                    cur = prev[cur]
                return out[::-1]
            for k in self.functions[cur].calls:
                if k in self.functions and k not in prev:
                    prev[k] = cur
                    todo.append(k)
        return None

    def scan(self, n, sp, bound, out):
        """Collect the free names a tree mentions and assigns (no types, no diagnostics)."""
        if n is None or isinstance(n, (KGQSym, KGChar, KGOp, KGLambda, np.ndarray, dict, bool, int, float)):
            return
        if isinstance(n, KGSym):
            s = _name(n)
            if s not in bound:
                out.refs.setdefault(s, sp)
            return
        if isinstance(n, str):
            return
        if isinstance(n, (KGCond, KGExprArray)) or type(n) is list:
            spans = getattr(n, "arg_spans", None) or ()
            for i, q in enumerate(n):
                self.scan(q, tuple(spans[i]) if i < len(spans) and spans[i] is not None else sp, bound, out)
            return
        if not isinstance(n, KGFn):
            return
        nsp = tuple(n.span) if n.span is not None else sp
        a = n.a
        if _is_chain(n):
            verb, operand = a[0].a, a[-1]
            self.scan(verb, tuple(n.op_span) if n.op_span is not None else nsp, bound, out)
            ops = operand if type(operand) is list else [operand]
            for i, q in enumerate(ops):
                self.scan(q, _arg_span(n, i, nsp), bound, out)
            return
        if isinstance(a, KGLambda):
            return
        if isinstance(a, KGOp):
            args = n.args
            if type(args) is not list:
                self.scan(args, _arg_span(n, 0, nsp), bound, out)
                return
            if a.a == "::" and len(args) == 2:
                left = args[0]
                targets = left if type(left) is KGDestructure else [left] if isinstance(left, KGSym) else []
                for t in targets:
                    if _name(t) not in bound:
                        out.assigns.setdefault(_name(t), _arg_span(n, 0, nsp))
                self.scan(args[1], _arg_span(n, 1, nsp), bound, out)
                return
            for i, q in enumerate(args):
                self.scan(q, _arg_span(n, i, nsp), bound, out)
            return
        if _is_fn_literal(n):
            inner = bound | set(_params(n)) | set(_locals(n))
            stmts, spans = _body(n)
            for s, ssp in zip(stmts, spans):
                self.scan(s, ssp or nsp, inner, out)
            return
        # a call, an infix call or a projection
        callee_sp = tuple(n.op_span) if n.op_span is not None else nsp
        if isinstance(a, KGSym):
            self.scan(a, callee_sp, bound, out)
        elif type(a) is KGBody or type(a) is list:
            for q in a:
                self.scan(q, callee_sp, bound, out)
        else:
            self.scan(a, callee_sp, bound, out)
        for i, q in enumerate(_args(n)):
            self.scan(q, _arg_span(n, i, nsp), bound, out)

    # -- roots ----------------------------------------------------------------------
    def roots(self):
        fns = self.functions
        entries = self.report.entries
        for e in entries:
            f = fns.get(e.name)
            if f is None:
                continue
            self.src = f.source
            r = self.call_function(f, [_from_ty(p) for p in e.signature.params], f.span)
            self.check_entry_result(e, f, r)
        if not entries or self.all_functions:
            for f in list(fns.values()):
                self.src = f.source
                self.call_function(f, [UNKNOWN] * f.arity, f.span)

    def check_entry_result(self, e, f, r):
        want = e.signature.result
        if want.elem == "void" or r.kind == "any":
            return
        ssp = e.span
        if r.kind == "void":
            self.diag("entry", ssp, f"entry {e.name} is declared to return {want}, but the function returns nothing "
                                    "(its last statement is a call of a void extern)", source=e.source)
            return
        if (is_fixed(r.elem) or is_fixed(want.elem)) and r.elem is not None and r.elem != want.elem:
            self.diag("entry", ssp, f"entry {e.name} is declared to return {want}, but the function returns {r}",
                      "a fixed-point result must be exactly the declared type: convert it with .fixed(type;value)", source=e.source)
            return
        if r.elem == "float" and want.elem == "int":
            self.diag("entry", ssp, f"entry {e.name} is declared to return {want}, but the function returns {r}",
                      "an int result cannot hold a float: declare float, or floor the result with _", source=e.source)
            return
        if r.shape is not None and tuple(r.shape) != tuple(want.shape):
            self.diag("entry", ssp, f"entry {e.name} is declared to return {want}, but the function returns {r}", source=e.source)

    def finish(self):
        rep = self.report
        fns = list(self.functions.values())
        order = {}
        for d in rep.diagnostics:
            order.setdefault(d.source.name, len(order))
            for f in fns:
                if f.source is d.source and f.span[0] <= d.span[0] < f.span[1]:
                    d.function = f.name
                    f.in_profile = False
                    break
        changed = True
        while changed:
            changed = False
            for f in fns:
                if f.in_profile and any(k in self.functions and not self.functions[k].in_profile for k in f.calls):
                    f.in_profile = False
                    changed = True
        rep.diagnostics.sort(key=lambda d: (order[d.source.name], d.span[0], d.span[1], d.rule))
        for f in fns:
            f.specializations = {k[1]: v for k, v in f.specializations.items()}

    # -- functions ---------------------------------------------------------------------
    def call_function(self, f, args, sp):
        """The abstract result of calling function f with these argument values."""
        if len(args) != f.arity:
            if len(args) < f.arity:
                self.diag("projection", sp, f"{f.name} takes {_plural(f.arity, 'argument')}, got {len(args)}: "
                                            "the call is a projection, which is a function value")
            else:
                self.diag("arity", sp, f"{f.name} takes {_plural(f.arity, 'argument')}, got {len(args)}")
            return ANY
        if f.name in self.stack:
            return ANY          # recursion: reported from the call graph
        for a in args:
            if a.kind == "void":
                self.diag("void-value", sp, f"an argument of {f.name} is the result of a void extern")
                return ANY
        args = [a.plain() for a in args]
        key = (f.name, tuple(a.key() for a in args))
        if key in self.memo and (self.mute or key in self.reported):
            return self.memo[key]
        saved = self.src
        self.src = f.source
        self.stack.append(f.name)
        try:
            r = self.run_literal(f.node, args, None).plain()
        finally:
            self.stack.pop()
            self.src = saved
        self.memo[key] = r
        if not self.mute:
            self.reported.add(key)
            f.specializations[(f.name, tuple(a.ty() for a in args))] = r.ty()
        return r

    def run_literal(self, node, args, parent):
        """
        Walk the body of a function literal with these argument values. The
        type of a local is the join of its assignments, so the body is walked
        (silently) until the types of its variables stop changing, then once
        more to report.
        """
        params = _params(node)
        names = _locals(node)
        stmts, spans = _body(node)
        carried = {}
        self.mute += 1
        try:
            for _ in range(_MAX_PASSES):
                fr = self.frame(params, names, args, parent, carried)
                self.body(stmts, spans, node, fr)
                now = {k: v.ty.elem for k, v in fr.vars.items() if v.ty is not None and v.ty.kind == "num"}
                if now == carried:
                    break
                carried = now
        finally:
            self.mute -= 1
        fr = self.frame(params, names, args, parent, carried)
        return self.body(stmts, spans, node, fr)

    def frame(self, params, names, args, parent, carried):
        fr = _Frame(parent)
        for p, a in zip(params, args):
            if carried.get(p) == "float" and a.kind == "num" and a.elem == "int":
                a = _num("float", a.shape)
            fr.vars[p] = _Var(a, True)
        for q in names:
            if q not in fr.vars:
                fr.vars[q] = _Var(None, False, carried.get(q))
        return fr

    def body(self, stmts, spans, node, fr):
        r = ANY
        default = tuple(node.span) if node.span is not None else None
        for s, sp in zip(stmts, spans):
            r = self.ev(s, tuple(sp) if sp is not None else default, fr)
        return r

    # -- expressions -------------------------------------------------------------------
    def ev(self, n, sp, fr):
        """The abstract value of expression n; sp is its span when the node carries none."""
        nsp = getattr(n, "span", None)
        if nsp is not None:
            sp = tuple(nsp)
        if n is None:
            return ANY
        if isinstance(n, KGQSym):
            self.diag("symbol", sp, f"the symbol :{_name(n)} is not a run-time value")
            return ANY
        if isinstance(n, KGSym):
            return self.read(_name(n), sp, fr)
        if isinstance(n, KGChar):
            self.diag("string", sp, "a character is not a run-time value")
            return ANY
        if isinstance(n, str):
            self.diag("string", sp, "a string is not a run-time value")
            return ANY
        if isinstance(n, (bool, np.bool_)):
            return _num("int", (), int(n))
        if isinstance(n, (int, np.integer)):
            return self.int_literal(int(n), sp)
        if isinstance(n, (float, np.floating)):
            return _num("float", (), float(n))
        if isinstance(n, np.ndarray):
            return self.list_literal(n, sp)
        if isinstance(n, KGCond):
            return self.cond(n, sp, fr)
        if isinstance(n, KGExprDict):
            self.diag("dict", sp, "a dictionary is not a run-time value")
            return ANY
        if isinstance(n, KGExprArray):
            return self.elist(n, sp, fr)
        if isinstance(n, KGOp):
            self.diag("fn-value", sp, f"the operator {n.a} is used as a value")
            return ANY
        if isinstance(n, dict):
            self.diag("dict", sp, "a dictionary is not a run-time value")
            return ANY
        if not isinstance(n, KGFn):
            self.diag("unsupported-op", sp, f"a {type(n).__name__} is not part of the profile")
            return ANY
        a = n.a
        if isinstance(a, KGLambda):
            self.diag("dict", sp, "a dictionary is not a run-time value")
            return ANY
        if _is_chain(n):
            return self.adverb(n, sp, fr)
        if isinstance(a, KGOp):
            if type(n.args) is not list:
                if a.a == "-" and isinstance(n.args, int) and not isinstance(n.args, bool):
                    return self.int_literal(-n.args, sp)      # -2147483648 is in range; 2147483648 alone is not
                return self.ty_monad(a.a, self.ev(n.args, _arg_span(n, 0, sp), fr), sp)
            if len(n.args) != 2:
                self.diag("unsupported-op", sp, f"the operator {a.a} is applied to {len(n.args)} operands")
                return ANY
            if a.a == "::":
                return self.assign(n, sp, fr)
            if a.a == ":=":
                return self.amend(n, sp, fr)
            if a.a == "@" and isinstance(n.args[0], KGSym) and self.is_function(_name(n.args[0]), fr):
                self.ev(n.args[1], _arg_span(n, 1, sp), fr)
                name = _name(n.args[0])
                self.diag("fn-value", _arg_span(n, 0, sp), f"{name}@x applies the function {name} as a value",
                          f"call it by name: {name}(...)")
                return ANY
            right = self.ev(n.args[1], _arg_span(n, 1, sp), fr)     # the right operand is evaluated first
            left = self.ev(n.args[0], _arg_span(n, 0, sp), fr)
            return self.ty_dyad(a.a, left, right, sp)
        if _is_fn_literal(n):
            self.diag("fn-value", sp, "a function literal is used as a value")
            return ANY
        return self.call(n, sp, fr)

    @staticmethod
    def _frames(fr):
        while fr is not None:
            yield fr
            fr = fr.parent

    def is_function(self, name, fr):
        f = fr
        while f is not None:
            if name in f.vars:
                return False
            f = f.parent
        return name in self.functions or name in self.externs

    def int_literal(self, v, sp):
        lo, hi = INT_RANGE[self.int_bits]
        if not lo <= v <= hi:
            self.diag("int-range", sp, f"{v} is outside the range of a {self.int_bits}-bit kg_int")
            return _num("int", ())
        return _num("int", (), v)

    def list_literal(self, arr, sp):
        kind = arr.dtype.kind
        if kind == "O":
            bad = _first_non_number(arr)
            if isinstance(bad, KGSym):
                self.diag("symbol", sp, f"the list holds the symbol :{_name(bad)}: a name inside [...] is not evaluated")
            elif isinstance(bad, KGChar):
                self.diag("string", sp, "the list holds a character")
            elif isinstance(bad, str):
                what = f"the string \"{bad}\"" if bad in (";", "(", ")", "{", "}", ":[", ":|") else "a string"
                self.diag("string", sp, f"the list holds {what}")
            elif isinstance(bad, (dict, KGCall)):
                self.diag("dict", sp, "the list holds a dictionary")
            elif isinstance(bad, KGOp):
                self.diag("list-literal", sp, f"the list holds the operator {bad.a}: inside [...] nothing is evaluated",
                          "write the numbers only; a negative number is written -1 with no space")
            else:
                self.diag("list-literal", sp, "the list is ragged: its rows have different lengths, or it mixes numbers and lists")
            return ANY
        if kind not in "iubf":
            self.diag("string", sp, "the list holds strings")
            return ANY
        shape = tuple(int(d) for d in arr.shape)
        if arr.size == 0:
            self.diag("array-shape", sp, "an empty list has a dimension of 0")
            return ANY
        if len(shape) > MAX_RANK:
            self.diag("array-shape", sp, f"the list has rank {len(shape)}; the rank is 1 to {MAX_RANK}")
            return ANY
        if kind == "f":
            return _num("float", shape, arr if arr.size <= 4096 else _NC)
        lo, hi = INT_RANGE[self.int_bits]
        if int(arr.min()) < lo or int(arr.max()) > hi:
            worst = int(arr.max()) if int(arr.max()) > hi else int(arr.min())
            self.diag("int-range", sp, f"{worst} is outside the range of a {self.int_bits}-bit kg_int")
        return _num("int", shape, arr if arr.size <= 4096 else _NC)

    def read(self, name, sp, fr):
        f = fr
        while f is not None:
            v = f.vars.get(name)
            if v is not None:
                if not v.set:
                    self.diag("unset-local", sp, f"the local {name} is read before it is assigned")
                    return ANY
                return v.ty if v.ty is not None else ANY
            f = f.parent
        g = self.globals.get(name)
        if g is not None:
            return g.at.plain() if g.assigned else g.at
        if name in self.functions:
            self.diag("fn-value", sp, f"the function {name} is used as a value")
            return ANY
        if name in self.externs:
            self.diag("fn-value", sp, f"the extern {name} is used as a value")
            return ANY
        if name == ".f":
            self.diag("recursion", sp, ".f is the running function: calling it is recursion")
            return ANY
        if name.startswith("."):
            self.diag("sys-fn", sp, f"{name} is a system name; it does not exist in a compiled program")
            return ANY
        if name in reserved_fn_args:
            self.diag("undefined-name", sp, f"{name} is not an argument of this function")
            return ANY
        if name in self.broken:
            return ANY
        self.diag("undefined-name", sp, f"{name} is not defined",
                  "a run-time function sees its parameters, its locals, globals defined at top level, functions and externs")
        return ANY

    def cond(self, n, sp, fr):
        if len(n) != 3:
            self.diag("unsupported-op", sp, "a conditional has a condition and two branches")
            return ANY
        c = self.ev(n[0], _arg_span(n, 0, sp), fr)
        if c.kind == "void":
            self.diag("void-value", _arg_span(n, 0, sp), "the condition is the result of a void extern")
        elif c.kind == "num" and c.shape:
            self.diag("cond-scalar", _arg_span(n, 0, sp), f"the condition is {c}; it must be a scalar")
        before = {k: v.set for k, v in fr.vars.items()}
        t = self.ev(n[1], _arg_span(n, 1, sp), fr)
        after_t = {k: v.set for k, v in fr.vars.items()}
        for k, v in fr.vars.items():
            v.set = before[k]
        e = self.ev(n[2], _arg_span(n, 2, sp), fr)
        for k, v in fr.vars.items():
            v.set = v.set and after_t[k]
        if t.kind == "void" and e.kind == "void":
            return VOID
        if t.kind == "any" or e.kind == "any":
            return ANY
        if t.kind == "void" or e.kind == "void":
            self.diag("cond-shape", sp, "one branch of the conditional is a value and the other is the result of a void extern")
            return ANY
        if t.shape is not None and e.shape is not None and t.shape != e.shape:
            self.diag("cond-shape", sp, f"the branches of the conditional are {t} and {e}; they must have the same shape")
            return ANY
        elem = self.join_elem([t, e], sp, "the branches of the conditional")
        if elem is False:
            return ANY
        return _num(elem, t.shape if t.shape == e.shape else None)

    def elist(self, n, sp, fr):
        items = [self.ev(q, _arg_span(n, i, sp), fr) for i, q in enumerate(n)]
        if not items:
            self.diag("array-shape", sp, "an empty list has a dimension of 0")
            return ANY
        shape = items[0].shape if items[0].kind == "num" else None
        for i, it in enumerate(items):
            if it.kind == "void":
                self.diag("void-value", _arg_span(n, i, sp), "a list element is the result of a void extern")
                return ANY
            if it.kind != "num":
                return ANY
            if it.shape is None:
                shape = None
            elif shape is not None and it.shape != shape:
                self.diag("list-literal", sp, f"the elements of the list are {items[0]} and {it}; they must have the same shape")
                return ANY
        elem = self.join_elem(items, sp, "the list")
        if elem is False:
            return ANY
        if shape is None:
            return _num(elem, None)
        return self.mk(elem, (len(items),) + tuple(shape), sp)

    def mk(self, elem, shape, sp):
        """A number of this type, after checking a known shape against the limits of section 2."""
        if shape is not None:
            if any(d <= 0 for d in shape):
                self.diag("array-shape", sp, "the result would have a dimension of 0: every dimension is a positive integer")
                return ANY
            if len(shape) > MAX_RANK:
                self.diag("array-shape", sp, f"the result would have rank {len(shape)}; the rank is 1 to {MAX_RANK}")
                return ANY
            shape = tuple(shape)
        return _num(elem, shape)

    # -- assignment ----------------------------------------------------------------------
    def assign(self, n, sp, fr):
        left, right = n.args
        lsp = _arg_span(n, 0, sp)
        if _is_fn_literal(right):
            self.diag("fn-value", _arg_span(n, 1, sp), "a function is assigned to a variable inside a function",
                      "define the function at top level")
            v = ANY
        else:
            v = self.ev(right, _arg_span(n, 1, sp), fr)
        if v.kind == "void":
            self.diag("void-value", _arg_span(n, 1, sp), "the result of a void extern is assigned")
            v = ANY
        v = v.plain()
        if type(left) is KGDestructure:
            names = [_name(q) for q in left]
            item = ANY
            if v.kind == "num":
                if v.shape is None:
                    item = _num(v.elem, None)
                elif not v.shape:
                    self.diag("shape-mismatch", sp, f"cannot destructure {v} into {len(names)} names: it is not an array")
                elif v.shape[0] != len(names):
                    self.diag("shape-mismatch", sp, f"cannot destructure {v} into {_plural(len(names), 'name')}: "
                                                    f"it has {_plural(v.shape[0], 'item')}")
                else:
                    item = _num(v.elem, v.shape[1:])
            for q in names:
                self.store(q, item, lsp, fr)
            return v
        if not isinstance(left, KGSym) or isinstance(left, KGQSym):
            self.diag("strict-syntax", lsp, "the left side of :: must be a name")
            return v
        self.store(_name(left), v, lsp, fr)
        return v

    def store(self, name, v, sp, fr):
        var = fr.vars.get(name)
        if var is not None:
            old = var.ty
            if v.kind != "num" or (old is not None and old.kind != "num"):
                var.ty = ANY
            elif old is None:
                var.ty = _num("float" if (var.hint == "float" and v.elem == "int") else v.elem, v.shape)
            else:
                shape = old.shape
                if old.shape is not None and v.shape is not None and old.shape != v.shape:
                    self.diag("shape-change", sp, f"{name} was {old} and is assigned {v}: the shape of a variable is fixed")
                elif v.shape is None:
                    shape = None
                if (is_fixed(old.elem) or is_fixed(v.elem)) and old.elem is not None and v.elem is not None \
                        and old.elem != v.elem:
                    self.diag("fixed", sp, f"{name} was {old} and is assigned {v}: a variable does not change its type",
                              f'convert the value: .fixed("{old.elem if is_fixed(old.elem) else v.elem}";value)')
                    var.ty = ANY
                    var.set = True
                    return
                elem = "float" if "float" in (old.elem, v.elem) else (old.elem if old.elem == v.elem else None)
                var.ty = _num(elem, shape)
            var.set = True
            return
        f = fr.parent
        while f is not None:
            if name in f.vars:
                self.diag("closure-assign", sp, f"the nested function assigns {name}, a variable of the function it is written in")
                f.vars[name].set = True      # reported once, here; not again where it is read
                f.vars[name].ty = ANY
                return
            f = f.parent
        g = self.globals.get(name)
        if g is not None:
            have = g.at
            if g.param:
                self.diag("param", sp, f"{name} is a parameter: a compiled function reads it and cannot assign it",
                          "a parameter is written by the host between runs; keep the running value in another global")
                return
            if v.kind == "num" and have.kind == "num":
                if have.shape is not None and v.shape is not None and have.shape != v.shape:
                    self.diag("shape-change", sp, f"the global {name} is {have.plain()} and is assigned {v}: "
                                                  "the shape of a global is fixed")
                elif have.elem is not None and v.elem is not None and have.elem != v.elem:
                    hint = ("state must not change type: floor the value with _, or define the global as a float"
                            if have.elem == "int" else
                            "state must not change type: write the value as a float (0.0, or multiply by 1.0)"
                            if have.elem == "float" else
                            f'state must not change type: convert the value with .fixed("{have.elem}";value)')
                    self.diag("shape-change", sp, f"the global {name} is {have.plain()} and is assigned {v}: "
                                                  "the type of a global is fixed", hint)
            return
        if name in self.functions or name in self.externs:
            what = "function" if name in self.functions else "extern"
            self.diag("redefinition", sp, f"{name} is a {what}; it cannot be assigned")
            return
        if name.startswith("."):
            self.diag("sys-fn", sp, f"{name} is a system name; it cannot be assigned in a compiled program")
            return
        self.diag("undefined-name", sp, f"{name} is assigned but is neither a local nor a global",
                  f"declare it as a local ({{[{name}];...}}) or define the global at top level first")

    # -- calls -----------------------------------------------------------------------------
    def call(self, n, sp, fr):
        callee = n.a
        args = _args(n)
        csp = tuple(n.op_span) if n.op_span is not None else sp
        if any(q is None for q in args):
            for i, q in enumerate(args):
                if q is not None:
                    self.ev(q, _arg_span(n, i, sp), fr)
            who = _name(callee) if isinstance(callee, KGSym) else "the function"
            self.diag("projection", sp, f"the call of {who} leaves an argument out: that is a projection, a function value")
            return ANY
        if not isinstance(callee, KGSym) or isinstance(callee, KGQSym):
            for i, q in enumerate(args):
                self.ev(q, _arg_span(n, i, sp), fr)
            self.diag("fn-value", csp, "a function literal is called in place",
                      "give it a name at top level and call it by name")
            return ANY
        name = _name(callee)
        f = fr
        while f is not None:
            if name in f.vars:
                for i, q in enumerate(args):
                    self.ev(q, _arg_span(n, i, sp), fr)
                self.diag("fn-value", csp, f"{name} is a variable called as a function: functions cannot be passed or stored")
                return ANY
            f = f.parent
        if name in self.functions:
            ats = [self.arg(q, _arg_span(n, i, sp), fr) for i, q in enumerate(args)]
            return self.call_function(self.functions[name], ats, sp)
        if name in self.externs:
            return self.call_extern(self.externs[name], n, args, sp, fr)
        if name == ".f":
            self.diag("recursion", csp, ".f is the running function: calling it is recursion")
            return ANY
        if name in (".fixed", ".float"):
            return self.conversion(name, n, args, sp, csp, fr)
        if name.startswith("."):
            self.diag("sys-fn", csp, f"{name} is a system function; it does not exist in a compiled program")
            return ANY
        for i, q in enumerate(args):
            self.ev(q, _arg_span(n, i, sp), fr)
        if name in self.globals:
            self.diag("fn-value", csp, f"{name} is a global, not a function")
        elif name not in self.broken:
            self.diag("undefined-name", csp, f"the function {name} is not defined",
                      "a function provided by the target is declared with .extern")
        return ANY

    def conversion(self, name, n, args, sp, csp, fr):
        """.fixed(type;x) and .float(x): the conversions between fixed-point numbers and the other types."""
        want = 2 if name == ".fixed" else 1
        if len(args) != want:
            for i, q in enumerate(args):
                if not _is_plain_str(q):
                    self.ev(q, _arg_span(n, i, sp), fr)
            self.diag("fixed", csp, f"{name} takes {_plural(want, 'argument')}, got {len(args)}")
            return ANY
        ty = None
        if name == ".fixed":
            tsp = _arg_span(n, 0, sp)
            if not _is_plain_str(args[0]):
                self.diag("fixed", tsp, "the type of .fixed must be a string literal such as \"fixed<32,16>\"")
            else:
                try:
                    ty = parse_type(args[0])
                except SignatureError as e:
                    self.diag("signature", tsp, f".fixed type: {e.message}")
                    ty = None
                if ty is not None and (not is_fixed(ty.elem) or ty.shape):
                    self.diag("fixed", tsp, f".fixed converts to fixed<W,I>, not {ty}")
                    ty = None
        v = self.arg(args[-1], _arg_span(n, len(args) - 1, sp), fr)
        if v.kind == "void":
            self.diag("void-value", sp, f"the argument of {name} is the result of a void extern")
            return ANY
        if v.kind != "num" or (name == ".fixed" and ty is None):
            return ANY
        if name == ".fixed":
            return _num(ty.elem, v.shape)
        return _num("float", v.shape)

    def arg(self, q, sp, fr):
        if _is_fn_literal(q) or (isinstance(q, KGSym) and not isinstance(q, KGQSym) and self.is_function(_name(q), fr)):
            self.diag("fn-value", sp, "a function is passed as an argument",
                      "a function may only stand directly as the verb of an adverb")
            return ANY
        return self.ev(q, sp, fr)

    def call_extern(self, decl, n, args, sp, fr):
        sig = decl.signature
        if len(args) != sig.arity:
            for i, q in enumerate(args):
                if not _is_plain_str(q):
                    self.ev(q, _arg_span(n, i, sp), fr)
            self.diag("extern-call", sp, f"{decl.name} is declared \"{sig}\" and takes {_plural(sig.arity, 'argument')}, got {len(args)}")
            return _from_ty(sig.result)
        for i, (q, ty) in enumerate(zip(args, sig.params)):
            asp = _arg_span(n, i, sp)
            if ty.elem == "str":
                if not _is_plain_str(q):
                    self.ev(q, asp, fr)
                    self.diag("extern-call", asp, f"argument {i + 1} of {decl.name} is declared str: it must be a string literal")
                continue
            self.extern_arg(decl, i, ty, self.arg(q, asp, fr), asp)
        return _from_ty(sig.result)

    def extern_arg(self, decl, i, ty, a, asp):
        what = f"argument {i + 1} of {decl.name}"
        if ty.elem == "str":
            self.diag("extern-call", asp, f"{what} is declared str: it must be a string literal")
            return
        if a.kind == "void":
            self.diag("void-value", asp, f"{what} is the result of a void extern")
            return
        if a.kind != "num":
            return
        if a.shape is not None and tuple(a.shape) != tuple(ty.shape):
            self.diag("extern-call", asp, f"{what} is declared {ty}, got {a}")
        elif a.elem == "float" and ty.elem == "int":
            self.diag("extern-call", asp, f"{what} is declared {ty}, got {a}", "floor the value with _ to pass an int")

    # -- adverbs ------------------------------------------------------------------------------
    def verb(self, v, need, sp, fr, role="verb"):
        """
        The verb of an adverb as a function from argument values to a result
        value, or None (after a diagnostic) if it cannot be one. need is the
        number of arguments the adverb gives it.
        """
        if isinstance(v, KGOp):
            glyph = v.a
            valence = "monad" if need == 1 else "dyad"
            if glyph in ("::", ":="):
                self.diag("unsupported-op", sp, f"{glyph} cannot be the verb of an adverb",
                          "wrap it in a function literal")
                return None
            if need == 1:
                return lambda args, at: self.ty_monad(glyph, args[0], at)
            return lambda args, at: self.ty_dyad(glyph, args[0], args[1], at)
        if isinstance(v, KGSym) and not isinstance(v, KGQSym):
            name = _name(v)
            f = fr
            while f is not None:
                if name in f.vars:
                    self.diag("fn-value", sp, f"{name} is a variable used as the {role} of an adverb: functions cannot be passed or stored")
                    return None
                f = f.parent
            if name in self.functions:
                fn = self.functions[name]
                if fn.arity != need:
                    self.diag("arity", sp, f"{name} takes {_plural(fn.arity, 'argument')}; here it is given {need}")
                    return None
                return lambda args, at: self.call_function(fn, args, at)
            if name in self.externs:
                decl = self.externs[name]
                if decl.signature.arity != need:
                    self.diag("extern-call", sp, f"{name} is declared \"{decl.signature}\"; here it is given {_plural(need, 'argument')}")
                    return None

                def call(args, at):
                    for i, (ty, a) in enumerate(zip(decl.signature.params, args)):
                        self.extern_arg(decl, i, ty, a, at)
                    return _from_ty(decl.signature.result)
                return call
            self.read(name, sp, fr)     # reports .f, a system name, an undefined name
            return None
        if _is_fn_literal(v):
            have = len(_params(v))
            if have != need:
                self.diag("arity", sp, f"the function literal takes {_plural(have, 'argument')}; here it is given {need}")
                return None
            return lambda args, at: self.run_literal(v, [a.plain() for a in args], fr).plain()
        if isinstance(v, KGFn):
            self.diag("fn-value", sp, f"the {role} of the adverb is computed: it must be an operator, a function name or a function literal")
        else:
            self.diag("fn-value", sp, f"the {role} of the adverb must be an operator, a function name or a function literal")
        return None

    def adverb(self, n, sp, fr):
        chain = n.a
        verb = chain[0].a
        advs = [q.a for q in chain[1:-1]]
        operand = chain[-1]
        dyadic = type(operand) is list
        vsp = tuple(n.op_span) if n.op_span is not None else sp
        if dyadic:
            right = operand[1]
            left = operand[0]
        if len(advs) != 1:
            self.operands(n, operand, dyadic, sp, fr)
            self.diag("unsupported-op", vsp, f"the adverb chain {''.join(advs)} is not in the profile",
                      "apply one adverb at a time: wrap the inner one in a function literal, {+/x}'a")
            return ANY
        glyph = advs[0]
        row = lookup(glyph, "adverb2" if dyadic else "adverb1")
        if row is None or row.status != "allowed":
            self.operands(n, operand, dyadic, sp, fr)
            if row is None:
                self.diag("unsupported-op", vsp, f"the adverb {glyph} is not in the profile")
            else:
                self.diag(row.rule, vsp, f"{glyph} ({row.name}): {row.note}")
            return ANY
        kind = row.id
        if kind == "while":
            b = self.ev(right, _arg_span(n, 1, sp), fr)
            pred = self.verb(left, 1, _arg_span(n, 0, sp), fr, role="predicate")
            f = self.verb(verb, 1, vsp, fr)
            return self.loop(kind, f, pred, b, sp, _arg_span(n, 0, sp))
        if kind == "iterate":
            b = self.ev(right, _arg_span(n, 1, sp), fr)
            count = self.ev(left, _arg_span(n, 0, sp), fr)
            if count.kind == "num" and (count.shape or count.elem == "float"):
                self.diag("operand", _arg_span(n, 0, sp), f"the count of Iterate is {count}; it must be an int scalar")
            refs = _Refs()
            self.scan(left, sp, set(), refs)
            hit = [k for k in refs.refs if k in self.report.params and not any(k in f_.vars for f_ in self._frames(fr))]
            if hit:
                self.diag("param", _arg_span(n, 0, sp), f"the count of Iterate is the parameter {hit[0]}: a loop bound cannot depend on "
                          "one, since the hardware is built for a fixed number of iterations",
                          "iterate a fixed maximum and keep the value once the parameter's count is reached")
            f = self.verb(verb, 1, vsp, fr)
            return self.loop(kind, f, None, b, sp, None)
        need = 1 if kind == "each" else 2
        if dyadic:
            b = self.ev(right, _arg_span(n, 1, sp), fr)
            a = self.ev(left, _arg_span(n, 0, sp), fr)
        else:
            b = self.ev(operand, _arg_span(n, 0, sp), fr)
            a = None
        f = self.verb(verb, need, vsp, fr)
        for v in (a, b):
            if v is not None and v.kind == "void":
                self.diag("void-value", sp, f"an operand of {glyph} ({row.name}) is the result of a void extern")
                return ANY
        if f is None or b.kind != "num" or (a is not None and a.kind != "num"):
            return ANY
        if b.shape == ():
            self.diag("operand", sp, f"{glyph} ({row.name}) needs an array; the operand is {b}")
            return ANY
        item = _num(b.elem, b.shape[1:] if b.shape is not None else None)
        n_items = b.shape[0] if b.shape is not None else None
        if kind == "each":
            r = f([item], sp)
            return self.stacked(r, n_items, sp)
        if kind == "each2":
            if a.shape == ():
                self.diag("operand", sp, f"{glyph} ({row.name}) needs two arrays; the left operand is {a}")
                return ANY
            if a.shape is not None and n_items is not None and a.shape[0] != n_items:
                self.diag("shape-mismatch", sp, f"{glyph} ({row.name}) pairs the items of {a} and {b}: the lengths differ",
                          "KlongPy stops at the shorter operand; compiled code needs equal lengths")
                return ANY
            if n_items is None and a.shape is not None:
                n_items = a.shape[0]
            r = f([_num(a.elem, a.shape[1:] if a.shape is not None else None), item], sp)
            return self.stacked(r, n_items, sp)
        if kind == "each-pair":
            r = f([item, item], sp)
            if n_items is not None and n_items < 2:
                self.diag("array-shape", sp, f"{glyph} ({row.name}) of {b} has no pairs: the result would be empty")
                return ANY
            return self.stacked(r, None if n_items is None else n_items - 1, sp)
        # over and scan, with or without a seed
        seed = a.plain() if a is not None else item
        if a is not None and is_fixed(item.elem) and a.elem == "int" and _fixed_holds(item.elem, a.const):
            seed = _num(item.elem, a.shape)         # a constant int seed converts exactly
        acc = self.accumulate(f, seed, item, sp, row)
        if acc.kind != "num":
            return acc
        if kind in ("over", "over-seed"):
            return acc
        if n_items is None or acc.shape is None:
            return _num(acc.elem, None)
        return self.mk(acc.elem, (n_items + (1 if kind == "scan-seed" else 0),) + tuple(acc.shape), sp)

    def operands(self, n, operand, dyadic, sp, fr):
        if dyadic:
            for i in (1, 0):
                if not _is_fn_literal(operand[i]):
                    self.ev(operand[i], _arg_span(n, i, sp), fr)
        else:
            self.ev(operand, _arg_span(n, 0, sp), fr)

    def stacked(self, r, n_items, sp):
        if r.kind == "void":
            return VOID
        if r.kind != "num":
            return ANY
        if n_items is None or r.shape is None:
            return _num(r.elem, None)
        return self.mk(r.elem, (n_items,) + tuple(r.shape), sp)

    def accumulate(self, f, acc, item, sp, row):
        """The fixed point of acc = f(acc, item), allowing one promotion from int to float."""
        for _ in range(2):
            self.mute += 1
            try:
                r = f([acc, item], sp)
            finally:
                self.mute -= 1
            if r.kind != "num" or acc.kind != "num":
                break
            if r.shape is not None and acc.shape is not None and r.shape != acc.shape:
                self.diag("loop-state", sp, f"{row.glyph} ({row.name}): the accumulator is {acc} and the verb returns {r}; "
                                            "its shape must not change")
                return ANY
            if r.elem == "float" and acc.elem == "int":
                acc = _num("float", acc.shape)
                continue
            if r.elem == "int" and acc.elem == "float":
                break       # the accumulator is float; the verb's int result is converted
            break
        r = f([acc, item], sp)
        if r.kind == "void":
            self.diag("void-value", sp, f"the verb of {row.glyph} ({row.name}) returns nothing")
            return ANY
        if r.kind != "num" or acc.kind != "num":
            return ANY
        return _num(_promote(acc.elem, r.elem) if acc.elem and r.elem else None,
                    acc.shape if acc.shape is not None else r.shape)

    def loop(self, kind, f, pred, state, sp, psp):
        name = "Iterate" if kind == "iterate" else "While"
        if state.kind == "void":
            self.diag("void-value", sp, f"the starting value of {name} is the result of a void extern")
            return ANY
        state = state.plain()
        if pred is not None:
            c = pred([state], psp)
            if c.kind == "num" and c.shape:
                self.diag("cond-scalar", psp, f"the predicate of While returns {c}; it must be a scalar")
            elif c.kind == "void":
                self.diag("void-value", psp, "the predicate of While returns nothing")
        if f is None:
            return ANY
        r = f([state], sp)
        if r.kind == "void":
            self.diag("void-value", sp, f"the verb of {name} returns nothing")
            return ANY
        if r.kind != "num" or state.kind != "num":
            return ANY
        if (r.shape is not None and state.shape is not None and r.shape != state.shape) or \
                (r.elem is not None and state.elem is not None and r.elem != state.elem):
            self.diag("loop-state", sp, f"{name}: the state starts as {state} and the verb returns {r}; "
                                        "its type and shape must not change")
            return ANY
        return _num(state.elem, state.shape)

    # -- operators -------------------------------------------------------------------------------
    def rejected(self, glyph, valence, sp):
        """None if the operator is allowed, else ANY after the diagnostic."""
        row = lookup(glyph, valence)
        if row is None:
            self.diag("unsupported-op", sp, f"the operator {glyph} is not in the profile")
            return None, ANY
        if row.status != "allowed":
            shown = {"monad": f"{glyph}a", "dyad": f"a{glyph}b"}[valence]
            self.diag(row.rule, sp, f"{shown} ({row.name}): {row.note}")
            return None, ANY
        return row, None

    def join_elem(self, ats, sp, what):
        """
        The element type of values that meet in one place (operands of an operator, branches of a
        conditional, items of a list). Fixed-point numbers do not mix with floats or with fixed-point
        numbers of another type, and an int joins a fixed-point number only as a constant that fits.
        False after a diagnostic.
        """
        elems = [x.elem for x in ats]
        fx = [e for e in elems if is_fixed(e)]
        if not fx:
            out = "int"
            for e in elems:
                out = _promote(out, e) if out is not None else None
            return out
        if None in elems:
            return None
        target = fx[0]
        for x in ats:
            if x.elem == target:
                continue
            if is_fixed(x.elem):
                self.diag("fixed", sp, f"{what} mixes {target} and {x.elem}",
                          f'convert one of them: .fixed("{x.elem}";value) or .fixed("{target}";value)')
                return False
            if x.elem == "float":
                self.diag("fixed", sp, f"{what} mixes {target} with a float: they do not convert implicitly",
                          f'write the float as a fixed-point constant .fixed("{target}";value), or convert the {target} with .float(value)')
                return False
            fits = _fixed_holds(target, x.const)
            if fits is None:
                self.diag("fixed", sp, f"{what} mixes {target} with the int {x}, which is not a constant: an int converts only where it is exact",
                          f'convert it explicitly: .fixed("{target}";value)')
                return False
            if not fits:
                self.diag("fixed", sp, f"{what}: the int constant does not fit {target} (its integer part has {fixed_spec(target)[1]} bits)",
                          f'convert it explicitly: .fixed("{target}";value) wraps')
                return False
        return target

    _FIXED_OPS = ("add", "sub", "mul", "div", "max", "min", "lt", "gt", "eq")

    def fixed_dyad(self, row, glyph, a, b, shape, sp):
        """The type of an atomic dyad with a fixed-point operand."""
        k = row.id
        elem = self.join_elem([a, b], sp, f"{glyph} ({row.name})")
        if elem is False:
            return ANY
        if k not in self._FIXED_OPS:
            self.diag("fixed", sp, f"{glyph} ({row.name}) is not defined on fixed-point numbers",
                      "the fixed-point operators are + - * % | & < > = (and _ to floor to an int); "
                      "use .float(x) to leave fixed-point for the others")
            return ANY
        if elem is not None and k == "div":
            if fixed_spec(elem)[0] > 32:
                self.diag("fixed", sp, f"% (Divide) of {elem}: fixed-point division is defined for 32 bits or fewer",
                          "a 64-bit quotient needs a 128-bit divider: divide narrower values and widen the result with .fixed")
                return ANY
            if b.const is not _NC and b.const is not _CU and np.any(np.asarray(b.const) == 0):
                self.diag("div-zero", sp, f"{glyph} ({row.name}) divides a fixed-point number by the constant 0",
                          "compiled code traps on a fixed-point division by zero")
                return ANY
        if k in ("lt", "gt", "eq"):
            return _num("int", shape)
        return _num(elem, shape)

    def ty_monad(self, glyph, a, sp):
        row, bad = self.rejected(glyph, "monad", sp)
        if row is None:
            return bad
        if a.kind == "void":
            self.diag("void-value", sp, f"the operand of {glyph} ({row.name}) is the result of a void extern")
            return ANY
        if a.kind != "num":
            return ANY
        k = row.id
        if is_fixed(a.elem) and k in ("recip",):
            self.diag("fixed", sp, f"{glyph} ({row.name}) of {a.elem}: a reciprocal is a float",
                      "divide instead: .fixed(\"" + a.elem + "\";1)%x, or 1%x")
            return ANY
        if k == "neg":
            c = a.const
            if c is not _NC and c is not _CU:
                c = -c
            return _num(a.elem, a.shape, c)
        if k == "floor":
            return _num("int", a.shape, a.const if _is_int_const(a) else _NC)
        if k == "recip":
            return _num("float", a.shape)
        if k == "not":
            return _num("int", a.shape)
        if k in ("size", "shape"):
            if a.shape == ():
                self.diag("size-of-atom", sp, f"{glyph} ({row.name}) of {a}: the operand is an atom")
                return ANY
            if a.shape is None:
                return _num("int", () if k == "size" else None, _CU)
            if k == "size":
                return _num("int", (), int(a.shape[0]))
            return _num("int", (len(a.shape),), np.array(a.shape, dtype=np.int64))
        if k == "first":
            if a.shape is None:
                return _num(a.elem, None)
            return _num(a.elem, a.shape[1:]) if a.shape else a.plain()
        if k == "reverse":
            return _num(a.elem, a.shape)
        if k == "transpose":
            if a.shape is None:
                return _num(a.elem, None)
            if len(a.shape) != 2:
                self.diag("operand", sp, f"+ (Transpose) needs a rank-2 array; the operand is {a}")
                return ANY
            return _num(a.elem, (a.shape[1], a.shape[0]))
        if k == "enumerate":
            c = self.count(a, sp, "!n (Enumerate)", "n")
            if c is None:
                return ANY
            if c is _CU:
                return _num("int", None)
            if c <= 0:
                self.diag("array-shape", sp, f"!{c} would be empty: n must be positive")
                return ANY
            return _num("int", (c,), np.arange(c, dtype=np.int64) if c <= 4096 else _NC)
        if k == "list":
            return self.mk(a.elem, None if a.shape is None else (1,) + tuple(a.shape), sp)
        return ANY

    def count(self, a, sp, what, word="count"):
        """The constant count of take, drop or enumerate: an int, _CU, or None after a diagnostic."""
        if a.shape:
            self.diag("operand", sp, f"the {word} of {what} is {a}; it must be an int scalar")
            return None
        if a.elem == "float":
            self.diag("operand", sp, f"the {word} of {what} is a float; it must be an int")
            return None
        if a.const is _NC:
            self.diag("const-count", sp, f"the {word} of {what} is not a compile-time constant: the result length would depend on run-time data")
            return None
        if a.const is _CU:
            return _CU
        return int(a.const)

    def prefix(self, a, b, sp, row):
        """The result shape of an atomic dyad: (True, shape) or (False, None) after a diagnostic."""
        if a.shape is None or b.shape is None:
            return True, None
        if not a.shape:
            return True, b.shape
        if not b.shape:
            return True, a.shape
        short, long_ = (a.shape, b.shape) if len(a.shape) <= len(b.shape) else (b.shape, a.shape)
        if long_[:len(short)] != short:
            self.diag("shape-mismatch", sp, f"cannot apply {row.glyph} ({row.name}) to {a} and {b}")
            return False, None
        return True, long_

    _FOLD = {
        "add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y,
        "max": max, "min": min,
    }

    def ty_dyad(self, glyph, a, b, sp):
        row, bad = self.rejected(glyph, "dyad", sp)
        if row is None:
            return bad
        if a.kind == "void" or b.kind == "void":
            self.diag("void-value", sp, f"an operand of {glyph} ({row.name}) is the result of a void extern")
            return ANY
        if a.kind != "num" or b.kind != "num":
            return ANY
        k = row.id
        if row.shape_rule == "prefix":
            ok, shape = self.prefix(a, b, sp, row)
            if not ok:
                return ANY
            zero = _is_int_const(b) and b.const == 0 and b.elem == "int"
            if zero and a.elem == "int" and (k in ("idiv", "rem") or (k == "div" and a.const is not _NC)):
                self.diag("div-zero", sp, f"{glyph} ({row.name}) divides an int by the constant 0",
                          "compiled code traps on an integer division by zero")
                return ANY
            if is_fixed(a.elem) or is_fixed(b.elem):
                return self.fixed_dyad(row, glyph, a, b, shape, sp)
            rule = row.type_rule
            if rule == "promote":
                elem = _promote(a.elem, b.elem)
            elif rule == "float":
                elem = "float"
            elif rule == "int":
                elem = "int"
            elif rule == "int-only":
                for side, v in (("left", a), ("right", b)):
                    if v.elem == "float":
                        self.diag("operand", sp, f":% (Integer-Divide) needs int operands; the {side} operand is {v}",
                                  "floor the operand with _ first, or divide with %")
                        return ANY
                elem = "int"
            else:   # power
                if a.elem == "int" and b.elem == "int" and _is_int_const(b) and b.const >= 0:
                    elem = "int"
                elif a.elem is None or b.elem is None or (a.elem == "int" and b.elem == "int" and b.const is _CU):
                    elem = None
                else:
                    elem = "float"
            const = _NC
            if a.const is not _NC and b.const is not _NC:
                const = _CU
                if _is_int_const(a) and _is_int_const(b):
                    if k in self._FOLD:
                        const = self._FOLD[k](a.const, b.const)
                    elif k in ("idiv", "rem") and b.const != 0:
                        q = abs(a.const) // abs(b.const)
                        q = q if (a.const < 0) == (b.const < 0) else -q     # truncates toward zero, as C does
                        const = q if k == "idiv" else a.const - q * b.const
            if shape is not None and len(shape) > MAX_RANK:
                return self.mk(elem, shape, sp)
            return _num(elem, shape, const if shape == () else _NC)
        if k == "match":
            if (is_fixed(a.elem) or is_fixed(b.elem)) and self.join_elem([a, b], sp, "~ (Match)") is False:
                return ANY
            if a.shape is not None and b.shape is not None and a.shape != b.shape:
                self.diag("shape-mismatch", sp, f"~ (Match) compares {a} with {b}: the shapes differ, so the answer is always 0",
                          "the profile matches operands of the same shape only")
                return ANY
            return _num("int", ())
        if k == "index":
            if a.shape == ():
                self.diag("operand", sp, f"cannot index {a}: @ (Index) needs an array on the left")
                return ANY
            if b.elem == "float":
                self.diag("operand", sp, f"the index is {b}; an index must be an int", "floor it with _")
                return ANY
            if a.shape is not None and b.const is not _NC and b.const is not _CU:
                n = a.shape[0]
                idx = np.asarray(b.const)
                if idx.size and (int(idx.min()) < -n or int(idx.max()) >= n):
                    worst = int(idx.max()) if int(idx.max()) >= n else int(idx.min())
                    self.diag("index-range", sp, f"index {worst} is out of range for {a}",
                              f"valid indices are 0 to {n - 1}, or -{n} to -1 counting from the end")
                    return ANY
            if a.shape is None or b.shape is None:
                return _num(a.elem, None)
            return self.mk(a.elem, tuple(b.shape) + tuple(a.shape[1:]), sp)
        if k == "take":
            c = self.count(a, sp, "n#a (Take)")
            if b.shape is not None and len(b.shape) != 1:
                self.diag("take-rank", sp, f"n#a (Take) from {b}: version 1 takes from rank-1 arrays only")
                return ANY
            if c is None:
                return ANY
            if c is _CU:
                return _num(b.elem, None)
            if c == 0:
                self.diag("array-shape", sp, "0#a would be empty: every dimension is a positive integer")
                return ANY
            return _num(b.elem, (abs(c),))
        if k == "drop":
            c = self.count(a, sp, "n_a (Drop)")
            if b.shape == ():
                self.diag("operand", sp, f"n_a (Drop) from {b}: the right operand must be an array")
                return ANY
            if c is None:
                return ANY
            if c is _CU or b.shape is None:
                return _num(b.elem, None)
            left = b.shape[0] - abs(c)
            if left <= 0:
                self.diag("array-shape", sp, f"dropping {abs(c)} from {b} leaves nothing: every dimension is a positive integer")
                return ANY
            return _num(b.elem, (left,) + tuple(b.shape[1:]))
        if k == "join":
            elem = self.join_elem([a, b], sp, ", (Join)")
            if elem is False:
                return ANY
            if a.shape is None or b.shape is None:
                return _num(elem, None)
            sa, sb = tuple(a.shape), tuple(b.shape)
            if not sa and not sb:
                return _num(elem, (2,))
            if not sa or not sb:
                arr = sb if not sa else sa
                if len(arr) != 1:
                    self.diag("shape-mismatch", sp, f"joining {a} with {b} would give a ragged list")
                    return ANY
                return _num(elem, (arr[0] + 1,))
            if len(sa) != len(sb) or sa[1:] != sb[1:]:
                hint = "to append a row to a matrix write m,,row" if abs(len(sa) - len(sb)) == 1 else None
                self.diag("shape-mismatch", sp, f"joining {a} with {b} would give a ragged list", hint)
                return ANY
            return _num(elem, (sa[0] + sb[0],) + sa[1:])
        if k == "rotate":
            if a.shape or a.elem == "float":
                self.diag("operand", sp, f"the count of n:+a (Rotate) is {a}; it must be an int scalar")
                return ANY
            if b.shape is not None and len(b.shape) != 1:
                self.diag("operand", sp, f"n:+a (Rotate) of {b}: the profile rotates rank-1 arrays only",
                          "KlongPy rotates the flattened data of a matrix")
                return ANY
            return _num(b.elem, b.shape)
        if k == "reshape":
            if a.elem == "float":
                self.diag("operand", sp, f"the shape of s:^a (Reshape) is {a}; it must hold ints")
                return ANY
            if a.const is _NC:
                self.diag("const-count", sp, "the shape of s:^a (Reshape) is not a compile-time constant: "
                                             "the result shape would depend on run-time data")
                return ANY
            if a.const is _CU:
                return _num(b.elem, None)
            dims = np.asarray(a.const)
            if dims.ndim > 1:
                self.diag("operand", sp, f"the shape of s:^a (Reshape) is {a}; it must be an int or a list of ints")
                return ANY
            if dims.ndim == 0 and b.shape is not None and len(b.shape) > 1:
                # KlongPy answers n:^matrix with rows or with elements depending on n; only the list form is defined here
                self.diag("operand", sp, f"n:^a (Reshape) of {b} with a scalar n: the profile reshapes a matrix by a list of dimensions only",
                          "write the shape as a list: [n]:^a")
                return ANY
            shape = tuple(int(d) for d in dims.reshape(-1))
            return self.mk(b.elem, shape, sp)
        if k == "amend":
            self.diag("amend-form", sp, ":= (Amend) cannot be the verb of an adverb")
            return ANY
        self.diag("unsupported-op", sp, f"{glyph} ({row.name}) cannot be used here")
        return ANY

    def amend(self, n, sp, fr):
        target, pair = n.args
        psp = _arg_span(n, 1, sp)
        if not (isinstance(pair, KGFn) and not isinstance(pair, KGCall) and isinstance(pair.a, KGOp) and pair.a.a == ","
                and type(pair.args) is list and len(pair.args) == 2):
            self.ev(pair, psp, fr)
            self.ev(target, _arg_span(n, 0, sp), fr)
            self.diag("amend-form", psp, "the right side of := (Amend) must be written v,i: one value, then the index or indices")
            return ANY
        if pair.span is not None:
            psp = tuple(pair.span)
        idx = self.ev(pair.args[1], _arg_span(pair, 1, psp), fr)
        val = self.ev(pair.args[0], _arg_span(pair, 0, psp), fr)
        arr = self.ev(target, _arg_span(n, 0, sp), fr)
        for v in (idx, val, arr):
            if v.kind == "void":
                self.diag("void-value", sp, "an operand of := (Amend) is the result of a void extern")
                return ANY
        if arr.kind != "num":
            return ANY
        ok = True
        if arr.shape is not None and len(arr.shape) != 1:
            self.diag("amend-form", sp, f":= (Amend) of {arr}: the profile amends rank-1 arrays only")
            ok = False
        if val.kind == "num" and val.shape:
            self.diag("amend-form", _arg_span(pair, 0, psp), f"the value of := (Amend) is {val}; it must be one scalar")
            ok = False
        if idx.kind == "num":
            if idx.elem == "float":
                self.diag("operand", _arg_span(pair, 1, psp), f"the index of := (Amend) is {idx}; an index must be an int")
                ok = False
            elif idx.shape is not None and len(idx.shape) > 1:
                self.diag("amend-form", _arg_span(pair, 1, psp), f"the index of := (Amend) is {idx}; it must be a scalar or a rank-1 list")
                ok = False
            elif ok and arr.shape is not None and idx.const is not _NC and idx.const is not _CU:
                size = arr.shape[0]
                where = np.asarray(idx.const)
                if where.size and (int(where.min()) < -size or int(where.max()) >= size):
                    worst = int(where.max()) if int(where.max()) >= size else int(where.min())
                    self.diag("index-range", _arg_span(pair, 1, psp), f"index {worst} is out of range for {arr}")
                    ok = False
        if not ok:
            return ANY
        if val.kind != "num":
            return _num(None, arr.shape)
        elem = self.join_elem([arr, val], sp, ":= (Amend)")
        if elem is False:
            return ANY
        if is_fixed(elem) and elem != arr.elem:
            self.diag("fixed", sp, f":= (Amend) of {arr} with {val}: the array would change its type",
                      f'convert the value: .fixed("{arr.elem}";value)')
            return ANY
        return _num(elem, arr.shape)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def check_source(text, name="<string>", *, int_bits=32, globals=None, base_dir=None, resolve=True, all_functions=False):
    """
    Check a program text against the static profile. Returns a StaticReport;
    ``report.ok`` is True when the program is inside the profile.

        text           the program
        name           its name in the diagnostics; also where ``.l("file")`` is looked up
        int_bits       32 (the default) or 64: the width of kg_int, for the int-range rule
        globals        {name: value or Ty}: the values of compile-time definitions, when the
                       caller has evaluated them (the driver of ticket 091 does). A value is
                       validated as a global (validate_global) and its type and shape are then
                       known to every rule. Without it, only literal definitions are known.
        base_dir       directory for ``.l`` lookups, instead of the directory of name
        resolve        False: do not read the files named by ``.l``
        all_functions  True: also check the functions no entry reaches, as roots with
                       arguments of unknown type (they are otherwise host-only and unchecked)

    Nothing in the program is evaluated and no other file is run.
    """
    if int_bits not in INT_RANGE:
        raise ValueError(f"int_bits must be 32 or 64, not {int_bits!r}")
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(name)) if os.path.isfile(name) else os.getcwd()
    source = text if isinstance(text, Source) else Source(name, text)
    return _Checker(source, base_dir, int_bits, globals, resolve, all_functions).run()


def check_file(path, **options):
    """Check a file. OSError and UnicodeDecodeError propagate."""
    with open(path, "r") as f:
        text = f.read()
    return check_source(text, path, **options)


def check_main(files, int_bits=32, verbose=False, stdout=None, stderr=None, color=False, streamable=False):
    """
    The body of ``kgpy --check-static FILE...``. Every problem of every file is
    printed to stderr, rendered like any KlongPy error; a file inside the
    profile gets one line on stdout. Returns the exit status: 1 if any file
    is outside the profile, 2 if a file could not be read (and none was
    outside), else 0. With verbose, the functions, globals and declarations
    of each file are listed. With streamable, a file inside the profile is
    also searched for what keeps a hardware kernel from taking one record
    per clock cycle (klongpy/static/streamable.py); each finding is printed
    like a profile violation, with the rule id of the streamable tier.
    """
    import sys
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    bad = unreadable = total = 0
    for path in files:
        try:
            if path == "-":
                report = check_source(sys.stdin.read(), "<stdin>", int_bits=int_bits)
            else:
                report = check_file(path, int_bits=int_bits)
        except (OSError, UnicodeDecodeError) as e:
            print(f"{path}: cannot read: {getattr(e, 'strerror', None) or e}", file=err)
            unreadable += 1
            continue
        if report.diagnostics:
            bad += 1
            total += len(report.diagnostics)
            for d in report.diagnostics:
                print(d.render(color), file=err)
        else:
            note = "" if report.entries else "; no .entry, so every function was checked with arguments of unknown type"
            if report.unresolved:
                n = len(report.unresolved)
                note += (f"; {_plural(n, 'global')} ({', '.join(report.unresolved[:4])}{', ...' if n > 4 else ''}) "
                         f"{'is' if n == 1 else 'are'} computed on the host: checked for form only until the program is compiled")
            print(f"{path}: inside the static profile ({report.summary()}){note}", file=out)
            if streamable and path != "-":
                from ..errors import KlongError
                from .streamable import check_streamable
                try:
                    found = check_streamable(path, int_bits=int_bits)
                except KlongError as e:
                    for d in getattr(e, "diagnostics", None) or [e]:
                        print(d.render(color), file=err)
                    bad += 1
                    total += 1
                    found = []
                if found:
                    bad += 1
                    total += len(found)
                    for d in found:
                        print(d.render(color), file=err)
                else:
                    print(f"{path}: streamable: nothing keeps a kernel from taking one record per clock cycle", file=out)
        if verbose:
            _describe(report, out)
    if total:
        print(f"{_plural(total, 'static error')} in {_plural(bad, 'file')}", file=err)
    if bad:
        return 1
    return 2 if unreadable else 0


def _describe(report, out):
    for e in report.entries:
        print(f"  entry   {e.name}: {e.signature}", file=out)
    for x in report.externs.values():
        print(f"  extern  {x.name}: {x.signature} (module {x.module})", file=out)
    for g in report.globals.values():
        kind = "state" if g.assigned else "const"
        ty = g.ty if g.ty is not None else "computed on the host; type not known without evaluating"
        print(f"  global  {g.name}: {ty} ({kind})", file=out)
    for f in report.functions.values():
        if not f.reachable:
            print(f"  fn      {f.name}: host only (no entry reaches it)", file=out)
            continue
        for args, result in f.specializations.items():
            shown = "; ".join(str(a) if a is not None else "?" for a in args)
            print(f"  fn      {f.name}({shown}) -> {result if result is not None else '?'}", file=out)
