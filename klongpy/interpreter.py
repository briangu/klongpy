import os
import time
from collections import deque

from .adverbs import get_adverb_fn
from .backends import get_backend
from .core import *
from .dyads import create_dyad_functions
from .monads import create_monad_functions
from .sys_fn import create_system_functions
from .sys_fn_autograd import create_system_functions_autograd
from .sys_fn_ipc import create_system_functions_ipc, create_system_var_ipc
from .sys_fn_timer import create_system_functions_timer
from .sys_var import *
from .types import KGQSym
from .errors import (KlongError, KlongInternalError, KlongNameError, KlongRangeError, KlongSyntaxError,
                     adverb_name, classify, compat, is_passthrough, operator_error, operator_name,
                     python_error, system_error)
from .lexer import Lexer, Source
from .parser import (SEMI, ProgResult, tk_check, tk_comment_end, tk_peek_adverb, tk_raise, tk_read_cond,
                     tk_read_data, tk_read_expr_array)
from .utils import ReadonlyDict
from .compiler import compile_guarded, run_compiled, NOT_COMPILED, LRUCache


def set_context_var(d, sym, v):
    """
    Sets a context variable, wrapping Python lambda/functions as appropriate.
    """
    assert isinstance(sym, KGSym)
    if callable(v) and not issubclass(type(v), KGLambda) :
        x = KGLambda(v)
        # A function value in call form with no arguments yet (args=None).
        v = KGCall(x,args=None,arity=x.get_arity())
    d[sym] = v


class _Value:
    """
    Marks an already-evaluated argument while it is merged with projection
    arguments, which are stored as unevaluated expressions (see apply()).
    """
    __slots__ = ('v',)

    def __init__(self, v):
        self.v = v


class KGModule(dict):
    """
    A module class that is used for optimizing when to scan for namespaced keys.
    """
    def __init__(self, name=None):
        self.name = name
        super().__init__()


class KlongContext():
    """

    Maintains current symbol and module context for the interpeter.

    TODO: Support 'it' system variable

        it                                                          [It]

    This variable holds the result of the most recent successful
    computation, so you do not have to re-type or copy-paste the
    previous result. E.g.:

            {(x+2%x)%2}:~2
    1.41421356237309504
            it^2
    1.99999999999999997

    """

    def __init__(self, system_contexts, strict_mode=1):
        self._context = deque([{}, *system_contexts])
        self._min_ctx_count = len(system_contexts)
        self._strict_mode = strict_mode

    def start_module(self, name):
        self.push(KGModule(name))
        self._min_ctx_count = len(self._context)

    def stop_module(self):
        self.push({})

    def __setitem__(self, k, v):
        assert isinstance(k, KGSym)

        if k not in reserved_fn_symbols:
            # Check if variable exists in any scope
            for d in self._context:
                if in_map(k, d):
                    d[k] = v
                    return k

        # Variable doesn't exist - check strict mode
        if self._strict_mode >= 1:
            # Check if we're inside a function (more than just global scope)
            in_function = len(self._context) > self._min_ctx_count + 1

            if in_function:
                # Inside function - disallow creating new variables
                raise KlongNameError(
                    f"undefined variable: {k}",
                    hint=f"declare it as a local in the parameter list, {{[{k}]; ...}}, "
                         f"or define the global before the function is called")

        # Create new variable in current scope
        set_context_var(self._context[0], k, v)
        return k

    _MISSING = object()

    def __getitem__(self, k):
        assert isinstance(k, KGSym)
        for d in self._context:
            v = d.get(k, self._MISSING)
            if v is not self._MISSING:
                return v
            if isinstance(d,KGModule):
                if  '`' in k:
                    p = k.split('`')
                    if KGSym(p[1]) == d.name:
                        k = KGSym(p[0])
                else:
                    tk = k + '`'
                    for dk in d.keys():
                        if dk.startswith(tk):
                            return d[dk]
        raise KeyError(k)

    def __delitem__(self, k):
        assert isinstance(k, KGSym)
        for d in self._context:
            if in_map(k, d) and not isinstance(d,ReadonlyDict):
                del d[k]
                return
        raise KeyError(k)

    def push(self, d):
        self._context.appendleft(d)

    def pop(self):
        return self._context.popleft() if len(self._context) > self._min_ctx_count else None

    def is_defined_sym(self, k):
        if isinstance(k, KGSym):
            for d in self._context:
                if in_map(k, d):
                    return True
                if isinstance(d,KGModule) and '`' not in k:
                    tk = k + '`'
                    for dk in d.keys():
                        if dk.startswith(tk):
                            return True
        return False

    def __iter__(self):
        seen = set()
        for d in self._context:
            for x in d.items():
                if x[0] not in seen:
                    yield x
                    seen.add(x[0])


def add_context_key_values(d, context):
    for k,fn in context.items():
        set_context_var(d, KGSym(k), fn)


def create_system_contexts():
    cin = eval_sys_var_cin()
    cout = eval_sys_var_cout()
    cerr = eval_sys_var_cerr()

    sys_d = {}
    add_context_key_values(sys_d, create_system_functions())
    add_context_key_values(sys_d, create_system_functions_autograd())
    add_context_key_values(sys_d, create_system_functions_ipc())
    add_context_key_values(sys_d, create_system_functions_timer())
    set_context_var(sys_d, KGSym('.e'), eval_sys_var_epsilon()) # TODO: support lambda
    set_context_var(sys_d, KGSym('.cin'), cin)
    set_context_var(sys_d, KGSym('.cout'), cout)
    set_context_var(sys_d, KGSym('.cerr'), cerr)

    sys_var = {}
    add_context_key_values(sys_var, create_system_var_ipc())
    set_context_var(sys_var, KGSym('.sys.cin'), cin)
    set_context_var(sys_var, KGSym('.sys.cout'), cout)
    set_context_var(sys_var, KGSym('.sys.cerr'), cerr)

    return [sys_var, ReadonlyDict(sys_d)]


def chain_adverbs(klong, arr):
    """

        Multiple Adverbs

        Multiple adverbs can be attached to a verb. In this case, the
        first adverb modifies the verb, giving a new verb, and the next
        adverb modifies the new verb. Note that subsequent adverbs must
        be adverbs of monadic verbs, because the first verb-adverb
        combination in a chain of adverbs forms a monad. So ,/' (Join-
        Over-Each) would work, but ,/:' (Join-Over-Each-Pair) would not,
        because :' expects a dyadic verb.

        Examples:

        +/' (Plus-Over-Each) would apply Plus-Over to each member of a
        list:

        +/'[[1 2 3] [4 5 6] [7 8 9]]  -->  [6 15 24]

        ,/:~ (Flatten-Over-Converging) would apply Flatten-Over until a
        fixpoint is reached:

        ,/:~[1 [2 [3 [4] 5] 6] 7]  -->  [1 2 3 4 5 6 7]

        ,/\~ (Flatten-Over-Scan-Converging) explains why ,/:~ flattens
        any object:

        ,/\~[1 [2 [3 [4] 5] 6] 7]  -->  [[1 [2 [3 [4] 5] 6] 7]
                                          [1 2 [3 [4] 5] 6 7]
                                           [1 2 3 [4] 5 6 7]
                                            [1 2 3 4 5 6 7]]

    """
    f = chain_fn(klong, arr)
    if arr[-2].arity == 1:
        return lambda a=arr[-1],f=f,k=klong: f(k.eval(a))
    return lambda a=arr[-1],f=f,k=klong: f(k.eval(a[0]),k.eval(a[1]))


def chain_fn(klong, arr):
    """
    The function a chain of adverbs makes of its verb: it takes the values of
    the operand(s), which are the last element of arr and are evaluated by
    the caller (KlongInterpreter.eval, or the thunk chain_adverbs returns).
    """
    # The verb receives values the adverb has already produced, so it is
    # applied to them directly and they are never evaluated a second time.
    a = arr[0].a
    f = None
    if isinstance(a, KGOp):
        f = (klong._vm if arr[0].arity == 1 else klong._vd).get(a.a)
    if f is None:
        if arr[0].arity == 1:
            f = lambda x,k=klong,a=a: k.apply(a, [x])
        else:
            f = lambda x,y,k=klong,a=a: k.apply(a, [x,y])
    for i in range(1,len(arr)-1):
        o = get_adverb_fn(klong, arr[i].a, arity=arr[i].arity)
        if arr[i].arity == 1:
            f = lambda x,f=f,o=o: o(f,x,op=arr[0].a)
        else:
            f = lambda x,y,f=f,o=o: o(f,x,y)
    return f


class KlongInterpreter():

    def __init__(self, backend=None, device=None, parser=None):
        """
        Initialize a Klong interpreter.

        Parameters
        ----------
        backend : str, optional
            Backend name ('numpy' or 'torch'). Defaults to 'numpy'.
        device : str, optional
            Device for torch backend ('cpu', 'cuda', 'mps'). Only applies
            when backend='torch'. If None, auto-selects best available device.
        parser : str, optional
            'token' (the default) reads programs through klongpy.lexer and
            records source spans. 'legacy' selects the previous
            character-cursor parser; it is deprecated and will be removed
            after one release. When not given, the environment variable
            KLONGPY_PARSER is consulted.
        """
        parser = parser or os.environ.get('KLONGPY_PARSER') or 'token'
        if parser not in ('token', 'legacy'):
            raise ValueError(f"unknown parser: {parser!r} (expected 'token' or 'legacy')")
        self._parser = parser
        self._backend = get_backend(backend, device=device)
        strict_mode = 0  # 0=unsafe (default for backward compat), 1=strict, 2=pedantic
        self._context = KlongContext(create_system_contexts(), strict_mode=strict_mode)
        self._vd = create_dyad_functions(self)
        self._vm = create_monad_functions(self)
        self._start_time = time.time()
        self._module = None
        self._parse_cache = LRUCache()
        self._compiled_cache = LRUCache()
        # The Source of each program being run, innermost last: pushed by
        # __call__ and exec, and so by .l, .E and file runs, which go through
        # them. See current_source().
        self._sources = []

    @property
    def backend(self):
        """Return the backend provider for this interpreter."""
        return self._backend

    @property
    def np(self):
        """Return the numpy-compatible array module for this interpreter."""
        return self._backend.np

    def __setitem__(self, k, v):
        k = k if isinstance(k, KGSym) else KGSym(k)
        self._context[k] = v
        # Compiled expressions assume operand types (tensor vs list vs scalar).
        # A type change (e.g., tensor -> list) would silently produce wrong
        # results since Python operators have different semantics per type.
        self._compiled_cache.clear()

    def __getitem__(self, k):
        k = k if isinstance(k, KGSym) else KGSym(k)
        r = self._context[k]
        # Pass the symbol name to avoid O(n) context search
        return KGFnWrapper(self, r, sym=k) if issubclass(type(r), KGFn) else r

    def __delitem__(self, k):
        k = k if isinstance(k, KGSym) else KGSym(k)
        del self._context[k]
        self._compiled_cache.clear()

    def _get_op_fn(self, s, arity):
        try:
            return self._vm[s] if arity == 1 else self._vd[s]
        except KeyError:
            # (ticket 042 makes this a parse-time error)
            raise compat(KlongSyntaxError, KeyError)(f"unknown operator {s}") from None

    def current_source(self):
        """The klongpy.lexer.Source of the program being run (the innermost one), or None."""
        return self._sources[-1] if self._sources else None

    def _is_monad(self, s):
        return isinstance(s,KGOp) and in_map(s.a, self._vm)

    def _is_dyad(self, s):
        return isinstance(s,KGOp) and in_map(s.a, self._vd)

    def start_module(self, name):
        self._context.start_module(name)

    def stop_module(self):
        self._context.stop_module()

    def parse_module(self, name):
        if safe_eq(name,0) or is_empty(name):
            self._module = None
        else:
            # Strip any module qualification that read_sym may have added
            # (e.g. m2`m1 -> m2 when switching from m1 to m2)
            s = str(name)
            if '`' in s:
                name = KGSym(s.split('`')[0])
            self._module = name

    def current_module(self):
        return self._module

    def process_start_time(self):
        return self._start_time

    # ------------------------------------------------------------------
    # Grammar on tokens (the default parser).
    #
    # These methods are the legacy grammar methods (kept below as _legacy_*)
    # rewritten to read tokens from a klongpy.lexer.Lexer instead of
    # characters from (t, i), decision for decision: every quirk of the legacy
    # parser is reproduced, with one deliberate exception (ticket 023): a
    # string or char literal is never mistaken for punctuation, because a
    # token knows its kind. See the notes on positions and newlines above the
    # tk_* readers in parser.py.
    #
    # Each method leaves lx.start at the start offset of the expression it
    # returned; its end is the end of the last token consumed. Source spans
    # are recorded on the nodes from these (see KGFn) and change nothing else.
    # ------------------------------------------------------------------

    def _apply_adverbs(self, lx, a, aa, arity, st, ost, dyad=False, dyad_value=None, lsp=None):
        """
        The read position is just after the first adverb aa of the verb a.
        st is the start offset of the whole expression, ost of the verb, lsp
        the span of the left operand of a dyadic chain.
        """
        aa_arity = get_adverb_arity(aa, arity)
        if isinstance(a,KGOp):
            a.arity = aa_arity
        arr = [KGAdverb(a, aa_arity), KGAdverb(aa, arity)]
        aa = tk_peek_adverb(lx)
        while aa is not None:
            arr.append(KGAdverb(aa, 1))
            aa = tk_peek_adverb(lx)
        toks = lx.toks
        oe = toks[lx.pos-1][2]
        # legacy: the operand is read without ignore_newline, wherever we are
        r = self._expr(lx)
        if r is SEMI:
            r = ';'
        arr.append([dyad_value,r] if dyad else r)
        # Parser-built: the operand expressions are the last element of arr
        # and are evaluated by chain_adverbs.
        c = KGCall(arr,args=None,arity=2 if dyad else 1)
        e = toks[lx.pos-1][2]
        rsp = None if r is None else (lx.start, e)
        c.span = (st, e)
        c.op_span = (ost, oe)
        c.arg_spans = (lsp, rsp) if dyad else (rsp,)
        c.source = lx.source
        return c

    def _read_fn_args(self, lx):
        """

        # Arguments are delimited by parentheses and separated by
        # semicolons. There are up to three arguments.

        a := '(' ')'
           | '(' e ')'
           | '(' e ';' e ')'
           | '(' e ':' e ';' e ')'

        # Projected argument lists are like argument lists (a), but at
        # least one argument must be omitted.

        P := '(' ';' e ')'
           | '(' e ';' ')'
           | '(' ';' e ';' e ')'
           | '(' e ';' ';' e ')'
           | '(' e ';' e ';' ')'
           | '(' ';' ';' e ')'
           | '(' ';' e ';' ')'
           | '(' e ';' ';' ')'

        The token at the read position is '(' or ':('. Returns the arguments
        (None for an omitted one) and their spans.

        """
        toks = lx.toks
        j = lx.pos + 1
        lx.pos = j
        arr = []
        spans = []
        tok = toks[j]
        if tok[0] == 'RPAREN' and not tok[4]: # nilad application
            lx.pos = j + 1
            return arr, spans
        # legacy: an argument is omitted only where a ';' or the ')' directly
        # follows the previous separator, with nothing in between.
        sep = j
        while True:
            j = lx.pos
            tok = toks[j]
            k = tok[0]
            while k == 'NEWLINE' and not tok[5]:
                j += 1
                tok = toks[j]
                k = tok[0]
            if k == 'SEMI' or k == 'NEWLINE':
                if j == sep and not tok[4]:
                    arr.append(None)
                    spans.append(None)
                lx.pos = sep = j + 1
                continue
            elif k == 'RPAREN':
                if j == sep and not tok[4]:
                    arr.append(None)
                    spans.append(None)
                break
            if k == 'LBRACK':
                # legacy: the lexeme is read here as a data list before the
                # expression reader sees that it is an evaluated list
                nt = toks[j+1]
                if nt[0] == 'SEMI' and not nt[4]:
                    tk_check(self, lx, j)
            a = self._expr(lx, True)
            if a is None:
                break
            arr.append(';' if a is SEMI else a)     # SEMI only from "(;)"
            spans.append((lx.start, toks[lx.pos-1][2]))
        # legacy: the ')' must follow the last argument directly
        tok = toks[lx.pos]
        if tok[0] != 'RPAREN' or tok[4]:
            raise UnexpectedChar(lx.text, lx.offset(), ')')
        lx.pos += 1
        return arr, spans

    def _read_fn(self, lx, st):
        """
        Read a function literal and the argument list that may follow it. The
        read position is just after the '{' whose offset is st.
        """
        toks = lx.toks
        body, bsp = self._prog(lx, True)
        a = body[0] if len(body) == 1 else body
        j = lx.pos
        tok = toks[j]
        while tok[0] == 'NEWLINE' and not tok[5]:
            j += 1
            tok = toks[j]
        if tok[0] != 'RBRACE':
            raise UnexpectedChar(lx.text, tok[1], '}')
        lx.pos = j = j + 1
        arity = get_fn_arity(a)
        nt = toks[j]
        nk = nt[0]
        if (nk == 'LPAREN' or nk == 'CALLP') and not nt[4]:
            fa, fsp = self._read_fn_args(lx)
            # Parser-built calls (here and below): fa holds unevaluated
            # argument expressions, which _eval_fn evaluates.
            f = KGFn(a, fa, arity) if has_none(fa) else KGCall(a, fa, arity)
            f.op_span = (st, tok[2])
            f.arg_spans = tuple(fsp)
        else:
            f = KGFn(a, args=None, arity=arity)
        f.span = (st, toks[lx.pos-1][2])
        f.body_spans = tuple(bsp)
        f.source = lx.source
        return f

    def _factor(self, lx, ignore_newline=False):
        """

        # A factor is a lexeme class (C) or a variable (V) applied to
        # arguments (a) or a function (f) or a function applied to
        # arguments or a monadic operator (m) applied to an expression
        # or a parenthesized expression or a conditional expression (c)
        # or a list (L) or a dictionary (D).

        x := C
           | V a
           | f
           | f a
           | m e
           | '(' e ')'
           | c
           | L
           | D

        # A function is a program delimited by braces. Deja vu? A
        # function may be projected onto on some of its arguments,
        # giving a projection. A variable can also be used to form
        # a projection.

        f := '{' p '}'
           | '{' p '}' P
           | V P

        Returns None at end of input and SEMI for a ';' (or a newline that
        acts as one). Any other punctuation that cannot start a factor is
        returned as the string it is written as, as the legacy parser did.

       """
        toks = lx.toks
        j = lx.pos
        tok = toks[j]
        k = tok[0]
        if ignore_newline:
            while k == 'NEWLINE' and not tok[5]:
                j += 1
                tok = toks[j]
                k = tok[0]
        st = tok[1]
        if k == 'NAME' or k == 'QNAME':
            v = tok[3]
            module = self._module
            a = reserved_fn_symbol_map.get(v) or KGSym(v if module is None or v[0] == '.' else f"{v}`{module}")
            if k == 'QNAME':
                # a quoted symbol in expression position is a KGQSym node
                a = KGQSym(a)
            lx.pos = j = j + 1
            nt = toks[j]
            nk = nt[0]
            if (nk == 'LPAREN' or nk == 'CALLP') and not nt[4]:
                fa, fsp = self._read_fn_args(lx)
                a = KGFn(a, fa, arity=len(fa)) if has_none(fa) else KGCall(a, fa, arity=len(fa))
                if v == '.comment':
                    # .comment("END") hides the raw text up to the marker
                    lx.seek(tk_comment_end(lx, st, a.args[0]))
                    return self._factor(lx, ignore_newline)
                elif v == '.module':
                    self.parse_module(fa[0])
                a.span = (st, toks[lx.pos-1][2])
                a.op_span = (st, tok[2])
                a.arg_spans = tuple(fsp)
                a.source = lx.source
            aa = tk_peek_adverb(lx)
            if aa is not None:
                a = self._apply_adverbs(lx, a, aa, 1, st, st)
        elif k == 'NUM' or k == 'STR' or k == 'CHAR':
            lx.pos = j + 1
            a = tok[3]
        elif k == 'OP' or k == 'COP' or k == 'CALLP':
            v = tok[3]
            a = KGOp(v, 0)
            lx.pos = j + 1
            if v in self._vm:
                a.arity = 1
                aa = tk_peek_adverb(lx)
                if aa is not None:
                    a = self._apply_adverbs(lx, a, aa, 1, st, st)
                else:
                    r = self._expr(lx, ignore_newline)
                    if r is SEMI:
                        r = ';'
                    a = KGFn(a, r, arity=1)
                    e = toks[lx.pos-1][2]
                    a.span = (st, e)
                    a.op_span = (st, tok[2])
                    a.arg_spans = (None if r is None else (lx.start, e),)
                    a.source = lx.source
                    if r is None:
                        a.missing_operand = True
        elif k == 'LBRACK':
            nt = toks[j+1]
            if nt[0] == 'SEMI' and not nt[4]:
                # evaluated array constructor [;expr1;expr2;...]
                lx.pos = j + 2
                a = tk_read_expr_array(self, lx, st)
            else:
                a = self._backend.kg_asarray(tk_read_data(self, lx, j, True))
        elif k == 'LPAREN':
            lx.pos = j + 1
            # legacy: "(;)" and "(" newline ")" give the ';' itself, which the
            # callers then treat as a separator: a is passed on as SEMI.
            a = self._expr(lx, ignore_newline)
            tok = toks[lx.pos]
            if tok[0] != 'RPAREN' or tok[4]:
                raise UnexpectedChar(lx.text, lx.offset(), ')')
            lx.pos += 1
        elif k == 'LBRACE': # read fn
            lx.pos = j + 1
            a = self._read_fn(lx, st)
            aa = tk_peek_adverb(lx)
            if aa is not None:
                a = self._apply_adverbs(lx, a, aa, 1, st, st)
        elif k == 'SEMI' or k == 'NEWLINE':
            lx.pos = j + 1
            a = SEMI
        elif k == 'EOF':
            lx.pos = j
            lx.at_end = True
            return None
        elif k == 'COND':
            lx.pos = j + 1
            a = tk_read_cond(self, lx, st)
        elif k == 'DICT':
            a = tk_read_data(self, lx, j, True)
        elif k == 'ERROR':
            tk_raise(lx, tok)
        else:
            # ) } ] :| where a factor was expected
            lx.pos = j + 1
            a = tok[3]
        lx.start = st
        return a

    def _expr(self, lx, ignore_newline=False):
        """

        # An expression is a factor or a dyadic operation applied to
        # a factor and an expression. I.e. dyads associate to the right.

        e := x
           | x d e

        """
        a = self._factor(lx, ignore_newline)
        if a is None or a is SEMI:
            return a
        st = lx.start
        toks = lx.toks
        first = True
        while True:
            j = lx.pos
            lend = toks[j-1][2]
            tok = toks[j]
            k = tok[0]
            if ignore_newline:
                while k == 'NEWLINE' and not tok[5]:
                    j += 1
                    tok = toks[j]
                    k = tok[0]
            if k == 'OP' or k == 'COP' or k == 'CALLP':
                aa = KGOp(tok[3], 0)
                # legacy: only the first operator read here has its arity set
                if first and tok[3] in self._vd:
                    aa.arity = 2
                lx.pos = j + 1
            elif k == 'NAME' or k == 'QNAME':
                v = tok[3]
                module = self._module
                aa = reserved_fn_symbol_map.get(v) or KGSym(v if module is None or v[0] == '.' else f"{v}`{module}")
                lx.pos = j + 1
                nt = toks[j+1]
                nk = nt[0]
                if (nk == 'LPAREN' or nk == 'CALLP') and not nt[4]:
                    fa, fsp = self._read_fn_args(lx)
                    aa = KGFn(aa, fa, arity=len(fa)) if has_none(fa) else KGCall(aa, fa, arity=len(fa))
                    aa.span = (tok[1], toks[lx.pos-1][2])
                    aa.op_span = (tok[1], tok[2])
                    aa.arg_spans = tuple(fsp)
                    aa.source = lx.source
            elif k == 'LBRACE': # read fn
                lx.pos = j + 1
                aa = self._read_fn(lx, tok[1])
            else:
                if k == 'LBRACK' or k == 'DICT' or k == 'ERROR':
                    # legacy: the lexeme was read to look at it
                    tk_check(self, lx, j)
                break
            first = False
            ost = tok[1]
            adv = tk_peek_adverb(lx)
            if adv is not None:
                a = self._apply_adverbs(lx, aa, adv, 2, st, ost, dyad=True, dyad_value=a, lsp=(st, lend))
            else:
                osp = (ost, toks[lx.pos-1][2])
                r = self._expr(lx, ignore_newline)
                if r is SEMI:
                    r = ';'
                a = KGFn(aa, [a, r], arity=2)
                e = toks[lx.pos-1][2]
                a.span = (st, e)
                a.op_span = osp
                a.arg_spans = ((st, lend), None if r is None else (lx.start, e))
                a.source = lx.source
                if r is None:
                    a.missing_operand = True
        if ignore_newline and isinstance(a, str) and a == '\n':
            # legacy: an expression that is just the newline character (0c
            # followed by a newline, or a string holding one newline) was taken
            # for a newline lexeme, and the whitespace and newlines after it
            # were skipped here. What follows is then read as if it were
            # adjacent, and a newline after a comment that stopped the skip no
            # longer acts as ';'. Restarting the lexer at the token the skip
            # stopped at gives exactly that state.
            j = lx.pos
            tok = toks[j]
            while tok[0] == 'NEWLINE' and not tok[5]:
                j += 1
                tok = toks[j]
            lx.pos = j
            lx.seek(tok[1])
        lx.start = st
        return a

    def _prog(self, lx, ignore_newline=False):
        """

        # A program is a ';'-separated sequence of expressions.

        p := e
           | e ';' p

        Returns the statements and one span per statement.

        """
        arr = []
        spans = []
        toks = lx.toks
        while True:
            q = self._expr(lx, ignore_newline)
            if q is None:
                break
            if q is SEMI:
                continue
            arr.append(q)
            j = lx.pos
            spans.append((lx.start, toks[j-1][2]))
            tok = toks[j]
            k = tok[0]
            if ignore_newline:
                while k == 'NEWLINE' and not tok[5]:
                    j += 1
                    tok = toks[j]
                    k = tok[0]
            if k == 'SEMI' or k == 'NEWLINE':
                lx.pos = j + 1
            else:
                if k == 'LBRACK' or k == 'DICT' or k == 'ERROR':
                    # legacy: the lexeme was read to look at it
                    tk_check(self, lx, j)
                # legacy: whatever follows a complete statement is dropped
                break
        return arr, spans

    def prog(self, t, i=0, ignore_newline=False):
        """

        Parse a Klong expression string into a Klong program, which can then be evaluated.

        # A program is a ';'-separated sequence of expressions.

        p := e
           | e ';' p

        t is the program text or a klongpy.lexer.Source (a text with a name).
        Returns the pair (offset where parsing stopped, list of statements).
        The pair also has two attributes: stmt_spans, one (start, end) span
        per statement, and source, the Source the spans refer to.

        """
        if self._parser == 'legacy':
            text = t.text if isinstance(t, Source) else t
            try:
                return ProgResult(self._legacy_prog(text, i, ignore_newline=ignore_newline))
            except KlongError as e:
                if e.source is None and e.span is not None:
                    e.source = t if isinstance(t, Source) else Source("<string>", text)
                raise
        lx = Lexer(t, i)
        try:
            arr, spans = self._prog(lx, ignore_newline)
        except KlongError as e:
            # a syntax error knows its span (offsets into this text)
            if e.source is None and e.span is not None:
                e.source = lx.source
            raise
        r = ProgResult((lx.offset(), arr))
        r.stmt_spans = spans
        r.source = lx.source
        return r

    # ------------------------------------------------------------------
    # Legacy grammar on the source string (parser="legacy"). Deprecated;
    # kept unchanged for one release so the two parsers can be compared.
    # ------------------------------------------------------------------

    def _legacy_apply_adverbs(self, t, i, a, aa, arity, dyad=False, dyad_value=None):
        aa_arity = get_adverb_arity(aa, arity)
        if isinstance(a,KGOp):
            a.arity = aa_arity
        a = KGAdverb(a, aa_arity)
        arr = [a, KGAdverb(aa, arity)]
        ii,aa = peek_adverb(t, i)
        while aa is not None:
            arr.append(KGAdverb(aa, 1))
            i = ii
            ii,aa = peek_adverb(t, i)
        i, aa = self._legacy_expr(t, i)
        arr.append([dyad_value,aa] if dyad else aa)
        # Parser-built: the operand expressions are the last element of arr
        # and are evaluated by chain_adverbs.
        return i,KGCall(arr,args=None,arity=2 if dyad else 1)

    def _legacy_read_fn_args(self, t, i=0):
        """

        # Arguments are delimited by parentheses and separated by
        # semicolons. There are up to three arguments.

        a := '(' ')'
           | '(' e ')'
           | '(' e ';' e ')'
           | '(' e ':' e ';' e ')'

        # Projected argument lists are like argument lists (a), but at
        # least one argument must be omitted.

        P := '(' ';' e ')'
           | '(' e ';' ')'
           | '(' ';' e ';' e ')'
           | '(' e ';' ';' e ')'
           | '(' e ';' e ';' ')'
           | '(' ';' ';' e ')'
           | '(' ';' e ';' ')'
           | '(' e ';' ';' ')'

        """
        if cmatch(t,i,'('):
            i += 1
        elif cmatch2(t,i,':', '('):
            i += 2
        else:
            raise UnexpectedChar(t,i,t[i])
        arr = []
        if cmatch(t, i, ')'): # nilad application
            return i+1,arr
        k = i
        while True:
            ii,c = kg_read(t, i, ignore_newline=True, module=self.current_module())
            if safe_eq(c, ';'):
                i = ii
                if k == i - 1:
                    arr.append(None)
                k = i
                continue
            elif safe_eq(c,')'):
                if k == ii - 1:
                    arr.append(None)
                break
            i,a = self._legacy_expr(t,i,ignore_newline=True)
            if a is None:
                break
            arr.append(a)
        i = cexpect(t,i,')')
        return i,arr


    def _legacy_factor(self, t, i=0, ignore_newline=False):
        """

        # A factor is a lexeme class (C) or a variable (V) applied to
        # arguments (a) or a function (f) or a function applied to
        # arguments or a monadic operator (m) applied to an expression
        # or a parenthesized expression or a conditional expression (c)
        # or a list (L) or a dictionary (D).

        x := C
           | V a
           | f
           | f a
           | m e
           | '(' e ')'
           | c
           | L
           | D

        # A function is a program delimited by braces. Deja vu? A
        # function may be projected onto on some of its arguments,
        # giving a projection. A variable can also be used to form
        # a projection.

        f := '{' p '}'
           | '{' p '}' P
           | V P

       """
        # Check for evaluated array constructor [;expr1;expr2;...]
        ii = skip(t, i, ignore_newline=ignore_newline)
        if cmatch2(t, ii, '[', ';'):
            i, exprs = read_expr_array(self, t, ii + 2)
            return i, KGExprArray(exprs)

        # qsym=True: a quoted symbol in expression position is a KGQSym node.
        i,a = kg_read_array(t, i, self._backend, ignore_newline=ignore_newline, module=self.current_module(), qsym=True)
        if a is None:
            return i,a
        if safe_eq(a, '{'): # read fn
            i,a = self._legacy_prog(t, i, ignore_newline=True)
            a = a[0] if len(a) == 1 else a
            i = skip(t, i, ignore_newline=True)
            i = cexpect(t, i, '}')
            arity = get_fn_arity(a)
            if cmatch(t, i, '(') or cmatch2(t,i,':','('):
                i,fa = self._legacy_read_fn_args(t,i)
                # Parser-built calls (here and below): fa holds unevaluated
                # argument expressions, which _eval_fn evaluates.
                a = KGFn(a, fa, arity) if has_none(fa) else KGCall(a, fa, arity)
            else:
                a = KGFn(a, args=None, arity=arity)
            ii, aa = peek_adverb(t, i)
            if aa:
                i,a = self._legacy_apply_adverbs(t, ii, a, aa, arity=1)
        elif isinstance(a, KGSym):
            if cmatch(t,i,'(') or cmatch2(t,i,':','('):
                i,fa = self._legacy_read_fn_args(t,i)
                a = KGFn(a, fa, arity=len(fa)) if has_none(fa) else KGCall(a, fa, arity=len(fa))
                if safe_eq(a.a, KGSym(".comment")):
                    i = read_sys_comment(t,i,a.args[0])
                    return self._legacy_factor(t,i, ignore_newline=ignore_newline)
                elif safe_eq(a.a, KGSym('.module')):
                    self.parse_module(fa[0])
            ii, aa = peek_adverb(t, i)
            if aa:
                i,a = self._legacy_apply_adverbs(t, ii, a, aa, arity=1)
        elif self._is_monad(a):
            a.arity = 1
            ii, aa = peek_adverb(t, i)
            if aa:
                i,a = self._legacy_apply_adverbs(t, ii, a, aa, arity=1)
            else:
                i, aa = self._legacy_expr(t, i, ignore_newline=ignore_newline)
                a = KGFn(a, aa, arity=1)
                if aa is None:
                    a.missing_operand = True
        elif safe_eq(a, '('):
            i,a = self._legacy_expr(t, i, ignore_newline=ignore_newline)
            i = cexpect(t, i, ')')
        elif safe_eq(a, ':['):
            return read_cond(self, t, i)
        return i, a

    def _legacy_expr(self, t, i=0, ignore_newline=False):
        """

        # An expression is a factor or a dyadic operation applied to
        # a factor and an expression. I.e. dyads associate to the right.

        e := x
           | x d e

        """
        i, a = self._legacy_factor(t, i, ignore_newline=ignore_newline)
        if a is None or safe_eq(a, ';'):
            return i,a
        ii, aa = kg_read(t, i, ignore_newline=ignore_newline, module=self.current_module())
        if self._is_dyad(aa):
            aa.arity = 2
        while isinstance(aa,(KGOp,KGSym)) or safe_eq(aa, '{'):
            i = ii
            if safe_eq(aa, '{'): # read fn
                i,aa = self._legacy_prog(t, i, ignore_newline=True)
                aa = aa[0] if len(aa) == 1 else aa
                i = skip(t, i, ignore_newline=True)
                i = cexpect(t, i, '}')
                arity = get_fn_arity(aa)
                if cmatch(t, i, '(') or cmatch2(t,i,':','('):
                    i,fa = self._legacy_read_fn_args(t,i)
                    aa = KGFn(aa, fa, arity=arity) if has_none(fa) else KGCall(aa, fa, arity=arity)
                else:
                    aa = KGFn(aa, args=None, arity=arity)
            elif isinstance(aa,KGSym) and (cmatch(t, i, '(') or cmatch2(t,i,':','(')):
                i,fa = self._legacy_read_fn_args(t,i)
                aa = KGFn(aa, fa, arity=len(fa)) if has_none(fa) else KGCall(aa, fa, arity=len(fa))
            ii, aaa = peek_adverb(t, i)
            if aaa:
                i,a = self._legacy_apply_adverbs(t, ii, aa, aaa, arity=2, dyad=True, dyad_value=a)
            else:
                i, aaa = self._legacy_expr(t, i, ignore_newline=ignore_newline)
                a = KGFn(aa, [a, aaa], arity=2)
                if aaa is None:
                    a.missing_operand = True
            ii, aa = kg_read(t, i, ignore_newline=ignore_newline, module=self.current_module())
        if ignore_newline and safe_eq(a, '\n'):
            i = skip(t, i, ignore_newline=True)
        return i, a

    def _legacy_prog(self, t, i=0, ignore_newline=False):
        """

        Parse a Klong expression string into a Klong program, which can then be evaluated.

        # A program is a ';'-separated sequence of expressions.

        p := e
           | e ';' p

        """
        arr = []
        while i < len(t):
            i, q = self._legacy_expr(t,i, ignore_newline=ignore_newline)
            if q is None or safe_eq(q, ';'):
                continue
            arr.append(q)
            ii, c = kg_read(t, i, ignore_newline=ignore_newline, module=self.current_module())
            if c != ';':
                break
            i = ii
        return i, arr

    def _resolve_fn(self, f, f_args, f_arity):
        """

        Resolve a Klong function to its final form and update its arguments and arity.

        This helper function is used to resolve function references, projections, and invocations
        while considering the complexities involved, such as:
        * Functions may contain references to symbols which need to be resolved.
        * Functions may be projections or an invocation of a projection, and arguments
        need to be "flattened".

        Args:
            f (KGFn or KGSym): The Klong function or symbol to resolve.
            f_args (list): The list of arguments currently being associated with the function for its evaluation.
            f_arity (int): The arity of the function.

        Returns:
            tuple: A tuple containing the resolved function, updated arguments list, and updated arity.

        Details:
        - f_args is the list of arguments provided to the function during the current invocation.
        - f.args (if f is an instance of KGFn) represents predefined arguments associated with the function.
        These are arguments that are already set by virtue of the function being a projection of another function.
        - During the resolution process, if the function f is a KGFn with predefined arguments (projections),
        these arguments are appended to the f_args list.
        - The resolution process ensures that if there are placeholders in the predefined arguments, they are
        filled in with values from the provided arguments. However, if the function itself is being projected,
        f_args can still contain a None indicating an empty projection slot.
        - By the end of the resolution, f_args contains the arguments that will be passed to the function for
        evaluation. It is possible for f_args to finally contain None if the function is itself being projected.
        - If `f.a` (the main function or symbol) resolves to another function `_f`, it might indicate a higher-order function scenario.
        - In such a case, `f_args` supplies a function as an argument to `f.a`.
        - The `f.args` are then arguments for the function provided by `f_args`.
        - Even if this function (`_f`) corresponds to a reserved symbol, it should still undergo the projection flattening process in subsequent recursive calls.

        """
        if isinstance(f, KGSym):
            try:
                _f = self._context[f]
                if isinstance(_f, (KGFn,KGLambda)) or not in_map(f, reserved_fn_symbols):
                    # if f is a symbol and it resolves to a function, then we resolve f as the function.
                    # In this case, the f_args are meant for the resolved function.
                    f = _f
                else:
                    # f is x, y or z and holds a value that is not a function.
                    # If that value is a symbol naming a function, call the
                    # function it names, as a call through any other variable
                    # holding a symbol does ({x(1)}'[f g]). The symbol is in
                    # call position here; as data it is never looked up.
                    _g = None
                    if isinstance(_f, KGSym) and not in_map(_f, reserved_fn_symbols):
                        try:
                            _g = self._context[_f]
                        except KeyError:
                            pass
                    if not isinstance(_g, (KGFn,KGLambda)):
                        return f, f_args, f_arity
                    f = _g
            except KeyError:
                if not in_map(f, reserved_fn_symbols):
                    raise KlongNameError(f"undefined function: {f}")
        if f_arity > 0 and isinstance(f, KGFn) and not f.is_op() and not f.is_adverb_chain():
            if f.args is None:
                # if f.args is None, then there are no projections in place and we use f_args entirely for the function.
                return f.a, f_args, f.arity
            elif has_none(f.args):
                f_args.append(f.args if isinstance(f.args, list) else [f.args])
                return f.a, f_args, f.arity
        return f, f_args, f_arity

    def _eval_fn(self, x: KGFn):
        """

        Evaluate a Klong function.

        The incoming "x" argument is the description of the function to be evaluated.

        x.a may be a symbol or an actual function. If it's a symbol and a reserved symbol, then it should be resolved to a function.
        x.args contains the arguments to be passed to the function.
        x.arity contains the arity of the function.

        Processing before function execution is to build the local context for the function so that
        when it references x,y and z it retreives the arguments provided to the function.

        In practice, its often way more complex to derived these arguemnts because:

        * Functions may contain references to symbols which need to be resolved
        * Functions may be projections or a invocation of a projection and
            arguments need to be "flattened".

        To address these points:

        * Projections are recursively merged with arguments and mapped to x,y, and z as appropriate.
        * A new runtime context is prepared and the function is invoked.
        * Locals are populated with identity first in the local context.
        * The system var .f is populated in the local context and made available to the function.

        Notes:

            In higher-order function scenarios, the x.args contains the function referenced by the symbol in by x.a.
            Subsequent processing will then use the arguments attached to the referenced function as the basis for projection flattening.

        The arguments in x.args are unevaluated expressions (the parser built
        this call, or call() turned a function into one). They are evaluated here,
        once, after projections are merged. A caller that already holds argument
        values must use apply() instead.

        """
        f = x.a
        f_arity = x.arity
        f_args = [None] if x.args is None else [x.args if isinstance(x.args, list) else [x.args]]

        # three passes as there are max three argumentes: x,y, and z
        f, f_args, f_arity = self._resolve_fn(f, f_args, f_arity)
        f, f_args, f_arity = self._resolve_fn(f, f_args, f_arity)
        f, f_args, f_arity = self._resolve_fn(f, f_args, f_arity)

        f_args.reverse()
        f_args = merge_projections(f_args)
        if (0 if f_args is None else len(f_args)) < f_arity or has_none(f_args):
            return x

        ctx = {} if f_args is None else {p: self.call(q) for p,q in zip(reserved_fn_symbols,f_args)}

        return self._run_fn(f, ctx)

    def apply(self, f, values, arity=None):
        """

        Apply a function to argument values that are already evaluated.

        The values are bound to x, y and z as they are: no value is evaluated
        again, so a symbol stays a symbol, a function stays a function and a
        list is never run as a program. Every call made at run time (adverbs,
        the apply form of @, KGFnWrapper, IPC and timer callbacks, autograd)
        goes through here. Calls written in the source go through _eval_fn,
        which evaluates the argument expressions first.

        Args:
            f: what to apply: a function value (KGFn), a symbol naming one, a
               KGLambda, an operator (KGOp), or a function body.
            values (list): the argument values.
            arity (int): the arity to assume for f when f does not carry its
               own (a body or a KGLambda). Defaults to len(values).

        If f is a projection, the arguments it already holds are expressions and
        are evaluated now, as in a written call; only "values" are exempt.

        If there are fewer values than f needs, the unapplied call is returned,
        as a written call does.

        """
        if isinstance(f, KGOp):
            try:
                return self._vm[f.a](values[0]) if len(values) == 1 else self._vd[f.a](values[0], values[1])
            except Exception as e:
                if isinstance(e, KlongError) or is_passthrough(e):
                    raise
                raise operator_error(e, f.a, operator_name(f.a, 1 if len(values) == 1 else 2), values) from e

        fn = f
        f_arity = len(values) if arity is None else arity
        f_args = [values]

        # three passes as there are max three argumentes: x,y, and z
        f, f_args, f_arity = self._resolve_fn(f, f_args, f_arity)
        f, f_args, f_arity = self._resolve_fn(f, f_args, f_arity)
        f, f_args, f_arity = self._resolve_fn(f, f_args, f_arity)

        if len(f_args) == 1:
            # no projection involved: the values are the arguments
            if len(values) < f_arity:
                return KGCall(fn, values, len(values) if arity is None else arity)
        else:
            # Projection arguments are unevaluated expressions; mark the values
            # so that only the projection arguments are evaluated after merging.
            f_args[0] = [_Value(v) for v in values]
            f_args.reverse()
            f_args = merge_projections(f_args)
            if len(f_args) < f_arity or has_none(f_args):
                return KGCall(fn, values, len(values) if arity is None else arity)
            values = [q.v if type(q) is _Value else self.call(q) for q in f_args]

        return self._run_fn(f, dict(zip(reserved_fn_symbols, values)))

    def _run_fn(self, f, ctx):
        """

        Run a resolved function with its arguments bound in ctx: declare its
        locals, set .f, and evaluate the body in a new context frame.

        """
        if is_list(f) and len(f) > 1 and is_list(f[0]) and len(f[0]) > 0:
            # Filter out semicolons and check if ALL remaining elements are symbols.
            # A mixed list like [a 1] is a normal array literal, not a local declaration.
            non_sep = [q for q in f[0] if q != ';']
            params = [q for q in non_sep if isinstance(q, KGSym)]
            have_locals = len(params) > 0 and len(params) == len(non_sep)
            if have_locals:
                for q in params:
                    # Don't overwrite function parameters (x, y, z)
                    if q not in ctx:
                        ctx[q] = q
                f = f[1:]

        ctx[reserved_dot_f_symbol] = f

        self._context.push(ctx)
        try:
            if issubclass(type(f), KGLambda):
                try:
                    return f(self, self._context)
                except KlongError:
                    raise
                except Exception as e:
                    # User Python (.py, .pyf, .pya) fails as a KlongPythonError.
                    # A system function's OSError becomes a KlongSystemError;
                    # anything else it raises is passed on as it is.
                    err = python_error(e) if f.user else system_error(e)
                    if err is e:
                        raise
                    raise err from e
            return self.call(f)
        finally:
            self._context.pop()

    def _eval_quoted_symbol(self, x):
        """

        Evaluate a quoted symbol (`:foo`, a KGQSym node).

        STRICT-LEVEL SWITCH (ticket 051): this is the one place that decides
        what a quoted symbol evaluates to.

        Strict levels 0 and 1 (all that exist today) keep the legacy rule: a
        quoted symbol is treated like the bare name, so it gives the variable's
        value when one is defined and the symbol otherwise (and, like a bare
        undefined name, it is then defined as itself).

        Strict level 2 is to return the plain symbol and never look it up,
        i.e. only the last two lines of this method.

        The result is never a KGQSym: the symbol value is a plain KGSym.

        """
        try:
            return self._context[x]
        except KeyError:
            pass
        s = reserved_fn_symbol_map.get(str.__str__(x))
        if s is None:
            s = KGSym(x)
            self._context[s] = s
        return s

    def call(self, x):
        """

        Invoke a Klong program (as produced by prog()), causing functions to be called and evaluated.

        A function that is not yet a call is put in call form here; its args, if
        any, are the unevaluated expressions it was parsed with.

        """
        if isinstance(x, KGFn):
            c = KGCall(x.a, x.args, x.arity)
            c._origin = getattr(x, '_origin', x)  # compile cache lives on the node as parsed
            return self.eval(c)
        return self.eval(x)

    def _compiled_for(self, x):
        """The CompiledExpr for an operator or adverb-chain node, or False.

        The result (including "not compilable") is kept on the node as parsed,
        so the KGCall that call() makes for every evaluation shares it.
        """
        node = getattr(x, '_origin', x)
        compiled = getattr(node, '_compiled', None)
        if compiled is None:
            compiled = compile_guarded(x, self) or False
            node._compiled = compiled
        return compiled

    def _locate(self, e, x):
        """
        Give the error e, which has no position yet, the position of x: an
        expression whose evaluation e is passing through on its way out. Since
        this runs at every level the error passes and only while e has no
        span, the innermost node that has one wins.

        A leaf (a name, a number) carries no span. It is remembered instead,
        and the node that holds it as an argument supplies the span of that
        argument from its arg_spans.
        """
        node = getattr(x, '_origin', x)
        if not isinstance(node, (KGFn, KGCond, KGExprArray)):
            if node is not None and '_leaf' not in e.__dict__:
                e._leaf = node
            return
        span = node.span
        if span is None:
            return
        leaf = e.__dict__.pop('_leaf', None)
        spans = node.arg_spans
        if leaf is not None and spans:
            if isinstance(node, KGFn):
                args = node.a[-1] if node.is_adverb_chain() else node.args
                if not isinstance(args, list):
                    args = [args]
            else:
                args = node
            for a, sp in zip(args, spans):
                if a is leaf and sp is not None:
                    span = sp
                    break
        e.span = span
        e.source = node.source

    def _no_operand(self, x):
        """Raise the syntax error for the monad node x, which has no operand ("-" alone)."""
        node = getattr(x, '_origin', x)
        raise KlongSyntaxError(f"missing operand for {x.a.a}", span=node.op_span,
                               source=node.source if node.op_span is not None else None)

    def _op_error(self, e, x, operands):
        """
        The error to raise for the exception e that the operator node x raised
        when applied to operands (their values); e itself when it is to be
        passed on. This is the only place operator messages are made:

            cannot apply + (Plus) to integer and string

        The classification table is klongpy.errors.classify. An error a
        system function raised is passed on untouched, as is a KlongError,
        which only gets the operator's position if it has none.

        A node the parser marked missing_operand ("1+") reports whatever went
        wrong as the syntax error it is.
        """
        node = getattr(x, '_origin', x)
        op = x.a
        if node.missing_operand:
            err = KlongSyntaxError(f"missing right operand for {op.a}" if op.arity != 1 else f"missing operand for {op.a}")
        elif isinstance(e, KlongError):
            err = e
            if err.span is not None:
                return err
        elif is_passthrough(e):
            return e
        else:
            arity = 1 if op.arity == 1 else 2
            err = operator_error(e, op.a, operator_name(op.a, arity), operands)
        if node.op_span is not None:
            err.__dict__.pop('_leaf', None)
            err.span = node.op_span
            err.source = node.source
        return err

    def _chain_error(self, e, x, operands):
        """As _op_error, for the adverb chain node x: "cannot apply +/ (Plus-Over) to list of 2"."""
        if is_passthrough(e):
            return e
        node = getattr(x, '_origin', x)
        if isinstance(e, KlongError):
            err = e
            if err.span is not None:
                return err
        else:
            arr = x.a
            verb = arr[0]
            names = []
            if isinstance(verb.a, KGOp):
                glyph = verb.a.a
                names.append(operator_name(glyph, verb.arity))
            else:
                glyph = 'f'
            for adv in arr[1:-1]:
                glyph += adv.a
                names.append(adverb_name(adv.a, adv.arity))
            err = operator_error(e, glyph, "-".join(n for n in names if n), operands)
        if node.op_span is not None:
            err.__dict__.pop('_leaf', None)
            err.span = node.op_span
            err.source = node.source
        return err

    def _compiled_error(self, e, x):
        """
        The error to raise for the exception e that the compiled function of
        the node x raised; e itself when it is to be passed on (anything that
        is not one of the library errors operators raise).

        The compiled function is the whole expression, so which operator
        failed on what is not known. The operands of x are evaluated to find
        out: a compiled expression is literals, variables and arithmetic, so
        that has no side effect, and x itself is not run again. An operand
        that fails is the error; otherwise x failed on these values.
        """
        if isinstance(e, KlongError) or is_passthrough(e) or classify(e) is KlongInternalError:
            return e
        try:
            if x.is_op():
                fa = x.args if isinstance(x.args, list) else [x.args]
                return self._op_error(e, x, tuple(self.eval(a) for a in fa))
            a = x.a[-1]
            return self._chain_error(e, x, (self.eval(a),) if x.a[-2].arity == 1 else (self.eval(a[0]), self.eval(a[1])))
        except KlongError as inner:
            return inner
        except Exception:
            return e

    def eval(self, x):
        """

        Evaluate a Klong program.

        The fundamental design ideas include:

        * Python lists contain programs and NumPy arrays contain data.
        * Functions (KGFn) are not invoked unless they are KGCall instances, allowing for function definitions to be differentiated from invocations.

        Errors. Nothing here runs unless something is raised (a try costs
        nothing until then). A KlongError on its way out is given the position
        of the innermost node it passes that has one (_locate). Any other
        exception that escapes an operator or an adverb chain is turned into a
        KlongError there (_op_error, _chain_error): that is the one place
        library exceptions (numpy, torch) are classified. SystemExit,
        KeyboardInterrupt and GeneratorExit are not Exceptions and are never
        caught.

        """
        try:
            if isinstance(x, KGSym):
                if type(x) is KGQSym:
                    return self._eval_quoted_symbol(x)
                try:
                    return self._context[x]
                except KeyError:
                    if x not in reserved_fn_symbols:
                        self._context[x] = x
                    return x
            elif isinstance(x, KGFn):
                if x.is_op():
                    # Try compiled path for operator expressions (skips assignment/gradient)
                    if x.a.a not in ('::','∇'):
                        compiled = self._compiled_for(x)
                        if compiled:
                            try:
                                r = run_compiled(self, compiled)
                            except Exception as e:
                                err = self._compiled_error(e, x)
                                if err is e:
                                    raise
                                raise err from e
                            if r is not NOT_COMPILED:
                                return r
                    f = self._get_op_fn(x.a.a, x.a.arity)
                    fa = (x.args if isinstance(x.args, list) else [x.args]) if x.args is not None else self._no_operand(x)
                    _y = self.eval(fa[1]) if x.a.arity == 2 else None
                    _x = fa[0] if x.a.a in ['::','∇'] else self.eval(fa[0])
                    try:
                        return f(_x) if x.a.arity == 1 else f(_x, _y)
                    except Exception as e:
                        err = self._op_error(e, x, (_x,) if x.a.arity == 1 else (_x, _y))
                        if err is e:
                            raise
                        raise err from e
                elif x.is_adverb_chain():
                    # Try compiled path for adverb chains (reduce/scan)
                    compiled = self._compiled_for(x)
                    if compiled:
                        try:
                            r = run_compiled(self, compiled)
                        except Exception as e:
                            err = self._compiled_error(e, x)
                            if err is e:
                                raise
                            raise err from e
                        if r is not NOT_COMPILED:
                            return r
                    arr = x.a
                    f = chain_fn(self, arr)
                    a = arr[-1]
                    if arr[-2].arity == 1:
                        _x = self.eval(a)
                        try:
                            return f(_x)
                        except Exception as e:
                            err = self._chain_error(e, x, (_x,))
                            if err is e:
                                raise
                            raise err from e
                    else:
                        _x = self.eval(a[0])
                        _y = self.eval(a[1])
                        try:
                            return f(_x, _y)
                        except Exception as e:
                            err = self._chain_error(e, x, (_x, _y))
                            if err is e:
                                raise
                            raise err from e
                elif isinstance(x, KGCall):
                    return self._eval_fn(x)
            elif isinstance(x, KGCond):
                q = self.call(x[0])
                p = not ((self._backend.is_number(q) and q == 0) or is_empty(q))
                return self.call(x[1]) if p else self.call(x[2])
            elif isinstance(x, KGExprArray):
                results = [self.call(e) for e in x]
                return self._backend.kg_asarray(results)
            elif isinstance(x,list) and len(x) > 0:
                return [self.call(y) for y in x][-1]
            return x
        except KlongError as e:
            if e.span is None:
                self._locate(e, x)
            raise

    def _top_error(self, e, n, spans, source):
        """
        A KlongError is leaving the top level from statement n of a program:
        a statement that is a bare leaf (a lone name) has no node to take a
        position from, so it gets the span of the statement.
        """
        if e.span is None and spans is not None and n < len(spans):
            e.__dict__.pop('_leaf', None)
            e.span = spans[n]
            e.source = source

    def __call__(self, x):
        """
        Convenience method for executing Klong programs.

        If the result only contains one entry, it's directly returned for convenience.

        x is the program text, or a klongpy.lexer.Source (a text with a name,
        e.g. a file path): errors then report positions under that name.
        """
        # Parse cache, keyed by (source, module, strict level): parsing depends
        # on the active module (symbols get module-qualified names) and syntax
        # checks depend on the strict level. An entry is the triple (tree,
        # statement spans, Source); tree is the statement itself when there
        # is one, else the list of them.
        module = self._module
        parse_key = (x, module, self._context._strict_mode)
        cached = self._parse_cache.get(parse_key)
        if cached is None:
            p = self.prog(x)
            prog = p[1]
            cached = (prog[0] if len(prog) == 1 else prog, p.stmt_spans, p.source)
            # A parse that switched the module (.module) has a side effect on
            # the parser that a cached tree would not repeat: do not cache it.
            m = self._module
            if m is module or (m is not None and module is not None and m == module):
                self._parse_cache[parse_key] = cached
        tree, spans, source = cached
        sources = self._sources
        sources.append(source)
        n = 0
        try:
            # Try compiled path (single expressions only)
            if type(tree) is not list:
                cache_key = (x, module)
                compiled = self._compiled_cache.get(cache_key)
                if compiled is None:
                    compiled = compile_guarded(tree, self) or False
                    self._compiled_cache[cache_key] = compiled
                if compiled:
                    try:
                        r = run_compiled(self, compiled)
                    except Exception as e:
                        err = self._compiled_error(e, tree)
                        if err is e:
                            raise
                        raise err from e
                    if r is not NOT_COMPILED:
                        return r
                return self.call(tree)

            # Existing interpreter path
            r = None
            for y in tree:
                r = self.call(y)
                n += 1
            return r
        except KlongError as e:
            self._top_error(e, n, spans, source)
            raise
        except RecursionError as e:
            raise compat(KlongRangeError, RecursionError)("expression nested too deeply") from e
        finally:
            sources.pop()

    def exec(self, x):
        """

        Execute a Klong program.

        Each subprogram is executed in order and the resulting array contains the resulst of each sub-program.

        """
        p = self.prog(x)
        spans, source = p.stmt_spans, p.source
        sources = self._sources
        sources.append(source)
        r = []
        try:
            for y in p[1]:
                r.append(self.call(y))
            return r
        except KlongError as e:
            self._top_error(e, len(r), spans, source)
            raise
        finally:
            sources.pop()
