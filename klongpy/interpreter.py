import time
from collections import deque

from .adverbs import get_adverb_fn
from .core import *
from .dyads import create_dyad_functions
from .monads import create_monad_functions
from .sys_fn import create_system_functions
from .sys_var import *
from .utils import ReadonlyDict


def set_context_var(d, sym, v):
    """
    Sets a context variable, wrapping Python lambda/functions as appropriate.
    """
    assert isinstance(sym, KGSym)
    if callable(v):
        x = KGLambda(v)
        v = KGCall(x,args=None,arity=x.get_arity())
    d[sym] = v


class KGModule(dict):
    """
    A module class that is used for optimizing when to scan for namespaced keys.
    """
    def __init__(self, name=None):
        self.name = name
        super().__init__()


class KlongContext():

    def __init__(self, system_contexts):
        self._context = deque([{}, *system_contexts])
        self._min_ctx_count = len(system_contexts)

    def start_module(self, name):
        self.push(KGModule(name))
        self._min_ctx_count = len(self._context)

    def stop_module(self):
        self.push({})

    def current_module(self):
        return self._module

    def __setitem__(self, k, v):
        assert isinstance(k, KGSym)
        if k not in reserved_fn_symbols:
            for d in self._context:
                if in_map(k, d):
                    d[k] = v
                    return k
        set_context_var(self._context[0], k, v)
        return k

    def __getitem__(self, k):
        assert isinstance(k, KGSym)
        for d in self._context:
            v = d.get(k)
            if v is not None:
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


def add_system_functions(d, system_functions):
    for k,fn in system_functions.items():
        set_context_var(d, KGSym(k), fn)


def create_system_contexts(klong):
    cin = eval_sys_var_cin()
    cout = eval_sys_var_cout()
    cerr = eval_sys_var_cerr()

    sys_d = {}
    add_system_functions(sys_d, create_system_functions())
    set_context_var(sys_d, KGSym('.e'), eval_sys_var_epsilon()) # TODO: this is probably a bug that this can't be a lambda
    set_context_var(sys_d, KGSym('.cin'), cin)
    set_context_var(sys_d, KGSym('.cout'), cout)
    set_context_var(sys_d, KGSym('.cerr'), cerr)

    sys_var = {}
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
    if arr[0].arity == 1:
        f = lambda x,k=klong,a=arr[0].a: k.eval(KGCall(a, [x], arity=1))
    else:
        f = lambda x,y,k=klong,a=arr[0].a: k.eval(KGCall(a, [x,y], arity=2))
    for i in range(1,len(arr)-1):
        o = get_adverb_fn(klong, arr[i].a, arity=arr[i].arity)
        if arr[i].arity == 1:
            f = lambda x,f=f,o=o: o(f,x)
        else:
            f = lambda x,y,f=f,o=o: o(f,x,y)
    if arr[-2].arity == 1:
        f = lambda a=arr[-1],f=f,k=klong: f(k.eval(a))
    else:
        f = lambda a=arr[-1],f=f,k=klong: f(k.eval(a[0]),k.eval(a[1]))
    return f


class KlongInterpreter():

    def __init__(self):
        self._context = KlongContext(create_system_contexts(self))
        self._vd = create_dyad_functions(self)
        self._vm = create_monad_functions(self)
        self._start_time = time.time()
        self._module = None

    def __setitem__(self, k, v):
        k = k if isinstance(k, KGSym) else KGSym(k)
        self._context[k] = v

    def __getitem__(self, k):
        k = k if isinstance(k, KGSym) else KGSym(k)
        return self._context[k]

    def _get_op_fn(self, s, arity):
        return self._vm[s] if arity == 1 else self._vd[s]

    def _is_monad(self, s):
        return isinstance(s,KGOp) and in_map(s.a, self._vm)

    def _is_dyad(self, s):
        return isinstance(s,KGOp) and in_map(s.a, self._vd)

    def start_module(self, name):
        self._context.start_module(name)

    def stop_module(self):
        self._context.stop_module()

    def parse_module(self, name):
        self._module = None if safe_eq(name,0) or is_empty(name) else name

    def current_module(self):
        return self._module

    def process_start_time(self):
        return self._start_time

    def _apply_adverbs(self, t, i, a, aa, arity, dyad=False, dyad_value=None):
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
        i, aa = self._expr(t, i)
        arr.append([dyad_value,aa] if dyad else aa)
        return i,KGCall(arr,args=None,arity=2 if dyad else 1)

    def _read_fn_args(self, t, i=0):
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
            ii,c = kg_read(t,i,ignore_newline=True,module=self.current_module())
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
            i,a = self._expr(t,i,ignore_newline=True)
            if a is None:
                break
            arr.append(a)
        i = cexpect(t,i,')')
        return i,arr


    def _factor(self, t, i=0, ignore_newline=False):
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
        i,a = kg_read(t, i, ignore_newline=ignore_newline, module=self.current_module())
        if a is None:
            return i,a
        if isinstance(a,KGChar) or is_number(a):
            pass
        elif safe_eq(a, '{'): # read fn
            i,a = self.prog(t, i, ignore_newline=True)
            a = a[0] if len(a) == 1 else a
            i = cexpect(t, i, '}')
            arity = get_fn_arity(a)
            if cmatch(t, i, '(') or cmatch2(t,i,':','('):
                i,fa = self._read_fn_args(t,i)
                a = KGFn(a, fa, arity) if has_none(fa) else KGCall(a, fa, arity)
            else:
                a = KGFn(a, args=None, arity=arity)
            ii, aa = peek_adverb(t, i)
            if aa:
                i,a = self._apply_adverbs(t, ii, a, aa, arity=1)
        elif isinstance(a, KGSym):
            if cmatch(t,i,'(') or cmatch2(t,i,':','('):
                i,fa = self._read_fn_args(t,i)
                a = KGFn(a, fa, arity=len(fa)) if has_none(fa) else KGCall(a, fa, arity=len(fa))
                if safe_eq(a.a, KGSym(".comment")):
                    i = read_sys_comment(t,i,a.args[0])
                    return self._factor(t,i, ignore_newline=ignore_newline)
                elif safe_eq(a.a, KGSym('.module')):
                    self.parse_module(fa[0])
            ii, aa = peek_adverb(t, i)
            if aa:
                i,a = self._apply_adverbs(t, ii, a, aa, arity=1)
        elif self._is_monad(a):
            a.arity = 1
            ii, aa = peek_adverb(t, i)
            if aa:
                i,a = self._apply_adverbs(t, ii, a, aa, arity=1)
            else:
                i, aa = self._expr(t, i, ignore_newline=ignore_newline)
                a = KGFn(a, aa, arity=1)
        elif safe_eq(a, '('):
            i,a = self._expr(t, i, ignore_newline=ignore_newline)
            i = cexpect(t, i, ')')
        elif safe_eq(a, ':['):
            return read_cond(self, t, i)
        return i, a

    def _expr(self, t, i=0, ignore_newline=False):
        """

        # An expression is a factor or a dyadic operation applied to
        # a factor and an expression. I.e. dyads associate to the right.

        e := x
           | x d e

        """
        i, a = self._factor(t, i, ignore_newline=ignore_newline)
        if a is None or safe_eq(a, ';'):
            return i,a
        ii, aa = kg_read(t, i, ignore_newline=ignore_newline, module=self.current_module())
        if self._is_dyad(aa):
            aa.arity = 2
        while isinstance(aa,(KGOp,KGSym)) or safe_eq(aa, '{'):
            i = ii
            if safe_eq(aa, '{'): # read fn
                i,aa = self.prog(t, i, ignore_newline=True)
                aa = aa[0] if len(aa) == 1 else aa
                i = cexpect(t, i, '}')
                arity = get_fn_arity(aa)
                if cmatch(t, i, '(') or cmatch2(t,i,':','('):
                    i,fa = self._read_fn_args(t,i)
                    aa = KGFn(aa, fa, arity=arity) if has_none(fa) else KGCall(aa, fa, arity=arity)
                else:
                    aa = KGFn(aa, args=None, arity=arity)
            elif isinstance(aa,KGSym) and (cmatch(t, i, '(') or cmatch2(t,i,':','(')):
                i,fa = self._read_fn_args(t,i)
                aa = KGFn(aa, fa, arity=len(fa)) if has_none(fa) else KGCall(aa, fa, arity=len(fa))
            ii, aaa = peek_adverb(t, i)
            if aaa:
                i,a = self._apply_adverbs(t, ii, aa, aaa, arity=2, dyad=True, dyad_value=a)
            else:
                i, aaa = self._expr(t, i, ignore_newline=ignore_newline)
                a = KGFn(aa, [a, aaa], arity=2)
            ii, aa = kg_read(t, i, ignore_newline=ignore_newline, module=self.current_module())
        return i, a

    def prog(self, t, i=0, ignore_newline=False):
        """

        Parse a Klong expression string into a Klong program, which can then be evaluated.

        # A program is a ';'-separated sequence of expressions.

        p := e
           | e ';' p

        """
        arr = []
        while i < len(t):
            i, q = self._expr(t,i, ignore_newline=ignore_newline)
            if q is None or safe_eq(q, ';'):
                continue
            arr.append(q)
            ii, c = kg_read(t, i, ignore_newline=ignore_newline, module=self.current_module())
            if c != ';':
                break
            i = ii
        return i, arr

    def _eval_fn(self, x):
        """

        Evaluate a Klong function.

        The basic idea is the prepare a local context for the function so that
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

        Frankly, the best way to understand this code is to step through it with examples.

        """
        f = x.a
        f_arity = x.arity
        f_args = [(x.args if isinstance(x.args, list) else [x.args]) if x.args is not None else x.args]

        if self._context.is_defined_sym(f) or (isinstance(f,KGFn) and self._context.is_defined_sym(f.a)):
            while self._context.is_defined_sym(f) or (isinstance(f,KGFn) and self._context.is_defined_sym(f.a)):
                if (isinstance(f,KGFn) and self._context.is_defined_sym(f.a)):
                    f_args.append(f.args)
                    f_arity = f.arity
                    f = f.a
                f = self.eval(f)
                if isinstance(f,KGFn):
                    if f.args is not None:
                        f_args.append((f.args if isinstance(f.args, list) else [f.args]))
                    f_arity = f.arity
                    f = f.a
            f_args.reverse()
        elif isinstance(f,KGFn) and not (f.is_op() or f.is_adverb_chain()):
            # TODO: should this be recursive towards convergence?
            if f.args is not None:
                f_args.append(f.args)
            f_args.reverse()
            f_arity = f.arity
            f = f.a

        f_args = merge_projections(f_args)
        if (f_args is None and f_arity > 0) or (is_list(f_args) and len(f_args) < f_arity) or has_none(f_args):
            return x

        ctx = {} if f_args is None else {reserved_fn_symbol_map[p]: self.call(q) for p,q in zip(reserved_fn_args,f_args)}

        if is_list(f) and len(f) > 1 and is_list(f[0]) and len(f[0]) > 0:
            have_locals = True
            for q in f[0]:
                if not isinstance(q, KGSym):
                    have_locals = False
                    break
            if have_locals:
                for q in f[0]:
                    ctx[q] = q
                f = f[1:]

        self._context.push(ctx)
        try:
            # TODO: more testing on whether this is actualy the correct f
            ctx[KGSym('.f')] = f
            return f(self, self._context) if isinstance(f, KGLambda) else self.call(f)
        finally:
            # always pop because if we don't, then when a function has an exception in debugging it will leave the stack corrupted
            self._context.pop()

    def call(self, x):
        """

        Invoke a Klong program (as produced by prog()), causing functions to be called and evaluated.

        """
        return self.eval(KGCall(x.a, x.args, x.arity) if isinstance(x, KGFn) else x)

    def eval(self, x):
        """
        Evaluate a Klong program.

        The fundamental design ideas include:

        * Python lists contain programs and NumPy arrays contain data.
        * Functions (KGFn) are not invoked unless they are KGCall instances, allowing for function definitions to be differentiated from invocations.

        """

        if isinstance(x, KGSym):
            try:
                return self[x]
            except KeyError:
                if x not in reserved_fn_symbols:
                    self[x] = x
                return x
        elif isinstance(x, KGCond):
            q = self.call(x[0])
            p = not ((is_number(q) and q == 0) or is_empty(q))
            return self.call(x[1]) if p else self.call(x[2])
        elif isinstance(x, KGCall) and not (x.is_op() or x.is_adverb_chain()):
            return self._eval_fn(KGFn(x.a,x.args,x.arity))
        elif isinstance(x, KGFn) and x.is_op():
            f = self._get_op_fn(x.a.a, x.a.arity)
            fa = (x.args if isinstance(x.args, list) else [x.args]) if x.args is not None else x.args
            # eval y before x to preserve right to left semantics
            _y = self.eval(fa[1]) if x.a.arity == 2 else None
            _x = fa[0] if x.a.a == '::' else self.eval(fa[0])
            return f(_x) if x.a.arity == 1 else f(_x, _y)
        elif isinstance(x, KGFn) and x.is_adverb_chain():
            return chain_adverbs(self, x.a)()
        elif isinstance(x,list) and len(x) > 0:
            return [self.call(y) for y in x][-1]
        return x

    def __call__(self, x, *args, **kwds):
        """
        Convience method for executing Klong programs.

        If the result only contains one entry, it's directly returned for convenience.

        Example:

        klong = KlongInterpreter()
        r = klong("1+1")
        assert r == 2

        or more succinctly

        assert 2 == KlongInterpreter()("1+1")

        """
        return self.exec(x)[-1]

    def exec(self, x):
        """
        Execute a Klong program.

        Each subprogram is executed in order and the resulting array contains the resulst of each sub-program.
        """
        return [self.call(y) for y in self.prog(x)[1]]
