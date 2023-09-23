import time
from collections import deque

from .adverbs import get_adverb_fn
from .core import *
from .dyads import create_dyad_functions
from .monads import create_monad_functions
from .sys_fn import create_system_functions
from .sys_fn_ipc import create_system_functions_ipc, create_system_var_ipc
from .sys_fn_timer import create_system_functions_timer
from .sys_var import *
from .utils import ReadonlyDict


def set_context_var(d, sym, v):
    """
    Sets a context variable, wrapping Python lambda/functions as appropriate.
    """
    assert isinstance(sym, KGSym)
    if callable(v) and not issubclass(type(v), KGLambda) :
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
    if arr[0].arity == 1:
        f = lambda x,k=klong,a=arr[0].a: k.eval(KGCall(a, [x], arity=1))
    else:
        f = lambda x,y,k=klong,a=arr[0].a: k.eval(KGCall(a, [x,y], arity=2))
    for i in range(1,len(arr)-1):
        o = get_adverb_fn(klong, arr[i].a, arity=arr[i].arity)
        if arr[i].arity == 1:
            f = lambda x,f=f,o=o: o(f,x,op=arr[0].a)
        else:
            f = lambda x,y,f=f,o=o: o(f,x,y)
    if arr[-2].arity == 1:
        f = lambda a=arr[-1],f=f,k=klong: f(k.eval(a))
    else:
        f = lambda a=arr[-1],f=f,k=klong: f(k.eval(a[0]),k.eval(a[1]))
    return f


class KlongInterpreter():

    def __init__(self):
        self._context = KlongContext(create_system_contexts())
        self._vd = create_dyad_functions(self)
        self._vm = create_monad_functions(self)
        self._start_time = time.time()
        self._module = None

    def __setitem__(self, k, v):
        k = k if isinstance(k, KGSym) else KGSym(k)
        self._context[k] = v

    def __getitem__(self, k):
        k = k if isinstance(k, KGSym) else KGSym(k)
        r = self._context[k]
        return KGFnWrapper(self, r) if issubclass(type(r), KGFn) else r

    def __delitem__(self, k):
        k = k if isinstance(k, KGSym) else KGSym(k)
        del self._context[k]

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
        if safe_eq(a, '{'): # read fn
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
                    return f, f_args, f_arity
            except KeyError:
                if not in_map(f, reserved_fn_symbols):
                    raise KlongException(f"undefined: {f}")
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

        ctx[reserved_dot_f_symbol] = f

        self._context.push(ctx)
        try:
            return f(self, self._context) if issubclass(type(f), KGLambda) else self.call(f)
        finally:
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
                return self._context[x]
            except KeyError:
                if x not in reserved_fn_symbols:
                    self._context[x] = x
                return x
        elif isinstance(x, KGFn):
            if x.is_op():
                f = self._get_op_fn(x.a.a, x.a.arity)
                fa = (x.args if isinstance(x.args, list) else [x.args]) if x.args is not None else x.args
                _y = self.eval(fa[1]) if x.a.arity == 2 else None
                _x = fa[0] if x.a.a == '::' else self.eval(fa[0])
                return f(_x) if x.a.arity == 1 else f(_x, _y)
            elif x.is_adverb_chain():
                return chain_adverbs(self, x.a)()
            elif isinstance(x, KGCall):
                return self._eval_fn(x)
        elif isinstance(x, KGCond):
            q = self.call(x[0])
            p = not ((is_number(q) and q == 0) or is_empty(q))
            return self.call(x[1]) if p else self.call(x[2])
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
        r = self.exec(x)
        return r[-1] if r else None

    def exec(self, x):
        """

        Execute a Klong program.

        Each subprogram is executed in order and the resulting array contains the resulst of each sub-program.

        """
        return [self.call(y) for y in self.prog(x)[1]]
