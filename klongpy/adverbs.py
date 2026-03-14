import functools
import itertools
import math

import numpy as _numpy

from .core import *

_ndarray = _numpy.ndarray

# Dispatch tables for ufunc reduce/accumulate — avoids if/elif chains and hasattr checks
_reduce_dispatch = {
    '+': lambda np, a: np.add.reduce(a),
    '-': lambda np, a: np.subtract.reduce(a),
    '*': lambda np, a: np.multiply.reduce(a),
    '%': lambda np, a: np.divide.reduce(a),
}

_accumulate_dispatch = {
    '+': lambda np, a: np.add.accumulate(a),
    '-': lambda np, a: np.subtract.accumulate(a),
    '*': lambda np, a: np.multiply.accumulate(a),
    '%': lambda np, a: np.divide.accumulate(a),
}


def eval_adverb_converge(f, a, op, backend):
    """
        f:~a                                                  [Converge]

        Find the fixpoint of f(a), if any. The fixpoint of "f" is a value
        "a" for which f(a) = a. For example,

        {(x+2%x)%2}:~2

        converges toward the square root of two using Newton's method.
        Starting with x=2:

        (2+2%2)%2              -->  1.5
        (1.5+2%1.5)%2          -->  1.41666
        (1.41666+2%1.41666)%2  -->  1.41421  :"next value is the same"
        (1.41421+2%1.41421)%2  -->  1.41421

        (Of course, the precision of the actual implementation will
         probably be higher.)

        Example: ,/:~["f" ["l" "at"] "ten"]  -->  "flatten"

    """
    def _e(p,q):
        if type(p) is not type(q):
            return False
        if backend.is_number(p):
            # math.isclose is ~90x faster than np.isclose for Python scalars
            tp = type(p)
            if tp is int or tp is float:
                return p == q or math.isclose(p, q, rel_tol=1e-05, abs_tol=1e-08)
            return backend.np.isclose(p,q)
        elif backend.is_array(p):
            return backend.kg_equal(p, q)
        return p == q
    x = f(a)
    xx = f(x)
    while not _e(x,xx):
        x = xx
        xx = f(x)
    return x


def eval_adverb_each(f, a, op, backend):
    """

        f'a                                                       [Each]

        If "a" is a list, apply "f" to each member of "a":

        f'a  -->  f(a1),...,f(aN)

        If "a" is an atom, return f(a). If "a" is [], ignore "f" and
        return [].

        If "a" is a dictionary, apply "f" to each tuple stored in the
        dictionary. The resulting list will be in some random order.
        Applying {x} (the identity function) to a dictionary turns it
        into a list of tuples.

        Example: -'[1 2 3]  -->  [-1 -2 -3]

    """
    if isinstance(a,str):
        if is_empty(a):
            return a
        has_str = False
        r = []
        for x in backend.str_to_chr_arr(a):
            u = f(x)
            has_str |= isinstance(u,str)
            r.append(u)
        return ''.join(r) if has_str else backend.kg_asarray(r)
    if is_iterable(a):
        r = [f(x) for x in a]
        return a if is_empty(a) else backend.kg_asarray(r)
    elif is_dict(a):
        r = [f(backend.kg_asarray(x)) for x in a.items()]
        return backend.kg_asarray(r)
    return f(a)


def eval_adverb_each2(f, a, b):
    """

        a f'b                                                   [Each-2]

        Each-2 is like each, but applies "f" pairwise to elements of "a"
        and "b":

        a f'b  -->  f(a1;b1),...,f(aN;bN)

        If both "a" and "b" are atoms, return f(a;b). If either "a" or
        "b" is [], ignore "f" and return []. When the lengths of "a" and
        "b" differ, ignore any excess elements of the longer list.

        Example: [1 2 3],'[4 5 6]  -->  [[1 4] [2 5] [3 6]]

    """
    if is_empty(a) or is_empty(b):
        return bknp.asarray([]) if is_list(a) or is_list(b) else ""
    if is_atom(a) and is_atom(b):
        return f(a,b)
    r = bknp.asarray([f(x,y) for x,y in zip(a,b)])
    return ''.join(r) if r.dtype == '<U1' else r


def eval_adverb_each_left(f, a, b, backend):
    """
        a f:\b                                              [Each-Left]
        a f:/b                                              [Each-Right]

        If "b" is a list, both of these adverbs combine "a" with each
        element of "b", where :\ uses "a" as the left operand of "f",
        and :/ uses it as its right operand:

        a f:\b  -->  f(a;b1),...,f(a;bN)
        a f:/b  -->  f(b1;a),...,f(bN;a)

        If "b" is an atom, then

        a f:\b  -->  f(a;b)
        a f:/b  -->  f(b;a)

        When "b" is [], ignore "a" and "f" and return [].

        Examples: 1,:\[2 3 4]  -->  [[1 2] [1 3] [1 4]]
                  1,:/[2 3 4]  -->  [[2 1] [3 1] [4 1]]
    """
    b = backend.str_to_chr_arr(b) if isinstance(b,str) else b
    return backend.kg_asarray([f(a,x) for x in b])


def eval_adverb_each_right(f, a, b, backend):
    """
    see: eval_dyad_adverb_each_left
    """
    b = backend.str_to_chr_arr(b) if isinstance(b,str) else b
    return backend.kg_asarray([f(x,a) for x in b])


def eval_adverb_each_pair(f, a, op, backend):
    """

        f:'a                                                 [Each-Pair]

        If "a" is a list of more than one element, apply "f" to each
        consecutive pair of "a":

        f:'a  -->  f(a1;a2),f(a2;a3),...,f(aN-1;aN)

        If "a" is an atom or a single-element list, ignore "f" and
        return "a".

        Example: ,:'[1 2 3 4]  -->  [[1 2] [2 3] [3 4]]

    """
    if is_atom(a) or (is_iterable(a) and len(a) == 1):
        return a
    j = isinstance(a, str)
    a = backend.str_to_chr_arr(a) if j else a
    # Vectorized fast path for built-in dyadic ops on numpy arrays
    if type(a) is _ndarray and type(op) is KGOp:
        op_a = op.a
        if op_a == '-':
            return a[:-1] - a[1:]
        elif op_a == '+':
            return a[:-1] + a[1:]
        elif op_a == '*':
            return a[:-1] * a[1:]
        elif op_a == '%':
            return a[:-1] / a[1:]
        elif op_a == '|':
            return backend.np.maximum(a[:-1], a[1:])
        elif op_a == '&':
            return backend.np.minimum(a[:-1], a[1:])
    return backend.kg_asarray([f(x,y) for x,y in zip(a[::],a[1::])])


def eval_dyad_adverb_iterate(f, a, b):
    """

        a f:*b                                                 [Iterate]

        Apply "f" recursively to "b" "a" times. More formally:

        - if "a" is zero, return b
        - else assign b::f(b) and a::a-1 and start over

        Example: 3{1,x}:*[]  -->  [1 1 1]

    """
    while a != 0:
        b = f(b)
        a = a - 1
    return b


_over_cache = {}

def eval_adverb_over(f, a, op, backend):
    """
        f/a                                                       [Over]

        If "a" is a list, fold "f" over "a":

        f/a  -->  f(...f(f(a1;a2);a3)...;aN))
        +/a  -->  ((...(a1+a2)+...)+aN)

        If "a" is a single-element list, return the single element.

        If "a" is an atom, ignore "f" and return "a".

        Example: +/[1 2 3 4]  -->  10
    """
    # Inline is_atom + len check for common ndarray case
    ta = type(a)
    if ta is _ndarray:
        if a.ndim == 0 or len(a) == 0:
            return a
        if len(a) == 1:
            return a[0]
    elif is_atom(a):
        return a
    elif len(a) == 1:
        return a[0]
    # Use backend's ufunc reduce when available for better performance
    np_backend = backend.np
    if type(op) is KGOp:
        op_a = op.a
        # Cache reduce results for immutable arrays
        if type(a) is _ndarray and not a.flags.writeable:
            cache_key = (op_a, id(a))
            cached = _over_cache.get(cache_key)
            if cached is not None:
                return cached
        _reduce_fn = _reduce_dispatch.get(op_a)
        if _reduce_fn is not None:
            result = _reduce_fn(np_backend, a)
        elif op_a == '&' and a.ndim == 1:
            result = np_backend.min(a)
        elif op_a == '|' and a.ndim == 1:
            result = np_backend.max(a)
        elif op_a == ',' and np_backend.isarray(a) and a.dtype != 'O':
            result = a if a.ndim == 1 else np_backend.concatenate(a, axis=0)
        else:
            return functools.reduce(f, a)
        # Cache for immutable arrays
        if type(a) is _ndarray and not a.flags.writeable:
            _over_cache[(op_a, id(a))] = result
        return result
    return functools.reduce(f, a)


def eval_adverb_over_neutral(f, a, b):
    """

        a f/b                                             [Over-Neutral]

        This is like "/", but with a neutral element "a" that will be
        returned when "b" is [] or combined with the first element of
        "b" otherwise:

        a f/[]  -->  a
        a f/b   -->  f(...f(f(a;b1);b2)...;bN)

        For example, +/[] will give [], but 0+/[] will give 0.

        Of course, dyadic "/" can also be used to abbreviate an
        expression by supplying a not-so-neutral "neutral element".
        For instance, a++/b can be abbreviated to a+/b.

        If both "a" and "b" are atoms, "a f/b" will give f(a;b).

        Formally, "a f/b" is equal to f/a,b

        Example: 0,/[1 2 3]  -->  [0 1 2 3]
                 1+/[2 3 4]  -->  10

    """
    if is_empty(b):
        return a
    if is_atom(b):
        return f(a,b)
    return functools.reduce(f,b[1:],f(a,b[0]))


def eval_adverb_scan_over_neutral(f, a, b, backend):
    """

        f\a                                                  [Scan-Over]
        a f\b                                        [Scan-Over-Neutral]

        "\" is like "/", but collects intermediate results in a list and
        returns that list. In the resulting list,

        - the first slot will contain a1
        - the second slot will contain f(a1;a2)
        - the third slot will contain f(f(a1;a2);a3)
        - the last slot will contain f(...f(a1;a2)...;aN)
          (which is the result of f/a)

        If only one single argument is supplied, the argument will be
        returned in a list, e.g.: +\1 --> [1].

        "a f\b" is equal to f\a,b.

        Examples:  ,\[1 2 3]  -->  [1 [1 2] [1 2 3]]
                  0,\[1 2 3]  -->  [0 [0 1] [0 1 2] [0 1 2 3]]
    """
    if is_empty(b):
        return a
    if is_atom(b):
        b = [b]
    b = [f(a,b[0]), *b[1:]]
    r = list(itertools.accumulate(b,f))
    q = backend.kg_asarray(r)
    r = [a, *q]
    return backend.kg_asarray(r)


_scan_cache = {}

def eval_adverb_scan_over(f, a, op, backend):
    """
        see eval_adverb_scan_over_neutral
    """
    if is_atom(a):
        return a
    # Use backend's ufunc accumulate when available for better performance
    np_backend = backend.np
    if type(op) is KGOp:
        op_a = op.a
        # Cache scan results for immutable arrays
        if type(a) is _ndarray and not a.flags.writeable:
            cache_key = (op_a, id(a))
            cached = _scan_cache.get(cache_key)
            if cached is not None:
                return cached
        _accum_fn = _accumulate_dispatch.get(op_a)
        if _accum_fn is not None:
            result = _accum_fn(np_backend, a)
        else:
            return backend.kg_asarray(list(itertools.accumulate(a, f)))
        # Cache for immutable arrays
        if type(a) is _ndarray and not a.flags.writeable:
            result.flags.writeable = False
            _scan_cache[(op_a, id(a))] = result
        return result
    r = list(itertools.accumulate(a, f))
    return backend.kg_asarray(r)


def eval_adverb_scan_converging(f, a, op, backend):
    """

        f\~a                                           [Scan-Converging]

        Monadic \~ is like monadic :~, but returns a list of all
        intermediate results instead of just the end result. The
        last element of the list will be same as the result of a
        corresponding :~ application. For instance:

        {(x+2%x)%2}\~2

        will produce a list containing a series that converges toward
        the square root of 2.

        Example: ,/\~["a" ["b"] "c"]  -->  [["a" ["b"] "c"]
                                            ["a" "b" "c"]
                                            "abc"]

    """
    x = a
    xx = f(a)
    r = [a, xx]
    while not backend.kg_equal(x, xx):
        x = xx
        xx = f(x)
        r.append(xx)
    r.pop()
    return backend.kg_asarray(r)


def eval_adverb_scan_while(klong, f, a, b, backend):
    """

        a f\~b                                              [Scan-While]

        This adverb is (almost) like is non-scanning counterpart, :~,
        but it collects intermediate results in a list and returns that
        list.

        However, \~ will only collect values of X that satisfy a(X),
        while :~ will return the first value that does *not* satisfy
        a(X). E.g.:

        {x<10}{x+1}:~1  -->  10
        {x<10}{x+1}:\1  -->  [1 2 3 4 5 6 7 8 9]

        Example: {x<100}{x*2}\~1  -->  [1 2 4 8 16 32 64]

    """
    r = [b]
    # TODO: fix arity
    while klong.eval(KGCall(a, b, arity=1)):
        b = f(b)
        r.append(b)
    r.pop()
    return backend.kg_asarray(r)


def eval_adverb_scan_iterating(f, a, b, backend):
    """

        a f\*b                                          [Scan-Iterating]

        This adverbs is like its non-scanning counterpart, but collects
        intermediate results in a list and return that list.

        Example: 3{1,x}\*[]  -->  [[] [1] [1 1] [1 1 1]]

    """
    if a == 0:
        return b
    r = [b]
    while a != 0:
        b = f(b)
        r.append(b)
        a = a - 1
    return backend.kg_asarray(r)


def eval_adverb_while(klong, f, a, b):
    """

        a f:~b                                                   [While]

        Compute b::f(b) while a(b) is true. Formally:

        - if a(b) is false, return b
        - else assign b::f(b) and start over

        Example: {x<1000}{x*2}:~1  -->  1024

    """
    while klong.eval(KGCall(a, b, arity=1)):
        b = f(b)
    return b


def get_adverb_fn(klong, s, arity):
    backend = klong._backend

    if s == "'":
        return eval_adverb_each2 if arity == 2 else lambda f,a,op: eval_adverb_each(f,a,op,backend)
    elif s == '/':
        return eval_adverb_over_neutral if arity == 2 else lambda f,a,op: eval_adverb_over(f,a,op,backend)
    elif s == '\\':
        return (lambda f,a,b: eval_adverb_scan_over_neutral(f,a,b,backend)) if arity == 2 else lambda f,a,op: eval_adverb_scan_over(f,a,op,backend)
    elif s == '\\~':
        return (lambda f,a,b: eval_adverb_scan_while(klong,f,a,b,backend)) if arity == 2 else lambda f,a,op: eval_adverb_scan_converging(f,a,op,backend)
    elif s == '\\*':
        return lambda f,a,b: eval_adverb_scan_iterating(f,a,b,backend)
    elif s == ':\\':
        return lambda f,a,b: eval_adverb_each_left(f,a,b,backend)
    elif s == ':\'':
        return lambda f,a,op: eval_adverb_each_pair(f,a,op,backend)
    elif s == ':/':
        return lambda f,a,b: eval_adverb_each_right(f,a,b,backend)
    elif s == ':*':
        return eval_dyad_adverb_iterate
    elif s == ':~':
        return (lambda f,a,b: eval_adverb_while(klong,f,a,b)) if arity == 2 else lambda f,a,op: eval_adverb_converge(f,a,op,backend)
    raise RuntimeError(f"unknown adverb: {s}")
