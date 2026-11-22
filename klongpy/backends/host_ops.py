"""
The primitives of Klong on the backend's arrays, for the native engine's host table (klongpy/rust_host.py): monads, dyads and
adverbs written on backend operations (klongpy/backends/), taking and giving the backend's array type (a NumPy array, a torch
tensor on any device). A function-valued operand is called through klong.apply. This is the array code that was the Python
engine's monads, dyads and adverbs; the evaluator parts of those (the parse-tree walk, `::`) are not here.

    monad_table(klong), dyad_table(klong)    {glyph: f(x) / f(x, y)} for the primitives (the host table skips `::` and the gradient operators)
    get_adverb_fn(klong, glyph, arity)       the adverb's implementation: f(verb, x, op_or_y)
    converged(backend, p, q)                 the test of the converging adverbs
    format2(a, b, backend)                   a$b with broadcasting (.fmt)
"""
import numbers
import operator
import functools
import itertools

import numpy

from ..core import *


# ---- from monads.py ----

def eval_monad_atom(a):
    """

        @a                                                        [Atom]

        @ returns 1, if "a" is an atom and otherwise 0. All objects
        except for non-empty lists and non-empty strings are atoms.

        Examples:      @""  -->  1
                       @[]  -->  1
                      @123  -->  1
                  @[1 2 3]  -->  0

    """
    return kg_truth(is_atom(a))


def eval_monad_char(a, backend):
    """

        :#a                                                       [Char]

        Return the character at the code point "a".

        Monadic :# is an atomic operator.

        Examples: :#64  -->  0cA
                  :#10  -->  :"newline character"

    """
    if backend.is_backend_array(a):
        a = backend.to_numpy(a)  # code points are not differentiable; chr() needs plain integers
    return backend.rec_fn(a, lambda x: KGChar(chr(x))) if is_list(a) else KGChar(chr(a))


def eval_monad_enumerate(a, backend):
    """

        !a                                                   [Enumerate]

        Create a list of integers from 0 to a-1. !0 gives [].

        Examples: !0   -->  []
                  !1   -->  [1]
                  !10  -->  [0 1 2 3 4 5 6 7 8 9]

    """
    if is_dict(a):
        raise KlongTypeError("!: enumerate needs an integer, not a dictionary")
    if not backend.is_integer(a):
        from ..errors import compat, describe
        raise compat(KlongTypeError, RuntimeError)(f"!: enumerate needs an integer, not {describe(a)}")
    return backend.np.arange(int(a))


def eval_monad_expand_where(a, backend):
    """

        &a                                                [Expand/Where]

        Expand "a" to a list of subsequent integers X, starting at 0,
        where each XI is included aI times. When "a" is zero or an
        empty list, return nil. When "a" is a positive integer, return
        a list of that many zeros.

        In combination with predicates this function is also called
        Where, since it compresses a list of boolean values to indices,
        e.g.:

         [1 2 3 4 5]=[0 2 0 4 5]  -->  [0 1 0 1 1]
        &[1 2 3 4 5]=[0 2 0 4 5]  -->  [1 3 4]

        Examples:           &0  -->   []
                            &5  -->   [0 0 0 0 0]
                      &[1 2 3]  -->   [0 1 1 2 2 2]
                  &[0 1 0 1 0]  -->   [1 3]

    """
    arr = a if is_list(a) else [a]
    np_backend = backend.np
    if is_list(a) and len(a) == 0:
        # an empty list of any dtype (np.repeat rejects a float or object count array)
        return np_backend.arange(0)
    return np_backend.repeat(np_backend.arange(len(arr)), arr)


def eval_monad_first(a):
    """

        *a                                                       [First]

        Return the first element of "a", i.e. the first element of a
        list or the first character of a string. When "a" is an atom,
        return that atom.

        Examples:  *[1 2 3]  -->  1
                     *"abc"  --> 0ca
                        *""  -->  ""
                        *[]  -->  []
                         *1  -->  1

    """
    return a if is_empty(a) or not is_iterable(a) else a[0]


def eval_monad_floor(a, backend):
    """

        _a                                                       [Floor]

        Return "a" rounded toward negative infinity. When "a" is an
        integer, this is an identity operation. If "a" can be converted
        to integer without loss of precision after rounding, it will be
        converted. Otherwise, a floored real number will be returned.

        Note: loss of precision is predicted by comparing real number
        precision to the exponent, which is a conservative guess.

        Examples:   _123  -->  123
                  _123.9  -->  123
                  _1e100  -->  1.0e+100  :"if precision < 100 digits"

    """
    return backend.vec_fn(a, backend.floor_to_int)


def eval_monad_format(a, backend):
    """

        $a                                                      [Format]

        Write the external representation of "a" to a string and return
        it. The "external representation" of an object is the form in
        which Klong would display it.

        "$" is an atomic operator.

        Examples:    $123  -->  "123"
                  $123.45  -->  "123.45"
                  $"test"  -->  "test"
                     $0cx  -->  "x"
                    $:foo  -->  ":foo"

    """
    if isinstance(a, KGSym):
        return f":{a}"
    if is_dict(a):
        # the form in which it is displayed, :{[k v] ...}
        return kg_write(a, backend)
    if is_list(a):
        # "$" is atomic: format element by element (a numeric array holds atoms)
        return backend.kg_asarray([eval_monad_format(x, backend) for x in a])
    if backend.is_backend_array(a):
        a = backend.to_display(a)  # the number as it prints, not the repr of a 0-d tensor
    if isinstance(a, int) and not isinstance(a, bool):
        return kg_write_integer(a)  # an exact power can exceed Python's 4300-digit str() limit
    return str(a)


def eval_monad_grade_up(a, backend):
    """

        <a                                                    [Grade-Up]

        Impose the given order ("<" = ascending, ">" = descending") onto
        the elements of "a", which must be a list or string. Return a
        list of indices reflecting the desired order. Elements of "a"
        must be comparable by dyadic "<" (Less).

        In addition, "<" and ">" will compare lists by comparing their
        elements pairwise and recursively. E.g. [1 [2] 3] is considered
        to be "less" than [1 [4] 0], because 1=1 and 2<4 (3>0 does not
        matter, because 2<4 already finishes the comparison).

        When "a" is a string, these operators will grade its characters.

        To sort a list "a", use a@<a ("a" At Grade-Up "a") or a@>a.

        Examples:     <[1 2 3 4 5]  -->  [0 1 2 3 4]
                      >[1 2 3 4 5]  -->  [4 3 2 1 0]
                   <"hello, world"  -->  [6 5 11 1 0 2 3 10 8 4 9 7]
                    >[[1] [2] [3]]  -->  [2 1 0]

    """
    if isinstance(a, str) and a:
        return _grade_string(a, backend)
    return kg_argsort(backend.kg_asarray(a), backend)


def _grade_string(a, backend, descending=False):
    """The grade of a string by code point, ties by position (descending: the reverse of the
    ascending order), without making a list of characters."""
    return backend.grade_string(a, descending)


def eval_monad_grade_down(a, backend):
    """

        >a                                                  [Grade-Down]

        See [Grade-Up].

    """
    if isinstance(a, str) and a:
        return _grade_string(a, backend, descending=True)
    return kg_argsort(backend.kg_asarray(a), backend, descending=True)


def eval_monad_groupby(a, backend):
    """

        =a                                                       [Group]

        Return a list of lists ("groups") where each group contains the
        index of each occurrence of one element within "a". "a" must be
        a list or string. The indices of all elements of "a" that are
        equal according to "~" (Match) will appear in the same group in
        ascending order. The groups are ordered by the first appearance
        of their value in "a".

        ="" and =[] will yield [].

        Examples:   =[1 2 3 4]  -->  [[0] [1] [2] [3]]
                  ="hello foo"  -->  [[0] [1] [2 3] [4 7 8] [5] [6]]

    """
    arr = backend.kg_asarray(a)
    if backend.array_size(arr) == 0:
        return arr
    kind = backend.get_dtype_kind(arr)
    if kind in ('b', 'i', 'u', 'f', 'U') and getattr(arr, 'ndim', 1) == 1:
        # fast path: sort-based unique, then order the groups by first appearance
        arr_np = backend.to_numpy(arr) if backend.is_backend_array(arr) else arr
        _, first, inverse = bknp.unique(arr_np, return_index=True, return_inverse=True)
        inverse = inverse.reshape(-1)
        order = bknp.argsort(first, kind='stable')
        idx = bknp.argsort(inverse, kind='stable')
        parts = bknp.split(idx, bknp.cumsum(bknp.bincount(inverse))[:-1])
        groups = [parts[i] for i in order]
    else:
        # mixed, nested or ragged input: key on (type, value), dict keeps insertion order
        d = {}
        for i, x in enumerate(arr):
            d.setdefault(_value_key(x), []).append(i)
        groups = [bknp.asarray(v) for v in d.values()]
    return backend.kg_asarray(groups)


def _value_key(x):
    """Hashable key comparing by type and value (not by str()); NaN equals NaN."""
    if isinstance(x, KGChar):
        return ('c', str(x))
    if isinstance(x, str):
        return ('s', str(x))
    if isinstance(x, (bool, bknp.bool_, int, bknp.integer)):
        return ('i', int(x))
    if isinstance(x, (float, bknp.floating)):
        return ('f', 'nan' if x != x else float(x))
    if hasattr(x, 'tolist') and getattr(x, 'ndim', 1) == 0:
        return _value_key(x.tolist())
    if isinstance(x, (list, tuple)) or hasattr(x, '__len__') and hasattr(x, 'dtype'):
        return ('l', tuple(_value_key(e) for e in x))
    try:
        hash(x)
        return (type(x).__name__, x)
    except TypeError:
        return ('id', id(x))


def eval_monad_list(a, backend):
    """

        ,a                                                        [List]

        "," packages any object in a single-element list.

        Examples:    ,1  -->  [1]
                  ,:foo  -->  [:foo]
                  ,"xyz" -->  ["xyz"]
                   ,[1]  -->  [[1]]
    """
    if is_char(a):
        return str(a)
    return backend.kg_asarray([a])


def eval_monad_negate(a, backend):
    """

        -a                                                      [Negate]

        Return 0-a; "a" must be a number.

        "-" is an atomic operator.

        Examples:    -1  -->  -1
                  -1.23  -->  -1.23

    """
    return backend.vec_fn(a, lambda x: backend.np.negative(backend.kg_asarray(x)))


def eval_monad_not(a, backend):
    """

        ~a                                                         [Not]

        Return the negative truth value of "a", as explained in the
        section on CONDITIONALS. It will return 1 for 0, [], and "",
        and 0 for all other values.

        Examples:    ~0  -->  1
                     ~1  -->  0
                    ~[]  -->  1
                  ~:foo  -->  0

    """
    def _neg(x):
        if is_empty(x):
            return 1
        if is_dict(x) or isinstance(x, (KGFn, KGSym)):
            return 0
        if backend.is_backend_array(x):
            return kg_truth(backend.np.logical_not(x))
        arr = bknp.asarray(x)
        # Numeric/boolean input keeps a CLEAN (non-object) result so it can be
        # used as an index or by & (where). Forcing dtype=object here (as the old
        # code did unconditionally) tainted the result: logical_not on an object
        # array stays object, and then `&~v` / `RR@(&~v)` fail to cast O->int64.
        # Non-numeric input (strings/chars) still gets the object path so Python's
        # `not` semantics apply (e.g. ~"string" -> 0).
        if arr.dtype.kind not in ('b', 'i', 'u', 'f', 'c'):
            arr = bknp.asarray(x, dtype=object)
        return kg_truth(bknp.logical_not(arr))
    return backend.vec_fn(a, _neg) if not is_empty(a) else _neg(a)


def eval_monad_range(a, backend):
    """

        ?a                                                       [Range]

        Return a list containing unique elements from "a" in order of
        appearance. "a" may be a list or string.

        Examples:   ?[1 2 3 4]  -->  [1 2 3 4]
                  ?[1 1 1 2 2]  -->  [1 2]
                  ?"aaabbcccd"  -->  "abcd"

    """
    np_backend = backend.np
    if isinstance(a, str):
        return ''.join(dict.fromkeys(a))
    elif np_backend.isarray(a):
        dtype_kind = backend.get_dtype_kind(a)
        if dtype_kind != 'O' and a.ndim == 1:
            a_np = backend.to_numpy(a) if backend.is_backend_array(a) else a
            _, ids = bknp.unique(a_np, return_index=True)
            ids.sort()
            return a[ids]
        elif dtype_kind != 'O' and a.ndim > 1:
            # Use numpy for unique with return_index across backends
            a_np = backend.to_numpy(a) if backend.is_backend_array(a) else a
            _, ids = bknp.unique(a_np, axis=0, return_index=True)
            ids.sort()
            return a[ids]
        else:
            # handle the jagged / mixed array case
            # TODO: Make UNIQUE work. this feels so dirty.
            s = set()
            arr = []
            for x in a:
                sx = _value_key(x)
                if sx not in s:
                    s.add(sx)
                    arr.append(x)
            return backend.kg_asarray(arr)
    return a


def eval_monad_reciprocal(a, backend):
    """

        %a                                                  [Reciprocal]

        Return 1%a. "a" must be a number.

        "%" is an atomic operator.

        Examples:    %1  -->  1.0
                     %2  -->  0.5
                   %0.1  -->  10.0

    """
    if not is_list(a) and backend.is_number(a):
        a_val = backend.scalar_to_python(a) if backend.is_backend_array(a) else a
        if a_val == 0:
            return KLONG_UNDEFINED
    # 1%0 inside an array gives inf; numpy would warn, so scope the silence to this call
    with bknp.errstate(divide='ignore'):
        return backend.vec_fn(a, backend.reciprocal)


def eval_monad_reverse(a, backend):
    """

        |a                                                     [Reverse]

        Return a new list/string that contains the elements of "a" in
        reverse order. When "a" is neither a list nor a string, return
        it unchanged.

        Examples:       |[1 2 3]  -->  [3 2 1]
                  |"hello world"  -->  "dlrow olleh"
                              |1  -->  1

    """
    if is_atom(a):
        return a
    if backend.is_backend_array(a):
        np_mod = backend.np
        if hasattr(np_mod, 'flip'):
            try:
                return np_mod.flip(a, dims=[0])  # torch style
            except TypeError:
                return np_mod.flip(a, axis=0)  # numpy style
    return a[::-1]


def eval_monad_shape(a, backend):
    """

        ^a                                                       [Shape]

        Return the shape of "a". The shape of an atom is 0. The shape of
        a list L of atoms is ,#L. Such a list is also called a 1-array
        or a vector. The shape of a list of lists of equal length (M) is
        (#M),#*M. Such a list is called a 2-array or a matrix. A list of
        lists of unequal length is a vector.

        This principle is extended to higher dimensions. An N-array A is
        is an array with equal-sized sub-arrays in each dimension. Its
        shape is (#A),(#*A),...,(#*...*A), where there are N-1 "*"
        operators in the last group of that expression. All shapes are
        written in row-major notation.

        For example:

        [1 2 3 4 5]    is a vector (shape [5])

        [[1 2]
         [2 4]
         [5 6]]        is a matrix (shape [3 2])

        [[[1 2 3 4]
          [5 6 7 8]]
         [[9 0 1 2]
          [3 4 5 6]]
         [[7 8 9 0]
          [1 2 3 4]]]  is a 3-array (shape [3 2 4])

        [[1] [2 3]]    is a vector (shape [2])

        The shape of a string S is ,#S. A list of equally-sized strings
        is a matrix of characters. Strings may form the innermost level
        of higher-dimensional arrays.

        Examples:        ^1  -->  0
                      ^:xyz  -->  0
                       ^[0]  -->  [1]
                   ^[1 2 3]  -->  [3]
                   ^"hello"  -->  [5]
                   ^[[1 2]
                     [3 4]
                     [5 6]]  -->  [3 2]
                   ^[1 [2]]  -->  [2]
                  ^["abcd"
                    "efgh"]  -->  [2 4]

    """

    def _normalize_backend_array(x):
        return backend.to_numpy(x) if backend.is_backend_array(x) else x

    def _shape(x):
        # shape of nested lists/strings; unequal-length sub-lists make a vector
        x = _normalize_backend_array(x)
        if isinstance(x, str):
            return [len(x)] if not isinstance(x, (KGSym, KGChar)) else []
        if not is_list(x):
            return []
        if getattr(x, 'dtype', None) is not None and x.dtype != object:
            return list(x.shape)
        n = len(x)
        if n == 0:
            return [0]
        subs = [_shape(y) for y in x]
        first = subs[0]
        if first and all(sh == first for sh in subs):
            return [n] + first
        return [n]

    a = _normalize_backend_array(a)
    if is_atom(a):
        return 0
    return backend.np.asarray(_shape(a), dtype=int)


def eval_monad_size(a, backend):
    """

        #a                                                        [Size]

        Return the size/magnitude of "a".

        For lists, the size of "a" is the number of its elements.
        For strings, the size is the number of characters.
        For numbers, the size is the magnitude (absolute value).
        For characters, the size is the ASCII code.

        Examples:     #[1 2 3]  -->  3
                  #[1 [2 3] 4]  -->  3
                  #"123456789"  -->  9
                         #-123  -->  123
                          #0cA  -->  65

    """
    return backend.np.abs(a) if backend.is_number(a) else ord(a) if is_char(a) else len(a)


def eval_monad_transpose(a, backend):
    """

        +a                                                   [Transpose]

        Return the transpose of the matrix (2-array) "a".

        Examples:     +[[1] [2] [3]]  -->  [[1 2 3]]
                  +[[1 2 3] [4 5 6]]  -->  [[1 4] [2 5] [3 6]]
                                 +[]  -->  []

    """
    np_backend = backend.np
    return np_backend.transpose(a if np_backend.isarray(a) else np_backend.asarray(a))


def eval_monad_undefined(a, backend):
    """

        :_a                                                  [Undefined]

        Return truth, if "a" is undefined, i.e. the result of an
        operation that cannot yield any meaningful result, like
        division by zero or trying to find a non-existent key in
        a dictionary. Else return 0.

        Examples:        :_1%0  -->  1
                  :_:{[1 2]}?3  -->  1
                      :_:valid  -->  0

    """
    return kg_truth(a is None or a is KLONG_UNDEFINED)


def eval_monad_track(a):
    """

        ˙a                                                    [Track]

        Identity: returns "a". Internal. It is reserved for marking a value
        for gradient tracking and currently has no effect. It has no ASCII
        form.

    """
    return a


# ---- from dyads.py ----

def eval_dyad_add(a, b, backend):
    """

        a+b                                                       [Plus]

        Add "b" to "a" and return the result. "a" and "b" must both be
        numbers.

        Dyadic "+" is an atomic operator.

        Integers are 64 bits wide and a sum that does not fit wraps
        around, but if an operand is an integer outside 64 bits (such
        as 9223372036854775808) the sum is exact. The same holds for
        "-", "*", "<", ">", "|" and "&".

        Examples:  12+3  -->  15
                  12+-3  -->  9
                  1+0.3  -->  1.3

    """
    return backend.vec_fn2(a, b, backend.np.add)


def eval_dyad_amend(a, b, backend):
    """

        a:=b                                                     [Amend]

        "a" must be a list or string and "b" must be a list where the
        first element can have any type and the remaining elements must
        be integers. It returns a new object of a's type where a@b2
        through a@bN are replaced by b1. When "a" is a string, b1 must
        be a character or a string. The first element of "a" has an
        index of 0.

        When both "a" and b1 are strings, Amend replaces each substring
        of "a" starting at b2..bN by b1. Note that no index b2..bN must
        be larger than #a or a range error will occur. When b1 is
        replaced at a position past (#a)-#b1, the amended string will
        grow by the required amount. For instance:

        "aa":="bc",1 --> "abc".

        Examples:    "-----":=0cx,[1 3]  -->  "-x-x-"
                           [1 2 3]:=0,1  -->  [1 0 3]
                  "-------":="xx",[1 4]  -->  "-xx-xx-"
                         "abc":="def",3  -->  "abcdef"

    """
    np_backend = backend.np
    if not (isinstance(a, (str,list)) or np_backend.isarray(a)):
        from ..errors import compat
        raise compat(KlongTypeError, RuntimeError)(f":=: a must be list or str: {a}")
    if len(b) <= 1:
        return a
    if isinstance(a, str):
        if not isinstance(b[0], str):
            raise KlongTypeError(f":=: value for a string must be a character or string, got {b[0]!r}")
        n = len(a)
        r = a
        for i in b[1:]:
            i = _amend_index(i, n, allow_end=True)
            r = r[:i] + b[0] + r[i+len(b[0]):]
        return str(r)
    r = np_backend.array(a) # clone
    n = len(r)
    idx = [_amend_index(i, n, allow_end=False) for i in b[1:]]
    if is_list(b[0]): # TOOD: use bknp.put if we can
        r = r.tolist()
        for i in idx:
            r[i] = b[0]
        r = backend.kg_asarray(r)
    else:
        r = backend.put(_amend_promote(r, b[0]), idx, b[0])
    return r


def _amend_index(i, n, allow_end):
    """Normalise an amend index: -n..n-1 (and n when allow_end) are valid."""
    if getattr(i, 'ndim', None) == 0 and hasattr(i, 'item'):
        i = i.item()
    if isinstance(i, (str, bool)) or not isinstance(i, numbers.Real) or i != i or i != int(i):
        raise KlongTypeError(f":=: index must be an integer, got {i!r}")
    i = int(i)
    j = i + n if i < 0 else i
    if j < 0 or j > n or (j == n and not allow_end):
        raise KlongRangeError(f":=: index {i} out of range for length {n}")
    return j


def _amend_promote(r, v):
    """Widen the dtype of r when v cannot be held by it (so put does not truncate)."""
    kind = getattr(getattr(r, 'dtype', None), 'kind', None)
    if kind is None:
        return r
    if isinstance(v, (numbers.Integral, numpy.bool_)):
        return r
    if isinstance(v, numbers.Real):
        return r.astype(float) if kind in 'iub' else r
    return r.astype(object) if kind != 'O' else r


def _e_dyad_amend_in_depth(p, q, v):
    if bknp.isarray(q) and len(q) > 1:
        r = _e_dyad_amend_in_depth(p[q[0]], q[1:] if len(q) > 2 else q[1], v)
        p = bknp.array(p, dtype=r.dtype)
        p[q[0]] = r
        return p
    else:
        # a float value into an integer array: the array becomes a float array
        p = bknp.array(p, dtype=object) if isinstance(v, (str, KGSym)) else _amend_promote(bknp.array(p), v)
        p[q] = v
        return p


def eval_dyad_amend_in_depth(a, b, backend):
    """

        a:-b                                            [Amend-in-Depth]

        :- is like :=, but "a" may be a multi-dimensional array. The :-
        operator replaces one single element in that array. The sequence
        of indices b1..bN is used to locate the target element in an
        N-dimensional array. The number of indices must match the rank
        of the array.

        Example: [[1 2] [3 4]]:-42,[0 1]  -->  [[1 42] [3 4]]
                      [[[0]]]:-1,[0 0 0]  -->  [[[1]]]

    """
    if backend.is_backend_array(a):
        return backend.amend_at(a, b[1:], b[0])
    if backend.is_backend_array(b):
        b = backend.to_numpy(b)
    v, q = b[0], b[1:]
    if backend.get_dtype_kind(b) == 'f':
        # A float value joined to the indices made one float list (5.5,[0 1]):
        # the indices are integers again, and the value a float for the
        # promotion of an integer array.
        if (q != q.astype(int)).any():
            raise KlongTypeError(f":-: index must be an integer, got {q[q != q.astype(int)][0]!r}")
        q = q.astype(int)
        v = float(v)
    return _e_dyad_amend_in_depth(a, q, v)


def eval_dyad_cut(a, b, backend):
    """

        a:_b                                                       [Cut]

        Cut the list "b" before the elements at positions given in "a".
        "a" must be an integer or a list of integers. When it is a list
        of integers, its elements must be in monotonically increasing
        order. :_ returns a new list containing consecutive segments of
        "b".

        When "a" is zero or #b or contains two subsequent equal indices,
        nil (or an empty string if "b" is a string) will be inserted.

        Examples:      2:_[1 2 3 4]  -->  [[1 2] [3 4]]
                  [2 3 5]:_"abcdef"  -->  ["ab" "c" "de" "f"]
                             0:_[1]  -->  [[] [1]]
                           3:_"abc"  -->  ["abc" ""]
                       [1 1]:_[1 2]  -->  [[1] [] [2]]

    """
    j = isinstance(b, str)
    np_backend = backend.np
    b = backend.str_to_chr_arr(b) if j else b
    b = b if np_backend.isarray(b) else np_backend.asarray(b)
    a = a if np_backend.isarray(a) and a.ndim > 0 else [a]
    r = backend.array_split(b, a)
    if len(b) == 0 and len(a) > 0:
        r = r[1:]
    return bknp.asarray(["".join(x) for x in r]) if j else backend.kg_asarray(r)


def _normalize_index(i, n, what='list'):
    """a@i accepts -n <= i < n; a negative index counts from the end."""
    if getattr(i, 'ndim', None) == 0 and hasattr(i, 'item'):
        i = i.item()
    if not isinstance(i, numbers.Integral):
        return i  # not an integer index: let the container decide (e.g. nested lists)
    i = int(i)
    if i < -n or i >= n:
        from ..errors import compat
        # also an IndexError, which is what an index out of range was before
        raise compat(KlongRangeError, IndexError)(f"index {i} out of range for {what} of {n}")
    return i + n if i < 0 else i


def _dict_lookup(d, k, backend):
    """The value of key k in the dictionary d, :undefined when there is none (Find and At)."""
    v = d.get(backend.dict_key(k))
    return KLONG_UNDEFINED if v is None else v


def _is_int_vector(b, backend):
    """True for a non-empty one-dimensional integer array (not a boolean mask, not a scalar)."""
    return (backend.is_array(b) and b.ndim == 1 and len(b) > 0
            and backend.get_dtype_kind(b) in ('i', 'u'))


def _take_by_index(a, b, backend):
    """a[b] for a numeric array a and an integer vector b with -len(a) <= b < len(a), else None
    (the caller then reports the error the slow way)."""
    n = len(a)
    lo, hi = int(b.min()), int(b.max())
    if lo < -n or hi >= n:
        return None
    if lo < 0:
        b = backend.np.where(b < 0, b + n, b)
    if backend.get_dtype_kind(a) == 'O':
        # members of any type: the result is typed from its members, as for any list
        if a.ndim != 1:
            return None
        if backend.is_backend_array(b):
            b = backend.to_numpy(b)
        return backend.kg_asarray(a[b].tolist())
    return a[b]


def _index_string(a, b, backend):
    """a@b for a string a, indexing the string directly (code points, O(1) per index): a character
    for an integer index, a string of characters for an integer vector. None for anything else
    (nested or non-integer indices and errors), which the caller handles the general way."""
    n = len(a)
    if backend.is_integer(b):
        i = _normalize_index(b, n, 'string')
        return KGChar(a[i])
    if _is_int_vector(b, backend):
        idx = b.tolist()
        if min(idx) < -n or max(idx) >= n:
            return None
        return "".join([a[i] for i in idx])
    if (backend.is_array(b) and b.ndim == 2 and b.shape[0] > 0 and b.shape[1] > 0
            and backend.get_dtype_kind(b) in ('i', 'u')):
        rows = b.tolist()
        lo, hi = min(map(min, rows)), max(map(max, rows))
        if lo < -n or hi >= n:
            return None
        return backend.object_array(["".join([a[i] for i in row]) for row in rows])
    return None


def eval_dyad_at_index(klong, a, b):
    """

        a@b                                                   [At/Index]
        a@b                                                   [At/Apply]

        Extract one or multiple elements from "a" at (zero-based)
        positions given in "b". In this case "a" may be a list or a
        string.

        When "b" is an integer, extract a single element at the given
        position and return it.

        When "b" is a list, return a list containing the extracted
        elements. All members of "b" must be integers in this case.
        The order of indices in "b" does not matter. The same index
        may occur multiple times.

        Indices -n..n-1 are valid for a list or string of length n; a
        negative index counts from the end. Any other index is an error.

        When "a" is a dictionary, return the value associated with the
        key "b", as a?b does, and :undefined when there is no such key.
        When "b" is a list, return the list of the values of its members.

        When "a" is a function, "b" (if it is an atom) or the members
        of "b" (if it is a list) will be passed as arguments to the
        function and the result will be returned.

        Examples:         [1 2 3 4 5]@2  -->  3
                    [1 2 3 4 5]@[1 2 3]  -->  [2 3 4]
                    [1 2 3 4 5]@[0 0 0]  -->  [1 1 1]
                  "hello world"@[3 7 2]  -->  "lol"
                         [1 2 3 4 5]@-1  -->  5
                         {y+x*x}@[2 3]   -->  7
                        :{["k" 1]}@"k"   -->  1

    """
    if isinstance(a, (KGFn, KGSym)) or issubclass(type(a), KGLambda):
        # b holds values: a list supplies one argument per member, an atom is
        # the single argument. apply() binds them without evaluating them again.
        b = [x for x in b] if klong._backend.is_array(b) and getattr(b, 'ndim', 1) > 0 else b
        return klong.apply(a, b if isinstance(b, list) else [b], 1)
    backend = klong._backend
    if is_dict(a):
        # A dictionary (or a table): a@b is the lookup a?b for every kind of
        # key, and :undefined for a key that is not there. A list is one
        # lookup per member (a list cannot be a key).
        if is_list(b):
            if is_empty(b):
                return backend.np.asarray([])
            return backend.kg_asarray([_dict_lookup(a, x, backend) for x in b])
        return _dict_lookup(a, b, backend)
    j = isinstance(a,str)
    if j:
        fast = _index_string(a, b, backend)
        if fast is not None:
            return fast
    elif _is_int_vector(b, backend) and backend.is_array(a) and a.ndim > 0 and backend.get_dtype_kind(a) in ('i', 'f', 'O'):
        # an array indexed by an integer vector within range: one take
        fast = _take_by_index(a, b, backend)
        if fast is not None:
            return fast
    if j and backend.is_array(b) and getattr(b, 'ndim', 0) >= 3:
        # a string takes a vector of indexes or a matrix of them, no deeper
        raise KlongTypeError(f"cannot index a string by a list nested {b.ndim} deep")
    a = backend.str_to_chr_arr(a) if j else a
    if j and backend.is_backend_array(b):
        b = backend.to_numpy(b)   # the characters are a NumPy array: a tensor on a device cannot index it
    n = None if not hasattr(a, '__len__') else len(a)
    if is_list(b):
        if is_empty(b):
            r = backend.np.asarray([])
        else:
            for x in b:
                if not (is_list(x) or backend.is_integer(x)):
                    # an index vector holds whole numbers (a string, a symbol, a float)
                    raise KlongRangeError(f"index vector holds {type(x).__name__}, not a whole number")
            if n is not None:
                b = [_normalize_index(x, n, 'string' if j else 'list') for x in b]
            r = backend.kg_asarray([a[x] for x in b])
    elif backend.is_integer(b):
        r = a[b if n is None else _normalize_index(b, n, 'string' if j else 'list')]
        j = False
    else:
        r = a
    if j:
        if backend.np.isarray(r) and r.ndim > 1:
            return bknp.asarray(["".join(x) for x in r], dtype=object)
        return "".join(r)
    return r


def _undefined_on_zero_divisor(fn, backend):
    """Wrap an elementwise numeric f(x, y) so that positions where y == 0 give :undefined.

    A scalar zero divisor gives :undefined itself; for arrays the result becomes
    a mixed (object) list with :undefined at the offending positions.  When there
    is no zero divisor the fast path and the dtype are unchanged.  Backends that
    cannot hold :undefined in an array (torch) keep their IEEE results.
    """
    if not backend.supports_object_dtype():
        def ieee(x, y):
            # an atom divided by the atom 0 is :undefined on every backend; a
            # tensor keeps the IEEE inf/nan (it cannot hold :undefined) and
            # stays in the autograd graph
            if getattr(y, 'ndim', 0) == 0 and getattr(x, 'ndim', 0) == 0 and backend.is_number(y) and y == 0:
                return KLONG_UNDEFINED
            return fn(x, y)
        return ieee
    np_ = backend.np

    def g(x, y):
        yn = np_.asarray(y)
        if yn.dtype.kind not in 'biuf':
            return fn(x, y)
        zero = yn == 0
        if not zero.any():
            return fn(x, y)
        with numpy.errstate(all='ignore'):
            r = fn(x, np_.where(zero, 1, yn))
        if numpy.ndim(r) == 0:
            return KLONG_UNDEFINED
        o = np_.asarray(r).astype(object)
        o[numpy.broadcast_to(zero, o.shape)] = KLONG_UNDEFINED
        return o
    g.object_safe = True  # see BackendProvider._exact_atomic; fn is np.divide or np.fmod
    return g


def _ordering(py_cmp, ufunc):
    """Atomic < or >: strings compare with Python, everything else with the numpy ufunc.

    `object_safe`: when an operand is an integer outside int64 (numpy raises
    OverflowError) vec_fn2 repeats the call on Python integers.
    """
    def f(x, y):
        return py_cmp(x, y) if (isinstance(x, str) and isinstance(y, str)) else ufunc(x, y)
    f.object_safe = True
    return f


def eval_dyad_divide(a, b, backend):
    """

        a%b                                                     [Divide]

        Return the quotient of "a" and "b". The result is always a real
        number, even if the result has a fractional part of 0.

        "%" is an atomic operator.

        Examples: 10%2  -->  5.0
                  10%8  -->  1.25

    """
    if not is_list(a) and not is_list(b) and backend.is_number(b):
        b_val = backend.scalar_to_python(b) if backend.is_backend_array(b) or (hasattr(b, 'ndim') and b.ndim == 0) else b
        if b_val == 0:
            return KLONG_UNDEFINED
    return backend.vec_fn2(a, b, _undefined_on_zero_divisor(backend.np.divide, backend))


def eval_dyad_drop(a, b, backend):
    """

        a_b                                                       [Drop]

        When "b" is a list or string, drop "a" elements or characters
        from it, returning the remaining list. Dropping more elements
        than contained in "b" will yield the empty list/string. A
        negative value for "a" will drop elements from the end of "b".

        When "b" is a dictionary, remove the entry with the key "a" from
        it. Dictionary removal is in situ, i.e. the dictionary will be
        modified. Other objects will be copied.

        Examples: 3_[1 2 3 4 5]  -->  [4 5]
                  (-3)_"abcdef"  -->  "abc"
                     17_[1 2 3]  -->  []
                       (-5)_"x"  -->  ""

    """
    if is_dict(b):
        try:
            del b[backend.dict_key(a)] # biased towards presence perf
        except KeyError:
            pass
        return b
    return b[a:] if a >= 0 else b[:a]


def _safe_equal(x, y, backend):
    """Compare two values, handling backend arrays correctly."""
    return kg_truth(backend.safe_equal(x, y))


def eval_dyad_equal(a, b, backend):
    """

        a=b                                                      [Equal]

        Return 1, if "a" and "b" are equal, otherwise return 0.

        Numbers are equal, if they have the same value.
        Characters are equal, if (#a)=#b.
        Strings and symbols are equal, if they contain the same
        characters in the same positions.

        "=" is an atomic operator. In particular it means that it
        cannot compare lists, but only elements of lists. Use "~"
        (Match) to compare lists.

        Real numbers should not be compared with "=". Use "~" instead.

        Examples:             1=1  -->  1
                      "foo"="foo"  -->  1
                        :foo=:foo  -->  1
                          0cx=0cx  -->  1
                  [1 2 3]=[1 4 3]  -->  [1 0 1]

    """
    return backend.vec_fn2(a, b, lambda x, y: _safe_equal(x, y, backend))


def finditer(s, sub):
    i = 0
    while True:
        i = s.find(sub, i)
        if i == -1:
            break
        yield i
        i += 1


def eval_dyad_find(a, b, backend):
    """

        a?b                                                       [Find]

        Find each occurrence of "b" in "a". "a" must be a list, string,
        or dictionary. When "a" is a dictionary, return the value
        associated with the given key. When "a" is a list or string,
        return a list containing the position of each match.

        When both "a" and "b" are strings, return a list containing each
        position of the substring "b" inside of "a". The empty string ""
        is contained between any two characters of a string, even before
        the first and after the last character.

        In any case a return value of nil indicates that "b" is not
        contained in "a", except when "a" is a dictionary. When a key
        cannot be found in a dictionary, Find will return :undefined.
        (See [Undefined].)

        Examples: [1 2 3 1 2 1]?1  -->  [0 3 5]
                        [1 2 3]?4  -->  []
                      "hello"?0cl  -->  [2 3]
                    "xyyyyz"?"yy"  -->  [1 2 3]
                            ""?""  -->  [0]
                      :{[1 []]}?1  -->  []

    """
    if isinstance(a,str):
        return backend.np.asarray(list(finditer(a,str(b))))
    elif is_dict(a):
        v = a.get(backend.dict_key(b))
        return KLONG_UNDEFINED if v is None else v
    if is_list(b):
        return backend.np.asarray([i for i,x in enumerate(a) if backend.kg_equal(x, b)])
    return backend.find_equal(a, b)


def __e_dyad_form(a, b, backend):
    if isinstance(a,KGSym):
        if is_empty(b):
            return KLONG_UNDEFINED
        return KGSym(b[1:] if isinstance(b,str) and b.startswith(":") else b)
    if backend.is_integer(a):
        if not isinstance(b, str) or is_empty(b):
            return KLONG_UNDEFINED
        try:
            return int(b)
        except ValueError:
            return KLONG_UNDEFINED
    if backend.is_float(a):
        if is_empty(b):
            return KLONG_UNDEFINED
        try:
            return float(b)
        except (ValueError, TypeError):
            return KLONG_UNDEFINED
    if isinstance(a,KGChar):
        b = str(b)
        if len(b) != 1:
            return KLONG_UNDEFINED
        return KGChar(str(b)[0])
    return b


def _e_dyad_form(a, b, backend):
    """
    Unravel the broadcasting of a and b and apply __e_dyad_form
    """
    if backend.np.isarray(a) and backend.np.isarray(b) and a.ndim > 0 and b.ndim > 0:
        return backend.np.asarray([backend.vec_fn2(x, y, lambda x, y: _e_dyad_form(x, y, backend)) for x,y in zip(a,b)])
    if backend.np.isarray(b) and b.ndim > 0 and not backend.np.isarray(a):
        # numeric list on the right (e.g. 1:$[5]): convert element by element
        return backend.kg_asarray([_e_dyad_form(a, x, backend) for x in b])
    return __e_dyad_form(a, b, backend)


def eval_dyad_form(a, b, backend):
    """

        a:$b                                                      [Form]

        Convert string "b" to the type of the object of "a". When "b"
        can be converted to the desired type, an object of that type
        will be returned. When such a conversion is not possible, :$
        will return :undefined.

        When "a" is an integer, "b" may not represent a real number.
        When "a" is a real number, a real number will be returned, even
        if "b" represents an integer. When "a" is a character, "b" must
        contain exactly one character. When "a" is a symbol, "b" must
        contain the name of a valid symbol (optionally including a
        leading ":" character).

        :$ is an atomic operator.

        Examples:     1:$"-123"  -->  -123
                    1.0:$"1.23"  -->  1.23
                       0c0:$"x"  -->  0cx
                   "":$"string"  -->  "string"
                   :x:$"symbol"  -->  :symbol
                  :x:$":symbol"  -->  :symbol

    """
    return backend.vec_fn2(a, b, lambda x, y: _e_dyad_form(x, y, backend))


def __e_dyad_format2(a, b, backend):
    if type(a) is float or (hasattr(a, 'ndim') and a.ndim == 0):
        a = backend.to_display(a)  # .item(), or the float32 number as it prints: 8.2, not 8.199999809265137
    if type(b) is float or (hasattr(b, 'ndim') and b.ndim == 0):
        b = backend.to_display(b)
    if isinstance(a, float) and not (isinstance(b, float) and not isinstance(b, bool)):
        raise KlongTypeError(f"$: a real width needs a real number to format, got {b!r}")
    if safe_eq(int(a), 0):
        return str(b)
    if (backend.is_float(b) and not isinstance(b,int)) and (backend.is_float(a) and not isinstance(a,int)):
        b = "{:Xf}".replace("X",str(a)).format(b)
        p = b.split('.')
        p[0] = p[0].rjust(int(a))
        b = ".".join(p)
        return b
    b = f":{b}" if isinstance(b, KGSym) else b
    r = str(b).ljust(abs(a)) if a >= 0 else str(b).rjust(abs(a))
    return r


def _e_dyad_format2(a, b, backend):
    """
    Unravel the broadcasting of a and b and apply __e_dyad_format2
    """
    if is_list(a) and is_list(b):
        return backend.kg_asarray([backend.vec_fn2(x, y, lambda x, y: _e_dyad_format2(x, y, backend)) for x, y in zip(to_list(a), to_list(b))])
    if backend.np.isarray(a) and backend.np.isarray(b) and a.ndim > 0 and b.ndim > 0:
        return backend.np.asarray([backend.vec_fn2(x, y, lambda x, y: _e_dyad_format2(x, y, backend)) for x, y in zip(a, b)])
    return __e_dyad_format2(a, b, backend)


def eval_dyad_format2(a, b, backend):
    """

        a$b                                                    [Format2]

        Dyadic "$" is like its monadic cousin, but also pads its result
        with blanks. The minimal size of the output string is specified
        in "a", which must be an integer. "b" is the object to format.
        When the value of "a" is negative, the result string is padded
        to the left, else it is padded to the right.

        When "a" is real number of the form n.m and "b" is also a real
        number, the representation of "b" will have "n" integer digits
        and "m" fractional digits. The integer part will be padded with
        blanks and the fractional part will be padded with zeros.

        "$" is an atomic operator.

        Examples:     0$123  -->  "123"
                  (-5)$-123  -->  " -123"
                    5$"xyz"  -->  "xyz  "
                  (-5)$:foo  -->  " :foo"
                 5.3$123.45  -->  "  123.450"

    """
    return backend.vec_fn2(a, b, lambda x, y: _e_dyad_format2(x, y, backend))


def eval_dyad_index_in_depth(a, b, backend):
    """

        a:@b                                            [Index-in-Depth]

        :@ is like "@" but, when applied to an array, extracts a single
        element from a multi-dimensional array. The indices in "b" are
        used to locate the element. The number of indices must match
        the rank of the array.

        If "a" is a function, :@ is equal to "@".

        Examples: [[1 2] [3 4]]:@[0 1]  -->  2
                      [[[1]]]:@[0 0 0]  -->  1
                        {y+x*x}:@[2 3]  -->  7

    """
    return backend.index_at(a, b) if not is_empty(b) else b


def _e_dyad_integer_divide(x, y, backend):
    np_backend = backend.np
    a = np_backend.divide(x, y)
    a = backend.kg_asarray(backend.rec_fn(a, np_backend.trunc)) if np_backend.isarray(a) else a
    return backend.to_int_array(a)


def eval_dyad_integer_divide(a, b, backend):
    """

        a:%b                                            [Integer-Divide]

        Return the integer part of the quotient of "a" and "b". Both "a"
        and "b" must be integers. The result is always an integer.

        Formally, a = (b*a:%b) + a!b .

        ":%" is an atomic operator.

        Examples: 10:%2  -->  5
                  10:%8  -->  1

    """
    if not is_list(a) and not is_list(b) and backend.is_number(b):
        b_val = backend.scalar_to_python(b) if backend.is_backend_array(b) or (hasattr(b, 'ndim') and b.ndim == 0) else b
        if b_val == 0:
            return KLONG_UNDEFINED
    return backend.vec_fn2(a, b, _undefined_on_zero_divisor(lambda x, y: _e_dyad_integer_divide(x, y, backend), backend))


def _arr_to_list(a):
    """ Convert a to a list if it is not already one """
    return a if is_list(a) else [a]# if not is_list(a) else a


def _numeric_atom_vector(x, backend):
    """A numeric atom (an int64-range integer, a float, a numeric 0-d array) as a one-element array,
    else None. Booleans and big integers are not (they take the general path)."""
    if type(x) is int:
        return backend.np.asarray([x]) if -2**63 <= x < 2**63 else None
    if isinstance(x, numbers.Real) and not isinstance(x, bool):
        return backend.np.asarray([x])
    if backend.is_array(x) and x.ndim == 0 and backend.get_dtype_kind(x) in ('i', 'f'):
        return x.reshape(1)
    return None


def _join_numeric_atom(a, b, backend):
    """a,b for a numeric vector and a numeric atom (either order): one concatenation instead of
    a round trip through a list of elements. None when the pair is anything else."""
    if backend.is_array(a) and a.ndim == 1 and len(a) > 0 and backend.get_dtype_kind(a) in ('i', 'f'):
        v = _numeric_atom_vector(b, backend)
        if v is not None:
            return backend.np.concatenate((a, v))
    elif backend.is_array(b) and b.ndim == 1 and len(b) > 0 and backend.get_dtype_kind(b) in ('i', 'f'):
        v = _numeric_atom_vector(a, backend)
        if v is not None:
            return backend.np.concatenate((v, b))
    return None


def eval_dyad_join(a, b, backend):
    """

        a,b                                                       [Join]

        The "," operator joins objects of any type, forming lists or
        strings.

        If "a" and "b" are lists, append them.
        If "a" is a list and "b" is not, attach "b" at the end of "a".
        If "a" is a not list and "b" is one, attach "a" to the front of
        "b".
        If "a" and "b" are strings, append them.
        If "a" is a string and "b" is a char, attach "b" to the end of
        "a".
        If "a" is a char and "b" is a string, attach "a" to the front of
        "b".

        If "a" is a dictionary and "b" is a tuple (a list of two members)
        or vice versa, add the tuple to the dictionary. Any entry with
        the same key will be replaced by the new entry. The head of the
        tuple is the key and the second element is the payload.

        If "a" and "b" are dictionaries, add the entries of "b" to "a"
        and return "a". An entry of "b" replaces an entry of "a" with
        the same key.

        Otherwise, create a tuple containing "a" and "b" in that order.

        Join always returns a fresh list, but dictionaries will be
        updated by replacing old entries in situ.

        Examples:  [1 2 3],[4 5 6]  -->  [1 2 3 4 5 6]
                           1,[2 3]  -->  [1 2 3]
                           [1 2],3  -->  [1 2 3]
                       "abc","def"  -->  "abcdef"
                          "ab",0cc  -->  "abc"
                          0ca,"bc"  -->  "abc"
                               1,2  -->  [1 2]
                             "a",1  -->  ["a" 1]
                       [[1 2 3]],4  -->  [[1 2 3] 4]
                           1,2,3,4  -->  [1 2 3 4]
                    [1 2],:{[1 0]}  -->  :{[1 2]}
                    :{[1 0]},[1 2]  -->  :{[1 2]}
                  :{[1 2]},:{[3 4]}  -->  :{[1 2] [3 4]}

    """
    if (isinstance(a,str) and not isinstance(a,KGSym)) and (isinstance(b,str) and not isinstance(b,KGSym)):
        return a+b
    if isinstance(a,dict):
        if isinstance(b,dict):
            # merge: the entries of b go into a, b winning; a is updated in situ
            for k,v in b.items():
                a[k] = v
            return a
        if not (is_list(b) and len(b) == 2):
            raise KlongTypeError(",: a dictionary can only be joined with a tuple (a list of two members) "
                                 "or a dictionary",
                                 hint="d,[key value] adds an entry and d,d2 merges; to put a dictionary "
                                      "into a list write list,,d")
        a[backend.dict_key(b[0])] = b[1]
        return a
    if isinstance(b,dict) and is_list(a) and len(a) == 2:
        b[backend.dict_key(a[0])] = a[1]
        return b

    fast = _join_numeric_atom(a, b, backend)
    if fast is not None:
        return fast

    if backend.np.isarray(a) and backend.np.isarray(b):
        # Only use fast path for 1D+ arrays (not 0D scalars)
        a_is_1d_plus = hasattr(a, 'ndim') and a.ndim >= 1
        b_is_1d_plus = hasattr(b, 'ndim') and b.ndim >= 1
        if a_is_1d_plus and b_is_1d_plus:
            if len(a) == 0:
                return b
            if len(a.shape) == len(b.shape) and a.shape[-1] == b.shape[-1]:
                return backend.np.concatenate((a,b))

    aa = _arr_to_list(a)
    bb = _arr_to_list(b)

    r = [*aa,*bb]
    nr = backend.kg_asarray(r)
    # Check dtype kind for compatibility across backends
    dtype_kind = backend.get_dtype_kind(nr)
    if dtype_kind in ('i', 'f', 'u'):
        return nr
    # Use numpy directly for object arrays (backends without object dtype need this)
    # Convert backend arrays to numpy first (needed for device-backed arrays)
    r_numpy = [backend.to_numpy(x) if backend.is_array(x) else x for x in r]
    try:
        return numpy.asarray(r_numpy, dtype=object)
    except ValueError:
        # members that are arrays of different shape (a vector and a matrix)
        # do not broadcast into one array: a list of them, member by member
        return backend.object_array(r_numpy)


def eval_dyad_less(a, b, backend):
    """

        a<b                                                       [Less]

        Return 1, if "a" is less than "b", otherwise return 0.

        Numbers are compared by value.
        Characters are compared by ASCII code.
        Strings and symbols are compared lexicographically.

        "<" is an atomic operator; it cannot compare lists, but only
        elements of lists.

        Examples:              1<2  -->  1
                       "bar"<"foo"  -->  1
                         :abc<:xyz  -->  1
                           0c0<0c9  -->  1
                   [1 2 3]<[1 4 3]  -->  [0 1 0]

    """
    return kg_truth(backend.vec_fn2(a, b, _ordering(operator.lt, backend.np.less)))


def eval_dyad_match(a, b, backend):
    """

        a~b                                                      [Match]

        "~" is like "=", but can also compare lists and real numbers. It
        uses "=" (Equal) to compare integers, characters, symbols and
        strings.

        Two real numbers "a" and "b" match, if they are "sufficiently
        similar", where the exact definition of "sufficiently similar"
        is too complex to be discussed here. For the curious reader:
        the current implementation uses a relative epsilon algorithm.
        For instance, given

        sq2::{(x+2%x)%2}:~1 :"square root of 2"

        the following expression will be true:

        sq2~sq2+10*.e

        although the operands of Match differ by 10 times Epsilon.

        Two lists match if all of their elements match pairwise. "~"
        descends into sublists.

        Examples:                  1~1  -->  1
                           "foo"~"foo"  -->  1
                             :foo~:foo  -->  1
                               0cx~0cx  -->  1
                       [1 2 3]~[1 2 3]  -->  1
                   [1 [2] 3]~[1 [4] 3]  -->  0

    """
    return kg_truth(backend.kg_equal(a, b))


def eval_dyad_maximum(a, b, backend):
    """

        a|b                                                     [Max/Or]

        Return the larger one of two numbers.

        When both "a" and "b" are in the set {0,1} (booleans), then "|"
        acts as an "or" operator, as you can easily prove using a truth
        table:

        a  b  max/or
        0  0    0
        0  1    1
        1  0    1
        1  1    1

        Dyadic "|" is an atomic operator.

        Examples:       0|1  -->  1
                   123|-123  -->  123
                    1.0|1.1  -->  1.1

    """
    return backend.vec_fn2(a, b, backend.np.maximum)


def eval_dyad_minimum(a, b, backend):
    """

        a&b                                                    [Min/And]

        Return the smaller one of two numbers.

        When both "a" and "b" are in the set {0,1} (booleans), then "&"
        acts as an "and" operator, as you can easily prove using a truth
        table:

        a  b  min/and
        0  0     0
        0  1     0
        1  0     0
        1  1     1

        Dyadic "&" is an atomic operator.

        Examples:       0&1  -->  0
                   123&-123  -->  -123
                    1.0&1.1  -->  1.0

    """
    return backend.vec_fn2(a, b, backend.np.minimum)


def eval_dyad_more(a, b, backend):
    """

        a>b                                                       [More]

        Return 1, if "a" is greater than "b", otherwise return 0.

        See "<" (Less) for details on comparing objects.

        ">" is an atomic operator; it cannot compare lists, but only
        elements of lists.

        Examples:              2>1  -->  1
                       "foo">"bar"  -->  1
                         :xyz>:abc  -->  1
                           0c9>0c0  -->  1
                   [1 4 3]>[1 2 3]  -->  [0 1 0]

    """
    return kg_truth(backend.vec_fn2(a, b, _ordering(operator.gt, backend.np.greater)))


def eval_dyad_multiply(a, b, backend):
    """

        a*b                                                      [Times]

        Return "a" multiplied by "b". "a" and "b" must both be numbers.

        Dyadic "*" is an atomic operator.

        Examples:   3*4  -->  12
                   3*-4  -->  -12
                  0.3*7  -->  2.1

    """
    return backend.vec_fn2(a, b, backend.np.multiply)


# An exact integer power is computed only while the result stays small enough
# to be cheap (2^20 bits); beyond that the float computation answers, as before.
_EXACT_POWER_BITS = 1 << 20


def _int_scalar(x):
    """The Python int of an integer scalar (a Python or NumPy integer, or a 0-d integer
    array or tensor of any backend), else None. A bool is not one."""
    if isinstance(x, numbers.Integral):
        return None if isinstance(x, bool) else int(x)
    if getattr(x, 'ndim', None) == 0 and hasattr(x, 'item'):
        v = x.item()
        if type(v) is int:
            return v
    return None


def _exact_int_power(a, b):
    """a^b as an exact Python integer for two integer scalars and b >= 0, else None."""
    a = _int_scalar(a)
    b = _int_scalar(b)
    if a is None or b is None or b < 0 or a.bit_length() * b > _EXACT_POWER_BITS:
        return None
    return a ** b


def _power_kind(x, backend):
    """'i' or 'f' for a numeric operand of an array power (a plain or 0-d number, or a numeric
    array), else None: booleans, big integers, objects and anything with a gradient."""
    if isinstance(x, bool) or backend.has_gradient(x):
        return None
    if type(x) is int:
        return 'i' if -2**63 <= x < 2**63 else None
    if isinstance(x, numbers.Real):
        return 'i' if isinstance(x, numbers.Integral) else 'f'
    if backend.is_array(x):
        k = backend.get_dtype_kind(x)
        return k if k in ('i', 'f') else None
    return None


def _array_power(a, b, backend):
    """a^b when an operand is a numeric array, one rule for every arrangement (scalar^list,
    list^scalar, list^list): integers to non-negative integer powers wrap in int64 (integer
    arithmetic, never through float); a negative integer exponent or any float gives floats
    (as * does). None when this does not apply (the general path then)."""
    if not ((backend.is_array(a) and a.ndim > 0) or (backend.is_array(b) and b.ndim > 0)):
        return None
    ka, kb = _power_kind(a, backend), _power_kind(b, backend)
    if ka is None or kb is None:
        return None
    np_ = backend.np
    a, b = np_.asarray(a), np_.asarray(b)
    if ka == 'i' and kb == 'i' and not (b < 0).any():
        return backend.power(a, b)
    return backend.power(a * 1.0 if ka == 'i' else a, b * 1.0 if kb == 'i' else b)


def _e_dyad_power(a, b, backend):
    # An integer to a non-negative integer power is exact (Python integers do not
    # wrap); a negative or fractional exponent gives a float.
    exact = _exact_int_power(a, b)
    if exact is not None:
        return exact
    fast = _array_power(a, b, backend)
    if fast is not None:
        return fast
    # Check if input requires grad - if so, preserve float for autograd
    input_has_grad = backend.has_gradient(a)
    # Use backend power function which handles gradient-aware power
    r = backend.power(a, b)
    # If input had gradients, keep result as float to preserve autograd
    if input_has_grad:
        return r
    # Check if result is integer using vectorized operations
    r_val = backend.detach_if_needed(r)
    if backend.is_array(r_val) and backend.get_dtype_kind(r_val) in ('i', 'u'):
        return r  # an integer array power is already integers (wrapping, as documented)
    if is_list(r_val):
        # Vectorized check: trunc(r) == r for all elements
        trunc_r = numpy.trunc(r_val) if isinstance(r_val, numpy.ndarray) else r_val.trunc()
        br = bool((trunc_r == r_val).all())
    else:
        val = float(r_val) if hasattr(r_val, 'item') else r_val
        br = numpy.trunc(val) == val
    if br:
        return backend.to_int_array(r)
    return r


def eval_dyad_power(a, b, backend):
    """

        a^b                                                      [Power]

        Compute "a" to the power of "b" and return the result. Both "a"
        and "b" must be numbers. The result of a^b cannot be a complex
        number.

        Dyadic "^" is an atomic operator.

        Examples:   2^0  -->  1
                    2^1  -->  2
                    2^8  -->  256
                   2^-5  -->  0.03125
                  0.3^3  -->  0.027
                  2^0.5  -->  1.41421356237309504

    """
    return backend.vec_fn2(a, b, lambda x, y: _e_dyad_power(x, y, backend))


def eval_dyad_remainder(a, b, backend):
    """

        a!b                                                  [Remainder]

        Return the truncated division remainder of "a" and "b". Both
        "a" and "b" must be integers.

        Formally, a = (b*a:%b) + a!b .

        a!0 is :undefined (as is a%0 and a:%0), also element by element.

        Dyadic "!" is an atomic operator.

        Examples:    7!5  -->  2
                    7!-5  -->  2
                  (-7)!5  --> -2
                   -7!-5  --> -2

    """
    return backend.vec_fn2(a, b, _undefined_on_zero_divisor(backend.np.fmod, backend))


def eval_dyad_reshape(a, b, backend):
    """

        a:^b                                                   [Reshape]

        :^ reshapes "b" to the shape specified in "a". The shape is
        specified in the form returned by the "^" (Shape) operator: a
        list of dimensions in row-major order.

        The operand "b" may be in any shape. The elements of the new
        array will be taken from "b" in sequential order:

        [3 3]:^[1 2 3 4 5 6 7 8 9]  -->  [[1 2 3]
                                          [4 5 6]
                                          [7 8 9]]

        When the source array contains more elements that can be stored
        in an array of the shape "a", excess elements in "b" will be
        ignored. When the source array contains too few elements, :^
        will cycle through the source object, repeating the elements
        found there:

        [3 3]:^[0 1]  -->  [[0 1 0]
                            [1 0 1]
                            [0 1 0]

        When the value -1 appears in the shape parameter "a", it denotes
        half the size of the source vector, e.g.:

        [-1 2]:^!10  -->  [[0 1] [2 3] [4 5] [6 7] [8 9]]
        [2 -1]:^!10  -->  [[0 1 2 3 4] [5 6 7 8 9]]

        Both "a" and "b" may be atoms:

        5:^1  -->  [1 1 1 1 1]

        but when "b" is an atom (or a single-argument vector), then "a"
        may not contain the value -1.

        0:^x is an identity operation returning x.

        Examples:           5:^:x  -->  [:x :x :x :x :x]
                         [3]:^[1]  -->  [1 1 1]
                  [2 2 2]:^[1 2 3] -->  [[[1 2] [3 1]] [[2 3] [1 2]]]
                    [2]:^[[1 2 3]] -->  [[1 2 3] [1 2 3]]

    """
    np_backend = backend.np
    j = isinstance(b, str)
    b = backend.str_to_chr_arr(b) if j else b
    if backend.is_backend_array(a) and a.ndim == 0:
        a = backend.scalar_to_python(a)  # a 0-d tensor is an atom: the size, not a shape list
    if backend.is_backend_array(b) and b.ndim == 0:
        b = backend.scalar_to_python(b) if not backend.has_gradient(b) else b.reshape(1)
    if np_backend.isarray(a):
        if np_backend.isarray(a) and a.ndim > 0 and bool((a < -1).any()):
            raise KlongRangeError(f":^: only -1 may be negative in a shape, got {to_list(a)}")
        if np_backend.isarray(b):
            y = np_backend.where(a < 0)[0]
            if len(y) > 0:
                a = np_backend.copy(a)
                a[y] = backend.array_size(b) // 2
            b_s = backend.array_size(b)
            a_s = int(np_backend.prod(a))  # Ensure it's a Python int for comparison
            if b_s == 0 and a_s > 0:
                raise KlongRangeError(":^: cannot fill a non-empty shape from an empty source")
            # Convert shape to tuple of ints for backend compatibility
            a_shape = tuple(int(x) for x in (a.tolist() if hasattr(a, 'tolist') else a))
            if a_s > b_s:
                b = np_backend.tile(b.flatten(), (a_s // b_s))
                b = np_backend.concatenate((b, b[:a_s - backend.array_size(b)]))
                b_s = backend.array_size(b)
                r = b.reshape(a_shape)
                r = bknp.asarray(["".join(x) for x in r]) if j else r
                j = False
            elif a_s == b_s:
                r = b.reshape(a_shape)
            else:
                r = np_backend.resize(b, a_shape)
        else:
            if bool((a < 0).any()):
                raise KlongRangeError(":^: -1 in a shape needs a list as the source, not an atom")
            r = np_backend.full(a, b)
    else:
        if a < 0:
            raise KlongRangeError(f":^: size must not be negative, got {a}")
        if a == 0:
            r = b
        elif np_backend.isarray(b) and backend.array_size(b) == 0:
            raise KlongRangeError(":^: cannot fill a non-empty shape from an empty source")
        elif np_backend.isarray(b):
            if a < b.shape[0]:
                r = np_backend.resize(b, (a,))
            else:
                ns = np_backend.ones(len(b.shape),dtype=int)
                ns[0] = a // b.shape[0]
                r = np_backend.concatenate((np_backend.tile(b,ns), b[:a - b.shape[0]*ns[0]]))
        else:
            r = np_backend.full((a,), b)
    if j:
        if np_backend.isarray(r) and r.ndim > 1:
            return bknp.asarray(["".join(x) for x in r], dtype=object)
        return "".join(r)
    return r


def eval_dyad_rotate(a, b, backend):
    """

        a:+b                                                    [Rotate]

        Rotate the list or string "b" by "a" elements. "a" must be an
        integer. When "a" is positive, rotate elements to the "right",
        i.e. drop elements from the end of "b" and append them to the
        front. When "a" is negative, rotate "b" to the left, i.e. drop
        from the beginning, append to the end.

        "a" may be greater than #b. In this case, the number of elements
        rotated will be a!#b.

        Note that n:+M rotates the rows of a matrix M (i.e. it rotates
        it vertically); to rotate its columns (horizontally), use n:+:\\M
        (Rotate-Each-Left).

        Examples:           1:+[1 2 3 4 5]     -->  [5 1 2 3 4]
                            (-1):+[1 2 3 4 5]  -->  [2 3 4 5 1]
                       1:+[[1 2] [4 5] [5 6]]  -->  [[1 2] [4 5] [5 6]]
                   {1:+x}'[[1 2] [4 5] [5 6]]  -->  [[2 1] [5 4] [6 5]]

    """
    if a == 0 or not is_iterable(b):
        return b
    j = isinstance(b, str)
    b = backend.str_to_chr_arr(b) if j else b
    r = backend.np.roll(b, a)
    return "".join(r) if j else r


def eval_dyad_split(a, b, backend):
    """

        a:#b                                                     [Split]

        Split a list or string "b" into segments of the sizes given in
        "a". If "a" is an integer, all segments will be of the same size.
        If "a" is a list of more than one element, sizes will be taken
        from that list. When there are more segments than sizes, :# will
        cycle through "a". The last segment may be shorter than
        specified.

        Examples:         2:#[1 2 3 4]  -->  [[1 2] [3 4]]
                          3:#[1 2 3 4]  -->  [[1 2 3] [4]]
                          3:#"abcdefg"  -->  ["abc" "def" "g"]
                  [1 2]:#[1 2 3 4 5 6]  -->  [[1] [2 3] [4] [5 6]]

    """
    if len(b) == 0:
        return backend.np.asarray([])

    j = isinstance(b, str)
    b = backend.str_to_chr_arr(b) if j else b

    a = a if backend.np.isarray(a) and a.ndim > 0 else [a]
    if len(a) == 0:
        raise KlongRangeError(":#: segment sizes must not be empty")
    sizes = []
    for n in a:
        if getattr(n, 'ndim', None) == 0 and hasattr(n, 'item'):
            n = n.item()
        if isinstance(n, (str, bool)) or not isinstance(n, numbers.Real) or n != n or n in (float('inf'), float('-inf')) or n != int(n):
            raise KlongTypeError(f":#: segment sizes must be integers, got {n!r}")
        if n < 1:
            raise KlongRangeError(f":#: segment sizes must be positive, got {n!r}")
        sizes.append(int(n))
    a = sizes
    if len(a) == 1:
        if a[0] >= len(b):
            r = [b]
        else:
            k = len(b) // a[0]
            if (k*a[0]) < len(b):
                k += 1
            r = backend.array_split(b, k)
    else:
        p, q = 0, 0
        r = []
        while q < len(b):
            r.append(b[q:q+a[p]])
            q += a[p]
            p += 1
            if p >= len(a):
                p = 0

    return bknp.asarray(["".join(x) for x in r],dtype=object) if j else backend.kg_asarray(r)


def eval_dyad_subtract(a, b, backend):
    """

        a-b                                                      [Minus]

        Subtract "b" from "a" and return the result. "a" and "b" must be
        numbers.

        "-" is an atomic operator.

        Examples:  12-3  -->  9
                  12--3  -->  15
                  1-0.3  -->  0.7

    """
    return backend.vec_fn2(a, b, backend.np.subtract)


def eval_dyad_take(a, b, backend):
    """

        a#b                                                       [Take]

        Extract "a" elements from the front of "b". "a" must be an
        integer and "b" must be a list or string. If "a" is negative,
        extract elements from the end of "b". Extracting more elements
        than contained in "b" will fill the extra slots by cycling
        through "b". Taking 0 elements will result in an empty list
        or string. If "b" is an atom (a number or a symbol), it is
        repeated: 3#7 --> [7 7 7].

        Examples:     1#[1 2 3]  -->  [1]
                      2#[1 2 3]  -->  [1 2]
                      5#[1 2 3]  -->  [1 2 3 1 2]
                   (-2)#[1 2 3]  -->  [2 3]
                   (-5)#[1 2 3]  -->  [2 3 1 2 3]
                     3#"abcdef"  -->  "abc"
                  (-3)#"abcdef"  -->  "def"
                           0#[]  -->  []
                           0#""  -->  ""

    """
    np_backend = backend.np
    j = isinstance(b,str) and not isinstance(b, KGSym)
    if j:
        b = backend.str_to_chr_arr(b)
    elif isinstance(b, KGSym):
        b = backend.kg_asarray([b])  # a symbol is an atom: n#:s is [:s :s ...]
    else:
        if not np_backend.isarray(b):
            b = backend.kg_asarray(b)
        if b.ndim == 0:
            b = b.reshape(1)  # an atom: n#7 is [7 7 ...]
    abs_a = np_backend.abs(a)
    aa = int(abs_a) if hasattr(abs_a, 'item') else abs_a  # Convert tensor to int
    b_size = backend.array_size(b)
    if b_size == 0:
        # Handle empty array/string case
        r = b
    elif not j and len(getattr(b, 'shape', ())) >= 2:
        # rank 2 or more: cycle over the items of the first axis
        n = len(b)
        start = 0 if a >= 0 else (n - aa) % n
        r = b[[(start + i) % n for i in range(aa)]] if aa else b[:0]
    elif aa > b_size:
        b = np_backend.tile(b, aa // len(b))
        b = np_backend.concatenate((b, b[:aa-backend.array_size(b)]) if a > 0 else (b[-(aa-backend.array_size(b)):], b))
        r = b[a:] if a < 0 else b[:a]
    else:
        r = b[a:] if a < 0 else b[:a]
    return "".join(r) if j else r


# ---- from adverbs.py ----

def _iteration_count(a, name):
    """Validate the count of iterate/scan-iterate and return it as a Python int.

    Any integral real is accepted (int, numpy integer, float with an integral
    value) as long as it is >= 0.  Anything else raises instead of looping.
    """
    if getattr(a, 'ndim', None) == 0 and hasattr(a, 'item'):
        a = a.item()
    if isinstance(a, (bool, str)) or not isinstance(a, numbers.Real):
        raise KlongTypeError(f"{name}: count must be a non-negative integer, got {a!r}")
    if a != a or a in (float('inf'), float('-inf')) or a != int(a):
        raise KlongRangeError(f"{name}: count must be a non-negative integer, got {a!r}")
    if a < 0:
        raise KlongRangeError(f"{name}: count must not be negative, got {a!r}")
    return int(a)


def _is_scalar_operand(b):
    """True for an atom that each-left/each-right must not iterate (numbers, chars, symbols, ...)."""
    if isinstance(b, (str, dict)):
        return isinstance(b, (KGChar, KGSym))
    return not is_iterable(b)


def _converged(backend, p, q):
    """True when two successive values are equal; NaN equals NaN."""
    if backend.is_number(p) and backend.is_number(q):
        return bool(backend.np.isclose(p, q, equal_nan=True))
    if not isinstance(p, type(q)):
        return False
    if backend.is_array(p):
        return backend.kg_equal(p, q)
    return p == q


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
        return _converged(backend, p, q)
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
    if isinstance(a,str) and not isinstance(a, KGSym):  # a symbol is an atom: f(a) below
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


def eval_adverb_each_index(f, a, op, backend):
    """

        f@'a                                                 [Each-Index]

        If "a" is a list, apply "f" to [index;element] pairs for each
        member of "a":

        f@'a  -->  f([0;a1]),f([1;a2]),...,f([N-1;aN])

        Similar to Python's enumerate(), this gives the function access
        to both the index and the value at each position.

        Examples:  {x@0}@'[10 20 30]        -->  [0 1 2]
                   {x@1}@'[10 20 30]        -->  [10 20 30]
                   {(x@0)*(x@1)}@'[10 20 30]  -->  [0 20 60]
    """
    if is_empty(a):
        return a
    if is_iterable(a):
        r = [f(backend.kg_asarray([i, x])) for i, x in enumerate(a)]
        return backend.kg_asarray(r)
    return f(backend.kg_asarray([0, a]))


def eval_adverb_each2(f, a, b, backend):
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
        return backend.kg_asarray([]) if is_list(a) or is_list(b) else ""
    if is_atom(a) and is_atom(b):
        return f(a,b)
    # backend.kg_asarray stacks (torch.stack on the torch backend) so gradients
    # are preserved through each-dyad — unlike numpy asarray which detaches them.
    r = backend.kg_asarray([f(x,y) for x,y in zip(a,b)])
    dt = getattr(r, 'dtype', None)
    if dt == '<U1':
        return ''.join(r)
    # kg_asarray may return an object array of single-char strings (numpy
    # backend) where numpy.asarray would give '<U1'; rejoin those into a string.
    if dt == object and len(r) and all(isinstance(c, str) and len(c) == 1 for c in r):
        return ''.join(r)
    return r


def eval_adverb_each_left(f, a, b, backend):
    r"""
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
    if _is_scalar_operand(b):
        return f(a,b)
    b = backend.str_to_chr_arr(b) if isinstance(b,str) else b
    return backend.kg_asarray([f(a,x) for x in b])


def eval_adverb_each_right(f, a, b, backend):
    """
    see: eval_dyad_adverb_each_left
    """
    if _is_scalar_operand(b):
        return f(b,a)
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
    return backend.kg_asarray([f(x,y) for x,y in zip(a[::],a[1::])])


def eval_dyad_adverb_iterate(f, a, b):
    """

        a f:*b                                                 [Iterate]

        Apply "f" recursively to "b" "a" times. More formally:

        - if "a" is zero, return b
        - else assign b::f(b) and a::a-1 and start over

        Example: 3{1,x}:*[]  -->  [1 1 1]

    """
    for _ in range(_iteration_count(a, ":*")):
        b = f(b)
    return b


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
    if is_atom(a):
        return a
    if len(a) == 1:
        return a[0]
    if isinstance(a, str):
        # numpy string ufuncs have no min/max: use Python's on the characters
        if isinstance(op, KGOp) and safe_eq(op.a, '&'):
            return min(a)
        if isinstance(op, KGOp) and safe_eq(op.a, '|'):
            return max(a)
        return functools.reduce(f, list(a))
    # Use backend's ufunc reduce when available for better performance
    np_backend = backend.np
    if isinstance(op, KGOp):
        if safe_eq(op.a,'+'):
            return np_backend.add.reduce(a)
        elif safe_eq(op.a, '-'):
            return np_backend.subtract.reduce(a)
        elif safe_eq(op.a, '*') and hasattr(np_backend.multiply,'reduce'):
            return np_backend.multiply.reduce(a)
        elif safe_eq(op.a, '%') and hasattr(np_backend.divide,'reduce'):
            with np_backend.errstate(divide='ignore'):
                return np_backend.divide.reduce(a)
        elif safe_eq(op.a, '&') and getattr(a, 'ndim', 0) == 1:
            return np_backend.min(a)
        elif safe_eq(op.a, '|') and getattr(a, 'ndim', 0) == 1:
            return np_backend.max(a)
        elif safe_eq(op.a, ',') and np_backend.isarray(a) and a.dtype != 'O':
            return a if a.ndim == 1 else np_backend.concatenate(a, axis=0)
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
    r"""

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


def eval_adverb_scan_over(f, a, op, backend):
    """
        see eval_adverb_scan_over_neutral
    """
    if is_atom(a):
        return a
    # Use backend's ufunc accumulate when available for better performance
    np_backend = backend.np
    if isinstance(op, KGOp):
        if safe_eq(op.a, '+') and hasattr(np_backend.add, 'accumulate'):
            return np_backend.add.accumulate(a)
        elif safe_eq(op.a, '-') and hasattr(np_backend.subtract, 'accumulate'):
            return np_backend.subtract.accumulate(a)
        elif safe_eq(op.a, '*') and hasattr(np_backend.multiply, 'accumulate'):
            return np_backend.multiply.accumulate(a)
        elif safe_eq(op.a, '%') and hasattr(np_backend.divide, 'accumulate'):
            with np_backend.errstate(divide='ignore'):
                return np_backend.divide.accumulate(a)
    r = list(itertools.accumulate(a, f))
    return backend.kg_asarray(r)


def eval_adverb_scan_converging(f, a, op, backend):
    r"""

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
    while not _converged(backend, x, xx):
        x = xx
        xx = f(x)
        r.append(xx)
    r.pop()
    return backend.kg_asarray(r)


def eval_adverb_scan_while(klong, f, a, b, backend):
    r"""

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
    while klong.apply(a, [b]):
        b = f(b)
        r.append(b)
    r.pop()
    return backend.kg_asarray(r)


def eval_adverb_scan_iterating(f, a, b, backend):
    r"""

        a f\*b                                          [Scan-Iterating]

        This adverbs is like its non-scanning counterpart, but collects
        intermediate results in a list and return that list.

        The count must be a non-negative integer. If "a" is zero, "b" is
        returned as it is (not wrapped in a list), as in the Klong
        reference: 0{1,x}\*[]  -->  [] and 0,\*1  -->  1.

        Example: 3{1,x}\*[]  -->  [[] [1] [1 1] [1 1 1]]

    """
    n = _iteration_count(a, "\\*")
    if n == 0:
        return b
    r = [b]
    for _ in range(n):
        b = f(b)
        r.append(b)
    return backend.kg_asarray(r)


def eval_adverb_while(klong, f, a, b):
    """

        a f:~b                                                   [While]

        Compute b::f(b) while a(b) is true. Formally:

        - if a(b) is false, return b
        - else assign b::f(b) and start over

        Example: {x<1000}{x*2}:~1  -->  1024

    """
    while klong.apply(a, [b]):
        b = f(b)
    return b


def get_adverb_fn(klong, s, arity):
    backend = klong._backend

    if s == "'":
        return (lambda f,a,b: eval_adverb_each2(f,a,b,backend)) if arity == 2 else lambda f,a,op: eval_adverb_each(f,a,op,backend)
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
    elif s == "@'":
        return lambda f,a,op: eval_adverb_each_index(f,a,op,backend)
    raise RuntimeError(f"unknown adverb: {s}")


# ---- the tables ----

def monad_table(klong):
    """{glyph: f(x)} of the monads on the backend's arrays."""
    backend = klong._backend
    return {
        '@': eval_monad_atom,
        '*': eval_monad_first,
        '˙': eval_monad_track,
        '|': lambda a: eval_monad_reverse(a, backend),
        '&': lambda a: eval_monad_expand_where(a, backend),
        '+': lambda a: eval_monad_transpose(a, backend),
        ',': lambda a: eval_monad_list(a, backend),
        ':#': lambda a: eval_monad_char(a, backend),
        '!': lambda a: eval_monad_enumerate(a, backend),
        '_': lambda a: eval_monad_floor(a, backend),
        '$': lambda a: eval_monad_format(a, backend),
        '<': lambda a: eval_monad_grade_up(a, backend),
        '>': lambda a: eval_monad_grade_down(a, backend),
        '=': lambda a: eval_monad_groupby(a, backend),
        '-': lambda a: eval_monad_negate(a, backend),
        '~': lambda a: eval_monad_not(a, backend),
        '?': lambda a: eval_monad_range(a, backend),
        '%': lambda a: eval_monad_reciprocal(a, backend),
        '#': lambda a: eval_monad_size(a, backend),
        ':_': lambda a: eval_monad_undefined(a, backend),
        '^': lambda a: eval_monad_shape(a, backend),
    }


def dyad_table(klong):
    """{glyph: f(x, y)} of the dyads on the backend's arrays (`@` applies function values through klong.apply)."""
    backend = klong._backend
    return {
        '+': lambda a, b: eval_dyad_add(a, b, backend),
        '|': lambda a, b: eval_dyad_maximum(a, b, backend),
        '&': lambda a, b: eval_dyad_minimum(a, b, backend),
        '!': lambda a, b: eval_dyad_remainder(a, b, backend),
        '%': lambda a, b: eval_dyad_divide(a, b, backend),
        '*': lambda a, b: eval_dyad_multiply(a, b, backend),
        '-': lambda a, b: eval_dyad_subtract(a, b, backend),
        ':=': lambda a, b: eval_dyad_amend(a, b, backend),
        ':-': lambda a, b: eval_dyad_amend_in_depth(a, b, backend),
        '_': lambda a, b: eval_dyad_drop(a, b, backend),
        ':@': lambda a, b: eval_dyad_index_in_depth(a, b, backend),
        ':_': lambda a, b: eval_dyad_cut(a, b, backend),
        '=': lambda a, b: eval_dyad_equal(a, b, backend),
        '?': lambda a, b: eval_dyad_find(a, b, backend),
        ':$': lambda a, b: eval_dyad_form(a, b, backend),
        '$': lambda a, b: eval_dyad_format2(a, b, backend),
        ':%': lambda a, b: eval_dyad_integer_divide(a, b, backend),
        ',': lambda a, b: eval_dyad_join(a, b, backend),
        '<': lambda a, b: eval_dyad_less(a, b, backend),
        '~': lambda a, b: eval_dyad_match(a, b, backend),
        '>': lambda a, b: eval_dyad_more(a, b, backend),
        '^': lambda a, b: eval_dyad_power(a, b, backend),
        ':^': lambda a, b: eval_dyad_reshape(a, b, backend),
        ':+': lambda a, b: eval_dyad_rotate(a, b, backend),
        ':#': lambda a, b: eval_dyad_split(a, b, backend),
        '#': lambda a, b: eval_dyad_take(a, b, backend),
        '@': lambda a, b: eval_dyad_at_index(klong, a, b),
    }


def format2(a, b, backend):
    return _e_dyad_format2(a, b, backend)


def converged(backend, p, q):
    return _converged(backend, p, q)


def power(a, b, backend):
    return _e_dyad_power(a, b, backend)
