from .core import *
import re
import sys


def eval_dyad_add(a, b):
    """

        a+b                                                       [Plus]

        Add "b" to "a" and return the result. "a" and "b" must both be
        numbers.

        Dyadic "+" is an atomic operator.

        Examples:  12+3  -->  15
                  12+-3  -->  9
                  1+0.3  -->  1.3

    """
    return vec_fn2(a, b, np.add)


def eval_dyad_amend(a, b):
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
    if not (isinstance(a, (str,list)) or np.isarray(a)):
        raise RuntimeError(f"a must be list or str: {a}")
    if len(b) <= 1:
        return a
    if isinstance(a, str):
        r = str_to_chr_arr(a)
        q = str_to_chr_arr(b[0])
        for i in b[1:]:
            try:
                r[i:i+len(q)] = q
            except ValueError:
                r = r.astype(object)
                if i > len(r):
                    RangeError(i)
                elif i == len(r):
                    r = np.append(r, b[0])
                else:
                    r[i] = b[0]
        return "".join(["".join(x) for x in r])
    r = np.array(a) # clone
    if is_list(b[0]): # TOOD: use np.put if we can
        r = r.tolist()
        for i in b[1:]:
            r[i] = b[0]
        r = np.asarray(r)
    else:
        np.put(r, np.asarray(b[1:],dtype=int), b[0])
    return r


def eval_dyad_amend_in_depth(a, b):
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
    def _e(p, q, v):
        if np.isarray(q) and len(q) > 1:
            r = _e(p[q[0]], q[1:] if len(q) > 2 else q[1], v)
            p = np.array(p, dtype=r.dtype)
            p[q[0]] = r
            return p
        else:
            p = np.array(p, dtype=object) if isinstance(v, (str, KGSym)) else np.array(p)
            p[q] = v
            return p
    return _e(a, b[1:], b[0])


def eval_dyad_cut(a, b):
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
    b = np.asarray(str_to_chr_arr(b) if j else b)
    a = a if np.isarray(a) else [a]
    r = np.array_split(b, a)
    if len(b) == 0 and len(a) > 0:
        r = r[1:]
    return np.asarray(["".join(x) for x in r]) if j else np.asarray(r)


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

        When "a" is a function, "b" (if it is an atom) or the members
        of "b" (if it is a list) will be passed as arguments to the
        function and the result will be returned.

        Examples:         [1 2 3 4 5]@2  -->  3
                    [1 2 3 4 5]@[1 2 3]  -->  [2 3 4]
                    [1 2 3 4 5]@[0 0 0]  -->  [1 1 1]
                  "hello world"@[3 7 2]  -->  "lol"
                               {x}@:foo  -->  :foo
                         {y+x*x}@[2 3]   -->  7

    """
    if isinstance(a, (KGFn, KGSym)):
        # TODO: fix arity
        return klong.eval(KGCall(a, b.tolist() if np.isarray(b) else b, arity=2))
    j = isinstance(a,str)
    a = str_to_chr_arr(a) if j else a
    if is_list(b):
        if is_empty(b):
            r = np.asarray([])
        else:
            # TODO: return None for missing keys? or raise?
            r = np.asarray([a[x] for x in b])
    elif is_integer(b):
        r = a[b]
        j = False
    else:
        r = a
    return "".join(r) if j else r


def eval_dyad_define(klong, n, v):
    """

        a::b                                                    [Define]

        Assign "b" to the variable "a" and return "b". When a local
        variable named "a" exists, the value will be assigned to it,
        otherwise the global variable "a" will be assigned the value.

        Note that :: cannot be used to assign values to the function
        variables "x", "y", and "z" (they are read-only).

        Examples:        a::[1 2 3];a  -->  [1 2 3]
                  a::1;{[a];a::2}();a  -->  1

    """
    klong[n] = v
    return v


def eval_dyad_divide(a, b):
    """

        a%b                                                     [Divide]

        Return the quotient of "a" and "b". The result is always a real
        number, even if the result has a fractional part of 0.

        "%" is an atomic operator.

        Examples: 10%2  -->  5.0
                  10%8  -->  1.25

    """
    return vec_fn2(a, b, np.divide)


def eval_dyad_drop(a, b):
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
            del b[a] # biased towards presence perf
        except KeyError:
            pass
        return b
    return b[a:] if a >= 0 else b[:a]


def eval_dyad_equal(a, b):
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
    return vec_fn2(a, b, lambda x, y: kg_truth(np.asarray(x,dtype=object) == np.asarray(y,dtype=object)))


def eval_dyad_find(a, b):
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
        return np.asarray([m.start() for m in re.finditer(f"(?={b})", a)])
    elif is_dict(a):
        # NOTE: we don't use get or np.inf because value may be 0 or None
        return a[b] if b in a else np.inf # TODO: use undefined type
    if is_list(b):
        return np.asarray([i for i,x in enumerate(a) if array_equal(x,b)])
    return np.where(np.asarray(a) == b)[0]


def eval_dyad_form(a, b):
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
    if isinstance(a,KGSym):
        if is_empty(b):
            return np.inf
        return KGSym(b[1:] if isinstance(b,str) and b.startswith(":") else b)
    if is_integer(a):
        def _is_float(b):
            try:
                float(b)
                return True
            except ValueError:
                return False
        if is_float(b) or is_empty(b) or ('.' in b and _is_float(b)):
            return np.inf
        return int(b)
    if is_float(a):
        if is_empty(b):
            return np.inf
        return float(b)
    if isinstance(a,KGChar):
        b = str(b)
        if len(b) != 1:
            return np.inf
        return KGChar(str(b)[0])
    return b


def eval_dyad_format2(a, b):
    """

        a$b                                                    [Format2]

        Dyadic "$" is like its monadic cousin, but also pads its result
        with blanks. The minimal size of the output string is specified
        in "a", which must be an integer. "b" is the object to format.
        When the value of "a" is negative, the result string is padded
        to the right, else it is padded to the left.

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
    if safe_eq(int(a), 0):
        return str(b)
    if (is_float(b) and not isinstance(b,int)) and (is_float(a) and not isinstance(a,int)):
        b = "{:Xf}".replace("X",str(a)).format(b)
        p = b.split('.')
        p[0] = p[0].rjust(int(a))
        b = ".".join(p)
        return b
    b = f":{b}" if isinstance(b, KGSym) else b
    r = str(b).ljust(abs(a)) if a >= 0 else str(b).rjust(abs(a))
    return r


def eval_dyad_index_in_depth(a, b):
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
    return np.asarray(a)[tuple(b) if is_list(b) else b] if not is_empty(b) else b


def eval_dyad_integer_divide(a, b):
    """

        a:%b                                            [Integer-Divide]

        Return the integer part of the quotient of "a" and "b". Both "a"
        and "b" must be integers. The result is always an integer.

        Formally, a = (b*a:%b) + a!b .

        ":%" is an atomic operator.

        Examples: 10:%2  -->  5
                  10:%8  -->  1

    """
    def _e(x,y):
        a = np.trunc(np.divide(x, y))
        return np.asarray(a,dtype=int) if np.isarray(a) else int(a)
    return vec_fn2(a, b, _e)


def dyad_join_to_list(a):
    if np.isarray(a):
        if a.ndim == 1:
            return a
        elif a.shape[0] == 1:
            return [a.flatten()]
        elif a.shape[0] > 1:
            return a
    return [a]


def eval_dyad_join(a, b):
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

    """
    if (isinstance(a,str) and not isinstance(a,KGSym)) and (isinstance(b,str) and not isinstance(b,KGSym)):
        return a+b
    if isinstance(a,dict):
        a[b[0]] = b[1]
        return a
    if isinstance(b,dict) and is_list(a) and len(a) == 2:
        b[a[0]] = a[1]
        return b

    if np.isarray(a) and np.isarray(b):
        if len(a) == 0:
            return b
        if len(a.shape) == len(b.shape) and a.shape[-1] == b.shape[-1]:
            return np.concatenate((a,b))

    aa = dyad_join_to_list(a)
    bb = dyad_join_to_list(b)
 
    r = [*aa,*bb]
    nr = np.asarray(r)
    t = nr.dtype.type
    return nr if issubclass(t, np.integer) or issubclass(t, np.floating) else np.asarray(r,dtype=object)


def eval_dyad_less(a, b):
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
    return kg_truth(vec_fn2(a, b, lambda x,y: x < y if (isinstance(x,str) and isinstance(y,str)) else np.less(x,y)))


def eval_dyad_match(a,b):
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
    return kg_truth(all_fn2(a, b, lambda x,y: np.isclose(x, y) if is_number(x) and is_number(y) else np.all(x == y)))


def eval_dyad_maximum(a, b):
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
    return vec_fn2(a, b, np.maximum)


def eval_dyad_minimum(a, b):
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
    return vec_fn2(a, b, np.minimum)


def eval_dyad_more(a, b):
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
    return kg_truth(vec_fn2(a, b, lambda x,y: x > y if (isinstance(x,str) and isinstance(y,str)) else np.greater(x,y)))


def eval_dyad_multiply(a, b):
    """

        a*b                                                      [Times]

        Return "a" multiplied by "b". "a" and "b" must both be numbers.

        Dyadic "*" is an atomic operator.

        Examples:   3*4  -->  12
                   3*-4  -->  -12
                  0.3*7  -->  2.1

    """
    return vec_fn2(a, b, np.multiply)


def eval_dyad_power(a, b):
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
    def _e(a,b):
        r = np.power(float(a) if is_integer(a) else a, b)
        return np.dtype('int').type(r) if np.trunc(r) == r else r
    return vec_fn2(a, b, _e)


def eval_dyad_remainder(a, b):
    """

        a!b                                                  [Remainder]

        Return the truncated division remainder of "a" and "b". Both
        "a" and "b" must be integers.

        Formally, a = (b*a:%b) + a!b .

        Dyadic "!" is an atomic operator.

        Examples:    7!5  -->  2
                    7!-5  -->  2
                  (-7)!5  --> -2
                   -7!-5  --> -2

    """
    return vec_fn2(a, b, np.fmod)


def eval_dyad_reshape(a, b):
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
    j = isinstance(b, str)
    b = str_to_chr_arr(b) if j else b
    if np.isarray(a):
        if np.isarray(b):
            y = np.where(a < 0)[0]
            if len(y) > 0:
                a = np.copy(a)
                a[y] = b.size // 2
            b_s = b.size
            a_s = np.prod(a)
            if a_s > b_s:
                b = np.tile(b.flatten(), (a_s // b_s))
                b = np.concatenate((b, b[:a_s - b.size]))
                b_s = b.size
                r = b.reshape(a)
                r = np.asarray(["".join(x) for x in r]) if j else r
                j = False
            elif a_s == b_s:
                r = b.reshape(a)
            else:
                r = np.resize(b, a)
        else:
            r = np.ones(a)*b
    else:
        if a == 0:
            r = b
        elif np.isarray(b):
            if a < b.shape[0]:
                r = np.resize(b, (a,))
            else:
                ns = np.ones(len(b.shape),dtype=int)
                ns[0] = a // b.shape[0]
                r = np.concatenate((np.tile(b,ns), b[:a - b.shape[0]*ns[0]]))
        else:
            r = np.ones((a,))*b
    return "".join(r) if j else r


def eval_dyad_rotate(a, b):
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
        it vertically); to rotate its columns (horizontally), use n:+:\M
        (Rotate-Each-Left).

        Examples:           1:+[1 2 3 4 5]     -->  [5 1 2 3 4]
                            (-1):+[1 2 3 4 5]  -->  [2 3 4 5 1]
                       1:+[[1 2] [4 5] [5 6]]  -->  [[1 2] [4 5] [5 6]]
                   {1:+x}'[[1 2] [4 5] [5 6]]  -->  [[2 1] [5 4] [6 5]]

    """
    if a == 0 or not is_iterable(b):
        return b
    j = isinstance(b, str)
    b = str_to_chr_arr(b) if j else b
    r = np.roll(b, a)
    return "".join(r) if j else r


def eval_dyad_split(a, b):
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
        return np.asarray([])

    j = isinstance(b, str)
    b = str_to_chr_arr(b) if j else b

    a = a if np.isarray(a) else [a]
    if len(a) == 1:
        if a[0] >= len(b):
            r = [b]
        else:
            k = len(b) // a[0]
            if (k*a[0]) < len(b):
                k += 1
            r = np.array_split(b, k)
    else:
        p, q = 0, 0
        r = []
        while q < len(b):
            r.append(b[q:q+a[p]])
            q += a[p]
            p += 1
            if p >= len(a):
                p = 0

    return np.asarray(["".join(x) for x in r]) if j else np.asarray(r)


def eval_dyad_subtract(a, b):
    """

        a-b                                                      [Minus]

        Subtract "b" from "a" and return the result. "a" and "b" must be
        numbers.

        "-" is an atomic operator.

        Examples:  12-3  -->  9
                  12--3  -->  15
                  1-0.3  -->  0.7

    """
    return vec_fn2(a, b, np.subtract)


def eval_dyad_take(a, b):
    """

        a#b                                                       [Take]

        Extract "a" elements from the front of "b". "a" must be an
        integer and "b" must be a list or string. If "a" is negative,
        extract elements from the end of "b". Extracting more elements
        than contained in "b" will fill the extra slots by cycling
        through "b". Taking 0 elements will result in an empty list
        or string.

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
    j = isinstance(b,str)
    b = str_to_chr_arr(b) if j else np.asarray(b)
    aa = np.abs(a)
    if aa > b.size:
        b = np.tile(b,aa // len(b))
        b = np.concatenate((b, b[:aa-b.size]) if a > 0 else (b[-(aa-b.size):],b))
    r = b[a:] if a < 0 else b[:a]
    return "".join(r) if j else r


def create_dyad_functions(klong):
    def _get_name(s):
        s = s.strip()
        i = s.index("a")
        return s[i+1:i+s.index('b')]

    registry = {}

    m = sys.modules[__name__]
    for name in filter(lambda n: n.startswith("eval_dyad_"), dir(m)):
        fn = getattr(m,name)
        name = _get_name(fn.__doc__)
        if fn.__code__.co_argcount == 3:
            fn = lambda x,y,f=fn,klong=klong: f(klong, x, y)
        elif fn.__code__.co_argcount == 2 and 'klong' in fn.__code__.co_varnames:
            fn = lambda x,f=fn,klong=klong: f(klong, x)
        registry[name] = fn

    return registry
