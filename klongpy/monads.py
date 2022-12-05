from .core import *
import sys

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


def eval_monad_char(a):
    """

        :#a                                                       [Char]

        Return the character at the code point "a".

        Monadic :# is an atomic operator.

        Examples: :#64  -->  0cA
                  :#10  -->  :"newline character"

    """
    return rec_fn(a, lambda x: KGChar(chr(x))) if is_list(a) else KGChar(chr(a))


def eval_monad_enumerate(a):
    """

        !a                                                   [Enumerate]

        Create a list of integers from 0 to a-1. !0 gives [].

        Examples: !0   -->  []
                  !1   -->  [1]
                  !10  -->  [0 1 2 3 4 5 6 7 8 9]

    """
    return np.arange(a)


def eval_monad_expand_where(a):
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
    return np.concatenate([np.zeros(x, dtype=int) + i for i,x in enumerate(a if is_list(a) else [a])])


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


def eval_monad_floor(a):
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
    return vec_fn(a, lambda x: np.floor(np.asarray(x, dtype=float)))


def eval_monad_format(a):
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
    return f":{a}" if isinstance(a, KGSym) else vec_fn(a, eval_monad_format) if is_list(a) else str(a)


def eval_monad_grade_up(a):
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
    return kg_argsort(str_to_chr_arr(a) if isinstance(a,str) else a)


def eval_monad_grade_down(a):
    """

        >a                                                  [Grade-Down]

        See [Grade-Up].

    """
    return kg_argsort(str_to_chr_arr(a) if isinstance(a,str) else a, descending=True)


def eval_monad_groupby(a):
    """

        =a                                                       [Group]

        Return a list of lists ("groups") where each group contains the
        index of each occurrence of one element within "a". "a" must be
        a list or string. The indices of all elements of "a" that are
        equal according to "~" (Match) will appear in the same group in
        ascending order.

        ="" and =[] will yield [].

        Examples:   =[1 2 3 4]  -->  [[0] [1] [2] [3]]
                  ="hello foo"  -->  [[0] [1] [2 3] [4 7 8] [5] [6]]

    """
    q = np.asarray(str_to_chr_arr(a) if isinstance(a, str) else a)
    if len(q) == 0:
        return q
    a = q.argsort()
    u = np.unique(q[a], return_index=True)
    r = np.split(a, u[1][1:])
    return np.asarray(r, dtype=object)


def eval_monad_list(a):
    """

        ,a                                                        [List]

        "," packages any object in a single-element list.

        Examples:    ,1  -->  [1]
                  ,:foo  -->  [:foo]
                  ,"xyz" -->  ["xyz"]
                   ,[1]  -->  [[1]]
    """
    return str(a) if isinstance(a, KGChar) else np.asarray([a],dtype=object) # np interpets ':foo" as ':fo"


def eval_monad_negate(a):
    """

        -a                                                      [Negate]

        Return 0-a; "a" must be a number.

        "-" is an atomic operator.

        Examples:    -1  -->  -1
                  -1.23  -->  -1.23

    """
    return vec_fn(a, lambda x: np.negative(np.asarray(x, dtype=object)))


def eval_monad_not(a):
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
        return 1 if is_empty(x) else 0 if is_dict(x) or isinstance(x, (KGFn, KGSym)) else kg_truth(np.logical_not(np.asarray(x, dtype=object)))
    return vec_fn(a, _neg) if not is_empty(a) else _neg(a)


def eval_monad_range(a):
    """

        ?a                                                       [Range]

        Return a list containing unique elements from "a" in order of
        appearance. "a" may be a list or string.

        Examples:   ?[1 2 3 4]  -->  [1 2 3 4]
                  ?[1 1 1 2 2]  -->  [1 2]
                  ?"aaabbcccd"  -->  "abcd"

    """
    return ''.join(np.unique(str_to_chr_arr(a))) if isinstance(a, str) else np.unique(a)


def eval_monad_reciprocal(a):
    """

        %a                                                  [Reciprocal]

        Return 1%a. "a" must be a number.

        "%" is an atomic operator.

        Examples:    %1  -->  1.0
                     %2  -->  0.5
                   %0.1  -->  10.0

    """
    return vec_fn(a, lambda x: np.reciprocal(np.asarray(x,dtype=float)))


def eval_monad_reverse(a):
    """

        |a                                                     [Reverse]

        Return a new list/string that contains the elements of "a" in
        reverse order. When "a" is neither a list nor a string, return
        it unchanged.

        Examples:       |[1 2 3]  -->  [3 2 1]
                  |"hello world"  -->  "dlrow olleh"
                              |1  -->  1

    """
    return a[::-1]


def eval_monad_shape(a):
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

    def _a(x): # use numpy's natural shape by replacing all strings with arrays
        return np.asarray([np.empty(len(y)) if isinstance(y,str) else (_a(y) if is_list(y) else y) for y in x])
    return 0 if is_atom(a) else np.asarray([len(a)]) if isinstance(a,str) else np.asarray(_a(a).shape)


def eval_monad_size(a):
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
    return np.abs(a) if is_number(a) else ord(a) if is_char(a) else len(a)


def eval_monad_transpose(a):
    """

        +a                                                   [Transpose]

        Return the transpose of the matrix (2-array) "a".

        Examples:     +[[1] [2] [3]]  -->  [[1 2 3]]
                  +[[1 2 3] [4 5 6]]  -->  [[1 4] [2 5] [3 6]]
                                 +[]  -->  []

    """
    return np.transpose(np.asarray(a))


def eval_monad_undefined(a):
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
    return kg_truth(a is None or (np.isinf(a) if is_number(a) else False))


def create_monad_functions(klong):
    def _get_name(s):
        s = s.strip()
        return s[:s.index('a')]

    registry = {}

    m = sys.modules[__name__]
    for x in filter(lambda n: n.startswith("eval_monad_"), dir(m)):
        fn = getattr(m,x)
        name = _get_name(fn.__doc__)
        registry[name] = fn

    return registry
