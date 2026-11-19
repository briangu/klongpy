# Differences from Klong

KlongPy runs the Klong test suite, and a program written for Klong runs unchanged apart from the cases below. This page lists where KlongPy's behaviour differs from reference Klong (Nils M Holm's implementation, described at [t3x.org/klong](https://t3x.org/klong)) and why. Each example shows what KlongPy does.

The page has three parts. The first lists differences that are intended and tested. The second points to extensions and changes of meaning that are described elsewhere. The last lists known bugs: places where KlongPy gives a wrong answer and the right one is known. A bug is not a difference, and it is listed so that nobody relies on it.

## Differences that are intended

### A list literal takes tokens

Inside `[...]` with no semicolons, the elements are the tokens as written. A name is a symbol and an operator is an operator object, so `[!10]` has two elements and `[3*a^2]` has five. Reference Klong reads the operators as symbols (`[:! 10]`); in KlongPy an operator keeps its own type and prints as itself. A comma is not a separator, so `[7,7,7]` is five tokens, the three sevens and two commas, and its size is 5.

```kgpy
?> [!10]
[! 10]
?> [3*a^2]
[3 * :a ^ 2]
?> #[7,7,7]
5
?> [7 7 7]~[7 7 7]
1
```

For a list whose elements are evaluated, start it with `;`. Each element is an expression.

```kgpy
?> a::2
2
?> [;3*a^2;a]
[12 2]
?> [;!3;4]
[[0 1 2] 4]
```

Why: a list literal is data, not code, and `[;...]` is the form that marks its elements as expressions.

### Joining a two-element list and a dictionary

`list,dict` follows the [pair rule](language-additions.md#merge-and-the-pair-rule): a list of two members joined to a dictionary is read as a key and a value to add, not as an element to append. Reference Klong appends the dictionary, so appending an empty dictionary to a list that already holds two items fails in KlongPy at the third append.

```kgpy
?> [1 2],:{}
:{[1 2]}
?> A::[]
[]
?> A::A,:{}
[:{}]
?> A::A,:{}
[:{} :{}]
?> A::A,:{}
error: a dict cannot be a dictionary key
```

Write `A,,:{}` to append a dictionary as one element whatever the length of `A`.

```kgpy
?> A::[:{} :{}]
[:{} :{}]
?> A::A,,:{}
[:{} :{} :{}]
```

Why: joining a pair onto a dictionary, `d,[k v]`, is how Klong builds one, and the rule applies from either side, so a two-member list cannot also be an element.

### A function passed as a parameter does not curry

A projection such as `x(2;;)` needs the arity of `x` when the program is read. For a function written under its own name the interpreter has it. For a parameter, it does not, so a function that curries its argument, such as `foo::{x(2;;)}`, does not give the projection that `fn(2;;)` gives when `fn` is passed to it.

```kgpy
?> fn::{x+10+y*z}
:triad
?> p::fn(2;;)
:dyad
?> p(3;5)
27
?> foo::{x(2;;)}
:dyad
?> w::foo(fn;;)
:dyad
?> w(;3;5)
:monad
```

The last result is a projection that is still waiting, not 27, the value of `fn(2;3;5)`. Apply the function to all its arguments at once, or name the projection outside the function that receives it.

Why: arity is a property of the function that the reader knows from its text, and a parameter has none until the function runs.

### Calling a value that is not a function is an error

`a(args)` where `a` holds a number, a string, a list or a dictionary is a `type` error that names `a` and says what it holds. A dictionary is read with `d@k`.

```kgpy
?> a::5
5
?> a(1)
error: cannot call a: it is an integer, not a function
```

### Numbers

Reals are doubles, where Klong's are arbitrary precision. Integers are 64 bits wide and wrap on overflow, as in NumPy. An operand beyond 64 bits (a long literal, or `2^64`) makes `+ - * < > | &` exact. `%` gives a float. `^` of two integers with a non-negative exponent is exact on a scalar (`7^40`, `10^400`) while the operand's bit length times the exponent is at most 2^20 (`2^524288` works, `2^524289` does not); beyond that the answer is a float, which overflows to a range error (`10^100000000`). A negative or fractional exponent gives a float, and a list operand is computed in 64-bit integers like the other operators (`[2 3]^40` wraps), with the exceptions below.

```kgpy
?> 9223372036854775807+1
-9223372036854775808
?> 2^64
18446744073709551616
?> 1e308*10
inf
```

Why: the numbers are NumPy's, which is what makes array operations fast. Programs that need more digits than a double has will differ.

`^` follows one rule for a list operand, in every arrangement (`list^scalar`, `scalar^list`, `list^list`): integers to a non-negative integer power are computed in 64-bit integer arithmetic and wrap, as `+` and `*` do; a negative integer exponent, or a float anywhere, gives floats. Two integer scalars are exact instead (`2^64`). With the NumPy backend:

```kgpy
?> 2^64
18446744073709551616
?> [2 3]^64
[0 8733086111712066817]
?> 2^[62 63 64]
[4611686018427387904 -9223372036854775808 0]
?> [2 3]^[64 64]
[0 8733086111712066817]
?> [2 3]^(0-1)
[0.5 0.3333333333333333]
?> 2^[0 -1]
[1.0 0.5]
?> [2.0 3]^2
[4.0 9.0]
```

Use a scalar base and exponent when the result can exceed 64 bits. A list result of whole floats stays floats (`[4 9]^0.5` is `[2.0 3.0]`), where the scalar `4^0.5` gives the integer `2`.

## Described elsewhere

These are not repeated here. Each is a change from classic Klong that is documented where it is used.

- **Function arity.** A function has as many arguments as the highest of `x`, `y` and `z` its body uses, wherever it uses them, so `{-x}` is a monad. See [How many arguments a function takes](language-additions.md#how-many-arguments-a-function-takes).
- **Names with a dot and modules.** `a.b` is `a@"b"` on any dictionary, and a file that ends with `.export` is a module that `.import` brings in. Classic Klong has neither: its `.l` writes every name of a file into the globals, and a dot is part of a name. See [Modules](language-additions.md#modules).
- **Dictionary literals.** An operator or a function as a key, and an entry that is not exactly `[key value]`, are syntax errors. See [Dictionaries](language-additions.md#dictionaries).
- **Scope.** A function with named parameters, `f(a;b)::{...}`, is lexically scoped: a free name is found in its own frame and then the globals of its file or module, never in its caller's locals. A classic function keeps Klong's dynamic scope. See [Migrating to v2](migrating-to-v2.md#named-parameters-are-lexically-scoped).
- **Calls with too few arguments are projections**, `f(1)` being `f(1;)`, at every strict level. See the [release notes](release-notes.md).
- **Language additions.** Destructuring `[a b]::v`, named parameters `f(a;b;c;d)::{...}`, evaluated dictionaries, `d@k` on dictionaries, `.fmt`, `.grad`, `.split`, `.join`, `.words`, `.trim`, `.num`. The inputs whose meaning changed are in [Syntax that changed meaning](release-notes.md#syntax-that-changed-meaning). See [Language Additions](language-additions.md).
- **Errors as values and random numbers.** `.try` and `.err` catch and raise an error, `.seed` makes `.rn` repeatable and `.now` gives the time. See [System functions](system-functions.md).
- **Strict mode and lint.** Three [strict levels](strict.md) and [`kgpy lint`](lint.md) report or reject silent mistakes. The default is the classic behaviour.
- **Errors** carry a position and a kind. See [Errors](errors.md).
- **Python, IPC, autograd and the rest of the system functions**: `.py`, `.pyf`, `.srv`, `.cli`, `:>`, `.db` and others are additions. See [Python Integration](python_integration.md), [IPC](ipc_capabilities.md) and [PyTorch Backend and Autograd](torch_backend.md).
- **A static subset that compiles to C.** See [Compiling Klong to C](embedded.md).
- **Which command.** `kg` has no Python, so `.py`, `.pyf`, `.sys`, `.web` and the other functions that need Python are errors there; `kgpy` has them. See [What Runs Where](what-runs-where.md).

## Known bugs

These are cases where KlongPy's answer is wrong and the right answer is known. Each is a case in `conformance/pending/known-differences.cases`, under the name given, with the expected result; `python conformance/run.py --pending` runs them and reports any that has started to pass. They are not differences to rely on.

```kgpy
?> "abc"@0
0ca
?> *"abc"
"a"
?> [1 2 3]@1.0
[1 2 3]
?> %"a"
error: cannot apply % (Reciprocal) to string
```

- `pending.first-of-string-is-a-char` and `pending.first-of-reversed-string-is-a-char`: the first element of a string should be a character (`0ca`, as `"abc"@0` gives), not a one-character string. `*"abc"` and `*|"hello"` are strings. The reference suite passes only because `~` treats a character and a one-character string as equal.
- `pending.index-by-float`: `[1 2 3]@1.0` should be a type error (an index must be an integer). It returns the whole list.
- `pending.reciprocal-of-string` and `pending.where-of-string`: `%"a"` and `&"a"` should be type errors, as `-"a"` is. They are reported as range errors.

## See also

- [Conformance Suite](conformance.md): the implementation-neutral cases, including the pending list that the known bugs above come from.
- [Roadmap](roadmap.md): known limits and what is planned.
- [Migrating to v2](migrating-to-v2.md) and [What's new in v2](whats-new-v2.md).
