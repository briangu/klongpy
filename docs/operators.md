# Operator Reference

## Atomic dyads and nested lists

Atomic dyads (`+`, `-`, `*`, `%`, `:%`, `!`, `^`, `&`, `|`, `=`, `<`, `>`)
combine list elements pairwise, starting at the outermost dimension. When one
operand is a scalar, it applies to every element of the other operand, recursively.
Two lists must have equal lengths at every level where both operands are lists;
a singleton list is still a list, not a scalar.

```klong
[1 2 3]+[3 3]:^[1 2 3 4 5 6 7 8 9]
:" [[2 3 4] [6 7 8] [10 11 12]]"

[10 20]+[[1 2 3] [4 5 6]]
:" [[11 12 13] [24 25 26]]"
```

For rectangular numeric arrays, this pairs shared leading dimensions and extends
the lower-rank operand across the remaining dimensions. The rule applies to
interpreted and compiled Klong expressions. Python functions called through
interop retain their own behavior: NumPy operations use NumPy's trailing-axis
broadcasting. See the [Klong reference, semantics of arguments](https://t3x.org/klong/klong-ref.txt.html)
and [discussion #82](https://github.com/briangu/klongpy/discussions/82).

## Monads (Single Argument)

| Operator | Name | Description |
|----------|------|-------------|
| `@a` | Atom | Check if value is an atom |
| `:#a` | Char | Convert to character |
| `!a` | Enumerate | Generate sequence 0..a-1 |
| `&a` | Expand/Where | Indices where `a` is true (repeated by the count in `a`); `&3` is `[0 0 0]` |
| `*a` | First | First element |
| `_a` | Floor | Floor/truncate |
| `$a` | Format | Convert to string; on a list, converts each element (`$[1 2 3]` is `["1" "2" "3"]`) |
| `>a` | Grade-Down | Indices that would sort descending |
| `<a` | Grade-Up | Indices that would sort ascending |
| `=a` | Group | Indices of equal elements, one list per distinct value, in order of first appearance |
| `,a` | List | Wrap in list |
| `-a` | Negate | Negate value |
| `~a` | Not | Logical not |
| `%a` | Reciprocal | 1/a |
| `?a` | Range | Unique elements in order of first appearance (not sorted). For a random number use `.rn()` |
| `|a` | Reverse | Reverse order; an atom is returned unchanged |
| `^a` | Shape | Array dimensions |
| `#a` | Size | Length/count |
| `+a` | Transpose | Transpose matrix |
| `:_a` | Undefined | Check if undefined |
| `∇a` | Grad | Gradient function of `a`: by autograd on the torch backend, by finite differences otherwise. Glyph only |
| `˙a` | Track | Identity. Reserved for marking a value for gradient tracking; it returns its argument. No ASCII form |

## Dyads (Two Arguments)

| Operator | Name | Description |
|----------|------|-------------|
| `a:=b` | Amend | Amend array or string at index; see [Indexing and amend](#indexing-and-amend) |
| `a:-b` | Amend-in-Depth | Deep amend |
| `a:_b` | Cut | Cut array at indices |
| `a::b` | Define | Define variable. A list of names on the left destructures, and `f(a;b)::{...}` defines a function with named parameters; see [Language forms](#language-forms) |
| `a%b` | Divide | Division; a zero divisor gives `:undefined` |
| `a_b` | Drop | Drop elements |
| `a=b` | Equal | Equality test |
| `a?b` | Find | Find element |
| `a:$b` | Form | Convert string `b` to the type of `a`; `:undefined` if it does not convert |
| `a$b` | Format2 | Format with width: a positive `a` pads on the right, a negative `a` pads on the left; `a` of the form `5.2` formats a real |
| `a@b` | Index/Apply | Index or apply function; negative indices count from the end. On a dictionary, the value of key `b` (`:undefined` if missing), or the list of values for a list of keys; see [Dictionaries](#dictionaries) |
| `a:@b` | Index-in-Depth | Deep indexing |
| `a:%b` | Integer-Divide | Integer division; a zero divisor gives `:undefined` |
| `a,b` | Join | Concatenate. A two-member list joined to a dictionary is an entry, and two dictionaries merge; see [Dictionaries](#dictionaries) |
| `a<b` | Less | Less than |
| `a~b` | Match | Deep equality |
| `a\|b` | Max/Or | Maximum or logical or |
| `a&b` | Min/And | Minimum or logical and |
| `a-b` | Minus | Subtraction |
| `a>b` | More | Greater than |
| `a+b` | Plus | Addition |
| `a^b` | Power | Exponentiation |
| `a:^b` | Reshape | Reshape array |
| `a!b` | Remainder | Modulo; a zero divisor gives `:undefined` |
| `a:+b` | Rotate | Rotate elements |
| `a:#b` | Split | Split `b` into pieces of size `a` (or of the sizes in list `a`); sizes must be 1 or more |
| `a#b` | Take | Take elements; an atom `b` is repeated (`3#7` is `[7 7 7]`) |
| `a*b` | Times | Multiplication |
| `a∇b` | Grad | Numeric gradient of function `b` at `a`. ASCII: `.grad(b;a)` |
| `a:>b` | Autograd | Gradient of a at b (PyTorch autograd) |

## Gradient Operators

`∇` and `:>` compute gradients and `∂` computes Jacobians. On the numpy backend all three use finite differences; the torch backend uses autograd for `:>`, `∂` and the monad `∇`. The torch examples (`6.0` instead of `6.000000000838668`) are on [PyTorch Backend and Autograd](torch_backend.md). The [comparison table](#comparison) is the summary.

### Numeric Gradient: `∇` (nabla)

The dyad `point∇function` (and `.grad(function;point)`) **always** computes the gradient by numeric differentiation (finite differences), whichever backend is active. The monad `∇function` is different: it returns a gradient function that uses autograd on the torch backend and finite differences otherwise (see below).

**Dyad syntax:** `point∇function`

```kgpy
?> f::{x^2}
:monad
?> 3∇f
6.000000000838668
?> g::{+/x^2}
:monad
?> [1 2 3]∇g
[2.000000000279556 3.9999999996709334 6.000000000838668]
```

The first result is the derivative of `x^2` at 3 (about 6) and the second the gradient of the sum of squares (about `[2 4 6]`); finite differences leave an error near 1e-9 with float64 (NumPy, or torch on the CPU). On a torch device that computes in float32, such as a GPU, the error is near 1e-2: `3∇f` prints `5.993843`.

**ASCII form:** `.grad(function;point)` is the same computation with the function first, for keyboards without `∇`. The point is taken by value.

```kgpy
?> f::{x^2}
:monad
?> 3∇f
6.000000000838668
?> .grad(f;3)
6.000000000838668
?> .grad({+/x^2};[1 2 3])
[2.000000000279556 3.9999999996709334 6.000000000838668]
```

**Monad syntax:** `∇function` returns a gradient function. A name cannot contain `_` (it is the Drop operator), so the example calls it `gf`. On the numpy backend the result is by finite differences; on the torch backend it is by autograd and gives `6.0`.

```kgpy
?> f::{x^2}
:monad
?> gf::∇f
:monad
?> gf(3)
6.000000000838668
```

### PyTorch Autograd: `:>`

`function:>point` uses autograd on the torch backend and finite differences on numpy. The examples show the numpy result; on torch they are `6.0` and `[2.0 4.0 6.0]` (float32 tensors).

**Syntax:** `function:>point`

```kgpy
?> f::{x^2}
:monad
?> f:>3
6.000000000838668
?> g::{+/x^2}
:monad
?> g:>[1 2 3]
[2.000000000279556 3.9999999996709334 6.000000000838668]
```

### Multi-Parameter Gradients

A list of symbols on the right gives one gradient per symbol:

```kgpy
?> w::2.0;b::3.0
3.0
?> loss::{(w^2)+(b^2)}
:nilad
?> loss:>[w b]
[3.9999999996709334 6.000000000838668]
```

This is the numpy result; torch returns `[4.0 6.0]`. Rules:

- A single symbol returns the gradient itself, not a one-element list: `loss:>w` is `3.9999999996709334` on numpy and `4.0` on torch.
- A symbol that holds a vector gives the gradient vector.
- When the function uses its argument (`f::{x^2}`), a symbol on the right is a variable holding the point: `p::3.0` then `f:>p` is `f:>3.0`.

### Jacobian Operator: `∂`

The `∂` (partial derivative) operator computes the Jacobian matrix:

**Syntax:** `point∂function`

```kgpy
?> f::{x^2}
:monad
?> [1 2]∂f
[[2.000000000002 0.0] [0.0 4.000000000115023]]
?> g::{+/x^2}
:monad
?> [1 2 3]∂g
[[2.000000000279556 3.9999999996709334 6.000000000838668]]
```

These are the numpy results (finite differences). On torch the first is the exact diagonal Jacobian `[[2.0 0.0] [0.0 4.0]]` and the second is `[2.0 4.0 6.0]`.

Also available as `.jacobian(function;point)`.

### Multi-Parameter Jacobians

A list of symbols gives one Jacobian per symbol:

```kgpy
?> w::[1.0 2.0]
[1.0 2.0]
?> b::[3.0 4.0]
[3.0 4.0]
?> f::{w^2}
:nilad
?> [w b]∂f
[[[2.000000000002 0.0] [0.0 4.000000000115023]] [[0.0 0.0] [0.0 0.0]]]
```

`f` returns `[w0^2 w1^2]`, so the result is the list `[J_w J_b]`; `b` does not appear in `f`, so its Jacobian is zero. This is the numpy result; on torch the values are `2.0`, `4.0` and `0.0`.

### Comparison

The two gradient operators take their arguments in opposite orders: `point∇function` for numeric differentiation and `function:>point` for autograd. The ASCII names take the function first, like `:>`. This table is the one authoritative comparison; other pages link here.

| Feature | `∇` (nabla) | `:>` (autograd) | `∂` (jacobian) |
|---------|-------------|-----------------|----------------|
| Method | The dyad `point∇function` and `.grad`: finite differences on every backend. The monad `∇function`: autograd on torch, finite differences on numpy | Autograd on torch, finite differences on numpy | Autograd on torch, finite differences on numpy |
| Precision | The dyad is approximate: about 1e-9 in float64 (NumPy, torch on the CPU), about 1e-2 in float32 (torch on a GPU) | Exact up to float32 rounding on torch; approximate on numpy | Same as `:>` |
| Output | Scalar functions | Scalar functions | Vector functions |
| Syntax | `point∇function` | `function:>point` | `point∂function` |
| ASCII name | `.grad(function;point)` | `function:>point` | `.jacobian(function;point)` |
| Multi-param | No | `f:>[w b]` | `[w b]∂f` |

### Autograd System Functions

Five system functions need the PyTorch backend: `.gradcheck`, `.compile`, `.compilex`, `.cmodes` and `.export`. `.gradcheck` also needs the CPU device. [PyTorch Backend and Autograd](torch_backend.md#function-compilation-and-gradient-checks) describes them and has an example.

## Adverbs

| Adverb | Name | Description |
|--------|------|-------------|
| `f'a` | Each | Apply f to each element |
| `f@'a` | Each-Index | Apply f to [index;element] pairs |
| `f/a` | Over | Reduce with f |
| `f\a` | Scan | Scan with f |
| `a f'b` | Each-2 | Apply f to corresponding elements of `a` and `b`, like a zip (`[1 2 3]{x+y}'[10 20 30]` is `[11 22 33]`) |
| `f:'a` | Each-pair | Apply f to each pair of neighbours (`{x,y}:'[1 2 3 4]` is `[[1 2] [2 3] [3 4]]`) |
| `n f:*a` | Iterate | Apply f to a, n times; n must be a non-negative integer |
| `n f\*a` | Scan-Iterate | Like Iterate, returning every intermediate value |
| `f:~a` | Converge | Apply f until the result stops changing |
| `f\~a` | Scan-Converge | Like Converge, returning every intermediate value |

### Each-Index Adverb: `@'`

The `@'` adverb applies a function to `[index;element]` pairs, similar to Python's `enumerate()`:

```klong
:" Get indices"
{x@0}@'[10 20 30]           :" [0 1 2]"

:" Get values (same as regular each)"
{x@1}@'[10 20 30]           :" [10 20 30]"

:" Use both index and value"
{(x@0)*(x@1)}@'[10 20 30]   :" [0 20 60]"

:" Add index to each value"
{(x@0)+(x@1)}@'[100 200 300] :" [100 201 302]"
```

### Evaluated Array Constructors: `[;...]`

The `[;expr1;expr2;...]` syntax creates arrays by evaluating each expression:

```klong
:" Basic expressions"
[;1+1;2+2;3+3]              :" [2 4 6]"

:" With variables and function calls"
a::10; b::20
[;a;b;a+b]                  :" [10 20 30]"

:" With function results"
avg::{(+/x)%#x}
arr::[1 2 3 4 5]
[;avg(arr);+/arr;#arr]       :" [3.0 15 5]"

:" Inside a function"
make::{[;x;x*2;x*3]}
make(5)                      :" [5 10 15]"
```

## Conditionals

`:[c;a;b]` evaluates `c`. If `c` is true it evaluates and returns `a`, otherwise `b`. Only the chosen branch is evaluated.

A condition is false if it is a number equal to 0 (including `0.0`) or an empty value (`[]` or `""`). Everything else is true, including `:foo` and the empty dictionary.

```
:[1;"yes";"no"]              :" yes"
:[0;"yes";"no"]              :" no"
:[[];"yes";"no"]             :" no, the empty list is false"
```

### Else-if chains

`:|` in place of the `;` before a branch starts the next test: `:[c1;a:|c2;b;d]` is short for `:[c1;a;:[c2;b;d]]`. Both forms parse to the same program. Chains can be any length.

```
:[0;1:|1;2;3]                :" 2"
:[0;1:|0;2;3]                :" 3"

sign::{:[x<0;"neg":|x=0;"zero";"pos"]}
sign(-5)                     :" neg"
sign(0)                      :" zero"
sign(7)                      :" pos"
```

The chain form only replaces a conditional in the else (last) slot. A conditional in the then slot, as in `:[c1;:[c2;a;b];d]`, stays nested.

## Dictionaries

A dictionary literal is `:{[key value] [key value]}`, and `:{}` is the empty dictionary. The entries of that form are not evaluated. The evaluated form, `:{;key;value;key;value}`, evaluates its expressions from left to right each time it is evaluated; `:{;}` is the empty dictionary. A dictionary literal inside a dictionary or list literal is a dictionary.

| Expression | Result |
|---|---|
| `d?k` or `d@k` | The value of key `k`, or `:undefined` when `k` is missing. Not range checked. |
| `d@[k1 k2]` | The list of values, with `:undefined` for a missing key. |
| `:_d@k` | 1 when the key is missing. |
| `d,d2` | Copies the entries of `d2` into `d`, which changes in place; an entry of `d2` replaces one with the same key. |
| `d,[k v]`, `d,k,,v`, `[k v],d` | Adds the entry. A list of two members joined to a dictionary, on either side, is an entry. |
| `l,,d` | Appends the dictionary `d` to the list `l`. `l,d` does not: for a list of two members it is the pair rule above. |
| `$d` | The written form, as a string. |

```kgpy
?> n::7
7
?> d:::{;"n";n;"sq";n*n}
:{["n" 7] ["sq" 49]}
?> d@["n" "sq" "no"]
[7 49 :undefined]
?> d,:{["sq" 0] ["z" 1]}
:{["n" 7] ["sq" 0] ["z" 1]}
?> [1 2],,d
[1 2 :{["n" 7] ["sq" 0] ["z" 1]}]
```

A list cannot be a key, and an odd number of expressions in `:{;...}` is a syntax error. [Language Additions](language-additions.md#dictionaries) has the details with the classic form beside each.

## Language Forms

Two forms bind names with `::`. Both are described with examples, and with the classic equivalent, in [Language Additions](language-additions.md).

| Form | Meaning |
|---|---|
| `[a b c]::v` | Destructuring. Binds each name to the member of `v` in the same position; the count must match. As the first statement of a function, `{[a b]::x;...}`, the names are local to it. |
| `f(a;b;c;d)::{...}` | Defines `f` with the named parameters `a b c d`, any number of them. Too many arguments is an arity error; too few is a projection. `x`, `y` and `z` cannot be used in the body. |

A function's arity, which the display (`:monad`, `:dyad`, `:triad`, and `:fn/N` for four or more named parameters) shows and an adverb dispatches on, is the highest of `x`, `y` and `z` its body uses, not counting nested function literals, or the number of named parameters. `{-x}` is a monad and `{y}` a dyad.

```kgpy
?> f::{-x}
:monad
?> g::{y}
:dyad
?> h::{x+z}
:triad
?> k(a;b;c;d)::{a+b+c+d}
:fn/4
```

## Edge Cases and Errors

These rules apply to the interpreted and compiled paths and to the NumPy backend. In the transcripts below, an error is shown as `error: message`; the REPL adds its own prefix.

### Range and group order

`?a` and `=a` keep the order in which values first appear. Neither sorts.

```kgpy
?> ?[3 1 3 2 1]
[3 1 2]
?> ?"abcab"
"abc"
?> =[3 1 3 2 1]
[[0 2] [1 4] [3]]
?> ="abca"
[[0 3] [1] [2]]
```

### Indexing and amend

`a@i` accepts `-n..n-1` for a list or string of length `n`; a negative index counts from the end, as in NumPy. Any other index is an error. A list of indices is checked the same way.

```kgpy
?> [1 2 3]@-1
3
?> "abc"@[0 -1]
"ac"
?> [1 2 3]@3
error: index 3 out of range for list of 3
?> [1 2 3]@-4
error: index -4 out of range for list of 3
```

`a:=v,i` uses the same index rule. On a string, index `#a` appends and any larger index is an error. A real value put into an integer list turns the whole list into reals.

```kgpy
?> [1 2 3]:=9,-1
[1 2 9]
?> [1 2 3]:=9,3
error: :=: index 3 out of range for length 3
?> "abc":="xy",3
"abcxy"
?> "abc":="x",4
error: :=: index 4 out of range for length 3
?> [1 2 3]:=2.5,1
[1.0 2.5 3.0]
```

Dictionary lookups are not range checked: `d@k` for a key that is missing is `:undefined`.

### Division by zero

`%`, `:%` and `!` give `:undefined` for a zero divisor, one element at a time when an operand is a list. The result list then holds `:undefined` objects, and arithmetic on such a list raises an error.

```kgpy
?> 1%0
:undefined
?> [1 2 0]%[1 0 1]
[1.0 :undefined 0.0]
?> [5 6]![0 2]
[:undefined 0]
?> 1:%0
:undefined
```

### Shape mismatch

Two lists of different lengths in an atomic dyad are an error. So is filling a non-empty shape from an empty list.

```kgpy
?> [1 2 3]+[1 2]
error: cannot apply + (Plus) to list of 3 and list of 2
?> [2 3]:^[]
error: :^: cannot fill a non-empty shape from an empty source
```

### Format and Form

`$` converts an atom to a string and applies to each element of a list. `a:$b` converts the string `b` to the type of `a`, giving `:undefined` where the text does not convert, one element at a time for a list. `n$x` with a width of `5.2` needs a real `x`.

```kgpy
?> $[1 2 3]
["1" "2" "3"]
?> 1:$"12"
12
?> 1:$"x"
:undefined
?> [1 1]:$["12" "x"]
[12 :undefined]
?> 1.0:$"1.5"
1.5
?> 10$123
"123       "
?> (-10)$123
"       123"
?> 5.2$3.0
"    3.00"
?> 5.2$3
error: $: a real width needs a real number to format, got 3
```

### Split and iterate counts

Split sizes must be integers of 1 or more, and the size list may not be empty. The count of an iterate or scan-iterate must be an integer of 0 or more; an integral real such as `2.0` is accepted. A count of 0 returns the argument itself, for scan-iterate as well (not a one-element list).

```kgpy
?> 2:#[1 2 3 4 5]
[[1 2] [3 4] [5]]
?> [2 1]:#[1 2 3]
[[1 2] [3]]
?> 0:#[1 2 3]
error: :#: segment sizes must be positive, got 0
?> []:#[1 2 3]
error: :#: segment sizes must not be empty
?> 3{1,x}:*[]
[1 1 1]
?> 2.0{x+1}:*0
2
?> 3{x+1}\*0
[0 1 2 3]
?> 0{x+1}\*5
5
?> (-1){x+1}:*0
error: :*: count must not be negative, got -1
?> 1.5{x+1}:*0
error: :*: count must be a non-negative integer, got 1.5
```

### Take on nested lists

`n#a` on a list of lists takes whole rows, not elements of the rows. If `n` is larger than the number of rows, the rows repeat from the start. A negative `n` takes `|n|` rows and the result ends with the last row; when `|n|` is larger than the number of rows, the rows repeat.

```kgpy
?> 5#[[1 2] [3 4]]
[[1 2] [3 4] [1 2] [3 4] [1 2]]
?> 2#[[1 2] [3 4] [5 6]]
[[1 2] [3 4]]
?> (-3)#[[1 2] [3 4]]
[[3 4] [1 2] [3 4]]
?> 0#[[1 2] [3 4]]
[]
?> 5#[1 2]
[1 2 1 2 1]
```

### Integers beyond 64 bits

Integers are 64 bits wide. A sum, difference or product of two such integers that does not fit wraps around, as it does in NumPy. An operand that is itself outside 64 bits (a long literal, or the result of `2^64`) makes `+ - * < > | &` exact, while `%` gives a float:

```kgpy
?> 9223372036854775807+1
-9223372036854775808
?> 1+9223372036854775808
9223372036854775809
?> [1 2]+9223372036854775808
[9223372036854775809 9223372036854775810]
?> 9223372036854775808*3
27670116110564327424
?> 9223372036854775808<9223372036854775809
1
```

`^` of two integer scalars with a non-negative exponent is exact (`10^400` has all its digits) unless the result would need more than a million bits, which is an error. A negative or fractional exponent gives a float, and an error where the result does not fit a float. `!` is an error for an operand that does not fit 64 bits. Floats are doubles.

```kgpy
?> 2^64
18446744073709551616
?> 10^100000000
error: cannot apply ^ (Power) to integer and integer
?> 9223372036854775808!2
error: cannot apply ! (Remainder) to integer and integer
```

### Take of an atom

`n#x` with a number or a symbol `x` repeats it. The result is a list of `n` copies, and `0#7` is empty.

```kgpy
?> 3#7
[7 7 7]
?> 0#7
[]
?> 3#2.5
[2.5 2.5 2.5]
?> 2#:foo
[:foo :foo]
?> 3#"a"
"aaa"
```

### Indexing is `@`, not `q[1]`

Square brackets build lists. They do not index, so `q[1]` is the variable `q` followed by the list `[1]`. The parser reads `q`, warns that `[1]` is left over (see [Syntax warnings](errors.md#syntax-warnings)) and returns `q`. Inside parentheses or a function body it is an error, with a hint. Write `q@1`.

```kgpy
?> q::[10 20 30]
[10 20 30]
?> q@1
20
?> q[1]
[10 20 30]
?> {q[1]}
error: expected '}' but found '['
```

### Whitespace in a conditional

Spaces are allowed before and after the `;` that separates the parts of a conditional.

```kgpy
?> :[1 ;"a";"b"]
"a"
?> :[0 ;"a" ;"b"]
"b"
?> :[1 = 1 ; "yes" ; "no"]
"yes"
```

### An adverb touches its verb

An adverb must follow its verb with no space between them. A space is a syntax error at every strict level, and it is found when the text is parsed, so a function that contains one fails when it is defined.

```kgpy
?> f::{x+1}
:monad
?> f'[1 2]
[2 3]
?> f '[1 2]
error: adverb ' must directly follow its verb
?> {x+1} '[1 2]
error: adverb ' must directly follow its verb
?> (f)'[1 2]
error: adverb ' does not follow a verb
```

### Symbols are data

A symbol held in a list or a variable stays a symbol when it passes through an adverb, `@`, a projection or a Python or IPC callback. It is not looked up as a variable. To pass the values of variables, build the list from values with `[;a;b]`.

```kgpy
?> foo::5
5
?> l::[:foo :bar]
[:foo :bar]
?> {x}'l
[:foo :bar]
?> {x}@*l
:foo
?> {x}'[;foo;:bar]
[5 :bar]
?> s:::baz
:baz
?> {x}'s
:baz
```

Each over a symbol `s` is `f(s)`: the symbol is not iterated as a string.

### Projections

A call that does not supply every argument is a projection: a function of the arguments that are missing. An argument is left empty, `f(1;)` and `f(;2)`, or left off the end, `f(1)`; arguments left off are the trailing ones, so for a function of three arguments `f(1)`, `f(1;)` and `f(1;;)` are the same projection. This holds at every strict level, for functions of `x`, `y` and `z` and for functions with named parameters, and when `@` or an adverb applies a function to fewer values than it takes. A projection is displayed by the number of arguments it still takes.

```kgpy
?> f::{x,y,z}
:triad
?> p::f(1)
:dyad
?> q::p(;3)
:monad
?> q(2)
[1 2 3]
?> f@1
:dyad
```

Calling a projection fills its open slots in order, starting at the first one, and a slot the call leaves empty stays open. The function is looked up by name when the projection is completed, so redefining `f` changes what `p` calls.

A projection made inside a function holds the values its arguments had at that point. One made at top level keeps the argument expressions and evaluates them each time it is completed:

```kgpy
?> add::{x+y}
:dyad
?> {x(10)}'{add(x)}'[1 2 3]
[11 12 13]
?> a::1
1
?> p::add(a)
:monad
?> a::5
5
?> p(10)
15
```

Three cases are not projections. A function whose whole body is a projection, `g::{add(1;)}`, passes its call on, so `g(5)` is `add(1;5)`. A call through `.f` is never short: `.f(1)` in a function of two arguments runs it with `y` as the caller left it. A function with named parameters applied to too few values by `@` or an adverb is an arity error, as is a call from Python with too few arguments.

### Modules

`.module(:name)` opens a module and `.module(0)` closes it. A name assigned between the two belongs to the module, and every use of that name inside the module refers to the module's own variable. This matches Klong, whose test suite KlongPy runs.

The names are not hidden. After the module is closed, a module name is still reachable by its plain name, as long as the program has no variable of its own with that name:

```kgpy
?> .module(:geo);unit::10;scale::{x*unit};.module(0)
None
?> scale(4)
40
?> unit
10
```

Assigning the name outside the module makes a separate global variable. The module's functions keep using the module's own, so they cannot be broken by a program that happens to use the same name:

```kgpy
?> .module(:geo);unit::10;scale::{x*unit};.module(0)
None
?> unit::2
2
?> scale(4)
40
?> scale::{x}
:monad
?> unit,scale(4)
[2 4]
```

A global that exists before the module opens does not stop the module from having its own variable of that name, and afterwards the plain name reads the module's value: `v::1;.module(:m);v::5;.module(0);v` is 5. There is no qualified form such as `geo.unit`; the libraries in `klongpy/lib` name what they export with a prefix (`csv.read`) by convention only, and their helper functions without a prefix (`unesc`, `escape` in `csv.kg`) are reachable by their plain names too. When two modules define the same name, the plain name outside refers to the one loaded last.

### The undefined value

`:undefined` is what division by zero, a missing dictionary key and a failed Form give, and it can be written in a program. `:_a` and `a~:undefined` both test for it:

```kgpy
?> u::1%0
:undefined
?> u~:undefined
1
?> :_:undefined
1
```

## Unused Operators

The following operator combinations are reserved for future use:

```
:! :& :, :< :?
```

## See also

[Conformance Suite](conformance.md) for what the operators are tested against, [Differences from Klong](differences.md), [Roadmap](roadmap.md).
