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
| `∇a` | Grad | Numeric gradient function of a |

## Dyads (Two Arguments)

| Operator | Name | Description |
|----------|------|-------------|
| `a:=b` | Amend | Amend array or string at index; see [Indexing and amend](#indexing-and-amend) |
| `a:-b` | Amend-in-Depth | Deep amend |
| `a:_b` | Cut | Cut array at indices |
| `a::b` | Define | Define variable |
| `a%b` | Divide | Division; a zero divisor gives `:undefined` |
| `a_b` | Drop | Drop elements |
| `a=b` | Equal | Equality test |
| `a?b` | Find | Find element |
| `a:$b` | Form | Convert string `b` to the type of `a`; `:undefined` if it does not convert |
| `a$b` | Format2 | Format with width: a positive `a` pads on the right, a negative `a` pads on the left; `a` of the form `5.2` formats a real |
| `a@b` | Index/Apply | Index or apply function; negative indices count from the end |
| `a:@b` | Index-in-Depth | Deep indexing |
| `a:%b` | Integer-Divide | Integer division; a zero divisor gives `:undefined` |
| `a,b` | Join | Concatenate |
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
| `a#b` | Take | Take elements |
| `a*b` | Times | Multiplication |
| `a∇b` | Grad | Numeric gradient of b at a |
| `a:>b` | Autograd | Gradient of a at b (PyTorch autograd) |

## Gradient Operators

KlongPy provides two gradient operators:

### Numeric Gradient: `∇` (nabla)

The `∇` operator **always** computes gradients using numeric differentiation (finite differences), regardless of which backend is active.

**Dyad syntax:** `point∇function`

```klong
f::{x^2}
3∇f               :" Returns ~6.0 (derivative of x^2 at x=3)

g::{+/x^2}
[1 2 3]∇g         :" Returns [2 4 6] (gradient of sum of squares)
```

**Monad syntax:** `∇function` returns a gradient function

```klong
f::{x^2}
grad_f::∇f        :" Create gradient function
grad_f(3)         :" Compute gradient at x=3
```

### PyTorch Autograd: `:>`

The `:>` operator uses PyTorch's automatic differentiation when the torch backend is enabled (`--backend torch`). Falls back to numeric differentiation otherwise.

**Syntax:** `function:>point`

```klong
f::{x^2}
f:>3              :" Returns 6.0 (derivative of x^2 at x=3)

g::{+/x^2}
g:>[1 2 3]        :" Returns [2 4 6] (gradient of sum of squares)
```

### Multi-Parameter Gradients

The `:>` operator can compute gradients for multiple parameters simultaneously when given a list of symbols:

```klong
w::2.0;b::3.0
loss::{(w^2)+(b^2)}

:" Single parameter (returns gradient)
loss:>w               :" Returns 4.0

:" Multiple parameters (returns list of gradients)
loss:>[w b]           :" Returns [4.0 6.0]
```

### Jacobian Operator: `∂`

The `∂` (partial derivative) operator computes the Jacobian matrix:

**Syntax:** `point∂function`

```klong
f::{x^2}              :" Element-wise square
[1 2]∂f               :" [[2 0] [0 4]] - diagonal Jacobian

g::{+/x^2}            :" Sum of squares (scalar output)
[1 2 3]∂g             :" [2 4 6] - gradient vector
```

Also available as `.jacobian(function;point)`.

### Multi-Parameter Jacobians

Like gradients, Jacobians can be computed for multiple parameters using a list of symbols:

```klong
w::[1.0 2.0]
b::[3.0 4.0]
f::{w^2}              :" Returns [w0^2, w1^2]

:" Compute Jacobians for both w and b
[w b]∂f               :" Returns [J_w, J_b]
```

### Comparison

The two gradient operators take their arguments in opposite orders: `point∇function` for numeric differentiation and `function:>point` for autograd. This table is the one authoritative comparison; other pages link here.

| Feature | `∇` (nabla) | `:>` (autograd) | `∂` (jacobian) |
|---------|-------------|-----------------|----------------|
| Method | Numeric | PyTorch autograd | PyTorch/numeric |
| Precision | Approximate | Exact | Exact |
| Output | Scalar functions | Scalar functions | Vector functions |
| Syntax | `point∇function` | `function:>point` | `point∂function` |
| Multi-param | No | `f:>[w b]` | `[w b]∂f` |

### Autograd System Functions

Additional system functions for autograd (require PyTorch backend):

| Function | Description |
|----------|-------------|
| `.gradcheck(fn;input)` | Verify gradients against numeric computation |
| `.compile(fn;input)` | Compile function for optimized execution |
| `.compilex(fn;input;opts)` | Compile with extended options (mode, backend) |
| `.cmodes()` | Query available compilation modes and backends |
| `.export(fn;input;path)` | Export computation graph to file |

**Gradient Verification:**
```klong
f::{x^2}
.gradcheck(f;3.0)     :" Returns 1 if gradients are correct
```

**Function Compilation:**
```klong
f::{(x^3)+(2*x^2)+x}
cf::.compile(f;2.0)   :" Returns compiled (faster) function
cf(5.0)               :" Execute compiled function
```

**Extended Compilation:**
```klong
:" Fast compile for development
cf::.compilex(f;2.0;:{["mode" "reduce-overhead"]})

:" Maximum optimization for production
cf::.compilex(f;2.0;:{["mode" "max-autotune"]})

:" Debug mode (no C++ compiler needed)
cf::.compilex(f;2.0;:{["backend" "eager"]})

:" Query available options
info::.cmodes()
```

**Graph Export:**
```klong
info::.export(f;2.0;"model.pt2")
.p(info@"graph")      :" Print computation graph
```

## Adverbs

| Adverb | Name | Description |
|--------|------|-------------|
| `f'a` | Each | Apply f to each element |
| `f@'a` | Each-Index | Apply f to [index;element] pairs |
| `f/a` | Over | Reduce with f |
| `f\a` | Scan | Scan with f |
| `n f'a` | Each-n | Apply f to groups of n |
| `a f'b` | Each-pair | Apply f to pairs |
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

Dictionary lookups are not range checked.

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
```

## Unused Operators

The following operator combinations are reserved for future use:

```
:! :& :, :< :?
```
