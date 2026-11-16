# The static profile

Static Klong is the part of Klong that can be compiled ahead of time: every
type, every shape and every call target is known before the program runs. A
program inside the profile runs unchanged under KlongPy and gives the same
answers when compiled (the differences are listed [below](#differences-from-klongpy)).

```
kgpy --check-static prog.kg
```

checks a program against the profile without running it. Every problem is
reported, each with its position, the rule it breaks (the word in brackets)
and usually a hint:

```
prog.kg:3:5: static error: the count of n#a (Take) is not a compile-time constant: the result length would depend on run-time data [const-count]
  f::{y#x}
      ^^^
  hint: the count must be a literal, #a or ^a, or a global that no function assigns
```

The exit status is 1 if any file is outside the profile. `-v` lists what was
found (entries, externs, globals with their types, each function with the
argument types it was checked for); `--int64` checks integer constants
against a 64-bit `kg_int`. From Python: `klongpy.static.check_file(path)`
returns a report with `ok`, `diagnostics`, `functions`, `globals`, `externs`
and `entries`.

## Two stages

A program is split by what kind of top-level statement it is.

| Top-level statement | Stage | Meaning |
|---|---|---|
| `name::{...}` or `name(p;q)::{...}` | run time | a function to compile, if reachable from an entry |
| `name::<any other expression>` | compile time | evaluated once by KlongPy on the host; the result becomes an initialised global |
| `.extern(...)` | compile time | declares a function the target provides |
| `.entry(...)` | compile time | marks a function as a C entry point and fixes its argument types |
| `.l("file")` | compile time | loads another file under the same rules |
| any other expression | error | top-level effects have no meaning on the target |

Compile-time expressions may use the whole language, including the library,
`.py` and file I/O. That is how tables are built: `coef::{...}'!64` runs on the
host and the 64 results are emitted as data. Only the resulting value has to
fit the profile.

Run-time functions must fit the profile. Only functions an entry reaches are
compiled and checked; a function used only by compile-time definitions may
use anything. A file with no `.entry` at all is checked as a library: every
function, with arguments of unknown type.

The program must parse at strict level 2 (`kgpy --strict=2`).

## Types

| Type | C type | Notes |
|---|---|---|
| `int` | `kg_int` = `int32_t` (default) or `int64_t` (`--int64`) | booleans are ints 0 and 1 |
| `float` | `kg_float` = `float` (default) or `double` (`--float64`) | |
| `int[d1,...,dk]`, `float[d1,...,dk]` | pointer to `kg_int`/`kg_float`, row-major | every `d` a compile-time positive integer, rank 1 to 4 |
| `str` | `const char *` | only as a literal argument to an extern |

A compile-time value is accepted as a global if it is an int, a float, or a
rectangular numeric array. Python ints outside the `kg_int` range, object
arrays, strings, symbols, dicts and functions are rejected with the position
of the definition.

A global that some compiled function assigns is state and lives in RAM; one
that no function assigns is a constant, lives in flash, and counts as a
compile-time constant wherever one is required (a take count, for example).

## What a run-time function may contain

| Group | Forms |
|---|---|
| Literals | integers, floats, numeric list literals, evaluated lists `[;a;b]` of same-shaped elements |
| Names | parameters (`x y z` or named), locals, head-destructured names, globals, other compiled functions, externs |
| Arithmetic dyads | `+ - * %` (`%` always gives float), `:%` integer divide, `!` remainder, `^` power, <code>&#124;</code> max, `&` min |
| Comparison dyads | `< > =`, giving int; `~` match on equal-shaped operands, giving int |
| Monads | `-` negate, `_` floor (gives int), `%` reciprocal, `~` not, `#` size and `^` shape (arrays only; `#` is the first dimension), `*` first, <code>&#124;</code> reverse, `+` transpose (rank 2), `!n` enumerate with constant `n`, `,` wrap as one-element list |
| Structure | `a@i` index with a scalar or an int array; `n#a` take and `n_a` drop with constant `n`, negative meaning from the end; take is rank 1 only in version 1; `a,b` join; `n:+a` rotate (rank 1); `s:^a` reshape with constant `s`; `a:=v,i` amend of a rank-1 array with one scalar value and either a scalar index or an index list of known length; other forms are rejected |
| Control | `:[c;a;b]` and its chain form `:[c1;a:|c2;b;else]`; statement sequences; `name::expr` to a local or a global |
| Adverbs | `f'a` each; `a f'b` each-2; `f/a` over; `a f/b` over with seed; `f\a` scan; `a f\b` scan with seed; `f:'a` each-pair; `n f:*a` iterate; `p f:~a` while |
| Calls | compiled functions and externs with all arguments supplied |

Rejected, each with a positioned error naming the rule:

- Anything whose result length depends on run-time data: `&a` where, `a?b` find, `?a` range, `=a` group, `<a` and `>a` grade (grade is fixed-length and can be added later; it is left out of version 1 because it needs a sort routine in the runtime), `:#` split, `:_` cut, take or drop or reshape with a non-constant count.
- `f:~a` converge and the `\~` `\*` scan forms (result length unknown).
- Strings, chars, symbols, dicts, `:undefined` as run-time values.
- Functions as values: a function stored in a variable, returned, or passed as an argument other than directly as the verb of an adverb. Projections.
- Recursion, direct or mutual, including `.f`.
- `.E`, `.l`, `.py*`, channels and every other system function except `.extern`-declared names.
- A variable whose type or shape differs between two assignments, or a conditional whose branches have different shapes.
- Division of an int by zero when both are constants.

The tables below are generated from the checker's own data; each row and each
error has a stable id.

<!-- BEGIN GENERATED -->
### Operators

| Written | Row | Name | Operands | Type rule | Shape rule | Note |
|---|---|---|---|---|---|---|
| a<code>+</code>b | `add` | Plus | num , num | promote | prefix |  |
| a<code>-</code>b | `sub` | Minus | num , num | promote | prefix |  |
| a<code>*</code>b | `mul` | Times | num , num | promote | prefix |  |
| a<code>%</code>b | `div` | Divide | num , num | float | prefix | always float; a constant int divided by the constant 0 is an error |
| a<code>:%</code>b | `idiv` | Integer-Divide | int , int | int-only | prefix | truncates toward zero; KlongPy also accepts floats here, the profile does not |
| a<code>!</code>b | `rem` | Remainder | num , num | promote | prefix | the result has the sign of the dividend (C's %, fmod for floats) |
| a<code>^</code>b | `pow` | Power | num , num | power | prefix | KlongPy returns an int whenever the result is whole; the profile cannot depend on values |
| a<code>&#124;</code>b | `max` | Max/Or | num , num | promote | prefix |  |
| a<code>&amp;</code>b | `min` | Min/And | num , num | promote | prefix |  |
| a<code>&lt;</code>b | `lt` | Less | num , num | int | prefix |  |
| a<code>&gt;</code>b | `gt` | More | num , num | int | prefix |  |
| a<code>=</code>b | `eq` | Equal | num , num | int | prefix |  |
| a<code>~</code>b | `match` | Match | num , same-shape | int | scalar | operands of different shapes are an error (KlongPy answers 0) |
| a<code>@</code>b | `index` | At/Index | array , index | left | index | a negative index counts from the end; f@x (apply) is not in the profile |
| a<code>#</code>b | `take` | Take | const-int , rank1 | right | take | rank 1 only in version 1; -2#a is -(2#a): write (-2)#a |
| a<code>_</code>b | `drop` | Drop | const-int , array | right | drop | negative n drops from the end |
| a<code>,</code>b | `join` | Join | num , num | promote | join | joining ints with floats gives floats |
| a<code>:+</code>b | `rotate` | Rotate | int-scalar , rank1 | right | same | rank 1 only (KlongPy rotates the flattened data of a matrix); negative n rotates left |
| a<code>:^</code>b | `reshape` | Reshape | const-shape , num | right | reshape | the row-major data cycles; a scalar n takes a scalar or a rank-1 array only (write [n]:^a for a matrix) |
| a<code>:=</code>b | `amend` | Amend | rank1 , amend-pair | promote | amend | g::g:=v,i on a global is an in-place store |
| a<code>::</code>b | `define` | Define | - | right | same | name::expr assigns a local or a global and is the value assigned; see the forms table |
| <code>-</code>a | `neg` | Negate | num | same | same | a leading - negates the whole expression to its right |
| <code>_</code>a | `floor` | Floor | num | int | same | rounds toward minus infinity |
| <code>%</code>a | `recip` | Reciprocal | num | float | same |  |
| <code>~</code>a | `not` | Not | num | int | same |  |
| <code>#</code>a | `size` | Size | array | int | scalar | the first dimension, a compile-time constant |
| <code>^</code>a | `shape` | Shape | array | int | rank-vector | a compile-time constant |
| <code>*</code>a | `first` | First | num | same | item |  |
| <code>&#124;</code>a | `reverse` | Reverse | num | same | same | reverses along the first dimension |
| <code>+</code>a | `transpose` | Transpose | rank2 | same | transpose |  |
| <code>!</code>a | `enumerate` | Enumerate | const-int | int | enumerate | n must be a positive constant |
| <code>,</code>a | `list` | List | num | same | wrap |  |

### Adverbs

The verb `f` is an operator, the name of a function or extern, or a function literal.

| Written | Row | Name | Operands | Type rule | Shape rule | Note |
|---|---|---|---|---|---|---|
| f<code>'</code>a | `each` | Each | array | verb | each | f is called on each item along the first dimension |
| a f<code>'</code>b | `each2` | Each-2 | array , array | verb | each | both operands must have the same length (KlongPy stops at the shorter) |
| f<code>/</code>a | `over` | Over | array | acc | acc | the accumulator starts from the first item |
| a f<code>/</code>b | `over-seed` | Over-Neutral | num , array | acc | acc | the accumulator starts from the seed |
| f<code>\</code>a | `scan` | Scan-Over | array | acc | scan |  |
| a f<code>\</code>b | `scan-seed` | Scan-Over-Neutral | num , array | acc | scan-seed | returns n+1 items |
| f<code>:'</code>a | `each-pair` | Each-Pair | array | verb | each-pair | needs at least two items |
| a f<code>:*</code>b | `iterate` | Iterate | int-scalar , num | state | state | n f:*a applies f n times |
| a f<code>:~</code>b | `while` | While | predicate , num | state | state | p f:~a applies f while p holds |

### Other forms

| Form | Row | Rule |
|---|---|---|
| `1`, `2.5`, `1e3` | `number` | an integer literal is int; a literal with `.` or `e` is float |
| `[1 2 3]`, `[[1 2] [3 4]]` | `list` | a rectangular list of numbers; any float makes it float |
| `[;a;b;c]` | `evaluated-list` | elements of the same shape; ints are promoted when any element is float |
| `:[c;a;b]`, `:[c1;a:&#124;c2;b;else]` | `conditional` | the condition is a scalar; the branches have the same shape; int with float gives float |
| `{a;b;c}` | `sequence` | statements run in order; the last is the result |
| `{[a b];a::...}` | `local` | a local's type is the join of its assignments (int with float is float); its shape is fixed |
| `x y z`, or `f(a;b;c;d)::{...}` | `parameters` | named parameters allow more than three |
| `[a b]::v` | `destructure` | v is an array with exactly one item per name; as the first statement the names are locals |
| `name::expr` to a top-level name | `global` | the type and shape of a global never change |
| `g(a;b)`, `a g b` | `call` | a compiled function, with every argument supplied; specialised per argument types |
| `read.adc(0)` | `extern-call` | a name declared with .extern; a str parameter takes a string literal |
| `f'a` with f an operator, a function name or `{...}` | `verb` | a function literal is allowed only as the verb of an adverb or the predicate of While; it may read, but not assign, the locals of the function it is written in |

### Operators and adverbs outside the profile

| Written | Name | Error | Why |
|---|---|---|---|
| <code>&amp;</code>a | Expand/Where | `where` | result length depends on the data |
| a<code>?</code>b | Find | `find` | result length depends on the data |
| <code>?</code>a | Range | `range` | result length depends on the data |
| <code>=</code>a | Group | `group` | result is ragged |
| <code>&lt;</code>a | Grade-Up | `grade` | fixed length; left out of version 1 (needs a sort routine) |
| <code>&gt;</code>a | Grade-Down | `grade` | fixed length; left out of version 1 (needs a sort routine) |
| a<code>:#</code>b | Split | `split` | result is ragged |
| a<code>:_</code>b | Cut | `cut` | result is ragged |
| f<code>:~</code>a | Converge | `converge` | number of steps unknown |
| f<code>\~</code>a | Scan-Converging | `scan-dynamic` | result length unknown |
| a f<code>\~</code>b | Scan-While | `scan-dynamic` | result length unknown |
| a f<code>\*</code>b | Scan-Iterating | `scan-dynamic` | fixed length for a constant count, but not in the design's list; left out of version 1 |
| <code>$</code>a | Format | `string` | makes a string |
| a<code>$</code>b | Format2 | `string` | makes a string |
| a<code>:$</code>b | Form | `string` | reads a string |
| <code>:#</code>a | Char | `string` | makes a character |
| <code>@</code>a | Atom | `unsupported-op` | always a constant in a typed program |
| <code>:_</code>a | Undefined | `unsupported-op` | :undefined is not a run-time value |
| a<code>:-</code>b | Amend-in-Depth | `unsupported-op` | not in the design's list |
| a<code>:@</code>b | Index-in-Depth | `unsupported-op` | not in the design's list; write (a@i)@j |
| a f<code>:\</code>b | Each-Left | `unsupported-op` | not in the design's list; write {a f x}'b |
| a f<code>:/</code>b | Each-Right | `unsupported-op` | not in the design's list; write {x f b}'a |
| f<code>@'</code>a | Each-Index | `unsupported-op` | not in the design's list |
| <code>˙</code>a | Track | `unsupported-op` | autograd |
| <code>∇</code>a | Grad | `unsupported-op` | autograd |
| a<code>∇</code>b | Grad | `unsupported-op` | autograd |
| a<code>∂</code>b | Jacobian | `unsupported-op` | autograd |
| a<code>:&gt;</code>b | Autograd | `unsupported-op` | autograd |

### Operand kinds

| Kind | Meaning |
|---|---|
| `num` | an int or a float, scalar or array |
| `array` | an array (rank 1 to 4) |
| `rank1` | a rank-1 array |
| `rank2` | a rank-2 array |
| `int` | an int, scalar or array |
| `int-scalar` | an int scalar |
| `const-int` | an int scalar that is a compile-time constant |
| `const-shape` | a compile-time constant: a positive int, or a list literal of 1 to 4 positive ints |
| `index` | an int scalar or an int array; constants are range-checked |
| `same-shape` | a value of the same shape as the other operand |
| `amend-pair` | written v,i: a scalar value v and a scalar index or a rank-1 index list of known length |
| `predicate` | a monadic function (a name or a literal) that returns a scalar |

### Type rules

| Rule | Element type of the result |
|---|---|
| `promote` | float if either operand is float, else int |
| `float` | always float |
| `int` | always int |
| `int-only` | both operands must be int; the result is int |
| `power` | int when both operands are int and the exponent is a non-negative constant, else float |
| `same` | the element type of the operand |
| `right` | the element type of the right operand |
| `left` | the element type of the left operand |
| `verb` | the element type the verb returns |
| `acc` | the accumulator: the fixed point of acc = f(acc, item), with at most one int to float promotion |
| `state` | the type of the starting value, which the verb must return unchanged |

### Shape rules

| Rule | Shape of the result |
|---|---|
| `prefix` | scalar with scalar is a scalar; scalar with array is the array's shape; of two arrays the shorter shape must be a prefix of the longer, which is the result |
| `same` | the shape of the operand |
| `scalar` | a scalar |
| `item` | the operand's shape without its first dimension (a scalar stays a scalar) |
| `index` | a scalar index drops the first dimension; an index array of shape S gives S followed by the rest |
| `take` | [&#124;n&#124;] (the items cycle when &#124;n&#124; is larger than the length; negative n takes from the end) |
| `drop` | the first dimension shrinks by &#124;n&#124;, which must leave at least one item |
| `join` | scalar,scalar is [2]; rank 1 with a scalar is [n+1]; two arrays of the same rank whose items have the same shape are [n+m, ...] |
| `reshape` | the constant shape; the items cycle |
| `transpose` | [m,n] for an operand of shape [n,m] |
| `enumerate` | [n] |
| `wrap` | [1] followed by the operand's shape |
| `rank-vector` | [rank of the operand] |
| `amend` | the shape of the array |
| `each` | [n] followed by the shape the verb returns |
| `each-pair` | [n-1] followed by the shape the verb returns |
| `acc` | the shape of the accumulator |
| `scan` | [n] followed by the shape of the accumulator |
| `scan-seed` | [n+1] followed by the shape of the accumulator (the seed is the first item) |
| `state` | the shape of the starting value |

### Errors

Every rejection is a `static error` that ends in one of these rule ids.

| Rule id | Group | Rejected |
|---|---|---|
| `strict-syntax` | Program structure and declarations | the program does not parse at strict level 2 |
| `toplevel-effect` | Program structure and declarations | a top-level expression that is not a definition or a declaration |
| `decl-literal` | Program structure and declarations | an argument of .extern, .entry or .l that is not a string literal |
| `signature` | Program structure and declarations | a malformed signature |
| `extern` | Program structure and declarations | an extern that cannot be declared (bad or reserved name, declared twice with different signatures) |
| `entry` | Program structure and declarations | an entry that cannot be declared (unknown function, wrong number of parameters, reserved name, result that does not match the function) |
| `load` | Program structure and declarations | a file named by .l that cannot be found or read |
| `redefinition` | Program structure and declarations | a function or extern name defined more than once, or assigned |
| `global-value` | Compile-time values | a global whose compile-time value is a string, char, symbol, dictionary, function or :undefined |
| `global-array` | Compile-time values | a global whose compile-time value is a ragged, mixed or empty array, or has more than 4 dimensions |
| `int-range` | Compile-time values | an integer constant outside the range of kg_int |
| `string` | Values a run-time function may not hold | a string or a character as a run-time value |
| `symbol` | Values a run-time function may not hold | a symbol as a run-time value |
| `dict` | Values a run-time function may not hold | a dictionary as a run-time value |
| `list-literal` | Values a run-time function may not hold | a list that is not a rectangular list of numbers |
| `fn-value` | Values a run-time function may not hold | a function used as a value |
| `projection` | Values a run-time function may not hold | a projection: a call with fewer arguments than the function takes |
| `void-value` | Values a run-time function may not hold | the result of a void extern used as a value |
| `where` | Results whose length depends on run-time data | &a (Expand/Where): the result length depends on the data |
| `find` | Results whose length depends on run-time data | a?b (Find): the result length depends on the data |
| `range` | Results whose length depends on run-time data | ?a (Range): the result length depends on the data |
| `group` | Results whose length depends on run-time data | =a (Group): the result length depends on the data |
| `grade` | Results whose length depends on run-time data | <a and >a (Grade): left out of version 1 (needs a sort routine in the runtime) |
| `split` | Results whose length depends on run-time data | a:#b (Split): the result is ragged |
| `cut` | Results whose length depends on run-time data | a:_b (Cut): the result is ragged |
| `const-count` | Results whose length depends on run-time data | take, drop, reshape or enumerate with a count that is not a compile-time constant |
| `converge` | Results whose length depends on run-time data | f:~a (Converge): the number of steps is not known |
| `scan-dynamic` | Results whose length depends on run-time data | the \~ and \* scan forms: the result length is not known |
| `take-rank` | Forms outside the profile | take from anything but a rank-1 array |
| `amend-form` | Forms outside the profile | an amend other than a:=v,i on a rank-1 array with one scalar value |
| `size-of-atom` | Forms outside the profile | # or ^ of an atom |
| `unsupported-op` | Forms outside the profile | an operator, adverb or adverb chain that is not in the operator table |
| `recursion` | Forms outside the profile | recursion, direct or mutual, including .f |
| `sys-fn` | Forms outside the profile | a system function or variable in a run-time function |
| `undefined-name` | Forms outside the profile | a name that is not a parameter, a local, a global, a function or an extern |
| `unset-local` | Forms outside the profile | a local read before it is assigned |
| `closure-assign` | Forms outside the profile | a nested function literal that assigns a local of the enclosing function |
| `arity` | Forms outside the profile | a call or an adverb with the wrong number of arguments for the function |
| `operand` | Types and shapes | an operand of the wrong type or rank for the operator |
| `shape-mismatch` | Types and shapes | operands whose shapes do not agree |
| `shape-change` | Types and shapes | a variable whose type or shape differs between two assignments |
| `cond-shape` | Types and shapes | a conditional whose branches have different shapes |
| `cond-scalar` | Types and shapes | a condition that is not a scalar |
| `div-zero` | Types and shapes | division of an integer constant by the constant zero |
| `index-range` | Types and shapes | a constant index out of range |
| `array-shape` | Types and shapes | an array with more than 4 dimensions or with an empty dimension |
| `loop-state` | Types and shapes | an accumulator or loop state whose type or shape changes |
| `extern-call` | Types and shapes | a call of an extern that does not match its signature |
<!-- END GENERATED -->

## Typing rules

Inference is forward and monomorphic. Every expression gets one type from the types of its parts.

- **Literals.** An integer literal is `int`, a literal with `.` or `e` is `float`. A list literal is an array of the common element type (any float makes it float).
- **Atomic dyads** (`+ - * :% ! | & < > =`). Scalar with scalar gives a scalar; scalar with array gives the array's shape; of two arrays the shorter shape must be a prefix of the longer, and the result has the longer shape. Otherwise it is a shape error at compile time. The element type is float if either operand is float; comparisons give int.
- **`%`** always float. **`:%`** needs int operands and gives int. **`!`** gives int for two ints and float otherwise.
- **`^`** gives int only when both operands are int and the exponent is a non-negative constant; otherwise float. (KlongPy returns an int whenever the result happens to be whole; the profile cannot depend on values.)
- **`_`** gives int. **`#a`** and **`^a`** are compile-time constants.
- **Index.** `a@i` with scalar `i` drops the first axis. With an int array `i` of shape `S`, the result shape is `S` followed by the rest of `a`'s shape.
- **Conditional.** The condition must be a scalar. Branch types are unified (int with float gives float); shapes must be equal.
- **Each.** `f'a` calls `f` on each item along the first axis; the result shape is `[n]` followed by `f`'s result shape.
- **Each-pair.** `f:'a` gives `[n-1]` followed by `f`'s result shape.
- **Constant indices** are checked at compile time; one out of range is an error.
- **Over and scan.** The accumulator's type is the fixed point of `acc = f(acc, item)`, starting from the first item (or the seed); if it does not stabilise after one promotion step (int to float), it is an error. Scan's result is `[n]` (or `[n+1]` with a seed, matching Klong) followed by the accumulator's shape. So `,/a` (flatten) is outside the profile: write `[n]:^a`.
- **Iterate and while.** The state's type and shape must be the same before and after the body.
- **Functions** are specialised per distinct tuple of argument types and shapes. An entry's argument types come from its `.entry` signature; everything else follows from call sites. A constant passed as an argument is not a constant inside the function.
- **Globals** take the type of their compile-time value. Assigning a different type or shape is an error. An int global assigned a float is an error, not a promotion (state must not change type), and so is a float global assigned an int.
- **Numeric promotion at assignment** to a local: the local's type is the join of all assignments to it (int with float gives float); shapes must agree.
- **Compile-time constants** are literals, `#a` and `^a`, globals that no compiled function assigns, and `+ - * :% !`, max and min of constants.

## Things to know when writing numeric Klong

- A leading `-` negates the whole expression to its right: `-2#a` is `-(2#a)`, `-7!2` is `-(7!2)`. Write `(-2)#a`.
- Negative counts mean "from the end": `(-2)#[1 2 3]` is `[2 3]`, `(-1)_[1 2 3]` is `[1 2]`, `(-1):+a` rotates left.
- Take cycles: `5#[1 2]` is `[1 2 1 2 1]`.
- Scan with a seed returns `n+1` items: `0+\[1 2 3]` is `[0 1 3 6]`.
- Joining ints with floats gives floats: `[1 2],[3 4.5]`.
- Lists need brackets (`[1 2 3]`, not `1 2 3`), and a name inside brackets is a symbol, not the variable: write `[;a;b]`.
- `a!b` and `a:%b` are C's `%` and `/`: `(-7):%2` is -3, `(-7)!2` is -1 and `7!-2` is 1. Float remainder is `fmod` (`7.5!2` is 1.5). `_` rounds toward minus infinity (`_(-3.5)` is -4).
- Evaluation order: the right operand of a dyad is evaluated before the left; call arguments and the elements of `[;a;b]` left to right. Compiled code keeps this order.
- Names contain letters, digits and `.` but no underscore. A Klong name maps to a C identifier by replacing `.` with `_`, so the extern `read.adc` is C `read_adc`.
- A function reads its parameters, its locals and the globals. It does not see the locals of its caller (KlongPy's scoping is dynamic; the profile's is not). A function literal used as a verb may read the locals of the function it is written in, but not assign them.

## Host functions and entry points

```
.extern(name; signature)             :" a function the target provides; module defaults to env "
.extern(name; signature; module)
.entry(name; signature)              :" a function the host calls "
```

The name, the signature and the module are string literals. A signature is
`params -> result`: `params` is a space-separated list of types (possibly
empty) and `result` is a type or `void`. Types: `int`, `float`, `int[N]`,
`float[N,M]`, and `str` (extern parameters only). Examples: `"int -> float"`,
`"float[64] -> float"`, `"int float -> void"`, `"-> int"` (called as `name()`).

- In C an extern is a prototype under the mapped name (`read.adc` is `read_adc`) and an entry is `prog_<name>`. In wasm an extern is an import from `module` and an entry is an export.
- An extern name must not map to a C keyword or start with `kg.`; an entry cannot be called `init`, `t`, `size` or `storage.t` (`prog_init` and the others are generated).
- `.entry` needs the function to be defined before it and to take as many arguments as the signature has parameters. A function may have several entries with different shapes.
- A `str` parameter takes a string literal.
- The result of a `void` extern cannot be used as a value.

### Running a static program under KlongPy

The same file runs in KlongPy with Python functions standing in for the
externs:

```python
from klongpy import KlongInterpreter

klong = KlongInterpreter(strict=2)
klong.externs["read.adc"] = lambda channel: 0.3
klong.externs["write.pwm"] = lambda channel, duty: 0
klong('.l("examples/embedded/pid.kg")')
klong("step(0.31)")                      # 0.12005
```

or, in one call each:

```python
from klongpy.static import load_program, call_entry

klong = load_program("examples/embedded/pid.kg",
                     externs={"read.adc": lambda channel: 0.3, "write.pwm": lambda channel, duty: 0})
call_entry(klong, "step", 0.31)          # 0.12005, checked against the .entry signature
```

- `klong.externs` maps an extern name to a Python callable. It is looked up at each call, so a stand-in can be bound before or after the `.extern` line runs and replaced at any time. `klong.externs.declared` and `klong.externs.entries` hold what the program declared.
- A stand-in gets one positional argument per parameter: a Python `int` for `int`, a `float` for `float` (an int argument is converted), a `str` for `str`, a numpy `int64` or `float64` array of the declared shape for an array. It returns a value of the result type; for `void` the result is ignored and the call evaluates to `:undefined`.
- Arguments and result are checked against the signature on every call: a float where an int is declared is a type error, a wrong array shape a shape error, a wrong number of arguments an arity error, each naming the extern and positioned at the call. Calling an extern that has no stand-in is a system error naming it. An exception raised by a stand-in is a python error.
- `.entry` under KlongPy validates the declaration and records it; the function is called as usual. `call_entry` calls it the way compiled code is called, with arguments and result checked against the signature.

## Differences from KlongPy

By design, and exercised by tests.

| Case | KlongPy on the host | Compiled |
|---|---|---|
| Integer overflow | Python ints do not overflow; numpy int64 wraps | wraps at `kg_int` width (two's complement) |
| Float precision | float64 | `kg_float` width; exact with `--float64` |
| `x%0` | `:undefined`, per element for arrays | `KG_DIV0=trap` calls `kg_trap`; `KG_DIV0=ieee` (default) gives inf or nan |
| Integer `:%` or `!` by zero | `:undefined` | `kg_trap` always; a constant zero divisor is a compile error |
| `a@i` out of range | error; negative `i` counts from the end | with `KG_BOUNDS=1` (default) `kg_trap`; negative indices count from the end as in KlongPy; with `KG_BOUNDS=0` unchecked. A constant index out of range is a compile error |
| `^` result type | int when the result is whole (`2.0^2` is the int 4) | int only for int operands and a non-negative constant exponent, else float |
| `:%` of floats | accepted (`7.0:%2` is 3) | compile error: floor the operands first |
| `_` of a huge float | saturates to the int64 range | saturates to the `kg_int` range |
| `a~b` with different shapes | 0 | compile error |
| `a f'b` with different lengths | stops at the shorter | compile error |
| `n:+a` on a matrix | rotates the flattened data | compile error (rank 1 only) |
| Over on one item (`f/[5]`) | the item itself, whatever `f` returns | the item converted to the accumulator's type |
| A conditional with an int and a float branch | the type of the branch taken | float (the value is the same) |
| A callee reading its caller's locals | works (dynamic scope) | compile error (`undefined-name`) |

## What is checked without running the program

`kgpy --check-static` never evaluates anything. The type of a global is
known to it when the definition is a literal (`pos::0`,
`buf::[0.0 0.0 0.0 0.0]`); the value of any other compile-time definition
(`taps::raw%+/raw`) exists only once the host has evaluated it. Such a global
is treated as numeric of unknown type and shape, the rules that need a type
or shape are applied only where it is known, and the summary says how many
globals are in that state. Rules about forms (a rejected operator, a string,
a function used as a value, recursion, a system function, a count that is
not constant) never need types and are always decided. The compiler
evaluates the definitions and then applies the same rules with everything
known.

## Reference programs

`examples/embedded/` holds three programs that are inside the profile, each
with the results KlongPy gives (`.expect`):

- `mavg.kg`: a moving average over the last four samples; the window and its write position are state.
- `pid.kg`: one step of a PID controller that reads a sensor and writes an output through two externs.
- `fir.kg`: a 16-tap FIR filter over a block of 256 samples; the taps and the index table are built on the host.
