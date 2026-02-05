# Operator Reference

## Monads (Single Argument)

| Operator | Name | Description |
|----------|------|-------------|
| `@a` | Atom | Check if value is an atom |
| `:#a` | Char | Convert to character |
| `!a` | Enumerate | Generate sequence 0..a-1 |
| `&a` | Expand | Expand boolean mask |
| `*a` | First | First element |
| `_a` | Floor | Floor/truncate |
| `$a` | Format | Convert to string |
| `>a` | Grade-Down | Indices that would sort descending |
| `<a` | Grade-Up | Indices that would sort ascending |
| `=a` | Group | Group equal elements |
| `,a` | List | Wrap in list |
| `-a` | Negate | Negate value |
| `~a` | Not | Logical not |
| `%a` | Reciprocal | 1/a |
| `?a` | Range | Random or range |
| `|a` | Reverse | Reverse order |
| `^a` | Shape | Array dimensions |
| `#a` | Size | Length/count |
| `+a` | Transpose | Transpose matrix |
| `:_a` | Undefined | Check if undefined |
| `∇a` | Grad | Numeric gradient function of a |

## Dyads (Two Arguments)

| Operator | Name | Description |
|----------|------|-------------|
| `a:=b` | Amend | Amend array at index |
| `a:-b` | Amend-in-Depth | Deep amend |
| `a:_b` | Cut | Cut array at indices |
| `a::b` | Define | Define variable |
| `a%b` | Divide | Division |
| `a_b` | Drop | Drop elements |
| `a=b` | Equal | Equality test |
| `a?b` | Find | Find element |
| `a:$b` | Form | Format with precision |
| `a$b` | Format2 | Format with width |
| `a@b` | Index/Apply | Index or apply function |
| `a:@b` | Index-in-Depth | Deep indexing |
| `a:%b` | Integer-Divide | Integer division |
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
| `a!b` | Remainder | Modulo |
| `a:+b` | Rotate | Rotate elements |
| `a:#b` | Split | Split at indices |
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
| `f:*a` | Iterate | Iterate f until stable |
| `n f:*a` | Iterate-n | Iterate f n times |
| `f:~a` | Converge | Converge with condition |

### Each-Index Adverb: `@'`

The `@'` adverb applies a function to `[index;element]` pairs, similar to Python's `enumerate()`:

```klong
:" Get indices"
{x@0}@'[10 20 30]           :" [0 1 2]"

:" Get values (same as regular each)"
{x@1}@'[10 20 30]           :" [10 20 30]"

:" Use both index and value"
{((x@0)*10)+(x@1)}@'[5 6 7] :" [5 16 27]"

:" Enumerate-like behavior"
{x}@'["a" "b" "c"]          :" [[0 'a'] [1 'b'] [2 'c']]"

:" Filter by index (keep even indices)"
{:[0=(x@0)!2;x@1;0]}@'[10 20 30 40]  :" [10 0 30 0]"
```

## Evaluated Array Constructors

The `[;expr1;expr2;...]` syntax creates arrays where each element is evaluated as a full expression. This is useful when you need to build arrays from computed values rather than literals.

### Basic Syntax

```klong
:" Standard array (literal values only)"
[1 2 3]

:" Evaluated array (expressions are computed)"
[;1+1;2+2;3+3]              :" [2 4 6]"
```

### Use Cases

**Building arrays from variables:**
```klong
a::10; b::20
[;a;b;a+b]                  :" [10 20 30]"
```

**Arrays from function results:**
```klong
avg::{(+/x)%#x}
arr::[1 2 3 4 5]
[;avg(arr);+/arr;#arr]      :" [3.0 15 5]"
```

**Inside functions (deferred evaluation):**
```klong
makeTriple::{[;x;x*2;x*3]}
makeTriple(5)               :" [5 10 15]"
```

**Combining with other operations:**
```klong
[;|[1 2 3];+/[4 5 6]]       :" [[3 2 1] 15]"
```

### Why Use Evaluated Arrays?

In standard Klong, array literals `[...]` only accept raw tokens, not expressions. This means `[avg(x) sum(x)]` would be parsed as separate tokens, not function calls.

The `[;...]` syntax solves this by evaluating each semicolon-separated element:

| Standard Array | Evaluated Array |
|---------------|-----------------|
| `[1 2 3]` | `[;1;2;3]` |
| Literal values only | Expressions allowed |
| Parsed at read time | Evaluated at runtime |
| Fast for constants | Flexible for computed values |

## Unused Operators

The following operator combinations are reserved for future use:

```
:! :& :, :< :?
```
