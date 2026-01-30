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

The `∇` operator computes gradients using numeric differentiation (finite differences). Works with any backend.

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

The `:>` operator uses PyTorch's automatic differentiation when the torch backend is enabled (`USE_TORCH=1`). Falls back to numeric differentiation otherwise.

**Syntax:** `function:>point`

```klong
f::{x^2}
f:>3              :" Returns 6.0 (derivative of x^2 at x=3)

g::{+/x^2}
g:>[1 2 3]        :" Returns [2 4 6] (gradient of sum of squares)
```

### Comparison

| Feature | `∇` (nabla) | `:>` (autograd) |
|---------|-------------|-----------------|
| Method | Numeric (finite differences) | PyTorch autograd |
| Precision | Approximate (~1e-6 error) | Exact |
| Backend | Any | Torch preferred |
| Speed | Slower for complex functions | Faster |
| Syntax | `point∇function` | `function:>point` |

## Adverbs

| Adverb | Name | Description |
|--------|------|-------------|
| `f'a` | Each | Apply f to each element |
| `f/a` | Over | Reduce with f |
| `f\a` | Scan | Scan with f |
| `n f'a` | Each-n | Apply f to groups of n |
| `a f'b` | Each-pair | Apply f to pairs |
| `f:*a` | Iterate | Iterate f until stable |
| `n f:*a` | Iterate-n | Iterate f n times |
| `f:~a` | Converge | Converge with condition |

## Unused Operators

The following operator combinations are reserved for future use:

```
:! :& :, :< :?
```
