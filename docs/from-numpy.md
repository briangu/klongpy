# Klong for NumPy users

KlongPy runs on NumPy arrays, so most of what you know about array shapes and dtypes carries over. What changes is the notation. This page maps NumPy expressions to Klong, then lists the places where the same idea behaves differently. For the language from the beginning see the [tutorial](tutorial.md); for the full operator list see [Operators](operators.md).

## The mental model

There are no loops and no operator precedence. An expression is read from right to left, and each operator takes everything to its right as its right operand. A leading `-` is an operator too, which is the source of the most common mistake (see [Gotchas](#gotchas)).

Adverbs take the place of loops: `'` applies a function to each element, `/` reduces, `\` scans. A function is `{...}` and its arguments are `x`, `y` and `z`. The examples below are checked at strict level 2 (see [Strict mode](strict.md)).

<!-- check-docs: strict 2 -->
```kgpy
?> 2*3+4
14
?> (2*3)+4
10
?> a::[3 1 4 1 5]
[3 1 4 1 5]
?> +/a
14
?> avg::{(+/x)%#x}
:monad
?> avg(a)
2.8
?> {x*x}'a
[9 1 16 1 25]
?> a*a
[9 1 16 1 25]
```

`{x*x}'a` is the explicit loop, `a*a` the array form. Arithmetic on lists already works element by element, as in NumPy, so you rarely need `'` for it.

## Translation table

Each row is checked in the transcript that follows the table, in the same order. `a` and `b` are integer lists, `m` is the 2 by 3 matrix `[[1 2 3] [4 5 6]]`.

| NumPy | KlongPy | Note |
|---|---|---|
| `np.arange(n)` | `!n` | |
| `len(a)` | `#a` | |
| `a.sum()` | `+/a` | |
| `np.cumsum(a)` | `+\a` | |
| `np.prod(a)` | `*/a` | |
| `a.max()`, `a.min()` | `\|/a`, `&/a` | |
| `a.mean()` | `(+/a)%#a` | `%` is float division |
| `a[i]` | `a@i` | negative `i` counts from the end |
| `a[idx]` | `a@idx` | `idx` is a list of indices |
| `a[:n]` | `n#a` | |
| `a[n:]` | `n_a` | |
| `a[-n:]` | `(0-n)#a` | `-n#a` is not the same, see below |
| `a[::-1]` | `\|a` | |
| `np.where(a>25)[0]` | `&a>25` | |
| `a[a>25]` | `a@&a>25` | |
| `a.reshape(2,3)` | `[2 3]:^a` | |
| `m.T` | `+m` | |
| `m.sum(axis=0)` | `+/m` | reduces the leading axis |
| `m.sum(axis=1)` | `+/'m` | `'` applies the sum to each row |
| `np.concatenate((a,b))` | `a,b` | |
| `np.unique(a)` | `?a` | first-appearance order, not sorted |
| `np.sort(np.unique(a))` | `u::?a;u@<u` | |
| `np.argsort(a)` | `<a` | indices, as in NumPy |
| `np.sort(a)` | `a@<a` | |
| `np.argmax(a)` | `*>a` | first index of the grade-down |
| `np.dot(a,b)` | `+/a*b` | |
| `np.maximum(a,b)`, `np.minimum(a,b)` | `a\|b`, `a&b` | |
| `a>1` | `a>1` | a list of 0 and 1 |
| `(a>1).sum()` | `+/a>1` | |
| `np.array_equal(a,b)` | `a~b` | `a=b` is element by element |
| `np.full(3,7)` | `3#7` | |
| `np.zeros(3)` | `3#0` | integers, not floats |
| `a[1]=9` | `a:=9,1` | returns a new list, see [Gotchas](#gotchas) |

<!-- check-docs: strict 2 -->
```kgpy
?> !5
[0 1 2 3 4]
?> a::[10 20 30 40 50]
[10 20 30 40 50]
?> #a
5
?> +/a
150
?> +\a
[10 30 60 100 150]
?> */[1 2 3 4]
24
?> |/[3 1 2]
3
?> &/[3 1 2]
1
?> (+/a)%#a
30.0
?> a@2
30
?> a@-1
50
?> a@[0 2]
[10 30]
?> 2#a
[10 20]
?> 2_a
[30 40 50]
?> (0-2)#a
[40 50]
?> |a
[50 40 30 20 10]
?> &a>25
[2 3 4]
?> a@&a>25
[30 40 50]
?> [2 3]:^!6
[[0 1 2] [3 4 5]]
?> m::[[1 2 3] [4 5 6]]
[[1 2 3] [4 5 6]]
?> +m
[[1 4] [2 5] [3 6]]
?> +/m
[5 7 9]
?> +/'m
[6 15]
?> [1 2],[3]
[1 2 3]
?> b::[3 1 3 2 1]
[3 1 3 2 1]
?> ?b
[3 1 2]
?> u::?b;u@<u
[1 2 3]
?> <[30 10 20]
[1 2 0]
?> s::[30 10 20];s@<s
[10 20 30]
?> *>[3 9 2]
1
?> +/[1 2 3]*[4 5 6]
32
?> [1 5 3]|[4 2 6]
[4 5 6]
?> [1 5 3]&[4 2 6]
[1 2 3]
?> [1 2 3]>1
[0 1 1]
?> +/[1 2 3]>1
2
?> [1 2 3]~[1 2 3]
1
?> [1 2 3]=[1 5 3]
[1 0 1]
?> 3#7
[7 7 7]
?> 3#0
[0 0 0]
?> a:=9,1
[10 9 30 40 50]
```

The NumPy side of the less obvious rows, for comparison:

```python
np.unique([3, 1, 3, 2, 1])               # [1 2 3]   sorted; Klong's ?b gives [3 1 2]
np.array([1, 2, 3, 4, 5])[-2:]           # [4 5]     Klong: (0-2)#a
np.argsort([30, 10, 20])                 # [1 2 0]   same as <
np.argmax([3, 9, 2])                     # 1         same as *>
np.where(np.array([10, 20, 30, 40, 50]) > 25)[0]   # [2 3 4]
```

Some NumPy functions have no one-symbol form, for example `np.std` or `np.linspace`. Call them through Python, see [Calling NumPy from Klong](#calling-numpy-from-klong).

## Broadcasting

NumPy lines shapes up from the trailing axis. KlongPy's atomic operators (`+ - * % ^ & | = < >`) line them up from the leading axis: two lists must have the same length at the outermost level, and a shorter operand is extended across the remaining dimensions. The rule is the one in [Operators](operators.md#atomic-dyads-and-nested-lists).

For a vector and a matrix the results differ. Adding a list of 2 to a 2 by 3 matrix works in Klong and fails in NumPy. Adding a list of 3 works in NumPy and fails in Klong:

```python
m = np.array([[1, 2, 3], [4, 5, 6]])
m + np.array([10, 20])        # ValueError: operands could not be broadcast
m + np.array([10, 20, 30])    # [[11 22 33] [14 25 36]]
```

```kgpy
?> m::[[1 2 3] [4 5 6]]
[[1 2 3] [4 5 6]]
?> m+[10 20]
[[11 12 13] [24 25 26]]
?> m+[10 20 30]
error: cannot apply + (Plus) to 2x3 matrix and list of 3
```

To add along the trailing axis, transpose, add, and transpose back, or repeat the row:

```kgpy
?> m::[[1 2 3] [4 5 6]]
[[1 2 3] [4 5 6]]
?> +(+m)+[10 20 30]
[[11 22 33] [14 25 36]]
?> m+2#,[10 20 30]
[[11 22 33] [14 25 36]]
```

Two plain lists of different lengths are a shape error, where NumPy raises `ValueError`:

```kgpy
?> [1 2 3]+[10 20]
error: cannot apply + (Plus) to list of 3 and list of 2
```

The errors are `KlongShapeError` and the other classes in [Errors](errors.md). Python functions that you call from Klong keep NumPy's own broadcasting rules.

## Calling NumPy from Klong

`.pyf` imports named functions from a Python module, `.py` imports all of them. Klong lists go in as NumPy arrays and the results come back as Klong lists:

```kgpy
?> .pyf("numpy";["mean" "std" "sort"])
1
?> mean([1 2 3 4])
2.5
?> std([1 2 3 4])
1.118033988749895
?> sort([3 1 2])
[1 2 3]
```

A Python function is called with at most three arguments, and only its required parameters are passed by position. Parameters that have a default can be set only through `.pyc`, which takes the function name, a list of positional arguments and a dictionary of keywords:

```kgpy
?> .pyf("numpy";"linspace")
1
?> .pyc("linspace";[0 1];:{["num" 5]})
[0.0 0.25 0.5 0.75 1.0]
```

From Python you can set an array and read back the result. A Klong integer list is an `int64` array, and an operation on a NumPy array in the interpreter returns a NumPy array:

```python
import numpy as np
from klongpy import KlongInterpreter

klong = KlongInterpreter()
klong['a'] = np.arange(5.0)
klong('a*2')                  # array([0., 2., 4., 6., 8.])
klong('[1 2 3]').dtype        # int64
```

See [Python integration](python_integration.md) for more.

## Gradients beside PyTorch

`f:>x` is the gradient of `f` at `x`, where PyTorch needs a tensor with `requires_grad`, a forward pass and a call to `.backward()`:

```python
x = torch.tensor(3.0, requires_grad=True)
(x ** 2).backward()
x.grad                        # tensor(6.)
```

On the torch backend `f:>x` uses autograd. On the NumPy backend it falls back to finite differences, so the results below are close to the exact ones, not equal to them. `.grad(f;x)` is the ASCII spelling of `x∇f`, which is always numeric; note that the function comes first, as in `.jacobian(f;x)`, and the point first in `x∇f`. The comparison table is in [Operators](operators.md#comparison).

```kgpy
?> f::{x^2}
:monad
?> f:>3
6.000000000838668
?> .grad(f;3)
6.000000000838668
?> 3∇f
6.000000000838668
?> .grad({+/x^2};[1 2 3])
[2.000000000279556 3.9999999996709334 6.000000000838668]
```

As with `.backward()`, the function must return a scalar. A vector-valued function needs a reduction (`+/`) or `.jacobian`:

<!-- check-docs: continue -->
```kgpy
?> {x^2}:>[1.0 2.0]
error: Loss function must return a scalar, got shape (2,). Use sum (+/) or mean (%#) to reduce to a scalar.
```

Gradient descent by iterating the update three times from 5.0, where the exact answer for a step of 0.1 on `x^2` is 2.56:

<!-- check-docs: continue -->
```kgpy
?> 3{x-0.1*f:>x}:*5.0
2.559999999818814
```

For the torch backend, see [PyTorch Backend & Autograd](torch_backend.md).

## Gotchas

**A leading minus negates everything to its right.** `-1+2` is `-(1+2)`. A slice from the end is `(0-2)#a`, not `-2#a`, and `-2_a` is not a drop:

```kgpy
?> -1+2
-3
?> (-1)+2
1
?> a::[10 20 30 40 50]
[10 20 30 40 50]
?> -2#a
[-10 -20]
?> (0-2)#a
[40 50]
```

A negative number inside a list literal is a number: `[-1 2]` is `[-1 2]`.

**Names do not contain underscores.** `my_var` is `my _ var`, with `_` the Drop operator. Use `myVar` or `my.var`. `kgpy --lint` reports this, see [Lint](lint.md).

**`%` is float division, `:%` is integer division, `!` is remainder.** The integer forms truncate toward zero, as C does (`np.fmod`). NumPy's `//` rounds toward minus infinity and its `%` takes the sign of the divisor:

```python
-5 // 2        # -3, where (-5):%2 is -2
-5 % 3         #  1, where (-5)!3 is -2
```

```kgpy
?> 7%2
3.5
?> 7:%2
3
?> (-5):%2
-2
?> (-5)!3
-2
?> 1%0
:undefined
```

Division by zero gives `:undefined`, not `inf` or an error.

**`^` gives an integer when the result is a whole number.** `[1 4 9]^0.5` is `[1 2 3]` and `[2 3]^0.5` is a list of floats. NumPy's `**` with a float exponent always returns floats.

**Amend does not change its argument.** `a:=9,1` is the list with `9` at index 1 and leaves `a` as it was. There is no `a[1]=9`:

```kgpy
?> r::[1 2 3]
[1 2 3]
?> r:=0,0
[0 2 3]
?> r
[1 2 3]
```

**Integers are exact beyond 64 bits.** An operand outside the int64 range makes `+ - * % < > | &` use Python integers, where NumPy raises or wraps. Two int64 operands still wrap as in NumPy. `^` is computed in floating point, so `10^400` is an error.

```kgpy
?> 9223372036854775807+1
-9223372036854775808
?> 1+9223372036854775808
9223372036854775809
?> 2^64
18446744073709551616
?> [1 2]+9223372036854775808
[9223372036854775809 9223372036854775810]
```

**A list of names is a list of symbols.** `[a b]` holds the symbols `:a` and `:b`, not their values. `[;a;b]` evaluates its entries, like `np.array([a, b])`:

```kgpy
?> a::1;b::2
2
?> [a b]
[:a :b]
?> [;a;b]
[1 2]
```

**`?a` is not `np.unique`.** It keeps first-appearance order. `=a` groups the indices of equal values in the same order: `=[3 1 3 2 1]` is `[[0 2] [1 4] [3]]`.

**Indexing is zero-based and checked.** `a@5` on a list of 5 is a range error, where `a[5]` on an array is an `IndexError`. A list of indices is checked the same way.

**Square brackets build lists; they do not index.** `a[1]` is the variable `a` followed by the list `[1]`. Write `a@1`.

## Features NumPy code tends to want

These forms exist in KlongPy beside the classic ones. See [Friendlier syntax](language-additions.md) for each with its classic equivalent.

Destructuring takes the place of tuple unpacking, and fails when the count is wrong:

```kgpy
?> [lo hi]::[3 9]
[3 9]
?> hi-lo
6
?> [p q r]::[1 2]
error: cannot destructure 2 values into 3 names
```

Named parameters allow more than three arguments and read like a Python signature:

```kgpy
?> vol(w;h;d;t)::{w*h*d*t}
:triad
?> vol(1;2;3;4)
24
?> vol(1;2;3;4;5)
error: vol takes 4 arguments, got 5
```

`.fmt` takes the place of f-strings. `{}` and `{0}` select arguments, and the format after the colon is the Klong width. A positive width pads on the right and a negative one on the left, and `8.2` means 8 integer digits and 2 decimals, not a total width of 8:

```kgpy
?> .fmt("{}: {}";["pi" 3.14159])
"pi: 3.14159"
?> .fmt("{:5}|";,42)
"42   |"
?> .fmt("{:-5}|";,42)
"   42|"
?> .fmt("{:5.2}";3.0)
"    3.00"
```

`:{;key;value;...}` builds a dictionary from expressions, and `d@k` looks a key up, giving `:undefined` when it is missing:

```kgpy
?> d:::{;"n";2+1;"sq";3*3}
:{["n" 3] ["sq" 9]}
?> d@"sq"
9
?> d@["n" "sq"]
[3 9]
?> d@"zz"
:undefined
```
