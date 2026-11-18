# Language Additions

KlongPy adds a few forms to Klong. Each one gives a meaning to input that fails or does nothing useful in classic Klong, so a program written for classic Klong runs as it did, with the exceptions listed under [Syntax that changed meaning](release-notes.md#syntax-that-changed-meaning). Next to every new form this page shows the classic Klong that does the same thing, so you can use either.

| Form | Does | Classic Klong |
|---|---|---|
| `[a b c]::v` | [Destructuring](#destructuring): binds the members of a list to names | `a::v@0;b::v@1;c::v@2` |
| `{[a b]::x;...}` | The same as the first statement of a function; the names are local | `{[a b];a::x@0;b::x@1;...}` |
| `f(a;b;c;d)::{...}` | [Named parameters](#named-parameters), and more than three | `f::{...}` with `x y z`, or one list argument |
| `:{;k1;v1;k2;v2}` | An [evaluated dictionary](#dictionaries) | `:{[k1 v1] [k2 v2]}`, or `d,k,,v` |
| `d@k`, `d@[k1 k2]`, `d,d2` | Dictionary lookup with `@`, several keys at once, merge | `d?k` |
| `.fmt(template;args)`, `.pf(...)` | [Formatted text](#formatted-text) | joining strings with `,` and `$` |
| `.split(s;sep)` | [Strings](#strings): the pieces of `s` between the occurrences of `sep` | a loop over `s?sep`, written for each program |
| `.join(list;sep)` | The strings of `list` with `sep` between them | `{x,sep,y}/list`, which fails on an empty list |
| `.words(s)` | The pieces of `s` between runs of white space | a hand-written split that drops the empty pieces |
| `.trim(s)` | `s` without white space at either end | a loop that drops one blank at a time |
| `.num(s)` | The number that `s` spells, or `:undefined` | `.rs(s)`, which reads any data and stops after the first |
| `.grad(f;x)` | [Gradient](#gradient) in plain ASCII | `x∇f` |
| `it` | The [last result](#the-it-variable) at the REPL | assign it yourself |

## Destructuring

A list of names on the left of `::` binds each name to the member of the right side in the same position. The value of the statement is the right side.

<!-- check-docs: strict 2 -->
```kgpy
?> [a b c]::[1 2 3]
[1 2 3]
?> a
1
?> b+c
5
```

The classic form is one assignment for each name:

```kgpy
?> v::[1 2 3]
[1 2 3]
?> a::v@0;b::v@1;c::v@2
3
?> a,b,c
[1 2 3]
```

A list such as `[a b c]` written anywhere else holds the symbols `:a :b :c`, not their values; the left side of `::` is the one place where it names variables. Use `[;a;b;c]` to list values (see [lint W012](lint.md#w012-a-bare-name-in-a-data-list)).

The right side may be a list, an array or a string. A matrix gives its rows, and a string its characters:

<!-- check-docs: strict 2 -->
```kgpy
?> [r s]::[[1 2] [3 4]]
[[1 2] [3 4]]
?> s
[3 4]
?> [p q]::"hi"
"hi"
?> q
0ci
```

The number of names must equal the number of members, and the right side must be a list. Nothing is bound when it fails:

<!-- check-docs: strict 2 -->
```kgpy
?> [u v]::[1 2 3]
error: cannot destructure 3 values into 2 names
?> [u v]::5
error: cannot destructure integer into 2 names
?> u
error: undefined: u
```

### In a function

When a destructuring is the first statement of a function body, after an optional locals list, its names are local to the function. No locals list is needed, at any strict level, and a global of the same name is not touched:

<!-- check-docs: strict 2 -->
```kgpy
?> a::100
100
?> area::{[w h]::x;w*h}
:monad
?> area([4 5])
20
?> a
100
?> w
error: undefined: w
```

The classic form declares the locals and assigns each:

```kgpy
?> area::{[w h];w::x@0;h::x@1;w*h}
:monad
?> area([4 5])
20
```

This is the way to give an anonymous function more than three inputs: pass one list. Elsewhere in a body a destructuring follows the ordinary rule for assignment, so at [strict level 1 and 2](strict.md) the names must be declared as locals.

### What is not a destructuring

A list on the left of `::` is a destructuring only when it holds at least one name, and every member is a plain name, none is `x`, `y` or `z`, and no name is repeated. Any other list binds nothing. At strict level 0 it is accepted with a warning that says why. At level 1 and 2 it is a syntax error:

```kgpy
?> .strict(1)
1
?> [a [b c]]::[1 [2 3]]
error: cannot destructure: nested lists are not supported
?> [a a]::[1 2]
error: cannot destructure: a is named twice
?> [a x]::[1 2]
error: cannot destructure: x, y and z are the arguments of a function and cannot be assigned
?> []::[1 2]
error: cannot destructure: the list on the left of :: is empty
```

Parentheses make no difference: `([a b])::v` is a destructuring.

## Named parameters

`f(a;b)::{body}` defines `f` with the parameters `a` and `b`. It looks like the call it defines, and the right side must be a function in braces.

<!-- check-docs: strict 2 -->
```kgpy
?> f(a;b)::{a+b}
:dyad
?> f(1;2)
3
?> f::{x+y}
:dyad
?> f(1;2)
3
```

Any number of parameters is allowed. Classic functions stop at three (`x y z`):

<!-- check-docs: strict 2 -->
```kgpy
?> g(a;b;c;d)::{(a*b)+c*d}
:fn/4
?> g(1;2;3;4)
14
?> g@[1 2 3 4]
14
?> mx(a;b;c;d;e;f)::{a+b+c+d+e+f}
:fn/6
?> mx(1;2;3;4;5;6)
21
```

A function of four or more parameters displays as `:fn/N`, N being the number of parameters (`:fn/4`, `:fn/6`), which is how it is shown (over [IPC](ipc_capabilities.md) a function travels as its source text, named parameters included); `:nilad`, `:monad`, `:dyad` and `:triad` stay for zero to three. A function with no parameters is written `h()::{...}`:

<!-- check-docs: strict 2 -->
```kgpy
?> h()::{42}
:nilad
?> h()
42
```

With four inputs and classic Klong, pass a list and take it apart with a head [destructuring](#in-a-function): `g::{[a b c d]::x;(a*b)+c*d}` called as `g([1 2 3 4])`.

### Rules

The parameter names are local to the function. They shadow a global of the same name and leave it alone, and they need no locals list at strict level 1 or 2. The body may still start with a locals list, or with a head destructuring.

Parameters and locals are dynamically scoped: a function that is called from the body sees them, before the globals. The tutorial shows it in [Dynamic scoping](tutorial.md#dynamic-scoping), and why to pass arguments instead.

<!-- check-docs: strict 2 -->
```kgpy
?> a::100
100
?> k(a;b)::{[t];t::a*b;t+1}
:dyad
?> k(3;4)
13
?> a
100
```

`x`, `y` and `z` are not parameters of such a function. Using one directly is a syntax error found when the function is defined. A nested function literal has its own `x y z`, so it may use them:

<!-- check-docs: strict 2 -->
```kgpy
?> m(a;b)::{a+x}
error: x is not available in a function with named parameters
?> m(a;b)::{{x+1}'[;a;b]}
:dyad
?> m(1;2)
[2 3]
```

The right side must be a function in braces, at every strict level. The left side must be a call of a plain name whose arguments are plain names, none repeated and none of `x y z`. At strict level 0 a left side that fails that is accepted as it always was, with a warning, and defines nothing; at level 1 and 2 it is a syntax error:

```kgpy
?> f(a;b)::5
error: named-parameter definition needs a function body in braces
?> f(a;a)::{a}
:nilad
?> f
:f
?> .strict(1)
1
?> f(a;a)::{a}
error: cannot define f with named parameters: a is named twice
```

### Calling

A call works as for any function, and a function may be recursive by its own name or by `.f`:

<!-- check-docs: strict 2 -->
```kgpy
?> fib(n)::{:[n<2;n;fib(n-1)+fib(n-2)]}
:monad
?> fib(10)
55
?> fact(n)::{:[n<2;1;n*.f(n-1)]}
:monad
?> fact(5)
120
```

Leaving an argument empty, or leaving arguments off the end, makes a projection that takes the rest by position, for any number of parameters. `g(1;2)` is the same projection as `g(1;2;;)`. `f@list` spreads a list as the arguments. A function of one or two parameters works with the adverbs; one of more than two gives an arity error from the adverb.

<!-- check-docs: strict 2 -->
```kgpy
?> g(a;b;c;d)::{(a*b)+c*d}
:fn/4
?> p::g(1;;3;4)
:monad
?> p(10)
22
?> sub(a;b)::{a-b}
:dyad
?> sub/[10 3]
7
?> 3 sub 4
-1
?> g'[1 2]
error: g takes 4 arguments, got 1
```

More arguments than parameters is an arity error at every strict level, since no classic program can depend on it. Fewer written arguments is a projection, and `f@list` with too few members is an arity error:

<!-- check-docs: continue -->
```kgpy
?> g(1;2;3;4;5)
error: g takes 4 arguments, got 5
?> g@[1 2]
error: the function takes 4 arguments, got 2
```

### From Python

`klong['f']` is a Python callable. Arguments are positional or keyword, by parameter name. For a function written with `x`, `y` and `z` the keywords are `x`, `y` and `z`. Python callers do not get projections: too few, too many, an unknown keyword or a parameter given twice raises `KlongArityError`.

```python
from klongpy import KlongInterpreter

klong = KlongInterpreter()
klong('f(a;b;c;d)::{a+b*c-d}')
f = klong['f']
f(1, 2, 3, 4)          # -1
f(1, 2, d=4, c=3)      # -1
f(1, 2, 3)             # KlongArityError: f takes 4 arguments, got 3
f(1, 2, 3, 4, e=5)     # KlongArityError
```

### How many arguments a function takes

The arity of a function written with `x`, `y` and `z` is the highest of the three that its body uses, not counting nested function literals. It is what the display shows (`:monad`, `:dyad`), what Python callers pass, what an adverb dispatches on, and the limit at [strict level 2](strict.md#too-many-arguments).

<!-- check-docs: strict 2 -->
```kgpy
?> f::{1}
:nilad
?> f::{-x}
:monad
?> f::{#x}
:monad
?> f::{y}
:dyad
?> f::{x+z}
:triad
?> {-x}'[1 2]
[-1 -2]
?> {y}/[1 2 3]
3
```

A function that uses `x` only under a monad such as `-x` has one argument, so `f::{-x}` displays as `:monad`. `{y}` has two, so a call with one argument is a projection that waits for the second: `p::{y}(1);p(2)` is 2, and so is `{y}(1;2)`.

## Dictionaries

A dictionary literal is `:{[key value] [key value]}`. KlongPy keeps that form and adds an evaluated one, `:{;key;value;key;value}`: after `:{` a `;` with no space before it starts a list of expressions that alternate key and value. They are evaluated left to right each time the literal is evaluated.

<!-- check-docs: strict 2 -->
```kgpy
?> n::7
7
?> e:::{;"n";n;"sq";n*n}
:{["n" 7] ["sq" 49]}
?> :{["n" n]}
:{["n" :n]}
```

The plain literal does not evaluate its entries; `n` in it is the symbol `:n`. Classic Klong builds a dictionary from values by joining a pair `[key value]` onto a dictionary with `,`. The empty dictionary is `:{}`:

<!-- check-docs: strict 2 -->
```kgpy
?> n::7
7
?> e:::{}
:{}
?> e::e,"n",,n
:{["n" 7]}
?> e::e,"sq",,n*n
:{["n" 7] ["sq" 49]}
```

The literal `:{;}` is the empty dictionary too. An odd number of expressions, an empty slot and a trailing `;` are syntax errors. A list cannot be a key.

```kgpy
?> :{;"a"}
error: the last key of the dictionary literal has no value
?> :{;"a";1;}
error: missing expression in a dictionary literal
?> :{;[1 2];3}
error: a list of 2 cannot be a dictionary key
```

A dictionary literal inside a dictionary or list literal is a dictionary:

<!-- check-docs: strict 2 -->
```kgpy
?> d:::{["GET" :{["/" 2]}]}
:{["GET" :{["/" 2]}]}
?> d@"GET"
:{["/" 2]}
?> (d@"GET")@"/"
2
```

### Lookup

`d?k` looks up a key, and `d@k` does the same for every kind of key. `d@[k1 k2]` is the list of values. A missing key is `:undefined`. Test for it with `:_`.

<!-- check-docs: strict 2 -->
```kgpy
?> d:::{["a" 1] ["b" 2]}
:{["a" 1] ["b" 2]}
?> d?"a"
1
?> d@"a"
1
?> d@["a" "b"]
[1 2]
?> d@"zz"
:undefined
?> d@["a" "zz"]
[1 :undefined]
?> :_d@"zz"
1
```

### Merge, and the pair rule

`d,d2` copies the entries of `d2` into `d` and returns `d`, and an entry of `d2` replaces one with the same key. It changes `d` in place, so another name for the same dictionary sees the new entries:

<!-- check-docs: strict 2 -->
```kgpy
?> d:::{["x" 1]}
:{["x" 1]}
?> alias::d
:{["x" 1]}
?> d::d,:{["x" 5] ["y" 6]}
:{["x" 5] ["y" 6]}
?> alias
:{["x" 5] ["y" 6]}
```

Joining a list of two members to a dictionary adds an entry, whichever side the list is on: `d,[k v]`, `d,k,,v`, `[k v],d`. Anything else on the right of a dictionary is an error. The same rule means that `list,dict` does not append the dictionary to the list. To append it, wrap the dictionary with `,`:

<!-- check-docs: strict 2 -->
```kgpy
?> l::[1 2]
[1 2]
?> l,:{[3 4]}
:{[3 4] [1 2]}
?> l,,:{[3 4]}
[1 2 :{[3 4]}]
```

[Lint W020](lint.md#w020-a-two-element-list-joined-to-a-dictionary) flags a list of two members joined to a dictionary. A right operand that is neither a pair nor a dictionary is an error:

<!-- check-docs: strict 2 -->
```kgpy
?> :{["a" 1]},[1 2 3]
error: ,: a dictionary can only be joined with a tuple (a list of two members) or a dictionary
```

`$d` is the written form of the dictionary, as a string.

## Formatted text

`.fmt(template;args)` returns a string, and `.pf(template;args)` prints it with a newline and returns it. A call can leave out `args` when the template has no placeholders. The classic way is to join strings:

<!-- check-docs: strict 2 -->
```kgpy
?> x::3;y::4
4
?> .p("x=",($x),", y=",$y)
x=3, y=4
"x=3, y=4"
?> .pf("x={}, y={}";[;x;y])
x=3, y=4
"x=3, y=4"
?> .fmt("x={}, y={}";[;x;y])
"x=3, y=4"
```

`{}` takes the next argument, `{0}` and `{1}` take by position, and the two styles cannot be mixed. `{{` and `}}` are literal braces. A placeholder may carry a format after a colon, which is a number given to the [Format2](operators.md) operator `n$value`: a positive width pads on the right, a negative width on the left, and `5.2` means five integer digits and two decimals.

<!-- check-docs: strict 2 -->
```kgpy
?> .fmt("{1} {0}";["a" "b"])
"b a"
?> .fmt("{:8}|";42)
"42      |"
?> .fmt("{:-8}|";42)
"      42|"
?> .fmt("{:5.2}";3)
"    3.00"
?> .fmt("{{}} {}";7)
"{} 7"
```

Each argument is shown as `.d` shows it: strings and characters without quotes, lists as `[1 2 3]`, `:undefined` as `undefined`. If `args` is a list, its members are the arguments; any other value is the single argument. To pass one list as the only argument, wrap it with `,`. Build an argument list from variables with `[;a;b]`; `[a b]` would hold the symbols.

<!-- check-docs: strict 2 -->
```kgpy
?> .fmt("{}";[1 2 3])
error: .fmt: 1 placeholder(s) but 3 argument(s) in template '{}'
?> .fmt("{}";,[1 2 3])
"[1 2 3]"
?> .fmt("{} {}";[1])
error: .fmt: 2 placeholder(s) but 1 argument(s) in template '{} {}'
?> .fmt("{} {1}";[1 2])
error: .fmt: cannot mix {} and {N} placeholders in template '{} {1}'
```

A template with no placeholders can be given alone, as in `.pf("done")`, which prints `done`. A template that has placeholders needs its arguments, and the error counts them. A list of numbers is an array, so mixed integers and floats in one argument list all print as floats: in `[;3;2.5]` the `3` is already `3.0` when `.fmt` sees it. Write `$n` to keep an integer's form.

<!-- check-docs: strict 2 -->
```kgpy
?> .fmt("done")
"done"
?> .fmt("{}")
error: .fmt: 1 placeholder(s) ({}) but no arguments; call .fmt(template;arguments) in template '{}'
?> .fmt("{} {}";[;3;2.5])
"3.0 2.5"
?> .fmt("{} {}";[;$3;2.5])
"3 2.5"
```

## Strings

KlongPy has five functions for text that every program otherwise writes for itself. Each takes strings only; any other argument is a type error that names the function. Each counts characters, not bytes, as the rest of the language does. They are the same in the [native runtime](native/index.md), which runs the same [conformance cases](conformance.md).

`.split(s;sep)` returns the pieces of `s` between the occurrences of `sep`, a string of one or more characters or a character. Empty pieces are kept, so there is one more piece than there are separators, and the split of `""` is `[""]`. The result is always a list of strings. An empty `sep` is a range error. The classic way is a function that finds the separators with `?` and cuts between them; this one handles a single-character separator only:

<!-- check-docs: strict 2 -->
```kgpy
?> sp::{[p s e t];t::x;p::x?y;s::0,1+p;e::p,#x;{((e@x)-s@x)#(s@x)_t}'!#s}
:dyad
?> sp("a,,b";",")
["a" "" "b"]
?> .split("a,,b";",")
["a" "" "b"]
?> .split("";",")
[""]
?> .split("a::b::";"::")
["a" "b" ""]
?> .split("hé,é";0c,)
["hé" "é"]
?> .split("abc";"")
error: .split: y must not be empty
```

`.join(list;sep)` is the inverse: the strings of `list` with `sep` between neighbours. The empty list gives `""`. A member may be a character; a member of any other type is a type error, and so is a string given as the list. The classic way is a fold, which fails on an empty list:

<!-- check-docs: strict 2 -->
```kgpy
?> {x,", ",y}/["a" "b" "c"]
"a, b, c"
?> .join(["a" "b" "c"];", ")
"a, b, c"
?> .join([];", ")
""
?> .join([0ca "bc"];"-")
"a-bc"
?> .join(["a" 1];",")
error: .join: x must be a list of strings, got a list with integer
?> .join(.split("a,,b";",");",")
"a,,b"
```

`.words(s)` returns the non-empty pieces of `s` between runs of white space, and `.trim(s)` returns `s` without white space at its start and end. White space is the space, the tab, the newline and the carriage return.

<!-- check-docs: strict 2 -->
```kgpy
?> .words("  to be  or not ")
["to" "be" "or" "not"]
?> #.words("   ")
0
?> .trim("  a b  ")
"a b"
```

`.num(s)` returns the number that the whole of `s` spells: an integer, or a real in Klong's number syntax, with an optional `-` and optional white space around it. Anything else is `:undefined`: two numbers, text, or nothing. The classic way is `.rs`, which reads any data object and stops after the first one, so `.rs("7 8")` is `7` and `.rs("x")` is the symbol `x`.

<!-- check-docs: strict 2 -->
```kgpy
?> .num("12")
12
?> .num("-3.5")
-3.5
?> .num(" 7 ")
7
?> .num("7 8")
:undefined
?> .num("x")
:undefined
?> .num("")
:undefined
?> .rs("7 8")
7
```

To read a column of numbers from text, apply `.num` to each piece. A piece that is not a number gives `:undefined`, which the match operator `~` tests for:

<!-- check-docs: strict 2 -->
```kgpy
?> .num'.split("3,4.5";",")
[3.0 4.5]
?> :undefined~.num("x")
1
```

## Gradient

`.grad(f;x)` is the numeric gradient of `f` at `x`, the same as `x∇f`, for keyboards without the `∇` character.

The ASCII forms are `.grad(f;x)` for `x∇f` and `.jacobian(f;x)` for `x∂f`; both take the function first, and the operators take the point first. [Operators](operators.md#comparison) has the table that compares all the gradient operators.

The monad `∇f` and the multi-parameter `[w b]∂f` are available only as glyphs. The point is taken by value, so `.grad(f;p)` with a variable `p` uses its value.

<!-- check-docs: strict 2 -->
```kgpy
?> f::{x^2}
:monad
?> 3∇f
6.000000000838668
?> .grad(f;3)
6.000000000838668
?> .grad({+/x^2};[1 2 3])
[2.000000000279556 3.9999999996709334 6.000000000838668]
?> [1 2 3]∇{+/x^2}
[2.000000000279556 3.9999999996709334 6.000000000838668]
```

How to type `∇` and `∂` is in [PyTorch Backend and Autograd](torch_backend.md#typing-the-gradient-symbols).

## The `it` variable

In the REPL, `it` holds the value of the last expression that printed a result. In a program you assign the result yourself. Errors, `]` commands and expressions with no result leave `it` as it was, and a file run or `-e` never changes it. See [the REPL](repl.md#the-it-variable).

```kgpy
?> 1+2
3
?> it*2
6
?> r::it+1
7
```

## See also

[What's new in v2](whats-new-v2.md), [Migrating to v2](migrating-to-v2.md), [Differences from Klong](differences.md), [Conformance Suite](conformance.md), [Roadmap](roadmap.md).
