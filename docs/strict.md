# Strict mode

Klong accepts a lot of input that is almost certainly a mistake. `1 2 3` evaluates to `1` and drops the rest. An undefined name evaluates to its own symbol. A function called with too many arguments ignores the extras. Strict mode turns these into errors. It has three levels, and the default is level 0, which behaves as Klong always has.

| Level | What it adds |
|---|---|
| 0 | Nothing is rejected. Input that is probably a mistake gets a [warning](errors.md#syntax-warnings) and runs as it always did. |
| 1 | The parser's warnings become syntax errors. A function cannot create a variable by assignment, and cannot read one that is not defined. The left side of `::` must be a name. |
| 2 | Level 1, plus: reading an undefined name is an error everywhere, `:a` is always the symbol `a`, a declared local cannot be read before it is assigned, and a call with more arguments than the function takes is an error. |

Each level includes the rules of the one below it. Rules that are errors at every level (an adverb that does not touch its verb, say) are not strict-mode rules and are listed in [Errors](errors.md).

## Selecting a level

| From | How |
|---|---|
| The command line | `kgpy --strict` is level 1. `kgpy --strict=2` is level 2 (`kgpy --strict 2` also, as a lone `0`, `1` or `2` after the bare flag is its level). `--strict=0` is the default. A level other than 0, 1 or 2 is refused. |
| Python | `KlongInterpreter(strict=2)`. The level is the `strict` property of the interpreter, which can be assigned. Only the integers 0, 1 and 2 are accepted; anything else raises `ValueError`. |
| Klong code | `.strict()` sets level 1, `.strict(n)` sets level `n`, `.unsafe()` sets level 0. Each returns the new level. |

```console
$ kgpy -e 'a'
:a
$ kgpy --strict=2 -e 'a'
<expr>:1:1: name error: undefined: a
  a
  ^
$ kgpy --strict=3 -e 'a' >/dev/null 2>&1; echo $?
2
```

```python
from klongpy import KlongInterpreter

klong = KlongInterpreter(strict=2)
klong.strict        # 2
klong.strict = 1    # change it later
```

In a session, `.strict` and `.unsafe` set the level for what runs afterwards:

```kgpy
?> .strict(2)
2
?> .strict()
1
?> .unsafe()
0
?> .strict(3)
error: the strict level must be 0, 1 or 2
```

A file run with `kgpy --strict=N file.kg`, and a file loaded with `.l`, run at the level of the session. The level in force when a text is parsed decides the syntax checks for that text; the level in force when a statement runs decides the run-time checks. The same text can therefore be accepted at level 0 and rejected at level 1 by the same interpreter (the first example under [Input that is left over](#input-that-is-left-over)).

The default stays at level 0 so that existing programs keep their results. [`kgpy lint`](lint.md) finds many of the same mistakes in a file without running it.

## Level 1

### Input that is left over

At level 0 each of these is a warning and the program runs as it did. At level 1 each is a syntax error and nothing is evaluated.

A complete statement followed by more input:

```kgpy
?> 1 2 3
1
?> .strict(1)
1
?> 1 2 3
error: unexpected '2' after the end of the expression
```

A closer with nothing open:

```kgpy
?> 1+1 )
2
?> .strict(1)
1
?> 1+1 )
error: unexpected ')': nothing is open that it could close
```

A missing operand:

```kgpy
?> f::
None
?> .strict(1)
1
?> f::
error: missing right operand for ::
?> +/
error: missing operand for +/
```

A string, list, dictionary or comment that is not closed:

```kgpy
?> "abc
"abc"
?> [1 2 3
[1 2 3]
?> .strict(1)
1
?> "abc
error: unterminated string: the closing " is missing
?> [1 2 3
error: unterminated list: the closing ] is missing
?> :{[1 2]
error: unterminated dictionary: the closing } is missing
?> 3:"x
error: unterminated comment: the closing " is missing
```

A newline after a comment inside a function body is a separator at level 0, as if it were a `;`. At level 1 it is an ordinary newline, so the next line continues the expression:

```kgpy
?> f::{x :"note"
.. +2}
:monad
?> f(1)
2
?> .strict(1)
1
?> f::{x :"note"
.. +2}
:monad
?> f(1)
3
```

[Syntax warnings](errors.md#syntax-warnings) lists every code and what it does at level 0.

### The left side of `::`

Assigning to something that is not a name stored a meaningless entry at level 0. A name in parentheses, `(a)::5`, is still a name.

```kgpy
?> 1::5
5
?> "a"::5
5
?> .strict(1)
1
?> 1::5
error: cannot assign: the left side of :: must be a name
?> "a"::5
error: cannot assign: the left side of :: must be a name
?> (a)::5
5
```

A list on the left of `::` is a [destructuring](language-additions.md#destructuring) when it holds distinct plain names. Any other list was accepted at level 0 and bound nothing; at level 1 it is an error:

```kgpy
?> [a [b c]]::[1 [2 3]]
[1 [2 3]]
?> a
:a
?> .strict(1)
1
?> [a [b c]]::[1 [2 3]]
error: cannot destructure: nested lists are not supported
?> [a a]::[1 2]
error: cannot destructure: a is named twice
```

In a named-parameter definition, the parser checks the parameter names at every strict level. Reserved `x`, `y`, or `z` and repeated names raise a syntax error at the definition. [Named parameters](language-additions.md#named-parameters)

### Assigning inside a function

At level 0 `::` on an unknown name inside a function creates a variable that belongs to that call and disappears when it returns. It does not create a global, and `.unsafe()` does not change that. At level 1 the assignment is an error unless the name is a parameter, a declared local, or already defined.

```kgpy
?> f::{c::1}
:nilad
?> f()
1
?> .strict(1)
1
?> f()
error: undefined variable: c
```

Declare the name as a local, `{[c];c::1}`, or define the global before the function runs. To create a global from inside a function, use [`.E`](system-functions.md).

```kgpy
?> .strict(1)
1
?> f::{[c];c::1;c+1}
:nilad
?> f()
2
?> total::0
0
?> add::{total::total+x}
:monad
?> add(5)
5
?> total
5
```

### Reading an undefined name inside a function

```kgpy
?> g::{a+1}
:nilad
?> g()
error: cannot apply + (Plus) to symbol and integer
?> .strict(1)
1
?> g()
error: undefined: a
?> a::2
2
?> g()
3
```

At level 1 a name read at the top level that is not defined is still its own symbol: `a` prints `:a`. Level 2 closes that.

## Level 2

### Undefined names

Reading a name that is not defined is a name error, at the top level too. The message carries a "did you mean" hint when one defined name is close to it. The hint uses the names defined in the interpreter and a similarity cutoff of 0.7, so a one-letter slip is caught and a very different name gets no hint. The first example shows the error as the REPL prints it, with the hint line.

<!-- check-docs: strict 2; render -->
```kgpy
?> lenght::3
3
?> lenth
1:1: name error: undefined: lenth
  lenth
  ^^^^^
  hint: did you mean lenght?
?> zzz
1:1: name error: undefined: zzz
  zzz
  ^^^
```

A failed read defines nothing, so `a` is still undefined after `a+1` fails. Without strict mode, `a+1` fails inside the operator, with an error about a symbol and an integer, and `a` is defined as `:a`. Calling a function that is not defined is a name error at every level:

<!-- check-docs: strict 2 -->
```kgpy
?> a+1
error: undefined: a
?> a
error: undefined: a
?> nosuch(1)
error: undefined function: nosuch
```

### Quoted symbols

At levels 0 and 1, `:foo` is looked up as a variable when one called `foo` exists. At level 2 a quoted symbol is always the symbol. This is what lets a program keep symbols as data, such as the keys of a dictionary or the column names of a table, whatever variables happen to be defined.

```kgpy
?> foo::5
5
?> :foo
5
?> .strict(2)
2
?> :foo
:foo
?> {x}'[:foo :bar]
[:foo :bar]
```

A name that is not quoted is looked up, and an undefined name is an error at level 2, so `:foo` and `foo` are different things. To get the value of the variable, write `foo`. A list of symbols written `[:a :b]` is data at every level.

### Locals read before assignment

A declared local starts out as its own symbol at level 0 and 1. At level 2 reading it before it is assigned is an error:

```kgpy
?> f::{[a];a}
:nilad
?> f()
:a
?> .strict(2)
2
?> f()
error: local 'a' read before assignment
```

### Too many arguments

A function takes as many arguments as the highest of `x`, `y` and `z` it uses, or as many as it has named [parameters](language-additions.md#named-parameters). At levels 0 and 1 extra arguments are dropped. At level 2 they are an arity error. A call with fewer arguments, written `f(1;)` with the empty slot or `f(1)` without it, is a projection, and is not an error at any level.

```kgpy
?> {x+y}(1;2;3)
3
?> .strict(2)
2
?> {x+y}(1;2;3)
error: the function takes 2 arguments, got 3
?> f::{x}
:monad
?> f(1;2)
error: f takes 1 argument, got 2
?> {x+z}(1;2;3)
4
?> p::{x+y}(1;)
:monad
?> p(10)
11
?> q::{x+y}(1)
:monad
?> q(10)
11
```

`{x+z}` takes three arguments because it uses `z`, so a call with three is correct. The check also applies to functions called through adverbs and to system functions:

```kgpy
?> .strict(2)
2
?> {1}'[1 2]
error: the function takes no arguments, got 1
?> .p("a";"b")
error: .p takes 1 argument, got 2
```

An `x`, `y` or `z` inside a nested function literal does not count for the outer function. See [how many arguments a function takes](language-additions.md#how-many-arguments-a-function-takes).

## Which level to use

Use level 2 for new code. It reports a mistyped name at the place it is written, not several operations later as a numpy error about a string and an integer, and it catches calls that pass arguments a function never reads. Write new files so that they run under `--strict=2` and keep it on in tests.

Use level 1 to bring an existing program under strict checking, because it needs no change to how names are read: what it rejects is input that was already wrong or that never did what it appeared to do. Fix what it reports, then try level 2.

Leave level 0 on for programs written before these levels existed, and for a program that relies on one of the behaviours above. The interactive REPL also starts at level 0.

The standard library under `klongpy/lib` loads at levels 0, 1 and 2, so a strict program can load it with `.l`.

Run [`kgpy lint`](lint.md) over a file to see the same kinds of mistake without running anything.

See also: [Errors](errors.md), [Conformance Suite](conformance.md) (the `strict1` and `strict2` cases), [Migrating to v2](migrating-to-v2.md).
