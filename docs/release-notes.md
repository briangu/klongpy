# Release Notes

## Next release: behaviour changes

This list is the changes that can alter what an existing program does or prints. Each item gives the old behaviour, the new one and an example. The examples are run against the interpreter when the documentation is checked, so the new output shown is what it prints.

"Old" means the last release, 0.7.0. Where the old behaviour was an unrelated Python error, only its kind is given.

### Syntax that changed meaning

These inputs were accepted by the last release and ran without an error, and now mean something else. A program that contains one of them can give a different result. [`kgpy --lint`](lint.md) and [strict mode](strict.md) point at some of them. The old behaviour shown is that of the last release; the examples print what the interpreter now does.

**`[a b]::v` binds the names.** A list of names on the left of `::` used to be accepted, to evaluate the right side and to bind nothing. It is now a [destructuring](language-additions.md#destructuring): `a` and `b` are set to the members of `v`, so a name that already held a value changes. `a::5;[a b]::[1 2];a` was 5 and is 1. A list on the left that is not a list of distinct plain names (nested, repeated, `x y z`, empty) still binds nothing at strict level 0, now with a warning, and is a syntax error at level 1 and 2. A wrong number of members is a `shape` error, and a right side that is not a list is a `type` error.

```kgpy
?> a::5
5
?> [a b]::[1 2]
[1 2]
?> a
1
?> [a b]::[1 2 3]
error: cannot destructure 3 values into 2 names
```

**`f(a;b)::{...}` defines `f`.** It used to evaluate the call `f(a;b)` and define nothing, so `f` stayed undefined. If `f`, `a` and `b` existed, it called `f` and threw the result away. It is now a definition with [named parameters](language-additions.md#named-parameters), and it is not limited to three: `f(a;b;c;d)::{...}` is allowed.

```kgpy
?> f(a;b)::{a+b}
:dyad
?> f(1;2)
3
?> g(a;b;c;d)::{a+b+c+d}
:triad
?> g(1;2;3;4)
10
```

**`d@"k"` returns the value.** On a dictionary, `d@k` returned the dictionary itself when `k` was a string, symbol or real number, and raised a Python `KeyError` for a missing integer key. It is now the same as `d?k` for every key: the value, or `:undefined` when the key is missing. `d@[k1 k2]` is the list of values. `$d` was Python's text for a dictionary, `{'a': 1}`, and is now the Klong form. See [Dictionaries](language-additions.md#dictionaries).

```kgpy
?> d:::{["a" 1] ["b" 2]}
:{["a" 1] ["b" 2]}
?> d@"a"
1
?> d@"zz"
:undefined
?> d@["a" "b"]
[1 2]
?> $d
":{[""a"" 1] [""b"" 2]}"
```

**The arity of `{-x}` is one.** A function's arity is the highest of `x`, `y` and `z` its body uses, wherever it uses them. It was the highest used outside a monad or a call, so `{-x}`, `{#x}` and `{g(x+1)}` counted as having no arguments, and `{y}` as having one. The display, calls from Python, projections and the choice of monad or dyad under an adverb all follow the arity. `f::{-x}` displays as `:monad`, where it displayed as `:nilad`, and `klong['f'](5)` from Python is -5, where it raised an arity error. A written call such as `f(5)` worked before and still does. The arguments of a nested function literal do not count toward the enclosing function. See [How many arguments a function takes](language-additions.md#how-many-arguments-a-function-takes).

```kgpy
?> f::{-x}
:monad
?> f(5)
-5
?> g::{#x}
:monad
?> h::{x+z}
:triad
```

**`{y}(1)` is an incomplete call, not `:y`.** `{y}` has two arguments now. A call with one of them used to bind it to `x` and return the unbound `y` as the symbol `:y`. It now gives a function value, as `{x+y}(1)` always has, and the value is not completed by calling it again. A projection is written with the empty slot, `{y}(1;)`, and is completed by `p(5)`. A call with every argument, `{y}(1;2)`, is 2.

```kgpy
?> {y}(1;2)
2
?> p::{y}(1;)
:dyad
?> p(5)
5
```

**A recursive call through `.f` has its own locals.** `.f` called itself without declaring the function's locals, so the inner call changed the outer call's variable. `f::{[t];t::x;:[x<2;t;t+.f(x-1)]};f(3)` was 3, where the same function written with its own name gives 6.

```kgpy
?> f::{[t];t::x;:[x<2;t;t+.f(x-1)]}
:monad
?> f(3)
6
```

### Parser

**A string or character that looks like punctuation is a string or character.** `"{"`, `";"`, `")"`, `":["`, `0c{` and the like were taken for the opener, separator or closer they spell. `"{"` was a syntax error, `";"` at the start of a statement was an empty statement, `f(")")` was a syntax error and `["[" 1 2]` was the nested list `[[1 2]]`.

```kgpy
?> "{"
"{"
?> ";"
";"
?> ["[" 1 2]
["[" 1 2]
?> f::{x};f(")")
")"
```

**Whitespace before the `;` of a conditional is accepted.** `:[1 ;"a";"b"]` was a syntax error.

```kgpy
?> :[0 ;"a" ;"b"]
"b"
```

**Input that is left over is reported.** `1 2 3` evaluates `1` and drops the rest, as it always did, and now also warns. The same holds for a closer with nothing open, a missing operand, and an unterminated string, list, dictionary or comment. Under `.strict(1)` these are syntax errors. See [Syntax warnings](errors.md#syntax-warnings) for each case.

```kgpy
?> 1 2 3
1
?> .strict(1)
1
?> 1 2 3
error: unexpected '2' after the end of the expression
```

**An adverb that does not touch its verb is a syntax error when the text is parsed.** It used to fail with a `KeyError` when it was evaluated, so a function that contained one only failed when called.

```kgpy
?> {1 '2}
error: adverb ' must directly follow its verb
```

**A dictionary literal entry that is not a `[key value]` pair is a syntax error.** It was an `IndexError` or `TypeError`.

```kgpy
?> :{1}
error: each entry of a dictionary literal must be a [key value] pair
```

### Operators

**A symbol held as data is not looked up as a variable.** It stays a symbol when it passes through an adverb, `@`, a projection or a callback. `{x}'[:foo :bar]` gave `[5 :bar]` when `foo` was 5. To pass the values of variables, build the list from values with `[;a;b]`.

```kgpy
?> foo::5
5
?> {x}'[:foo :bar]
[:foo :bar]
?> {x}'[;foo;:bar]
[5 :bar]
```

**Amend (`:=`) checks its index and keeps the value.** A string amended at `#a` appends; a larger index is a range error (it used to change nothing); a negative index counts from the end. A real put into an integer list turns the list into reals (the value was truncated). An index out of range in a list is a range error, not a NumPy `IndexError`.

```kgpy
?> "abc":="x",3
"abcx"
?> "abc":="x",-1
"abx"
?> [1 2 3]:=2.5,1
[1.0 2.5 3.0]
?> [1 2 3]:=1,5
error: :=: index 5 out of range for length 3
```

**Group lists its groups in the order the values first appear, and range keeps mixed values apart.** `=[3 1 3 2 1]` listed the group of 1 first, and `=[10 "10" 10]` was a `TypeError`. `?[10 "10" 10]` was `[10]`.

```kgpy
?> =[3 1 3 2 1]
[[0 2] [1 4] [3]]
?> =[10 "10" 10]
[[0 2] [1]]
?> ?[10 "10" 10]
[10 "10"]
```

**Take on a list of lists takes rows.** `5#[[1 2] [3 4]]` gave `[[1 2 1 2] [3 4 3 4]]`.

```kgpy
?> 5#[[1 2] [3 4]]
[[1 2] [3 4] [1 2] [3 4] [1 2]]
```

**Two operators are equal when they are the same operator.** `[:+]~[:+]` was 0 because the two operator objects were compared by identity.

```kgpy
?> [:+]~[:+]
1
```

**Division by zero in a list gives `:undefined` for that element.** `%`, `:%` and `!` gave `inf`, or 0 for `!`, in a list. A scalar already gave `:undefined`.

```kgpy
?> [1 2 0]%[1 0 1]
[1.0 :undefined 0.0]
?> [5 6]![0 2]
[:undefined 0]
```

**Iterate, scan-iterate and split check their counts.** A count that is negative or not a whole number looped without end; split sizes of 0 raised a `ZeroDivisionError`. Whole-number reals such as `2.0` are accepted.

```kgpy
?> (-1){x+1}:*0
error: :*: count must not be negative, got -1
?> 2.0{x+1}:*0
2
?> 0:#[1 2 3]
error: :#: segment sizes must be positive, got 0
```

**Form (`:$`) gives `:undefined` for text that does not convert.** It raised a `ValueError`.

```kgpy
?> 1:$"x"
:undefined
?> [1 1]:$["12" "x"]
[12 :undefined]
```

**Several operators that crashed or looped now return a value or a Klong error.** `$[1 2 3]` recursed until Python stopped it. `|5` was a `TypeError`. `&/"abc"` and `+/"abc"` failed on a string. `[2 2]:^[]` and `5:^[]` raised a `ZeroDivisionError`. A dictionary joined with something that is not a two-member list was silently ignored.

```kgpy
?> $[1 2 3]
["1" "2" "3"]
?> |5
5
?> &/"abc"
"a"
?> +/"abc"
"abc"
?> [2 2]:^[]
error: :^: cannot fill a non-empty shape from an empty source
?> :{[1 2]},[1 2 3]
error: ,: a dictionary can only be joined with a tuple (a list of two members) or a dictionary
```

**Take with a number or a symbol on the right repeats it.** `3#7` was a Python `TypeError`, and `2#:foo` took two characters of the string `"foo"`. An atom is repeated, and `0#7` is the empty list.

```kgpy
?> 3#7
[7 7 7]
?> (-3)#7
[7 7 7]
?> 0#7
[]
?> 2#:foo
[:foo :foo]
```

**Integers beyond 64 bits are exact.** An operand outside the 64-bit range made `+ - * % < > | &` fail with a Python `OverflowError`, although the language already made such integers (`2^64`, a long literal). The operation is now done on exact integers, in a list as well as on a scalar, and a result that fits in 64 bits again is an ordinary integer. Operands that are both inside 64 bits keep their wrap-around, as before: `9223372036854775807+1` is `-9223372036854775808`. `^` is computed in floating point, so `10^400` is still an error, and `!` is an error for such an operand. The compiled fast path and the interpreter now give the same answer for these expressions.

```kgpy
?> 1+9223372036854775808
9223372036854775809
?> [1 2]+9223372036854775808
[9223372036854775809 9223372036854775810]
?> 9223372036854775808*3
27670116110564327424
?> 9223372036854775807+1
-9223372036854775808
?> 10^400
error: cannot apply ^ (Power) to integer and integer
```

**Each on a symbol passes the symbol.** `f'x` with a symbol `x` is `f(x)`. The symbol was taken for the string of its name and iterated over its characters.

```kgpy
?> s:::foo
:foo
?> {x}'s
:foo
```

### System functions

**`.sys` runs a shell command and returns its exit status.** It tried to start a program named by the whole string.

```kgpy
?> .sys("exit 3")
3
```

**`.timer` checks its arguments.** The interval may be a real number such as `0.5`. A negative or non-numeric interval, or a callback that is not a function, is an error; they were a returned string or a `ValueError`. An error in a callback is logged and cancels the timer.

```kgpy
?> .timer("t";-1;{1})
error: y must be a non-negative number
```

**`.pyc` passes positional arguments.** A Python function that takes its arguments by position lost them when called through `.pyc`. `hypot(3;4)` gave 0.0.

```kgpy
?> .pyf("math";"hypot")
1
?> .pyc("hypot";[3 4];:{})
5.0
```

**`.webc` stops a web server.** `.webc(h)` returned 0 and left the server running, because it did not recognise the handle that `.web` returns. It returns 1 and stops the server. See [the web server](web_server.md).

**A missing key in a `.kvs` store is `:undefined`.** It raised an error carrying the key as its message, where [the key-value store page](table_and_key_value_stores.md) says `:undefined`.

<!-- check-docs: requires duckdb pandas -->
```kgpy
?> .py("klongpy.db")
1
?> kvs::.kvs("/tmp/kvs-missing")
/tmp/kvs-missing:kvs
?> kvs?"nothing"
:undefined
```

**`.db` accepts a dictionary literal that holds tables.** `.db(:{["T" T]})` built a database whose queries failed with `'Database' object has no attribute 'user'`. It now works like `.db(:{},"T",,T)`.

<!-- check-docs: requires duckdb pandas -->
```kgpy
?> .py("klongpy.db")
1
?> T::.table([["a" [1 2 3]] ["b" [4 5 6]]])
a b
1 4
2 5
3 6
?> db::.db(:{["T" T]})
:db
?> db("select sum(a) from T")
6.0
```

**New: `.fmt`, `.pf` and `.grad`.** `.fmt(template;args)` and `.pf(template;args)` format text with `{}` placeholders, and `.grad(f;x)` is the ASCII name of `x∇f`. See [Friendlier syntax](language-additions.md#formatted-text).

### Strict mode

The default stays level 0, which behaves as before. See [Strict mode](strict.md) for the rules.

**`.strict(n)` sets level `n`.** `.strict(2)` and `.strict(0)` were accepted and silently set level 1, because the extra argument was dropped. The level is one of 0, 1 and 2, and anything else is an error. `.strict()` is still level 1, and `.unsafe()` is level 0. Each returns the level.

```kgpy
?> .strict(2)
2
?> .strict(0)
0
?> .strict(3)
error: the strict level must be 0, 1 or 2
```

**`kgpy --strict[=N]` and `KlongInterpreter(strict=N)` choose the level.** There was no flag and no constructor argument; the level could only be set from Klong code. Bare `--strict` is level 1. Files loaded with `.l` run at the level of the session.

```
$ kgpy --strict=2 -e 'a+1'
<expr>:1:1: name error: undefined: a
  a+1
  ^
```

**At level 1, `1::5` is a syntax error, and a read in a function reports the read.** The left side of `::` that is not a name was accepted and stored a meaningless entry. A function that read a name that was not defined was told to declare it in the parameter list, which is advice for an assignment. The message is now `undefined: a`. Level 1 also turns the parser's warnings into errors, as [Errors](errors.md#syntax-warnings) lists.

```kgpy
?> .strict(1)
1
?> 1::5
error: cannot assign: the left side of :: must be a name
?> f::{a+1}
:nilad
?> f()
error: undefined: a
```

**Level 2 is new.** Reading an undefined name is a name error with a "did you mean" hint, `:foo` is always the symbol, a declared local read before it is assigned is an error, and a call with more arguments than the function takes is an arity error.

```kgpy
?> a+1
error: cannot apply + (Plus) to symbol and integer
?> .strict(2)
2
?> c+1
error: undefined: c
?> foo::5
5
?> :foo
:foo
?> {x+y}(1;2;3)
error: the function takes 2 arguments, got 3
```

**There is no `global:` prefix.** The `.strict()` documentation promised that a function could create a global with `[global:counter]`. Nothing implemented it; `[global:counter]` was an ordinary locals list holding the names `global` and `counter`. The promise is deleted. To change a global from a function, define it first at the top level; to create one from inside a function, use `.E`.

### Friendlier syntax

New forms are described in [Friendlier syntax](language-additions.md). These are the changes in behaviour that come with them, apart from the five in [Syntax that changed meaning](#syntax-that-changed-meaning).

**A destructuring at the start of a function declares its names as locals.** `{[s k]::x;s+k}` needs no locals list at any strict level, and does not change a global named `s`.

**A dictionary literal inside a dictionary or list literal is a dictionary.** It was stored as an unusable call object. A list literal that holds one is copied each time it is evaluated, as a dictionary literal is.

```kgpy
?> d:::{["GET" :{["/" 2]}]}
:{["GET" :{["/" 2]}]}
?> d@"GET"
:{["/" 2]}
```

**`d,d2` merges two dictionaries.** It raised a Python `KeyError`. The entries of `d2` are copied into `d`, which changes in place, and an entry of `d2` replaces one with the same key. A right operand that is not a pair or a dictionary is still an error.

```kgpy
?> a:::{["x" 1]}
:{["x" 1]}
?> b:::{["y" 2]}
:{["y" 2]}
?> a,b
:{["x" 1] ["y" 2]}
```

**Python callers can pass arguments by name, and a named-parameter function takes any number of them.** `KGFnWrapper` ignored keyword arguments. `klong['f'](1, 2, d=4, c=3)` passes parameters by name, for a function written with named parameters or with `x`, `y` and `z`. Too few, too many, an unknown or a repeated name raises `KlongArityError`.

**A function with named parameters has no source text.** Over IPC it is sent like a projection, as an opaque value that the receiver can hold and send back but not call. See [How functions cross the wire](ipc_capabilities.md#how-functions-cross-the-wire).

### Errors

**Errors are `KlongError` classes with a position.** Every error KlongPy raises on purpose is a `KlongError` (`KlongTypeError`, `KlongRangeError` and so on, listed in [Errors](errors.md)), and an exception from NumPy inside an operator is reported as one, naming the operator. A Klong error that replaces a Python one is also an instance of the old class, so `except IndexError` and `except ValueError` still catch it.

```kgpy
?> 1+"a"
error: cannot apply + (Plus) to integer and string
?> [1 2 3]@3
error: index 3 out of range for list of 3
?> [1 2 3]+[1 2]
error: cannot apply + (Plus) to list of 3 and list of 2
```

The old messages were NumPy's, such as `ufunc 'add' did not contain a loop with signature matching types ...`, and `index 3 is out of bounds for axis 0 with size 3`.

**Messages changed.** A program that matches on message text needs the new text.

```kgpy
?> 1 f
error: undefined function: f
?> 1.2.3
error: malformed number '1.2.3'
```

These were `undefined: f` and a `ValueError` from `float()`. The error carries its position, its kind and the source: see [Reading an error](errors.md#reading-an-error).

### REPL and CLI

**Errors go to stderr, with the position and a caret.** At the REPL they were printed on stdout as `Error: (args...)`. A file run or `-e` printed a Python traceback; a traceback is now printed only with `-v` or `--traceback`.

```
$ kgpy -e '1+"a"'
<expr>:1:2: type error: cannot apply + (Plus) to integer and string
  1+"a"
   ^
```

**Exit status.** `kgpy file.kg` and `-e` exit with 1 after an error. Piped stdin exits 1 if any entry failed. `-t` runs every line, reports each failing line and exits after the last, with 1 if any failed; it no longer starts the REPL afterwards. Ctrl-C ends a file run with `interrupted` and status 130. A file that cannot be opened is `system error: cannot open ...`. See [the command line](errors.md#the-command-line).

**A multi-line entry is one entry.** When a bracket, string or comment is left open, the REPL prompts with `..` and reads on. Before, each line was evaluated alone, so `f::{` was an error. Piped input does the same.

```kgpy
?> f::{
.. x+1
.. }
:monad
?> f(2)
3
```

**`it` holds the last result.** `it` was an undefined name.

```kgpy
?> 1+2
3
?> it*2
6
```

**Line editing is built in.** History (`~/.kgpy_history`, or `KGPY_HISTORY`) and tab completion come from `readline`, so `rlwrap` is no longer needed. See [the REPL](repl.md).

### IPC

**The wire protocol is kdb+ IPC.** `.srv` and `.cli` used pickled Python objects, which let a peer run code in your process when a message was unpickled. The default is now the kdb+ protocol, and a q client can talk to a `.srv`. The old protocol remains for a transition and is deprecated: `KLONGPY_IPC=legacy` selects it, `auto` (the server default) accepts an old client, and `KLONGPY_IPC=kdb` refuses it. Upgrade servers first, then clients. See [the protocol](ipc_capabilities.md#the-protocol).

**An error on the server comes back as a Klong error and the connection stays open.** It used to drop the connection (`connection lost`), and the next call failed with a Python `AttributeError`.

```kgpy
server?> .srv(8888)
1
client?> f::.cli(8888)
remote[localhost:8888]:fn
client?> f("[1 2 3]@10")
error: index 10 out of range for list of 3
client?> f("1+1")
2
```

**`.srv` raises when it cannot listen.** A second `.srv` on a port that is in use returned 0 and did nothing. It raises `cannot listen on 0.0.0.0:8888: ...` followed by the system's reason. `.srv` on a server that is already running still returns 0.

**A call waits 30 seconds.** A call with no answer raises an error that contains `timed out`, closes the connection and fails every call waiting on it. Change the limit with `.clit`. See [Closing, Timeouts and `.clit`](ipc_capabilities.md#closing-timeouts-and-clit).

**Tables and functions cross the connection.** A `klongpy.db` table is sent as a kdb+ table and arrives as a table when `klongpy.db` is loaded; an index does not travel. A function literal is sent as its source text. See [the protocol](ipc_capabilities.md#the-protocol) and [How functions cross the wire](ipc_capabilities.md#how-functions-cross-the-wire).
