# Migrating to v2

This page lists what behaves differently from the released version, 0.7.0, for code that already exists. Each item gives what the old version did, what v2 does, why, and how to keep the old behaviour where a switch exists. The "before" results were run on the 0.7.0 source and the "now" results are the examples on this page, which are run when the documentation is checked.

Most programs need no change. The risk is in the first group, where the same input gives a different result and nothing complains.

## Check your code first

Before you upgrade, or right after, run these two on your programs.

`kgpy --lint` reads files without running them and reports several classic traps, such as a leading minus that negates more than you meant:

```console
$ cat > old.kg <<'EOF'
> total::0
> -1+2
> EOF
$ kgpy --lint old.kg
old.kg:2:1: W010 a leading - negates everything to its right: -1+... is -(1+...)
0 errors, 1 warning in 1 file
```

`kgpy --strict=1 file.kg` (or `--strict=2`) runs the program and turns the silent mistakes into errors, so the first one stops the run with its position:

```console
$ printf 'a::[1 2 3]\n.p(a)\nb::4 5 6\n' > prog.kg
$ kgpy prog.kg 2>&1
prog.kg:3:6: warning: unexpected '5' after the end of the expression; the rest of the input is not evaluated
  b::4 5 6
       ^
  hint: a list is written in brackets, [1 2 3]; statements are separated by ;
[1 2 3]
$ kgpy --strict=1 prog.kg 2>&1
prog.kg:3:6: syntax error: unexpected '5' after the end of the expression
  b::4 5 6
       ^
  hint: a list is written in brackets, [1 2 3]; statements are separated by ;
```

The first run warns and goes on, as the old version did without the warning, so `b` is 4. The second stops before anything runs. See [Lint](lint.md) and [Strict Mode](strict.md).

## Results that change without an error

### `[a b]::v` binds the names

Before: a list of names on the left of `::` evaluated the right side and bound nothing. Now it is a [destructuring](language-additions.md#destructuring), so a name that already had a value changes.

```kgpy
?> a::5
5
?> [a b]::[1 2]
[1 2]
?> a
1
```

`a::5;[a b]::[1 2];a` was 5 and is 1. Why: binding is what the form is for; it gave a silent no-op before. A list on the left that is not a list of distinct plain names (nested, repeated, empty) still binds nothing at level 0, now with a warning, and is a syntax error at strict levels 1 and 2. To keep the old reading, use the [legacy parser](#keep-the-old-reader).

### `f(a;b)::{...}` defines `f`

Before: it evaluated the call `f(a;b)` and defined nothing, so `f` stayed undefined (`undefined: f` on the next use). Now it defines `f` with named parameters, and it is not limited to three.

```kgpy
?> f(a;b)::{a+b}
:dyad
?> f(1;2)
3
```

### A function's arity is the highest of `x`, `y` and `z` it uses

Before: only the uses outside a monad or a call counted, so `{-x}`, `{#x}` and `{g(x+1)}` had no arguments and `{y}` had one. A function now has as many arguments as the highest of the three that its body uses, wherever it uses them. The display, calls from Python, projections and the choice of monad or dyad under an adverb all follow.

```kgpy
?> f::{-x}
:monad
?> f(5)
-5
```

`f::{-x}` displayed as `:nilad` before. From Python, `klong['f'](5)` raised `RuntimeError: Klong function called with 1 but expected 0` and is now -5; a wrong number of arguments from Python is a `KlongArityError`. A written call such as `f(5)` worked before and still does. The arguments of a nested function literal do not count toward the enclosing function. See [How many arguments a function takes](language-additions.md#how-many-arguments-a-function-takes).

### `{y}(1)` is a projection

Before: the call bound `1` to `x` and returned the unbound `y` as the symbol `:y`. Now `{y}` has two arguments and a call with one gives a projection that waits for the second.

```kgpy
?> p::{y}(1)
:monad
?> p(5)
5
?> {y}(1;2)
2
```

### A call with too few arguments is a projection

Before: `r::{x+y};s::r(1)` gave a value that could never be completed; `s(10)` returned it unchanged. Now `r(1)` is the same as `r(1;)`, a function that waits for the rest. It is allowed at every strict level. A function applied to too few values by `@` or an adverb gives a projection in the same way, so `{x+y}'[1 2 3]` is a list of three projections.

```kgpy
?> r::{x+y}
:dyad
?> s::r(1)
:monad
?> s(10)
11
```

Related changes that follow from one rule: a projection of a projection fills the first open slot (`f::{x,y,z};p::f(1;;);q::p(;3);q(2)` was `[1 None 3]` and is `[1 2 3]`), and a projection made inside a function keeps the values its arguments had there. A function that is passed as a parameter still does not curry; see [Differences from Klong](differences.md#a-function-passed-as-a-parameter-does-not-curry).

### `d@k` on a dictionary gives the value

Before: `d@k` returned the dictionary itself when `k` was a string, symbol or real number, and raised a Python `KeyError` for a missing integer key. Now it is the same as `d?k` for every key: the value, or `:undefined` when the key is missing. `d@[k1 k2]` is the list of values. `$d` was Python's text for a dictionary, `{'a': 1}`, and is now the Klong form.

```kgpy
?> d:::{["a" 1] ["b" 2]}
:{["a" 1] ["b" 2]}
?> d@"a"
1
?> d@"zz"
:undefined
?> $d
":{[""a"" 1] [""b"" 2]}"
```

Code that relied on `d@"a"` returning the dictionary needs `d`. Use `d?"a"` if you want code that reads the same in both versions.

### `:undefined` is one value

Before: `:undefined` written in a program was the symbol named `undefined`, which prints the same and is a different value, so `u::1%0;u~:undefined` was 0.

```kgpy
?> u::1%0
:undefined
?> u~:undefined
1
```

### Division by zero is `:undefined` on every path

Before: `1%0` was `:undefined` but `[1 2 0]%[1 0 1]`, `a%0` on a list and `:%` gave `inf` (or a huge integer), and `7!0` was 0. Now every path gives `:undefined` for the element: a scalar, a list, an array, a function body and `:%` and `!`.

```kgpy
?> [1 2 0]%[1 0 1]
[1.0 :undefined 0.0]
?> a::[1 2 3]
[1 2 3]
?> a%0
[:undefined :undefined :undefined]
?> 7!0
:undefined
?> [7 8]!0
[:undefined :undefined]
```

A list that holds `:undefined` is not numeric, so arithmetic on it is an error, where it went through as `inf` before. `+/[1 2 3]%0` was `inf` and is a type error.

```kgpy
?> +/[1 2 3]%0
error: cannot apply +/ (Plus-Over) to list of 3: the list holds :undefined
```

Why: a scalar already gave `:undefined`, and the same expression should not depend on whether the operands were a list. There is no switch. Test for zero before you divide, or use `:_` to find the undefined elements.

### A monad applied to a conditional gets the value

Before: `-:[0;2;3]` applied the monad to the condition and was 0. The conditional gives 2 or 3, and the monad now applies to that. In the same way `#[;1;2;3]` was the size of the first element, 1, and is 3. A function call whose argument is a conditional was not affected.

```kgpy
?> -:[0;2;3]
-3
?> #[;1;2;3]
3
```

### A symbol held as data is not looked up

Before: a symbol that passed through an adverb, `@`, a projection or a callback was looked up as a variable, so `{x}'[:foo :bar]` gave `[5 :bar]` when `foo` was 5. Now it stays a symbol. To pass the values of variables, build the list from values with `[;a;b]`. Each on a symbol passes the symbol: `s:::foo;{x}'s` iterated over the characters of `"foo"` and is now `:foo`.

```kgpy
?> foo::5
5
?> {x}'[:foo :bar]
[:foo :bar]
?> {x}'[;foo;:bar]
[5 :bar]
```

### Operators with new results

These gave a wrong or accidental result and now give the right one. Each is also in the [release notes](release-notes.md#operators).

```kgpy
?> "abc":="x",-1
"abx"
?> [1 2 3]:=2.5,1
[1.0 2.5 3.0]
?> =[3 1 3 2 1]
[[0 2] [1 4] [3]]
?> 5#[[1 2] [3 4]]
[[1 2] [3 4] [1 2] [3 4] [1 2]]
?> 2#:foo
[:foo :foo]
?> [:+]~[:+]
1
```

- Amend with a negative index counts from the end. It used to change nothing (`"abc":="x",-1` was `"abc"`). A real put into an integer list turns the list into reals; the value was truncated. An index out of range is a range error, not a NumPy `IndexError`.
- Group lists the groups in the order the values first appear; `=[3 1 3 2 1]` listed the group of 1 first. `=[10 "10" 10]` was a `TypeError`.
- Take on a list of lists takes rows; `5#[[1 2] [3 4]]` gave `[[1 2 1 2] [3 4 3 4]]`. `3#7` was a Python `TypeError`, and `2#:foo` was `"fo"`.
- `[:+]~[:+]` was 0.
- `1:$"x"` was a `ValueError` and is `:undefined`. `$[1 2 3]` recursed until Python stopped it. `|5`, `&/"abc"` and `+/"abc"` were Python errors, and `[2 2]:^[]` a `ZeroDivisionError`. `:{[1 2]},[1 2 3]` was silently ignored and is an error.
- Integers beyond 64 bits are exact for `+ - * % < > | &`, where they raised `OverflowError`. `10^400` is still an error, now a Klong one.

### A recursive call through `.f` has its own locals

Before: `.f` called itself without declaring the function's locals, so the inner call changed the outer call's variable. `f::{[t];t::x;:[x<2;t;t+.f(x-1)]};f(3)` was 3, where the same function written with its own name gives 6.

```kgpy
?> f::{[t];t::x;:[x<2;t;t+.f(x-1)]}
:monad
?> f(3)
6
```

### A Python function brought in with `.py` gets every argument

Before: a function with optional parameters was called with its required ones only, so `linspace(0;1;5)` made 50 points and `arange(2;8)` was `[0 1]`.

```kgpy
?> .pyf("numpy";"linspace")
1
?> #linspace(0;1;5)
5
```

## Input that is now reported

### Left-over input warns, and is an error under strict

Before: `1 2 3` evaluated `1` and dropped the rest without a word, and so did a closer with nothing open, a missing operand and an unterminated string, list, dictionary or comment. Now each prints a warning on stderr at strict level 0 and the program goes on with the same result. At strict level 1 and 2 each is a syntax error. [Errors](errors.md#syntax-warnings) lists every case.

```kgpy
?> 1 2 3
1
?> .strict(1)
1
?> 1 2 3
error: unexpected '2' after the end of the expression
```

Why: `1 2 3` is nearly always a list written without brackets. To silence the warning from Python, use `warnings.filterwarnings("ignore", category=KlongSyntaxWarning)`. The legacy parser does not warn.

### Two syntax errors are caught when the text is read

An adverb that does not touch its verb (`{1 '2}`) and a dictionary literal entry that is not a `[key value]` pair (`:{1}`) used to fail with a `KeyError` or `TypeError` when the code ran, so a function that held one failed only when called. Both are syntax errors when the text is parsed, at every level.

```kgpy
?> {1 '2}
error: adverb ' must directly follow its verb
?> :{1}
error: each entry of a dictionary literal must be a [key value] pair
```

### Strings that look like punctuation are strings

Before: `"{"`, `";"`, `")"` and `":["` were taken for the opener, separator or closer they spell. `"{"` was a syntax error, `";"` at the start of a statement was an empty statement, `f(")")` was a syntax error, and `["[" 1 2]` was the nested list `[[1 2]]`. Whitespace before the `;` of a conditional was a syntax error and is accepted.

```kgpy
?> "{"
"{"
?> ["[" 1 2]
["[" 1 2]
?> :[0 ;"a" ;"b"]
"b"
```

A program that worked around the old behaviour (a string written another way to dodge `"{"`) still runs.

## Strict mode

The default is still level 0, which behaves as before. What changed under it:

- `.strict(n)` sets level `n`. `.strict(2)` and `.strict(0)` were accepted and silently set level 1. Anything but 0, 1 or 2 is an error.
- `kgpy --strict[=N]` and `KlongInterpreter(strict=N)` choose the level; there was no flag and no constructor argument.
- At level 1, `1::5` is a syntax error where it stored a meaningless entry, and reading an undefined name in a function says `undefined: a`, where it gave advice for assignments.
- `x`, `y` and `z` can be assigned in a function at level 1 and 2. `h::{x::7};h(1)` was `undefined variable: x`. They are locals of the call and the caller's value does not change.
- Level 2 is new: an undefined name is an error, `:foo` is always the symbol, a declared local read before assignment is an error, and too many arguments is an arity error.
- The `[global:counter]` form that the old `.strict()` text promised never worked and is gone. To create a global from inside a function, use `.E`.

```kgpy
?> .strict(2)
2
?> {x+y}(1;2;3)
error: the function takes 2 arguments, got 3
```

## Errors

Every error KlongPy raises on purpose is a `KlongError` with a position, a kind and the source line. An exception from NumPy inside an operator is reported as one too, naming the operator. A Klong error that replaces a Python one is also an instance of the old class, so `except IndexError`, `except ValueError`, `except TypeError` and `except FileNotFoundError` still catch it, and `except KlongException` still catches every one. The classes are exported from `klongpy` (`KlongTypeError`, `KlongRangeError`, `KlongNameError` and so on).

```kgpy
?> 1+"a"
error: cannot apply + (Plus) to integer and string
?> [1 2 3]@3
error: index 3 out of range for list of 3
?> [1 2 3]+[1 2]
error: cannot apply + (Plus) to list of 3 and list of 2
?> 1.2.3
error: malformed number '1.2.3'
?> 1 f
error: undefined function: f
```

The messages changed, so a program that matches on message text needs the new text. These were `ufunc 'add' did not contain a loop with signature matching types ...`, `index 3 is out of bounds for axis 0 with size 3`, `atomic dyad shape mismatch: (3,) and (2,)`, a `ValueError` from `float()` and `undefined: f`. A Python class that was an unrelated error, such as the `KeyError` from `{1 '2}`, is now a Klong error of the right kind and is no longer that Python class. There is no switch. See [Errors](errors.md) for the kinds.

## The command line and REPL

- **Errors go to stderr.** At the REPL they were printed on stdout as `Error: (args...)`; a file run or `-e` printed a Python traceback. Now each prints the position, the source line and a caret on stderr, and a traceback only with `-v` or `--traceback`.
- **Exit status.** `kgpy file.kg` and `-e` exit with 1 after an error, and piped stdin exits 1 if any entry failed. `-t` runs every line, reports each failing line and exits after the last with 1 if any failed; it no longer starts the REPL afterwards. Ctrl-C ends a file run with `interrupted` and status 130.
- **Multi-line entry.** An open bracket, string or comment now makes the REPL read on. Before, each line was evaluated alone, so `f::{` was an error. Piped input does the same.
- **`it`** holds the last printed value. It was an undefined name, so a program that used `it` as a variable of its own still works.
- **`rlwrap` is not needed.** History, in `~/.kgpy_history` or `KGPY_HISTORY`, and Tab completion are built in.

```console
$ kgpy -e '1+"a"'; echo $?
<expr>:1:2: type error: cannot apply + (Plus) to integer and string
  1+"a"
   ^
1
```

## IPC speaks the kdb+ protocol

Before: `.srv` and `.cli` sent pickled Python objects. A peer could run code in your process when its message was unpickled. Now the default is the kdb+ IPC protocol: a q process can talk to a `.srv`, and nothing is unpickled from the network. See [the protocol](ipc_capabilities.md#the-protocol).

The old protocol remains for a transition and is deprecated. It still unpickles what it receives, which runs code chosen by the sender, so use it on trusted networks only. `KLONGPY_IPC` chooses:

| Value | `.srv` accepts | `.cli` and `.clid` speak |
|---|---|---|
| `auto` (the default for `.srv`) | both | |
| `kdb` (the default for `.cli`) | the kdb+ protocol only | the kdb+ protocol |
| `legacy` | the pickle protocol only | the pickle protocol |

Upgrade servers first, then clients. A server on `auto` still accepts an old client, with a `FutureWarning` on its first use of the pickle protocol. Set `KLONGPY_IPC=kdb` on the server to refuse old clients, or `KLONGPY_IPC=legacy` on a new client to talk to a server that has not been upgraded. A value other than the three is a system error:

```console
$ printf '.cli(8891)\n' | KLONGPY_IPC=bogus kgpy 2>&1 >/dev/null
1:1: system error: KLONGPY_IPC must be one of kdb, legacy, auto, not 'bogus'
  .cli(8891)
  ^^^^^^^^^^
```

Other changes that come with the new protocol:

- An error on the server comes back to the client as a Klong error and the connection stays open. It used to drop the connection (`connection lost`), and the next call failed with an `AttributeError`.
- `.srv` raises when it cannot listen. A second `.srv` on a port in use returned 0 and did nothing; it now raises `cannot listen on 0.0.0.0:8888: ...`. `.srv` on a server that is already running still returns 0.
- A call waits 30 seconds and then raises an error that contains `timed out`, closes the connection and fails every call waiting on it. Change the limit with `.clit`.
- A `klongpy.db` table is sent as a kdb+ table. A function is sent as its source text. A function with named parameters has no source text, so it is sent as an opaque value that the receiver can hold and send back but not call. See [How functions cross the wire](ipc_capabilities.md#how-functions-cross-the-wire).
- A numpy `datetime64` or `timedelta64` can be sent, as a kdb+ timestamp or timespan in nanoseconds. It was a type error.
- A server accepts every client unless the program defines `.srv.auth`. See [Credentials](ipc_capabilities.md#credentials-srvauth).

There is still no encryption. See the security model in the [README](https://github.com/briangu/klongpy#security-model-trusted-networks).

## System functions

These were wrong or crashed. Each is in the [release notes](release-notes.md#system-functions).

```kgpy
?> .sys("exit 3")
3
?> .pyf("math";"hypot")
1
?> .pyc("hypot";[3 4];:{})
5.0
?> .timer("t";-1;{1})
error: y must be a non-negative number
```

- `.sys` runs a shell command and returns its exit status. It tried to start a program named by the whole string and raised `FileNotFoundError`.
- `.pyc` passes positional arguments. `hypot(3;4)` gave 0.0.
- `.timer` checks its arguments. A negative or non-numeric interval is a `KlongTypeError`; the non-numeric one was a `ValueError`.
- `.webc(h)` stops a web server. It returned 0 and left the server running.
- A missing key in a `.kvs` store is `:undefined`. It raised an error.
- `.db(:{["T" T]})` works. It built a database whose queries failed.
- `.l` of a missing file is a `KlongSystemError`, which is still a `FileNotFoundError`.

## Keep the old reader

The reader (the parser) has a switch for the old reading of the text. It is deprecated and kept for one release so the two can be compared, so use it to get a program running and then fix the program.

```console
$ kgpy -e 'a::5;[a b]::[1 2];a'
1
$ KLONGPY_PARSER=legacy kgpy -e 'a::5;[a b]::[1 2];a'
5
```

From Python it is `KlongInterpreter(parser="legacy")`. It restores the old reading of the text: `[a b]::v` binds nothing, `f(a;b)::{...}` is a call, `"{"` is a syntax error, whitespace before the `;` of a conditional is an error, and left-over input is dropped without a warning. It does not bring back the old arity, projections, `:undefined`, division, operator or error behaviour, which are in the evaluator. Those have no switch.

## Quick list

| Change | Switch to keep the old behaviour |
|---|---|
| `[a b]::v` binds; `f(a;b)::{...}` defines | `KLONGPY_PARSER=legacy` |
| Strings such as `"{"`; whitespace before `;` in a conditional | `KLONGPY_PARSER=legacy` |
| Left-over input warns (level 0) or fails (strict) | Level 0 is the default; `warnings.filterwarnings("ignore", category=KlongSyntaxWarning)` hides the warning |
| Arity, projections, `d@k`, `:undefined`, `x%0`, conditional monads, symbols as data, operator results | None |
| Error classes and messages | None; `except KlongException` and the old Python classes still catch |
| Errors on stderr, exit statuses | None |
| IPC protocol | `KLONGPY_IPC=legacy` (or `auto` on the server), trusted networks only |
| Strict levels | `.strict(0)` or `.unsafe()`; the default is 0 |

The rest of v2 is in [What's New in v2](whats-new-v2.md).
