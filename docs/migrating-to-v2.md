# Migrating to v2

This page is for anyone with code written for KlongPy 0.7. It lists what behaves differently in v2, with the 0.7 behaviour, the v2 behaviour, why, and a switch to keep the old behaviour where one exists. [Release notes](release-notes.md) has the complete list of changes. Most programs need no change. The risk is in "Results that change without an error", where the same input gives a different result and nothing complains.

## Quick list

| Change | Switch to keep the old behaviour |
|---|---|
| IPC: 0.7 and 2.0 cannot talk to each other (the pickle protocol is removed) | None; [upgrade both ends](#ipc-speaks-the-kdb-protocol) |
| `[a b]::v` binds; `f(a;b)::{...}` defines | `KLONGPY_PARSER=legacy` |
| Strings such as `"{"`; whitespace before `;` in a conditional | `KLONGPY_PARSER=legacy` |
| Left-over input warns (level 0) or fails (strict) | Level 0 is the default; `warnings.filterwarnings("ignore", category=KlongSyntaxWarning)` hides the warning |
| A call on a number, string, list or dictionary is a `type` error | None |
| Arity, projections, `d@k`, `:undefined`, `x%0`, conditional monads, symbols as data, operator results | None |
| Error classes and messages | None; `except KlongException` and the old Python classes still catch |
| Errors on stderr, exit statuses | None |
| Strict levels | `.strict(0)` or `.unsafe()`; the default is 0 |
| A web handler that returns a dictionary sends a response; a failing handler answers 500, not 400 | None; [see below](#web-handlers) |
| `.rls()` strips the line end from each line | None; [see below](#other-changes) |
| The Polygon and Alpaca examples are removed | None; [a simulated feed replaces them](#other-changes) |

## Check your code first

Run these two on your programs before you upgrade, or right after.

`kgpy lint` reads files without running them and reports several classic traps, such as a leading minus that negates more than you meant:

```console
$ cat > old.kg <<'EOF'
> total::0
> -1+2
> EOF
$ kgpy lint old.kg
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

Before: `r::{x+y};s::r(1)` gave a value that could never be completed; `s(10)` returned it unchanged. Now `r(1)` is the same as `r(1;)`, a function that waits for the rest. It is allowed at every strict level.

```kgpy
?> r::{x+y}
:dyad
?> s::r(1)
:monad
?> s(10)
11
```

The same rule has these consequences:

* A function applied to too few values by `@` or an adverb gives a projection in the same way, so `{x+y}'[1 2 3]` is a list of three projections.
* A projection of a projection fills the first open slot: `f::{x,y,z};p::f(1;;);q::p(;3);q(2)` was `[1 None 3]` and is `[1 2 3]`.
* A projection made inside a function keeps the values its arguments had there.
* A function that is passed as a parameter still does not curry; see [Differences from Klong](differences.md#a-function-passed-as-a-parameter-does-not-curry).

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

These gave a different result in 0.7. Each is also in the [release notes](release-notes.md#operators).

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

- Amend with a negative index counts from the end; it changed nothing (`"abc":="x",-1` was `"abc"`). A real put into an integer list turns the list into reals; the value was truncated.
- Group lists the groups in the order the values first appear; `=[3 1 3 2 1]` listed the group of 1 first.
- Take on a list of lists takes rows; `5#[[1 2] [3 4]]` gave `[[1 2 1 2] [3 4 3 4]]`. `2#:foo` was `"fo"`.
- `[:+]~[:+]` was 0.
- `:{[1 2]},[1 2 3]` was silently ignored and is an error.
- `^` with a list operand follows one rule in 2.0: integers wrap in 64 bits for a non-negative integer exponent, and a negative exponent or any float gives floats. Two integer scalars stay exact. A program that raises a list to a power beyond 64 bits gets wrapped integers, so use a scalar base and exponent for an exact result. See [Numbers](differences.md#numbers).

### A recursive call through `.f` has its own locals

Before: `.f` called itself without declaring the function's locals, so the inner call changed the outer call's variable. `f::{[t];t::x;:[x<2;t;t+.f(x-1)]};f(3)` was 3, where the same function written with its own name gives 6.

```kgpy
?> f::{[t];t::x;:[x<2;t;t+.f(x-1)]}
:monad
?> f(3)
6
```

### A Python function brought in with `.py` gets every argument

Before: a function with optional parameters was called with its required ones only, so `linspace(0;1;5)` made 50 points and `arange(2;8)` was `[0 1]`. Now it gets every argument.

```kgpy
?> .pyf("numpy";"linspace")
1
?> #linspace(0;1;5)
5
```

`.pyc` passes positional arguments in the same way: `hypot(3;4)` gave 0.0.

```kgpy
?> .pyf("math";"hypot")
1
?> .pyc("hypot";[3 4];:{})
5.0
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

### Calling a value that is not a function

Before: `a::1.0;a([3 4])` returned 1.0. The arguments were dropped and the value of `a` came back, so a definition written without braces, `avg::(+/x)%#x`, failed silently: `avg(list)` printed the number that `avg` held.

Now: the call is a `type` error at every strict level, and the message says what was called and what it is.

```kgpy
?> a::1.0
1.0
?> a([3 4])
error: cannot call a: it is a float, not a function
?> a::[1 2 3]
[1 2 3]
?> a(1)
error: cannot call a: it is a list of 3, not a function
?> a@1
2
```

Fix: search your code for a call on a name that holds a value.

* Read a list or a dictionary with `a@i` or `d@k`.
* A call on a function, a projection, a Python function, an IPC connection or a database is unchanged.
* One idiom relied on the old behaviour: a function literal passed as an argument and then called. `time({work()})` evaluates the literal where it stands, so the callee `x()` called its value. Give the function a name and pass the name: `w::{work()};time(w)`.

There is no switch.

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

Every error KlongPy raises on purpose is a `KlongError` with a kind, and almost all have a position and the source line. A transcript in this page often shows a REPL error as `error: MESSAGE`, the message alone; [Reading an error](errors.md#reading-an-error) describes the full form. An exception from NumPy inside an operator is reported as one too, naming the operator. A Klong error that replaces a Python one is also an instance of the old class, so `except IndexError`, `except ValueError`, `except TypeError` and `except FileNotFoundError` still catch it, and `except KlongException` still catches every one. The classes are exported from `klongpy` (`KlongTypeError`, `KlongRangeError`, `KlongNameError` and so on).

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

Before: `.srv` and `.cli` sent pickled Python objects, and a peer could run code in your process when its message was unpickled. A 0.7 server could only be reached by a 0.7 client.

Now: the kdb+ IPC protocol is the only protocol. A q process can talk to a `.srv`, and nothing is unpickled from the network. A 0.7 process and a 2.0 process cannot exchange IPC messages in either direction:

- A 0.7 client connecting to a 2.0 server is closed at once, and the server logs that the client speaks the pickle protocol of 0.7. The client sees `connection lost`.
- A 2.0 client connecting to a 0.7 server fails at the handshake after 5 seconds with `handshake timed out`.
- The environment variable `KLONGPY_IPC` no longer exists and is ignored, whatever its value.

What to do: upgrade every KlongPy process that talks IPC to 2.0 or later, then remove `KLONGPY_IPC` from environments if you set it. A Python object with no kdb+ form (a set, a complex number, `None`) used to travel by pickle and is now a type error on the sending side; send a list, a string or a dictionary instead. Python `bytes` and `bytearray` cross as a kdb+ byte vector and arrive as an integer array. See [the protocol](ipc_capabilities.md#protocol-and-security) and [0.7 peers](ipc_capabilities.md#07-peers).

Other changes that come with the new protocol:

- An integer beyond 64 bits (`2^64`, a long literal) cannot cross the wire. Sending one is `range error: integer does not fit 64 bits`; 0.7's pickle carried it. Send it as a string or a float.
- An error on the server comes back to the client as a Klong error and the connection stays open. It used to drop the connection (`connection lost`), and the next call failed with an `AttributeError`.
- `.srv` raises when it cannot listen. A second `.srv` on a port in use returned 0 and did nothing; it now raises `cannot listen on 0.0.0.0:8888: ...`. `.srv` on a server that is already running still returns 0.
- A call waits 30 seconds and then raises an error that contains `timed out`, closes the connection and fails every call waiting on it. Change the limit with `.clit`.
- A `klongpy.db` table is sent as a kdb+ table. A function is sent as its source text, a function with named parameters included. A projection has none and is sent as an opaque value. See [How functions cross the wire](ipc_capabilities.md#how-functions-cross-the-wire).
- A numpy `datetime64` or `timedelta64` can be sent, as a kdb+ timestamp or timespan in nanoseconds. It was a type error.
- A server accepts every client unless the program defines `.srv.auth`. See [Credentials](ipc_capabilities.md#credentials-srvauth).

There is still no encryption. See the security model in the [README](https://github.com/briangu/klongpy#security-model-trusted-networks).

## Web handlers

Before: a handler that returned a dictionary sent Python's text for it as the body, and a handler that failed answered 400 with the body `Invalid request` and no log line. Now:

- A dictionary is a response: `"status"` (default 200), `"type"` (default `text/plain`) and `"body"` (default empty). Any other key is a type error, so a handler that returned a dictionary of data to be shown as text must convert it first, with `$d` or `.json(d)`.
- A handler that fails answers 500 with the body `handler failed` and logs one line with the route and the error. A client or a test that expected 400 needs 500.
- A `HEAD` request to a GET route is answered as the GET, without a body.

See [the web server](web_server.md#the-response).

## Other changes

- **`.rls()` strips the line end.** It returned each line with its `\n`, and a program that cut it off with `-1_` now loses a real character. Remove that code.
- **The Polygon and Alpaca examples are removed.** `examples/stocks/feed` is a simulated price feed that needs no account: a seeded random walk, a server with per-symbol subscriptions, and a consumer that computes a VWAP and a moving average.

[Release notes](release-notes.md#system-functions) lists the fixes to system functions (`.sys`, `.timer`, `.webc`, `.kvs`, `.db`), the operators that used to crash, and the libraries that now work.

## Keep the old reader

The reader (the parser) has a switch for the old reading of the text. It is deprecated and kept for one release so the two can be compared, so use it to get a program running and then fix the program.

```console
$ kgpy -e 'a::5;[a b]::[1 2];a'
1
$ KLONGPY_PARSER=legacy kgpy -e 'a::5;[a b]::[1 2];a'
5
```

From Python it is `KlongInterpreter(parser="legacy")`. It restores the old reading of the text: `[a b]::v` binds nothing, `f(a;b)::{...}` is a call, `"{"` is a syntax error, whitespace before the `;` of a conditional is an error, and left-over input is dropped without a warning. It does not bring back the old arity, projections, `:undefined`, division, operator or error behaviour, which are in the evaluator. Those have no switch.

The rest of v2 is in [What's New in v2](whats-new-v2.md).
