# What's New in v2

Version 2 adds a standalone command, shared desktop tables and SQL, and new language forms. [Start Here](getting-started.md) · [Migrating from the pure-Python line](migrating-to-v2.md)

## One engine, two commands

KlongPy 2 evaluates Klong on one interpreter, written in Rust. The Python implementation of the language that 0.7 had, with its parser, evaluator, linter and checker, is gone. The same interpreter is in both commands and in the firmware for microcontroller boards, so a program means the same thing in all three.

| Command | What it is |
|---|---|
| `kg` | The standalone interpreter with embedded DuckDB, column tables, and native Metal on supported macOS builds. |
| `kgpy` | The same interpreter hosted in Python, with NumPy and PyTorch arrays, Python callables, pandas and Arrow interchange, web services and FPGA card bindings. |

A v2 wheel installs both. The Rust tools `lint`, `check`, `compile c`, `fpga source` and `ast` run from either command. `kg --capabilities` reports which features the installed binary has. [What Runs Where](what-runs-where.md)

## Tables and time-series queries

`.table` creates ordered columns and `.index` adds a key for upserts. A later SQL query sees changes made through any reference to the same table. Use `.db("trades.duckdb")` for durable storage or pass named Klong tables to `.db` to query them in memory. Both hosts support the same Klong database code, and the Python host also exchanges tables with pandas and Arrow. [Columnar tables and SQL](fast_columnar_database.md)

The status of each target (stable, passes conformance, subset, experimental) is in the table on the [home page](index.md#targets-and-status), and the hardware status is in the [status table of the board firmware](native/index.md#status-what-has-and-has-not-been-proven).

## CPU: the NumPy backend

For plain NumPy users, version 2 adds [strict modes](strict.md), a [linter](lint.md), [errors with positions](errors.md), [language additions](language-additions.md) and [IPC](ipc_capabilities.md) in the kdb+ protocol. An integer operand beyond 64 bits makes arithmetic exact, and indexing a vector by an integer vector is a whole-array operation, so `x@<x` sorts in about the time of the grade. [Migrating to v2](migrating-to-v2.md) lists what changes for 0.7 code.

## GPU: the torch backend

The torch backend and the `:>` gradient operator are in 0.7. What changes in 2:

* The conformance suite runs on the torch backend, and passes except the cases tagged as NumPy only (float32 text, and `:undefined` inside a list, which a tensor cannot hold).
* An integer operand beyond 64 bits makes `+ - * < > | &` exact on both backends; two operands within 64 bits wrap.

`kg --device mps` selects native Metal for supported array arithmetic and gradients on macOS. `kgpy --backend torch` uses PyTorch. Automated tests cover its CPU backend and an Apple MPS release smoke workload. Default float calculations use float32 under PyTorch, and strings and mixed lists stay NumPy object arrays on the host. [Native Metal](what-runs-where.md#native-metal) · [PyTorch Backend and Autograd](torch_backend.md)

## FPGA: Klong functions as hardware

A function of the static profile can be built into a kernel for an FPGA card and called from a KlongPy session with ordinary arrays. The same source runs interpreted, as C and on the card. This is experimental: it has run on one Alveo U250. See [Klong functions as FPGA hardware](hardware.md).

## Microcontrollers: the interpreter and compiling to C

Two routes put Klong on a small device.

**The interpreter on a board** shares the desktop engine and can provide a serial console, uploads, kdb+ IPC, timers and the management library. The RC1 serial simulator passed 1,716 selected language cases. On two physical RP2040 Picos, RC1 passed the paired boot self-test, including a 4 KB echo and a management call. The ESP32-P4 has earlier physical console results. [Board test status](native/index.md#status-what-has-and-has-not-been-proven)

**Compiling to C** takes a subset of Klong, the static profile, in which every type, shape and call target is known before the program runs. It compiles to freestanding C99 with no heap, no recursion and no C library. `check` tells you whether a file is inside the subset, and `compile c` writes the C; both are commands of `kg` and of `kgpy`. Two RP2040 Picos ran the moving average and PID controller both compiled and interpreted. The compiled FIR was checked against a desktop reference stored in flash. [Measurements on a Pico](embedded.md#on-a-pico)

<!-- check-docs: link examples -->
```console
$ kgpy check examples/embedded/mavg.kg
examples/embedded/mavg.kg: inside the static profile (1 function, 1 entry, 0 externs, 2 globals)
```

See [Compiling Klong to C](embedded.md) and [The Static Profile](static-profile.md).

## One language on every target: the conformance suite

The `conformance/` directory holds test cases that do not depend on any implementation: a program, and the text the REPL prints or the kind of error it raises. The same files run through `kgpy`, through `kg` and on the board firmware. See [Conformance Suite](conformance.md).

<!-- check-docs: link conformance -->
```console
$ python conformance/run.py | tail -n 1 | grep -o '[0-9]* failed'
0 failed
```

## The stable core

These are in the NumPy and torch backends, on every platform that runs Python.

### A friendlier language

Classic Klong programs run as before. v2 adds forms for things that needed workarounds, and gives a meaning to input that did nothing. The full description is in [Language Additions](language-additions.md).

**Destructuring.** A list of names on the left of `::` binds each name to the member of the right side in the same position. At the start of a function it also declares the names as locals.

```kgpy
?> [a b c]::[10 20 30]
[10 20 30]
?> b
20
?> swap(pair)::{[p q]::pair;q,p}
:monad
?> swap([1 2])
[2 1]
```

**Named parameters.** `f(a;b)::{...}` defines a function whose arguments have names you choose. It is not limited to three arguments.

```kgpy
?> area(w;h)::{w*h}
:dyad
?> area(3;4)
12
```

**Dictionary literals and access.** `:{;k;v;k;v}` builds a dictionary from expressions. `d@k` looks a key up, `d@[k1 k2]` looks up several, `d.k` is `d@"k"`, and `d,d2` merges.

```kgpy
?> n::7
7
?> d:::{;"n";n;"sq";n*n}
:{["n" 7] ["sq" 49]}
?> d@"sq"
49
?> d@["n" "sq"]
[7 49]
```

**Projections for short calls.** A call with fewer arguments than the function takes gives a function that waits for the rest.

```kgpy
?> add(a;b)::{a+b}
:dyad
?> inc::add(1)
:monad
?> inc'[10 20 30]
[11 21 31]
```

**Modules.** A file that assigns the variable `export` is a module. `.l` runs it in a scope of its own and binds what it exported, and `a.b` reads or writes the entry `"b"` of any dictionary. See [Modules](language-additions.md#modules).

```console
$ printf 'sq(v)::{v*v}\nexport:::{;"sq";sq}\n' > geom.kg
$ kgpy -e '.l("geom");geom.sq(4)'
16
```

**Text functions.** `.split`, `.join`, `.words`, `.trim` and `.num` do what every program otherwise writes for itself, and `.pf` and `.fmt` fill a template. `.num` gives `:undefined` for text that is not a number.
`.json` and `.unjson` convert to and from JSON text.

```kgpy
?> .words("  to be  or not ")
["to" "be" "or" "not"]
?> .join(.split("a,b,c";",");"/")
"a/b/c"
?> .num'.split("3,4.5";",")
[3.0 4.5]
?> .pf("{} items";3)
3 items
"3 items"
```

**`:undefined`.** A missing dictionary key, a division by zero and a number that cannot be read from text are all the one value `:undefined`, and `:_` tests for it. It is the same value everywhere, so you can compare against it.

```kgpy
?> 1%0
:undefined
?> d:::{["a" 1]}
:{["a" 1]}
?> d@"b"
:undefined
?> :_d@"b"
1
?> 1:$"x"
:undefined
```

**Arity from the highest argument.** A function has as many arguments as the highest of `x`, `y` and `z` that its body uses, wherever it uses them. `{-x}` is a monad and `{x+z}` is a triad.

```kgpy
?> {-x}
:monad
?> {x+z}
:triad
```

### Readable errors with source positions

Almost every error KlongPy raises on purpose has a position, a kind and the source line with a caret. The exception is the range error for an expression nested too deeply, which has no position. The kinds are `syntax`, `name`, `type`, `range`, `shape`, `arity`, `system`, `python` and a few others. From Python each is a class under `KlongError` (`KlongTypeError`, `KlongRangeError`, and so on), and an old `except IndexError` or `except ValueError` still catches the ones that replace those. A file run stops at the first error with exit status 1 and prints a traceback only with `-v` or `--traceback`.

<!-- check-docs: render -->
```kgpy
?> [1 2 3]@3
1:8: range error: index 3 out of range for list of 3
  [1 2 3]@3
         ^
```

See [Errors](errors.md).

### Strict modes

Klong accepts input that is almost certainly a mistake: `1 2 3` evaluates to `1` and drops the rest, and an undefined name is its own symbol. At the default level 0 these still run, and the common ones print a warning. Level 1 makes them syntax errors, and level 2 also makes reading an undefined name an error and rejects a call with too many arguments. Choose the level with `kgpy --strict=N`, `KlongInterpreter(strict=N)` or `.strict(N)`.

```console
$ kgpy -e 'a'
:a
$ kgpy --strict=2 -e 'a'
<expr>:1:1: name error: undefined: a
  a
  ^
```

See [Strict Mode](strict.md).

### The linter

`kgpy lint FILE...` reads programs without running them and reports constructs that parse but usually do not mean what they look like. An example is `my_var`, which reads as `my _ var` because `_` is an operator.

```console
$ cat > demo.kg <<'EOF'
> my_var::5
> total::0
> -1+2
> EOF
$ kgpy lint demo.kg
demo.kg:1:1: W001 'my_var' reads as my _ var: _ is the Drop operator, names cannot contain underscores
demo.kg:3:1: W010 a leading - negates everything to its right: -1+... is -(1+...)
0 errors, 2 warnings in 1 file
```

See [Lint](lint.md).

### The REPL

The REPL reads a definition that spans lines, completes names with Tab, keeps a history between sessions (`~/.kgpy_history`) and stores the last printed value in `it`. `rlwrap` is not needed.

```kgpy
?> avg(v)::{
.. (+/v)%#v
.. }
:monad
?> avg([2 4 9])
5.0
?> it*2
10.0
```

See [REPL](repl.md).

### kdb+ IPC

`.srv` and `.cli` speak the kdb+ IPC protocol. Both desktop commands tested keyed and unkeyed table round trips with PeachQ. The wire also carries timestamps, timespans and function source text. Upgrade every peer on a pure-Python KlongPy network together because v2 uses a different protocol. [IPC Capabilities](ipc_capabilities.md) · [Migrating to v2](migrating-to-v2.md#ipc-speaks-the-kdb-protocol)

```kgpy
server?> .srv.auth(user;pw)::{(user~"bob")&(pw~"secret")}
:dyad
server?> .srv(8888)
1
client?> f::.cli("localhost:8888:bob:secret")
remote[localhost:8888]:fn
client?> f("1+1")
2
client?> .cli("localhost:8888:bob:wrong")
error: remote[localhost:8888] closed the connection during the handshake (bad credentials, or it does not speak the kdb+ protocol)
```

See [IPC Capabilities](ipc_capabilities.md) and [Klong for q Users](from-kdb.md#talking-to-q-ipc).

### Web handlers

A [web handler](web_server.md) runs on the same loop as timers and IPC callbacks, and may return a dictionary with a status, a content type and a body. A handler that fails answers 500 and logs one line. `examples/stocks/feed` is a simulated price feed over IPC that needs no account, and a file that serves or holds a connection stays up until it is stopped ([the command line](repl.md#options)).

## More

[Roadmap](roadmap.md) lists what is experimental and what is planned. [Release notes](release-notes.md) has the complete list of changes.
