# What's New in v2

This page tours what version 2 adds, from CPU and GPU to FPGA and microcontroller, and then the stable core. Each section is short and links to the page that covers it. If you have code written for 0.7, read [Migrating to v2](migrating-to-v2.md) as well. If you are new, start with [Start Here](getting-started.md).

The status of each target (stable, passes conformance, subset, experimental) is in the table on the [home page](index.md#targets-and-status), and the hardware status is in the [native runtime status table](native/index.md#status-what-has-and-has-not-been-proven).

## CPU: the NumPy backend

For plain NumPy users, version 2 adds [strict modes](strict.md), a [linter](lint.md), [errors with positions](errors.md), [language additions](language-additions.md) and [IPC](ipc_capabilities.md) in the kdb+ protocol. An integer operand beyond 64 bits makes arithmetic exact, and indexing a vector by an integer vector is a whole-array operation, so `x@<x` sorts in about the time of the grade. [Migrating to v2](migrating-to-v2.md) lists what changes for 0.7 code.

## GPU: the torch backend

The torch backend and the `:>` gradient operator are in 0.7. What changes in 2:

* The conformance suite runs on the torch backend, and passes except the cases tagged as NumPy only (float32 text, and `:undefined` inside a list, which a tensor cannot hold).
* An integer operand beyond 64 bits makes `+ - * < > | &` exact on both backends; two operands within 64 bits wrap.

`kgpy --backend torch` runs the arrays on PyTorch (CPU, CUDA or Apple MPS). Floats are float32, and strings and mixed lists stay NumPy object arrays on the host. See [PyTorch Backend and Autograd](torch_backend.md), and [Performance](performance.md) for what is measured.

## FPGA: Klong functions as hardware

A function of the static profile can be built into a kernel for an FPGA card and called from a KlongPy session with ordinary arrays. The same source runs interpreted, as C and on the card. This is experimental: it has run on one Alveo U250. See [Klong functions as FPGA hardware](hardware.md).

## Microcontrollers: the native runtime and compiling to C

Two routes put Klong on a small device. The generated C has not yet run on a device.

**The native runtime** is a second implementation of the language in Rust that runs without Python. It has a serial console and an upload command, kdb+ IPC, timers and a management node library. It passes the conformance suite on a desktop and has run it on one ESP32-P4 board, and two Raspberry Pi Picos have run it with IPC between them. See [Native Runtime](native/index.md) and, for the details of what has run on hardware, its [status table](native/index.md#status-what-has-and-has-not-been-proven).

**Compiling to C** takes a subset of Klong, the static profile, in which every type, shape and call target is known before the program runs. It compiles to freestanding C99 with no heap, no recursion and no C library. `kgpy --check-static` tells you whether a file is inside the subset, and `kgpy --emit-c` writes the C.

<!-- check-docs: link examples -->
```console
$ kgpy --check-static examples/embedded/mavg.kg
examples/embedded/mavg.kg: inside the static profile (1 function, 1 entry, 0 externs, 2 globals)
```

See [Compiling Klong to C](embedded.md) and [The Static Profile](static-profile.md).

## One language on every target: the conformance suite

The `conformance/` directory holds about 1,300 test cases that do not depend on any implementation: a program, and the text the REPL prints or the kind of error it raises. KlongPy is the reference, and the native runtime runs the same files with no Python. See [Conformance Suite](conformance.md).

<!-- check-docs: link conformance -->
```console
$ python conformance/run.py | tail -n 1
1302 passed, 0 failed, 0 skipped by tags (KlongPy (in process))
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
?> swap::{[p q]::x;q,p}
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

**Dictionary literals and access.** `:{;k;v;k;v}` builds a dictionary from expressions. `d@k` looks a key up, `d@[k1 k2]` looks up several, and `d,d2` merges.

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
?> add::{x+y}
:dyad
?> inc::add(1)
:monad
?> inc'[10 20 30]
[11 21 31]
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

`kgpy --lint FILE...` reads programs without running them and reports constructs that parse but usually do not mean what they look like. An example is `my_var`, which reads as `my _ var` because `_` is an operator.

```console
$ cat > demo.kg <<'EOF'
> my_var::5
> total::0
> -1+2
> EOF
$ kgpy --lint demo.kg
demo.kg:1:1: W001 'my_var' reads as my _ var: _ is the Drop operator, names cannot contain underscores
demo.kg:3:1: W010 a leading - negates everything to its right: -1+... is -(1+...)
0 errors, 2 warnings in 1 file
```

See [Lint](lint.md).

### The REPL

The REPL reads a definition that spans lines, completes names with Tab, keeps a history between sessions (`~/.kgpy_history`) and stores the last printed value in `it`. `rlwrap` is not needed.

```kgpy
?> avg::{
.. (+/x)%#x
.. }
:monad
?> avg([2 4 9])
5.0
?> it*2
10.0
```

See [REPL](repl.md).

### kdb+ IPC

`.srv` and `.cli` speak the kdb+ IPC protocol, so a q process can connect to a KlongPy server and a KlongPy client can call a q process. The wire carries tables (as kdb+ tables), timestamps and timespans, and functions as their source text. It is tested against PeachQ, a kdb+-compatible implementation, not against kdb+ itself. A server can check credentials with `.srv.auth`, a call that gets no answer fails after 30 seconds (change it with `.clit`), and an error on the server comes back as a Klong error while the connection stays open. The pickle protocol of 0.7 is gone, so 0.7 and 2.0 cannot talk to each other. See [Migrating to v2](migrating-to-v2.md#ipc-speaks-the-kdb-protocol).

```kgpy
server?> .srv.auth::{(x~"bob")&(y~"secret")}
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

## More

[Roadmap](roadmap.md) lists what is experimental and what is planned. [Release notes](release-notes.md) has the complete list of changes.
