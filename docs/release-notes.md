# Release Notes

These notes describe the changes in version 2 from the pure-Python 0.7 and v1 line. The older examples use the 0.7 behavior where stated. [What's New in v2](whats-new-v2.md) · [Migrating to v2](migrating-to-v2.md)

## 2.0

### Installing 2.0.0rc1

Install the candidate with `python -m pip install klongpy==2.0.0rc1`. It supports CPython 3.11–3.14. Linux x86-64 and ARM64 wheels require glibc 2.28 or later. Apple Silicon wheels target macOS 11 or later and Intel wheels target macOS 10.15 or later. The Windows wheel is deferred while the compiler and simulator tests are qualified on Windows. [Source distribution and available wheels](https://pypi.org/project/klongpy/2.0.0rc1/#files)

The 25,600-tick/s benchmark missed its 100 ms scheduled latency target on the four-CPU GitHub Intel Mac runner. In one run, `kg` recorded a 323 ms p99 latest-price query and `kgpy` a 140 ms p99 as-of query. Both numbers include the time the client waited before sending. [Intel qualification run](https://github.com/briangu/klongpy/actions/runs/44982548782)

### Headline additions

* [One interpreter, written in Rust](what-runs-where.md), behind two commands: `kgpy` (hosted in Python) and `kg` (no Python). The same interpreter runs on [microcontroller boards](native/index.md).
* [Column tables and embedded DuckDB](fast_columnar_database.md) in both desktop commands. Tables support compound keys and keyed updates, and SQL can query live Klong tables or a persistent DuckDB file.
* [Native Metal on macOS](what-runs-where.md#native-metal) for supported array operations and reverse-mode gradients. The Python host also supports device arrays and autograd through its PyTorch backend.
* [Modules](language-additions.md#modules): the `export` variable, `.l` as the one loader, and qualified names `a.b`.
* Errors as values: `.try` catches an error and `.err` raises one. `.seed` makes `.rn` repeatable and `.now` gives the time.
* The [conformance suite](conformance.md): more than 2,000 cases that every build of the interpreter must pass.
* The [static profile](static-profile.md) and [compiling Klong to C](embedded.md).
* [FPGA kernels](hardware.md), experimental.
* [Strict modes](strict.md), levels 0, 1 and 2.
* The [linter](lint.md), `kgpy lint`.
* [Structured errors](errors.md) with a position, a kind and the source line.
* [Language additions](language-additions.md): destructuring, named parameters, dictionary literals, calls with too few arguments, `.fmt`, `.pf` and `.grad`.
* kdb+-protocol [IPC](ipc_capabilities.md), tested against PeachQ.
* A [REPL](repl.md) with multi-line entry, history, completion and `it`.


#### One interpreter

**One Rust interpreter.** The parser, evaluator, linter, static checker and compilers run in both desktop commands. `kgpy` adds NumPy and PyTorch arrays, Python callables, pandas and Arrow interchange, web services and FPGA card bindings. [What Runs Where](what-runs-where.md)

**The standalone `kg` command.** A v2 wheel installs it next to `kgpy`. It runs files, expressions and a REPL. `kg --capabilities` emits one JSON object for scripts and tools. A function that needs Python reports that it needs `kgpy`.

```text
$ kg -e '.py("math")'
1:1: name error: .py needs kgpy (the Python-hosted command)
  .py("math")
  ^^^^^^^^^^^
```

**The tools are in the interpreter.** `lint`, `check`, `compile c`, `fpga source` and `ast` are the same Rust code under `kg` and `kgpy`, and they print the same text.

**There is no pure-Python version of 2.** `import klongpy` needs its compiled extension, and the error says how to get it: a wheel carries it, and a source tree builds it with `pip install -e .`, which needs a Rust toolchain. The Python modules of the old implementation (`klongpy.parser`, `klongpy.lexer`, `klongpy.compiler`, `klongpy.monads`, `klongpy.dyads` and `klongpy.adverbs`) do not exist. The Python API (`KlongInterpreter`, the error classes, the backends) is kept. Version 2 needs Python 3.11 to 3.14, where 0.7 accepted 3.9 and later; the wheel is one `abi3` file per platform.

#### Modules

**New: modules, `.l` as the one loader and qualified names.** A file that assigns the variable `export`, a dictionary written as a literal, is a module. `.l("geom")` runs it in a scope of its own, once, and binds the read-only dictionary it exported to the file's stem; `.l("geom";:g)` binds it to `g`, and `.l("geom";["sq"])` binds the entries named as plain names. `.l` returns the module. `a.b` is `a@"b"` for reading, writing and calling on any dictionary. A bare name is searched in the directory of the file that contains the `.l`, then in `KLONGPATH`, then in the bundled library, and a local file that shadows a library file gives a warning. See [Modules](language-additions.md#modules); `examples/modules` is a program of twelve files.

```console
$ printf 'sq(v)::{v*v}\nexport:::{;"sq";sq}\n' > geom.kg
$ kgpy -e '.l("geom");geom.sq(4)'
16
```

#### Static Klong and C

**New: a subset of Klong that compiles to C.** [The static profile](static-profile.md) is the part of the language in which every type, shape and call target is known before the program runs. `check FILE...` checks files against it without running them, and `compile c FILE... -o DIR` compiles a program inside it to freestanding C99 with no heap, no recursion and no C library. `adc::.extern("read_adc";"int -> float")` returns a function the target provides, kept in a module file and used as `read.adc(3)`, and `.entry("name";"float -> float")` declares one the host calls. Under KlongPy an extern is a Python function (`klong.externs["read_adc"] = f`, keyed by C symbol), and `klongpy.static.load_program` and `call_entry` run a program the way the compiled one is called. A program that uses neither is unaffected. See [Compiling Klong to C](embedded.md).

**`.entry` and `.extern` have the same checks under `kg` and `kgpy`.** A program of the static profile loads unchanged, and an extern resolves to a host function registered under its C symbol. Fixed-point programs (`.fixed`) run under `kgpy`; under `kg` `.fixed` is a system error.

**Compiled C on two RP2040 Picos.** On both boards, the compiled moving average and PID controller matched the on-device interpreter across 64 and 32 outputs respectively, within the checks’ numeric tolerance. The interpreted FIR workload exceeded the Pico heap. The compiled FIR matched 241 reference values stored in flash. [On a Pico](embedded.md#on-a-pico)

```kgpy
?> adc::.extern("read_adc";"int -> float")
:nilad
?> .entry("f";"-> int")
error: .entry: the function f is not defined
```

**Experimental: `.fpga` calls a function of the static profile as FPGA hardware.** `fir::.fpga("fir.kg")` binds the kernels of an image built with `kgpy fpga build fir.kg` for an Alveo U250 and `fir.run(x)` runs on the card, with the arguments checked against the `.entry` signature, an extra leading dimension meaning several calls, resident buffers (`.fpga.put`, `.fpga.get`, `.fpga.free`), `.fpga.reset` and `.fpga.info`. `KLONG_FPGA=sim` runs the same generated code on the host as a simulator, with no card. Nothing changes for a program that does not use `.fpga`, and nothing of it is imported at start. This is experimental: it has been run on one Alveo U250 with two images. The strategy kernel (a streaming kernel) is about ten times faster than the same program as C on a 32-thread CPU, and `fir` (a wrapped kernel) is slower than numpy. See [Klong functions as FPGA hardware](hardware.md).


#### Strict modes

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


#### Language additions

See [Language additions](language-additions.md) for the forms. These are the changes in behaviour that come with them, apart from the ones in [Syntax that changed meaning](#syntax-that-changed-meaning).

**New: `.try` and `.err`, errors as values.** `.try(f;args)` applies a function to a list of arguments and gives `[1 result]`, or `[0 e]` when it raises an error. `e` is a dictionary with `kind`, `message`, `source`, `line` and `col`. `.err(kind;message)` raises an error of that kind, and `.err(e)` raises again what `.try` caught. They are the same on every target. A function literal with no parameters, `{work()}`, is a function where `.try`, `.async` and `.timer` want one, and inside the evaluated forms `[;...]` and `:{;...}`; a second argument of `.try` that is not a list is a type error. See [`.try`](system-functions.md#try).

```kgpy
?> .try({x+1};[1])
[1 2]
?> ((.try({x+"a"};[1]))@1)@"kind"
"type"
```

**New: `.seed` and `.now`.** `.seed(n)` seeds the random number generator, and the draws of `.rn` that follow are the same on every machine and target. In 0.7 `.rn` drew from NumPy's generator, so `np.random.seed` made it repeat; NumPy's seed no longer reaches `.rn`, and `.seed(42);.rn()` is 0.1941059175341826. `.now()` is the time in seconds since 1970.

**A function with named parameters is lexically scoped.** `f(a;b)::{...}` resolves a free name in its own frame and then in the globals of its file or module, never in the locals of its caller. A classic function (`{x+y}`, or one with a `[t]` locals list) keeps dynamic scope, and so does a function it calls. A function that read its caller's local through a named-parameter form must be given the value as an argument.

```kgpy
?> t::9
9
?> g(a)::{t+a}
:monad
?> f::{[t];t::5;g(1)}
:nilad
?> f()
10
```

**New: `.fmt`, `.pf` and `.grad`.** `.fmt(template;args)` and `.pf(template;args)` format text with `{}` placeholders, and `.grad(f;x)` is the ASCII name of `x∇f`. A template with no placeholders needs no arguments, as in `.pf("done")`. A width with a leading zero, such as `{:05}`, is a range error, because a width pads with spaces. See [Language Additions](language-additions.md#formatted-text).

**New: `.split`, `.join`, `.words`, `.trim` and `.num`.** Five system functions for text, the same under `kg` and `kgpy`: `.split(s;sep)` and its inverse `.join(list;sep)`, `.words(s)`, `.trim(s)`, and `.num(s)`, which reads a number from the whole string and gives `:undefined` for anything else, where `.rs` stops after the first object. See [Language Additions](language-additions.md#strings).

**New: `.json` and `.unjson`.** `.json(x)` is the JSON text of a value and `.unjson(s)` the value of JSON text; neither needs a package. `null` reads as `:undefined`, and text that is not exactly one JSON value is a range error with its position. See [JSON](language-additions.md#json).

```kgpy
?> .json(:{;"a";[1 2]})
"{""a"":[1,2]}"
?> .unjson("{""a"":[1,2],""b"":null}")
:{["a" [1 2]] ["b" :undefined]}
```

**The gradient operators accept a projection that leaves one argument open.** With `f(a;b)::{a*b}`, `3∇f(2;)` is the derivative of `a*2` at 3.

**`.table` takes a dictionary of columns, and `#` of a store is its number of entries.**

<!-- check-docs: requires duckdb pandas -->
```kgpy
?> .py("klongpy.db")
1
?> .table(:{;"a";[1 2];"b";[3 4]})
a b
1 3
2 4
```

**`.l` looks next to the file that calls it first.** `.l("common.kg")` looks in the directory of the file that contains the `.l`, then in the directories of `KLONGPATH`, then in the bundled library, so `kgpy path/to/prog.kg` can load its neighbours from any current directory. See [`.l`](system-functions.md#l).

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

**A Python function given to Klong takes any parameter names and any number of parameters.** Before, a function assigned to a name, held in a dictionary or passed as a callback took at most three parameters, named `x`, `y` and `z`, and any other name failed with a Python `TypeError`. Now `klong['f'] = lambda a, b, c, d: ...` makes `f(1;2;3;4)` call it with the arguments by position. A function whose parameters are all among `x`, `y` and `z` is called as before. Optional parameters, `*args` and callables without a signature are accepted, and a call with the wrong number of arguments is a `KlongArityError`, where an extra argument to an `x`, `y`, `z` function used to be dropped. See [Python Integration](python_integration.md#the-parameter-rule).



#### Structured errors

Transcripts on this page and on [Migrating to v2](migrating-to-v2.md) often show a REPL error as `error: MESSAGE`, the message alone. [Reading an error](errors.md#reading-an-error) describes the full form with position and source line.

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


#### REPL and command line

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

**Persistent history and tab completion are built in.** Line editing existed in 0.7; history (`~/.kgpy_history`, or `KGPY_HISTORY`) and tab completion now come from `readline`, so `rlwrap` is not needed. See [the REPL](repl.md).

**Both commands report their version.** Run `kg --version` or `kgpy --version`.

**A file run stays up while it serves or is connected.** `kgpy file.kg` ends after its last line unless an IPC server (`.srv`) is listening, a connection opened with `.cli` is open, a web server (`.web`) is listening or a WebSocket connection (`.ws`) is open. A timer alone does not keep it up. `.x` in a callback (a timer, a request or a hook) exits the process with its status. See [the command line](repl.md#options).

**Tooling has subcommands.** `kgpy lint`, `kgpy check`, `kgpy compile c`, `kgpy fpga source`, `kgpy fpga build`, `kgpy fpga status` and `kgpy ast` check, compile and build programs without running them; running a file, `-e`, `-l`, `-s` and `-t` are as in 0.7. See [the command line](repl.md#command-line).


#### kdb+ IPC

**The wire protocol is kdb+ IPC.** `.srv` and `.cli` used pickled Python objects, which let a peer run code in your process when a message was unpickled. It is now the kdb+ protocol, and a q client can talk to a `.srv`. The pickle protocol was removed: a 0.7 and a 2.0 process cannot exchange IPC messages, so upgrade both ends. See [the protocol](ipc_capabilities.md#protocol-and-security).

**Tables and functions cross the connection.** Both desktop hosts send and receive keyed and unkeyed tables. Echo tests against PeachQ cover typed empty columns, symbols, timestamps and nulls. A function literal travels as source, preserving its parameter names and arity between Klong peers. A projection travels as an opaque value. [Types across the wire](ipc_capabilities.md#types-across-the-wire) · [How functions cross the wire](ipc_capabilities.md#how-functions-cross-the-wire)

**A numpy `datetime64` or `timedelta64` can be sent.** It goes as a kdb+ timestamp or timespan, in nanoseconds (a date is nanoseconds since 2000-01-01), and `NaT` is the null. It used to be a type error on the sending side. It arrives as the integer, as every kdb+ temporal does, so the receiver of `2000-01-02` sees 86400000000000. A numpy unit is converted to nanoseconds, a calendar unit (years, months) of a timedelta is refused, and a value outside the range of a kdb+ temporal is a range error.

```kgpy
server?> .py("numpy")
1
server?> .pyf("numpy";"datetime64")
1
server?> .srv(8888)
1
client?> f::.cli(8888)
remote[localhost:8888]:fn
client?> f("datetime64(""2000-01-02"")")
86400000000000
```


### Behaviour changes

These changes can alter what an existing program does or prints. The old behaviour is that of 0.7.


#### Syntax that changed meaning

These inputs were accepted by 0.7 and ran without an error, and now mean something else. [`kgpy lint`](lint.md) and [strict mode](strict.md) point at some of them.

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
:fn/4
?> g(1;2;3;4)
10
```

A function of four or more parameters displays as `:fn/N` (here `:fn/4`), the form IPC uses; zero to three keep `:nilad`, `:monad`, `:dyad` and `:triad`.

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

**`{y}(1)` is a projection, not `:y`.** `{y}` has two arguments now. A call with one of them used to bind it to `x` and return the unbound `y` as the symbol `:y`. It now gives a projection that waits for the second argument, the same value as `{y}(1;)` with the empty slot, and `p(5)` completes it. A call with every argument, `{y}(1;2)`, is 2.

```kgpy
?> {y}(1;2)
2
?> p::{y}(1)
:monad
?> p(5)
5
?> q::{y}(1;)
:monad
?> q(5)
5
```

**A call with too few arguments is a projection.** `r::{x+y};s::r(1)` used to give a value that looked like a function and could never be completed: `s(10)` returned it unchanged. A call with fewer arguments than the function takes is now a projection over the missing trailing arguments, `r(1)` being the same as `r(1;)`, for functions of `x`, `y` and `z` and for functions with named parameters. It is allowed at every strict level; too many arguments is still an arity error at level 2. A function applied to too few values by `@` or an adverb gives a projection in the same way, so `{x+y}'[1 2 3]` is a list of three projections. Three things follow from one rule for all of them:

- A projection is shown by the number of arguments it still takes. `{x+y}(1;)` displayed as `:dyad` and is `:monad`.
- A written call that leaves a slot empty and has fewer arguments than the function takes leaves the trailing arguments open as well. For a function `g` of three arguments, `h::g(1;)` waits for two arguments, where `h(2)` used to run `g` without its third.
- A projection made inside a function holds the values its arguments had there. `mk::{[a];a::x*2;r(a;)};p::mk(5);p(10)` is 20; it used to look `a` up when `p` was called, long after `mk` had returned, and fail. At top level a projection still evaluates its argument expressions each time it is completed, as before: `a::1;p::r(a;);a::5;p(10)` is 15.

```kgpy
?> r::{x+y}
:dyad
?> s::r(1)
:monad
?> s(10)
11
?> g::{x,y,z}
:triad
?> h::g(1)
:dyad
?> h(2;3)
[1 2 3]
?> {x(10)}'{r(x)}'[1 2 3]
[11 12 13]
```

**A projection of a projection fills the first open slot.** `f::{x,y,z};p::f(1;;);q::p(;3);q(2)` ran `f` without its middle argument, because each call went on from the slot where the one before it stopped. It is `[1 2 3]`. A projection called with too few arguments, `p(2;)` or `p(2)`, is a projection again; it used to run the function with the open slot unbound when written as a statement.

```kgpy
?> f::{x,y,z}
:triad
?> p::f(1;;)
:dyad
?> q::p(;3)
:monad
?> q(2)
[1 2 3]
```

**`:undefined` written in a program is the undefined value.** It used to be the symbol named `undefined`, which prints the same and is a different value, so `u::1%0;u~:undefined` was 0. It is 1, and `:_:undefined` is 1.

```kgpy
?> u::1%0
:undefined
?> u~:undefined
1
```

**`x`, `y` and `z` can be assigned at strict levels 1 and 2.** `h::{x::7};h(1)` was `undefined variable: x` at level 1 and 2. The arguments are locals of the call, as named parameters are, and the caller's value is not changed.


#### Parser

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


#### Operators

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

**Each on a symbol passes the symbol.** `f'x` with a symbol `x` is `f(x)`. The symbol was taken for the string of its name and iterated over its characters.

```kgpy
?> s:::foo
:foo
?> {x}'s
:foo
```

**A monad applied to a conditional or an evaluated list gets the value.** `-:[0;2;3]` applied the monad to the condition and was 0; the conditional gives 2 or 3 and the monad applies to that, so it is -3. In the same way `#[;a;b;c]` was the size of `a` and is 3. A function call whose argument is a conditional was not affected.

```kgpy
?> -:[0;2;3]
-3
?> -:[1;2;3]
-2
?> a::[7 8 9]
[7 8 9]
?> #[;a;1;2]
3
```


#### Calling a value that is not a function

Before: `name(args)` returned the value of `name` and dropped the arguments when `name` held a number, a string, a list or a dictionary, at every strict level. A definition written without its braces, `avg::(+/x)%#x`, made `avg` the number 1.0, and `avg([3 3 3 4 5 7])` printed 1.0.

Now:

* The call is a `type` error that names the callee and what it holds, with the source span of the call.
* It applies to every written call (`a(1)`, `a()`, `a(1;2)`, a name inside a function, `x(1)` on a parameter that holds a value) and to a name given to an adverb (`a'[1 2]`, `a/[1 2]`).
* A function, a projection, an operator, a Python function brought in with `.py` or `.pyf`, an IPC connection from `.cli` and a database from `.db` are called as before.
* A name that is not defined is still `undefined function`. A symbol that names a function is still called through (`g:::f;g(2)`).
* A dictionary is read with `d@k`; `d(k)` gave the dictionary and now fails.

```kgpy
?> avg::1.0
1.0
?> avg([3 3 3 4 5 7])
error: cannot call avg: it is a float, not a function
?> avg::{(+/x)%#x}
:monad
?> avg([3 3 3 4 5 7])
4.166666666666667
```

Fix: a function literal passed as an argument is evaluated where it stands, so `ap::{x()}` called as `ap({7})` calls the number 7 and now fails. Pass the function by name, `g::{7};ap(g)`.



#### Names with a dot, and dictionary literals

**A name with a dot is a qualified name, not a variable.** In 0.7 `a.b::5` defined a variable named `a.b`. Now `a.b` is `a@"b"`, so `a` must be a dictionary, and a program that wrote dotted names as variables fails with a name error that says so. Use a plain name or an entry of a dictionary. [`kgpy lint`](lint.md#w021-abv-where-a-is-not-a-dictionary) finds these writes (W021).

```kgpy
?> a.b::5
error: a is not defined (a.b means a@"b")
?> a:::{}
:{}
?> a.b::5
5
?> a
:{["b" 5]}
```

**The dictionary literal rejects more.** An operator or a function as a key, and an entry of the plain literal that is not exactly `[key value]`, are syntax errors found before the program runs. They were accepted: `:{[1 2 3]}` was the entry `1:2`, and `:{[1]}` was an `IndexError`.

```kgpy
?> :{[1 2 3]}
error: each entry of a dictionary literal must be a [key value] pair
?> :{[+ 1]}
error: an operator cannot be a dictionary key
```

#### Libraries and modules

**`csv` and `nstat` are modules, and `mgmt` is a new one.** `.l("csv")` binds the module to the file's name, so `csv.read` works as before; its helper names are private. `.l("nstat")` binds only `nstat`, where 0.7 put `mu`, `sd` and the rest in the global scope, so write `nstat.mu(v)`. `mgmt` is new in version 2: `mgmt.add`, and its settings in the dictionary `mgmt.cfg` (`mgmt.cfg.pi`, `mgmt.cfg.ct`, `mgmt.cfg.probe`, `mgmt.cfg.trap`). On a node the three names that `mgmt` installs are `.mgmt.run`, `.mgmt.job` and `.mgmt.r`.

**Dotted helper names in the other libraries are renamed**, because a dotted name is now a qualified name:

| Library | 0.7 | Now |
|---|---|---|
| `math` | `pi0.5`, `pi1.5` | `pihalf`, `pi3half` |
| `math` | `e.tlr`, `ln.tlr`, `ln.apx`, `ln.big` | `etlr`, `lntlr`, `lnapx`, `lnbig` |
| `math` | `sin.tlr`, `sin.pos`, `gln.c` | `sintlr`, `sinpos`, `glnc` |
| `nstat` | `rss.set`, `rse.set`, `mse.set`, `r2.set` | `nstat.rssset`, `nstat.rseset`, `nstat.mseset`, `nstat.r2set` |
| `nstat` | the lognormal family `ln.pdf`, `ln.cdf`, `ln.qf`, `ln.mu`, `ln.M`, `ln.mo`, `ln.var`, `ln.skew` | `nstat.lnorm.pdf`, `nstat.lnorm.cdf`, `nstat.lnorm.qf`, `nstat.lnorm.mu`, `nstat.lnorm.M`, `nstat.lnorm.mo`, `nstat.lnorm.var`, `nstat.lnorm.skew` |
| `nstat` | `F.pdf.n`, `F.pdf.d` | private helpers `Fpdfn`, `Fpdfd` of `nstat.F.pdf` |
| `help` | `help.add`, `op.db` | `helpadd`, `opdb` |

Its functions are `nstat.mu`, `nstat.sd` and the rest, and each distribution family is a dictionary inside it, as in `nstat.n.cdf(x;m;s)`.

```kgpy
?> .l("nstat");nstat.n.cdf(0;0;1)
0.5
?> n::5
5
?> n.cdf(0;0;1)
error: n.cdf: n is not a dictionary (it is integer)
```

#### Removed switches

**There is no engine switch and no parser switch.** The Rust interpreter is the only one. `KLONGPY_ENGINE`, `KLONGPY_PARSER` and `KLONGPY_IPC` were switches of pre-releases of version 2 (0.7 had none); they have no effect. `KlongInterpreter(parser=...)` is a `TypeError`, as any unknown keyword is. The old reading of the text that the parser switch gave is not available: `a::5;[a b]::[1 2];a` is 1.

#### Messages changed

A program that matches on message text needs the new text.

```kgpy
?> 1 f
error: undefined function: f
?> 1.2.3
error: malformed number '1.2.3'
```

These were `undefined: f` and a `ValueError` from `float()`. Some texts name the kind of an operand now, where they gave its value: `5.2$3` reports `got integer`, and a list as a dictionary key reports `a list cannot be a dictionary key`. The error carries its position, its kind and the source: see [Reading an error](errors.md#reading-an-error).


#### Command line

**Errors go to stderr, with the position and a caret.** At the REPL they were printed on stdout as `Error: (args...)`. A file run or `-e` printed a Python traceback; a traceback is now printed only with `-v` or `--traceback`.

```
$ kgpy -e '1+"a"'
<expr>:1:2: type error: cannot apply + (Plus) to integer and string
  1+"a"
   ^
```

**Exit status.** `kgpy file.kg` and `-e` exit with 1 after an error. Piped stdin exits 1 if any entry failed. `-t` runs every line, reports each failing line and exits after the last, with 1 if any failed; it no longer starts the REPL afterwards. Ctrl-C ends a file run with `interrupted` and status 130, and SIGTERM ends one that is waiting for a server or a connection with `terminated` and status 143. A file that cannot be opened is `system error: cannot open ...`. See [the command line](errors.md#the-command-line).


#### IPC

**The pickle protocol is removed.** A 0.7 client connecting to a 2.0 server is closed at once and the server logs the cause; a 2.0 client connecting to a 0.7 server fails after 5 seconds with `handshake timed out`. A Python object that has no kdb+ form (a set, a complex number, `None`) was carried by pickle and is now a type error on the sending side. Python `bytes` and `bytearray` cross as a kdb+ byte vector and arrive as an integer array. An integer beyond 64 bits was carried by pickle and is now `range error: integer does not fit 64 bits` on the sending side. See [0.7 peers](ipc_capabilities.md#07-peers).

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

**A synchronous call to the process's own server is an error.** The call would block the loop that has to answer it, so it is a `system` error that names `.async`. Use `.async(f;callback)`, or call from another process.

**A call waits 30 seconds.** A call with no answer raises an error that contains `timed out`, closes the connection and fails every call waiting on it. Change the limit with `.clit`. See [Closing, Timeouts and `.clit`](ipc_capabilities.md#closing-timeouts-and-clit).


#### Web server, files and examples

**A web handler that returns a dictionary sends a response.** In 0.7 the dictionary was converted to Python's text and sent as the body. The keys `"status"`, `"type"` and `"body"` now choose the status, the content type and the body, and any other key is a type error. A handler that returns text is as before. See [The response](web_server.md#the-response).

**A handler that fails answers 500, not 400.** The body is `handler failed`, and one line with the route and the error goes to stderr. It was 400 with the body `Invalid request`, and nothing said why. A body that cannot be read is still 400.

**A `HEAD` request to a GET route is answered as the GET, without a body.**

**Handlers run on the Klong loop.** A handler can start a timer, make an IPC call and call `.webc`. `.webc` in a handler completes the reply, stops the server and lets the run end. See [Where a handler runs](web_server.md#where-a-handler-runs).

**A program that waits for its own server on the Klong loop gets a 503, not a hang.** A blocking HTTP request from Klong code to the program's own `.web` server (`urlopen` through `.pyf`) waited for ever, because the Klong loop that had to answer was the one waiting. The server now answers such a request, after one second, with status 503 and a body that says why, and writes one line to stderr. Slow handlers and clients in other processes are not affected. See [Where a handler runs](web_server.md#where-a-handler-runs).

**`.rls()` strips the line end from each line.** The lines kept their `\n`.

**`.compile`, `.compilex` and `.cmodes` are removed.** They wrapped `torch.compile`, which needs a C++ compiler and cannot read a function that the interpreter runs. Calling a removed function is the ordinary undefined-function error. The torch backend's `.export(fn;input;path)` runs the function once and every tensor operation it executes is recorded with `torch.export` (non-strict). To compile a function, export it and compile the exported program with torch: `torch.compile(torch.export.load(path).module())`.

**The two broker examples are removed.** `examples/stocks` had a feed from Polygon and one from Alpaca, and both needed an account. A simulated price feed replaces them: a seeded random walk, a server that publishes each trade to the clients subscribed to its symbol, and a consumer that computes a volume-weighted average price and a moving average. It needs only `pip install klongpy`. See [`examples/stocks/feed`](https://github.com/briangu/klongpy/tree/main/examples/stocks/feed).

**The example programs are written in version 2 forms.** They use named parameters, destructuring, evaluated dictionaries and `.pf` where those read better, and keep the classic forms where those are better Klong. Their output is the same.


### Fixes

These were crashes, hangs or wrong results in 0.7 on features that existed then.


#### Operators and the parser

**`&[]` is `[]` on the NumPy backend.** Where of an empty list was a type error, although the manual says it returns nil and the torch backend and the interpreter on boards did. `d_k` on a dictionary, the wrong order for removing a key, now says in a hint that the key goes on the left, `k_d`.

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

**Several operators that crashed or looped now return a value or a Klong error.** `$[1 2 3]` recursed until Python stopped it. `|5` was a `TypeError`. `&/"abc"` and `+/"abc"` failed on a string; they now give `0ca` and a type error. `[2 2]:^[]` and `5:^[]` raised a `ZeroDivisionError`. A dictionary joined with something that is not a two-member list was silently ignored.

```kgpy
?> $[1 2 3]
["1" "2" "3"]
?> |5
5
?> &/"abc"
0ca
?> +/"abc"
error: cannot apply +/ (Plus-Over) to string
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

**`^` with a list operand follows one rule.** Integers wrap in 64 bits for a non-negative integer exponent, and a negative exponent or any float gives floats. Two integer scalars stay exact, so `2^64` is exact and `[2 3]^64` wraps. The rule is the same for `list^scalar`, `scalar^list` and `list^list`; see [Numbers](differences.md#numbers).

**Integers beyond 64 bits are exact.** An operand outside the 64-bit range made `+ - * < > | &` fail with a Python `OverflowError`, although the language already made such integers (`2^64`, a long literal). The operation is now done on exact integers, in a list as well as on a scalar, and a result that fits in 64 bits again is an ordinary integer. Operands that are both inside 64 bits keep their wrap-around, as before: `9223372036854775807+1` is `-9223372036854775808`. `^` of two integer scalars with a non-negative exponent is exact as well (`10^400`; a result beyond about a million bits is a range error), and `!` is an error for such an operand.

```kgpy
?> 1+9223372036854775808
9223372036854775809
?> [1 2]+9223372036854775808
[9223372036854775809 9223372036854775810]
?> 9223372036854775808*3
27670116110564327424
?> 9223372036854775807+1
-9223372036854775808
?> 7^22
3909821048582988049
?> 10^100000000
error: cannot apply ^ (Power) to integer and integer
```

**A recursive call through `.f` has its own locals.** `.f` called itself without declaring the function's locals, so the inner call changed the outer call's variable. `f::{[t];t::x;:[x<2;t;t+.f(x-1)]};f(3)` was 3, where the same function written with its own name gives 6.

```kgpy
?> f::{[t];t::x;:[x<2;t;t+.f(x-1)]}
:monad
?> f(3)
6
```

**A Python function with optional parameters gets every argument.** A function brought in with `.py` or `.pyf` was called with its required parameters only, so `linspace(0;1;5)` made 50 points and `arange(2;8)` was `[0 1]`. It now gets each argument written. It is displayed by its required parameters, as before.


#### Performance

On an AMD EPYC 7773X, finding distinct IDs among one million resident integers with 64 dense IDs fell from 29.73 ms to 2.80 ms against the preceding v2 build. Through Python, a 100,000-step Klong loop ran in 13.64 ms on RC1 and 5.72 s on v1. Returning a writable running sum of one million integers took 1.019 ms on RC1 and 0.753 ms on v1. On an Apple M2 Ultra, the native scan used 8.02 MB of peak interpreter heap, down from 16.02 MB. Group counts for a million integers supplied from Python took 48.63 ms on RC1 and 35.39 ms on v1. [Runtime benchmarks](performance.md)

All 232 cases in the strict release acceptance selection passed. On the default-placement Linux feed, the worst batch acknowledgement took 15.76 ms on `kg` and 76.24 ms on `kgpy`, within the 100 ms budget. Against PeachQ 0.92, the Linux wheel passed both controlled-placement series and all three default-placement series on the median bounds. At default placement, median as-of p99 ratios were 1.043×, 1.069× and 1.243× against a 1.25× bound, with one individual pair at 1.353×. [Streaming measurements](performance.md#streaming-tables-and-queries)

**Indexing and appending are faster.** Indexing a vector or a string by an integer vector is a whole-array operation, so `x@<x` sorts in about the time of the grade. Appending a number to a vector and `^` on integer arrays are faster, most of all on the torch backend. [Performance](performance.md) has the measurements and says which rows predate the change.


#### System functions

**`.sys` runs a shell command and returns its exit status.** It tried to start a program named by the whole string.

```kgpy
?> .sys("exit 3")
3
```

**`.timer` checks its arguments.** The interval may be a real number such as `0.5`. A negative or non-numeric interval, or a callback that is not a function, is an error; they were a returned string or a `ValueError`. An error in a callback is logged in one line and cancels the timer, and a callback that is not a function says what a callback must be.

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

**`.rs` checks its argument.** A value that is not a string is a type error, and empty text gives `:undefined`.

**`.l`, `kgpy lint`, `kgpy check` and `kgpy compile c` look for a file in one order.** The directory of the file that contains the `.l` (the working directory at the prompt), then the directories of `KLONGPATH` (default `.:lib` on Unix and `.;lib` on Windows), then the package library. A bare name is the same for classic files and modules, and a local file that shadows a library file gives a warning.

**`.unjson` keeps every integer of a list exact.** `.unjson("[9223372036854775808,0.5]")` turned the large integer into a float, because the list became a float array. A list that holds an integer beyond 64 bits is now a list of objects, as it is when it holds only integers.

**A SQL result of one row and one string column displays, and a SQL or store error names its cause once.** The word `error` was printed twice. A module that needs an extra says which: `klongpy.db needs the db extra, which is not installed (no module named duckdb): pip install "klongpy[db]"`; the same holds for `klongpy.web` and the torch backend.

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


#### Libraries

**`fft` and `eigenv` work.** `fft` in `math.kg` and `eigenv` in `eigenv.kg` raised errors. `eigenv` needed two operators to accept more than they did: amend in depth (`:-`) with a real value joined to integer indices, and join (`,`) of lists of different shape. Both do now. The library computes with its own sine and cosine, so `fft` is accurate to about four decimals, not to the last digit of a double.

```kgpy
?> .l("math.kg");0
0
?> fft([1.0 0.0 0.0 0.0])
[[1.0 0.0] [1.0 0.0] [1.0 0.0] [1.0 0.0]]
?> .l("eigenv.kg");0
0
?> eigenv([[2.0 1.0 0.0] [1.0 2.0 1.0] [0.0 1.0 2.0]];4)
[[3.4143 1.9998 0.5859] [[0.4993 -0.7073 0.5004] [0.7073 -0.0008 -0.7069] [0.5004 0.7069 0.4999]]]
```


#### Torch backend

**Torch scalar arithmetic uses float32.** A scalar now agrees with the matching element of a default tensor, and `(0.1+0.2)=0.3` is 1 on torch and 0 on NumPy. Explicit float64 arrays keep their width on devices that support it. `point∇function`, `.grad` and `point∂function` evaluate the function in float64 on the CPU and return float32. `.try` accepts a tensor as its argument list, `.float` converts a tensor, and `a.b` on a tensor is a type error. [PyTorch Backend and Autograd](torch_backend.md)

**The optimizers are in the package.** `SGDOptimizer` and `AdamOptimizer` are in `klongpy.optimizers`, so a `pip install` has them, and they work on Apple MPS, where the gradient of `x^2` at 0 is 0 and not NaN. See [Optimizers](torch_backend.md#optimizers).

**`.bkf` functions accept scalars and `.gradcheck` accepts a constant matrix.**
