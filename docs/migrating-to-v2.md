# Migrating to v2

This guide covers upgrades from the pure-Python line, including 0.7 and the source line named v1. The shorter 0.7 examples below describe that line's earlier releases. To stay on the published pure-Python line, install `klongpy<2`. Check [Results that change without an error](#results-that-change-without-an-error) before you upgrade. [Release notes](release-notes.md)

## Quick list

| Change | What to do |
|---|---|
| There is no pure-Python KlongPy 2; the interpreter is written in Rust | Install a wheel, or build with a Rust toolchain; [see below](#the-interpreter-and-the-commands) |
| `a.b` is `a@"b"`, not a variable named `a.b` | Use a plain name or a dictionary; [see below](#names-with-a-dot) |
| An operator or function as a dictionary-literal key, and an entry that is not `[key value]`, are syntax errors | [Write the pair](#dictionary-literals) |
| The libraries `csv` and `nstat` are modules (so is `mgmt`, which is new), and dotted helper names in `math`, `nstat` and `help` are renamed | [See the table](#libraries) |
| `.l` is the one loader: a file that assigns `export` is a module, and `.l(path)`, `.l(path;:s)` and `.l(path;[names])` bind it; a bare name is searched in the calling file's directory first | [See below](#modules-and-l) |
| `.extern(c_symbol;signature)` returns a function; a target's functions live in a module file, and stand-ins are keyed by C symbol | [See below](#extern-returns-a-function) |
| `.compile`, `.compilex` and `.cmodes` are removed; `.export(fn;input;path)` stays | [See below](#torch-compile) |
| `KlongInterpreter(parser=...)` is removed, and there is no engine switch | Remove them; [see below](#removed-switches) (only pre-releases of 2 had them) |
| Python 3.11 or later | [See below](#python-version) |
| A function with named parameters does not see its caller's locals | [Pass the value](#named-parameters-are-lexically-scoped) |
| A function literal with no parameters is a function wherever it is written | [See below](#a-function-literal-with-no-parameters) |
| `.l("nstat")` binds `nstat` only, `.rn` is seeded with `.seed`, and numpy's seed no longer reaches it | [See below](#libraries) and [random numbers](#random-numbers) |
| Torch scalar arithmetic uses float32, so `(0.1+0.2)=0.3` is 1 | [See below](#torch-float32-and-numeric-gradients) |
| `kg` is the Rust binary and `kgpy` the Python-hosted command; Python-only functions under `kg` say they need `kgpy` | [See below](#the-interpreter-and-the-commands) |
| Both desktop commands include tables and DuckDB; importing `klongpy.db` selects the legacy Python adapter in that interpreter | [Check which database API your code uses](#tables-and-sql) |
| New names: `.try`, `.err`, `.seed`, `.now` | [See below](#new-system-functions-that-take-names) |
| String indexing gives a character; `[:a :b]?:b`, `&/"abc"`, `+/"abc"` and others changed | [See below](#bugs-of-07-that-2-does-not-have) |
| IPC: pure-Python and 2.0 peers use incompatible protocols | [Upgrade both ends](#ipc-speaks-the-kdb-protocol) |
| `[a b]::v` binds; `f(a;b)::{...}` defines | Write the program for the new reading |
| Strings such as `"{"`; whitespace before `;` in a conditional | None needed: the program that worked around them still runs |
| Left-over input warns (level 0) or fails (strict) | Level 0 is the default; `warnings.filterwarnings("ignore", category=KlongSyntaxWarning)` hides the warning |
| A call on a number, string, list or dictionary is a `type` error | [Name the function](#calling-a-value-that-is-not-a-function) |
| Arity, projections, `d@k`, `:undefined`, `x%0`, conditional monads, symbols as data, operator results | None |
| Error classes and messages | None; `except KlongException` and the old Python classes still catch |
| Errors on stderr, exit statuses | None |
| Strict levels | `.strict(0)` or `.unsafe()`; the default is 0 |
| A web handler that returns a dictionary sends a response; a failing handler answers 500, not 400 | [See below](#web-handlers) |
| `.rls()` strips the line end from each line | [See below](#other-changes) |
| The Polygon and Alpaca examples are removed | [A simulated feed replaces them](#other-changes) |

## The interpreter and the commands

Version 2 evaluates Klong on one interpreter written in Rust. `pip install klongpy` installs `kgpy` for Python-hosted programs and `kg` for standalone programs. A service that calls Python, starts a web server or opens a WebSocket keeps using `kgpy`. A function such as `.py`, `.pyf`, `.pyc` or `.web` raises a name error under `kg` that identifies `kgpy` as the required command. [What Runs Where](what-runs-where.md)

What this changes for your code:

* **The Python modules of the old implementation are gone.** `import klongpy.parser`, `klongpy.lexer`, `klongpy.compiler`, `klongpy.monads`, `klongpy.dyads` and `klongpy.adverbs` fail with `ModuleNotFoundError`. `KlongInterpreter`, the error classes, `klong['name']`, the backends and `.py`/`.pyf` are kept. Code that built or walked a parse tree needs `kgpy ast` or `kg ast`, which print the tree as text.
* **`import klongpy` needs its compiled extension.** A wheel carries it. A source tree builds it with `pip install -e .`, which needs a Rust toolchain. Without it the import fails with a message that says so, and there is no slower fallback.
* **The tools are Rust.** `kgpy lint`, `check`, `compile c`, `fpga source` and `ast` call the same code as `kg lint`, `kg check` and the rest. Their output is the same text as before except where a message names a new rule. `klongpy.lint.lint_source` still works from Python.
* **Warnings are issued before the program runs.** The syntax warnings of a file come out after it is parsed and before its first statement, so they appear above the program's output.

## Tables and SQL

Desktop `kg` and `kgpy` include the same `.table`, `.index`, `.insert` and `.db` primitives. `.db("trades.duckdb")` opens a persistent DuckDB file. `.db(:{;"trades";t})` exposes a Klong table as a live SQL view. Start with the built-ins for new Klong code.

Existing Python programs can keep importing `klongpy.db` with the `db` extra for its pandas-backed tables, Python DuckDB interface and key-value stores. That import makes the Python adapter's functions available to the hosted interpreter. When switching a program to the built-in SQL interface, check comparisons and downstream Python conversions for exact integer aggregates and `:undefined` for SQL NULL. Select `epoch_ns(ts)` when reading a timestamp as nanoseconds. [Columnar tables and SQL](fast_columnar_database.md)

## Python version

Version 2 needs Python 3.11 to 3.14. 0.7 accepted Python 3.9 and later. Upgrade the interpreter before you upgrade KlongPy. The wheel is one `abi3` file per platform.

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

`a::5;[a b]::[1 2];a` was 5 and is 1. Why: binding is what the form is for; it gave a silent no-op before. A list on the left that is not a list of distinct plain names (nested, repeated, empty) still binds nothing at level 0, now with a warning, and is a syntax error at strict levels 1 and 2.

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

### Named parameters are lexically scoped

Before: every function was dynamically scoped, so a function saw the locals and parameters of whoever called it. Now a function written with named parameters, `f(a;b)::{...}`, resolves a free name in its own frame and then in the globals of its file or module, and never in the caller's locals. A classic function (`{x+y}`, or `{[t];...}` with a locals list) keeps dynamic scope, and so does a function it calls.

```kgpy
?> t::9
9
?> g(a)::{t+a}
:monad
?> f::{[t];t::5;g(1)}
:nilad
?> f()
10
?> h::{t}
:nilad
?> f2::{[t];t::5;h()}
:nilad
?> f2()
5
```

`f()` is 10 and not 6: `g` has named parameters, so it reads the global `t`, and `h`, a classic function, reads the `t` of `f2` that called it. A program whose named-parameter function read a caller's local must pass the value as an argument. Why: a named-parameter function is the form that [design principle 4](design-principles.md) asks for, and its names are visible where it is written.

### A function literal with no parameters

Before: a function literal that uses none of `x`, `y` and `z`, such as `{work()}`, was evaluated at the place where it was written when it was an argument (see [Calling a value that is not a function](#calling-a-value-that-is-not-a-function)). Now it is a function in the places that want one: the function argument of `.try`, `.async` and `.timer`, and an element of a list or dictionary written with `;`, such as `:{;"on";{work()}}`. A second argument of `.try` that is not a list is a type error.

```kgpy
?> d:::{;"on";{1}}
:{["on" :nilad]}
?> .try({1%0};[])
[1 :undefined]
```

### Random numbers

Before: `.rn` drew from NumPy's global generator, so `np.random.seed(7)` made the draws repeat, and there was no `.seed`. Now `.seed(n)` seeds the interpreter's own generator, and the draws that follow are the same on every machine and on every target. NumPy's seed no longer reaches `.rn`, and the sequences differ from the ones NumPy gave: `.seed(42);.rn()` is 0.1941059175341826. A program that wants repeatable draws calls `.seed`.

```kgpy
?> .seed(42);.rn()
0.1941059175341826
```

### Torch: float32 and numeric gradients

The torch backend computes Klong scalars and default floating-point arrays in float32. On the pure-Python line, a scalar was a Python float (float64), so a scalar and the matching tensor element could differ. Now `(0.1+0.2)=0.3` is 1 under torch and 0 under NumPy. Explicit float64 arrays retain their width on devices that support it. A passive float64 IPC echo or symbol lookup keeps its wire precision. Numeric gradients (`point∇function` and `.grad`) use float64 working precision and return arrays on the selected device in its Klong compute dtype. MPS evaluates the perturbations on the CPU. [PyTorch Backend and Autograd](torch_backend.md)

### New system functions that take names

`.try`, `.err`, `.seed` and `.now` are system functions of version 2, and 0.7 had none of them. A program that defined a variable or function of one of these names keeps the definition but loses the system function. `.try(f;args)` gives `[1 result]` or `[0 e]` and `.err(kind;message)` raises, `.now()` is the time in seconds since 1970. See [System functions](system-functions.md).

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

Why: `1 2 3` is nearly always a list written without brackets. To silence the warning from Python, use `warnings.filterwarnings("ignore", category=KlongSyntaxWarning)`.

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

## Names with a dot

`a.b` means `a@"b"`: the entry `"b"` of the dictionary `a`. 0.7 read `a.b` as one variable name, so a program could write `geom.sq(v)::{...}` and `count.total::0`. Now the first is a name error, because `geom` is not a dictionary, and so is the second.

```kgpy
?> count.total::0
error: count is not defined (count.total means count@"total")
?> count:::{}
:{}
?> count.total::0
0
?> count.total::count.total+1
1
```

What to write instead:

* A plain name: `countTotal::0`. The libraries did this, so [their dotted names changed](#libraries).
* A dictionary, when the dots grouped values: `count:::{;"total";0}` and `count.total`.
* A module, when the dots grouped functions of a file: the file ends with `export:::{;"sq";sq}`, and `.l("geom")` gives `geom.sq(v)`. See [Modules](language-additions.md#modules).

`kgpy lint` finds the writes (rule W021). A name that starts with a dot and has dots inside, such as `.srv.auth`, is one name, as before. [Qualified names](language-additions.md#qualified-names) has the rules, and `.module(:name)` still exists for a classic file but gives no qualified names.

## Dictionary literals

An operator or a function as a key of a dictionary literal, and an entry of the plain literal that is not exactly `[key value]`, are syntax errors found before the program runs. 0.7 accepted them: `:{[+ 1]}` made a dictionary keyed by the operator, `:{[1 2 3]}` made `1:2`, `:{[{x} 1]}` made a key from the text `{` and `:{[1]}` raised an `IndexError`.

```kgpy
?> :{[+ 1]}
error: an operator cannot be a dictionary key
?> :{[1 2 3]}
error: each entry of a dictionary literal must be a [key value] pair
```

Write each entry as a pair of one key and one value. To have the keys and values evaluated, use `:{;key1;value1;key2;value2}`, in which a function is a value and not a key. A key is a number, a string, a character or a symbol.

## Libraries

The libraries in `klongpy/lib` are loaded with `.l`, as before. Three are modules now and the others lost their dotted helper names, because a dotted name is now a qualified name.

* **`csv` is a module.** `.l("csv")` binds the module to the name `csv`, so `csv.read`, `csv.split`, `csv.make`, `csv.write` and `csv.load` work as before. Its helpers are private: `escape` in `csv` is no longer a global. In 0.7 these were variables whose names contained a dot.
* **`nstat` is a module.** 0.7 put its functions in the global scope after `.l("nstat")`, as `mu(v)` and `n.cdf(x;m;s)`. Now `.l("nstat")` binds only `nstat`, and a call is `nstat.mu(v)` or `nstat.n.cdf(x;m;s)`. A bare `mu(v)` is an undefined-function error.
* **`mgmt` is new in version 2** and is a module: `.l("mgmt")` binds `mgmt`, with `mgmt.add` and the settings in the dictionary `mgmt.cfg` (`mgmt.cfg.pi`, `mgmt.cfg.ct`, `mgmt.cfg.probe` and `mgmt.cfg.trap`). Write a setting as `mgmt.cfg.ct::60`. A node gets `.mgmt.run`, `.mgmt.job` and `.mgmt.r`. Release candidates of version 2 had `mgmt.trap`; it is `mgmt.cfg.trap`. 0.7 had no `mgmt`.
* **The other names changed as follows.**

| Library | 0.7 | Now |
|---|---|---|
| `math` | `pi0.5`, `pi1.5` | `pihalf`, `pi3half` |
| `math` | `e.tlr`, `ln.tlr`, `ln.apx`, `ln.big` | `etlr`, `lntlr`, `lnapx`, `lnbig` |
| `math` | `sin.tlr`, `sin.pos`, `gln.c` | `sintlr`, `sinpos`, `glnc` |
| `nstat` | `rss.set`, `rse.set`, `mse.set`, `r2.set` | `nstat.rssset`, `nstat.rseset`, `nstat.mseset`, `nstat.r2set` |
| `nstat` | `ln.pdf`, `ln.cdf`, `ln.qf`, `ln.mu`, `ln.M`, `ln.mo`, `ln.var`, `ln.skew` | `nstat.lnorm.pdf`, `nstat.lnorm.cdf`, `nstat.lnorm.qf`, `nstat.lnorm.mu`, `nstat.lnorm.M`, `nstat.lnorm.mo`, `nstat.lnorm.var`, `nstat.lnorm.skew` |
| `nstat` | `F.pdf.n`, `F.pdf.d` | private helpers `Fpdfn`, `Fpdfd` (use `nstat.F.pdf`) |
| `help` | `help.add`, `op.db` | `helpadd`, `opdb` |

`sin` of a negative number works in `math`; `sin(-0.5)` raised an undefined-name error in 0.7.

```kgpy
?> .l("nstat");nstat.n.cdf(0;0;1)
0.5
?> n::5
5
?> n.cdf(0;0;1)
error: n.cdf: n is not a dictionary (it is integer)
```

## Modules and `.l`

`.l` of a file is still the way to load it, and it is now the only one. A file that assigns the variable `export` at its top level is a module; any other file is classic Klong, and `.l` loads it into the globals as in 0.7 and returns what it returned.

* **The `export` variable.** A module states its interface with a dictionary written as a literal: `export:::{;"mean";mean;"sd";sd}`. The interface is read and frozen when the file finishes loading, and the other top-level names stay private.
* **Three forms.** `.l("nstat")` binds the module to the file's stem, `nstat`. `.l("nstat";:s)` binds it to `s` instead. `.l("nstat";["mu" "sd"])` binds those entries as plain names, and not the stem; a missing entry is an error that names it. `.l` returns the module. The Python analogues are `import nstat`, `import nstat as s` and `from nstat import mu, sd`.
* **A bundled library binds one name.** `csv`, `nstat` and `mgmt` are modules, so `.l("csv")` binds `csv` and nothing else; see [Libraries](#libraries).
* **`a.b` means `a@"b"`.** A module is a dictionary, so `nstat.mu(v)` is `(nstat@"mu")(v)`, and it is read-only from outside. See [Names with a dot](#names-with-a-dot).
* **Errors.** A second argument on a classic file, a second argument that is not a symbol or a list of strings, and a stem that is not a name (letters and digits, starting with a letter) without an alias are errors that say what to write. At strict levels 0 and 1 `:s` is the value of `s` when `s` is defined, so choose an alias that is not in use.
* **Search order.** A bare name is searched in the directory of the file that contains the `.l` (the working directory at the REPL), then in the `KLONGPATH` directories (default `.:lib` on Unix and `.;lib` on Windows), then in the bundled library. 0.7 searched `KLONGPATH` first and the calling file's directory last. A file found before the library that has the name of a library file gives a warning, and `kgpy lint` reports it as W023. `./x` and `../x` are relative to the calling file. The order is the same for classic files and modules.

## `.extern` returns a function

`.extern(c_symbol;signature)` returns the host function as a value: `adc::.extern("read_adc";"int -> float")`. The statement form `.extern("read.adc";"int -> float")`, whose value is discarded, is an error that names the new form, and `kgpy lint` reports it as E005. The third argument is gone. A target's functions live in an ordinary module file, `read.kg` with `export:::{;"adc";adc}`, and a program calls `.l("read")` and `read.adc(3)`. The C symbol is a C identifier: underscore allowed, no dot, not a C keyword, no `kg_` prefix. A host registers its implementation under the C symbol, and `klong.externs["read_adc"]` is keyed the same way. See [Compiling Klong to C](embedded.md).

## Torch: `.compile`

`.export(fn;input;path)` is the torch graph export, as in 0.7. `.compile`, `.compilex` and `.cmodes` are removed; see [Other changes](#other-changes) for what to use.

## Removed switches

`KlongInterpreter(parser=...)` is a `TypeError`, as any unknown keyword is, and there is no engine switch: the Rust interpreter is the only one. 0.7 had neither, so only a program written against a pre-release of version 2 is affected. `KLONGPY_ENGINE`, `KLONGPY_PARSER` and `KLONGPY_IPC` were switches of those pre-releases and have no effect now; remove them from scripts and service files. The program `a::5;[a b]::[1 2];a` is 1 whatever the environment says.

```console
$ KLONGPY_PARSER=legacy kgpy -e 'a::5;[a b]::[1 2];a'
1
```

## Bugs of 0.7 that 2 does not have

These gave a wrong or failing result in 0.7. They are listed because a program that worked around one now gets the right result and may do the work twice.

| Expression | 0.7 | Now |
|---|---|---|
| `[:a :b]?:b` | `[]` | `[1]` |
| `[:a :b]?:a` | `[]` | `[0]` |
| `&/"abc"` | an `AttributeError` | `0ca` |
| `\|/"abc"` | an `AttributeError` | `0cc` |
| `+/"abc"` | an error from NumPy | a `type` error: `cannot apply +/ (Plus-Over) to string` |
| `["1234" "5678"]:@[0 1]` | an `IndexError` | `0c2` |
| `,\"abc"` | `["a" "ab" "abc"]` | `[0ca "ab" "abc"]` |
| `"abc"@1` | `"b"` | `0cb` |
| `"abc"@-1` | `"c"` | `0cc` |

Indexing a string with an integer gives a character, as in classic Klong, where 0.7 gave a one-character string. `*"abc"` is still the string `"a"`; see [Known bugs](differences.md#known-bugs). Indexing with a list of integers gives a string, as before: `"abc"@[0 1]` is `"ab"`. A program that compares an indexed character with a string needs `0cb` and not `"b"`; the two print differently and compare equal under `~` only in the cases that [Differences from Klong](differences.md#known-bugs) lists.

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

Now the listener speaks kdb+ IPC and accepts q peers. The older pickle protocol used by 0.7 and v1 is incompatible in either direction.

- A 0.7 client connecting to a 2.0 server is closed at once, and the server logs that the client speaks the pickle protocol of 0.7. The client sees `connection lost`.
- A 2.0 client connecting to a 0.7 server fails at the handshake after 5 seconds with `handshake timed out`.

What to do: upgrade every KlongPy process that talks IPC to 2.0 or later. A Python object with no kdb+ form (a set, a complex number, `None`) used to travel by pickle and is now a type error on the sending side; send a list, a string or a dictionary instead. Python `bytes` and `bytearray` cross as a kdb+ byte vector and arrive as an integer array. See [the protocol](ipc_capabilities.md#protocol-and-security) and [0.7 peers](ipc_capabilities.md#07-peers).

Named table keys cross the wire as a kdb+ keyed table. Column order and supported q column types, including typed empty columns and nulls, survive a passive table round trip between desktop peers. [Types across the wire](ipc_capabilities.md#types-across-the-wire)

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
- **`.compile`, `.compilex` and `.cmodes` are removed.** They wrapped `torch.compile`, which needs a C++ compiler and cannot read a function that the interpreter runs. Use `.export(fn;input;path)`, which records the tensor operations the function executes, and compile the exported program with torch: `torch.compile(torch.export.load(path).module())`. A call to a removed function is an undefined-function error.
- **The Polygon and Alpaca examples are removed.** `examples/stocks/feed` is a simulated price feed that needs no account: a seeded random walk, a server with per-symbol subscriptions, and a consumer that computes a VWAP and a moving average.

[Release notes](release-notes.md#system-functions) lists the fixes to system functions (`.sys`, `.timer`, `.webc`, `.kvs`, `.db`), the operators that used to crash, and the libraries that now work.

The rest of v2 is in [What's New in v2](whats-new-v2.md).
