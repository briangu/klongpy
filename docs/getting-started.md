# Start Here

KlongPy runs Klong, a small array language. In an array language you apply an operation to a whole list of numbers at once instead of writing a loop. The sum of the numbers 1 to 5 is written `+/1+!5`. This page takes you from installing KlongPy to writing a small script, then points to the part of the documentation that fits what you want to do.

You need Python 3.9 to 3.14. You do not need to know any other array language.

## Install

```bash
pip install klongpy
```

That installs the interpreter, the `kgpy` command and its two requirements, NumPy and colorama. It is enough for everything on this page. Extra features have their own packages. Ask for them in square brackets:

| Install | Adds | Use it for |
|---|---|---|
| `pip install "klongpy[torch]"` | PyTorch | The torch backend: GPU arrays and gradients with `:>`. Start it with `kgpy --backend torch`. See [PyTorch Backend and Autograd](torch_backend.md). |
| `pip install "klongpy[db]"` | pandas and DuckDB | Tables, SQL on arrays, and the key-value store. See [Columnar tables and SQL](fast_columnar_database.md). |
| `pip install "klongpy[web]"` | aiohttp | The `.web` HTTP server. See [Web Server](web_server.md). |
| `pip install "klongpy[ws]"` | websockets | WebSocket clients; there is no server. See [WebSockets](websockets.md). |
| `pip install "klongpy[all]"` | Everything above | One command for all of it. `klongpy[full]` adds pytest to that, for running the tests. |
| `pip install "klongpy[repl]"` | Nothing new | Kept so that older instructions still work. The REPL needs no extra. |

Two more extras are for people who work on KlongPy itself: `dev` (the test tools) and `docs` (the documentation build). Work in a virtual environment if you want the packages kept apart from other projects:

```bash
python3 -m venv venv
source venv/bin/activate
pip install klongpy
```

Check that it works:

```console
$ kgpy -e '1+1'
2
```

`kgpy -e` evaluates one expression, prints the value and exits. `kgpy file.kg` runs a file.

## Your first five minutes

Type `kgpy` to start the REPL. The banner shows the version and the backend. The REPL has line editing, history and tab completion built in, so `rlwrap` is not needed. It reads one entry, evaluates it and prints the value. Quit with Ctrl-D or `]q`. The lines below that start with `?> ` are what you type, and the line after each is what KlongPy prints.

```kgpy
?> 1+1
2
?> 10-4
6
```

### Reading right to left

Klong has no operator precedence. An expression is read from right to left, and each operator takes everything to its right as its second argument. `2*3+4` is `2*(3+4)`, not `(2*3)+4`:

```kgpy
?> 2*3+4
14
?> (2*3)+4
10
```

Use parentheses to change the order. Read each expression from the right end: `2*3+4` is "add 3 and 4, then multiply by 2".

### Lists

`!5` makes the list of the numbers from 0 up to but not including 5. An operation on a list applies to every member:

```kgpy
?> !5
[0 1 2 3 4]
?> 1+!5
[1 2 3 4 5]
?> 2*1+!5
[2 4 6 8 10]
```

`+/` is the sum of a list: `+` with the adverb `/` (read "over"), which puts `+` between the members. `#` is the length, and `|` reverses:

```kgpy
?> a::[3 1 4 1 5 9 2 6]
[3 1 4 1 5 9 2 6]
?> +/a
31
?> #a
8
?> |a
[6 2 9 5 1 4 1 3]
?> a@0
3
```

`::` assigns. `a@0` is the member of `a` at index 0. Strings are lists of characters, and the same operators work on them:

<!-- check-docs: continue -->
```kgpy
?> #"hello"
5
?> |"hello"
"olleh"
```

### Your first function

A function is code between braces, and `name(parameters)::{...}` gives it a name and names its parameters. The mean of a list is its sum divided by its length:

<!-- check-docs: continue -->
```kgpy
?> avg(v)::{(+/v)%#v}
:monad
?> avg(a)
3.875
?> avg(1+!5)
3.0
```

`%` is division. The REPL shows the value of the definition: `:monad` means a function of one argument (`:dyad` is two, `:triad` is three, and `:fn/4` is four). The names are local to the function. A call with fewer arguments than the function takes waits for the rest:

<!-- check-docs: continue -->
```kgpy
?> add(a;b)::{a+b}
:dyad
?> add(2;3)
5
?> inc::add(1)
:monad
?> inc(10)
11
```

Klong also has unnamed arguments: inside a function written `{...}` alone, `x` is the first argument, `y` the second and `z` the third, so `{x+y}` is a dyad. They suit a one-line function given to an adverb, such as `{x*x}'!5`; the [Tutorial](tutorial.md#4-functions) teaches them first.

`.p` prints a value followed by a newline. The REPL then shows the value that `.p` returned, which is why the text appears twice:

<!-- check-docs: continue -->
```kgpy
?> .p("mean: ",$avg(a))
mean: 3.875
"mean: 3.875"
```

`$` turns a number into text, and `,` joins two lists, here two strings. In a script file nothing echoes values, so you see the text once.

### Your first script

Put the program in a file whose name ends in `.kg`. Here the shell commands make the file and run it:

```console
$ cat > stats.kg <<'EOF'
> avg(v)::{(+/v)%#v}
> data::[3 1 4 1 5 9 2 6]
> .p("mean: ",$avg(data))
> .p("largest: ",$|/data)
> EOF
$ kgpy stats.kg
mean: 3.875
largest: 9
```

`|/` is the maximum of a list: `|` (the larger of two) with `/`. A script does not print the value of each line. It prints what you print with `.p`.

### Reading an error

An error says where it happened, what kind of mistake it is, and shows the line with a caret under the place. Adding a number to a string is a type error:

<!-- check-docs: render -->
```kgpy
?> 1+"a"
1:2: type error: cannot apply + (Plus) to integer and string
  1+"a"
   ^
```

The position is `line:column`. The word before `error` is the kind: `type`, `name` (an undefined function), `syntax`, `range` (an index out of bounds), `shape` (lists of different lengths) and a few more. [Errors](errors.md) lists them all and says how the short form `error: MESSAGE`, which other pages use in transcripts, relates to this one. The REPL stays open after an error. In a script the error also names the file, and the run stops at the first error with exit status 1:

```console
$ cat > bad.kg <<'EOF'
> a::[1 2 3]
> .p(a@5)
> EOF
$ kgpy bad.kg; echo $?
bad.kg:2:5: range error: index 5 out of range for list of 3
  .p(a@5)
      ^
1
```

### Getting help

At the REPL, `]h` lists every topic and `]h` followed by an operator or a name explains it, with examples. `]a` searches the help for a word. The help covers the operators, the adverbs and the system functions, which are the names that start with a dot, such as `.p`:

```console
$ printf ']h #\n' | kgpy 2>&1 | tail -n +8 | head -3
?> #a                  [Size]

    Return the size/magnitude of "a".
```

Other places to look:

- [REPL](repl.md) lists the `]` commands, multi-line entry, completion and history. Multi-line entry means that a definition with an open `{` can continue on the next line. `it` holds the last value printed.
- [Operators](operators.md) and [System Functions](system-functions.md) are the reference pages.
- [Errors](errors.md) explains every message and exit status.

`kgpy lint file.kg` reports likely mistakes in a file without running it, and `kgpy --strict file.kg` turns silent mistakes into errors while it runs. See [Lint](lint.md) and [Strict Mode](strict.md).

## Choose your path

**Coming from NumPy.** If you think in arrays and broadcasting, [Klong for NumPy Users](from-numpy.md) has a translation table from NumPy calls to Klong and lists the mistakes that people make first.

**Coming from q or kdb+.** [Klong for q Users](from-kdb.md) shows what carries over, what is different (a leading minus, no booleans, division), and how a KlongPy process and a q process talk to each other.

**Learning the language.** The [Tutorial](tutorial.md) takes you from the first expression to functions, adverbs, dictionaries and files. [Examples](examples.md) has programs to read.

**Using Python.** `.py` and `.pyf` import Python modules and functions into Klong, and a Klong function can be called from Python code. See [Python Integration](python_integration.md).

**Gradients with torch.** With the torch backend, `f:>x` gives the exact gradient of a function. See [PyTorch Backend and Autograd](torch_backend.md).

**Tables and the database.** `.table` makes a table from arrays, and `.db` runs SQL on it with DuckDB. See [Columnar tables and SQL](fast_columnar_database.md) and [Table and Key-Value Stores](table_and_key_value_stores.md).

**Talking between processes.** `.srv` serves a KlongPy process and `.cli` calls one, over the kdb+ IPC protocol, so a q process can be on either end. See [IPC Capabilities](ipc_capabilities.md).

**Compiling to C for microcontrollers.** A static subset of Klong compiles to C99 with no heap. Three examples have run compiled on two Raspberry Pi Picos, and the C has run on no other device. See [Compiling Klong to C](embedded.md) and [The Static Profile](static-profile.md).

**The native runtime on boards.** A separate implementation in Rust runs the full language without Python, including on an ESP32-P4. See [Native Runtime](native/index.md), [Building and Running](native/building.md), [Serial Console and Upload](native/console.md) and [ESP32-P4](native/esp32-p4.md).

If you used an earlier KlongPy, read [What's New in v2](whats-new-v2.md) and [Migrating to v2](migrating-to-v2.md).

For a question or a problem, start a [GitHub Discussion](https://github.com/briangu/klongpy/discussions). The project does not accept issues directly; maintainers promote discussions to issues.
