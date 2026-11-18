# Lint: `kgpy --lint`

`kgpy --lint FILE...` reads Klong programs and reports constructs that parse and run without complaint but usually do not mean what they look like. It tokenises and parses. It never evaluates anything, never starts the REPL or an event loop, never imports a Python module, and never runs a file that `.l` loads. A file that contains `.x(3)` lints with exit status 0.

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

A finding is one line, `file:line:col: CODE message`. The findings of a file come in order of position, and files in the order they were named. The last line is the summary. Codes that start with `W` are warnings; codes that start with `E` are errors. Add `-v` after the file names to print a `hint:` line under each finding:

<!-- check-docs: continue -->
```console
$ echo 'my_var::5' | kgpy --lint - -v
<stdin>:1:1: W001 'my_var' reads as my _ var: _ is the Drop operator, names cannot contain underscores
  hint: rename it (my.var or myVar), or put spaces around _ if you mean Drop
0 errors, 1 warning in 1 file
```

The file name `-` reads standard input, which the examples on this page use to stay short. The options may come in any order: `kgpy --lint -v demo.kg` and `kgpy --lint demo.kg -v` are the same. `--lint` with no file name is an error.

Lint finds the same sort of mistake as [strict mode](strict.md), but in a file, before it runs, and without needing the mistake to happen on the path that is taken.

## Options and exit status

| Option | Meaning |
|---|---|
| `--select W001,W010` | Report only these codes. |
| `--ignore W002` | Leave these codes out. An ignored error no longer counts toward the exit status. |
| `--define t,helper` | Names that the host defines before the program runs. W002 accepts them. |
| `-v` | Print the hint of each finding. |

Codes are written in upper case or lower case, separated by commas, and must exist. A file name of `-` is standard input.

| Exit status | When |
|---|---|
| 0 | No error was reported. Warnings do not change the status. |
| 1 | At least one `E` finding was reported, after `--select` and `--ignore`. |
| 2 | A file could not be read, or a code is unknown. The other files are still checked. |

<!-- check-docs: continue -->
```console
$ echo 'f::{x}' | kgpy --lint - ; echo $?
0 errors, 0 warnings in 1 file
0
$ echo '1+' | kgpy --lint - ; echo $?
<stdin>:1:2: E001 missing right operand for +
1 error, 0 warnings in 1 file
1
$ echo 'my_var::5' | kgpy --lint - --select W010 ; echo $?
0 errors, 0 warnings in 1 file
0
$ echo '1+' | kgpy --lint - --ignore E001 ; echo $?
0 errors, 0 warnings in 1 file
0
$ kgpy --lint nothing.kg demo.kg --select W001 ; echo $?
nothing.kg: cannot read: No such file or directory
demo.kg:1:1: W001 'my_var' reads as my _ var: _ is the Drop operator, names cannot contain underscores
0 errors, 1 warning in 1 file
2
$ echo 'a' | kgpy --lint - --select W999 ; echo $?
kgpy --lint: unknown lint code 'W999'; codes are W001, W002, W003, W010, W011, W012, W013, W014, W015, W016, W017, W018, W019, W020, E001, E002
2
```

The `--select W001` example reads `demo.kg`, written by the first example on this page.

## Suppressing a finding

A comment on a line removes a code on that line, and only that line. The comment is `:"lint:ignore CODE"`, and several codes may follow, separated by spaces or commas. A comment without a code does nothing.

```console
$ cat > ignore.kg <<'EOF'
> a::1
> [a a]
> [a a] :"lint:ignore W012"
> EOF
$ kgpy --lint ignore.kg
ignore.kg:2:2: W012 'a' in a list is the symbol :a, not the value of a
ignore.kg:2:4: W012 'a' in a list is the symbol :a, not the value of a
0 errors, 2 warnings in 1 file
```

The third line is clean. A code can also be left out for the whole run with `--ignore`.

## Codes

Each example below is the whole file, read from standard input.

### W001: an underscore between name characters

`_` is the Drop operator, so `my_var` is `my _ var`. Write `myVar` or `my.var`, or put spaces around `_` when Drop is meant. A name that is `x`, `y` or `z` next to an underscore (`x_a`) is not reported, because that is how Drop is usually written.

```console
$ echo 'my_var::5' | kgpy --lint -
<stdin>:1:1: W001 'my_var' reads as my _ var: _ is the Drop operator, names cannot contain underscores
0 errors, 1 warning in 1 file
```

### W002: a name used before it is defined, or never defined

An undefined name evaluates to its own symbol, so a misspelled name does not fail where it is written. A use inside a function body may come before a later top-level definition, since the function runs later.

```console
$ printf 'b::a\na::1\n' | kgpy --lint -
<stdin>:1:4: W002 'a' is used before its definition (defined at line 2)
0 errors, 1 warning in 1 file
$ echo 'g::{zz}' | kgpy --lint -
<stdin>:1:5: W002 'zz' is never defined in this file
0 errors, 1 warning in 1 file
$ printf 'f::{a}\na::1\n' | kgpy --lint -
0 errors, 0 warnings in 1 file
```

See [How names are resolved](#how-names-are-resolved) for what counts as defined.

### W003: a declared local that is never used

```console
$ echo 'f::{[a;b];a::1;a}' | kgpy --lint -
<stdin>:1:8: W003 local 'b' is declared but never used
0 errors, 1 warning in 1 file
```

### W010: a leading minus before a number and an operator

`-1+2` is `-(1+2)`, which is -3. A minus fuses with a number only inside `[...]`. Write `(-1)+2` to negate only the number. A leading minus before `*` or `%` is not reported, since the sign commutes with those.

```console
$ echo '-1+2' | kgpy --lint -
<stdin>:1:1: W010 a leading - negates everything to its right: -1+... is -(1+...)
0 errors, 1 warning in 1 file
```

### W011: a function that uses `y` but not `x`, called with too few arguments

A function takes as many arguments as the highest of `x`, `y` and `z` it uses, so `{y}` takes two. A call with one argument does not give a value. The rule fires only when the file shows such a call. A function that is never called short is not reported, because callbacks with a fixed `(x;y)` signature that ignore `x` are correct.

```console
$ printf 'f::{y}\nf(5)\n' | kgpy --lint -
<stdin>:1:4: W011 function uses y but not x: the call at line 2 passes 1 argument and gets a projection, not a value
0 errors, 1 warning in 1 file
```

### W012: a bare name in a data list

A name in `[a b]` is the symbol `:a`, not the value of `a`. Use `[;a;b]` to list values. Not reported: `x`, `y`, `z`, the keys of a dictionary literal, and lists that the language reads as names (the left side of a destructuring, the argument of `.pyc`, `f:>[w b]`).

```console
$ printf 'a::1\n[a 2]\n' | kgpy --lint -
<stdin>:2:2: W012 'a' in a list is the symbol :a, not the value of a
0 errors, 1 warning in 1 file
```

### W013: a conditional in the else branch of a conditional

`:[c;a;:[d;b;e]]` can be written with the chain form `:[c;a:|d;b;e]`, which the [operator reference](operators.md#else-if-chains) describes.

```console
$ echo ':[1;"a";:[0;"b";"c"]]' | kgpy --lint -
<stdin>:1:9: W013 :[ in the else branch of :[ can be written as the :| chain
0 errors, 1 warning in 1 file
```

### W014: whitespace before an argument list or an adverb

`f (1)` is not a call: the parenthesised part is dropped. `{x} '[1 2]` is a syntax error. The same position is also an `E001` finding, so a file with either construct exits with status 1.

```console
$ printf 'f::{x}\nf (1)\n' | kgpy --lint -
<stdin>:2:3: E001 unexpected '(' after the end of the expression; the rest of the input is not evaluated
<stdin>:2:3: W014 whitespace between f and (: this is not a call, the parenthesised part is dropped
1 error, 1 warning in 1 file
```

```console
$ echo "{x} '[1 2]" | kgpy --lint -
<stdin>:1:5: E001 adverb ' must directly follow its verb
<stdin>:1:5: W014 whitespace between } and the adverb ': an adverb must follow its verb directly
1 error, 1 warning in 1 file
```

### W015: a newline right after a comment

Inside a function body, a conditional, an argument list or an evaluated list, the newline after a comment acts as `;` at strict level 0. At level 1 and 2 it is an ordinary newline. The finding is placed on the comment, so a `lint:ignore` goes in that comment. Not reported at the top level, where a newline is a separator anyway.

```console
$ printf 'f::{x :"c"\n+2}\n' | kgpy --lint -
<stdin>:1:7: W015 newline after a comment acts as ';' inside a function body
0 errors, 1 warning in 1 file
```

### W016: a colon before a number

`:123` is the number 123. The colon is ignored.

```console
$ echo ':123' | kgpy --lint -
<stdin>:1:1: W016 ':123' is the number 123: the colon is ignored
0 errors, 1 warning in 1 file
```

### W017: a `;`, bracket or operator inside a data list

In `[1;2]` the `;` is the string `";"`, and in `[1 + 2]` the `+` is an operator value. Separate data with spaces; use `[;1;2]` for values that are computed. One finding is reported for each list, at its first offending token.

```console
$ echo '[1;2]' | kgpy --lint -
<stdin>:1:3: W017 ';' inside a list is the string ";"
0 errors, 1 warning in 1 file
$ echo '[1 + 2]' | kgpy --lint -
<stdin>:1:4: W017 operator + inside a list is an operator object, not a computation
0 errors, 1 warning in 1 file
```

### W018: a number directly followed by a name

`10c` is the number 10 followed by a call of `c`. Add a space if `c` is an infix function.

```console
$ printf 'c::{x+y}\n10c 2\n' | kgpy --lint -
<stdin>:2:1: W018 '10c' is the number 10 followed by the name c
0 errors, 1 warning in 1 file
```

### W019: assignment to `x`, `y` or `z`

Assigning to an argument of a function is legal, and is usually a mistake. A function that declares the name as a local is not reported.

```console
$ echo 'f::{x::5;x}' | kgpy --lint -
<stdin>:1:5: W019 assignment to x, which is an argument of the function
0 errors, 1 warning in 1 file
```

### W020: a two-element list joined to a dictionary

A list of two members on the left of `,` with a dictionary on the right is a dictionary entry, not a list with one more member. To append the dictionary, wrap it: `list,,dict`. See [Dictionaries](language-additions.md#dictionaries). The rule needs a list of two elements written out in the file, or a name whose every definition is such a list.

```console
$ printf 'a::[1 2]\na,:{[3 4]}\n' | kgpy --lint -
<stdin>:2:2: W020 'a', a two-element list, joined to a dictionary with , becomes a dictionary entry
0 errors, 1 warning in 1 file
```

### E001: a syntax error at strict level 1

Everything that [strict level 1](strict.md) rejects, found without running the file: input left over, a missing operand, an unterminated string, a closer with nothing open, and the like. All of a file's problems are listed; parsing resumes after each.

```console
$ echo '1+' | kgpy --lint -
<stdin>:1:2: E001 missing right operand for +
1 error, 0 warnings in 1 file
$ echo '"abc' | kgpy --lint -
<stdin>:1:1: E001 unterminated string: the closing " is missing; it runs to the end of the input
1 error, 0 warnings in 1 file
```

### E002: more arguments than the function takes

The extra arguments are dropped at strict levels 0 and 1 and are an arity error at level 2. The limit is the number of named parameters, or the highest of `x`, `y` and `z` the function uses, so `{x+z}(1;2;3)` is fine and `{x}(1;2)` is not. The callee must be known: a name bound to a function literal at the top level of the file or of a file it loads with `.l`, or a literal called directly.

```console
$ printf 'f::{x}\nf(1;2)\n' | kgpy --lint -
<stdin>:2:1: E002 f takes 1 argument but is called with 2: the extra arguments are dropped
1 error, 0 warnings in 1 file
$ printf 'f::{x+z}\nf(1;2;3)\n' | kgpy --lint -
0 errors, 0 warnings in 1 file
$ printf 'f(a;b)::{a+b}\nf(1;2;3)\n' | kgpy --lint -
<stdin>:2:1: E002 f takes 2 arguments but is called with 3: the extra arguments are dropped
1 error, 0 warnings in 1 file
```

## How names are resolved

W002, W012 and E002 use a table of what the file defines: top-level `name::...`, the destructuring targets, the locals and parameters of each function, the names the system defines, and the names that the files it loads define. A file loaded with `.l("file")`, with a literal argument, is found as `.l` finds it (`file` and `file.kg` in each `KLONGPATH` directory) relative to the linted file, then the current directory, then `klongpy/lib`. It is parsed, not run.

```console
$ printf 'inc::{x+1}\n' > inc.kg
$ printf '.l("inc.kg")\ninc(1)\n' > main.kg
$ kgpy --lint main.kg
0 errors, 0 warnings in 1 file
$ echo 'inc(1)' | kgpy --lint -
<stdin>:1:1: W002 'inc' is never defined in this file
0 errors, 1 warning in 1 file
```

Some definitions cannot be seen without running code. A `.py` or `.pyf` import, a `.pya`, `.E`, a `.l` with a computed argument, and a `.l` whose file is not found, define names that lint does not know. W002 is switched off for such a file, and the other rules still run:

```console
$ printf '.pyf("math";"sqrt")\nsqrt(2)\n' | kgpy --lint -
0 errors, 0 warnings in 1 file
```

Test files that call a helper defined by the harness (`t(...)` in the repository's own tests) are linted with `--define t`.

## What is not reported

A rule that needs run-time knowledge reports only what the text shows. W002 compares the order of the source with the order a program runs, which is right to left inside a statement and dynamic between functions, and it does not follow calls. W020 needs a two-element list written out. E002 knows only functions bound to a literal. There are no type checks, no formatting rules and no automatic fixes.

## From Python

```python
from klongpy.lint import lint_source

for d in lint_source("a::1\n[a 2]\n", "demo.kg"):
    print(d.code, d.severity, d.line, d.col, d.message, d.hint)
```

`lint_source(text, name, *, select, ignore, define, base_dir, resolve)` returns `Diagnostic` objects in order of position. `str(d)` is the line the command prints. `resolve=False` never reads other files. `lint_file(path, ...)` reads a file.

See also: [Strict Mode](strict.md), [Errors](errors.md).
