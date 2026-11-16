# Errors

KlongPy reports a failed program with the position of the failure, the source line and a caret under it. This page lists the kinds of error, shows how to read one, and gives the command-line flags and exit codes.

## Reading an error

```
demo.kg:5:4: range error: index 5 out of range for list of 2
    a@5}
     ^
```

- The first line is `name:line:column: kind error: message`. The column counts from 1. The name is the file path for a file run, `<expr>` for `-e`, and is left out for text typed at the REPL, where the first line starts with `line:column`.
- The second line is the source line (tabs shown as single spaces, clipped to 120 columns around the caret).
- The third line puts carets under the part of the program that failed. An operator error points at the operator; a failed call points at the whole call.
- A `hint:` line follows when there is a piece of advice.

Without a position (a file that cannot be opened, for example) only the first line is printed, without the name and numbers:

```
system error: cannot open nope.kg: No such file or directory
```

The line number is the line inside the text that was run. In a function loaded with `.l`, it is the line in that file, whatever file the call came from:

```
$ cat demo.kg
total::0

scale::{[a];
  a::x;
  a@5}

scale([1 2])
$ kgpy demo.kg
demo.kg:5:4: range error: index 5 out of range for list of 2
    a@5}
     ^
```

At the REPL each entry is its own text, so in a multi-line entry the line is the line of the entry. This entry is typed over three prompts; here it is piped in, which gives the same result:

```
$ printf ':[1=1;\n2+"b";\n3]\n' | kgpy >/dev/null
2:2: type error: cannot apply + (Plus) to integer and string
  2+"b";
   ^
```

## Kinds of error

Every error KlongPy raises on purpose is a `KlongError`. The `kind` is the short name shown in the first line and is stable; the message text can change.

| Class | Kind | Raised for |
|---|---|---|
| `KlongSyntaxError` | `syntax` | text the lexer or parser rejects, for example `(1]` |
| `KlongNameError` | `name` | an undefined function, or an undefined variable in strict mode |
| `KlongArityError` | `arity` | the wrong number of arguments |
| `KlongTypeError` | `type` | an operand of the wrong kind, for example `1+"a"` |
| `KlongShapeError` | `shape` | lengths or shapes that do not agree, for example `[1 2 3]+[1 2]` |
| `KlongRangeError` | `range` | an index out of range, or a value outside an operator's domain |
| `KlongSystemError` | `system` | a system function failed for an outside reason (missing file, refused connection) |
| `KlongPythonError` | `python` | an exception inside Python code called through `.py`, `.pyf` or `.pyc` |
| `KlongInternalError` | `internal` | any other exception inside an operator; this is a bug in KlongPy |
| `KlongMemoryError` | `memory` | an allocation failed |
| `KlongInterrupt` | `interrupt` | evaluation stopped by Ctrl-C |
| `KlongStaticError` | `static` | a program outside the static compile profile |
| `KlongRemoteError` | `remote` | an IPC peer answered with an error KlongPy does not know |

`KlongException` is another name for `KlongError`. From Python:

```python
from klongpy import KlongInterpreter
from klongpy.errors import KlongError

klong = KlongInterpreter()
try:
    klong('[1 2 3]+[1 2]')
except KlongError as e:
    print(e.kind)      # shape
    print(e.message)   # cannot apply + (Plus) to list of 3 and list of 2
    print(e.render())  # the multi-line form shown above
```

`str(e)` is the message alone. `e.span` is a `(start, end)` pair of offsets into `e.source.text`; both are `None` when the error has no position. When an operator fails because of a Python exception, that exception is the error's `__cause__`.

## Exceptions that are not Klong errors

A Python exception that is not a `KlongError` is a bug, in KlongPy or in host code. It is shown on one line, with how to get the rest:

```
internal error: AttributeError: 'int' object has no attribute 'channel_dir'
  hint: run with -v for the Python traceback
```

## The command line

| Mode | Value results | Errors | Exit status |
|---|---|---|---|
| `kgpy file.kg` | nothing printed unless the program prints | rendered to stderr; the run stops at the first error | 0, or 1 on an error; `.x(0)` exits with 0 and `.x(n)` for any other value with 1 |
| `kgpy -e EXPR` | the result to stdout | rendered to stderr | 0, or 1 on an error |
| `kgpy -t file` | the lines are echoed to stdout | each failing line rendered to stderr; the run goes on | 1 if any line failed, else 0; the REPL is not started |
| `kgpy -l file` | | a failing load is rendered to stderr and the process exits, without the REPL | 1 |
| REPL on a terminal | results to stdout | rendered to stderr; the loop goes on | 0 at exit; `]q` gives 0 |
| REPL on piped stdin | results to stdout | rendered to stderr; the loop goes on | 1 if any entry failed, else 0 |

A file that cannot be read is a `system` error and exits 1. Ctrl-C ends a file, `-e` or `-t` run with `interrupted` on stderr and status 130.

Piped input is the REPL reading from a pipe, so a failed entry does not stop the run; the status reports that something failed:

```
$ printf '1+2\n1+"a"\n3*4\n' | kgpy 2>err.txt >/dev/null; echo $?
1
$ cat err.txt
1:2: type error: cannot apply + (Plus) to integer and string
  1+"a"
   ^
```

### Traceback

No mode prints a Python traceback unless asked. `-v` (or `--traceback`, which does nothing else) adds the traceback after the rendered error, including the exception that caused it:

```
$ kgpy --traceback -e '1+"a"'
<expr>:1:2: type error: cannot apply + (Plus) to integer and string
  1+"a"
   ^
Traceback (most recent call last):
  File ".../klongpy/interpreter.py", line 1523, in eval
  ...
numpy._core._exceptions._UFuncNoLoopError: ufunc 'add' did not contain a loop ...

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  ...
klongpy.errors.KlongTypeError: cannot apply + (Plus) to integer and string
```

### Colour

The first line of an error is red, and only on a terminal. Nothing is coloured when the stream is a file or a pipe, or when the `NO_COLOR` environment variable is set to a non-empty value.

## Warnings

A construct that KlongPy still accepts for compatibility, but that is probably a mistake, raises a Python warning of class `KlongSyntaxWarning` (or another class whose name starts with `Klong`). The REPL, file runs, `-e` and `-t` show it on stderr every time it happens, in the same layout as an error with `warning:` in place of `<kind> error:`, and the run goes on:

```
name:line:column: warning: message
  the source line
        ^^^
  hint: advice, when there is some
```

A warning without a position is the single line `warning: message`. Warnings do not change the exit status. Python code that uses the interpreter directly sees them through the standard `warnings` module. Other Python warnings (from numpy, for example) are left to Python's own handling.

### Syntax warnings

Input that the parser used to accept without a word is now reported. At the default level (strict level 0) each case below is a `KlongSyntaxWarning` and the program runs as it always did, so existing programs keep their results. The warning text adds what happens to the input. Every warning has a `code`:

| `code` | Input | What happens at the default level |
|---|---|---|
| `trailing-input` | `1 2 3`, `"a" "b"`, `f(1 2)` | The first expression is evaluated and the rest is dropped. In a call the extra item is passed as one more argument. |
| `stray-closer` | `1+1 )`, `x::1 }` | A closer with nothing open is ignored. In a place where an expression is expected (`1;)`) it is read as a string. |
| `missing-operand` | `f::`, `+/`, `1 f` | The operator has no operand. The statement evaluates to nothing, or fails when it is run (`1+` is also a syntax error). |
| `unterminated-string` | `"abc` | The string runs to the end of the input. |
| `unterminated-list` | `[1 2 3` | The list runs to the end of the input. For nested lists only the outermost is reported. |
| `unterminated-dict` | `:{[1 2]` | The dictionary runs to the end of the input. |
| `unterminated-comment` | `3:"x` | The comment runs to the end of the input. |

At the default level the REPL returns these values (each with a warning on stderr):

```kgpy
?> 1 2 3
1
?> 1+1 )
2
?> f::
None
?> "abc
"abc"
?> [1 2 3
[1 2 3]
?> :{[1 2]
:{[1 2]}
?> 3:"x
3
```

The same inputs are errors under `.strict(1)`: a `KlongSyntaxError` with the same message, without the sentence about what would have happened, and nothing is evaluated.

```kgpy
?> .strict(1)
1
?> 1 2 3
error: unexpected '2' after the end of the expression
?> 1+1 )
error: unexpected ')': nothing is open that it could close
?> f::
error: missing right operand for ::
?> +/
error: missing operand for +/
?> "abc
error: unterminated string: the closing " is missing
?> [1 2 3
error: unterminated list: the closing ] is missing
?> :{[1 2]
error: unterminated dictionary: the closing } is missing
?> 3:"x
error: unterminated comment: the closing " is missing
```

A few inputs are errors at every level, because they were never valid: an adverb that does not directly follow a verb (`{1 '2}`, `(f)'3`), and a dictionary literal entry that is not a `[key value]` pair (`:{;1;2}`). They fail when the text is parsed, so a function that contains one now fails when it is defined, not when it is called.

#### From Python

A warning belongs to the text, not to one run of it. `klong(text)` issues the warnings of `text` through `warnings.warn_explicit` before it evaluates anything, each time it is called, also for text that comes from the parse cache. The file name is the name of the source (`<string>` for text passed as a string). `KlongSyntaxWarning` is exported by `klongpy`. Its attributes are `message` (also `str(w)`), `code`, `span` (a `(start, end)` pair into the source text), `source`, `hint` and `render()`.

Python's default filter shows a warning once for each place it is issued from. To collect them, or to see every one, use `warnings.catch_warnings`:

```python
import warnings
from klongpy import KlongInterpreter, KlongSyntaxWarning

klong = KlongInterpreter()

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    klong("1 2 3")                      # evaluates to 1
for w in caught:
    print(w.message.code)               # trailing-input
    print(w.message.render())
```

To make them errors, turn the category into an `error` filter. The warning is then raised before anything in the text runs, and every time the text is run:

```python
warnings.simplefilter("error", KlongSyntaxWarning)
try:
    klong("1+1 )")
except KlongSyntaxWarning as w:
    print(w.code)                       # stray-closer
```

`warnings.simplefilter("error")`, `python -W error` and `pytest -W error::klongpy.KlongSyntaxWarning` work as well. `python -W error::klongpy.KlongSyntaxWarning` does not, and neither does `PYTHONWARNINGS=error::klongpy.KlongSyntaxWarning`. Python resolves the category name while it starts, before `klongpy` can be imported (importing it needs numpy), so it prints `Invalid -W option ignored: invalid module name: 'klongpy'` and the filter is not installed. Install the filter from code, as above, or use `-W error` for all warnings.

Only the default (token) parser warns. The legacy parser (`KLONGPY_PARSER=legacy`) accepts the same input silently at every level.

## Interrupt

In the REPL, Ctrl-C during a long evaluation stops it, prints `interrupted` and returns to the prompt. Python code stops at its next step; one long call into numpy finishes first. If the evaluation has not stopped after two seconds the REPL says so, and a second Ctrl-C quits. Ctrl-C at the prompt only abandons the line.

## See also

- [Conformance Suite](conformance.md): the error kinds above are the ones the implementation-neutral suite checks, by kind and not by message.
- [Strict Mode](strict.md) and [Lint](lint.md): turning silent mistakes into errors or reports.
- [Differences from Klong](differences.md) and [Roadmap](roadmap.md) for known limits.
