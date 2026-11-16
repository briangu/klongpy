# REPL Reference

KlongPy comes with an interactive Read Eval Print Loop (REPL) which is helpful for experimenting with the language.

`kgpy` is installed with the package (`pip install klongpy`); its one extra requirement, `colorama`, is a core dependency, so the `repl` extra adds nothing and is kept only so that `pip install "klongpy[repl]"` still works.

Launch the REPL using `kgpy`. Line editing, history and tab completion come from Python's `readline` module, so `rlwrap` is not needed:

```bash
$ kgpy
```

Once running you can evaluate expressions directly:

```kgpy
?> 1+1
2
```

Several system commands prefixed with `]` are available:

| Command | Description |
|---------|-------------|
| `]h topic` | Show help for an operator, adverb, system function or variable. A glyph (`]h /`), a name (`]h Over`, `]h .p`) or part of a name works. `]h` lists all topics by group and `]h all` shows every entry. |
| `]! cmd` | Run a shell command. |
| `]a topic` | List help entries whose name or title contains `topic` (for example `]a scan`). |
| `]i dir` | List `.kg` files in a directory. Defaults to `KLONGPATH`. |
| `]l file` | Load and execute a Klong source file. |
| `]T expr` | Time the evaluation of an expression. Use `]T:n` to repeat `n` times. |
| `]t file` | Append input and results to `file`. `]t` with no file stops the transcript. |
| `]q` | Exit the REPL. |

A bare `]` lists the commands. A command with a missing or bad argument prints a usage line.

Press `Ctrl-D` to quit as well.

### Multi-line entry

When a line leaves a bracket open (`(`, `[`, `{`, `:[`, `:{`, `:(`), a string unterminated, or a `:"` comment or `.comment("X")` unfinished, the REPL prompts with `.. ` and keeps reading until the entry is complete, then evaluates it as one entry:

```kgpy
?> f::{
.. x+1
.. }
:monad
?> f(2)
3
```

An empty line or Ctrl-C at the `.. ` prompt cancels the entry. Ctrl-C at the `?> ` prompt only abandons the line; Ctrl-D quits. A line that ends in an operator (`1+`) is complete and evaluated at once (it is a syntax error). `]` commands are only recognised at the `?>` prompt. A multi-line entry is one history item. Input piped into `kgpy` goes through the same loop, so a multi-line definition in a pipe runs as one entry (blank lines inside a pending entry are kept when stdin is not a terminal).

### The `it` variable

In the REPL, `it` holds the value of the last expression that printed a result:

```kgpy
?> 1+2
3
?> it*2
6
```

Errors, `]` commands and expressions with no result leave it unchanged; assigning to `it` is allowed and the next result overwrites it. Programs run from files, `-l` and `-e` never change `it`.

### Completion and history

Tab completes defined names (including system functions such as `.p` to `.pc .py ...`), `]` commands, and file names after `]l ` and inside `.l("`. History is kept in `~/.kgpy_history` (at most 1000 entries, saved on exit); set `KGPY_HISTORY` to another path, or to an empty value to disable it. Without the `readline` module (for example on Windows) the REPL works without editing, completion or history.

### Errors

A failed entry prints the position, the source line and a caret on stderr, and the REPL stays in the loop. The first line is red on a terminal (never when `NO_COLOR` is set). In a multi-line entry the line number counts inside the entry. This entry is typed over three prompts; piped in, it gives the same output:

```
$ printf ':[1=1;\n2+"b";\n3]\n' | kgpy >/dev/null
2:2: type error: cannot apply + (Plus) to integer and string
  2+"b";
   ^
```

An exception that is a bug in KlongPy prints one `internal error: Type: message` line and how to get the traceback (`-v`). Ctrl-C during a long evaluation stops it and prints `interrupted`. [Errors](errors.md) lists the kinds, the flags and the exit statuses.

### CLI options

The `kgpy` executable supports a few useful flags:

- `-e EXPR` Evaluate an expression and exit. The value goes to stdout, an error to stderr with status 1.
- `-l FILE` Load a file into the REPL on startup. If the load fails, the error is printed and the process exits with status 1 without starting the REPL.
- `-s PORT` Start the IPC server (e.g., `kgpy -s 8888`).
- `-t FILE` Run a test file (non-interactive), one line at a time. A failing line is reported and the run goes on; the process exits after the last line with status 1 if any line failed. It does not start the REPL.
- `-v` Verbose output. Errors also print the Python traceback.
- `--traceback` Print the Python traceback with an error, without the other verbose output.
- `--strict N` Start at [strict level](strict.md) 0, 1 or 2.
- `--backend NAME` and `--device DEVICE` Choose the array backend and device (see [PyTorch Backend](torch_backend.md)). `--backend` offers `torch` only where torch is installed.
- `--lint`, `--check-static`, `--emit-c` and `--dump-ast` check or translate files without running them: see [Lint](lint.md), [Static Profile](static-profile.md), [Compiling Klong to C](embedded.md) and [Conformance Suite](conformance.md).

A file run (`kgpy file.kg`) stops at the first error, prints it with the file name, line and column (`file.kg:3:6: ...`) and exits with status 1. Input piped into the REPL exits with status 1 if any entry failed.

## See also

[Getting Started](getting-started.md), [Errors](errors.md), [System Functions](system-functions.md).
