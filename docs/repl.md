# REPL Reference

KlongPy comes with an interactive Read Eval Print Loop (REPL) which is helpful for experimenting with the language.

Install the REPL extras first:

```bash
pip install "klongpy[repl]"
# or
pip install "klongpy[all]"
```

Launch the REPL using `kgpy` (installing `rlwrap` via your OS package manager is recommended for command history and line editing):

```bash
$ rlwrap kgpy
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

### CLI options

The `kgpy` executable supports a few useful flags:

- `-e EXPR` Evaluate an expression and exit.
- `-l FILE` Load a file into the REPL on startup.
- `-s PORT` Start the IPC server (e.g., `kgpy -s 8888`).
- `-t FILE` Run a test file (non-interactive).
- `-v` Verbose output.
