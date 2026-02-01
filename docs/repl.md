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
| `]h topic` | Show help text for an operator or topic. |
| `]! cmd` | Run a shell command. |
| `]a topic` | Search help topics (currently a no-op). |
| `]i dir` | List `.kg` files in a directory. Defaults to `KLONGPATH`. |
| `]l file` | Load and execute a Klong source file. |
| `]T expr` | Time the evaluation of an expression. Use `]T:n` to repeat `n` times. |
| `]q` | Exit the REPL. |

Press `Ctrl-D` to quit as well.

### CLI options

The `kgpy` executable supports a few useful flags:

- `-e EXPR` Evaluate an expression and exit.
- `-l FILE` Load a file into the REPL on startup.
- `-s PORT` Start the IPC server (e.g., `kgpy -s 8888`).
- `-t FILE` Run a test file (non-interactive).
- `-v` Verbose output.
