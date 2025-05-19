# REPL Reference

KlongPy comes with an interactive Read Eval Print Loop (REPL) which is helpful for experimenting with the language.

Launch the REPL using `kgpy` (installing `rlwrap` is recommended for command history and line editing):

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
| `]a topic` | Search help topics. |
| `]i dir` | List `.kg` files in a directory. Defaults to `KLONGPATH`. |
| `]l file` | Load and execute a Klong source file. |
| `]T expr` | Time the evaluation of an expression. Use `]T:n` to repeat `n` times. |
| `]q` | Exit the REPL. |

Press `Ctrl-D` can also be used to quit.
