# The kgpy command and the REPL

`kgpy` runs Klong files and expressions and starts an interactive Read Eval Print Loop (REPL). This page lists its commands, entry rules and options. It ships with `pip install klongpy`.

Start the REPL with `kgpy`. Line editing, history and tab completion come from Python's `readline` module, so `rlwrap` is not needed:

```bash
$ kgpy
```

Type an expression to evaluate it:

```kgpy
?> 1+1
2
```

Commands that start with `]`:

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

## Multi-line entry

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

## The `it` variable

In the REPL, `it` holds the value of the last expression that printed a result:

```kgpy
?> 1+2
3
?> it*2
6
```

Errors, `]` commands and expressions with no result leave it unchanged; assigning to `it` is allowed and the next result overwrites it. Programs run from files, `-l` and `-e` never change `it`.

## Completion and history

Tab completes defined names (including system functions such as `.p` to `.pc .py ...`), `]` commands, and file names after `]l ` and inside `.l("`. History is kept in `~/.kgpy_history` (at most 1000 entries, saved on exit); set `KGPY_HISTORY` to another path, or to an empty value to disable it. Without the `readline` module (for example on Windows) the REPL works without editing, completion or history.

## Errors

A failed entry prints the position, the source line and a caret on stderr, and the REPL stays in the loop. In a multi-line entry the line number counts inside the entry. [Errors](errors.md) shows the layout, the kinds, the flags and the exit statuses. Ctrl-C during a long evaluation stops it and prints `interrupted`.

## Options

| Option | Effect |
|---|---|
| `--version` | Print `kgpy` and the version number, and exit. |
| `-e EXPR` | Evaluate an expression and exit. The value goes to stdout, an error to stderr with status 1. The system context is the same as for a file run, so `.timer`, `.srv` and `.cli` work, but the process ends after the expression and closes what it opened: a timer does not fire and a server stops. |
| `-l FILE` | Load a file into the REPL on startup. If the load fails, the error is printed and the process exits with status 1 without starting the REPL. |
| `-s PORT` | Start the IPC server, for example `kgpy -s 8888`. |
| `-t FILE` | Run a test file one line at a time, without the REPL. A failing line is reported and the run goes on; the exit status is 1 if any line failed. |
| `-v` | Verbose: an error also prints the Python traceback. |
| `--traceback` | Print the Python traceback with an error, without the other verbose output. |
| `--strict=N` | Start at [strict level](strict.md) 0, 1 or 2. `--strict` alone is level 1. A lone `0`, `1` or `2` after a bare `--strict` is read as its level, so `--strict 2` is `--strict=2`; write `--strict=N` in scripts. |
| `--backend NAME`, `--device DEVICE` | Choose the array backend and device (see [PyTorch Backend and Autograd](torch_backend.md)). `torch` needs the torch extra (`pip install "klongpy[torch]"`); without it `--backend torch` says so. |

A file run (`kgpy file.kg`) stops at the first error, prints it with the file name, line and column (`file.kg:3:6: ...`) and exits with status 1. When the last line has run the process exits, unless one of these holds:

- an IPC server is listening (`.srv`);
- a connection opened with `.cli` or `.clid` is open;
- a web server is listening (`.web`);
- a WebSocket connection opened with `.ws` is open.

The publish and subscribe scripts in [IPC Capabilities](ipc_capabilities.md) rely on this. The run ends when none is left: the server stopped with `.srv(0)` or `.webc`, the connection closed with `.clic`, `.wsc` or by the peer. A callback may do either, and `.x` in a callback (a timer, a request or a hook) exits the process with the status `.x` gives. A [timer](timer.md) does not keep a file run up: it fires while the run is up for one of the reasons above, and a file whose only live thing is a timer ends after its last line. What the file printed is flushed before it waits and before the exit.

Ctrl-C ends a waiting run with `interrupted` on stderr and status 130. SIGTERM ends it with `terminated` and status 143. Both close the server and the connections first. Input piped into the REPL exits with status 1 if any entry failed.

## Command line

| Command | What it does |
|---|---|
| `kgpy` | Start the REPL. |
| `kgpy FILE [-- ARGS]` | Run FILE; the words after `--` are in `.os.argv`, a list of strings (`kgpy feed.kg -- 8888` gives `["8888"]`; the empty list when there are none), and the process environment is in `.os.env`. The extension is optional. Accepts the options above. |
| `kgpy lint FILE...` | Report the classic Klong traps without running the files: see [Lint](lint.md). Options `--select`, `--ignore`, `--define`, `-v`. |
| `kgpy check FILE...` | Check that the files are inside the [static profile](static-profile.md). Options `--int64`, `--streamable`, `--dump-ir`, `-v`. |
| `kgpy compile c FILE...` | Compile static-profile programs to C99: see [Compiling Klong to C](embedded.md). Options `-o`, `--int64`, `--float64`, `--line-directives`, `--prefix`, `--dump-ir`. |
| `kgpy fpga source FILE...` | Experimental. Write the HLS C++ of each `.entry`: see [Klong functions as FPGA hardware](hardware.md). Options `-o`, `--int64`, `--prefix`, `--clock-mhz`, `--lanes`, `--wrapped`, `--reassociate`, `--synth`. |
| `kgpy fpga build [FILE...]` | Experimental. Build an Alveo U250 bitstream. Options `--int64`, `--clock-mhz`, `--lanes`, `--wrapped`, `--reassociate`, `--wait`, `--force`, `--emu`. |
| `kgpy fpga status [FILE...]` | Experimental. Report the progress of the build. |
| `kgpy ast FILE` | Print the canonical parse tree of FILE: see [Conformance suite](conformance.md). |

`kgpy SUBCOMMAND --help` lists the options of one subcommand, which may come before or after the file names. A file named like a subcommand is run as `kgpy ./lint`.

## See also

[Start Here](getting-started.md), [Errors](errors.md), [System Functions](system-functions.md).
