# Release Notes

## Next release: behaviour changes

This list is the changes that can alter what an existing program does or prints. Each item gives the old behaviour, the new one and an example. The examples are run against the interpreter when the documentation is checked, so the new output shown is what it prints.

"Old" means the last release, 0.7.0. Where the old behaviour was an unrelated Python error, only its kind is given.

### Parser

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

**An adverb that does not touch its verb is a syntax error when the text is parsed.** It used to fail with a `KeyError` when it was evaluated, so a function that contained one only failed when called.

```kgpy
?> {1 '2}
error: adverb ' must directly follow its verb
```

**A dictionary literal entry that is not a `[key value]` pair is a syntax error.** It was an `IndexError` or `TypeError`.

```kgpy
?> :{;1;2}
error: each entry of a dictionary literal must be a [key value] pair
```

### Operators

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

**Several operators that crashed or looped now return a value or a Klong error.** `$[1 2 3]` recursed until Python stopped it. `|5` was a `TypeError`. `&/"abc"` and `+/"abc"` failed on a string. `[2 2]:^[]` and `5:^[]` raised a `ZeroDivisionError`. A dictionary joined with something that is not a two-member list was silently ignored.

```kgpy
?> $[1 2 3]
["1" "2" "3"]
?> |5
5
?> &/"abc"
"a"
?> +/"abc"
"abc"
?> [2 2]:^[]
error: :^: cannot fill a non-empty shape from an empty source
?> :{[1 2]},[1 2 3]
error: ,: a dictionary can only be joined with a tuple (a list of two members)
```

### System functions

**`.sys` runs a shell command and returns its exit status.** It tried to start a program named by the whole string.

```kgpy
?> .sys("exit 3")
3
```

**`.timer` checks its arguments.** The interval may be a real number such as `0.5`. A negative or non-numeric interval, or a callback that is not a function, is an error; they were a returned string or a `ValueError`. An error in a callback is logged and cancels the timer.

```kgpy
?> .timer("t";-1;{1})
error: y must be a non-negative number
```

### Errors

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

**Messages changed.** A program that matches on message text needs the new text.

```kgpy
?> 1 f
error: undefined function: f
?> 1.2.3
error: malformed number '1.2.3'
```

These were `undefined: f` and a `ValueError` from `float()`. The error carries its position, its kind and the source: see [Reading an error](errors.md#reading-an-error).

### REPL and CLI

**Errors go to stderr, with the position and a caret.** At the REPL they were printed on stdout as `Error: (args...)`. A file run or `-e` printed a Python traceback; a traceback is now printed only with `-v` or `--traceback`.

```
$ kgpy -e '1+"a"'
<expr>:1:2: type error: cannot apply + (Plus) to integer and string
  1+"a"
   ^
```

**Exit status.** `kgpy file.kg` and `-e` exit with 1 after an error. Piped stdin exits 1 if any entry failed. `-t` runs every line, reports each failing line and exits after the last, with 1 if any failed; it no longer starts the REPL afterwards. Ctrl-C ends a file run with `interrupted` and status 130. A file that cannot be opened is `system error: cannot open ...`. See [the command line](errors.md#the-command-line).

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

**Line editing is built in.** History (`~/.kgpy_history`, or `KGPY_HISTORY`) and tab completion come from `readline`, so `rlwrap` is no longer needed. See [the REPL](repl.md).

### IPC

**The wire protocol is kdb+ IPC.** `.srv` and `.cli` used pickled Python objects, which let a peer run code in your process when a message was unpickled. The default is now the kdb+ protocol, and a q client can talk to a `.srv`. The old protocol remains for a transition and is deprecated: `KLONGPY_IPC=legacy` selects it, `auto` (the server default) accepts an old client, and `KLONGPY_IPC=kdb` refuses it. Upgrade servers first, then clients. See [the protocol](ipc_capabilities.md#the-protocol).

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

**A call waits 30 seconds.** A call with no answer raises an error that contains `timed out`, closes the connection and fails every call waiting on it. Change the limit with `.clit`. See [Closing, Timeouts and `.clit`](ipc_capabilities.md#closing-timeouts-and-clit).

**Tables and functions cross the connection.** A `klongpy.db` table is sent as a kdb+ table and arrives as a table when `klongpy.db` is loaded; an index does not travel. A function literal is sent as its source text. See [the protocol](ipc_capabilities.md#the-protocol) and [How functions cross the wire](ipc_capabilities.md#how-functions-cross-the-wire).
