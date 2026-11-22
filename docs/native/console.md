# Serial Console and Upload

This page describes the console that a board running the native Klong runtime offers on its serial port: the prompt and line editor, the `^A` command mode that uploads files and runs programs, the simulated board `klong-sim` that you can try all of it on without hardware, and `native/tools/kgupload.py`, the desktop tool that speaks the protocol. It is for people who use a board, write tools for one, or port the runtime to one.

The transcripts below use the simulated board, because a serial device is not available to everyone. The console is the same code on a simulated board and on a real one. What has run on real hardware is in the [status table](index.md#status-what-has-and-has-not-been-proven).

Run the examples from the repository root with `klong-sim` on your `PATH`. The workspace build includes the diagnostic and wide-argument features used by these examples. To build the simulator by itself with those features:

```sh
cargo build --locked --release --manifest-path native/Cargo.toml \
  -p klong-sim --features klong-console/diag,klong-core/wide-args
```

The transcripts normalize the installed version to `VERSION`. [Building and Running](building.md)

## The console

A board with the console prints a banner when it starts and then a prompt:

```
Klong 0.0.1 (esp32p4, int64 float64)
klong>
```

The banner is `Klong VERSION (BOARD, int INTWIDTH float FLOATWIDTH)`, and `BOARD` is the name the port gives. You type a line, press Enter, and the board evaluates it and prints the value or the error, then a new prompt. The REPL of the desktop `kg` program has no editor. This one has an editor, history, and continuation lines.

### A first session

`klong-sim` with no option talks over its standard input and output, so a pipe stands for the serial line. The board echoes what it receives, as a real one does, so each line you send appears after the prompt, and the answer follows. Lines are ended by CR, LF or CR LF; a real terminal sends CR.

<!-- check-docs: requires-file native/target/release/klong-sim; path native/target/release -->
```console
$ printf '1+2\n+/!10\nf::{x*x}\nf(12)\n1+"a"\n' | klong-sim | sed 's/^Klong [^ ]*/Klong VERSION/'
Klong VERSION (sim, int64 float64)
klong> 1+2
3
klong> +/!10
45
klong> f::{x*x}
:monad
klong> f(12)
144
klong> 1+"a"
1:2: type error: cannot apply + (Plus) to integer and string
  1+"a"
   ^
klong>
```

The `sim` in the banner is the board's name. The `klong>` at the end is the prompt that was waiting when the input ended.

### Multi-line entry

When a line leaves a bracket, a string or a comment open, the console does not evaluate it. It prints the continuation prompt, five blanks and `>`, and takes more lines, until the entry is complete. The rule is the one that KlongPy's own REPL uses to decide when an entry is complete (`check_complete`), and it was checked against KlongPy on the shared cases and on 6000 random strings. An entry is at most 8192 bytes (`max_entry`) and one line at most 1024 (`max_line`).

<!-- check-docs: requires-file native/target/release/klong-sim; path native/target/release -->
```console
$ printf 'f::{x+\n   1}\nf(2)\n' | klong-sim | sed 's/^Klong [^ ]*/Klong VERSION/'
Klong VERSION (sim, int64 float64)
klong> f::{x+
     >    1}
:monad
klong> f(2)
3
klong>
```

### Keys

Everything is bounded, and the editor works on bytes that arrive one at a time, so it does not depend on how the bytes are grouped by the serial line.

| Key | Bytes | Does |
|---|---|---|
| Enter | CR, LF or CR LF | ends the line |
| Backspace | `0x08` or `0x7f` | deletes the character before the cursor |
| Delete | `ESC [ 3 ~` | deletes the character at the cursor |
| Left, Right | `ESC [ D`, `ESC [ C` | move the cursor |
| Home, End | `ESC [ H`, `ESC [ F` (also `ESC [ 1 ~`, `ESC [ 4 ~`, `ESC O H`, `ESC O F`) | go to the start or end of the line |
| Up, Down | `ESC [ A`, `ESC [ B` | recall earlier lines; the history holds 32 lines and at most 2048 bytes |
| Ctrl-U | `0x15` | empties the line |
| Tab | `0x09` | inserts a space |
| Ctrl-C | `0x03` | abandons the line or entry being typed; during an evaluation it stops it |
| Ctrl-D | `0x04` | ignored |
| Ctrl-A | `0x01` | as the first byte of an empty line, starts a [command](#command-mode); anywhere else it is an ordinary character |

Text is UTF-8: a multibyte character is one position of the cursor. A character is taken to be one column wide, so a line wider than the terminal moves the cursor as well as the terminal lets it.

### Ctrl-C

A program that runs for a long time is stopped with Ctrl-C. The console looks at the serial line from inside the evaluation, every 4096 steps of the interpreter (a few milliseconds), so no interrupt handler needs to know about Ctrl-C. The program ends with an `interrupt` error, the prompt comes back, and the names the program defined stay defined. On the ESP32-P4 a runaway loop was stopped this way on 2026-11-16. Here it is on the simulated board; the Ctrl-C is sent one second after the loop is started.

<!-- check-docs: requires-file native/target/release/klong-sim; path native/target/release -->
```console
$ (printf '{1}{x+1}:~0\n'; sleep 1; printf '\003'; sleep 1; printf '1+1\n') | klong-sim | sed 's/^Klong [^ ]*/Klong VERSION/'
Klong VERSION (sim, int64 float64)
klong> {1}{x+1}:~0
1:1: interrupt error: interrupted
  {1}{x+1}:~0
  ^^^^^^^^^^^
klong> 1+1
2
klong>
```

Input that arrives while a program runs is kept, in order, up to 4096 bytes, and the rest is dropped. A Ctrl-C ends the running program and discards the typed-ahead input that arrived before it; what comes after it is kept. One Ctrl-C ends one program: a second one is read at the prompt (it abandons the line being typed). It makes no difference how the bytes are split. Here the Ctrl-C comes in the same burst as the line that starts the loop, and the `2+` typed before it is discarded, while `1+1` after it is run:

<!-- check-docs: requires-file native/target/release/klong-sim; path native/target/release -->
```console
$ printf '{1}{x+1}:~0\n2+\003\n1+1\n' | klong-sim | sed 's/^Klong [^ ]*/Klong VERSION/'
Klong VERSION (sim, int64 float64)
klong> {1}{x+1}:~0
1:1: interrupt error: interrupted
  {1}{x+1}:~0
  ^^^^^^^^^^^
klong>
klong> 1+1
2
klong>
```

A Ctrl-C that arrives after the program has already finished is read at the prompt like any other. The `kg` program's own REPL has no Ctrl-C byte: on a pipe nothing can interrupt a runaway loop, and on a terminal Ctrl-C is the terminal's signal and ends the process.

While a program runs the console is not answering anything else. A program that blocks the board for longer than the watchdog allows is reset by the watchdog: on the ESP32-P4 firmware the task watchdog is 30 seconds. Whether a runaway loop survives that on a board has not been tested; see the [porting checklist](porting.md).

### Output and the line end

A board sends CR LF at the end of an output line, because a terminal on a serial line expects it. `klong-sim --crlf` does the same. Tools accept both. `.p` and the other output functions write straight to the serial line.

## Command mode

The byte `0x01` (Ctrl-A, written `^A` below) as the first byte of an empty line starts a command. Tools use it to put files on the board and to run programs without the echo and the editor getting in the way. Every reply is one line that starts with `0x01`. A human at a terminal does not need it.

A command line is ended like any line. Where a command is followed by data, the board answers `^A go` when it has accepted the command, and the sender waits for that before it sends the data. Without the wait, data of a refused command would be read as input lines.

| Command | After `^A go`, the sender sends | Reply |
|---|---|---|
| `^A ver` | nothing | `^A ok ver=VERSION int=64 float=64 features=LIST platform=NAME fs=0\|1 heap_free=N heap_used=N heap_peak=N heap_limit=N max_line=N max_run=N max_file=N` |
| `^A put PATH LENGTH CRC32` | `LENGTH` raw bytes | `^A ok LENGTH CRC32`, followed by ` warn syntax: MESSAGE at LINE:COL` for a `.kg` file that does not parse, or ` unchecked` for a file larger than `max_run` |
| `^A run LENGTH CRC32` | `LENGTH` bytes of program | the output of the program, then the value on its own line, then `^A ok value N` (the value is `N` bytes long) or `^A ok` when there is no value; or `^A err KIND: MESSAGE` |
| `^A hex PATH` | Intel HEX records, one per line | per record `^A ack N` or `^A nak N REASON`; at the end-of-file record `^A ok LENGTH CRC32` |
| any failure | | `^A err KIND: MESSAGE` |

`CRC32` is the zlib CRC-32 of the data in eight hexadecimal digits. Paths have no blanks. The reply to `put` and `hex` carries the length and CRC-32 of the file as the board stored it (for `hex` they are read back from the stored file), and the sender compares them with its own. `max_file` (16 MiB by default) limits `put` and `hex`, and `max_run` (64 KiB) limits `run`.

A transfer that gets no byte for 5 seconds is abandoned with `^A err system: transfer timed out`, the partly written file is removed and the prompt returns. Ctrl-C abandons a `hex` transfer. A file is written under the name `PATH.tmp` first and renamed at the end, so a transfer that fails leaves the old file.

`put` of a `.kg` file also parses it on a second, empty interpreter and reports a syntax error as a warning. It still stores the file. The running interpreter keeps its names.

`hex` takes records of types 00 (data), 01 (end of file), 02 and 04 (addresses); types 03 and 05 are acknowledged and ignored, and any other type is refused. Records may come in any order, and a gap reads as zero, so a binary file of any size can be sent in pieces. `N` in `ack` and `nak` counts the non-blank lines since `go`.

These are the commands on the simulated board. The `--dumb` option turns the echo and the editor off, which is how a tool sees a board. The `cat -v` shows the byte `0x01` as `^A`. The `ver` examples keep the stable fields and the simulator's base features. The full response includes any additional features enabled by the build and the heap figures. `crc32 of 1+2` is `644ffa5c`.

<!-- check-docs: requires-file native/target/release/klong-sim; path native/target/release -->
```console
$ printf '\001ver\n' | klong-sim --dumb | cat -v | grep -a ver= | cut -d' ' -f2-8 | sed 's/\(features=files,lib,timers,ipc\),[^ ]*/\1/' | sed 's/ver=[^ ]*/ver=VERSION/'
^Aok ver=VERSION int=64 float=64 features=files,lib,timers,ipc platform=sim fs=1
$ printf '\001run 3 644ffa5c\n1+2' | klong-sim --dumb | cat -v | sed 's/^Klong [^ ]*/Klong VERSION/'
Klong VERSION (sim, int64 float64)
klong> ^Ago
3
^Aok value 1
klong>
$ printf '\001run 3 00000000\n1+2' | klong-sim --dumb | cat -v | sed -n 3p
^Aerr system: check sum 644ffa5c, expected 00000000
$ printf '\001frob\n' | klong-sim --dumb | cat -v | sed -n 2p
klong> ^Aerr syntax: unknown command frob: the commands are put, hex, run, ver
```

The error kinds are those of the language: `syntax name arity type shape range system memory interrupt`. A damaged transfer is `system`. A program that is longer than `max_run` is `range`. A board without a file system answers `put` and `hex` with `err system: this board has no file system`, and `run` still works:

<!-- check-docs: requires-file native/target/release/klong-sim; path native/target/release -->
```console
$ printf '\001put a.kg 3 644ffa5c\n' | klong-sim --dumb --no-fs | cat -v | sed -n 2p
klong> ^Aerr system: this board has no file system
$ printf '\001run 1 83dcefb7\n1' | klong-sim --dumb --max-run 0 | cat -v | sed -n 2p
klong> ^Aerr range: 1 bytes is more than a program sent with run may have (0)
```

### Weak spots

* The command line has no check sum. A damaged length makes the board take too few or too many raw bytes. The CRC of the data then fails, but surplus bytes of a text file are read as input lines. `kgupload.py` retries a damaged transfer and, to get a stuck board back, stays silent for longer than the board's 5 second timeout.
* A program that prints the byte `0x01` confuses a tool that reads the output of `run`, because that byte starts a reply.
* The console has no flow control. A board that cannot receive at full speed while it writes flash needs either a receive buffer big enough for the longest write, or a sender that paces itself (`kgupload.py --chunk` and `--chunk-delay`), or `hex`, where the sender never runs more than `--window` records ahead of the acknowledgements.

## The simulated board

`klong-sim` is a board that is not there: the real console and the real runtime on a platform whose serial line is in memory, a pipe, or a pseudo-terminal. It is how the console is tested, and it lets you develop tools and programs without a board.

| Option | Meaning |
|---|---|
| (none) | the console on standard input and output; end of input ends the program |
| `--pty` | the console on a pseudo-terminal. The first line of standard output is `pty: DEVICE`; open the device as you would a serial port |
| `--dumb` | no echo, no editing, no history (scripts and tools) |
| `--crlf` | end output lines with CR LF, as a board does |
| `--no-fs` | no file system: `put` and `hex` fail |
| `--fs-bytes N` | size of the RAM file system (default 8M) |
| `--heap-limit N` | heap budget, as `kg --heap-limit` |
| `--stack-budget N` | recursion limits as a board with a task stack of `N` bytes would have (see [Building and Running](building.md#memory-and-stack)) |
| `--stack N` | native stack of the evaluating thread |
| `--strict N` | strict level 0, 1 or 2 |
| `--max-line N`, `--max-entry N`, `--max-run N`, `--max-file N` | the console limits named above |
| `--timeout-ms N` | the transfer timeout in milliseconds (default 5000) |
| `--check-complete` | a test aid: reads one entry per line as hex digits and answers `complete`, `incomplete` or `error` for each, using the console's check |

Sizes take `K` and `M`. `.x()` resets the board: a new runtime, an empty file system, and the banner again, as on a real board. The simulation has IPC and timers, which a board image may not have (`.ver()` lists the features).

<!-- check-docs: requires-file native/target/release/klong-sim; path native/target/release -->
```console
$ printf '1+1\n' | klong-sim --crlf | od -c | sed -n 3,4p
0000040    a   t   6   4   )  \r  \n   k   l   o   n   g   >       1   +
0000060    1  \r  \n   2  \r  \n   k   l   o   n   g   >
```

### `--pty` and the upload tool

With `--pty` the simulated board shows up as a serial device that any program can open. This is how the tests drive it, with the same upload tool that is used on a real board. Start it in the background, keep the device name, and talk to it.

<!-- check-docs: requires-file native/target/release/klong-sim; path native/target/release; link native -->
```console
$ klong-sim --pty > sim.txt 2>&1 & echo $! > sim.pid; sleep 1; sed -n 's/^pty: //p' sim.txt > port; test -s port && echo started
started
$ python native/tools/kgupload.py $(cat port) ver | cut -d' ' -f1-6 | sed 's/\(features=files,lib,timers,ipc\),[^ ]*/\1/' | sed 's/ver=[^ ]*/ver=VERSION/'
ver=VERSION int=64 float=64 features=files,lib,timers,ipc platform=sim fs=1
```

(The device name differs on every run, `/dev/ttys007` on the machine where this was written, so the commands read it from the file `port`.) This block and the next three share one directory and one running simulation, and the last one stops it.

## Uploading with kgupload.py

`native/tools/kgupload.py` is a single Python file that uses only the standard library. It opens a serial device (or the pseudo-terminal of a simulated board), makes the port raw, gets the board to its prompt, and speaks the command protocol above. It works on POSIX systems (it needs `termios`); on Windows it would need a serial library, and nothing was tried there.

```
kgupload.py [OPTIONS] PORT SUBCOMMAND ...
```

The options may stand before or after the subcommand (`kgupload.py PORT --chunk 100 upload f.kg` and `kgupload.py PORT upload --chunk 100 f.kg` are the same).

| Subcommand | Does |
|---|---|
| `ver` | prints the answer of `^A ver` as `key=value` words: version, widths, features, platform, free heap and the console limits |
| `upload FILE [REMOTE] [--hex \| --raw]` | stores a file on the board. `REMOTE` is the name there and defaults to the file's name. A file that is valid UTF-8 text goes with `put`; anything else goes as Intel HEX. `--hex` and `--raw` force one or the other |
| `run FILE` or `run -e TEXT` | evaluates a program on the board without storing it. It prints what the program writes and then its value |
| `reset` | evaluates `.x(0)`: the board restarts, the tool waits for the banner and prints the `ver` line |

| Option | Meaning |
|---|---|
| `--baud N` | the speed of a real serial port (default 115200, the speed of the boards' console; 0 leaves the port as it is; a pseudo-terminal has none) |
| `--timeout SECONDS` | wait this long for one answer (default 10) |
| `--run-timeout SECONDS` | wait this long for the result of `run` (default 60) |
| `--retries N` | how often a damaged transfer is tried again (default 3) |
| `--chunk BYTES`, `--chunk-delay MS` | send raw data in pieces of this size with a pause after each, for a board whose UART has no flow control (defaults 1024 and 0) |
| `--record BYTES` | data bytes per Intel HEX record (default 16, at most 255) |
| `--window N` | HEX records sent before their acknowledgements are read (default 32) |
| `--board-timeout SECONDS` | how long the board waits before it abandons a transfer (default 5); the tool stays quiet that long to get a stuck board back |
| `-v` | show the conversation with the board on standard error |

The exit status is 0 on success, 1 when the board answers with an error (the message is `error: KIND: MESSAGE` on standard error), 2 for a bad command line or a file that cannot be read, and 3 when the board does not answer or says something the tool cannot understand.

### Upload and run

<!-- check-docs: requires-file native/target/release/klong-sim; path native/target/release; link native; continue -->
```console
$ printf 'sq::{x*x}\n.p("loaded")\n' > lib.kg
$ python native/tools/kgupload.py $(cat port) upload lib.kg | sed 's/, [0-9.]* s$//'
lib.kg: 23 bytes, crc32 b76d0bc1, put
$ python native/tools/kgupload.py $(cat port) run -e '.l("lib.kg");sq(12)'
loaded
144
$ python native/tools/kgupload.py $(cat port) upload lib.kg other.kg | sed 's/, [0-9.]* s$//'
other.kg: 23 bytes, crc32 b76d0bc1, put
$ python native/tools/kgupload.py $(cat port) run lib.kg
loaded
"loaded"
```

The `sed` removes the transfer time from the line. The last command runs the file `lib.kg` from the desktop, with no copy on the board, and prints what it wrote. The value of the last statement is printed after the output, here the string that `.p` returns.

A Klong error comes back as `error: KIND: MESSAGE at LINE:COL` and status 1:

<!-- check-docs: requires-file native/target/release/klong-sim; path native/target/release; link native; continue -->
```console
$ python native/tools/kgupload.py $(cat port) run -e '1+"a"'; echo $?
error: type: cannot apply + (Plus) to integer and string at 1:2
1
```

### Binary files and the syntax warning

A file that is not UTF-8 text is sent as Intel HEX, and the answer carries the CRC-32 of what the board read back. The tests send a 1 MiB binary file this way and also with `put`.

<!-- check-docs: requires-file native/target/release/klong-sim; path native/target/release; link native; continue -->
```console
$ python -c 'import sys; sys.stdout.buffer.write(bytes(range(256)) * 4)' > blob.bin
$ python native/tools/kgupload.py $(cat port) upload blob.bin | sed 's/, [0-9.]* s$//'
blob.bin: 1024 bytes, crc32 b70b4c26, hex
$ printf 'sq::{x*x\n' > bad.kg
$ python native/tools/kgupload.py $(cat port) upload bad.kg >upload.out 2>upload.err; cat upload.err upload.out | sed 's/, [0-9.]* s$//'
kgupload: bad.kg: warn syntax: expected '}' but found end of input at 2:1
bad.kg: 9 bytes, crc32 f44f241c, put
$ python native/tools/kgupload.py $(cat port) upload nofile.kg; echo $?
kgupload: nofile.kg: No such file or directory
2
```

A `.kg` file that does not parse is stored all the same, with the warning. The files of the board are lost when it resets. On a simulated board and on the ESP32-P4 they live in RAM:

<!-- check-docs: requires-file native/target/release/klong-sim; path native/target/release; link native; continue -->
```console
$ python native/tools/kgupload.py $(cat port) reset | cut -d' ' -f1-5 | sed 's/\(features=files,lib,timers,ipc\),[^ ]*/\1/' | sed 's/ver=[^ ]*/ver=VERSION/'
ver=VERSION int=64 float=64 features=files,lib,timers,ipc platform=sim
$ python native/tools/kgupload.py $(cat port) run -e '.l("lib.kg")'; echo $?
error: system: file does not exist: lib.kg (looked for lib.kg, lib/lib.kg, then the bundled library) at 1:1
1
```

### The conversation

`-v` shows what the tool sends and what the board answers, which is the protocol above in use. The `grep` hides the `ver` with which the tool starts, whose answer holds heap figures that change.

<!-- check-docs: requires-file native/target/release/klong-sim; path native/target/release; link native; continue -->
```console
$ printf 'sq::{x*x}\n' > t.kg
$ python native/tools/kgupload.py $(cat port) -v upload t.kg 2>&1 | grep -v 'ver' | sed 's/, [0-9.]* s$//'
-> put t.kg 10 465cfac9
<- go
<- ok 10 465cfac9
t.kg: 10 bytes, crc32 465cfac9, put
$ python native/tools/kgupload.py $(cat port) -v run -e '.p("x");1+1' 2>&1 | grep -v 'ver'
-> run 11 b741af65
<- go
<- ok value 1
x
2
$ kill $(cat sim.pid)
```

### Library use

`kgupload.py` can be imported: `Board(Port.open(path))` has the methods `ver()`, `put(data, remote)`, `hex(data, remote)`, `run(text)` and `reset()`. `native/scripts/console_conformance.py` uses it that way.

## The conformance suite through the console

`native/scripts/console_conformance.py` runs every case of the conformance suite through the console, with `^A run`, and judges what comes back over the wire. For each case it sends `.x(0)` (the board resets and prints its banner, which gives a fresh interpreter), `.strict(N)` when the case has a strict tag, and then the program. It compares the value, the console output and the error kind with the case.

Without `--port` it starts simulated boards, one per job, and talks to each through its pseudo-terminal with `kgupload.py`, exactly as it talks to a serial port. With `--port` it uses that one board. Options after `--` go to `klong-sim`.

<!-- check-docs: requires-file native/target/release/klong-sim; path native/target/release; link native conformance -->
```console
$ python native/scripts/console_conformance.py | tail -1 | cut -c1-40
total                         1716  1716
```

That passed 1716 of 1716 on 2026-11-19 in about 3 seconds with 8 simulated boards. To make the simulation as small as a board, add `-- --stack-budget 64K --heap-limit 1M --crlf`; the simulated runs of the project gave 1265 of 1266 (the suite of 2026-11-16; see [why the counts differ](../conformance.md#why-the-counts-differ)) with a 64 KiB stack and 1264 with 32 KiB, because the missing cases recurse deeply and end with a `range` error. The same script ran against a real board, an ESP32-P4: 1266 of 1266 over the serial port in 1877 seconds, see the [ESP32-P4](esp32-p4.md) page.

The runner needs a `System::reset` that really resets, because without it all cases share one interpreter and some fail. It needs a banner within `--timeout`, and a `max_run` of at least the largest case (under 4 KiB). The `sysio` cases write files, and on a board without storage you leave them out with `--tags`. `--narrow` selects the cases of a build with `int32,float32`.
