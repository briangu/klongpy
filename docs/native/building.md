# Building and Running

This page covers building the native Klong runtime from source on a desktop computer, running the `kg` program (a REPL, `-e`, and files), the build options, and running the shared conformance suite against it. It is for anyone who wants to try the native runtime before putting it on a board. For what the runtime is, start with the [overview](index.md).

Conditions for the measured figures: an Apple M2 Ultra desktop, 2026-11-16, `cargo build --release`. Paths are relative to the root of the KlongPy repository.

## Prerequisites

You need a Rust toolchain with `cargo`. The host build has been run with a nightly `rustc` 1.90 and also with stable 1.88. The runtime has one dependency, the `libm` crate, which `cargo` fetches. You do not need Python for the build. Python is needed only for the scripts that run the conformance suite and talk to a board, and for the tests that connect the runtime to KlongPy.

## Build

```sh
cd native
cargo build --release
```

This builds the whole workspace, and it produces two programs in `native/target/release/`:

| Program | What it is |
|---|---|
| `kg` | the runtime on the desktop: a REPL, `-e EXPR`, and running a file |
| `klong-sim` | a simulated board with the serial console, see [Serial Console and Upload](console.md) |

The examples on this page, and on the other pages of this section, use `kg` and `klong-sim` by name. Put the release directory on your `PATH` for the shell you work in:

```sh
export PATH="$PWD/native/target/release:$PATH"
```

(run from the root of the repository, after `cd ..` if you are still in `native/`).

<!-- check-docs: requires-file native/target/release/kg; path native/target/release -->
```console
$ kg -e '1+1'
2
$ kg -e '+/!100'
4950
```

## Running

`kg` takes one of these forms.

| Form | What it does |
|---|---|
| `kg` | read-eval-print loop on standard input |
| `kg -e EXPR` | evaluate `EXPR`, print its value |
| `kg FILE [ARG ...]` | run a file; the arguments are in `.a` |
| `kg --dump-ast FILE` | print the canonical parse tree of a file and exit |
| `kg --conformance` | the runner protocol of the conformance suite: the program is read from standard input |

`kg --help` (or `-h`) prints the forms and every option and exits with status 0; `kg --version` prints the version. An option that `kg` does not know prints the same text on standard error and exits with status 2. The output:

<!-- check-docs: requires-file native/target/release/kg; path native/target/release -->
```console
$ kg --help
usage: kg [OPTIONS] [FILE [ARG ...] | -e EXPR | --conformance | --dump-ast FILE]

Without FILE or -e: a read-eval-print loop on standard input.
  FILE [ARG ...]        run a file; the arguments are in .a
  -e EXPR               evaluate EXPR and print its value
  --conformance         the runner protocol of conformance/README.md (program on stdin)
  --dump-ast FILE       print the canonical parse tree of FILE and exit

Options (before FILE or -e; sizes accept the suffixes K and M):
  --strict N            strict level 0, 1 or 2
  --heap-limit BYTES    most the program may allocate (default: no limit)
  --max-depth N         most nested evaluations before a range error (default 512)
  --steps N             stop after about N calls and adverb iterations (interrupt)
  --stack BYTES         native stack of the evaluating thread (default 64M)
  --stack-budget BYTES  set the limits as a board with a task stack of this size would
  --max-nest N          most levels of nested lists (default 128)
  --stats               print peak heap, nesting and stack figures on standard error
  --bench N             with -e: best time of N runs on standard error, in nanoseconds
  --memfs               the file functions work on an empty file system in memory
  --fail-every N        make every Nth large allocation fail (out-of-memory testing)
  -h, --help            print this text and exit
  --version             print the version and exit
$ kg --version
kg 2.0.0-rc.1
```

`klong-sim` has the same two options:

<!-- check-docs: requires-file native/target/release/klong-sim; path native/target/release -->
```console
$ klong-sim --help
usage: klong-sim [OPTIONS]

A simulated board: the real serial console on standard input and output.
  --pty                 serve a pseudo-terminal instead; the first output line is `pty: DEVICE`
  --check-complete      read one entry per line as hex digits, answer complete, incomplete or error
  --dumb                no echo, no editing
  --crlf                end output lines with CR LF
  --no-fs               a board without a file system
  --fs-bytes N          size of the RAM file system (default 8M)
  --heap-limit N        heap budget of the program (default: none)
  --stack-budget N      set the limits as a board with a task stack of N bytes would
  --stack N             native stack of the evaluating thread (default 64M)
  --strict N            strict level 0, 1 or 2
  --max-line N          longest input line
  --max-entry N         longest multi-line entry
  --max-run N           longest program sent with ^A run
  --max-file N          longest file sent with ^A put or ^A hex
  --timeout-ms N        a transfer with no byte for this long is abandoned
  -h, --help            print this text and exit
  --version             print the version and exit
Sizes take the suffixes K and M. End of input ends the program.
```

### `-e`

`-e` evaluates an expression and prints its value in the form the REPL uses, so a string is shown with its quotes. Output that the program writes with `.p`, `.d` and `.w` comes first. An error is printed to standard error, with the position and the source line, and the exit status is 1.

<!-- check-docs: requires-file native/target/release/kg; path native/target/release -->
```console
$ kg -e '.p("sum of squares");+/{x*x}'"'"'!10'
sum of squares
285
$ kg -e '1+"a"'
1:2: type error: cannot apply + (Plus) to integer and string
  1+"a"
   ^
$ kg -e '1+"a"' 2>/dev/null; echo $?
1
```

### The REPL

With no argument `kg` reads one line at a time from standard input and evaluates each line as a program. The prompt is two blanks. A value is printed after the prompt, an error goes to standard error and the loop continues. The line `]q` or the end of input leaves.

There is no line editing, no history and no continuation: a definition has to fit on one line. (The [serial console](console.md) of a board has all three.) While the REPL waits for a line it still runs timers and serves IPC connections, so a `kg` that was told `.srv(...)` is a working server.

<!-- check-docs: requires-file native/target/release/kg; path native/target/release -->
```console
$ printf 'x::5\nx*x\n"abc"\n1%%0\n]q\n' | kg | grep -v '^ *$'
  5
  25
  "abc"
  :undefined
```

The two leading blanks are the prompt. The last prompt, before `]q`, is removed by the `grep`.

### Running a file

`kg FILE` runs the file and prints only what the program writes. It does not print the value of the last statement. The remaining arguments are available as the list of strings `.a`. After the file has run, `kg` keeps going while there is something to wait for: a timer, a server that listens, a queued call. It exits when there is not.

<!-- check-docs: requires-file native/target/release/kg; path native/target/release -->
```console
$ cat > hello.kg <<'EOF'
> greet::{.p("hello, ",x)}
> greet'.a
> EOF
$ kg hello.kg world klong
hello, world
hello, klong
```

An error stops the file, is printed in the same form as above, and the status is 1. The line and column are those of the file.

### Strict levels

`--strict N` (0, 1 or 2) sets the [strict level](../strict.md) before the program is parsed, as `.strict(N)` does.

<!-- check-docs: requires-file native/target/release/kg; path native/target/release -->
```console
$ kg -e 'zz'
:zz
$ kg --strict 2 -e 'zz'
1:1: name error: undefined: zz
  zz
  ^^
```

## Options

Options go before `FILE` or `-e`. Sizes accept the suffixes `K` and `M` (1024 and 1048576).

| Option | Meaning |
|---|---|
| `--strict N` | strict level 0, 1 or 2 |
| `--heap-limit BYTES` | the program may allocate at most this much above the interpreter's own state; going over is a `memory` error. The default is no limit. 16 KiB are held back so that the error can be reported |
| `--max-depth N` | the most nested evaluations (one Klong call is two or three) before a `range` error. The default is 512 |
| `--steps N` | stop after about `N` function calls and adverb iterations with an `interrupt` error (a guard against a runaway loop) |
| `--stack BYTES` | the native stack of the thread that evaluates. The default is 64M. Three quarters of it can be used before a `range` error |
| `--stack-budget BYTES` | configure the limits as a board whose task has a stack of this size would, see [memory and stack](#memory-and-stack) |
| `--max-nest N` | the most levels of nested lists. The default is 128 |
| `--stats` | after the run, print the peak heap, the deepest nesting and the stack used on standard error |
| `--bench N` | with `-e`: time `N` runs of the expression and print the best on standard error, in nanoseconds |
| `--memfs` | the file functions work on an empty file system in memory, not on the real files (for fuzzing) |
| `--fail-every N` | make every `N`th large allocation fail, to test out-of-memory handling |
| `-h`, `--help` | print the usage and exit with status 0 |
| `--version` | print the version and exit |

`.l` looks for a file in the directories of the environment variable `KLONGPATH` (default `.:lib`, colon separated), then in the library built into the program, and last in the directory of the file that is being run or loaded.

## The embedded library

The twelve library files of `klongpy/lib` (`csv`, `edt`, `eigenv`, `help`, `huffman`, `math`, `nstat`, `print`, `set`, `spline`, `time`, `util`) are part of the program as text, and `.l` loads them by name. A file of the same name on the load path is used in preference. In a build with IPC the management library `mgmt` is carried too, see [Management Node](management-node.md).

<!-- check-docs: requires-file native/target/release/kg; path native/target/release -->
```console
$ kg -e '.l("nstat");.p(nstat.mu([1 2 3 4]));.p(nstat.sd([1 2 3 4]))'
2.5
1.1180339889579018
"1.1180339889579018"
$ kg -e '.l("nosuch")' 2>&1 | sed -n '1s/ (looked for.*//p'
1:1: system error: file does not exist: nosuch
```

(The message goes on to list the places `.l` looked in; the example cuts it after the name.)

(The third line is the value of the expression, which `-e` prints.) Loading costs heap, because parsed code takes 20 to 30 bytes per byte of source, but a load reads one statement at a time, so the peak is about what stays: `.l("csv")` peaks at about 35 KB, `.l("math")` at about 141 KB and `.l("nstat")`, which loads `math`, at about 281 KB (a whole-file load needed 50 KiB, 495 KiB and 1.04 MiB): the functions of a loaded file are kept as the place of their text and parsed when called. `.mem()` returns `[in use, peak, limit]` in bytes. The interpreter itself is about 15 KiB after start-up, and `nstat` takes the peak over 250 KB. The exact numbers change a little from run to run and between builds, so the transcript tests them:

<!-- check-docs: requires-file native/target/release/kg; path native/target/release -->
```console
$ kg -e '(.mem()@0)<20000'
1
$ kg -e '.l("nstat");(.mem()@1)>250000'
1
```

## Feature flags

The workspace has these cargo features. They are chosen when you build, for example `cargo build --release --features int32,float32` in `native/klong-cli`.

| Feature | Crates | Effect |
|---|---|---|
| `int32` | `klong-cli`, `klong-sim`, `klong-board`, `klong-ffi` | integers are 32 bits wide and wrap at that width |
| `float32` | the same | floats are 32 bits wide and print with the digits of a 32-bit float |
| `files` | `klong-rt`, `klong-board`, `klong-ffi` (default on) | channels, `.l` of files and the file functions; needs a file system from the platform |
| `lib` | the same (default on) | the twelve library files as text in the program (about 42 KiB) |
| `timers` | `klong-rt`, `klong-ffi` (default on in `klong-ffi`) | `.timer` and `.timerc` |
| `ipc` | `klong-rt` | `.cli .clid .clic .clit .srv .async` and the `.srv.*` hooks, and the `mgmt` library |
| `compress` | `klong-rt`, `klong-ipc` | compress large messages when sending (decompression is always there) |

`klong` and `klong-sim` are built with `timers`, `ipc` and `compress` always on, and with `files` and `lib` on. The two width features are the ones you switch: `klong-cli` and `klong-sim` pass them down, so a narrow host build is

```sh
cd native
cargo build --release -p klong-cli --features int32,float32 --target-dir target/narrow
```

A narrow build wraps integers at 32 bits and computes floats in 32 bits:

<!-- check-docs: requires-file native/target/release/kg; path native/target/release -->
```console
$ kg -e '2147483647+1'
2147483648
$ kg -e '9223372036854775807+1'
-9223372036854775808
$ kg -e '.ver()'
:{["version" "2.0.0-rc.1"] ["int" 64] ["float" 64] ["features" ["files" "lib" "timers" "ipc"]] ["platform" "host"]}
```

The default build is 64 bits wide, as the first two lines show: `2147483647+1` is an ordinary number and `9223372036854775807+1` wraps. In the narrow build `.ver()` reports `["int" 32] ["float" 32]`, `2147483647+1` is `-2147483648`, and `1%3` prints `0.33333334`. (This paragraph is not a checked transcript: only the default build is run for the examples.)

Two integers within the build width wrap at it, with no error. As in KlongPy, an integer beyond 64 bits is exact: a long literal, a scalar `^` of two integers, and an operand beyond 64 bits in `+ - * % :% < > = | &`. The native runtime does it with the `num-bigint` crate behind the cargo feature `bigint`, which is on for `klong-cli` (the `kg` binary) and the Python extension. A result that fits goes back to an ordinary integer; ordinary integers stay machine integers, and the feature costs `fib(24)` about 2 percent. A board build leaves the feature off (it adds 22 KB of flash on `thumbv6m-none-eabi` and `thumbv7em-none-eabihf` and 32 KB on `riscv32imac-unknown-none-elf`), and there an integer literal that does not fit the build is a `syntax` error (`malformed number`); `klong-board` and `klong-ffi` have the feature `bigint` for a build that has the flash. See [How it relates to KlongPy](index.md#how-it-relates-to-klongpy).

`strings` and `dicts` exist as features of `klong-core` but do not remove anything yet.

## Memory and stack

The runtime recurses on the native stack of the thread that evaluates, and it allocates from one heap. Both are bounded so that running out is an error at the prompt and not a crash.

* **Heap.** `--heap-limit BYTES` gives the program a budget. A program that goes over it fails with `memory error: out of memory`, and the interpreter stays usable.
* **Recursion.** Without `--stack-budget` the limit is `--max-depth`, 512 nested evaluations. A Klong call is two or three evaluations, so a function can call itself about 160 times deep with the defaults. `--max-depth` and `--stack` raise it.
* **Stack budget.** `--stack-budget BYTES` sets the limits the way a board with a task stack of that size would: recursion is checked against the real stack address and a quarter of the budget is kept for leaf work. This is how you check on the desktop whether a program fits a board.

<!-- check-docs: requires-file native/target/release/kg; path native/target/release -->
```console
$ kg --heap-limit 100K -e '!1000000'
1:1: memory error: out of memory
  !1000000
  ^^^^^^^^
$ kg -e 'f::{:[x<1;0;1+f(x-1)]};f(150)'
150
$ kg -e 'f::{:[x<1;0;1+f(x-1)]};f(200)'
1:17: range error: expression nested too deeply
  f::{:[x<1;0;1+f(x-1)]};f(200)
                  ^^^
$ kg --max-depth 100000 -e 'f::{:[x<1;0;1+f(x-1)]};f(5000)'
5000
$ kg --stack-budget 64K -e 'f::{:[x<1;0;1+f(x-1)]};f(20)'
20
$ kg --stack-budget 64K -e 'f::{:[x<1;0;1+f(x-1)]};f(40)'
1:17: range error: expression nested too deeply
  f::{:[x<1;0;1+f(x-1)]};f(40)
                  ^^^
  hint: the native stack budget of this build is used up
$ kg --steps 1000 -e '{1}{x+1}:~0'
1:1: interrupt error: interrupted
  {1}{x+1}:~0
  ^^^^^^^^^^^
```

The figures in the stack-budget lines are for this 64-bit host. The span of a stack-budget error marks the evaluation that crossed the limit, and which one that is depends on how much stack each step of the build uses (here the argument `x-1` of the call that overflowed); the span of a `--max-depth` error does not depend on the build. On a 32-bit board a level of Klong recursion costs about 0.85 KiB: 864 bytes by the frames of the thumbv6m image, against about 0.8 to 2 KiB on the host. The stack budget is used as `bytes` less what is kept below the deepest level (a quarter by default; a board can say, `Config::for_stack_keep`: the Pico keeps 16 KiB of a 108 KiB budget), so 64 KiB gives about 57 levels and 256 KiB about 228 at the default.

With `--stats` the program reports what it used, on standard error: the peak heap, the heap in use, the limit and the number of refused allocations (`heap: peak 23174 bytes, in use 15679 bytes, limit 0, refused 0` for the program below; the heap numbers vary a little), the deepest nesting of evaluations and the native stack. The heap figures are bytes requested from the allocator and include the interpreter's own state of about 15 KiB on the desktop.

<!-- check-docs: requires-file native/target/release/kg; path native/target/release -->
```console
$ kg --stats -e '+/!1000' 2>&1 | grep -v -e '^heap' -e '^stack'
499500
depth: 3 nested evaluations at most
```

The last line it prints is the native stack, such as `stack: 768 bytes of native stack at most`. The number depends on how `kg` was built.

## Running the conformance suite

The language is checked by the cases in `conformance/cases`, which KlongPy and the native runtime both run (see the repository's `conformance/README.md` for the case format). Each case runs a program in a fresh interpreter and compares the printed value, the console output and the kind of the error. `native/scripts/conformance.py` runs the cases that the default build declares, with the `kg --conformance` protocol, one process per case, and prints counts per file and a total.

In the commands below, `python` is the interpreter you use for KlongPy, and the scripts are in the repository.

<!-- check-docs: requires-file native/target/release/kg; link native conformance -->
```console
$ python native/scripts/conformance.py | tail -1 | cut -c1-40
total                         2155  2155
```

Run it without the `tail` to see the files. The two numbers are cases passed and cases run: 2155 of 2155 on 2026-11-18, in about three seconds. `--narrow` runs the cases that a build with `int32,float32` declares, and `KLONG_NATIVE` names the binary to test:

```sh
KLONG_NATIVE=native/target/narrow/release/kg python native/scripts/conformance.py --narrow
```

That gave 1408 of 1408 on 2026-11-18 with the narrow build above. The parse trees of the golden files match too: `python conformance/run.py --trees --cmd native/target/release/kg` reported `169 of 169 trees match`. The same suite runs through the serial console of a simulated board and of a real one, see [Serial Console and Upload](console.md).

Not every case runs on every build. A case carries tags, and a build runs the cases whose tags it declares. The desktop binary declares `bigint`; a build without the feature leaves it out (it wraps), and so do all builds leave out `torch-only` and `kgpy-only` (KlongPy's own), `desktop` (Python interop, databases, web) and the `timers` and `ipc` tiers, which have no cases. The narrow build also leaves out `float`, `string-fmt` and `width64`, the cases that pin 64-bit results.

The native tests that need KlongPy (`tests/test_native_conformance.py`, `tests/test_native_ipc.py`, `tests/test_native_mgmt.py`) skip themselves when `native/target/release/kg` does not exist, so building the binary turns them on. On the machine of the conditions above 96 passed and 11 were skipped; the 11 need a PeachQ executable (a kdb+-compatible q implementation), see [IPC and Timers](ipc.md). `cargo test --release --workspace` in `native/` runs the Rust tests.

## Sizes and speed

Conditions: an Apple M2 Ultra desktop, 2026-11-16, nightly `rustc` 1.90.

**Programs on the desktop** (`ls -l`; the `release` profile, and the `min-size` profile that `cargo build --profile min-size -p klong-cli` selects: opt-level `z`, LTO, one code generation unit, `panic = "abort"`, stripped):

| Build | `kg` size |
|---|---|
| `release` | 1,566,816 bytes |
| `min-size` | 753,808 bytes |
| `min-size`, `int32,float32` | 753,824 bytes |

The width features do not change the size of a 64-bit program in any useful way. They matter on 32-bit targets, where they halve the size of a value (16 to 8 bytes) and remove the 64-bit arithmetic.

**Images for boards** (`sh native/scripts/measure.sh` builds these; the numbers here are from `cargo +stable build -p klong-board --profile min-size --target TARGET` and `native/scripts/sections.py`, bytes; `klong-board-image` is the board skeleton with the console, linked but not bootable):

| Target | Features | `.text` | `.rodata` | flash |
|---|---|---|---|---|
| `thumbv7em-none-eabihf` | `files`, `lib` | 176,788 | 72,656 | 249,464 |
| | `files`, `lib`, `int32`, `float32` | 162,564 | 72,440 | 235,024 |
| | none | 167,188 | 29,676 | 196,884 |
| `riscv32imac-unknown-none-elf` | `files`, `lib` | 218,990 | 109,092 | 328,102 |
| | `files`, `lib`, `int32`, `float32` | 205,518 | 107,984 | 313,522 |
| | none | 207,056 | 63,664 | 270,740 |

Two cargo features of `klong-core` are on for the desktop binary (`klong-cli`, and the Python extension) and off for these images. `bigint` (exact integers beyond 64 bits, with the `num-bigint` crate) adds 22.6 KB of flash on `thumbv6m-none-eabi`, 22.3 KB on `thumbv7em-none-eabihf` and 32.4 KB on `riscv32imac-unknown-none-elf`, more than a board is given for it, so the boards keep the range error (`klong-board` has the feature `bigint` for a build with the flash). `diag` (the parser's warnings and the hints of its errors, the whole-operand text of an operator's error, the description of a function value) is left out for flash too. Measured on 2026-11-18, `klong-board-image` without either: 309,092 bytes on `thumbv6m-none-eabi`, 299,168 on `thumbv7em-none-eabihf` and 395,220 on `riscv32imac-unknown-none-elf`, which is 5.7, 6.0 and 7.5 KB more than before the behaviours that KlongPy's tests found (`.fmt` with a template alone, an index vector that holds a float, and the like) and the messages.

These images have no IPC and no timers. IPC adds about 42 KB on `thumbv7em` and 59 KB on `riscv32imac`, timers 1.7 and 2.5 KB, and the `mgmt` library 12.6 KB. The firmware for the [ESP32-P4](esp32-p4.md) is 692 KB (691,328 bytes flashed), but it is built with `opt-level` 3 and carries ESP-IDF.

**Speed.** The native runtime is much faster than KlongPy on interpreter work (recursion, loops, lambdas, dictionaries) and slower on whole-array work that NumPy does in one call. [Performance](../performance.md) has the measured table. `python native/scripts/bench.py` repeats the comparison, and `kg --bench N -e PROGRAM` times one program in process. Starting the program and exiting takes about 4 ms. On the ESP32-P4 `fib(24)` takes 4.2 seconds, against about 35 ms on the desktop; see [ESP32-P4](esp32-p4.md).
