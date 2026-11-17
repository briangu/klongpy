# Building and Running

This page covers building the native Klong runtime from source on a desktop computer, running the `klong` program (a REPL, `-e`, and files), the build options, and running the shared conformance suite against it. It is for anyone who wants to try the native runtime before putting it on a board. For what the runtime is, start with the [overview](index.md).

Every command and every output below was run on 2026-11-16 on an Apple-silicon Mac, and the transcripts are run again whenever the documentation is checked and the binary has been built. Paths are relative to the root of the KlongPy repository.

## Prerequisites

You need a Rust toolchain with `cargo`. The host build was run here with a nightly `rustc` 1.90 and also with stable 1.88. The runtime has one dependency, the `libm` crate, which `cargo` fetches. You do not need Python for the build. Python is needed only for the scripts that run the conformance suite and talk to a board, and for the tests that connect the runtime to KlongPy.

## Build

```sh
cd native
cargo build --release
```

This builds the whole workspace, and it produces two programs in `native/target/release/`:

| Program | What it is |
|---|---|
| `klong` | the runtime on the desktop: a REPL, `-e EXPR`, and running a file |
| `klong-sim` | a simulated board with the serial console, see [Serial Console and Upload](console.md) |

The examples on this page, and on the other pages of this section, use `klong` and `klong-sim` by name. Put the release directory on your `PATH` for the shell you work in:

```sh
export PATH="$PWD/native/target/release:$PATH"
```

(run from the root of the repository, after `cd ..` if you are still in `native/`).

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ klong -e '1+1'
2
$ klong -e '+/!100'
4950
```

## Running

`klong` takes one of these forms.

| Form | What it does |
|---|---|
| `klong` | read-eval-print loop on standard input |
| `klong -e EXPR` | evaluate `EXPR`, print its value |
| `klong FILE [ARG ...]` | run a file; the arguments are in `.a` |
| `klong --dump-ast FILE` | print the canonical parse tree of a file and exit |
| `klong --conformance` | the runner protocol of the conformance suite: the program is read from standard input |

`klong --help` (or `-h`) prints the forms and every option and exits with status 0; `klong --version` prints the version. An option that `klong` does not know prints the same text on standard error and exits with status 2. This is the output, and it is checked against the program:

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ klong --help
usage: klong [OPTIONS] [FILE [ARG ...] | -e EXPR | --conformance | --dump-ast FILE]

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
$ klong --version
klong 0.0.1
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

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ klong -e '.p("sum of squares");+/{x*x}'"'"'!10'
sum of squares
285
$ klong -e '1+"a"'
1:1: type error: cannot apply + (Plus) to integer and string
  1+"a"
  ^^^^^
$ klong -e '1+"a"' 2>/dev/null; echo $?
1
```

### The REPL

With no argument `klong` reads one line at a time from standard input and evaluates each line as a program. The prompt is two blanks. A value is printed after the prompt, an error goes to standard error and the loop continues. The line `]q` or the end of input leaves.

There is no line editing, no history and no continuation: a definition has to fit on one line. (The [serial console](console.md) of a board has all three.) While the REPL waits for a line it still runs timers and serves IPC connections, so a `klong` that was told `.srv(...)` is a working server.

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ printf 'x::5\nx*x\n"abc"\n1%%0\n]q\n' | klong | grep -v '^ *$'
  5
  25
  "abc"
  :undefined
```

The two leading blanks are the prompt. The last prompt, before `]q`, is removed by the `grep`.

### Running a file

`klong FILE` runs the file and prints only what the program writes. It does not print the value of the last statement. The remaining arguments are available as the list of strings `.a`. After the file has run, `klong` keeps going while there is something to wait for: a timer, a server that listens, a queued call. It exits when there is not.

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ cat > hello.kg <<'EOF'
> greet::{.p("hello, ",x)}
> greet'.a
> EOF
$ klong hello.kg world klong
hello, world
hello, klong
```

An error stops the file, is printed in the same form as above, and the status is 1. The line and column are those of the file.

### Strict levels

`--strict N` (0, 1 or 2) sets the [strict level](../strict.md) before the program is parsed, as `.strict(N)` does.

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ klong -e 'zz'
:zz
$ klong --strict 2 -e 'zz'
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

`.l` looks for a file in the directories of the environment variable `KLONGPATH` (default `.:lib`, colon separated) and, when it finds none, in the library built into the program.

## The embedded library

The twelve library files of `klongpy/lib` (`csv`, `edt`, `eigenv`, `help`, `huffman`, `math`, `nstat`, `print`, `set`, `spline`, `time`, `util`) are part of the program as text, and `.l` loads them by name. A file of the same name on the load path is used in preference. In a build with IPC the management library `mgmt` is carried too, see [Management Node](management-node.md).

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ klong -e '.l("nstat");.p(mu([1 2 3 4]));.p(sd([1 2 3 4]))'
2.5
1.1180339889579018
"1.1180339889579018"
$ klong -e '.l("nosuch")'
1:1: system error: file does not exist: nosuch
  .l("nosuch")
  ^^^^^^^^^^^^
```

(The third line is the value of the expression, which `-e` prints.) Loading costs heap, because parsed code takes 20 to 30 bytes per byte of source: `.l("csv")` needs about 50 KiB, `.l("math")` about 495 KiB and `.l("nstat")`, which loads `math`, about 1.04 MiB. `.mem()` returns `[in use, peak, limit]` in bytes. The interpreter itself is about 15 KiB after start-up, and `nstat` takes the peak over a megabyte. The exact numbers change a little from run to run and between builds, so the transcript tests them:

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ klong -e '(.mem()@0)<20000'
1
$ klong -e '.l("nstat");(.mem()@1)>1000000'
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

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ klong -e '2147483647+1'
2147483648
$ klong -e '9223372036854775807+1'
-9223372036854775808
$ klong -e '.ver()'
:{["version" "0.0.1"] ["int" 64] ["float" 64] ["features" ["files" "lib" "timers" "ipc"]] ["platform" "host"]}
```

The default build is 64 bits wide, as the first two lines show: `2147483647+1` is an ordinary number and `9223372036854775807+1` wraps. In the narrow build `.ver()` reports `["int" 32] ["float" 32]`, `2147483647+1` is `-2147483648`, and `1%3` prints `0.33333334`. (The narrow program was built and run for this page, but the checker runs only the default build, so this paragraph is not a checked transcript.)

Integers wrap at the build width, with no error and no promotion to a larger type. KlongPy wraps at 64 bits too, but it promotes an operand beyond 64 bits (a long literal) to an exact integer for `+ - * < > | &`. The native runtime has no such promotion, and an integer literal that does not fit the build is a `syntax` error (`malformed number`). See [How it relates to KlongPy](index.md#how-it-relates-to-klongpy).

`strings` and `dicts` exist as features of `klong-core` but do not remove anything yet.

## Memory and stack

The runtime recurses on the native stack of the thread that evaluates, and it allocates from one heap. Both are bounded so that running out is an error at the prompt and not a crash.

* **Heap.** `--heap-limit BYTES` gives the program a budget. A program that goes over it fails with `memory error: out of memory`, and the interpreter stays usable.
* **Recursion.** Without `--stack-budget` the limit is `--max-depth`, 512 nested evaluations. A Klong call is two or three evaluations, so a function can call itself about 160 times deep with the defaults. `--max-depth` and `--stack` raise it.
* **Stack budget.** `--stack-budget BYTES` sets the limits the way a board with a task stack of that size would: recursion is checked against the real stack address and a quarter of the budget is kept for leaf work. This is how you check on the desktop whether a program fits a board.

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ klong --heap-limit 100K -e '!1000000'
1:1: memory error: out of memory
  !1000000
  ^^^^^^^^
$ klong -e 'f::{:[x<1;0;1+f(x-1)]};f(150)'
150
$ klong -e 'f::{:[x<1;0;1+f(x-1)]};f(200)'
1:17: range error: expression nested too deeply
  f::{:[x<1;0;1+f(x-1)]};f(200)
                  ^^^
$ klong --max-depth 100000 -e 'f::{:[x<1;0;1+f(x-1)]};f(5000)'
5000
$ klong --stack-budget 64K -e 'f::{:[x<1;0;1+f(x-1)]};f(20)'
20
$ klong --stack-budget 64K -e 'f::{:[x<1;0;1+f(x-1)]};f(40)'
1:7: range error: expression nested too deeply
  f::{:[x<1;0;1+f(x-1)]};f(40)
        ^^^
  hint: the native stack budget of this build is used up
$ klong --steps 1000 -e '{1}{x+1}:~0'
1:1: interrupt error: interrupted
  {1}{x+1}:~0
  ^^^^^^^^^^^
```

The figures in the stack-budget lines are for this 64-bit host. On a 32-bit board a level of Klong recursion costs about 0.8 KiB (16 KiB of stack: about 14 levels, 64 KiB: about 60, 256 KiB: about 250, from the porting notes), against about 1.3 to 2 KiB on the host.

With `--stats` the program reports what it used, on standard error: the peak heap, the heap in use, the limit and the number of refused allocations (`heap: peak 23174 bytes, in use 15679 bytes, limit 0, refused 0` for the program below, on the machine of this page; the heap numbers vary a little), the deepest nesting of evaluations and the native stack. The heap figures are bytes requested from the allocator and include the interpreter's own state of about 15 KiB on the desktop.

<!-- check-docs: requires-file native/target/release/klong; path native/target/release -->
```console
$ klong --stats -e '+/!1000' 2>&1 | grep -v '^heap'
499500
depth: 3 nested evaluations at most
stack: 1048 bytes of native stack at most
```

## Running the conformance suite

The language is checked by the cases in `conformance/cases`, which KlongPy and the native runtime both run (see the repository's `conformance/README.md` for the case format). Each case runs a program in a fresh interpreter and compares the printed value, the console output and the kind of the error. `native/scripts/conformance.py` runs the cases that the default build declares, with the `klong --conformance` protocol, one process per case, and prints counts per file and a total.

`link` and the `python` below are as the repository's own checker runs them: `python` is the interpreter you use for KlongPy, and the scripts are in the repository.

<!-- check-docs: requires-file native/target/release/klong; link native conformance -->
```console
$ python native/scripts/conformance.py | tail -1 | cut -c1-40
total                         1276  1276
```

Run it without the `tail` to see the files. The two numbers are cases passed and cases run: 1276 of 1276 on this machine on 2026-11-16, in about two seconds. `--narrow` runs the cases that a build with `int32,float32` declares, and `KLONG_NATIVE` names the binary to test:

```sh
KLONG_NATIVE=native/target/narrow/release/klong python native/scripts/conformance.py --narrow
```

That gave 1160 of 1160 with the narrow build above. The parse trees of the golden files match too: `python conformance/run.py --trees --cmd native/target/release/klong` reported `169 of 169 trees match`. The same suite runs through the serial console of a simulated board and of a real one, see [Serial Console and Upload](console.md).

Not every case runs on every build. A case carries tags, and a build runs the cases whose tags it declares. The default build leaves out the tags `bigint` (it wraps), `torch-only` and `kgpy-only` (KlongPy's own), `desktop` (Python interop, databases, web) and the `timers` and `ipc` tiers, which have no cases. The narrow build also leaves out `float`, `string-fmt` and `width64`, the cases that pin 64-bit results.

The native tests that need KlongPy (`tests/test_native_conformance.py`, `tests/test_native_ipc.py`, `tests/test_native_mgmt.py`) skip themselves when `native/target/release/klong` does not exist, so building the binary turns them on. On this machine 86 passed and 11 were skipped; the 11 need a PeachQ executable, see [IPC and Timers](ipc.md). `cargo test --release --workspace` in `native/` runs the Rust tests.

## Sizes and speed

Measured on 2026-11-16, Apple-silicon Mac, nightly `rustc` 1.90.

**Programs on the desktop** (`ls -l`; the `release` profile, and the `min-size` profile that `cargo build --profile min-size -p klong-cli` selects: opt-level `z`, LTO, one code generation unit, `panic = "abort"`, stripped):

| Build | `klong` size |
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

These images have no IPC and no timers. IPC adds about 42 KB on `thumbv7em` and 59 KB on `riscv32imac`, timers 1.7 and 2.5 KB, and the `mgmt` library 12.7 KB (figures from the work on those features; they were not re-measured for this page). The firmware for the [ESP32-P4](esp32-p4.md) is 692 KB, but it is built with `opt-level` 3 and carries ESP-IDF.

**Speed** (`python native/scripts/bench.py`: best of five, in process, parse and run, KlongPy on numpy against the native `release` build):

| Program | KlongPy | native | ratio |
|---|---|---|---|
| `fib(24)`, `f::{:[x<2;x;f(x-1)+f(x-2)]}` | 1440 ms | 36.0 ms | 40 times |
| `+/!100000` | 0.029 ms | 0.021 ms | 1.4 times |
| `{x+y}/!100000` | 341 ms | 17.7 ms | 19 times |
| `#{x*x}'!100000` | 338 ms | 17.1 ms | 20 times |
| `+/{x*x}'!1000000` | 3396 ms | 169 ms | 20 times |
| append 2000 items one by one | 124 ms | 1.6 ms | 79 times |
| `#,/$'!20000` | 29.5 ms | 1.5 ms | 20 times |
| index a non-ASCII string 2000 times | 569 ms | 0.42 ms | 1356 times |
| dictionary, 50000 integer keys | 510 ms | 27.1 ms | 19 times |

Starting the program and exiting takes about 4 ms. KlongPy is faster for whole-array work that numpy does in one call, which is why the ratio of `+/!100000` is close to 1. These are desktop figures. On the ESP32-P4, measured on the board on 2026-11-16, `fib(24)` took 4.2 seconds against about 35 ms on the Mac. The cause of that difference has not been investigated. See [ESP32-P4](esp32-p4.md).
