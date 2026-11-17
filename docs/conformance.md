# Conformance suite

The conformance suite is a set of test cases for the Klong language that does not depend on any implementation. Each case is a program and the text the REPL prints for its value, or the kind of error it must raise. KlongPy runs the cases in process. The [native runtime](native/index.md) runs the same files through a small command-line protocol, with no Python involved. A third implementation would do the same.

KlongPy is the reference. When KlongPy fails a case, either KlongPy has a bug (the case goes to the pending list, below) or the case is wrong.

The suite has 1294 cases in 17 files under `conformance/cases/`, 8 pending cases under `conformance/pending/` (that KlongPy fails), and the golden parse trees under `tests/golden/` (169 pairs of source and tree). The full specification is `conformance/README.md`; this page is the short version for users and implementers.

## What is in the repository

```text
conformance/
  README.md            the specification of the format and the protocol
  run.py               case parser, runner and authoring aid (Python 3, standard library only)
  shim/kgpy_shim.py    KlongPy as an executable of the runner protocol
  cases/*.cases        the cases, one area per file
  pending/*.cases      cases whose correct result KlongPy does not give yet
```

`tests/test_conformance.py` runs the suite as part of the KlongPy tests. `tests/test_native_conformance.py` runs it against the native binary and is skipped when the binary is not built.

## Case file format

A case file is plain UTF-8 text. Each line is blank, a comment, or a one-character sigil followed by an optional space and the text.

| Sigil | Meaning |
|---|---|
| `#` | Comment (only at the start of a line). |
| `@ NAME TAG...` | Starts a case. `NAME` is `area.topic` and is unique in the suite; the area is the file name without its number. |
| `> TEXT` | One line of source. The lines are joined with newlines. A bare `>` is an empty line. |
| `= TEXT` | One line of the expected result, which is what the REPL prints for the value of the last statement. Several lines are joined with newlines. |
| `! KIND` | Instead of `=`: the program must fail with this error kind. |
| `~ TEXT` | Optional: the expected console output (what the program writes with `.p`, `.d` and `.w`). |
| `: KEY TEXT` | Metadata (`reason`, `actual`, `ticket`), used in `pending/`. |
| `?` | Authoring only: "fill the expectation in". Never committed. |

A case with neither `=` nor `!` expects a program that produces no value. Examples from the suite:

```text
@ lists.take-last-more-cycles core
> (-7)#[1 2 3]
= [3 1 2 3 1 2 3]

@ errors.err-range-index core
> [1 2 3]@3
! range

@ printing.console-d-no-newline-then-p strings
> .d("a");.d("b");.p("");0
= 0
~ ab
```

Every case runs in a fresh interpreter with the default library and no other state. The source is a whole program: the statements run in order and the value of the last one is the result. The result is compared as printed text, byte for byte. An error is compared by kind only, never by message. The kinds are `syntax`, `name`, `arity`, `type`, `shape`, `range`, `system`, `memory` and `interrupt`.

## Tags

A runtime runs a case only if it declares every tag of the case. There is no "expected failure": a tag says "skip this case here".

| Tag | Meaning |
|---|---|
| `core` | Parsing, numbers, lists, symbols, functions, adverbs, conditionals. |
| `strings`, `dicts`, `sysio` | Strings and chars; dictionaries; files and channels. |
| `timers`, `ipc`, `desktop` | Tiers with no cases yet. |
| `float` | Needs floating point numbers. |
| `string-fmt` | Depends on the exact float text (Python's `repr`). |
| `width64` | Needs 64-bit integers or floats; a 32-bit build skips it. |
| `bigint` | Needs integers beyond the build width. No case uses it yet. |
| `unicode` | Counts or indexes non-ASCII text (KlongPy counts code points; the native runtime counts bytes). |
| `device` | Needs a depth limit, a heap limit or a console. |
| `strict1`, `strict2` | The runner starts the interpreter at that strict level. |
| `numpy-only`, `torch-only`, `kgpy-only` | Behaviour that belongs to one KlongPy backend. |

Every case has at least one tier tag (`core` alone, or the other tiers it needs). `tests/test_conformance.py` checks that the tags are honest: a `core` case may not use a string literal, `$` or a dictionary literal, and a case that uses a float without the `float` tag is rejected.

## The runner protocol

An implementation is an executable. The runner starts it once per case:

```text
CMD [--strict N]        program text on stdin (UTF-8)
CMD --dump-ast FILE     the parse trees of FILE
```

* On success the exit status is 0 and stdout holds the printed value and one newline (nothing at all if the program has no value).
* On a Klong error the exit status is 1 and stdout holds one line, `error: <kind>`.
* stderr holds only what the program wrote with `.p`, `.d` and `.w`, byte for byte.
* Anything else (another exit status, a signal, no answer within 20 seconds) is a crash and fails the case.
* `--strict N` runs the program at strict level N for the whole program.

You can try the protocol by hand with the native binary:

```text
$ echo '1+2' | native/target/release/klong --conformance
3
$ echo '[1 2 3]@3' | native/target/release/klong --conformance; echo $?
error: range
1
```

## Running it against KlongPy

The commands below are checked by `scripts/check_docs.py`, so the counts are what the code gives today.

<!-- check-docs: link conformance -->
```console
$ python conformance/run.py
1294 passed, 0 failed, 0 skipped by tags (KlongPy (in process))
```

The in-process run takes about a second. To run KlongPy through the same executable protocol that a native implementation uses, give the reference shim as the command (about 17 seconds with 8 jobs on the machine of the measurements below):

```text
$ python conformance/run.py --cmd "python -W ignore conformance/shim/kgpy_shim.py" --jobs 8
1294 passed, 0 failed, 0 skipped by tags (python -W ignore conformance/shim/kgpy_shim.py)
```

`--tags a,b` is the list of tags the implementation declares; other cases are skipped and counted. `--list` shows what would run. A file or directory argument limits the run. A failure prints the file, the line and the case name with the expected and the actual text.

<!-- check-docs: continue -->
```console
$ python conformance/run.py conformance/cases/90-errors.cases
57 passed, 0 failed, 0 skipped by tags (KlongPy (in process))
$ python conformance/run.py --list --tags core | wc -l
     771
```

`python -m pytest tests/test_conformance.py --backend torch --device cpu` runs the in-process cases on the torch backend. The numpy backend declares every tag except `torch-only`, `timers`, `ipc` and `desktop`. The torch backend declares `torch-only` instead of `numpy-only`, because float32 text and `:undefined` inside a list differ there by design.

## Running it against the native binary

Build the binary, then point the runner at it. The tags are what the default native build declares.

```text
$ cd native && cargo build --release
$ cd ..
$ python conformance/run.py --cmd "native/target/release/klong --conformance" \
      --tags core,strings,dicts,sysio,float,string-fmt,width64,strict1,strict2,numpy-only,device,unicode --jobs 8
1293 passed, 0 failed, 1 skipped by tags (native/target/release/klong --conformance)
```

Measured on 2026-11-16 on an Apple M2 Ultra: 1293 of 1293 in about 2 seconds with 8 jobs. With `--tags core` alone, 771 cases run and 523 are skipped by tags. The native runtime does not declare `bigint`, `torch-only`, `kgpy-only`, `timers`, `ipc` or `desktop`.

`native/scripts/conformance.py` prints the same run as a table of counts per file. The golden parse trees run the same way:

```text
$ python conformance/run.py --trees --cmd "native/target/release/klong"
169 of 169 trees match
```

A build for a 32-bit target (features `int32,float32`) declares fewer tags: `core,strings,dicts,unicode,sysio,strict1,strict2,numpy-only,device`.

## Running it through the simulated serial console

A board is driven through its serial console, not through the `klong` binary. `native/scripts/console_conformance.py` sends each case to a simulated board (`klong-sim` on a pseudo-terminal) with the console's `^A run` command and judges what comes back over the wire. It resets the board for each case, as the fresh-interpreter rule requires.

```text
$ python native/scripts/console_conformance.py
...
total                         1293  1293   in 2.4 s through console of native/target/release/klong-sim, 8 jobs
```

`--narrow` runs the tags of a 32-bit build (1160 cases passed on 2026-11-16). With `--port /dev/ttyUSB0` the same script drives a real board over a serial port, one case at a time. On an ESP32-P4 the native runtime passed 1266 of 1266 this way on 2026-11-16 (see [ESP32-P4](native/esp32-p4.md)). That is the only hardware result so far.

## The pending list

`conformance/pending/*.cases` hold cases whose correct result is known but which KlongPy does not give yet. The expectation in a pending case is the correct result. Each case also has `: reason` (why the current answer is wrong) and `: actual` (what KlongPy answers today).

<!-- check-docs: continue -->
```console
$ python conformance/run.py --pending
8 of 8 pending cases still fail as expected
```

The eight cases that KlongPy fails today: `&` of an empty list and of a string, `[:a :b]?:b` (find a symbol in a list), `*"abc"` and the first item of `,\"abc"` (a string where a char is expected), `[1 2 3]@1.0` (indexing by a real), and the reciprocal of a string (a range error where a type error is expected).

`tests/test_conformance.py` checks that each pending case still fails and prints a warning when one starts to pass. When that happens, move the case to `cases/`. Setting `CONFORMANCE_STRICT_PENDING=1` turns the warning into a failure. A runtime that is not KlongPy does not run the pending list.

A pending case of another kind is possible: when KlongPy changes first, the cases for the change pass in KlongPy and fail in the native runtime. Their `: reason` starts with `native:`, and `run.py --pending` and the test count them apart (`N more pass in KlongPy and wait for the native runtime`) instead of asking to move them; they move to `cases/` when the native runtime gives the same result. There are none now: the 15 cases for calling a value that is not a function (a `type` error, see [Differences from Klong](differences.md#calling-a-value-that-is-not-a-function-is-an-error)) waited for the native runtime, which gives the same error since 2026-11-16, and are in `cases/60-functions.cases`.

## Adding a case

1. Write the source in a `.cases` file with a `?` line where the expectation goes.
2. Run `python conformance/run.py --bless FILE`. It records what KlongPy prints.
3. Read what it wrote. A KlongPy bug gets blessed too. Check the printed form against the [Klong reference](https://t3x.org/klong/klong-ref.txt.html) and `tests/kgtests/language/test_suite.kg`. If you believe the result is wrong, put the case in `pending/` with the correct result, a `: reason` and an `: actual`.

<!-- check-docs: continue -->
```console
$ printf '@ demo.sum core\n> 1+2\n?\n\n@ demo.range-error core\n> [1 2 3]@3\n?\n' > demo.cases
$ python conformance/run.py --bless demo.cases > /dev/null
$ cat demo.cases
@ demo.sum core
> 1+2
= 3

@ demo.range-error core
> [1 2 3]@3
! range
$ python conformance/run.py demo.cases
2 passed, 0 failed, 0 skipped by tags (KlongPy (in process))
```

Rules of thumb:

* Write a few sharp cases per behaviour, with the boundary values; one behaviour per case.
* Put a leading negative number in parentheses: `-2#x` is `-(2#x)`. In a data list, `[1 -2]` is fine.
* Avoid the printed form of integral floats unless that is what the case tests (`4%2` prints `2.0`).
* Do not test function display (`:monad`, `:dyad`) or anything printed as a Python object.
* Put the area in the file name and the case name: a case in `40-lists.cases` is named `lists.something`. Names are unique across `cases/` and `pending/`.

Run `python -m pytest tests/test_conformance.py -q` after adding a case. It rejects unknown tags, duplicate names, unfilled `?` lines and dishonest tags.

## See also

* [Contributing](contributing.md)
* [Native runtime](native/index.md)
* [Roadmap](roadmap.md), for what is not run yet
* [Differences from Klong](differences.md)
