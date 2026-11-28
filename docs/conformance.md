# Conformance suite

The conformance suite is a set of test cases for the Klong language that does not depend on any implementation. This page is for someone who wants to run the suite, read its counts, or add a case. `conformance/README.md` is the specification of the case format, the tags and the runner protocol.

Each case is a program and the text the REPL prints for its value, or the kind of error it must raise. KlongPy runs the cases in process. The [native runtime](native/index.md) runs the same files through a small command-line protocol, with no Python. KlongPy is the reference: when it fails a case, either KlongPy has a bug (the case goes to the pending list, below) or the case is wrong.

The suite has 2227 cases in 25 files under `conformance/cases/`, 5 pending cases under `conformance/pending/` (that KlongPy fails), and 169 golden parse trees under `tests/golden/`.

## A case

A case is a name, some tags, the source and the expected result:

```text
@ lists.take-last-more-cycles core
> (-7)#[1 2 3]
= [3 1 2 3 1 2 3]

@ errors.err-range-index core
> [1 2 3]@3
! range
```

Each case runs in a fresh interpreter. The result is compared as printed text, byte for byte. An error is compared by kind only, never by message. A tag names a capability the case needs; a runtime that does not declare it skips the case (for example `width64` for a 32-bit build). A runtime runs a case only if it declares every tag of the case. There is no "expected failure". `tests/test_conformance.py` checks that the tags are honest.

An implementation is an executable that reads the program on stdin and prints the value, or `error: KIND` with exit status 1. You can try the protocol with the native binary:

```text
$ echo '1+2' | native/target/release/kg --conformance
3
$ echo '[1 2 3]@3' | native/target/release/kg --conformance; echo $?
error: range
1
```

`tests/test_conformance.py` runs the suite as part of the KlongPy tests. `tests/test_native_conformance.py` runs it against the native binary and is skipped when the binary is not built.

## Running it against KlongPy

<!-- check-docs: link conformance -->
```console
$ python conformance/run.py
2206 passed, 0 failed, 21 skipped by tags (KlongPy (in process))
```

The in-process run takes about a second. To run KlongPy through the same executable protocol that a native implementation uses, give the reference shim as the command (about 17 seconds with 8 jobs):

```text
$ python conformance/run.py --cmd "python -W ignore conformance/shim/kgpy_shim.py" --jobs 8
2206 passed, 0 failed, 21 skipped by tags (python -W ignore conformance/shim/kgpy_shim.py)
```

`--tags a,b` is the list of tags the implementation declares; other cases are skipped and counted. `--list` shows what would run. A file or directory argument limits the run. A failure prints the file, the line and the case name with the expected and the actual text.

<!-- check-docs: continue -->
```console
$ python conformance/run.py conformance/cases/90-errors.cases
63 passed, 0 failed, 0 skipped by tags (KlongPy (in process))
$ python conformance/run.py --list --tags core | awk 'END { print NR }'
808
```

`python -m pytest tests/test_conformance.py --backend torch --device cpu` runs the in-process cases on the torch backend. The numpy backend declares every tag except `torch-only`, `timers`, `ipc` and `desktop`. The torch backend declares `torch-only` instead of `numpy-only`, because float32 text and `:undefined` inside a list differ there by design.

## Running it against the native binary

Build the binary, then point the runner at it. Without `--tags` the runner declares what the default native build has: every tag of KlongPy's and `kg-only` (what only the standalone binary does: a system function of the Python-hosted command is a `name` error there that says so).

```text
$ cd native && cargo build --release
$ cd ..
$ python conformance/run.py --cmd "native/target/release/kg --conformance" \
      --jobs 8
2227 passed, 0 failed, 0 skipped by tags (native/target/release/kg --conformance)
```

The desktop run passed all 2227 cases in 4.0 seconds with eight jobs on an Apple M2 Ultra on 2026-11-27. With `--tags core` alone, 808 cases run. The desktop binary declares `bigint` (it is built with the feature `bigint`, on by default for `klong-cli`), `clock` and `kg-only`; it does not declare `torch-only`, `kgpy-only`, `timers`, `ipc` or `desktop`. Without `bigint` the run is smaller by the cases of big integers: a board build leaves them out.

`native/scripts/conformance.py` prints the same run as a table of counts per file. The golden parse trees run the same way:

```text
$ python conformance/run.py --trees --cmd "native/target/release/kg"
169 of 169 trees match
```

A build for a 32-bit target (features `int32,float32`) declares fewer tags: `core,strings,dicts,unicode,sysio,static,strict1,strict2,numpy-only,device` (and `clock` where the board has one).

## Running it through the simulated serial console

A board is driven through its serial console, not through the `kg` binary. `native/scripts/console_conformance.py` sends each case to a simulated board (`klong-sim` on a pseudo-terminal) with the console's `^A run` command and judges what comes back over the wire. It resets the board for each case, as the fresh-interpreter rule requires.

```text
$ python native/scripts/console_conformance.py
...
total                         1720  1720   in 3.3 s through console of native/target/release/klong-sim, 8 jobs
```

`--narrow` runs the tags of a 32-bit build (1527 cases passed on 2026-11-19). With `--port /dev/ttyUSB0` the same script drives a real board over a serial port, one case at a time. It is the only conformance run on a board; the other hardware results are on the [native status table](native/index.md#status-what-has-and-has-not-been-proven).

## Counts

| Implementation and run | Cases run | Passed | Date |
|---|---|---|---|
| KlongPy, in process | 2206 (21 `kg-only` skipped) | 2206 | 2026-11-27 |
| KlongPy through the executable shim | 2206 (21 `kg-only` skipped) | 2206 | 2026-11-27 |
| Native runtime, desktop, default build | 2227 | 2227 | 2026-11-27 |
| Native runtime, simulated boards through the console | 1720 | 1720 | 2026-11-27 |
| Native runtime, desktop, without `bigint`, and the 32-bit build (`--narrow`) | not run since the suite grew | | 2026-11-18 (1570 and 1408 of an earlier, smaller suite) |
| Native runtime on an ESP32-P4 over the serial port | 1266 | 1266 | 2026-11-16 |

### Why the counts differ

The hosted run skips the 21 `kg-only` cases and runs 2206 of 2227. The board and simulator omit cases for features they do not declare, such as `bigint`. A 32-bit build also omits tests requiring the desktop numeric widths. The physical ESP32-P4 result records the 1266-case suite of 2026-11-16. The desktop and simulator counts above use the expanded suite.

## The pending list

`conformance/pending/*.cases` hold cases whose correct result is known but which KlongPy does not give yet. The expectation in a pending case is the correct result. Each case also has `: reason` (why the current answer is wrong) and `: actual` (what KlongPy answers today).

<!-- check-docs: continue -->
```console
$ python conformance/run.py --pending
5 of 5 pending cases still fail as expected
```

The five pending cases are KlongPy's known bugs. They are listed once, with examples, in [Differences from Klong](differences.md#known-bugs).

`tests/test_conformance.py` checks that each pending case still fails and prints a warning when one starts to pass. When that happens, move the case to `cases/`. Setting `CONFORMANCE_STRICT_PENDING=1` turns the warning into a failure. A runtime that is not KlongPy does not run the pending list.

A pending case of another kind passes in KlongPy and fails in the native runtime, because the language changed in KlongPy first. Its `: reason` starts with `native:`. `run.py --pending` and the test count these apart and do not ask to move them; they move to `cases/` when the native runtime gives the same result. There are none.

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
