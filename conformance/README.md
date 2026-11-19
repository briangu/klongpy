# Language conformance suite

Test cases for the Klong language that do not depend on any implementation.
KlongPy runs them in `tests/test_conformance.py`; a second implementation (the
native runtime) runs the same files through the small
command-line protocol below, with no Python. KlongPy is the reference: a case
that KlongPy fails is either a KlongPy bug (it goes to `pending/`) or a wrong
case.

```
conformance/
  README.md            this file
  run.py               case parser, runner, authoring aid (Python 3, standard library only)
  shim/kgpy_shim.py    KlongPy as an executable of the runner protocol (the reference shim)
  cases/*.cases        the cases, one area per file
  pending/*.cases      cases whose correct result KlongPy does not give yet
```

Golden parse trees are not copied: `tests/golden/{language,lib,fixtures}/*.sexp`
(with their sources) are part of the suite. See "Parse trees" below. The reader's view of
the suite, with the counts per implementation, is `docs/conformance.md`.

## Case file format

Plain text, UTF-8, one case after another. A line is blank, a comment, or a
one-character **sigil** followed by an optional single space and the text.
The text is everything after that space, trailing blanks included.

| Sigil | Meaning |
|---|---|
| `#` | comment, ignored (only at the start of a line) |
| `@ NAME TAG...` | starts a case. NAME is `area.topic`, unique in the suite. Tags below. |
| `> TEXT` | one line of source. The source is the `>` lines joined with `\n`, no final newline. A bare `>` is an empty line. |
| `= TEXT` | one line of the expected result: what the REPL prints for the value of the last statement. Several `=` lines are joined with `\n` (a string holding a newline prints on several lines). A bare `=` is an empty line. |
| `! KIND` | instead of `=`: evaluating must fail with this error kind. |
| `~ TEXT` | optional: the expected console output (what the program writes with `.p .d .w`), lines joined with `\n`; a final newline is not compared. |
| `: KEY TEXT` | metadata (`reason`, `actual`, `issue`); used in `pending/`. |
| `?` | authoring only: "fill me in", see `run.py --bless`. Never committed. |

A case with neither `=` nor `!` expects a program that produces no value.
Examples, from `cases/40-lists.cases`, `cases/90-errors.cases` and
`cases/11-printing.cases`:

```
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

Each case runs in a **fresh interpreter** with the default library and no
other state. The source is a whole program: all statements run in order and
the value of the last one is the result. The result is compared as printed
text, byte for byte; an error is compared by kind only, never by message.
Programs that write with `.p .d .w` end in a plain value (`;0`) so the
console text is the only thing under test.

### Error kinds

`syntax name arity type shape range system memory interrupt`, the kinds of
`docs/errors.md`. (`python`, `internal`,
`static` and `remote` are not part of this suite; `internal` is always a bug.)
A runtime maps its own errors to these.

### Tags

A tag names a capability the case needs; a runtime that does not declare it
skips the case. A case runs on a runtime if **every tag is one the runtime
declares**. There is no "expected failure".

Tier tags (at least one per case; `core` alone, or the other tiers the case
needs, `core` being implied by them):

| Tag | Needs |
|---|---|
| `core` | parsing, numbers, lists, symbols, functions, projections, adverbs, conditionals |
| `strings` | string and char values, `$`, `:$` and everything printing them |
| `dicts` | dictionaries |
| `sysio` | files and channels (`.ic`, `.l`, ...) |
| `timers`, `ipc`, `desktop` | tiers with no cases yet |

Marks (documented differences, listed in `docs/native/index.md`):

| Tag | Meaning |
|---|---|
| `width64` | needs 64-bit integers or floats; a 32-bit build skips it |
| `bigint` | needs integers beyond the build width (a scalar integer power, for one) |
| `unicode` | counts or indexes non-ASCII text, in code points (KlongPy and the native runtime both count code points; a runtime that counts bytes does not declare it) |
| `device` | needs a depth limit, a heap limit or a console |

Capability tags:

| Tag | Meaning |
|---|---|
| `float` | needs floating point numbers; the cases expect `f64` (64-bit) results, so a build with 32-bit floats (the native `float32` feature) does not declare it |
| `string-fmt` | depends on the exact float text (Python `repr`: `1e+22`, `1e-07`, `0.30000000000000004`) |
| `strict1`, `strict2` | the runner starts the interpreter at that strict level (`--strict N`) |
| `static` | the declarations of the static profile run as they stand, `.extern` and `.entry` (docs/static-profile.md); a case runs with no host function, so calling an extern is a `system` error |
| `qualified` | the qualified name `a.b`, which means `a@"b"` (reading, writing `a.b::v`, calling `a.b(x)`, nesting `a.b.c`); a missing key is a `name` error, a base that is not a dictionary a `type` error |
| `modules` | `.import` and `.export` (a file is a module and a module is a value). The cases need files: see "Module cases" |
| `clock` | the platform has a wall clock that knows the date (`.now()`): a desktop does, a board that was not told the time does not |

Implementation tags: `numpy-only`, `torch-only`, `kgpy-only` mark behaviour
that belongs to one KlongPy backend, and `kg-only` marks what only the standalone binary `kg` does: a system function that exists only in the
Python-hosted command (`kgpy`) is a `name` error there that says so (`.py needs kgpy (the Python-hosted command)`). KlongPy does not declare
`kg-only`; `run.py --cmd CMD` declares it for any executable but the KlongPy shim. `numpy-only` is the result of the
reference (numpy) backend that the torch backend cannot give by design: text
of a float64 value (torch computes in float32 by default) and `:undefined`
inside a list (a tensor cannot hold it and keeps the IEEE `inf`/`nan`). Each such case has a `# numpy-only:` comment with the reason; the
expected result stays the reference one, and another runtime that gives it
declares the tag. No case uses `torch-only` or `kgpy-only` yet. KlongPy on
the numpy backend declares every tag except `torch-only`, `timers`, `ipc` and
`desktop` (`KLONGPY_TAGS` in `run.py`); on the torch backend it declares
`torch-only` instead of `numpy-only` (`KLONGPY_TORCH_TAGS`,
`klongpy_tags(backend)`).

## Runner protocol

An implementation is an executable. The runner starts it **once per case**.

```
CMD [--strict N]        program text on stdin (UTF-8)
CMD --dump-ast FILE     parse trees, see below
```

* stdout and exit status: on success exit status 0 and the printed result on
  stdout followed by one newline (nothing at all if the program has no
  value); on a Klong error exit status 1 and one line `error: <kind>`.
* stderr: only what the program wrote with `.p .d .w`, byte for byte.
  Warnings and diagnostics are not written there.
* Anything else (another exit status, a signal, no answer within 20 s) is a
  crash and fails the case.
* `--strict N` runs the program at strict level N (as `.strict(N)` before the
  program is parsed); the level applies to the whole program.

Run it:

```
python conformance/run.py                                   all cases against KlongPy, in process
python conformance/run.py --cmd "CMD" --tags core,strings --jobs 8
python conformance/run.py --list --tags core                what would run
python conformance/run.py --pending                         the pending list; reports any that pass
python conformance/run.py conformance/cases/40-lists.cases  one file
```

`--tags` is the list of tags the implementation declares; cases with any
other tag are skipped and counted. A failure prints as
`FAIL conformance/cases/40-lists.cases:12 lists.take-last-more-cycles` with the
expected and the actual text. A native build can port the runner (`parse_text`
is about 60 lines) or keep using this one.

`kgpy_shim.py` is the reference executable (it calls KlongPy's API and prints
as above); `tests/test_conformance.py` drives it through the external path.

## Module cases

`cases/87-modules.cases` is generated by `conformance/tools/gen_modules.py` (edit the generator and run it, see its
docstring). A runner has no fixtures, so a case that needs module files writes them itself at the start of the
program (`.oc`, `.tc`, `.p`, `.cc`), runs its own statements, deletes the files (`.df`) and ends with the value of its last
statement. The files are `/tmp/kgm87-<case>-<name>.kg`, unique per case; a case that ends in an error or fails cannot delete them, so `run.py` removes what is left when it finishes. In the
comments of the case file `{P}` is the absolute path of such a file, `{R}` the `./` path (used from one module file to
another, relative to the importing file), `{U}` the `../tmp/` path (relative, through `..`; written as `/tmp/../tmp/...`
in the main program, which is read from standard input and has no directory of its own) and `{B}` the bare name (looked
up in the library path only, never in the importing file's directory or the working directory). The cases that use
`csv` import the bundled library by its bare name. The runner stops at the first error, so a scenario that needs a
program to go on after a failed import (a failed import is not cached) is not expressible here.

## Parse trees

`kgpy ast FILE` prints the canonical dump (implemented by `klongpy/dump.py`) of every top-level form of FILE.
The golden files are `tests/golden/<group>/<name>.sexp`; groups and sources
are listed in `tests/golden/README.md` and enumerated by
`tests/golden/regen.py`. The native parser must print the same bytes:

```
python conformance/run.py --trees --cmd "CMD"      CMD --dump-ast FILE for every golden pair
```

A golden file that holds `(error ClassName)` (the three syntax error classes)
means the implementation must exit 1 with `error: syntax` on stdout.

## Pending

`pending/*.cases` hold cases whose **correct** result is known but KlongPy does
not produce it. Each has `: reason` (why it is a bug, or which change will
fix it) and `: actual` (what KlongPy answers). The expectation is the
correct result. `tests/test_conformance.py` checks that each still fails and
prints a note when one starts to pass; move it to `cases/` then
(`CONFORMANCE_STRICT_PENDING=1` makes that a failure).

A case whose `: reason` starts with `native:` is the other kind: KlongPy passes it and
another runtime does not yet (the language changed in KlongPy first). `run.py --pending`
and the test count those apart and do not ask to move them; move them to `cases/` when
the other runtime passes.

## Adding cases

Write the source, put `?` where the expectation goes, run
`python conformance/run.py --bless FILE`, then **read what it wrote**: the
bless step records what KlongPy prints, so a KlongPy bug gets blessed too.
Check the printed form against the reference
(https://t3x.org/klong/klong-ref.txt.html) and
`tests/kgtests/language/test_suite.kg`; a result you believe is wrong goes to
`pending/` with the correct result.

Rules of thumb:

* A few sharp cases per behaviour, boundary values included; one behaviour per case.
* Put a leading negative number in parentheses: `-2#x` is `-(2#x)`. In a data
  list `[1 -2]` is fine.
* Avoid the printed form of integral floats unless that is what is tested
  (`4%2` prints `2.0`).
* Tags must be honest: `tests/test_conformance.py` rejects a `core` case that
  uses a string literal, `$` or a dictionary literal, and a case with a float
  literal or result that lacks `float`.
* Do not test operator values, function display (`:monad`, `:dyad`) or
  anything printed with a Python object repr; their form is not fixed.
