# Contributing

This page is for someone who has cloned the repository and wants to build it, run the tests and change something. How to propose a change (discussion first, then pull request) is in `CONTRIBUTING.md` at the root of the repository. Read that first.

## Repository layout

| Path | What it holds |
|---|---|
| `klongpy/` | The Python implementation (package layout below). |
| `native/` | The Rust implementation of the language, a Cargo workspace (crates below). |
| `conformance/` | The implementation-neutral test cases and their runner. See [Conformance suite](conformance.md). |
| `tests/` | The Python test suite, the golden parse trees and the benchmark scripts (`perf_*.py`). |
| `scripts/` | `check_docs.py` (the documentation checker) and `gen_system_functions.py` (a generator). |
| `examples/` | Example programs, one directory per topic. Several have their own `README.md`. |
| `docs/` | These pages. Built with MkDocs; `mkdocs.yml` holds the navigation. |
| `.github/workflows/` | `unittests.yml` and `gh-pages.yml`. |
| `pyproject.toml` | Package metadata, the optional extras and the pytest configuration (`testpaths = ["tests"]`). |

### The `klongpy/` package

| Module or package | Role |
|---|---|
| `interpreter.py`, `core.py`, `types.py` | The interpreter, its core types and type checks. |
| `lexer.py`, `parser.py` | The tokenizer with source spans and the parser. |
| `monads.py`, `dyads.py`, `adverbs.py` | The operators. |
| `compiler.py` | The expression compiler that turns operator chains into backend code. See [Expression Compiler Notes](compiler-roadmap.md). |
| `backends/` | The array backends: `numpy_backend.py`, `torch_backend.py`, the shared `base.py` and the `registry.py`. `backend.py` selects one. |
| `autograd.py`, `sys_fn_autograd.py` | Gradients and the autograd system functions. |
| `sys_fn.py`, `sys_var.py`, `sys_fn_timer.py`, `sys_fn_ipc.py`, `sys_fn_static.py` | System functions and variables. `.entry` and `.extern` are in `sys_fn_static.py`. |
| `errors.py`, `writer.py` | The error classes with positions, and the output formatting. |
| `lint.py`, `dump.py` | `kgpy --lint`, and the canonical parse-tree dump used by the golden trees. |
| `cli.py`, `repl.py`, `repl_input.py`, `help_index.py` | The `kgpy` command, the REPL and its help. |
| `ipc/` | The kdb+ IPC codec, sessions, compression and endpoints. |
| `static/` | The static profile: checker, type inference, the IR and the C emitter. |
| `db/`, `web/`, `ws/` | The optional database, web server and WebSocket system functions. |
| `lib/` | Klong libraries (`math.kg`, `mgmt.kg`, ...) loaded with `.l`. |

### The `native/` workspace

| Crate | Role |
|---|---|
| `klong-core` | Lexer, parser, values and evaluator. `no_std` with `alloc`. |
| `klong-ipc` | The kdb+ IPC codec, session and serial framing, with no I/O. |
| `klong-rt` | The runtime around the core: the `Platform` trait, system functions, timers, heap accounting. |
| `klong-host` | The `Platform` for desktop operating systems. |
| `klong-cli` | The `klong` binary: file, `-e`, REPL, `--dump-ast` and `--conformance`. |
| `klong-console` | The serial console: line editor and the `^A` command mode. |
| `klong-sim` | A simulated board with the real console on top, used as a test bench. |
| `klong-board` | The skeleton of a board port, and a linkable image. |
| `klong-ffi` | The board runtime as a static library with a C ABI (`include/klong.h`). |
| `klong-probe` | A size probe, outside the workspace (its own `Cargo.lock`). |
| `boards/esp32p4/` | The ESP-IDF application for the ESP32-P4. |
| `scripts/`, `tools/`, `tests/` | Measurement and conformance scripts, `kgupload.py` (upload over a serial port) and the console tests in Python. |

See [Native runtime](native/index.md) for what the crates do together.

## Setting up an environment

You need Python 3.9 to 3.14.

```text
$ python -m venv .venv
$ .venv/bin/pip install -e ".[full]"
$ .venv/bin/pip install pytest-asyncio
```

`[full]` includes `pytest`, which the test suite needs (see below); the second line adds `pytest-asyncio`, the other half of the `dev` extra. The extras in `pyproject.toml`:

| Extra | Adds |
|---|---|
| (none) | `numpy` and `colorama`, which the `kgpy` command needs. |
| `torch` | PyTorch, for the torch backend and autograd. |
| `repl` | `colorama` (already a core dependency; the extra is kept for old install lines). |
| `web` | `aiohttp`, for `.web`. |
| `db` | `pandas` and `duckdb`, for `.db` and the table functions. |
| `ws` | `websockets`, for `.ws`. |
| `docs` | `mkdocs`, `mkdocs-material` and `mike`. |
| `dev` | `pytest` and `pytest-asyncio`. |
| `all` | torch, colorama, aiohttp, pandas, duckdb and websockets. It does not include `dev` or `docs`. |
| `full` | `all` and `pytest`, which several test modules import. Use it for `pip install -e ".[full]"` before running the tests. It does not include `pytest-asyncio` or `docs`. |

Torch is optional. KlongPy runs without it, and tests and examples that need it skip or are skipped by the documentation checker. To work on the torch backend, install `torch`, and see the rule below on where `torch` may be imported.

## Running the tests

Run the full suite with:

```text
$ python -m unittest discover tests
```

or with pytest:

```text
$ python -m pytest tests -q
```

The full suite takes several minutes: about 7 minutes with pytest and about 6 with `unittest discover` when torch, pandas, duckdb, aiohttp and websockets are installed and the native binary is built. Single files are much faster; `tests/test_conformance.py` takes about 11 seconds.

Four test modules (`test_atomic_broadcasting`, `test_autograd_parametrized`, `test_nested_grade` and `test_ascii_gradient_names`) import `pytest`, so `unittest discover` needs it installed too; the `full` extra includes it. Without torch installed, the 33 tests of `tests/test_compiler.py` that use torch are skipped.

pytest also takes the options defined in `tests/conftest.py`, which run every `KlongInterpreter` in the tests on another backend:

```text
$ python -m pytest tests -q --backend torch --device cpu
$ python -m pytest tests -q --backend torch --device mps
```

`--backend` is `numpy` (the default), `torch` or `test_backend` (a stub backend in `tests/custom_backend.py`). `--device` is `cpu`, `cuda` or `mps`, and applies to torch only. These options need pytest; `unittest discover` always uses numpy.

To run one file or one test:

```text
$ python -m pytest tests/test_conformance.py -q -p no:cacheprovider
$ python -m unittest tests.test_cli_errors.TestInterrupt -v
```

### Tests that skip without a tool

| Test | Needs | If missing |
|---|---|---|
| `tests/test_native_conformance.py`, `tests/test_native_ipc.py`, `tests/test_native_mgmt.py` | The native binary: `cd native && cargo build --release`. `KLONG_NATIVE` names another binary. | Skipped. |
| PeachQ interop tests (`tests/test_ipc_peachq.py` and the PeachQ classes in `tests/test_native_ipc.py`) | `KLONG_TEST_PEACHQ` set to the path of a PeachQ `q` executable. | Skipped. They are opt-in. |
| `native/tests/test_console.py` | The native binaries, including `klong-sim`. | Skipped. |
| `tests/static/` (the emitted C is compiled and run) | A C compiler: `$KG_TEST_CC`, else `cc`, `clang` or `gcc` on the path. | Skipped. |
| `tests/test_backend.py` and others that name torch | PyTorch. | Skipped. |
| `tests/test_compiler.py` | PyTorch, for 33 of its 69 tests. | Those tests are skipped; CI still excludes the file on the legs without torch. |
| Tests of `.db`, `.web` and `.ws` | `pandas`, `duckdb`, `aiohttp`, `websockets`. | Skipped. |
| `tests/test_docs_examples.py` | Blocks in the docs that need an optional package or a PeachQ session. | The checker skips those blocks. |

The native tests that need the binary run the conformance suite against it and the IPC tests against a simulated board. Build it first (see [Building and running](native/building.md)). With it built, `tests/test_native_conformance.py`, `tests/test_native_mgmt.py` and `tests/test_native_ipc.py` pass in about a minute, skipping the PeachQ tests, and `native/tests/test_console.py` takes about 20 seconds.

## The golden parse trees

`tests/golden/` holds one `.sexp` file per source: every `.kg` under `tests/kgtests/language/`, every `klongpy/lib/*.kg`, and the small inputs in `tests/golden/fixtures/`. `tests/test_golden_trees.py` parses each source and requires the canonical dump of `klongpy/dump.py` to match the file byte for byte. The native parser must print the same bytes (`conformance/run.py --trees`).

After an intended change to the parser or the dump:

```text
$ python tests/golden/regen.py --check
all golden files match
```

`--check` lists the files that differ and exits 1 if any does. Running `python tests/golden/regen.py` without `--check` rewrites every snapshot; review the diff. To see the tree of one file, run `python -m klongpy.dump FILE`. `tests/golden/README.md` lists the intended changes.

## How documentation examples are checked

This is the one page that explains how examples in the docs are verified.

`scripts/check_docs.py` runs the examples in `README.md` and `docs/*.md` and compares what they print with what the page says. `tests/test_docs_examples.py` runs it on every page, so the pytest and unittest runs include it.

```text
$ python scripts/check_docs.py                  # every page (under a minute)
$ python scripts/check_docs.py docs/errors.md   # one page
```

It prints a line like `1512 examples checked, 0 mismatches; 7 blocks skipped (...)` and exits 1 on a mismatch. Run it with an environment that has the extras installed, or the blocks that need `aiohttp`, `duckdb` or `pandas` are skipped. The docstring at the top of the script lists every block form and directive. The short version:

A fenced block whose first line starts with `?> ` runs in a fresh REPL. The lines after an input line, up to the next `?> `, are what the REPL must print, ending with the value:

```kgpy
?> 1+2
3
?> .p("hi")
hi
"hi"
```

This block is itself checked. A shell session is a block with the fence tag `console`, whose first line starts with `$ `. `kgpy` runs this checkout's command line, `python` runs the interpreter that runs the checker, and the output must match exactly:

<!-- check-docs: link examples -->
```console
$ kgpy -e '1+2'
3
```

A comment on the line before a block changes how it runs:

| Directive | Effect |
|---|---|
| `<!-- check-docs: continue -->` | Run in the interpreters (or directory) of the previous block. |
| `<!-- check-docs: requires duckdb pandas -->` | Run only if these packages import. |
| `<!-- check-docs: requires-cmd clang -->` | Run only if these programs are on the `PATH`. |
| `<!-- check-docs: link examples -->` | Make these repository paths readable in a `console` block. |
| `<!-- check-docs: strict 2 -->` | Run at strict level 2. |
| `<!-- check-docs: render -->` | Show an error as the REPL prints it, with position and hint. |
| `<!-- check-docs: skip: reason -->` | Do not run; the reason is required. Use it only for an example that cannot run here. |

Directives combine with `;`. A block without `?> ` or `$ ` at the start is not run, so output of the native binary and other things the checker cannot start are shown as plain `text` blocks. Say on the page that those are not checked.

## Generated files

Two pages are generated. Do not edit them by hand; edit the source and regenerate.

| Page | Generator | Source |
|---|---|---|
| `docs/static-profile.md` | `python -m klongpy.static.profile_doc` (`--check` to test, `--write` to refresh) | `klongpy/static/optable.py` and `rules.py` for the tables between the `BEGIN GENERATED` and `END GENERATED` markers; the text around them is `HEAD` and `TAIL` in `klongpy/static/profile_doc.py`. |
| `docs/system-functions.md` | `scripts/gen_system_functions.py` | The `eval_sys_*` docstrings of the modules in `MODULES`, with `DESC` and `GROUPS` in the script. |

```text
$ python -m klongpy.static.profile_doc --check docs/static-profile.md; echo $?
0
$ PYTHONPATH=. python -W ignore scripts/gen_system_functions.py > /tmp/sf.md && mv /tmp/sf.md docs/system-functions.md
```

The system functions generator needs an environment with the extras (`duckdb`, `aiohttp`, `websockets`), and runs the examples in the docstrings while it generates. Write to a temporary file first, as above, so that a failure does not leave an empty page. A new module that defines system functions must be added to `MODULES` in the script, and a new function needs an entry in `DESC` and `GROUPS` or it falls into "Other" with its first docstring sentence as its description.

## One rule about torch

Torch is imported only in `klongpy/backends/torch_backend.py`. No other module, including the interpreter, the compiler and the operators, may `import torch`. `tests/test_backend_isolation.py` checks it. Other code detects the backend like this:

* To test for an array or tensor, use `klong._backend.np.ndarray`. It is `torch.Tensor` on the torch backend and `numpy.ndarray` on numpy.
* To detect the torch backend, use `hasattr(klong._backend, '_torch_backend')`.
* To reach the torch module when the torch backend is already active, use `sys.modules.get('torch')`.
* For an operation that differs between backends, add or call a backend method, for example `klong._backend.np.cumsum`.

This keeps torch optional and avoids import-time failures where it is not installed.

## The Rust workspace

You need a Rust toolchain (stable; the bare-metal checks below also use the nightly toolchain). Dependencies other than the workspace crates are one crate, `libm`.

```text
$ cd native
$ cargo build --release
$ cargo test --release --workspace
```

`cargo build --release` produces `native/target/release/klong` (the language), `klong-sim` and `klong-board-image`. The test run takes a few seconds once the build is cached. See [Building and running](native/building.md).

One test, `data_nesting_is_bounded` in `native/klong-core/tests/full.rs`, fails about one run in ten. If it fails and you did not touch the evaluator, rerun it. See [Roadmap](roadmap.md#known-limits).

### Bare-metal link checks

The core and the board skeleton must link for bare-metal targets. `native/scripts/measure.sh` builds the host binaries, the `no_std` core without default features, the probe for `wasm32-unknown-unknown`, `riscv32imac-unknown-none-elf` and `thumbv7em-none-eabihf`, and the board image for the last two, and prints the section sizes. A target that is not installed is skipped (`rustup +stable target add TARGET`).

```text
$ sh native/scripts/measure.sh
```

The script takes about a minute and a half with a warm cache. To check one target only:

```text
$ cd native
$ cargo +stable build -p klong-board --profile min-size --target thumbv7em-none-eabihf --target-dir target/board-console
$ cargo +stable build -p klong-core --no-default-features
```

### The firmware build

The ESP32-P4 firmware is built by `native/boards/esp32p4/build.sh`, which needs Docker with the ESP-IDF image pinned in the script and the `riscv32imafc-unknown-none-elf` Rust target. The steps, the flashing commands and the board's facts are in `native/boards/esp32p4/README.md` and [ESP32-P4](native/esp32-p4.md).

### Conformance runs

The native runtime, the console and the shim are all run against the same cases. See [Conformance suite](conformance.md) for the commands and counts.

## Documentation

The site is built with MkDocs and the Material theme. `mkdocs.yml` holds the navigation; a page that is not in it, or a link that does not resolve, fails the strict build.

```text
$ mkdocs build --strict -d /tmp/site
```

To preview while you edit, run `mkdocs serve` and open the address it prints .

### Versioned docs

The published site keeps one directory per release line, built with [mike](https://github.com/jimporter/mike) on the `gh-pages` branch. The workflow `.github/workflows/gh-pages.yml` does this:

* A push to `main` (or `master`) installs `".[docs]"` and `mike>=2.1`, runs `mkdocs build --strict` as a check, then publishes the docs of that commit as the version `major.minor` of `pyproject.toml` and moves the alias `latest` to it (`mike deploy --push --update-aliases VERSION latest`, then `mike set-default --push latest`). Older versions stay on the branch. The `gh-pages` branch is then uploaded to GitHub Pages.
* A manual run (`workflow_dispatch`) takes two optional inputs: `ref`, a commit, tag or branch, and `version`, the name to publish it as, for example `0.7`. With a `version` it runs `mike deploy --push VERSION` and leaves `latest` where it is. Use it to keep the docs of the previous release. If the older tree has no version selector in its `mkdocs.yml`, the workflow adds `extra.version.provider: mike` for that build.

`mkdocs.yml` sets `extra.version.provider: mike`, which adds the version selector to the header. The site URL is `https://www.klongpy.org/`.

To preview the versioned site without touching the real `gh-pages` branch, deploy to a scratch branch, serve it, and delete the branch afterwards. Do not push it.

```text
$ mike deploy -b mike-scratch 0.7-preview latest -u
$ mike list -b mike-scratch
0.7-preview [latest]
$ mike serve -b mike-scratch -a 127.0.0.1:18765
$ git branch -D mike-scratch
```

`mike deploy` only writes to the named branch; it does not change your working tree.

## What CI runs

Two workflows in `.github/workflows/`.

`unittests.yml` runs on a push to `main`, on every pull request and by hand:

| Job | What it does |
|---|---|
| `linter` | Python 3.12. `flake8 klongpy/` with the selected error codes `F,E112,E113,E304,E502,E702,E703,E71,E72,W191,W6,W605`. |
| `testcpu` | A matrix of Python 3.9, 3.10, 3.11, 3.12, 3.13 and 3.14 on Ubuntu, with `pip install -e ".[web,db,ws,repl,dev]"`. Every leg except 3.12 runs `python -m pytest tests/ -v --ignore=tests/test_compiler.py`. The 3.12 leg installs CPU torch, runs the whole suite under `coverage`, and on a pull request checks that new lines are at least 80% covered (`diff-cover`). |
| `testtorch` | Python 3.14 with CPU torch: `python -m pytest tests/ -v --backend torch --device cpu`. |
| `testbackend` | Python 3.14: `python -m pytest tests/ -v --backend test_backend --ignore=tests/test_compiler.py`. |

Because `tests/test_docs_examples.py` is part of `tests/`, these jobs also run the documentation checker on every page; blocks that need a package that is not installed are skipped.

`gh-pages.yml` publishes the docs, as described above, on a push to `main` and by hand.

CI does not build the Rust workspace, does not run `cargo test`, does not run `conformance/run.py` against the native binary or the console, and does not build the firmware. The native test files in `tests/` skip there because the binary is not built. CI does not run the PeachQ tests or any test on macOS or MPS.

## See also

* [Conformance suite](conformance.md)
* [Native runtime](native/index.md) and [Porting to a board](native/porting.md)
* [Roadmap](roadmap.md)
* [Compiling Klong to C](embedded.md)
