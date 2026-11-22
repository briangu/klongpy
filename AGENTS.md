# Repo guidelines

## Development setup

```bash
pip install -e ".[dev]"
```

## Running tests

```bash
python3 -m pytest tests/
```

## Code guidelines

1. Never use embedded imports (imports inside functions, methods, or conditional blocks) except for traceback.
2. Don't reference torch outside of `klongpy/backends/torch_backend.py`.

## Use kg to develop kg

Use the checkout's native `kg` in the development loop. For a change someone can observe from Klong, first write a small `.kg` program that demonstrates the expected behavior. Run it against the current binary, rebuild, and run the same program again. Keep regression cases with the existing Klong suites. From the repo root, `native/target/release/kg tests/v2/programs/test_language.kg` runs one of those suites directly.

Prefer Klong for new development tools that fit the language, including fixture generation, array transformations, and result checks. Use Python for Python interoperability and independent reference checks, and Rust or C for runtime and board interfaces. When a missing `kg` capability forces a workaround, capture the gap in a minimal reproducer that a future change can run.

As you use `kg`, keep the [RC2 dogfooding notes](tests/v2/rc2-dogfooding.md) up to date. Record what you tried, what happened, and the workaround or regression check. Mark an entry fixed when you verify the fix against its reproducer.

The [post-freeze RC2 notes](tests/v2/rc2-dogfooding-post-rc1.md) cover board console checks and the first run of a script from the Intel Mac wheel. The [Rosetta startup measurements](docs/performance.md#first-launch-under-rosetta) include comparisons with warm queries.

### Running native kg from an agent

- Install from source with `cargo install --locked --path native/klong-cli`. For checkout tests, build with `cargo build --locked --release --manifest-path native/Cargo.toml -p klong-cli` and invoke `native/target/release/kg`.
- Query `kg --capabilities` for the installed runtime's JSON feature list. Use `kg -e EXPR` for a single expression and `kg FILE [ARG ...]` for scripts. Pass arguments as a vector. Files get their arguments in `.a`.
- Have the script print `.p(.json(result))` for machine-readable output. Check its exit status before consuming stdout. Runtime errors include source positions on stderr. `kg ast` and `kg lint` inspect generated code; `kg check` checks static-profile code.
- Both desktop commands use the Rust engine in `native/klong-core` and `native/klong-rt`. The standalone CLI is in `native/klong-cli`. The Python bridge lives in `native/klong-py` and `klongpy/rust_engine.py`. See `tests/v2/README.md` for release acceptance commands.

## Architecture quick map

- Both commands parse and evaluate Klong in `native/klong-core/src/`. Start with `lex.rs` and `parse.rs` for syntax, `interp.rs` for evaluation, `ops.rs` and `adverbs.rs` for primitives, and `value.rs` for value representation.
- `native/klong-rt/src/tables.rs` exposes the portable table operations implemented in `native/klong-core/src/table.rs`. Desktop SQL lives in `native/klong-host/src/database/`.
- `klongpy/interpreter.py` exports the Python-facing `KlongInterpreter` implemented in `klongpy/rust_engine.py`. The Python objects representing functions, symbols and calls live in `klongpy/types.py`.
- `klongpy/backends/`: backend registry + providers (NumPy default, PyTorch optional).
- `klongpy/autograd.py`: numeric gradients + torch autograd helpers.
- `klongpy/sys_fn*.py`, `klongpy/sys_var.py`: system functions/vars and IO helpers.
- `klongpy/repl.py`, `klongpy/cli.py`: REPL + CLI entrypoint (`kgpy`).
- `klongpy/db`, `klongpy/web`, `klongpy/ws`: optional extras (DuckDB, HTTP server, WebSockets).
- `klongpy/lib/*.kg` contains standard library modules loaded with `.l`.
- `tests/`: unit + perf tests.

## Backend architecture (important for edits)

- `klongpy/backends/base.py` defines `BackendProvider` (dtype support, conversion, array ops).
- `klongpy/backends/__init__.py` registers backends and exposes `get_backend()`.
- `NumpyBackendProvider` is the default; supports object dtype + strings.
- `TorchBackendProvider` is optional; uses `TorchBackend` to supply a numpy-like API, no object dtype/strings, supports autograd + device selection.
- `KlongInterpreter` owns a backend instance (`self._backend`) and exposes `self.np` as the backend’s numpy-like module.
- `klongpy/rust_host.py` installs the selected provider's hooks in the native engine. `klongpy/backends/host_ops.py` implements operations on provider-owned values. `klongpy/writer.py` formats values displayed from Python.
- `klongpy/backend.py` is legacy compatibility: module-level `np` always maps to the default numpy backend; new code should use per-interpreter `backend`.

## Optional subsystems and how they wire in

- REPL (`klongpy/repl.py`) sets `.system` in the interpreter with asyncio loops; IPC/web/ws modules expect this.
- IPC (`klongpy/sys_fn_ipc.py`): asyncio TCP, kdb+ IPC protocol only (`klongpy/ipc/`); the 0.7 pickle protocol was removed in 2.0.
- Web server (`klongpy/web/sys_fn_web.py`): aiohttp handlers that wrap Klong functions.
- WebSockets (`klongpy/ws/sys_fn_ws.py`): websockets client/server helpers; dispatches into the Klong loop.
- DB (`klongpy/db/sys_fn_db.py` + `sys_fn_kvs.py`): DuckDB + pandas tables; `Table` and `Database` types.

## Known issues / review notes (as of 2026-02-03)

- Logging formatting bug in `klongpy/web/sys_fn_web.py` (uses `logging.info("msg", value)` with no format placeholder).
- IPC/WS modules still use sentinel `np.inf` for “undefined” (see TODOs in `sys_fn_ipc.py`/`sys_fn_ws.py`).
- Web/WS/IPC expect `.system` loop handles; using `KlongInterpreter()` directly without REPL setup will raise lookup errors unless `.system` is set manually.
