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

## Architecture quick map

- `klongpy/interpreter.py`: `KlongInterpreter` + `KlongContext`; main parse/eval loop (`prog` -> `call` -> `eval` -> `_eval_fn`).
- `klongpy/types.py`: core AST/runtime types (`KGSym`, `KGFn`, `KGCall`, `KGOp`, `KGAdverb`, etc.) + type helpers.
- `klongpy/parser.py`: lexer + parser; `kg_read_array` converts lists to backend arrays.
- `klongpy/monads.py`, `klongpy/dyads.py`, `klongpy/adverbs.py`: Klong verb/adverb semantics; most array ops delegate to backend.
- `klongpy/backends/`: backend registry + providers (NumPy default, PyTorch optional).
- `klongpy/autograd.py`: numeric gradients + torch autograd helpers.
- `klongpy/sys_fn*.py`, `klongpy/sys_var.py`: system functions/vars and IO helpers.
- `klongpy/repl.py`, `klongpy/cli.py`: REPL + CLI entrypoint (`kgpy`).
- `klongpy/db`, `klongpy/web`, `klongpy/ws`: optional extras (DuckDB, HTTP server, WebSockets).
- `klongpy/lib/*.kg`: standard library modules loaded via `.l` / `.module`.
- `tests/`: unit + perf tests.

## Backend architecture (important for edits)

- `klongpy/backends/base.py` defines `BackendProvider` (dtype support, conversion, array ops).
- `klongpy/backends/__init__.py` registers backends and exposes `get_backend()`.
- `NumpyBackendProvider` is the default; supports object dtype + strings.
- `TorchBackendProvider` is optional; uses `TorchBackend` to supply a numpy-like API, no object dtype/strings, supports autograd + device selection.
- `KlongInterpreter` owns a backend instance (`self._backend`) and exposes `self.np` as the backend’s numpy-like module.
- Parser + writer + ops are backend-aware:
  - `parser.kg_read_array()` converts list literals using `backend.kg_asarray`.
  - `monads/dyads/adverbs` call backend methods for math/array behavior.
  - `writer.kg_write()` uses backend for display conversion.
- `klongpy/backend.py` is legacy compatibility: module-level `np` always maps to the default numpy backend; new code should use per-interpreter `backend`.

## Optional subsystems and how they wire in

- REPL (`klongpy/repl.py`) sets `.system` in the interpreter with asyncio loops; IPC/web/ws modules expect this.
- IPC (`klongpy/sys_fn_ipc.py`): asyncio TCP + pickle; remote calls marshal `KGFn`/`KGLambda`.
- Web server (`klongpy/web/sys_fn_web.py`): aiohttp handlers that wrap Klong functions.
- WebSockets (`klongpy/ws/sys_fn_ws.py`): websockets client/server helpers; dispatches into the Klong loop.
- DB (`klongpy/db/sys_fn_db.py` + `sys_fn_kvs.py`): DuckDB + pandas tables; `Table` and `Database` types.

## Known issues / review notes (as of 2026-02-03)

- Logging formatting bug in `klongpy/web/sys_fn_web.py` (uses `logging.info("msg", value)` with no format placeholder).
- IPC/WS modules still use sentinel `np.inf` for “undefined” (see TODOs in `sys_fn_ipc.py`/`sys_fn_ws.py`).
- Web/WS/IPC expect `.system` loop handles; using `KlongInterpreter()` directly without REPL setup will raise lookup errors unless `.system` is set manually.
