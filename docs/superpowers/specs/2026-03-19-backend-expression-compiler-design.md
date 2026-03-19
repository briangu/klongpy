# Backend Expression Compiler

## Problem

`compiler.py` hardcodes torch-specific code generation (tensor method calls like `.sum()`, `.cumsum(0)`) and gates compilation on `hasattr(klong._backend, '_torch_backend')`. This violates the backend abstraction: the compiler knows about a specific backend, other backends can't participate in compilation, and adding a new backend requires modifying the compiler.

## Goal

Refactor the expression compiler to follow a Triton-inspired frontend/backend separation:
- **Frontend** (`compiler.py`): walks the AST, produces a backend-neutral IR
- **Backend** (`BackendProvider` subclasses): takes the IR, generates optimized code for its platform

Non-math expressions (Python interop, I/O, library calls) pass through to the interpreter unchanged.

## IR Format

A tree of tuples representing compute operations. Pure data, no backend references:

```python
('literal', value)                  # int or float
('var', symbol, param_name)         # variable reference, e.g. ('var', KGSym('a'), '_v0')
('binop', op, left_ir, right_ir)    # binary op: '+', '-', '*', '%', '^'
('cmp', op, left_ir, right_ir)      # comparison: '=', '>', '<' (Klong operators)
('negate', child_ir)                # unary negation
('reduce', op, arg_ir)              # reduce adverb: '+', '*', '|', '&'
('scan', op, arg_ir)                # scan adverb: '+', '*', '|', '&'
```

The IR uses Klong operator characters (`%` for division, `^` for power) — backends map these to their platform's operations.

## Changes

### `compiler.py`

Replace `_ast_to_source()` with `_ast_to_ir()`:
- Same AST walking logic
- Emits IR tuples instead of Python source strings
- No backend-specific knowledge — no operator mappings, no method names

`compile_expr(ast, klong)`:
- Calls `_ast_to_ir(ast, klong, var_refs)` to produce IR
- If IR is `None` or no variable references, returns `None`
- Delegates to `klong._backend.compile_expr_ir(ir, var_syms)` for code generation
- Returns `(callable, var_syms)` or `None`

Removed from this module:
- `_KLONG_TO_PY`, `_KLONG_CMP_OPS` mappings
- `_REDUCE_TO_METHOD`, `_REDUCE_TO_VALMETHOD` mappings
- `_SCAN_TO_METHOD`, `_SCAN_TO_VALMETHOD` mappings
- `hasattr(klong._backend, '_torch_backend')` check
- All `exec()` calls and Python source generation

### `backends/base.py`

Add to `BackendProvider`:

```python
def compile_expr_ir(self, ir, var_syms):
    """Compile a compute IR tree to a callable.

    Args:
        ir: Tuple-based IR tree from compiler.py
        var_syms: List of KGSym variable references in parameter order

    Returns:
        (callable, var_syms) or None
    """
    return None
```

Default returns `None` — future backends opt in to compilation.

### `backends/numpy_backend.py`

Implements `compile_expr_ir()`:
- Walks IR tree, emits Python source using numpy operations
- Arithmetic operators: uses Python operators (numpy overloads `+`, `-`, `*`, `/`, `**`)
- Comparisons: uses Python operators with `*1` for bool-to-numeric conversion
- Reduce: `np.sum()`, `np.prod()`, `np.max()`, `np.min()`
- Scan: `np.cumsum()`, `np.cumprod()`. Returns `None` for `|\\` and `&\\` (cummax/cummin) since numpy lacks these — interpreter handles them
- Injects `numpy` module as `np` into `exec()` namespace so generated code like `np.sum(_v0)` resolves
- Catches `SyntaxError`/`Exception` from `exec()` and returns `None` (graceful degradation)
- Returns `(callable, var_syms)` or `None`

### `backends/torch_backend.py`

Implements `compile_expr_ir()`:
- Walks IR tree, emits Python source using torch tensor methods
- Arithmetic operators: uses Python operators (torch overloads them)
- Comparisons: uses Python operators with `*1`
- Reduce: `.sum()`, `.prod()`, `.max()`, `.min()`
- Scan: `.cumsum(0)`, `.cumprod(0)`, `.cummax(0).values`, `.cummin(0).values`
- No extra imports needed (methods are on the tensor objects)
- Catches `SyntaxError`/`Exception` from `exec()` and returns `None` (graceful degradation)
- Returns `(callable, var_syms)` or `None`

### `interpreter.py`

No changes. Two existing call sites remain as-is:
- `eval()`: calls `compile_expr(x, self)` for adverb chains (reduce/scan)
- `__call__()`: calls `compile_expr(cached, self)` for all single expressions (arithmetic, comparisons, negation, adverb chains) with caching

Cache and invalidation logic unchanged.

## Compilation flow

```
Expression: a+b*c
         |
   [compiler.py: _ast_to_ir]
         |
   ('binop', '+',
     ('var', a, '_v0'),
     ('binop', '*',
       ('var', b, '_v1'),
       ('var', c, '_v2')))
         |
   [backend.compile_expr_ir(ir, [a, b, c])]
         |
    +-----------+-----------+
    |                       |
  numpy:                  torch:
  "def _expr(..):           "def _expr(..):
    return _v0+_v1*_v2"       return _v0+_v1*_v2"
    (same for arithmetic)   (same for arithmetic)
         |                       |
  reduce/scan differ:      reduce/scan differ:
  "np.sum(_v0)"            "_v0.sum()"
  "np.cumsum(_v0)"         "_v0.cumsum(0)"
         |                       |
      exec() → callable    exec() → callable
         |
   [interpreter uses callable, falls back on failure]
```

## What does NOT change

- **Torch isolation**: torch imports stay in `torch_backend.py`
- **Fallback behavior**: `None` at any stage means interpreter handles it
- **Graceful degradation**: `try/except` around compiled calls, fall through on failure
- **Cache invalidation**: cleared on variable assignment/deletion
- **Supported expression scope**: same math patterns (arithmetic, comparison, negate, reduce, scan)

## Testing

### `tests/test_compiler.py`

- **IR generation tests**: verify `_ast_to_ir` produces correct IR tuples for each expression type
- **Round-trip tests**: IR → backend compilation → execution matches interpreter results
- **Both backends**: tests run with numpy and (when available) torch
- **Fallback tests**: non-math AST nodes produce `None` IR
- **Behavioral change**: `test_returns_none_when_no_torch` updates — the numpy backend now compiles expressions too, so the test becomes "numpy backend compiles with numpy ops" rather than "returns None without torch"
- **Integration tests**: existing performance and cache tests continue to pass

### New test file: `tests/test_backend_compiler.py` (optional)

- Unit tests for each backend's `compile_expr_ir()` in isolation
- Feed IR tuples directly, verify generated callables produce correct results

## File summary

| File | Change |
|------|--------|
| `compiler.py` | Replace `_ast_to_source` with `_ast_to_ir`, remove all backend-specific mappings, delegate to backend |
| `backends/base.py` | Add `compile_expr_ir()` with default `None` return |
| `backends/numpy_backend.py` | Add `compile_expr_ir()` with numpy code generation |
| `backends/torch_backend.py` | Add `compile_expr_ir()` with torch code generation |
| `interpreter.py` | No changes |
| `tests/test_compiler.py` | Update tests for IR generation + backend compilation |
