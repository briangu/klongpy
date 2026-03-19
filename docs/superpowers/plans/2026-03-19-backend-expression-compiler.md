# Backend Expression Compiler Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the expression compiler so `compiler.py` produces backend-neutral IR and each backend generates its own optimized code.

**Architecture:** Frontend/backend split inspired by Triton. `compiler.py` walks ASTs and emits tuple-based IR. `BackendProvider.compile_expr_ir()` takes IR and returns compiled callables. NumPy and torch backends each implement their own code generation.

**Tech Stack:** Python, numpy, torch (optional)

**Spec:** `docs/superpowers/specs/2026-03-19-backend-expression-compiler-design.md`

---

## Chunk 1: IR Generation and Base Interface

### Task 1: Add `compile_expr_ir()` to `BackendProvider`

**Files:**
- Modify: `klongpy/backends/base.py`
- Test: `tests/test_compiler.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_compiler.py`, add a test that calls `compile_expr_ir` on the base backend:

```python
class TestBackendCompileExprIr(unittest.TestCase):
    """Test BackendProvider.compile_expr_ir interface."""

    def test_base_returns_none(self):
        from klongpy.backends.numpy_backend import NumpyBackendProvider
        backend = NumpyBackendProvider()
        # Before numpy implements it, base class returns None
        ir = ('binop', '+', ('literal', 1), ('literal', 2))
        result = backend.compile_expr_ir(ir, [])
        self.assertIsNone(result)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_compiler.py::TestBackendCompileExprIr::test_base_returns_none -v`
Expected: FAIL with `AttributeError: 'NumpyBackendProvider' object has no attribute 'compile_expr_ir'`

- [ ] **Step 3: Add `compile_expr_ir` and `_collect_params` to `BackendProvider`**

In `klongpy/backends/base.py`, add after the `klong_gradcheck` method (around line 322).
`_collect_params` is shared across all backends since it is a pure IR tree walk with no backend-specific logic:

```python
def compile_expr_ir(self, ir, var_syms):
    """Compile a compute IR tree to a callable.

    Args:
        ir: Tuple-based IR tree from compiler.py.
            Node types:
            - ('literal', value)
            - ('var', param_name)
            - ('binop', op, left_ir, right_ir)  -- op: '+','-','*','%','^'
            - ('cmp', op, left_ir, right_ir)     -- op: '=','>','<'
            - ('negate', child_ir)
            - ('reduce', op, arg_ir)             -- op: '+','*','|','&'
            - ('scan', op, arg_ir)               -- op: '+','*','|','&'
        var_syms: List of KGSym variable references in parameter order.

    Returns:
        (callable, var_syms) or None. Default returns None (no compilation).
    """
    return None

@staticmethod
def _collect_params(ir):
    """Collect unique parameter names from IR tree in order."""
    params = []
    seen = set()

    def _walk(node):
        if node[0] == 'var':
            name = node[1]
            if name not in seen:
                seen.add(name)
                params.append(name)
        elif node[0] in ('binop', 'cmp'):
            _walk(node[2])
            _walk(node[3])
        elif node[0] == 'negate':
            _walk(node[1])
        elif node[0] in ('reduce', 'scan'):
            _walk(node[2])  # node[1] is the op char, node[2] is the arg IR

    _walk(ir)
    return params
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_compiler.py::TestBackendCompileExprIr::test_base_returns_none -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add klongpy/backends/base.py tests/test_compiler.py
git commit -m "feat: add compile_expr_ir to BackendProvider interface"
```

---

### Task 2: Replace `_ast_to_source` with `_ast_to_ir` in `compiler.py`

**Files:**
- Modify: `klongpy/compiler.py`
- Modify: `tests/test_compiler.py`

- [ ] **Step 1: Write IR generation tests**

**Delete the entire `TestAstToSource` class** from `tests/test_compiler.py` (lines 44-135) and replace it with `TestAstToIr`. These tests use the numpy backend (no torch required) since `_ast_to_ir` is backend-neutral. The function needs variables to be numpy arrays or scalars to pass the type check.

```python
class TestAstToIr(unittest.TestCase):
    """Test the AST-to-IR walker on parsed expressions."""

    def _get_ast(self, klong, expr):
        return klong.prog(expr)[1][0]

    def test_simple_addition(self):
        from klongpy.compiler import _ast_to_ir
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.array([1.0, 2.0, 3.0])
        klong['b'] = np.array([4.0, 5.0, 6.0])
        ast = self._get_ast(klong, 'a+b')
        var_refs = {}
        ir = _ast_to_ir(ast, klong, var_refs)
        self.assertEqual(ir[0], 'binop')
        self.assertEqual(ir[1], '+')
        self.assertEqual(ir[2], ('var', var_refs[list(var_refs.keys())[0]]))
        self.assertEqual(len(var_refs), 2)

    def test_compound_expression(self):
        from klongpy.compiler import _ast_to_ir
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.array([1.0, 2.0])
        klong['b'] = np.array([3.0, 4.0])
        ast = self._get_ast(klong, 'a*2+b')
        var_refs = {}
        ir = _ast_to_ir(ast, klong, var_refs)
        self.assertIsNotNone(ir)
        self.assertEqual(ir[0], 'binop')

    def test_division(self):
        from klongpy.compiler import _ast_to_ir
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.array([10.0])
        klong['b'] = np.array([3.0])
        ast = self._get_ast(klong, 'a%b')
        var_refs = {}
        ir = _ast_to_ir(ast, klong, var_refs)
        self.assertEqual(ir[0], 'binop')
        self.assertEqual(ir[1], '%')  # Klong operator preserved in IR

    def test_comparison_greater(self):
        from klongpy.compiler import _ast_to_ir
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.array([1.0])
        klong['b'] = np.array([2.0])
        ast = self._get_ast(klong, 'a>b')
        var_refs = {}
        ir = _ast_to_ir(ast, klong, var_refs)
        self.assertEqual(ir[0], 'cmp')
        self.assertEqual(ir[1], '>')

    def test_comparison_less(self):
        from klongpy.compiler import _ast_to_ir
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.array([1.0])
        klong['b'] = np.array([2.0])
        ast = self._get_ast(klong, 'a<b')
        var_refs = {}
        ir = _ast_to_ir(ast, klong, var_refs)
        self.assertEqual(ir[0], 'cmp')
        self.assertEqual(ir[1], '<')

    def test_comparison_equal(self):
        from klongpy.compiler import _ast_to_ir
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.array([1.0])
        klong['b'] = np.array([2.0])
        ast = self._get_ast(klong, 'a=b')
        var_refs = {}
        ir = _ast_to_ir(ast, klong, var_refs)
        self.assertEqual(ir[0], 'cmp')
        self.assertEqual(ir[1], '=')

    def test_negate(self):
        from klongpy.compiler import _ast_to_ir
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.array([1.0, -2.0])
        ast = self._get_ast(klong, '-a')
        var_refs = {}
        ir = _ast_to_ir(ast, klong, var_refs)
        self.assertEqual(ir[0], 'negate')

    def test_reduce(self):
        from klongpy.compiler import _ast_to_ir
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.array([1.0, 2.0, 3.0])
        ast = self._get_ast(klong, '+/a')
        var_refs = {}
        ir = _ast_to_ir(ast, klong, var_refs)
        self.assertEqual(ir[0], 'reduce')
        self.assertEqual(ir[1], '+')

    def test_scan(self):
        from klongpy.compiler import _ast_to_ir
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.array([1.0, 2.0, 3.0])
        ast = self._get_ast(klong, '+\\a')
        var_refs = {}
        ir = _ast_to_ir(ast, klong, var_refs)
        self.assertEqual(ir[0], 'scan')
        self.assertEqual(ir[1], '+')

    def test_repeated_variable_single_param(self):
        from klongpy.compiler import _ast_to_ir
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.array([1.0, 2.0])
        ast = self._get_ast(klong, 'a*a')
        var_refs = {}
        ir = _ast_to_ir(ast, klong, var_refs)
        self.assertIsNotNone(ir)
        self.assertEqual(len(var_refs), 1)

    def test_unsupported_op_returns_none(self):
        from klongpy.compiler import _ast_to_ir
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.array([1.0])
        ast = self._get_ast(klong, 'a@0')
        var_refs = {}
        ir = _ast_to_ir(ast, klong, var_refs)
        self.assertIsNone(ir)

    def test_literal_in_expression(self):
        from klongpy.compiler import _ast_to_ir
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.array([1.0, 2.0])
        ast = self._get_ast(klong, 'a+1')
        var_refs = {}
        ir = _ast_to_ir(ast, klong, var_refs)
        self.assertIsNotNone(ir)
        # One child should be a literal
        self.assertTrue(
            ir[2] == ('literal', 1) or ir[3] == ('literal', 1)
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_compiler.py::TestAstToIr -v`
Expected: FAIL with `ImportError: cannot import name '_ast_to_ir'`

- [ ] **Step 3: Implement `_ast_to_ir` and update `compile_expr`**

Replace the contents of `klongpy/compiler.py`:

```python
"""
Expression compiler for KlongPy.

Compiles Klong expression ASTs to a backend-neutral IR (tuple tree),
then delegates to the active backend for platform-specific code generation.
Falls back to None (interpreter handles it) for anything it can't compile.
"""
from .types import KGSym, KGFn, KGCall, KGOp, KGAdverb, reserved_fn_symbols


# Arithmetic ops the IR supports
_ARITH_OPS = {'+', '-', '*', '%', '^'}

# Comparison ops
_CMP_OPS = {'>', '<', '='}

# Supported reduce/scan ops
_REDUCE_SCAN_OPS = {'+', '*', '|', '&'}


def compile_expr(ast, klong):
    """Try to compile an AST node to a callable.

    Returns (compiled_fn, [var_syms]) or None.
    """
    var_refs = {}
    ir = _ast_to_ir(ast, klong, var_refs)
    if ir is None:
        return None
    if not var_refs:
        return None  # pure constant — not worth compiling

    var_syms = list(var_refs.keys())
    return klong._backend.compile_expr_ir(ir, var_syms)


def _ast_to_ir(node, klong, var_refs):
    """Walk AST and emit IR tuples. Returns IR tree or None."""
    t = type(node)

    # Literals
    if t is int or t is float:
        return ('literal', node)

    # Symbols -> variable references
    if t is KGSym:
        if node in reserved_fn_symbols:
            return None
        try:
            val = klong._context[node]
        except KeyError:
            return None
        tv = type(val)
        if tv is int or tv is float:
            if node not in var_refs:
                var_refs[node] = f'_v{len(var_refs)}'
            return ('var', var_refs[node])
        if isinstance(val, klong._backend.np.ndarray):
            if node not in var_refs:
                var_refs[node] = f'_v{len(var_refs)}'
            return ('var', var_refs[node])
        return None

    # Operator expressions
    if isinstance(node, KGFn) and node.is_op():
        op_char = node.a.a
        arity = node.a.arity

        if arity == 2:
            args = node.args
            if not isinstance(args, list):
                args = [args] if args is not None else None
            if args is None or len(args) != 2:
                return None
            left = _ast_to_ir(args[0], klong, var_refs)
            right = _ast_to_ir(args[1], klong, var_refs)
            if left is None or right is None:
                return None
            if op_char in _ARITH_OPS:
                return ('binop', op_char, left, right)
            if op_char in _CMP_OPS:
                return ('cmp', op_char, left, right)
            return None

        if arity == 1 and op_char == '-':
            arg = node.args
            if isinstance(arg, list):
                arg = arg[0]
            child = _ast_to_ir(arg, klong, var_refs)
            if child is None:
                return None
            return ('negate', child)

    # Adverb chains: op/arg (reduce) and op\arg (scan)
    if isinstance(node, KGCall) and node.is_adverb_chain():
        chain = node.a
        if isinstance(chain, list) and len(chain) == 3:
            verb_adv = chain[0]
            adverb = chain[1]
            arg = chain[2]
            if (isinstance(verb_adv, KGAdverb) and isinstance(verb_adv.a, KGOp)
                    and isinstance(adverb, KGAdverb)):
                op_char = verb_adv.a.a
                adv_char = adverb.a
                if op_char not in _REDUCE_SCAN_OPS:
                    return None
                arg_ir = _ast_to_ir(arg, klong, var_refs)
                if arg_ir is None:
                    return None
                if adv_char == '/':
                    return ('reduce', op_char, arg_ir)
                elif adv_char == '\\':
                    return ('scan', op_char, arg_ir)

    return None
```

- [ ] **Step 4: Run IR tests to verify they pass**

Run: `python -m pytest tests/test_compiler.py::TestAstToIr -v`
Expected: PASS

- [ ] **Step 5: Run fallback tests**

The `TestCompilerFallback` tests need updating. `test_returns_none_when_no_torch` should now return a result (numpy backend compiles too, once implemented). But since `NumpyBackendProvider.compile_expr_ir()` still returns `None` (inherits base), this test should still pass for now.

Run: `python -m pytest tests/test_compiler.py::TestCompilerFallback -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add klongpy/compiler.py tests/test_compiler.py
git commit -m "refactor: replace _ast_to_source with _ast_to_ir in compiler"
```

---

## Chunk 2: NumPy Backend Compiler

### Task 3: Implement `compile_expr_ir` for NumPy backend

**Files:**
- Modify: `klongpy/backends/numpy_backend.py`
- Modify: `tests/test_compiler.py`

- [ ] **Step 1: Write failing tests for numpy compilation**

Add to `tests/test_compiler.py`:

```python
class TestNumpyBackendCompiler(unittest.TestCase):
    """Test numpy backend's compile_expr_ir."""

    def _compile_and_run(self, klong, expr):
        """Compile and execute, return (compiled_result, interp_result)."""
        from klongpy.compiler import compile_expr
        import numpy as np
        ast = klong.prog(expr)[1][0]
        result = compile_expr(ast, klong)
        self.assertIsNotNone(result, f"Failed to compile: {expr}")
        fn, var_syms = result
        args = [klong._context[s] for s in var_syms]
        compiled_val = fn(*args)
        interp_val = klong.eval(ast)
        return compiled_val, interp_val

    def test_add(self):
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.array([1.0, 2.0, 3.0])
        klong['b'] = np.array([4.0, 5.0, 6.0])
        compiled, interp = self._compile_and_run(klong, 'a+b')
        np.testing.assert_array_almost_equal(compiled, interp)

    def test_subtract(self):
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.random.randn(100)
        klong['b'] = np.random.randn(100)
        compiled, interp = self._compile_and_run(klong, 'a-b')
        np.testing.assert_array_almost_equal(compiled, interp)

    def test_divide(self):
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.random.randn(100)
        klong['b'] = np.random.randn(100) + 1.0
        compiled, interp = self._compile_and_run(klong, 'a%b')
        np.testing.assert_array_almost_equal(compiled, interp)

    def test_power(self):
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.abs(np.random.randn(100)) + 0.1
        compiled, interp = self._compile_and_run(klong, 'a^2')
        np.testing.assert_array_almost_equal(compiled, interp)

    def test_comparison(self):
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.random.randn(100)
        klong['b'] = np.random.randn(100)
        compiled, interp = self._compile_and_run(klong, 'a>b')
        np.testing.assert_array_equal(compiled, interp)

    def test_negate(self):
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.random.randn(100)
        compiled, interp = self._compile_and_run(klong, '-a')
        np.testing.assert_array_almost_equal(compiled, interp)

    def test_compound_expression(self):
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.random.randn(100)
        klong['b'] = np.random.randn(100)
        compiled, interp = self._compile_and_run(klong, 'a*2+b')
        np.testing.assert_array_almost_equal(compiled, interp)

    def test_sum_reduce(self):
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.random.randn(100)
        compiled, interp = self._compile_and_run(klong, '+/a')
        np.testing.assert_almost_equal(float(compiled), float(interp))

    def test_product_reduce(self):
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.random.rand(10) + 0.5
        compiled, interp = self._compile_and_run(klong, '*/a')
        np.testing.assert_almost_equal(float(compiled), float(interp))

    def test_max_reduce(self):
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.random.randn(100)
        compiled, interp = self._compile_and_run(klong, '|/a')
        np.testing.assert_almost_equal(float(compiled), float(interp))

    def test_min_reduce(self):
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.random.randn(100)
        compiled, interp = self._compile_and_run(klong, '&/a')
        np.testing.assert_almost_equal(float(compiled), float(interp))

    def test_cumsum_scan(self):
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.random.randn(100)
        compiled, interp = self._compile_and_run(klong, '+\\a')
        np.testing.assert_array_almost_equal(compiled, interp)

    def test_cumprod_scan(self):
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.random.rand(20) + 0.5
        compiled, interp = self._compile_and_run(klong, '*\\a')
        np.testing.assert_array_almost_equal(compiled, interp)

    def test_cummax_scan_returns_none(self):
        """Numpy lacks cummax — compiler should return None, interpreter handles it."""
        from klongpy.compiler import compile_expr
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.random.randn(100)
        ast = klong.prog('|\\a')[1][0]
        result = compile_expr(ast, klong)
        self.assertIsNone(result)

    def test_cummin_scan_returns_none(self):
        """Numpy lacks cummin — compiler should return None, interpreter handles it."""
        from klongpy.compiler import compile_expr
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.random.randn(100)
        ast = klong.prog('&\\a')[1][0]
        result = compile_expr(ast, klong)
        self.assertIsNone(result)

    def test_fused_sum_product(self):
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.random.randn(100)
        klong['b'] = np.random.randn(100)
        compiled, interp = self._compile_and_run(klong, '+/a*b')
        np.testing.assert_almost_equal(float(compiled), float(interp))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_compiler.py::TestNumpyBackendCompiler -v`
Expected: FAIL — `compile_expr` returns `None` because numpy backend's `compile_expr_ir` returns `None`

- [ ] **Step 3: Implement `compile_expr_ir` in NumPy backend**

Add to `klongpy/backends/numpy_backend.py`, as a method on `NumpyBackendProvider`:

```python
def compile_expr_ir(self, ir, var_syms):
    """Compile IR tree to a callable using numpy operations."""
    source = self._ir_to_source(ir)
    if source is None:
        return None

    param_names = list(self._collect_params(ir))
    fn_source = f"def _expr({', '.join(param_names)}): return {source}"
    ns = {'np': np}
    try:
        exec(fn_source, ns)
    except Exception:
        return None
    return (ns['_expr'], var_syms)

def _ir_to_source(self, ir):
    """Convert IR tree to Python source string with numpy operations."""
    node_type = ir[0]

    if node_type == 'literal':
        return repr(ir[1])

    if node_type == 'var':
        return ir[1]  # param name like '_v0'

    if node_type == 'binop':
        op, left, right = ir[1], ir[2], ir[3]
        l = self._ir_to_source(left)
        r = self._ir_to_source(right)
        if l is None or r is None:
            return None
        py_op = {'+': '+', '-': '-', '*': '*', '%': '/', '^': '**'}.get(op)
        if py_op is None:
            return None
        return f'({l}{py_op}{r})'

    if node_type == 'cmp':
        op, left, right = ir[1], ir[2], ir[3]
        l = self._ir_to_source(left)
        r = self._ir_to_source(right)
        if l is None or r is None:
            return None
        py_cmp = {'=': '==', '>': '>', '<': '<'}.get(op)
        if py_cmp is None:
            return None
        return f'(({l}{py_cmp}{r})*1)'

    if node_type == 'negate':
        child = self._ir_to_source(ir[1])
        if child is None:
            return None
        return f'(-{child})'

    if node_type == 'reduce':
        op, arg = ir[1], ir[2]
        arg_src = self._ir_to_source(arg)
        if arg_src is None:
            return None
        method = {'+': 'np.sum', '*': 'np.prod', '|': 'np.max', '&': 'np.min'}.get(op)
        if method is None:
            return None
        return f'{method}({arg_src})'

    if node_type == 'scan':
        op, arg = ir[1], ir[2]
        arg_src = self._ir_to_source(arg)
        if arg_src is None:
            return None
        method = {'+': 'np.cumsum', '*': 'np.cumprod'}.get(op)
        if method is None:
            return None  # |\ and &\ not supported in numpy
        return f'{method}({arg_src})'

    return None
```

Note: `_collect_params` is inherited from `BackendProvider` (added in Task 1).

- [ ] **Step 4: Run numpy compiler tests**

Run: `python -m pytest tests/test_compiler.py::TestNumpyBackendCompiler -v`
Expected: PASS

- [ ] **Step 5: Update fallback test**

In `tests/test_compiler.py`, update `test_returns_none_when_no_torch`:

```python
def test_numpy_backend_compiles_expressions(self):
    from klongpy.compiler import compile_expr
    from klongpy import KlongInterpreter
    import numpy as np
    # numpy backend now compiles — should return a result for array expressions
    klong = KlongInterpreter(backend='numpy')
    klong['a'] = np.array([1.0, 2.0])
    klong['b'] = np.array([3.0, 4.0])
    ast = klong.prog("a+b")[1][0]
    result = compile_expr(ast, klong)
    self.assertIsNotNone(result)
    fn, var_syms = result
    args = [klong._context[s] for s in var_syms]
    np.testing.assert_array_equal(fn(*args), np.array([4.0, 6.0]))
```

Note: The old `test_returns_none_when_no_torch` (which tested `1+2` with no array variables) should still return `None` because it has no variable references (pure constants). Keep that as a separate test or rename for clarity:

```python
def test_returns_none_for_pure_constant_expr(self):
    from klongpy.compiler import compile_expr
    from klongpy import KlongInterpreter
    klong = KlongInterpreter(backend='numpy')
    ast = klong.prog("1+2")[1][0]
    result = compile_expr(ast, klong)
    self.assertIsNone(result)  # No variable refs — not worth compiling
```

- [ ] **Step 6: Run all fallback tests**

Run: `python -m pytest tests/test_compiler.py::TestCompilerFallback -v`
Expected: PASS

- [ ] **Step 7: Update `TestEvalIntegration.test_numpy_backend_eval_unchanged`**

The numpy backend now compiles `+/a`, so this test should verify the compiled path works:

```python
def test_numpy_backend_eval_compiled(self):
    """Eval on numpy backend should compile reduce expressions."""
    from klongpy import KlongInterpreter
    import numpy as np
    klong = KlongInterpreter(backend='numpy')
    klong['a'] = np.array([1.0, 2.0, 3.0])
    ast = klong.prog('+/a')[1][0]
    result = klong.eval(ast)
    self.assertAlmostEqual(float(result), 6.0)
```

- [ ] **Step 8: Run full test suite**

Run: `python -m pytest tests/test_compiler.py -v`
Expected: PASS (except torch-dependent tests if torch not installed)

- [ ] **Step 9: Commit**

```bash
git add klongpy/backends/numpy_backend.py tests/test_compiler.py
git commit -m "feat: implement numpy backend expression compiler"
```

---

## Chunk 3: Torch Backend Compiler

### Task 4: Implement `compile_expr_ir` for torch backend

**Files:**
- Modify: `klongpy/backends/torch_backend.py`
- Modify: `tests/test_compiler.py`

- [ ] **Step 1: Implement `compile_expr_ir` in torch backend**

Add to `TorchBackendProvider` in `klongpy/backends/torch_backend.py`:

```python
def compile_expr_ir(self, ir, var_syms):
    """Compile IR tree to a callable using torch tensor operations."""
    source = self._ir_to_source(ir)
    if source is None:
        return None

    param_names = list(self._collect_params(ir))
    fn_source = f"def _expr({', '.join(param_names)}): return {source}"
    ns = {}
    try:
        exec(fn_source, ns)
    except Exception:
        return None
    return (ns['_expr'], var_syms)

def _ir_to_source(self, ir):
    """Convert IR tree to Python source string with torch operations."""
    node_type = ir[0]

    if node_type == 'literal':
        return repr(ir[1])

    if node_type == 'var':
        return ir[1]

    if node_type == 'binop':
        op, left, right = ir[1], ir[2], ir[3]
        l = self._ir_to_source(left)
        r = self._ir_to_source(right)
        if l is None or r is None:
            return None
        py_op = {'+': '+', '-': '-', '*': '*', '%': '/', '^': '**'}.get(op)
        if py_op is None:
            return None
        return f'({l}{py_op}{r})'

    if node_type == 'cmp':
        op, left, right = ir[1], ir[2], ir[3]
        l = self._ir_to_source(left)
        r = self._ir_to_source(right)
        if l is None or r is None:
            return None
        py_cmp = {'=': '==', '>': '>', '<': '<'}.get(op)
        if py_cmp is None:
            return None
        return f'(({l}{py_cmp}{r})*1)'

    if node_type == 'negate':
        child = self._ir_to_source(ir[1])
        if child is None:
            return None
        return f'(-{child})'

    if node_type == 'reduce':
        op, arg = ir[1], ir[2]
        arg_src = self._ir_to_source(arg)
        if arg_src is None:
            return None
        method = {'+': 'sum', '*': 'prod', '|': 'max', '&': 'min'}.get(op)
        if method is None:
            return None
        return f'({arg_src}).{method}()'

    if node_type == 'scan':
        op, arg = ir[1], ir[2]
        arg_src = self._ir_to_source(arg)
        if arg_src is None:
            return None
        methods = {
            '+': 'cumsum(0)',
            '*': 'cumprod(0)',
            '|': 'cummax(0).values',
            '&': 'cummin(0).values',
        }
        method = methods.get(op)
        if method is None:
            return None
        return f'({arg_src}).{method}'

    return None
```

Note: `_collect_params` is inherited from `BackendProvider` (added in Task 1).

- [ ] **Step 2: Run existing torch tests**

Run: `python -m pytest tests/test_compiler.py::TestCompileExprCorrectness tests/test_compiler.py::TestCompileReductions tests/test_compiler.py::TestEvalIntegration tests/test_compiler.py::TestInterpreterIntegration -v`
Expected: PASS — all existing torch correctness, reduction, eval, and integration tests pass through the new IR pipeline

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/test_compiler.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add klongpy/backends/torch_backend.py
git commit -m "feat: implement torch backend expression compiler"
```

---

### Task 5: Update `test_returns_none_for_numpy_arrays_on_torch_backend`

**Files:**
- Modify: `tests/test_compiler.py`

The test at line 31 checks that numpy arrays on the torch backend don't compile. This should still pass because the IR walker checks `isinstance(val, klong._backend.np.ndarray)` which is `torch.Tensor` on the torch backend, and numpy arrays won't match.

- [ ] **Step 1: Verify the test passes**

Run: `python -m pytest tests/test_compiler.py::TestCompilerFallback::test_returns_none_for_numpy_arrays_on_torch_backend -v`
Expected: PASS

- [ ] **Step 2: If it fails, investigate and fix**

The test should still pass because `klong._backend.np.ndarray` is `torch.Tensor` on the torch backend, and `np.array` is not a `torch.Tensor`. No code change expected.

---

## Chunk 4: Cleanup and Documentation

### Task 6: Remove old `_ast_to_source` references from tests

**Files:**
- Modify: `tests/test_compiler.py`

- [ ] **Step 1: Verify no remaining references to `_ast_to_source`**

Search for any remaining `_ast_to_source` references in the test file. The `TestAstToSource` class was replaced by `TestAstToIr` in Task 2. Ensure no imports or calls remain.

Run: `grep -n "_ast_to_source" tests/test_compiler.py`
Expected: No output (no remaining references)

- [ ] **Step 2: Run the full project test suite**

Run: `python -m unittest discover tests`
Expected: ALL PASS

- [ ] **Step 3: Commit if any cleanup was needed**

```bash
git add tests/test_compiler.py
git commit -m "chore: remove stale _ast_to_source test references"
```

---

### Task 7: Update README.md benchmarks

**Files:**
- Modify: `README.md`
- Reference: `examples/bench_compiler.kg`

- [ ] **Step 1: Run benchmarks on numpy backend**

Run: `python -m klongpy --backend numpy examples/bench_compiler.kg`
Capture the output.

- [ ] **Step 2: Run benchmarks on torch backend**

Run: `python -m klongpy --backend torch --device cpu examples/bench_compiler.kg`
Capture the output.

- [ ] **Step 3: Update the Performance section in README.md**

Update the benchmark numbers in the Performance section (around lines 140-176) with fresh results. Update the description to mention that both numpy and torch backends now have expression compilers, not just torch.

Key text changes:
- Update the section that says the torch backend includes an expression compiler — now both backends have one
- Update benchmark numbers with fresh results
- Keep the same table format

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: update README benchmarks for backend expression compiler"
```

---

### Task 8: Final verification

- [ ] **Step 1: Run full test suite**

Run: `python -m unittest discover tests`
Expected: ALL PASS

- [ ] **Step 2: Verify torch isolation**

Run: `grep -rn "import torch" klongpy/ --include="*.py" | grep -v torch_backend | grep -v __pycache__`
Expected: No output — torch is only imported in `torch_backend.py`

- [ ] **Step 3: Verify compiler.py has no backend-specific code**

Run: `grep -n "torch\|numpy\|\.sum\|\.cumsum\|\.prod\|\.max\|\.min" klongpy/compiler.py`
Expected: No output — compiler.py is backend-neutral

- [ ] **Step 4: Final commit if needed, then done**
