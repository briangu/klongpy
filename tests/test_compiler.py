import unittest


class TestCompilerFallback(unittest.TestCase):
    """Test that the compiler returns None for things it can't compile."""

    def test_returns_none_for_integer_literal(self):
        from klongpy.compiler import compile_expr
        from klongpy import KlongInterpreter
        klong = KlongInterpreter()
        # Pure integer — no variables, not worth compiling
        result = compile_expr(42, klong)
        self.assertIsNone(result)

    def test_returns_none_for_string(self):
        from klongpy.compiler import compile_expr
        from klongpy import KlongInterpreter
        klong = KlongInterpreter()
        result = compile_expr("hello", klong)
        self.assertIsNone(result)

    def test_returns_none_for_pure_constant_expr(self):
        from klongpy.compiler import compile_expr
        from klongpy import KlongInterpreter
        # 1+2 has no variable refs — not worth compiling
        klong = KlongInterpreter(backend='numpy')
        ast = klong.prog("1+2")[1][0]
        result = compile_expr(ast, klong)
        self.assertIsNone(result)

    def test_numpy_backend_compiles_expressions(self):
        from klongpy.compiler import compile_expr
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.array([1.0, 2.0, 3.0])
        klong['b'] = np.array([4.0, 5.0, 6.0])
        ast = klong.prog("a+b")[1][0]
        result = compile_expr(ast, klong)
        self.assertIsNotNone(result, "numpy backend should compile array expressions")

    def test_returns_none_for_numpy_arrays_on_torch_backend(self):
        from klongpy.compiler import compile_expr
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='torch')
        klong['a'] = np.array([1.0, 2.0])
        klong['b'] = np.array([3.0, 4.0])
        ast = klong.prog("a+b")[1][0]
        # Must return None — compiling would bypass torch backend coercion
        result = compile_expr(ast, klong)
        self.assertIsNone(result)


class TestBackendCompileExprIr(unittest.TestCase):
    """Test BackendProvider.compile_expr_ir interface."""

    def test_base_returns_none(self):
        from klongpy.backends.base import BackendProvider
        # Create a minimal concrete subclass that doesn't override compile_expr_ir
        class _StubBackend(BackendProvider):
            @property
            def name(self): return 'stub'
            @property
            def np(self): return None
            def supports_object_dtype(self): return False
            def supports_strings(self): return False
            def supports_float64(self): return False
            def str_to_char_array(self, s): pass
            def kg_asarray(self, a): pass
            def is_array(self, x): return False
            def is_backend_array(self, x): return False
            def get_dtype_kind(self, arr): return None
            def to_numpy(self, x): return x
            def is_scalar_integer(self, x): return False
            def is_scalar_float(self, x): return False
            def argsort(self, a, descending=False): pass
            def array_size(self, a): return 0
        backend = _StubBackend()
        ir = ('binop', '+', ('literal', 1), ('literal', 2))
        result = backend.compile_expr_ir(ir, [])
        self.assertIsNone(result)

    def test_numpy_backend_compiles_ir(self):
        from klongpy.backends.numpy_backend import NumpyBackendProvider
        backend = NumpyBackendProvider()
        ir = ('binop', '+', ('var', '_v0'), ('var', '_v1'))
        result = backend.compile_expr_ir(ir, ['a', 'b'])
        self.assertIsNotNone(result)


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
        self.assertEqual(ir[1], '%')

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
        self.assertTrue(
            ir[2] == ('literal', 1) or ir[3] == ('literal', 1)
        )


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


class TestCompileExprCorrectness(unittest.TestCase):
    """Test that compiled functions produce correct results."""

    def _get_ast(self, klong, expr):
        return klong.prog(expr)[1][0]

    def _assert_compiled_matches_interp(self, klong, expr):
        """Compile expr, run both paths, compare results."""
        from klongpy.compiler import compile_expr
        import torch
        ast = self._get_ast(klong, expr)
        result = compile_expr(ast, klong)
        self.assertIsNotNone(result, f"Failed to compile: {expr}")
        fn, var_syms = result
        args = [klong._context[s] for s in var_syms]
        compiled_val = fn(*args)
        interp_val = klong.eval(ast)
        # Normalize to CPU tensors for comparison (devices may differ)
        def _to_cpu(v):
            if isinstance(v, torch.Tensor):
                return v.cpu()
            return torch.tensor(v)
        c = _to_cpu(compiled_val)
        i = _to_cpu(interp_val)
        # Dtypes should match (e.g., comparisons should be int, not bool)
        self.assertEqual(c.dtype, i.dtype,
            f"Dtype mismatch for: {expr} — compiled={c.dtype}, interp={i.dtype}")
        torch.testing.assert_close(c, i, msg=f"Mismatch for: {expr}")

    def test_add(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.randn(100)
        klong['b'] = torch.randn(100)
        self._assert_compiled_matches_interp(klong, 'a+b')

    def test_subtract(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.randn(100)
        klong['b'] = torch.randn(100)
        self._assert_compiled_matches_interp(klong, 'a-b')

    def test_multiply(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.randn(100)
        klong['b'] = torch.randn(100)
        self._assert_compiled_matches_interp(klong, 'a*b')

    def test_divide(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.randn(100)
        klong['b'] = torch.randn(100) + 1.0
        self._assert_compiled_matches_interp(klong, 'a%b')

    def test_power(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.rand(100) + 0.1
        self._assert_compiled_matches_interp(klong, 'a^2')

    def test_comparison(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.randn(100)
        klong['b'] = torch.randn(100)
        self._assert_compiled_matches_interp(klong, 'a>b')

    def test_negate(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.randn(100)
        self._assert_compiled_matches_interp(klong, '-a')

    def test_compound_expression(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.randn(1000)
        klong['b'] = torch.randn(1000)
        self._assert_compiled_matches_interp(klong, 'a*2+b')

    def test_complex_expression(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.randn(1000) + 2.0
        klong['b'] = torch.randn(1000)
        self._assert_compiled_matches_interp(klong, '(a*2+b)%(a+1)')

    def test_mixed_scalar_tensor(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.randn(100)
        klong['alpha'] = 0.5
        self._assert_compiled_matches_interp(klong, 'a*alpha')


class TestCompileReductions(unittest.TestCase):
    """Test compilation of reduce (op/) and scan (op\\) expressions."""

    def _get_ast(self, klong, expr):
        return klong.prog(expr)[1][0]

    def _assert_compiled_matches_interp(self, klong, expr):
        from klongpy.compiler import compile_expr
        import torch
        ast = self._get_ast(klong, expr)
        result = compile_expr(ast, klong)
        self.assertIsNotNone(result, f"Failed to compile: {expr}")
        fn, var_syms = result
        args = [klong._context[s] for s in var_syms]
        compiled_val = fn(*args)
        interp_val = klong.eval(ast)
        def _to_cpu(v):
            if isinstance(v, torch.Tensor):
                return v.cpu()
            return torch.tensor(v)
        torch.testing.assert_close(
            _to_cpu(compiled_val).float(),
            _to_cpu(interp_val).float(),
            msg=f"Mismatch for: {expr}"
        )

    def test_sum_reduce(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.randn(100)
        self._assert_compiled_matches_interp(klong, '+/a')

    def test_product_reduce(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.rand(10) + 0.5  # keep positive for stable product
        self._assert_compiled_matches_interp(klong, '*/a')

    def test_max_reduce(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.randn(100)
        self._assert_compiled_matches_interp(klong, '|/a')

    def test_min_reduce(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.randn(100)
        self._assert_compiled_matches_interp(klong, '&/a')

    def test_fused_sum_product(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.randn(100)
        klong['b'] = torch.randn(100)
        self._assert_compiled_matches_interp(klong, '+/a*b')

    def test_fused_max_expression(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.randn(100)
        klong['b'] = torch.randn(100)
        self._assert_compiled_matches_interp(klong, '|/a-b')

    def test_cumsum_scan(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.randn(100)
        self._assert_compiled_matches_interp(klong, '+\\a')

    def test_cumprod_scan(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.rand(20) + 0.5
        self._assert_compiled_matches_interp(klong, '*\\a')

    def test_running_max_scan(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.randn(100)
        self._assert_compiled_matches_interp(klong, '|\\a')

    def test_running_min_scan(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.randn(100)
        self._assert_compiled_matches_interp(klong, '&\\a')

    def test_unsupported_reduce_returns_none(self):
        from klongpy.compiler import compile_expr
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.randn(10)
        # -/ is not a standard reduce we support
        ast = self._get_ast(klong, '-/a')
        result = compile_expr(ast, klong)
        self.assertIsNone(result)


class TestEvalIntegration(unittest.TestCase):
    """Test that eval() uses the compiler for adverb chains."""

    def test_scan_in_loop_uses_compiled(self):
        """|\a inside a loop should use torch.cummax, not element-by-element."""
        from klongpy import KlongInterpreter
        import torch, time
        klong = KlongInterpreter(backend='torch', device='cpu')
        klong['a'] = torch.randn(100000)
        # Run |\a via eval (simulates inner-loop usage)
        ast = klong.prog('|\\a')[1][0]
        t0 = time.perf_counter()
        for _ in range(10):
            result = klong.eval(ast)
        elapsed = time.perf_counter() - t0
        # Should complete in well under 1 second (compiled: ~0.01s, interpreter: ~50s)
        self.assertLess(elapsed, 1.0, f"|\\ too slow ({elapsed:.2f}s) — compiler not active in eval()")
        # Verify correctness
        expected = torch.cummax(klong['a'].cpu(), 0).values
        torch.testing.assert_close(result.cpu(), expected)

    def test_reduce_in_loop_uses_compiled(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch', device='cpu')
        klong['a'] = torch.randn(100000)
        ast = klong.prog('+/a')[1][0]
        result = klong.eval(ast)
        expected = klong['a'].sum()
        torch.testing.assert_close(result.cpu().float(), expected.cpu().float())

    def test_fused_reduce_in_eval(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch', device='cpu')
        klong['a'] = torch.randn(1000)
        klong['b'] = torch.randn(1000)
        ast = klong.prog('+/a*b')[1][0]
        result = klong.eval(ast)
        expected = (klong['a'] * klong['b']).sum()
        torch.testing.assert_close(result.cpu().float(), expected.cpu().float())

    def test_cumsum_in_eval(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch', device='cpu')
        klong['a'] = torch.randn(1000)
        ast = klong.prog('+\\a')[1][0]
        result = klong.eval(ast)
        expected = torch.cumsum(klong['a'], 0)
        torch.testing.assert_close(result.cpu().float(), expected.cpu().float())

    def test_numpy_backend_eval_compiled(self):
        """Eval on numpy backend should compile reduce expressions."""
        from klongpy import KlongInterpreter
        import numpy as np
        klong = KlongInterpreter(backend='numpy')
        klong['a'] = np.array([1.0, 2.0, 3.0])
        ast = klong.prog('+/a')[1][0]
        result = klong.eval(ast)
        self.assertAlmostEqual(float(result), 6.0)


class TestInterpreterIntegration(unittest.TestCase):
    """Test that __call__ uses the compiler when available."""

    def test_call_uses_compiled_path(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.tensor([1.0, 2.0, 3.0])
        klong['b'] = torch.tensor([4.0, 5.0, 6.0])
        result = klong('a+b')
        torch.testing.assert_close(result.cpu(), torch.tensor([5.0, 7.0, 9.0]))

    def test_call_populates_compiled_cache(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.tensor([1.0, 2.0])
        klong['b'] = torch.tensor([3.0, 4.0])
        klong('a+b')
        self.assertIn(('a+b', None), klong._compiled_cache)

    def test_cache_cleared_on_setitem(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.tensor([1.0])
        klong['b'] = torch.tensor([2.0])
        klong('a+b')
        self.assertIn(('a+b', None), klong._compiled_cache)
        klong['a'] = torch.tensor([10.0])
        self.assertEqual(len(klong._compiled_cache), 0)

    def test_cache_cleared_on_delitem(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.tensor([1.0])
        klong['b'] = torch.tensor([2.0])
        klong('a+b')
        self.assertIn(('a+b', None), klong._compiled_cache)
        del klong['a']
        self.assertEqual(len(klong._compiled_cache), 0)

    def test_type_change_does_not_produce_wrong_result(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.tensor([10.0, 20.0])
        klong['b'] = torch.tensor([1.0, 2.0])
        r1 = klong('a+b')
        torch.testing.assert_close(r1.cpu(), torch.tensor([11.0, 22.0]))
        # Rebind to lists — cache must be invalidated so we don't get
        # list concatenation from the compiled lambda
        klong['a'] = [10, 20]
        klong['b'] = [1, 2]
        r2 = klong('a+b')
        # Interpreter returns element-wise addition, not list concat
        self.assertFalse(isinstance(r2, list) and r2 == [10, 20, 1, 2],
            "Compiled path returned list concatenation instead of element-wise addition")

    def test_numpy_backend_still_works(self):
        from klongpy import KlongInterpreter
        klong = KlongInterpreter(backend='numpy')
        result = klong('1+2')
        self.assertEqual(result, 3)

    def test_multi_statement_falls_through(self):
        from klongpy import KlongInterpreter
        klong = KlongInterpreter(backend='torch')
        result = klong('a::5;b::3;a+b')
        self.assertEqual(int(result), 8)

    def test_assignment_falls_through(self):
        from klongpy import KlongInterpreter
        klong = KlongInterpreter(backend='torch')
        klong('a::42')
        self.assertEqual(int(klong['a']), 42)


if __name__ == '__main__':
    unittest.main()
