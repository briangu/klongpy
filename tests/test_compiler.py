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

    def test_returns_none_when_no_torch(self):
        from klongpy.compiler import compile_expr
        from klongpy import KlongInterpreter
        # numpy backend — compiler should return None
        klong = KlongInterpreter(backend='numpy')
        ast = klong.prog("1+2")[1][0]
        result = compile_expr(ast, klong)
        self.assertIsNone(result)


class TestAstToSource(unittest.TestCase):
    """Test the AST-to-source walker on parsed expressions."""

    def _get_ast(self, klong, expr):
        """Parse an expression and return the AST node."""
        return klong.prog(expr)[1][0]

    def test_simple_addition(self):
        from klongpy.compiler import _ast_to_source
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.tensor([1.0, 2.0, 3.0])
        klong['b'] = torch.tensor([4.0, 5.0, 6.0])
        ast = self._get_ast(klong, 'a+b')
        var_refs = {}
        source = _ast_to_source(ast, klong, var_refs)
        self.assertIsNotNone(source)
        self.assertIn('+', source)
        self.assertEqual(len(var_refs), 2)

    def test_compound_expression(self):
        from klongpy.compiler import _ast_to_source
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.tensor([1.0, 2.0])
        klong['b'] = torch.tensor([3.0, 4.0])
        ast = self._get_ast(klong, 'a*2+b')
        var_refs = {}
        source = _ast_to_source(ast, klong, var_refs)
        self.assertIsNotNone(source)

    def test_division_maps_to_truediv(self):
        from klongpy.compiler import _ast_to_source
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.tensor([10.0])
        klong['b'] = torch.tensor([3.0])
        ast = self._get_ast(klong, 'a%b')
        var_refs = {}
        source = _ast_to_source(ast, klong, var_refs)
        self.assertIn('/', source)

    def test_negate_monad(self):
        from klongpy.compiler import _ast_to_source
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.tensor([1.0, -2.0])
        ast = self._get_ast(klong, '-a')
        var_refs = {}
        source = _ast_to_source(ast, klong, var_refs)
        self.assertIsNotNone(source)
        self.assertIn('-', source)

    def test_repeated_variable_single_param(self):
        from klongpy.compiler import _ast_to_source
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.tensor([1.0, 2.0])
        ast = self._get_ast(klong, 'a*a')
        var_refs = {}
        source = _ast_to_source(ast, klong, var_refs)
        self.assertIsNotNone(source)
        self.assertEqual(len(var_refs), 1)

    def test_unsupported_op_returns_none(self):
        from klongpy.compiler import _ast_to_source
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.tensor([1.0])
        ast = self._get_ast(klong, 'a@0')
        var_refs = {}
        source = _ast_to_source(ast, klong, var_refs)
        self.assertIsNone(source)

    def test_scalar_literal_in_expression(self):
        from klongpy.compiler import _ast_to_source
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.tensor([1.0, 2.0])
        ast = self._get_ast(klong, 'a+1')
        var_refs = {}
        source = _ast_to_source(ast, klong, var_refs)
        self.assertIsNotNone(source)
        self.assertIn('1', source)


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
        self.assertIn('a+b', klong._compiled_cache)

    def test_cache_cleared_on_setitem(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.tensor([1.0])
        klong['b'] = torch.tensor([2.0])
        klong('a+b')
        self.assertIn('a+b', klong._compiled_cache)
        klong['a'] = torch.tensor([10.0])
        self.assertEqual(len(klong._compiled_cache), 0)

    def test_cache_cleared_on_delitem(self):
        from klongpy import KlongInterpreter
        import torch
        klong = KlongInterpreter(backend='torch')
        klong['a'] = torch.tensor([1.0])
        klong['b'] = torch.tensor([2.0])
        klong('a+b')
        self.assertIn('a+b', klong._compiled_cache)
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
