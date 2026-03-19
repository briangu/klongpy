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
        # Normalize to CPU float tensors for comparison
        # (devices may differ, comparison ops return bool vs int)
        def _to_cpu_float(v):
            if isinstance(v, torch.Tensor):
                return v.cpu().float()
            return torch.tensor(v).float()
        torch.testing.assert_close(
            _to_cpu_float(compiled_val),
            _to_cpu_float(interp_val),
            msg=f"Mismatch for: {expr}"
        )

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


if __name__ == '__main__':
    unittest.main()
