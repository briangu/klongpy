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


if __name__ == '__main__':
    unittest.main()
