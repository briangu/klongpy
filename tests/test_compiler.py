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


if __name__ == '__main__':
    unittest.main()
