"""
Tests that validate reported bugs in the core interpreter/parser.

Each test is expected to FAIL against the current (unfixed) code,
proving the bug is real. After fixes are applied, all tests should pass.
"""
import os
import sys
import tempfile
import unittest

from klongpy import KlongInterpreter
from klongpy.core import KGSym
from klongpy.parser import read_num, read_sys_comment


class TestBugP1_SiblingModuleImports(unittest.TestCase):
    """
    P1: import_file_module() executes the target module before adding its
    directory to sys.path, so sibling imports inside the loaded module fail
    with ModuleNotFoundError.
    """

    def test_file_import_resolves_sibling_module(self):
        """A .py file loaded via .py() should be able to import a sibling module."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a helper module
            helper_path = os.path.join(tmpdir, "myhelper.py")
            with open(helper_path, "w") as f:
                f.write("def greet():\n    return 'hello from helper'\n")

            # Create a main module that imports the sibling
            main_path = os.path.join(tmpdir, "mymain.py")
            with open(main_path, "w") as f:
                f.write("import myhelper\n")
                f.write("def sayhello():\n    return myhelper.greet()\n")

            klong = KlongInterpreter()
            # This should not raise ModuleNotFoundError
            klong(f'.py("{main_path}")')
            result = klong("sayhello()")
            self.assertEqual(result, "hello from helper")

    def test_file_import_siblings_from_different_directories(self):
        """Loading modules with distinct sibling names from different dirs
        should each resolve their own siblings correctly."""
        with tempfile.TemporaryDirectory() as dir_a, \
             tempfile.TemporaryDirectory() as dir_b:
            # dir_a has helperA
            with open(os.path.join(dir_a, "helperA.py"), "w") as f:
                f.write("def greet():\n    return 'A'\n")
            with open(os.path.join(dir_a, "mainA.py"), "w") as f:
                f.write("import helperA\ndef sayhello():\n    return helperA.greet()\n")

            # dir_b has helperB
            with open(os.path.join(dir_b, "helperB.py"), "w") as f:
                f.write("def greet():\n    return 'B'\n")
            with open(os.path.join(dir_b, "mainB.py"), "w") as f:
                f.write("import helperB\ndef sayhello():\n    return helperB.greet()\n")

            klong_a = KlongInterpreter()
            klong_a(f'.py("{os.path.join(dir_a, "mainA.py")}")')
            self.assertEqual(klong_a("sayhello()"), "A")

            klong_b = KlongInterpreter()
            klong_b(f'.py("{os.path.join(dir_b, "mainB.py")}")')
            self.assertEqual(klong_b("sayhello()"), "B")

    def test_file_import_adds_module_dir_to_sys_path(self):
        """Loading a file module should add its directory to sys.path
        so that spawned subprocesses can resolve siblings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "mymain.py"), "w") as f:
                f.write("def sayhello():\n    return 'hi'\n")

            klong = KlongInterpreter()
            klong(f'.py("{os.path.join(tmpdir, "mymain.py")}")')
            self.assertIn(tmpdir, sys.path)


class TestBugP2_NoneContextLookup(unittest.TestCase):
    """
    P2: KlongContext.__getitem__ uses `if v is not None` which means
    a variable explicitly set to None cannot be retrieved.
    """

    def test_none_value_is_retrievable(self):
        """A variable assigned None (via Python interop) should be readable."""
        klong = KlongInterpreter()
        klong["x"] = None
        # Should return None, not raise KeyError
        result = klong["x"]
        self.assertIsNone(result)

    def test_none_value_does_not_fall_through_to_outer_scope(self):
        """A None value in inner scope should not fall through to outer scope."""
        klong = KlongInterpreter()
        klong["x"] = 42
        # Overwrite with None — should shadow the outer 42
        klong["x"] = None
        result = klong["x"]
        self.assertIsNone(result)

    def test_python_function_returning_none(self):
        """Calling a Python function that returns None and storing the result
        should allow the result to be retrieved."""
        klong = KlongInterpreter()
        klong["pynone"] = lambda: None
        # r::pynone(); r should be None, not undefined
        klong("r::pynone()")
        result = klong["r"]
        self.assertIsNone(result)


class TestBugP2_UnterminatedComment(unittest.TestCase):
    """
    P2: read_sys_comment() returns a RuntimeError instead of raising it
    when the end marker is missing, causing a downstream TypeError.
    """

    def test_unterminated_comment_raises_error(self):
        """An unterminated .comment() should raise RuntimeError, not return one."""
        # The text after the .comment() call has no end marker
        text = "some text without end marker"
        marker = "END"
        with self.assertRaises(RuntimeError):
            read_sys_comment(text, 0, marker)

    def test_unterminated_comment_via_interpreter(self):
        """Parsing an unterminated .comment() via the interpreter should give
        a clear error, not a confusing TypeError."""
        klong = KlongInterpreter()
        with self.assertRaises(RuntimeError):
            klong('.comment("END")\nsome text but no end marker')


class TestBugP3_ScientificNotationPlus(unittest.TestCase):
    """
    P3: read_num() handles 1e3 and 1e-3 but does not handle 1e+3,
    causing a ValueError.
    """

    def test_parse_1e_plus_3(self):
        """1e+3 should parse as 1000.0"""
        i, val = read_num("1e+3", 0)
        self.assertEqual(val, 1000.0)
        self.assertEqual(i, 4)  # consumed all 4 chars

    def test_parse_2_5e_plus_2(self):
        """2.5e+2 should parse as 250.0"""
        i, val = read_num("2.5e+2", 0)
        self.assertEqual(val, 250.0)
        self.assertEqual(i, 6)

    def test_parse_neg_1e_plus_3(self):
        """-1e+3 should parse as -1000.0"""
        i, val = read_num("-1e+3", 0)
        self.assertEqual(val, -1000.0)
        self.assertEqual(i, 5)

    def test_existing_notation_still_works(self):
        """Ensure 1e3 and 1e-3 still work after any fix."""
        i, val = read_num("1e3", 0)
        self.assertEqual(val, 1000.0)

        i, val = read_num("1e-3", 0)
        self.assertEqual(val, 0.001)

    def test_1e_plus_3_via_interpreter(self):
        """The interpreter should evaluate 1e+3 as 1000.0"""
        klong = KlongInterpreter()
        result = klong("1e+3")
        self.assertEqual(result, 1000.0)


class TestBugP1_ModuleRequalification(unittest.TestCase):
    """
    P1: read_sym() qualifies every non-dot symbol with the current module,
    including the argument to .module(). So .module(:m2) while m1 is active
    turns the module name into m2`m1 instead of m2.
    """

    def test_nested_module_switch(self):
        """Switching modules inside a module should use the new name, not a
        qualified version of it."""
        klong = KlongInterpreter()
        klong('.module(:m1)')
        klong('.module(:m2)')
        self.assertEqual(str(klong.current_module()), 'm2')

    def test_module_then_exit_then_reenter(self):
        """After exiting a module, entering a new one should work correctly."""
        klong = KlongInterpreter()
        klong('.module(:m1)')
        klong('a::1')
        klong('.module(0)')
        klong('.module(:m2)')
        klong('a::2')
        klong('.module(0)')
        # Both module-qualified names should exist via Python API
        self.assertEqual(klong[KGSym('a`m1')], 1)
        self.assertEqual(klong[KGSym('a`m2')], 2)


class TestBugP1_ParseCacheModuleSensitive(unittest.TestCase):
    """
    P1: __call__() keys _parse_cache only by raw source string, but parsing
    depends on current_module(). The same source in different modules reuses
    a stale AST.
    """

    def test_same_source_different_modules(self):
        """The same source string evaluated in different modules should
        resolve to the correct module's binding."""
        klong = KlongInterpreter()
        klong('.module(:m1)')
        klong('a::10')
        klong('.module(0)')
        klong('.module(:m2)')
        klong('a::20')
        klong('.module(0)')
        # Both should return the correct value, not a cached stale one
        self.assertEqual(klong[KGSym('a`m1')], 10)
        self.assertEqual(klong[KGSym('a`m2')], 20)


class TestBugP2_MixedLocalDeclaration(unittest.TestCase):
    """
    P2: The {[locals]; ...} detection only checks whether the first list
    contains at least one symbol, not whether every element is a symbol.
    So [a 1] is misread as a local declaration.
    """

    def test_array_literal_not_treated_as_locals(self):
        """A function whose body starts with a mixed array [a 1] should
        not have that array stripped as a local declaration."""
        klong = KlongInterpreter()
        klong('a::42')
        result = klong('{[a 1];a}()')
        # Should return 42 (global a), not the symbol :a
        self.assertEqual(result, 42)

    def test_pure_symbol_locals_still_work(self):
        """Legitimate local declarations like {[p q]; ...} should still work."""
        klong = KlongInterpreter()
        result = klong('{[p];p::5;p}()')
        self.assertEqual(result, 5)


class TestBugP2_NoneEvalPath(unittest.TestCase):
    """
    P2: The evaluator (not just the public API) should see None bindings.
    KlongContext.__getitem__() treats None as missing, so the evaluator
    falls back to the raw symbol.
    """

    def test_none_visible_in_klong_eval(self):
        """k['n']=None; k('n') should evaluate to None, not the symbol :n."""
        klong = KlongInterpreter()
        klong['n'] = None
        result = klong('n')
        self.assertIsNone(result)

    def test_none_assignment_in_klong(self):
        """Assigning None from a Python function and reading it back via eval."""
        klong = KlongInterpreter()
        klong['pynone'] = lambda: None
        klong('r::pynone()')
        result = klong('r')
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
