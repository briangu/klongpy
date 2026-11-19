"""The backend abstraction.

* importing klongpy must not change process-wide numpy state;
* one KGChar class;
* operator modules may not grow new direct numpy uses (the budgets below can
  only go down; each remaining use is one the backend mechanism
  could not take over mechanically);
* torch is imported only by klongpy/backends/torch_backend.py.
"""
import ast
import os
import subprocess
import sys
import textwrap
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "klongpy")

# Direct `np.x`, `numpy.x` and `bknp.x` attribute uses (AST walk) left in each operator module.
NUMPY_BUDGET = {
    "adverbs.py": 0,
    "monads.py": 16,
    "dyads.py": 17,
}
NUMPY_NAMES = {"np", "numpy", "bknp"}


def count_direct_numpy(path):
    with open(path) as f:
        tree = ast.parse(f.read())
    return sum(1 for n in ast.walk(tree)
               if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in NUMPY_NAMES)


def run_python(code):
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", textwrap.dedent(code)],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    return r


class TestNoGlobalNumpyState(unittest.TestCase):
    def test_import_does_not_touch_numpy_globals(self):
        r = run_python("""
            import warnings
            import numpy as np
            before = (np.geterr(), list(warnings.filters), hasattr(np, 'isarray'))
            import klongpy
            from klongpy.backends.numpy_backend import NumpyBackendProvider
            from klongpy import KlongInterpreter
            NumpyBackendProvider()
            k = KlongInterpreter()
            k('1+2'); k('%[0 1 2]'); k('[1 2 3]%0')
            after = (np.geterr(), list(warnings.filters), hasattr(np, 'isarray'))
            assert before[2] is False
            assert before == after, (before, after)
            print('ok')
        """)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ok", r.stdout)

    def test_backend_namespace_has_isarray(self):
        from klongpy.backends import get_backend
        import numpy
        np_ = get_backend('numpy').np
        self.assertTrue(np_.isarray(numpy.zeros(2)))
        self.assertFalse(np_.isarray([1]))
        self.assertFalse(hasattr(numpy, 'isarray'))

    def test_divide_by_zero_does_not_warn(self):
        import warnings
        from klongpy import KlongInterpreter
        k = KlongInterpreter()
        for src in ("%[0 1 2]", "0^-1.0", "x::[1.0 2];x%0", "%/[1.0 0]", "%\\[1.0 0]"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    k(src)
                except RuntimeWarning as e:
                    self.fail(f"{src}: {e}")
                except Exception:
                    pass  # a Klong error is fine, a numpy warning is not


class TestSingleKGChar(unittest.TestCase):
    def test_single_kgchar_class(self):
        from klongpy import types
        from klongpy.backends import numpy_backend
        self.assertIs(types.KGChar, numpy_backend.KGChar)

    def test_is_char(self):
        from klongpy.types import KGChar, is_char
        self.assertTrue(is_char(KGChar("a")))
        self.assertFalse(is_char("a"))


class TestOperatorNumpyBudget(unittest.TestCase):
    def test_direct_numpy_uses_do_not_grow(self):
        for name, budget in NUMPY_BUDGET.items():
            n = count_direct_numpy(os.path.join(PKG, name))
            self.assertLessEqual(n, budget,
                                 f"{name}: {n} direct numpy uses, budget {budget}; route through backend.np "
                                 f"(and lower the budget when you remove one)")

    def test_budget_is_tight(self):
        # keeps the budget honest: lower NUMPY_BUDGET when uses are removed
        for name, budget in NUMPY_BUDGET.items():
            n = count_direct_numpy(os.path.join(PKG, name))
            self.assertEqual(n, budget, f"{name}: {n} uses, budget {budget}; lower the budget")


class TestTorchIsolation(unittest.TestCase):
    def test_torch_imported_only_by_torch_backend(self):
        offenders = []
        for dirpath, _, files in os.walk(PKG):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                if path == os.path.join(PKG, "backends", "torch_backend.py"):
                    continue
                with open(path) as f:
                    tree = ast.parse(f.read())
                for n in ast.walk(tree):
                    mods = []
                    if isinstance(n, ast.Import):
                        mods = [a.name for a in n.names]
                    elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                        mods = [n.module]
                    if any(m == "torch" or m.startswith("torch.") for m in mods):
                        offenders.append(f"{os.path.relpath(path, ROOT)}:{n.lineno}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
