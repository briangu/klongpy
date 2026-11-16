"""scripts/gen_system_functions.py documents every registered system function."""
import importlib.util
import os
import unittest

from klongpy import KlongInterpreter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Registered by the interpreter but not functions: handler hooks that a program
# defines (see ipc_capabilities.md) and aliases of the channel variables.
NOT_FUNCTIONS = {".srv.auth", ".srv.c", ".srv.e", ".srv.o", ".sys.cin", ".sys.cout", ".sys.cerr"}


def load_generator():
    spec = importlib.util.spec_from_file_location("gen_system_functions",
                                                  os.path.join(ROOT, "scripts", "gen_system_functions.py"))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        mod.collect()
    except ImportError as e:      # an optional extra (duckdb, aiohttp, websockets) is missing
        raise unittest.SkipTest(f"the generator needs the optional extras: {e}")
    return mod


class TestGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = load_generator()
        cls.items = cls.gen.collect()

    def test_every_registered_system_function_is_collected(self):
        k = KlongInterpreter()
        registered = {str(n) for n, _ in k._context if str(n).startswith(".")}
        missing = sorted(registered - set(self.items) - NOT_FUNCTIONS)
        self.assertEqual(missing, [])

    def test_static_module_is_listed(self):
        self.assertIn("klongpy.sys_fn_static", self.gen.MODULES)
        self.assertIn(".entry", self.items)
        self.assertIn(".extern", self.items)

    def test_every_collected_function_has_a_description_and_a_group(self):
        grouped = {n for _, _, names in self.gen.GROUPS for n in names.split()}
        for n in self.items:
            with self.subTest(name=n):
                if n in (".clit", ".grad"):
                    continue    # fall back to the first docstring sentence and the "Other" group
                self.assertIn(n, self.gen.DESC)
                self.assertIn(n, grouped)

    def test_the_page_has_a_section_for_each(self):
        with open(os.path.join(ROOT, "docs", "system-functions.md"), encoding="utf-8") as f:
            page = f.read()
        for n in self.items:
            with self.subTest(name=n):
                self.assertIn(f"### `{n}`", page)


if __name__ == "__main__":
    unittest.main()
