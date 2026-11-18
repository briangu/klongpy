"""`.table` takes a dictionary of columns; `#` counts the rows of a table and the entries of a store."""
import shutil
import tempfile
import unittest

try:
    import pandas  # noqa: F401
    import duckdb  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False

from klongpy import KlongInterpreter
from klongpy.errors import KlongShapeError, KlongTypeError


@unittest.skipUnless(HAVE, "requires pandas and duckdb")
class TestTableFromDictionary(unittest.TestCase):
    def setUp(self):
        self.k = KlongInterpreter()
        self.k('.py("klongpy.db")')

    def test_columns_in_dictionary_order(self):
        self.k('t::.table(:{;"b";[1 2 3];"a";[4 5 6]})')
        self.assertEqual(list(self.k(".schema(t)")), ["b", "a"])
        self.assertEqual(list(self.k('t?"a"')), [4, 5, 6])
        self.assertEqual(self.k("#t"), 3)

    def test_same_table_as_the_pair_list(self):
        self.k('u::.table([["a" [1 2 3]] ["b" [2 3 4]]]);v::.table(:{;"a";[1 2 3];"b";[2 3 4]})')
        self.assertEqual(str(self.k("u")), str(self.k("v")))

    def test_errors_are_klong_errors(self):
        with self.assertRaises(KlongShapeError) as cm:
            self.k('.table(:{;"a";[1 2 3];"b";[1 2]})')
        self.assertIn("different lengths", str(cm.exception))
        for src in ('.table(:{;1;[1 2]})', '.table(:{;:a;[1 2]})', '.table(:{;"a";5})'):
            with self.assertRaises(KlongTypeError, msg=src):
                self.k(src)


@unittest.skipUnless(HAVE, "requires pandas and duckdb")
class TestSizeOfTablesAndStores(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.k = KlongInterpreter()
        self.k('.py("klongpy.db")')

    def test_sizes(self):
        k = self.k
        k('t::.table(:{;"a";[1 2 3]})')
        k(f'tbs::.tables("{self.dir}/t");kv::.kvs("{self.dir}/k")')
        self.assertEqual(k("#t"), 3)
        self.assertEqual((k("#tbs"), k("#kv")), (0, 0))
        k('tbs,:{;"one";t;"two";t}')
        k('kv,:{;"x";[1 2];"y";"s";"z";5}')
        self.assertEqual((k("#tbs"), k("#kv")), (2, 3))

    def test_no_operator_gives_an_internal_error(self):
        k = self.k
        k('t::.table(:{;"a";[1 2 3]})')
        k(f'tbs::.tables("{self.dir}/t");kv::.kvs("{self.dir}/k")')
        for v in ("t", "tbs", "kv"):
            for op in ("#V", "*V", "|V", "@V", "V?\"a\"", "V,V", "$V", ",V", "_V", "%V", "&V", "-V", "!V"):
                src = op.replace("V", v)
                try:
                    k(src)
                except KlongTypeError:
                    pass
                except Exception as e:      # an internal error, or a stray Python exception
                    self.fail(f"{src}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    unittest.main()
