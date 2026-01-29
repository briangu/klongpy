import unittest

import numpy as np
import pandas as pd

from klongpy import KlongInterpreter
from klongpy.core import kg_equal
from tests.backend_compat import requires_strings


class TestLocalScopeBehavior(unittest.TestCase):
    """
    KlongPy uses the DuckDb's ability to scope in dataframes into table space.
    """

    @staticmethod
    def add_to_local_scope(d):
        for k,v in d.items():
            locals()[k] = v

    def test_locals_scope_behavior(self):
        d = {'hello': "world"}
        TestLocalScopeBehavior.add_to_local_scope(d)
        self.assertTrue(locals().get("hello") is None)


class TestTablePrint(unittest.TestCase):

    @requires_strings
    def test_table_print(self):
        s = """
        .py("klongpy.db")
        T::.table(,"a",,[1 2 3])
        .p(T)
        """
        klong = KlongInterpreter()
        r = klong(s)
        self.assertEqual(r, "a\n1\n2\n3")

    @requires_strings
    def test_long_table_print(self):
        s = """
        .py("klongpy.db")
        T::.table(,"a",,!100)
        .p(T)
        """
        klong = KlongInterpreter()
        r = klong(s)
        seq = "".join([str(x)+"\n" for x in range(10)])
        self.assertEqual(r, f"""a\n{seq}...\nrows=100\n""")


class TestTableDataFrame(unittest.TestCase):

    def test_table_from_empty_df(self):
        df = pd.DataFrame()
        klong = KlongInterpreter()
        klong['df'] = df
        klong('.py("klongpy.db")')
        klong('T::.table(df)')
        r = klong('#T')
        self.assertEqual(r, 0)

    def test_table_from_a_df_with_single_row_and_two_columns(self):
        data = {'col1': [1], 'col2': [3]}
        df = pd.DataFrame(data)
        klong = KlongInterpreter()
        klong['df'] = df
        klong('.py("klongpy.db")')
        klong('T::.table(df)')
        r = klong('#T')
        self.assertEqual(r, 1)
        r = klong('.schema(T)')
        self.assertTrue(kg_equal(r, ["col1", "col2"]))

    def test_table_from_a_df_with_one_column_many_rows(self):
        data = {'col1': np.arange(10)}
        df = pd.DataFrame(data)
        klong = KlongInterpreter()
        klong['df'] = df
        klong('.py("klongpy.db")')
        klong('T::.table(df)')
        r = klong('#T')
        self.assertEqual(r, 10)
        r = klong('.schema(T)')
        self.assertTrue(kg_equal(r, ["col1"]))
        r = klong('T?"col1"')
        self.assertTrue(kg_equal(r, data['col1']))
        # TODO: @ should work the same as a dictionary
        # r = klong('T@"col1"')
        # self.assertTrue(kg_equal(r, data['col1']))
