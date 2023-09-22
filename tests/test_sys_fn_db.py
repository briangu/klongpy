import unittest

from klongpy import KlongInterpreter


def add_to_local_scope(d):
    for k,v in d.items():
        locals()[k] = v


class TestSysFnDb(unittest.TestCase):

    def test_locals_scope_behavior(self):
        """ KlongPy uses the DuckDb's ability to scope in dataframes into table space. """
        d = {'hello': "world"}
        add_to_local_scope(d)
        self.assertTrue(locals().get("hello") is None)

    def test_table_print(self):
        s = """
        .py("klongpy.db")
        T::.table(,"a",,[1 2 3])
        .p(T)
        """
        klong = KlongInterpreter()
        r = klong(s)
        self.assertEqual(r, "a\n1\n2\n3")


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
