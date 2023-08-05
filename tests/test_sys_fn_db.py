import unittest

def add_to_local_scope(d):
    for k,v in d.items():
        locals()[k] = v


class TestSysFnDb(unittest.TestCase):

    def test_locals_scope_behavior(self):
        """ KlongPy uses the DuckDb's ability to scope in dataframes into table space. """
        d = {'hello': "world"}
        add_to_local_scope(d)
        self.assertTrue(locals().get("hello") is None)
