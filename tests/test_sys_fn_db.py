import unittest

from utils import *
from klongpy import KlongInterpreter


class TestSysFnDb(unittest.TestCase):

    def test_initial_prototype(self):
        s = """
.py("klongpy.db")
a::[1 2 3]
b::[2 3 4]
c::[3 4 5]
d:::{}
d,"a",,a
d,"b",,b
t::.table(d)
t,"c",,c
q:::{}
q,"T",,t
db::.db(q)
db("select * from T")
"""
        klong = KlongInterpreter()
        r = klong(s)
        self.assertTrue(kg_equal(r, np.array([[1,2,3],[2,3,4],[3,4,5]])))
