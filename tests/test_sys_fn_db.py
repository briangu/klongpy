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


    def test_multi_table(self):
        s = """
.py("klongpy.db")
a::[1 2 3]
b::[2 3 4]
c::[3 4 5]

d:::{}
d,"a",,a
d,"b",,b
t::.table(d)

e:::{}
e,"c",,c
u::.table(e)

q:::{}
q,"T",,t
q,"G",,u
db::.db(q)
"""
        klong = KlongInterpreter()
        klong(s)
        r = klong('db("select * from T")')
        self.assertTrue(kg_equal(r, np.array([[1,2],[2,3],[3,4]])))
        r = klong('db("select * from G")')
        self.assertTrue(kg_equal(r, np.array([[3],[4],[5]])))
        r = klong('db("select * from T join G on G.c = T.b")')
        self.assertTrue(kg_equal(r, np.array([[2,3,3],[3,4,4]])))
