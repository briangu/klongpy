import time
import unittest

from utils import *

from klongpy import KlongInterpreter


def add_to_local_scope(d):
    for k,v in d.items():
        locals()[k] = v


def run_lines(s, klong=None):
    klong = klong or KlongInterpreter()
    for line in s.splitlines():
        r = klong(line)
    return r


class TestSysFnDb(unittest.TestCase):

    def test_locals_scope_behavior(self):
        d = {'hello': "world"}
        add_to_local_scope(d)
        self.assertTrue(locals().get("hello") is None)

    def test_create_empty_table(self):
        s = """
.py("klongpy.db")
t::.table(,"c",,[])
q:::{}
q,"T",,t
db::.db(q)
db("select * from T")
"""
        klong = KlongInterpreter()
        r = run_lines(s, klong)
        self.assertTrue(kg_equal(r, np.array([])))
        r = klong(".schema(t)")
        self.assertTrue(kg_equal(r, np.array(["c"], dtype=object)))

    def test_create_one_col_table(self):
        s = """
.py("klongpy.db")
a::[1 2 3]

e::[]
e::e,,"a",,a
t::.table(e)

q:::{}
q,"T",,t
db::.db(q)
db("select * from T")
"""
        klong = KlongInterpreter()
        r = run_lines(s, klong)
        self.assertTrue(kg_equal(r, np.array([1,2,3])))
        r = klong(".schema(t)")
        self.assertTrue(kg_equal(r, np.array(["a"], dtype=object)))


    def test_create_multi_col_table(self):
        s = """
.py("klongpy.db")
a::[1 2 3]
b::[2 3 4]
c::[3 4 5]

e::[]
e::e,,"a",,a
e::e,,"b",,b
e::e,,"c",,c
t::.table(e)

q:::{}
q,"T",,t
db::.db(q)
db("select * from T")
"""
        klong = KlongInterpreter()
        r = run_lines(s, klong)
        self.assertTrue(kg_equal(r, np.array([[1,2,3],[2,3,4],[3,4,5]])))
        r = klong(".schema(t)")
        self.assertTrue(kg_equal(r, np.array(["a", "b", "c"], dtype=object)))

    def test_table_via_dict_behavior(self):
        s = """
.py("klongpy.db")
a::[1 2 3]
b::[2 3 4]
c::[3 4 5]
d::[]
d::d,,"a",,a
d::d,,"b",,b
t::.table(d)
q:::{}
q,"T",,t
db::.db(q)
"""
        klong = KlongInterpreter()
        klong(s)
        r = klong('db("select * from T")')
        self.assertTrue(kg_equal(r, np.array([[1,2],[2,3],[3,4]])))
        r = klong(".schema(db)")
        self.assertEqual(len(r), 1)
        self.assertTrue("T" in r)
        self.assertTrue(kg_equal(r["T"], np.array(["a", "b"], dtype=object)))

        klong('t,"c",,c')
        r = klong('db("select * from T")')
        self.assertTrue(kg_equal(r, np.array([[1,2,3],[2,3,4],[3,4,5]])))
        r = klong(".schema(db)")
        self.assertEqual(len(r), 1)
        self.assertTrue("T" in r)
        self.assertTrue(kg_equal(r["T"], np.array(["a", "b", "c"], dtype=object)))

    def test_multi_table_join(self):
        s = """
.py("klongpy.db")
a::[1 2 3]
b::[2 3 4]
c::[3 4 5]

d::[]
d::d,,"a",,a
d::d,,"b",,b
t::.table(d)

e::[]
e::e,,"c",,c
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
        self.assertTrue(kg_equal(r, np.array([3,4,5])))
        r = klong('db("select * from T join G on G.c = T.b")')
        self.assertTrue(kg_equal(r, np.array([[2,3,3],[3,4,4]])))
        r = klong(".schema(db)")
        self.assertEqual(len(r), 2)
        self.assertTrue("T" in r)
        self.assertTrue("G" in r)
        self.assertTrue(kg_equal(r["T"], np.array(["a", "b"], dtype=object)))
        self.assertTrue(kg_equal(r["G"], np.array(["c"], dtype=object)))

    def test_table_modification_array_effects(self):
        s = """
.py("klongpy.db")
a::[1 2 3]
b::[2 3 4]
c::[3 4 5]

e::[]
e::e,,"a",,a
e::e,,"b",,b
e::e,,"c",,c
t::.table(e)

q:::{}
q,"T",,t
db::.db(q)
"""
        klong = KlongInterpreter()
        klong(s)
        r = klong('db("select * from T")')
        self.assertTrue(kg_equal(r, np.array([[1,2,3],[2,3,4],[3,4,5]])))
        klong(".insert(t; [4 5 6])")
        # original arrays are not affected by the insert
        r = klong("a")
        self.assertTrue(kg_equal(r, np.array([1,2,3])))
        r = klong("b")
        self.assertTrue(kg_equal(r, np.array([2,3,4])))
        r = klong("c")
        self.assertTrue(kg_equal(r, np.array([3,4,5])))
        r = klong('db("select a from T")')
        self.assertTrue(kg_equal(r, np.array([1,2,3,4])))

    def test_array_modification_table_effects(self):
        s = """
.py("klongpy.db")
a::[1 2 3]
b::[2 3 4]
c::[3 4 5]

e::[]
e::e,,"a",,a
e::e,,"b",,b
e::e,,"c",,c
t::.table(e)

q:::{}
q,"T",,t
db::.db(q)
"""
        klong = KlongInterpreter()
        klong(s)
        # appending to the arrays after they are used to create the table has no effect
        klong('a::a,4')
        klong('b::b,5')
        klong('c::c,6')
        r = klong('db("select a from T")')
        self.assertTrue(kg_equal(r, np.array([1,2,3])))


    def test_array_updaet_table_effects(self):
        s = """
.py("klongpy.db")
a::[1 2 3]
b::[2 3 4]
c::[3 4 5]

e::[]
e::e,,"a",,a
e::e,,"b",,b
e::e,,"c",,c
t::.table(e)

q:::{}
q,"T",,t
db::.db(q)
"""
        klong = KlongInterpreter()
        klong(s)
        # updating to the arrays after they are used to create the table has no effect
        # the table may be updated to reflect the new column
        r = klong('db("select a from T")')
        self.assertTrue(kg_equal(r, np.array([1,2,3])))
        klong('a::a:=0,1')
        r = klong("a")
        self.assertTrue(kg_equal(r, np.array([1,0,3])))
        r = klong('db("select a from T")')
        self.assertTrue(kg_equal(r, np.array([1,2,3])))
        klong('t,"a",,a')
        r = klong('db("select a from T")')
        self.assertTrue(kg_equal(r, np.array([1,0,3])))

    def test_multi_insert_no_index(self):
        s = """
.py("klongpy.db")
a::[1 2 3]
b::[2 3 4]
c::[3 4 5]

e::[]
e::e,,"a",,a
e::e,,"b",,b
e::e,,"c",,c
t::.table(e)

q:::{}
q,"T",,t
db::.db(q)
"""
        klong = KlongInterpreter()
        klong(s)
        r = klong('db("select * from T")')
        self.assertTrue(kg_equal(r, np.array([[1,2,3],[2,3,4],[3,4,5]])))
        klong(".insert(t; [4 5 6])")
        r = klong('db("select * from T")')
        self.assertTrue(kg_equal(r, np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])))

    def test_multi_insert_with_single_index(self):
        s = """
.py("klongpy.db")
a::[1 2 3]
b::[2 3 4]
c::[3 4 5]

e::[]
e::e,,"a",,a
e::e,,"b",,b
e::e,,"c",,c
t::.table(e)

.index(t;["a"])

q:::{}
q,"T",,t
db::.db(q)
"""
        klong = KlongInterpreter()
        klong(s)
        r = klong('db("select * from T")')
        self.assertTrue(kg_equal(r, np.array([[1,2,3],[2,3,4],[3,4,5]])))
        klong(".insert(t; [4 5 6])")
        r = klong('db("select * from T")')
        self.assertTrue(kg_equal(r, np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])))

        klong(".rindex(t)")
        r = klong('db("select * from T")')
        self.assertTrue(kg_equal(r, np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])))

    def test_multi_insert_with_multi_index(self):
        s = """
.py("klongpy.db")
a::[1 2 3]
b::[2 3 4]
c::[3 4 5]

e::[]
e::e,,"a",,a
e::e,,"b",,b
e::e,,"c",,c
t::.table(e)

.index(t;["a" "b"])

q:::{}
q,"T",,t
db::.db(q)
"""
        klong = KlongInterpreter()
        klong(s)
        r = klong('db("select * from T")')
        self.assertTrue(kg_equal(r, np.array([[1,2,3],[2,3,4],[3,4,5]])))
        klong(".insert(t; [4 5 6])")
        r = klong('db("select * from T")')
        self.assertTrue(kg_equal(r, np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])))
        
        klong(".rindex(t)")
        r = klong('db("select * from T")')
        self.assertTrue(kg_equal(r, np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])))
