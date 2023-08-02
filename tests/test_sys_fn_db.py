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

    def test_table_dict(self):
        s = """
.py("klongpy.db")
a::[1 2 3]
b::[2 3 4]
c::[3 4 5]
d::[]
d::d,,"a",,a
d::d,,"b",,b
t::.table(d)
t,"c",,c
q:::{}
q,"T",,t
db::.db(q)
db("select * from T")
"""
        klong = KlongInterpreter()
        r = run_lines(s, klong)
        self.assertTrue(kg_equal(r, np.array([[1,2,3],[2,3,4],[3,4,5]])))
        r = klong(".schema(db)")
        self.assertEqual(len(r), 1)
        self.assertTrue("T" in r)
        self.assertTrue(kg_equal(r["T"], np.array(["a", "b", "c"], dtype=object)))


    def test_multi_table_db(self):
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

    def test_insert_no_index(self):
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

    def test_insert_with_single_index(self):
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
        klong('.index(t;["a"])')
        r = klong('db("select * from T")')
        self.assertTrue(kg_equal(r, np.array([[1,2,3],[2,3,4],[3,4,5]])))
        klong(".insert(t; [4 5 6])")
        r = klong('db("select * from T")')
        self.assertTrue(kg_equal(r, np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])))

        klong(".rindex(t)")
        r = klong('db("select * from T")')
        self.assertTrue(kg_equal(r, np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])))

    def test_insert_with_multi_index(self):
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
        klong('.index(t;["a" "b"])')
        r = klong('db("select * from T")')
        self.assertTrue(kg_equal(r, np.array([[1,2,3],[2,3,4],[3,4,5]])))
        klong(".insert(t; [4 5 6])")
        r = klong('db("select * from T")')
        self.assertTrue(kg_equal(r, np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])))
        
        klong(".rindex(t)")
        r = klong('db("select * from T")')
        self.assertTrue(kg_equal(r, np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])))


def perf_single_col_no_pre_alloc():
    s = """
.py("klongpy.db")
a::[]

e::[]
e::e,,"a",,a
t::.table(e)

q:::{}
q,"T",,t
db::.db(q)
"""
    klong = KlongInterpreter()
    klong(s)
    klong("tfn::{[t0];t0::.pc();x@[];.pc()-t0}")
    klong("afn::{{.insert(t; x)}'!10000}")
    r = klong("tfn(afn)")
    pr = r / 10000
    print("single col (no index) (no pre alloc)", r, pr)
    r = klong('db("select count(*) from T")')
    assert r == 10000


def perf_single_index_no_pre_alloc():
    s = """
.py("klongpy.db")
a::[]

e::[]
e::e,,"a",,a
t::.table(e)

.index(t;["a"])

q:::{}
q,"T",,t
db::.db(q)
"""
    klong = KlongInterpreter()
    klong(s)
    klong("tfn::{[t0];t0::.pc();x@[];.pc()-t0}")
    klong("afn::{{.insert(t; x)}'!10000}")
    r = klong("tfn(afn)")
    pr = r / 10000
    print("single col (index) (no pre alloc)", r, pr)
    r = klong('db("select count(*) from T")')
    assert r == 10000


def perf_single_col_index_pre_alloc():
    s = """
.py("klongpy.db")
a::!10000

e::[]
e::e,,"a",,a
t::.table(e)

.index(t;["a"])

q:::{}
q,"T",,t
db::.db(q)
"""
    klong = KlongInterpreter()
    klong(s)
    klong("tfn::{[t0];t0::.pc();x@[];.pc()-t0}")
    klong("afn::{{.insert(t; x)}'!10000}")
    r = klong("tfn(afn)")
    pr = r / 10000
    print("single col (index) (pre alloc)", r, pr)
    r = klong('db("select count(*) from T")')
    assert r == 10000


def perf_multi_col_pre_alloc():
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
    klong("tfn::{[t0];t0::.pc();x@[];.pc()-t0}")
    klong("afn::{{.insert(t; x,(x+1),(x+2))}'!10000}")
    r = klong("tfn(afn)")
    pr = r / 10000
    print("multi col (no index) (pre alloc)", r, pr)
    r = klong('db("select count(*) from T")')
    assert r == 10003



def perf_multi_col_single_index_pre_alloc():
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
    klong("tfn::{[t0];t0::.pc();x@[];.pc()-t0}")
    klong("afn::{{.insert(t; x,(x+1),(x+2))}'!10000}")
    r = klong("tfn(afn)")
    pr = r / 10000
    print("multi col (single index) (pre alloc)", r, pr)
    r = klong('db("select count(*) from T")')
    assert r == 10000


if __name__ == "__main__":
    perf_single_col_no_pre_alloc()
    perf_single_index_no_pre_alloc()
    perf_single_col_index_pre_alloc()
    perf_multi_col_pre_alloc()
    perf_multi_col_single_index_pre_alloc()
