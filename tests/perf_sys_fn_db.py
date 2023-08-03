import time
import unittest

from utils import *

from klongpy import KlongInterpreter


def run_lines(s, klong=None):
    klong = klong or KlongInterpreter()
    for line in s.splitlines():
        r = klong(line)
    return r


def time_sql(klong, sql):
    start_t = time.time()
    try:
        return klong(sql)
    finally:
        print(sql, " : ", time.time() - start_t)


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
    print("single col (no index) (no pre alloc)", r, pr, int(1/pr))
    r = time_sql(klong, 'db("select count(*) from T")')
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
    print("single col (index) (no pre alloc)", r, pr, int(1/pr))
    r = time_sql(klong, 'db("select count(*) from T")')
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
    print("single col (index) (pre alloc)", r, pr, int(1/pr))
    r = time_sql(klong, 'db("select count(*) from T")')
    assert r == 10000


def perf_multi_col():
    s = """
.py("klongpy.db")
a::[]
b::[]
c::[]

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
    klong("afn::{{.insert(t; x,x,x)}'!10000}")
    r = klong("tfn(afn)")
    pr = r / 10000
    print("multi col (no index) (no pre alloc)", r, pr, int(1/pr))
    r = time_sql(klong, 'db("select count(*) from T")')
    assert r == 10000


def perf_multi_col_single_index():
    s = """
.py("klongpy.db")
a::[]
b::[]
c::[]

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
    klong("afn::{{.insert(t; x,x,x)}'!10000}")
    klong("bfn::{{.insert(t; x,x,x)}'10000+!10}")
    r = klong("tfn(afn)")
    pr = r / 10000
    print("multi col (single index) (no pre alloc)", r, pr, int(1/pr))
    r = time_sql(klong, 'db("select count(*) from T")')
    assert r == 10000
    r = klong("tfn(bfn)")
    pr = r / 10
    print("multi col (single index) (no pre alloc) [10]", r, pr, int(1/pr))
    r = time_sql(klong, 'db("select count(*) from T")')
    assert r == 10010



def perf_multi_col_single_index_pre_alloc():
    s = """
.py("klongpy.db")
a::!10000
b::!10000
c::!10000

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
    klong("afn::{{.insert(t; x,x,x)}'!10000}")
    r = klong("tfn(afn)")
    pr = r / 10000
    print("multi col (single index) (pre alloc)", r, pr, int(1/pr))
    r = time_sql(klong, 'db("select count(*) from T")')
    assert r == 10000


def perf_multi_col_multi_index():
    s = """
.py("klongpy.db")
a::[]
b::[]
c::[]

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
    klong("tfn::{[t0];t0::.pc();x@[];.pc()-t0}")
    klong("afn::{{.insert(t; x,x,x)}'!10000}")
    r = klong("tfn(afn)")
    pr = r / 10000
    print("multi col (multi index) (no pre alloc)", r, pr, int(1/pr))
    r = time_sql(klong, 'db("select count(*) from T")')
    assert r == 10000


def perf_multi_col_multi_index_pre_alloc():
    s = """
.py("klongpy.db")
a::!10000
b::!10000
c::!10000

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
    klong("tfn::{[t0];t0::.pc();x@[];.pc()-t0}")
    klong("afn::{{.insert(t; x,x,x)}'!10000}")
    r = klong("tfn(afn)")
    pr = r / 10000
    print("multi col (multi index) (pre alloc)", r, pr, int(1/pr))
    r = time_sql(klong, 'db("select count(*) from T")')
    assert r == 10000


if __name__ == "__main__":
    perf_single_col_no_pre_alloc()
    perf_single_index_no_pre_alloc()
    perf_single_col_index_pre_alloc()
    perf_multi_col()
    perf_multi_col_single_index()
    perf_multi_col_single_index_pre_alloc()
    perf_multi_col_multi_index()
    perf_multi_col_multi_index_pre_alloc()
