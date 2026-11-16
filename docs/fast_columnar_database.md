# Fast Columnar Database

KlongPy provides a module `klongpy.db` that includes DuckDB integration. DuckDB can operate directly on NumPy arrays, which allows for zero-copy SQL execution over pre-existing NumPy data.

Install the database extras first:

```bash
pip install "klongpy[db]"
```

## Tables

<!-- check-docs: requires duckdb pandas -->
```
?> .py("klongpy.db")
1
?> t::.table([["a" [1 2 3]] ["b" [2 3 4]]])
a b
1 2
2 3
3 4
?> t,"c",,[3 4 5]
a b c
1 2 3
2 3 4
3 4 5
```

You can also create a table directly from a Pandas DataFrame.

Indexes (one or more columns) can be created on a table. `.index` returns the indexed columns.

<!-- check-docs: continue -->
```
?> .index(t;["a"])
["a"]
```

When a column is indexed, it appears with an asterisk in the pretty-print format:

<!-- check-docs: continue -->
```
?> t
a* b c
 1 2 3
 2 3 4
 3 4 5
```

Inserting a row with a pre-existing value at an index results in an update:

<!-- check-docs: continue -->
```
?> .insert(t;[3 5 6])
a* b c
 1 2 3
 2 3 4
 3 5 6
```

Indexes may be reset via .rindex(). 1 is returned if an index was reset and 0 if there was none.

<!-- check-docs: continue -->
```
?> .rindex(t)
1
?> .rindex(t)
0
```

## Database

Databases are created from a map of table names to table instances.  A database instance is a function which accepts SQL and runs it over the underlying tables.  SQL results are NumPy arrays and can be directly used in normal KlongPy operations.

<!-- check-docs: requires duckdb pandas -->
```
?> .py("klongpy.db")
1
?> T::.table(,"a",,[1 2 3])
a
1
2
3
?> db::.db(:{},"T",,T)
:db
?> db("select * from T")
[1 2 3]
```

Since KlongPy uses DuckDB under the hood, you can perform sophisticated SQL over the underlying NumPy arrays.

For example, it's easy to use JOIN with this setup:

<!-- check-docs: requires duckdb pandas -->
```
?> .py("klongpy.db")
1
?> d::[]
[]
?> d::d,,"a",,[1 2 3]
[["a" [1 2 3]]]
?> d::d,,"b",,[2 3 4]
[["a" [1 2 3]] ["b" [2 3 4]]]
?> T::.table(d)
a b
1 2
2 3
3 4
?> e::,"c",,[3 4 5]
[["c" [3 4 5]]]
?> G::.table(e)
c
3
4
5
?> q:::{}
:{}
?> q,"T",,T
:{["T" a b
1 2
2 3
3 4]}
?> q,"G",,G
:{["T" a b
1 2
2 3
3 4] ["G" c
3
4
5]}
?> db::.db(q)
:db
```

We can now issue a JOIN SQL:

<!-- check-docs: continue -->
```
?> db("select * from T join G on G.c = T.b")
[[2 3 3] [3 4 4]]
```

## Pandas DataFrame integration

Tables are backed by Pandas DataFrames, so it's easy to integrate Pandas directly into KlongPy via DuckDB.

Run on 2026-11-16 with pandas and duckdb installed, this prints `[25 30 35 40]`:

```Python
from klongpy import KlongInterpreter
import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40]}
df = pd.DataFrame(data)

klong = KlongInterpreter()
klong['df'] = df
r = klong("""
.py("klongpy.db")
t::.table(df)
db::.db(:{},"people",t)
db("select Age from people")
""")
```

## See also

- [Table and key-value stores](table_and_key_value_stores.md) to keep tables on disk.
- [IPC capabilities](ipc_capabilities.md) and [Roadmap](roadmap.md).
