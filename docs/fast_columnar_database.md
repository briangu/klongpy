# Columnar tables and SQL

`klongpy.db` adds tables and SQL queries to Klong. A table is a set of named columns of equal length, stored as a pandas DataFrame; it behaves like a dictionary from column name to column, and key/value updates are column operations. A database is a set of named tables that DuckDB queries with SQL; DuckDB reads the NumPy arrays of the columns directly, and a query returns plain Klong values.

Install the extra, which brings pandas and DuckDB, and load the module with `.py("klongpy.db")`:

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

A table can also be made from a pandas DataFrame (see [below](#pandas-dataframe-integration)).

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

### Duplicate index values

The two ways of writing into an indexed table treat a duplicate index value differently:

| Write | Row whose index value already exists |
|---|---|
| `.insert(t;row)` | The new row replaces the existing row (an update). |
| `tbs,"name",,t` on a [table store](table_and_key_value_stores.md) | The stored row is kept and the new row is dropped. |

A table with no index appends every row. An example of `.insert` on an indexed table:

<!-- check-docs: continue -->
```
?> .insert(t;[3 5 6])
a* b c
 1 2 3
 2 3 4
 3 5 6
```

`.rindex` removes the indexes. It returns 1 if an index was removed and 0 if there was none.

<!-- check-docs: continue -->
```
?> .rindex(t)
1
?> .rindex(t)
0
```

## Database

`.db` takes a dictionary from table name to table and returns a database. A database is a function that takes an SQL string and runs it over the tables. The result is a NumPy array (one column), a list of rows (several columns) or a number, and can be used in ordinary Klong code.

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

Any DuckDB SQL works, for example a join:

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

The join:

<!-- check-docs: continue -->
```
?> db("select * from T join G on G.c = T.b")
[[2 3 3] [3 4 4]]
```

## Pandas DataFrame integration

A pandas DataFrame stored in the interpreter becomes a table with `.table`. This returns `[25 30 35 40]`:

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
