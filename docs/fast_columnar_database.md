# Fast Columnar Database

KlongPy provides a module `klongpy.db` that includes DuckDB integration. DuckDB can operate directly on NumPy arrays, which allows for zero-copy SQL execution over pre-existing NumPy data.

Install the database extras first:

```bash
pip install "klongpy[db]"
```

## Tables

```
?> .py("klongpy.db")
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

Indexes (one or more columns) can be created on a table. The current indexes can be seen in the table description prefix.

```
?> .index(t; ["a"])
['a']
```

When a column is indexed, it appears with an asterisk in the pretty-print format:

```
?> t
a* b
 1 2
 2 3
 3 5
```

Inserting a row with a pre-existing value at an index results in an update:

```
?> .insert(t, [3 5 6])
a* b c
 1 2 3
 2 3 4
 3 5 6
```

Indexes may be reset via .rindex().  True is returned if the index was reset.

```
?> .rindex(t)
1
```

## Database

Databases are created from a map of table names to table instances.  A database instance is a function which accepts SQL and runs it over the underlying tables.  SQL results are NumPy arrays and can be directly used in normal KlongPy operations.

```
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

```
d::[]
d::d,,"a",,[1 2 3]
d::d,,"b",,[2 3 4]
T::.table(d)

e::,"c",,[3 4 5]
G::.table(e)

q:::{}
q,"T",,T
q,"G",,G
db::.db(q)
```

We can now issue a JOIN SQL:

```
?> db("select * from T join G on G.c = T.b")
[[2 3 3]
 [3 4 4]]
```

## Pandas DataFrame integration

Tables are backed by Pandas DataFrames, so it's easy to integrate Pandas directly into KlongPy via DuckDB.

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
