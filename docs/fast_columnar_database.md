# Columnar tables and SQL

`klongpy.db` adds tables and SQL queries to Klong. It is for programs that keep columns of data and want to select, join and aggregate them with SQL.

A table is a set of named columns of equal length, stored as a pandas DataFrame. It behaves like a dictionary from column name to column, and key/value updates are column operations. A database is a set of named tables that DuckDB queries with SQL. DuckDB reads the NumPy arrays of the columns directly, and a query returns plain Klong values.

Install the extra, which brings pandas and DuckDB, and load the module with `.py("klongpy.db")`:

```bash
pip install "klongpy[db]"
```

## Tables

<!-- check-docs: requires duckdb pandas -->
```
?> .py("klongpy.db")
1
?> t::.table(:{;"a";[1 2 3];"b";[2 3 4]})
a b
1 2
2 3
3 4
?> t,:{;"c";[3 4 5]}
a b c
1 2 3
2 3 4
3 4 5
```

`.table` takes a dictionary of column name to column values: the columns are in the dictionary's order, the names must be strings and the columns must have the same length. It also takes a list of `[name values]` pairs, as `.table([["a" [1 2 3]] ["b" [2 3 4]]])`, and a pandas DataFrame (see [below](#pandas-dataframe-integration)).

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
| `tbs,:{;"name";t}` on a [table store](table_and_key_value_stores.md) | The stored row is kept and the new row is dropped. |

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

`.db` takes a dictionary from table name to table and returns a database. A database is a function that takes an SQL string and runs it over the tables. The result is a number, a string, a list or a list of rows, depending on its [shape](#the-shape-of-a-result), and can be used in ordinary Klong code.

<!-- check-docs: requires duckdb pandas -->
```
?> .py("klongpy.db")
1
?> T::.table([["a" [1 2 3]]])
a
1
2
3
?> db::.db(:{;"T";T})
:db
?> db("select * from T")
[1 2 3]
```

Any DuckDB SQL works, for example a join:

<!-- check-docs: requires duckdb pandas -->
```
?> .py("klongpy.db")
1
?> T::.table([["a" [1 2 3]] ["b" [2 3 4]]])
a b
1 2
2 3
3 4
?> G::.table([["c" [3 4 5]]])
c
3
4
5
?> q:::{;"T";T;"G";G}
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

### The shape of a result

The shape of a result depends on how many rows and columns the query returns, so write code that tests the shape or fix it in the query. In the examples, `T` has the columns `id` (integers), `name` (strings) and `x` (reals).

| Rows by columns | Result |
|---|---|
| 1 by 1 | the bare value: a number, a string, or `:undefined` for null |
| n by 1 | a list of the n values |
| 1 by m | a flat list of the m values of the row |
| n by m | a list of n rows, each a list of m values |
| no rows | `[]` |

<!-- check-docs: requires duckdb pandas -->
```kgpy
?> .py("klongpy.db")
1
?> T::.table(:{;"id";[1 2 3];"name";["a" "b" "c"];"x";[1.5 2.5 3.5]})
id name   x
 1    a 1.5
 2    b 2.5
 3    c 3.5
?> db::.db(:{;"T";T})
:db
?> db("select id from T where id=2")
2
?> db("select name from T where id=2")
"b"
?> db("select name from T")
["a" "b" "c"]
?> db("select name, x from T where id=2")
["b" 2.5]
?> db("select id, x from T")
[[1.0 1.5] [2.0 2.5] [3.0 3.5]]
?> db("select * from T where id>9")
[]
?> db("select null")
:undefined
```

A row with columns of different types is a list of mixed values (`["b" 2.5]`). A result of several columns of numbers is one array, so the integers `id` and the reals `x` become reals together. A query that selects a column with a `where` that matches one row gives a bare value, not a list of one: ask for the row count first (`select count(*) from T where ...`) when the code cannot know.

### Missing values

A column of a table is one type. A list that contains `:undefined`, such as `[1.5 2.5 :undefined]`, is not a column of numbers, so the whole column becomes text, and the missing value is the text `undefined`:

<!-- check-docs: requires duckdb pandas -->
```kgpy
?> .py("klongpy.db")
1
?> U::.table(:{;"id";[1 2 3];"f";[1.5 2.5 :undefined]})
id         f
 1       1.5
 2       2.5
 3 undefined
?> e::.db(:{;"U";U})
:db
?> e("select typeof(f) from U limit 1")
"VARCHAR"
?> e("select f from U")
["1.5" "2.5" "undefined"]
```

Arithmetic on that column is a SQL error (`avg(VARCHAR)`). Choose a value that cannot occur, such as -1, store that, and let SQL turn it into null with `nullif`, which aggregates then skip:

<!-- check-docs: requires duckdb pandas -->
```kgpy
?> .py("klongpy.db")
1
?> V::.table(:{;"id";[1 2 3];"f";[1.5 2.5 -1]})
id    f
 1  1.5
 2  2.5
 3 -1.0
?> g::.db(:{;"V";V})
:db
?> g("select avg(nullif(f,-1)) from V")
2.0
?> g("select count(f), count(nullif(f,-1)) from V")
[3 2]
```

A null in a result (`select null`, or a missing value of a pandas DataFrame) is returned as `:undefined`.

### Errors

A SQL error is a Klong error that carries DuckDB's message, which names the statement and the position:

<!-- check-docs: requires duckdb pandas -->
```kgpy
?> .py("klongpy.db")
1
?> d0::.db(:{})
:db
?> d0("select nope from")
error: Parser Error: syntax error at or near "from"
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
db::.db(:{;"people";t})
db("select Age from people")
""")
```

## See also

- [Table and key-value stores](table_and_key_value_stores.md) to keep tables on disk.
- [IPC capabilities](ipc_capabilities.md) and [Roadmap](roadmap.md).
