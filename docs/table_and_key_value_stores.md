# Table and Key-Value Stores

> **Security:** keys are used as file names under the store root and values are pickled on disk. Treat the root as trusted storage. See the [security model](https://github.com/briangu/klongpy#security-model-trusted-networks).

`klongpy.db` can keep [tables](fast_columnar_database.md) and other values on disk. A store behaves like a dictionary whose entries are files: setting a key writes the value, and reading a key returns it, or `:undefined` when it is not stored. There are two kinds. A table store (`.tables`) writes tables and merges a table with the one already stored under the same key. A key-value store (`.kvs`) writes any value as pickled bytes and does not look at its contents.

Install the extra, which brings pandas and DuckDB, and load the module with `.py("klongpy.db")`:

```bash
pip install "klongpy[db]"
```

## TableStore

`.tables(path)` opens a table store under a directory. This stores a table called `prices`:

<!-- check-docs: requires duckdb pandas -->
```
?> .py("klongpy.db")
1
?> prices::.table([["sym" ["A" "B"]] ["px" [1.5 2.5]]])
sym  px
  A 1.5
  B 2.5
?> tbs::.tables("/tmp/tables")
/tmp/tables:tables
?> tbs,"prices",,prices
/tmp/tables:tables
```

Reading a key returns the table:

<!-- check-docs: continue -->
```
?> prices::tbs?"prices"
sym  px
  A 1.5
  B 2.5
```

Writing a table under a key that is already stored merges the two by index: a row with a new index value is added, and for a row whose index value is already stored the stored row is kept. (`.insert` into an indexed table behaves the other way: the new row replaces the old one; see [Duplicate index values](fast_columnar_database.md#duplicate-index-values).) Tables that have no index set (as `prices` above) are numbered from 0, so a second write of such a table finds every index value taken and adds nothing. To merge by a column, index it first:

<!-- check-docs: requires duckdb pandas -->
```
?> .py("klongpy.db")
1
?> tbs::.tables("/tmp/merge")
/tmp/merge:tables
?> a::.table([["sym" ["A" "B"]] ["px" [1.5 2.5]]]);.index(a;["sym"])
["sym"]
?> tbs,"p",,a
/tmp/merge:tables
?> b::.table([["sym" ["B" "C"]] ["px" [9.0 3.5]]]);.index(b;["sym"])
["sym"]
?> tbs,"p",,b
/tmp/merge:tables
?> tbs?"p"
sym  px
  A 1.5
  B 2.5
  C 3.5
```

## Generic key-value store

`.kvs(path)` opens a key-value store under a directory:

<!-- check-docs: requires duckdb pandas -->
```
?> .py("klongpy.db")
1
?> kvs::.kvs("/tmp/kvs")
/tmp/kvs:kvs
?> kvs,"hello",,"world"
/tmp/kvs:kvs
```

The file `/tmp/kvs/hello` now holds the pickled value `"world"`. Only read stores that you trust.

Read it back:

<!-- check-docs: continue -->
```
?> kvs?"hello"
"world"
```

## See also

- [Columnar tables and SQL](fast_columnar_database.md) for tables and SQL.
- [IPC capabilities](ipc_capabilities.md) and [Roadmap](roadmap.md).
