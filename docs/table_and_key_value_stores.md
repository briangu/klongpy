# Table and Key-Value Stores

> **Security:** keys are used as file names under the store root and values are pickled on disk. Treat the root as trusted storage. See the [security model](https://github.com/briangu/klongpy#security-model-trusted-networks).

To support the KlongPy database capabilities, the `klongpy.db` module includes a key-value store capability that allows for saving and retrieving tables from disk. There is a more generic key-value store as well as a TableStore. The TableStore merges tables when writing to disk, while the generic key-value store writes raw serialized data and doesn't consider the contents.

Install the database extras first:

```bash
pip install "klongpy[db]"
```

Key-value stores operate as dictionaries, so setting a value updates the contents on disk and reading a value retrieves it.  Similar to Klong dictionaries, if the value does not exist, then the undefined value is returned.

### TableStore

Since KlongPy Tables are backed by Pandas DataFrames, it's convenient to be able to save/load them from disk.  For this we use the .tables() command.  If table is already present on disk, then the set results in the merge of the two DataFrames.

Let's consider that we have a table called 'prices' and we want to store it on disk.

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

Similarly, reading values is the same as getting a value from a dict:

<!-- check-docs: continue -->
```
?> prices::tbs?"prices"
sym  px
  A 1.5
  B 2.5
```

When a table is written under a name that is already on disk, the two are combined by their index: rows with a new index value are added, and a row whose index value is already stored keeps the stored row. Tables that have no index set (as `prices` above) are numbered from 0, so a second write of such a table finds every index value taken and adds nothing. To merge by a column, index it first:

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

Reading a name that is not stored returns the undefined value (`:undefined`), for tables and for the generic store below.

### Generic key-value store

A simple key-value store backed by disk is available via the .kvs() command.

<!-- check-docs: requires duckdb pandas -->
```
?> .py("klongpy.db")
1
?> kvs::.kvs("/tmp/kvs")
/tmp/kvs:kvs
?> kvs,"hello",,"world"
/tmp/kvs:kvs
```

Now a file `/tmp/kvs/hello` exists with a pickled instance of "hello". Only unpickle data you trust.

Retrieving a value is the same as reading from a dictionary:

<!-- check-docs: continue -->
```
?> kvs?"hello"
"world"
```

## See also

- [Fast columnar database](fast_columnar_database.md) for tables and SQL.
- [IPC capabilities](ipc_capabilities.md) and [Roadmap](roadmap.md).
