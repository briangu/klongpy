# Table and Key-Value Stores

To support the KlongPy database capabilities, the `klongpy.db` module includes a key-value store capability that allows for saving and retrieving tables from disk. There is a more generic key-value store as well as a TableStore. The TableStore merges tables when writing to disk, while the generic key-value store writes raw serialized data and doesn't consider the contents.

Install the database extras first:

```bash
pip install "klongpy[db]"
```

Key-value stores operate as dictionaries, so setting a value updates the contents on disk and reading a value retrieves it.  Similar to Klong dictionaries, if the value does not exist, then the undefined value is returned.

### TableStore

Since KlongPy Tables are backed by Pandas DataFrames, it's convenient to be able to save/load them from disk.  For this we use the .tables() command.  If table is already present on disk, then the set results in the merge of the two DataFrames.

Let's consider that we have a table called 'prices' and we want to store it on disk.

```
?> tbs::.tables("/tmp/tables")
/tmp/tables:tables
?> tbs,"prices",prices
/tmp/tables:tables
```

Similarly, reading values is the same as getting a value from a dict:

```
?> prices::tbs?"prices"
```

### Generic key-value store

A simple key-value store backed by disk is available via the .kvs() command.

```
?> kvs::.kvs("/tmp/kvs")
/tmp/kvs:kvs
?> kvs,"hello",,"world"
/tmp/kvs:kvs
```

Now a file `/tmp/kvs/hello` exists with a pickled instance of "hello". Only unpickle data you trust.

Retrieving a value is the same as reading from a dictionary:

```
?> kvs?"hello"
world
```
