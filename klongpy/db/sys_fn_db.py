import sys

import duckdb
import numpy as np
import pandas as pd

from klongpy.core import (KGCall, KGLambda, KGSym, KlongException, KLONG_UNDEFINED,
                          reserved_fn_args, reserved_fn_symbol_map)
from klongpy.backend import np as backend_np
from klongpy.errors import KlongShapeError, KlongTypeError


class KlongDbException(KlongException):
    def __init__(self, x, e):
        self.x = x
        super().__init__(str(e))


def _host_column(v):
    """A column as something pandas can hold: a tensor becomes a NumPy array, anything else is unchanged."""
    if hasattr(v, "detach") and hasattr(v, "cpu"):
        return v.detach().cpu().numpy()
    return v


class Table(dict):
    def __init__(self, data, columns=None):
        # If the input is a pandas DataFrame, use it directly
        if isinstance(data, pd.DataFrame):
            self._df = data.copy()
            self.columns = data.columns.tolist()
        # If the input is a dictionary, create a DataFrame using the provided columns
        elif isinstance(data, dict) and columns is not None:
            # a column held on a device (a tensor on a GPU) is copied to host memory: a DataFrame holds NumPy arrays
            data = {k: _host_column(v) for k, v in data.items()}
            self._df = pd.DataFrame(data, columns=columns, copy=False)
            self.columns = columns
        else:
            raise ValueError("Input must be either a dictionary with columns or a pandas DataFrame.")

        self.idx_cols = None
        self.buffer = []

    def __getitem__(self, x):
        return self.get(x)

    def __setitem__(self, x, y):
        return self.set(x, y)

    def __contains__(self, x):
        raise NotImplementedError()

    def get(self, x):
        v = self._df.get(x)
        return KLONG_UNDEFINED if v is None else v.values

    def set(self, x, y):
        self._df[x] = y
        self.columns = list(self._df.columns)

    def schema(self):
        return np.array(self._df.columns, dtype=object)

    @staticmethod
    def _create_index_from_cols(df: pd.DataFrame, idx_cols):
        iic = []
        for ic in idx_cols:
            k = f"{ic}_idx"
            df[k] = df[ic]
            iic.append(k)
        df.set_index(iic, inplace=True)
        df.sort_index(inplace=True)
        df.drop_duplicates(inplace=True)
        return df

    def set_index(self, idx_cols):
        df = self.get_dataframe()
        self._df = self._create_index_from_cols(df, idx_cols)
        self.idx_cols = idx_cols

    def reset_index(self):
        df = self.get_dataframe()
        self._df = df.reset_index()
        if self.idx_cols is not None:
            iic = [f"{ic}_idx" for ic in self.idx_cols]
            self._df.drop(columns=iic, inplace=True)
            self.idx_cols = None

    def has_index(self):
        # TODO: add tests for loading pandas dataframes with indexes but no idx_cols
        # return (not isinstance(self._df.index, pd.RangeIndex)) or self.idx_cols is not None
        return self.idx_cols is not None

    def get_dataframe(self):
        self.commit()
        return self._df

    def insert(self, y):
        self.buffer.append(y)

    def insertb(self, y):
        self.buffer.extend(y)

    def commit(self):
        if not self.buffer:
            return
        if self.has_index():
            buffer_df = pd.DataFrame(self.buffer, columns=self.columns)
            buffer_df = self._create_index_from_cols(buffer_df, self.idx_cols)

            # Update existing rows and append new rows
            common_idx = self._df.index.intersection(buffer_df.index)
            self._df.loc[common_idx] = buffer_df.loc[common_idx]
            self._df = pd.concat([self._df, buffer_df.loc[~buffer_df.index.isin(common_idx)]])
            self._df.sort_index(inplace=True)
        else:
            values = np.concatenate([self._df.values] + [y.reshape(1, -1) for y in self.buffer])
            self._df = pd.DataFrame(values, columns=self.columns, copy=False)
        self.buffer = []

    def __len__(self):
        return len(self.get_dataframe())

    def __str__(self):
        full_df = self.get_dataframe()
        df = full_df.head(10)
        idx_cols = self.idx_cols or []
        header = [f"{k}{'*' if k in idx_cols else ''}" for k in df.columns]
        if df.empty:
            return str(" ".join(header))
        col_space = [len(x) for x in header]
        s = df.to_string(header=header, index=False, col_space=col_space)
        if len(full_df) > len(df):
            s += "\n...\n"
            s += f"rows={len(full_df)}\n"
        return s


class Database(KGLambda):
    # __init__ does not call KGLambda.__init__: the flag the interpreter reads
    user = False

    def __init__(self, tables):
        self.tables = tables
        self.con = duckdb.connect(':memory:')

    def __call__(self, _, ctx):
        x = ctx[reserved_fn_symbol_map[reserved_fn_args[0]]]

        # register the table dataframes so DuckDB can reference them by name in the SQL.
        for k,v in self.tables.items():
            self.con.register(k, v.get_dataframe())

        try:
            df = self.con.execute(x).fetchdf()
        except Exception as e:
            # TODO: we need to expose the sql errors in an actionable way
            raise KlongDbException(x, e)
        # one row and one column squeezes to a 0-d array: a number stays one, any other cell
        # (text, null) becomes the Python value, since a 0-d object array has no length
        v = df.values.squeeze()
        if isinstance(v, np.ndarray) and v.ndim == 0 and v.dtype == object:
            v = v.item()
            return KLONG_UNDEFINED if v is None or v is pd.NA or v is pd.NaT or (isinstance(v, float) and v != v) else v
        return v

    def schema(self):
        return {k: v.schema() for k,v in self.tables.items()}

    def get_arity(self):
        return 1

    def __str__(self):
        return ":db"


def _table_from_dict(x):
    """A table from a dictionary of column name to column values (insertion order is column order)."""
    cols = {}
    for k, v in x.items():
        if not isinstance(k, str) or isinstance(k, KGSym):
            raise KlongTypeError(f".table: column names must be strings, got {type(k).__name__ if not isinstance(k, KGSym) else 'a symbol'} {k!r}",
                                 hint='use a string key, as in :{;"a";[1 2 3]}')
        if isinstance(v, (str, bytes, dict)) or not hasattr(v, '__len__') or getattr(v, 'ndim', 1) == 0:
            raise KlongTypeError(f'.table: column "{k}" must be a list of values',
                                 hint="a column is a list or array, one value per row")
        cols[k] = v
    sizes = {k: len(v) for k, v in cols.items()}
    if len(set(sizes.values())) > 1:
        detail = ", ".join(f'"{k}" has {n}' for k, n in sizes.items())
        raise KlongShapeError(f".table: columns have different lengths ({detail})",
                              hint="every column needs the same number of rows")
    return Table(cols, columns=list(cols))


def eval_sys_fn_create_table(x):
    """

        .table(x)                                         [Table-Create]

        Create a table from a dictionary of column name to column values,
        or from a sequence of tuples, each of which describes a column and
        its values.  Column order is the dictionary's insertion order, or
        the order in the tuple sequence.  Column names must be strings
        and every column must have the same number of values.

        The table can be used like a dictionary, and key/value updates are
        column operations.

        Examples:

            t::.table(:{;"a";[1 2 3];"b";[2 3 4]})
            t,"c",,[3 4 5]

            d::[]
            d::d,,"a",,[1 2 3]
            d::d,,"b",,[2 3 4]
            t::.table(d)

    """
    if isinstance(x, dict):
        return _table_from_dict(x)
    if backend_np.isarray(x):
        return Table({k:v for k,v in x}, columns=[k for k,_ in x])
    elif isinstance(x, pd.DataFrame):
        return Table(x)
    raise KlongDbException(x, "tables must be created from a dictionary of columns, an array of column data map or a Pandas DataFrame.")


def eval_sys_fn_index(x, y):
    """

        .index(x;y)                                 [Create-Table-Index]

        Creates an index on the table "x" as specified by the columns in array "y".
        An index may be one or more columns.

        t::.table(d)
        .index(t;["a"])

    """
    if not isinstance(x, Table):
        raise KlongDbException(x, "An index may only be created on a table.")
    if x.has_index():
        raise KlongDbException(x, "Table already has an index.")
    if not backend_np.isarray(y):
        raise KlongDbException(x, "An index must be a list of column names")
    for q in y:
        if q not in x.columns:
            raise KlongDbException(x, f"An index column {q} not found in table")
    x.set_index(list(y))
    return x.idx_cols or []


def eval_sys_fn_reset_index(x):
    """

        .rindex(x)                                  [Reset-Table-Index]

        Removes indexes from a table.

    """
    if not isinstance(x, Table):
        raise KlongDbException(x, "An index may only be created on a table.")
    if x.has_index():
        x.reset_index()
        return 1
    return 0


def eval_sys_fn_schema(x):
    """

        .schema(x)                                        [Table-Schema]

        Returns the schema of either a table or dictionary "x".  If "x" is a table,
        then the columns are returned.  If "x" is a database, then a dict of
        table name to table is returned.

    """
    if isinstance(x, KGCall):
        x = x.a
    if not isinstance(x, (Database, Table)):
        raise KlongDbException(x, "A schema is available only for a table or database.")
    return x.schema()


def eval_sys_fn_insert_table(x, y):
    """

        .insert(x;y)                                      [Table-Insert]

        Insert values "y" into a table "x".  The values provided by "y" must be in the
        corresponding column position as specified when the table was created.
        If the table is indexed, then the appropriate columns will be used as keys when
        inserting values.  If the table is unindexed, then the values are appended.

        Examples:

            t::.table(d)
            .insert(t;[1 2 3])

            for batch inserts supply an array of arrays:

            .insert(t;[[1 2 3] [4 5 6]])

    """
    if not isinstance(x,Table):
        raise KlongDbException(x, "Inserts must be applied to a table")
    if not backend_np.isarray(y):
        raise KlongDbException(x, "Values to insert must be a list")
    batch = len(y.shape) > 1
    y_cols = len(y[0]) if batch else len(y)
    if y_cols != len(x.columns):
        raise KlongDbException(x, f"Expected {len(x.columns)} values, received {len(y)}")
    if batch:
        x.insertb(y)
    else:
        x.insert(y)
    return x


def eval_sys_fn_create_db(klong, x):
    """

        .db(x)                                               [Create-db]

        Create a database from a map of tables "x".
        The keys are table names and values are the tables.



    """
    if not isinstance(x,dict):
        return "a db must be created from a dict"
    # a dictionary literal, :{["T" T]}, holds the symbol T, not its value
    return Database({k: klong[v] if isinstance(v, KGSym) else v for k, v in x.items()})


def create_system_functions_db():
    def _get_name(s):
        i = s.index(".")
        return s[i : i + s[i:].index("(")]

    registry = {}

    m = sys.modules[__name__]
    for x in filter(lambda n: n.startswith("eval_sys_"), dir(m)):
        fn = getattr(m, x)
        registry[_get_name(fn.__doc__)] = fn

    return registry


def _table_from_wire(names, columns):
    """A table received over IPC (kdb+ type 98) as a Table."""
    names = [str(n) for n in names]
    data = {}
    for n, c in zip(names, columns):
        data[n] = np.array(list(c), dtype=object) if isinstance(c, str) else c
    return Table(data, columns=names)


try:
    from klongpy.ipc.codec import set_table_factory
    set_table_factory(_table_from_wire)
except ImportError:  # the IPC package is optional
    pass
