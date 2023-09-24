import sys

import duckdb
import numpy as np
import pandas as pd

from klongpy.core import (KGCall, KGLambda, KlongException, reserved_fn_args,
                          reserved_fn_symbol_map)


class KlongDbException(KlongException):
    def __init__(self, x, e):
        self.x = x
        super().__init__(e)


class Table(dict):
    def __init__(self, data, columns=None):
        # If the input is a pandas DataFrame, use it directly
        if isinstance(data, pd.DataFrame):
            self._df = data.copy()
            self.columns = data.columns.tolist()
        # If the input is a dictionary, create a DataFrame using the provided columns
        elif isinstance(data, dict) and columns is not None:
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
        return np.inf if v is None else v.values

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

    def __init__(self, tables):
        self.tables = tables
        self.con = duckdb.connect(':memory:')

    def __call__(self, _, ctx):
        x = ctx[reserved_fn_symbol_map[reserved_fn_args[0]]]

        # add the table column -> dataframe into local scope so DuckDB can reference them by name in the SQL.
        for k,v in self.tables.items():
            locals()[k] = v.get_dataframe()

        try:
            df = self.con.execute(x).fetchdf()
            return df.values.squeeze()
        except Exception as e:
            # TODO: we need to expose the sql errors in an actionable way
            raise KlongDbException(x, e)

    def schema(self):
        return {k: v.schema() for k,v in self.tables.items()}

    def get_arity(self):
        return 1

    def __str__(self):
        return ":db"


def eval_sys_fn_create_table(x):
    """

        .table(x)                                         [Table-Create]

        Creates a table created from a sequence of tuples where each tuple
        describes a column and it's associated values.  Column order is
        determined by the order in the tuple sequence.

        The resulting table treatable as a dict and key/value updates are
        column operations.

        Examples:

            a::[1 2 3]
            b::[2 3 4]
            c::[3 4 5]
            d::[]
            d::d,,"a",,a
            d::d,,"b",,b
            t::.table(d)
            t,"c",,c

    """
    if np.isarray(x):
        return Table({k:v for k,v in x}, columns=[k for k,_ in x])
    elif isinstance(x, pd.DataFrame):
        return Table(x)
    raise KlongDbException(x, "tables must be created from an array of column data map or a Pandas DataFrame.")


def eval_sys_fn_index(x, y):
    """

        .index(x)                                   [Create-Table-Index]

        Creates an index on a table as specified by the columns in array "x".
        An index may be one or more columns.

        t::.table(d)
        .index(t, ["a"])

    """
    if not isinstance(x, Table):
        raise KlongDbException(x, "An index may only be created on a table.")
    if x.has_index():
        raise KlongDbException(x, "Table already has an index.")
    if not np.isarray(y):
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

        .insert(x, y)                                     [Table-Insert]

        Insert values "y" into a table "x".  The values provided by "y" must be in the
        corrensponding column position as specified when the table was created.
        If the table is indexed, then the appropriate columns will be used as keys when
        inserting values.  If the table is unindexed, then the values are appended.

        Examples:

            t::.table(d)
            .insert(t, [1;2;3])

    """
    if not isinstance(x,Table):
        raise KlongDbException(x, "Inserts must be applied to a table")
    if len(x.columns) > 1:
        if not np.isarray(y):
            raise KlongDbException(x, "Values to insert must be a list")
    elif not np.isarray(y):
            y = np.array([y])
    if len(y) != len(x.columns):
        raise KlongDbException(x, f"Expected {len(x.columns)} values, received {len(y)}")
    x.insert(y)
    return x


def eval_sys_fn_create_db(x):
    """

        .db(x)                                               [Create-db]

        Create a database from a map of tables "x".
        The keys are table names and values are the tables.



    """
    if not isinstance(x,dict):
        return "a db must be created from a dict"
    return Database(x)


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
