import asyncio
import sys
import threading

import duckdb
import numpy as np
import pandas as pd

from klongpy.core import KlongException, KGCall, KGLambda, reserved_fn_args, reserved_fn_symbol_map

import time

# _main_loop = asyncio.get_event_loop()
# _main_tid = threading.current_thread().ident


class KlongDbException(KlongException):
    def __init__(self, x, e):
        self.x = x
        super().__init__(e)


class Table(dict):
    def __init__(self, d, columns):
        self._df = pd.DataFrame(d, columns=columns, copy=False)
        self.idx_cols = None
        self.idx_cols_loc = None
        self.columns = columns
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

    def schema(self):
        return np.array(self._df.columns, dtype=object)

    @staticmethod
    def _create_index_from_cols(df, idx_cols):
        iic = []
        for ic in idx_cols:
            k = f"{ic}_idx"
            df[k] = df[ic]
            iic.append(k)
        df.set_index(iic, inplace=True)
        return df

    def set_index(self, idx_cols):
        self._df = self._create_index_from_cols(self._df, idx_cols)
        self.idx_cols = idx_cols
        self.idx_cols_loc = np.where(np.isin(np.asarray(self.columns), np.asarray(self.idx_cols)))[0]

    def reset_index(self):
        self._df = self._df.reset_index()
        iic = [f"{ic}_idx" for ic in self.idx_cols]
        self._df.drop(columns=iic, inplace=True)
        self.idx_cols = None
        self.idx_cols_loc = None

    def has_index(self):
        return self.idx_cols is not None

    def get_dataframe(self):  # Method to access the DataFrame
        start_t = time.time()
        self.commit()  # Commit changes before returning the DataFrame
        print(time.time() - start_t)
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
        return len(self._df.columns)
    
    def __str__(self):
        return f"{self.idx_cols or ''}:{self.columns}:table"


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

    """
    if not np.isarray(x):
        raise KlongDbException(x, "tables must be created from an array of column data map")
    return Table({k:v for k,v in x}, columns=[k for k,_ in x])


def eval_sys_fn_index(x, y):
    """

        .index(x)                                   [Create-Table-Index]

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
    return x.set_index(list(y))


def eval_sys_fn_reset_index(x):
    """

        .rindex(x)                                  [Reset-Table-Index]

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

    """
    if isinstance(x, KGCall):
        x = x.a
    if not isinstance(x, (Database, Table)):
        raise KlongDbException(x, "A schema is available only for a table or database.")
    return x.schema()


def eval_sys_fn_insert_table(x, y):
    """

        .insert(x, y)                                     [Table-Insert]

        Examples:

            t::.table(d)
            it::.insert(t, [1;2;3;4])

    """
    if not isinstance(x,Table):
        raise KlongDbException(x, "Inserts must be applied to a table")
    if len(x.columns) > 1:
        if not np.isarray(y):
            raise KlongDbException(x, f"Values to insert must be a list")
    elif not np.isarray(y):
            y = np.array([y])
    if len(y) != len(x.columns):
        raise KlongDbException(x, f"Expected {len(x.columns)} values, received {len(y)}")
    x.insert(y)
    return x


def eval_sys_fn_create_db(x):
    """

        .db(x)                                               [Create-db]

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
