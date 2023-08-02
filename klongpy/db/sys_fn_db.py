import asyncio
import sys
import threading

import duckdb
import numpy as np
import pandas as pd

from klongpy.core import KlongException, KGCall, KGLambda, reserved_fn_args, reserved_fn_symbol_map

_main_loop = asyncio.get_event_loop()
_main_tid = threading.current_thread().ident


class KlongDbException(KlongException):
    def __init__(self, x, e):
        self.x = x
        super().__init__(e)


class Table(dict):
    def __init__(self, d, columns):
        self.df = pd.DataFrame(d, columns=columns)
        self.index = None
        self.columns = columns
        self.val_cols = columns

    def __getitem__(self, x):
        return self.get(x)

    def __setitem__(self, x, y):
        return self.set(x, y)

    def __contains__(self, x):
        raise NotImplementedError()

    def get(self, x):
        v = self.df.get(x)
        return np.inf if v is None else v.values

    def set(self, x, y):
        self.df[x] = y

    def schema(self):
        return np.array(self.df.columns, dtype=object)

    def set_index(self, columns):
        self.df.set_index(columns, inplace=True)
        self.index = columns
        self.val_cols = [k for k in self.columns if k not in self.index]

    def reset_index(self):
        self.df.reset_index()
        self.index = None
        self.val_cols = self.columns

    # def close(self):
    #     return self.nc.close()

    # def is_open(self):
    #     return self.nc.is_open()

    def __len__(self):
        return len(self.df.columns)
    
    def __str__(self):
#        return f":table"
        return str(self.df)


class Database(KGLambda):

    def __init__(self, tables):
        self.tables = tables
        self.con = duckdb.connect(':memory:')

    def __call__(self, _, ctx):
        x = ctx[reserved_fn_symbol_map[reserved_fn_args[0]]]

        # add the table column -> dataframe into local scope so DuckDB can reference them by name in the SQL.
        for k,v in self.tables.items():
            locals()[k] = v.df

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
    if x.index is not None:
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
    return x.reset_index(x)


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
    if not np.isarray(y):
        raise KlongDbException(x, f"Values to insert must be a list")
    if len(y) != len(x.columns):
        raise KlongDbException(x, f"Expected {len(x.columns)} values, received {len(y)}")
    if x.index is None:
        x.df.loc[len(x.df)] = y
    elif len(x.index) == 1:
        idx_val = None
        values = []
        for k,v in zip(x.columns, y):
            if k in x.index:
                idx_val = v
            else:
                values.append(v)
        x.df.loc[idx_val] = values
    else:
        idx_vals = []
        values = []
        for k,v in zip(x.columns, y):
            if k in x.index:
                idx_vals.append(v)
            else:
                values.append(v)
        x.df.loc[tuple(idx_vals), *x.val_cols] = values
    return 


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
