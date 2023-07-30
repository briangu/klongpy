import asyncio
import logging
import pickle
import socket
import struct
import sys
import threading
from asyncio import StreamReader, StreamWriter
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from klongpy.core import (KGCall, KGFn, KGFnWrapper, KGLambda, KGSym,
                          get_fn_arity_str, is_list, reserved_fn_args,
                          reserved_fn_symbol_map)

_main_loop = asyncio.get_event_loop()
_main_tid = threading.current_thread().ident


class Table(dict):
    def __init__(self, d):
        self.df = pd.DataFrame(d)

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

    # def close(self):
    #     return self.nc.close()

    # def is_open(self):
    #     return self.nc.is_open()

    def __len__(self):
        return len(self.df.columns)
    
    def __str__(self):
#        return f":table"
        return str(self.df)


def eval_sys_fn_create_table(x):
    """

        .table(x)                                         [Create-table]

    """
    if not isinstance(x,dict):
        return "tables must be created from a dict"
    return Table(x)


class Database(KGLambda):

    def __init__(self, tables):
        self.tables = tables
        self.con = duckdb.connect(':memory:')

    def __call__(self, _, ctx):
        x = ctx[reserved_fn_symbol_map[reserved_fn_args[0]]]
        for k,v in self.tables.items():
            locals()[k] = v.df

        try:
            df = self.con.execute(x).fetchdf()
            return df.values.squeeze()
        except Exception as e:
            print(e)
        finally:
            for k in self.tables.keys():
                del locals()[k]

    def get_arity(self):
        return 1

    def __str__(self):
        return ":db"

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
