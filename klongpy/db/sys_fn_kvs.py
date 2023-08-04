import asyncio
import sys
import threading

import duckdb
import numpy as np
import pandas as pd

from klongpy.core import KlongException, KGCall, KGLambda, reserved_fn_args, reserved_fn_symbol_map

import time

from sys_fn_db import Table

from df_cache import PandasDataFrameCache
from file_cache import FileCache


class KlongDfsException(KlongException):
    def __init__(self, x, e):
        self.x = x
        super().__init__(e)


class TableStorage(dict):
    def __init__(self, filepath, max_memory=None):
        self.cache = PandasDataFrameCache(root_path=filepath, max_memory=max_memory)

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
        if isinstance(y, Table):
            pass
        else:
            pass
    
    def __len__(self):
        return len(self._df.columns)
    
    def __str__(self):
        return f"{self.idx_cols or ''}:{self.columns}:table"


class KeyValueStorage(dict):
    def __init__(self, filepath, max_memory=None):
        self.cache = PandasDataFrameCache(root_path=filepath, max_memory=max_memory)

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
        if hasattr(y, "get_dataframe"):
            pass
        else:
            pass
    
    def __len__(self):
        return len(self._df.columns)
    
    def __str__(self):
        return f"{self.idx_cols or ''}:{self.columns}:table"


def eval_sys_fn_create_table_storage(x):
    """

        .tables(x)                                 [TableStorage-Create]
            
        Creates Table storage at file path "x".

        Table storage is similar to key-value storage except that setting a table value
        causes a merge with the existing table on disk.

        Examples:

            ts::.tables("/tmp/tables")
            prices::.table(d)
            ts,"prices",prices

            prices::ts?"prices"

    """
    if not np.isarray(x):
        raise KlongDfsException(x, "tables must be created from an array of column data map")
    return TableStorage(x)


def eval_sys_fn_create_kvs(x):
    """

        .kvs(x)                                 [KeyValueStorage-Create]
            
        Creates file backed key-value storage at file path "x".

        Values are stored as bytes on disk via pickle.

        Examples:
                
            kvs::.kvs("/tmp/storage")
            v::"value"
            kvs,"key",v
            v::kvs?"key"

    """
    if not np.isarray(x):
        raise KlongDfsException(x, "tables must be created from an array of column data map")
    return KeyValueStorage(x)


def create_system_functions_kvs():
    def _get_name(s):
        i = s.index(".")
        return s[i : i + s[i:].index("(")]

    registry = {}

    m = sys.modules[__name__]
    for x in filter(lambda n: n.startswith("eval_sys_"), dir(m)):
        fn = getattr(m, x)
        registry[_get_name(fn.__doc__)] = fn

    return registry
