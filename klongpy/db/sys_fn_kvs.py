import sys

import numpy as np

from klongpy.core import KlongException

from .df_cache import PandasDataFrameCache
from .file_cache import FileCache
from .helpers import *
from .sys_fn_db import Table


class KlongKvsException(KlongException):
    def __init__(self, x, e):
        self.x = x
        super().__init__(e)


class TableStorage(dict):
    def __init__(self, root_path, max_memory=None):
        self.cache = PandasDataFrameCache(root_path=root_path, max_memory=max_memory)

    def __getitem__(self, x):
        return self.get(x)

    def __setitem__(self, x, y):
        return self.set(x, y)

    def __contains__(self, x):
        raise NotImplementedError()

    def get(self, x):
        if not isinstance(x,str):
            raise KlongKvsException(x, "key must be a str")
        df = self.cache.get_dataframe(key_to_file_path(x), default_empty=False)
        return np.inf if df is None else Table(df)

    def set(self, x, y):
        if not isinstance(x,str):
            raise KlongKvsException(x, "key must be a str")
        if not isinstance(y, Table):
            raise KlongKvsException(y, "value must be a Table")
        self.cache.update(key_to_file_path(x), y.get_dataframe())

    def __len__(self):
        raise NotImplementedError("len not implemented")
    
    def __str__(self):
        return f"{self.cache.root_path}:tables"


class KeyValueStorage(dict):
    def __init__(self, root_path, max_memory=None):
        self.cache = FileCache(root_path=root_path, max_memory=max_memory)

    def __getitem__(self, x):
        return self.get(x)

    def __setitem__(self, x, y):
        return self.set(x, y)

    def __contains__(self, x):
        raise NotImplementedError()

    def get(self, x):
        if not isinstance(x,str):
            raise KlongKvsException(x, "key must be a str")
        return deserialize_obj(self.cache.get_file(key_to_file_path(x)))

    def set(self, x, y):
        if not isinstance(x,str):
            raise KlongKvsException(x, "key must be a str")
        self.cache.update_file(key_to_file_path(x), serialize_obj(y), use_fsync=True)
    
    def __len__(self):
        raise NotImplementedError("len not implemented")
    
    def __str__(self):
        return f"{self.cache.root_path}:kvs"


def eval_sys_fn_create_table_storage(x):
    """

        .tables(x)                                 [TableStorage-Create]
            
        Creates Table storage at file path "x".

        Table storage is similar to key-value storage except that setting a table value
        causes a merge with the existing table on disk.

        Examples:

            ts::.tables("/tmp/tables")

            cols::["s" "c" "v"]
            colsFromNames::{{x,,[]}'x}
            prices:::.table(colsFromNames(cols))
            ts,"tables/prices",prices

            .p(ts?"tables/prices")

            updateDb::{[u d];u::x;d::{u?x}'cols;.p(d);.insert(prices;d)}
            updateDb(:{["s" 1] ["c" 2] ["v" 3]})

            .p(prices)

            ts,"tables/prices",prices

            .p(ts?"tables/prices")

    """
    if not isinstance(x,str):
        raise KlongKvsException(x, "root path must be a string")
    return TableStorage(x, max_memory=1024*1024*1024*10)


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
    if not isinstance(x,str):
        raise KlongKvsException(x, "root path must be a string")
    return KeyValueStorage(root_path=x)


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
