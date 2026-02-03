import os
import shutil
import tempfile
import unittest

import pandas as pd

from klongpy.core import KLONG_UNDEFINED
from klongpy.db.sys_fn_kvs import (
    TableStorage,
    KeyValueStorage,
    KlongKvsException,
    eval_sys_fn_create_table_storage,
    eval_sys_fn_create_kvs,
    create_system_functions_kvs,
)
from klongpy.db.sys_fn_db import Table


class TestTableStorage(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.storage = TableStorage(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_set_and_get_table(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        table = Table(df)
        self.storage.set("test_table", table)
        result = self.storage.get("test_table")
        self.assertIsInstance(result, Table)
        pd.testing.assert_frame_equal(result.get_dataframe(), df)

    def test_setitem_and_getitem(self):
        df = pd.DataFrame({'X': [10, 20], 'Y': [30, 40]})
        table = Table(df)
        self.storage["my_table"] = table
        result = self.storage["my_table"]
        self.assertIsInstance(result, Table)
        pd.testing.assert_frame_equal(result.get_dataframe(), df)

    def test_get_nonexistent_returns_undefined(self):
        result = self.storage.get("nonexistent")
        self.assertIs(result, KLONG_UNDEFINED)

    def test_set_non_string_key_raises(self):
        df = pd.DataFrame({'A': [1]})
        table = Table(df)
        with self.assertRaises(KlongKvsException) as ctx:
            self.storage.set(123, table)
        self.assertIn("key must be a str", str(ctx.exception))

    def test_get_non_string_key_raises(self):
        with self.assertRaises(KlongKvsException) as ctx:
            self.storage.get(123)
        self.assertIn("key must be a str", str(ctx.exception))

    def test_set_non_table_value_raises(self):
        with self.assertRaises(KlongKvsException) as ctx:
            self.storage.set("key", "not a table")
        self.assertIn("value must be a Table", str(ctx.exception))

    def test_contains_raises(self):
        with self.assertRaises(NotImplementedError):
            _ = "key" in self.storage

    def test_len_raises(self):
        with self.assertRaises(NotImplementedError):
            _ = len(self.storage)

    def test_str(self):
        result = str(self.storage)
        self.assertIn(self.temp_dir, result)
        self.assertIn("tables", result)


class TestKeyValueStorage(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.storage = KeyValueStorage(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_set_and_get_value(self):
        self.storage.set("my_key", "my_value")
        result = self.storage.get("my_key")
        self.assertEqual(result, "my_value")

    def test_setitem_and_getitem(self):
        self.storage["key1"] = {"data": 123}
        result = self.storage["key1"]
        self.assertEqual(result, {"data": 123})

    def test_set_and_get_list(self):
        self.storage.set("list_key", [1, 2, 3, 4, 5])
        result = self.storage.get("list_key")
        self.assertEqual(result, [1, 2, 3, 4, 5])

    def test_set_non_string_key_raises(self):
        with self.assertRaises(KlongKvsException) as ctx:
            self.storage.set(123, "value")
        self.assertIn("key must be a str", str(ctx.exception))

    def test_get_non_string_key_raises(self):
        with self.assertRaises(KlongKvsException) as ctx:
            self.storage.get(456)
        self.assertIn("key must be a str", str(ctx.exception))

    def test_contains_raises(self):
        with self.assertRaises(NotImplementedError):
            _ = "key" in self.storage

    def test_len_raises(self):
        with self.assertRaises(NotImplementedError):
            _ = len(self.storage)

    def test_str(self):
        result = str(self.storage)
        self.assertIn(self.temp_dir, result)
        self.assertIn("kvs", result)


class TestCreateFunctions(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_eval_sys_fn_create_table_storage(self):
        result = eval_sys_fn_create_table_storage(self.temp_dir)
        self.assertIsInstance(result, TableStorage)

    def test_eval_sys_fn_create_table_storage_non_string_raises(self):
        with self.assertRaises(KlongKvsException) as ctx:
            eval_sys_fn_create_table_storage(123)
        self.assertIn("root path must be a string", str(ctx.exception))

    def test_eval_sys_fn_create_kvs(self):
        result = eval_sys_fn_create_kvs(self.temp_dir)
        self.assertIsInstance(result, KeyValueStorage)

    def test_eval_sys_fn_create_kvs_non_string_raises(self):
        with self.assertRaises(KlongKvsException) as ctx:
            eval_sys_fn_create_kvs(123)
        self.assertIn("root path must be a string", str(ctx.exception))


class TestCreateSystemFunctionsKvs(unittest.TestCase):
    def test_create_system_functions_kvs(self):
        registry = create_system_functions_kvs()
        self.assertIsInstance(registry, dict)
        self.assertIn(".tables", registry)
        self.assertIn(".kvs", registry)


if __name__ == '__main__':
    unittest.main()
