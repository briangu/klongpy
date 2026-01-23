import os
import tempfile
import unittest

import pandas as pd

from klongpy.db.helpers import (
    file_path_to_key_path,
    key_to_file_path,
    save_gzip_df_file,
    read_gzip_df_file,
    save_df_file,
    read_df_file,
    df_memory_usage,
    serialize_gzip_df,
    deserialize_gzip_df,
    serialize_obj,
    deserialize_obj,
    serialize_df,
    deserialize_df,
)


class TestHelpersFunctions(unittest.TestCase):
    def test_file_path_to_key_path(self):
        path = os.path.join("foo", "bar", "baz")
        result = file_path_to_key_path(path)
        self.assertEqual(result, ["foo", "bar", "baz"])

    def test_key_to_file_path(self):
        key = "some/path/to/file"
        result = key_to_file_path(key)
        self.assertEqual(result, key)


class TestDataFrameFileIO(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_read_gzip_df_file(self):
        file_path = os.path.join(self.temp_dir, "test.pkl.gz")
        save_gzip_df_file(file_path, self.df)
        result = read_gzip_df_file(file_path)
        pd.testing.assert_frame_equal(result, self.df)

    def test_save_and_read_df_file(self):
        file_path = os.path.join(self.temp_dir, "test.pkl")
        save_df_file(file_path, self.df)
        result = read_df_file(file_path)
        pd.testing.assert_frame_equal(result, self.df)

    def test_df_memory_usage(self):
        import numpy as np
        result = df_memory_usage(self.df)
        self.assertTrue(np.issubdtype(type(result), np.integer) or isinstance(result, (int, float)))
        self.assertGreater(result, 0)


class TestDataFrameSerialization(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})

    def test_serialize_and_deserialize_gzip_df(self):
        data = serialize_gzip_df(self.df)
        self.assertIsInstance(data, bytes)
        result = deserialize_gzip_df(data)
        pd.testing.assert_frame_equal(result, self.df)

    def test_serialize_and_deserialize_df(self):
        data = serialize_df(self.df)
        self.assertIsInstance(data, bytes)
        result = deserialize_df(data)
        pd.testing.assert_frame_equal(result, self.df)


class TestObjectSerialization(unittest.TestCase):
    def test_serialize_and_deserialize_dict(self):
        obj = {'key': 'value', 'num': 42}
        data = serialize_obj(obj)
        self.assertIsInstance(data, bytes)
        result = deserialize_obj(data)
        self.assertEqual(result, obj)

    def test_serialize_and_deserialize_list(self):
        obj = [1, 2, 3, 'a', 'b', 'c']
        data = serialize_obj(obj)
        result = deserialize_obj(data)
        self.assertEqual(result, obj)

    def test_serialize_and_deserialize_nested(self):
        obj = {'list': [1, 2, {'nested': True}], 'value': 123}
        data = serialize_obj(obj)
        result = deserialize_obj(data)
        self.assertEqual(result, obj)


if __name__ == '__main__':
    unittest.main()
