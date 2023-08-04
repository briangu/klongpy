import os
import platform
import tempfile
import threading
import unittest

import pandas as pd

from klongpy.db.df_cache import PandasDataFrameCache

# TODO: add MacOS RAM disk
# hdiutil attach -nomount ram://$((2 * 1024 * 100))
# diskutil eraseVolume HFS+ RAMDisk /dev/disk3
# https://stackoverflow.com/questions/1854/how-to-identify-which-os-python-is-running-on
tmp_dir = "/dev/shm" if platform.system() == "Linux" else None

class TestPandasDataFrameCache(unittest.TestCase):
    def setUp(self):
        self.test_file_1 = tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir)
        self.df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=[1, 2, 3])
        self.cache = PandasDataFrameCache(max_memory=1024)
        self.cache.update(self.test_file_1.name, self.df)

    def test_get_dataframe(self):
        start = 2
        end = 3
        result = self.cache.get_dataframe(self.test_file_1.name, start, end)
        expected = self.df.loc[start:end]
        pd.testing.assert_frame_equal(result, expected)

    def test_append(self):
        new_df = pd.DataFrame({'A': [4, 5, 6], 'B': [7, 8, 9]}, index=[4, 5, 6])
        result = self.cache.update(self.test_file_1.name, new_df)
        expected = pd.concat([self.df, new_df]).sort_index().drop_duplicates()
        pd.testing.assert_frame_equal(result, expected)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.cache.get_file('not_found.pickle')

    def test_memory_error(self):
        with self.assertRaises(MemoryError):
            self.cache = PandasDataFrameCache(max_memory=1)
            self.cache.update(self.test_file_1.name, self.df)

    def tearDown(self):
        os.unlink(self.test_file_1.name)


class TestMultithreadedPandasDataFrameCache(unittest.TestCase):
    def setUp(self):
        self.test_file_1 = tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir)
        self.test_file_2 = tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir)
        self.cache = PandasDataFrameCache(max_memory=2**30)

    def test_append_concurrently(self):
        def run_append_1(cache, start_i):
            values = list(range(start_i, start_i+100))
            new_df = pd.DataFrame({'A': values, 'B': values}, index=values)
            cache.update(self.test_file_1.name, new_df)
        def run_append_2(cache, start_i):
            values = list(range(start_i, start_i+100))
            new_df = pd.DataFrame({'C': values, 'D': values}, index=values)
            cache.update(self.test_file_2.name, new_df)

        threads = []
        for i in range(100):
            thread = threading.Thread(target=run_append_1, args=(self.cache, i*100))
            thread.start()
            threads.append(thread)
        for i in range(150):
            thread = threading.Thread(target=run_append_2, args=(self.cache, i*100))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

        result = self.cache.get_file(self.test_file_1.name)
        self.assertEqual(len(result), 10000)
        result = self.cache.get_file(self.test_file_2.name)
        self.assertEqual(len(result), 15000)

    def tearDown(self):
        os.remove(self.test_file_1.name)
        os.remove(self.test_file_2.name)
