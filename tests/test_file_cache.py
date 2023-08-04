import os
import platform
import random
import tempfile
import threading
import time
import unittest
from multiprocessing.pool import ThreadPool

from klongpy.db.file_cache import FileCache

# TODO: add MacOS RAM disk
# hdiutil attach -nomount ram://$((2 * 1024 * 100))
# diskutil eraseVolume HFS+ RAMDisk /dev/disk3
# https://stackoverflow.com/questions/1854/how-to-identify-which-os-python-is-running-on

def gen_data():
    return bytearray(os.urandom(random.randint(10,50)))

def gen_file(tmp_dir=None):
    tmp_dir = tmp_dir or ("/dev/shm" if platform.system() == "Linux" else None)
    f = tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir)
    d = gen_data()
    f.write(d)
    f.seek(0)
    return f,d,len(d)

class FileCacheTests(unittest.TestCase):
    def setUp(self):
        self.file_contents = {i:gen_file() for i in range(10)}
        # make cache just under capactiy to hold all 4 files to force eviction
        self.file_cache = FileCache(max_memory=(sum([x[2] for x in self.file_contents.values()]) - 1))

    def test_get_file(self):
        info = self.file_contents[0]
        contents = self.file_cache.get_file(info[0].name)
        self.assertEqual(self.file_cache.current_memory_usage, info[2])
        self.assertTrue(info[0].name in self.file_cache.file_futures)
        self.assertEqual(self.file_cache.file_futures[info[0].name][1], info[2])
        self.assertEqual(contents, info[1])
        self.assertEqual(len(self.file_cache.file_access_times), 1)
        self.assertEqual(info[0].name, self.file_cache.file_access_times[0][1])

    def test_get_file_with_eviction(self):
        for i in range(2):
            for info in self.file_contents.values():
                expected_memory_usage = self.file_cache.current_memory_usage + info[2]
                if expected_memory_usage > self.file_cache.max_memory:
                    oldest_size = self.file_cache.file_futures[self.file_cache.file_access_times[0][1]][1]
                    expected_memory_usage -= oldest_size
                contents = self.file_cache.get_file(info[0].name)
                self.assertEqual(self.file_cache.current_memory_usage, expected_memory_usage)
                self.assertTrue(info[0].name in self.file_cache.file_futures)
                self.assertEqual(self.file_cache.file_futures[info[0].name][1], info[2])
                self.assertEqual(contents, info[1])
                self.assertEqual(info[0].name, self.file_cache.file_access_times[-1][1])

    def test_get_file_multithreaded(self):
        def get_file_thread(file, data, size):
            self.assertEqual(self.file_cache.get_file(file.name), data)

        with ThreadPool() as pool:
            pool.starmap(get_file_thread, random.choices(list(self.file_contents.values()), k=100))

    def test_unload_file(self):
        for info in self.file_contents.values():
            self.file_cache.get_file(info[0].name)
            self.assertEqual(self.file_cache.current_memory_usage, info[2])
            self.file_cache.unload_file(info[0].name)
            self.assertFalse(info[0].name in self.file_cache.file_futures)
            self.assertEqual(self.file_cache.current_memory_usage, 0)
            self.assertEqual(len(self.file_cache.file_access_times), 0)

    def test_unload_file_multithreaded(self):
        def unload_file_thread(file, data, size):
            self.file_cache.get_file(file.name)
            self.assertTrue(file.name in self.file_cache.file_futures)
            self.file_cache.unload_file(file.name)
            self.assertFalse(file.name in self.file_cache.file_futures)

        with ThreadPool() as pool:
            for _ in range(10):
                pool.starmap(unload_file_thread, list(self.file_contents.values()))

    def test_update_file(self):
        info = self.file_contents[0]
        content = self.file_cache.get_file(info[0].name)
        self.assertEqual(content, info[1])

        # Update the file in the cache
        new_contents = gen_data()
        updated = self.file_cache.update_file(info[0].name, new_contents)
        self.assertTrue(updated)
        self.assertTrue(info[0].name in self.file_cache.file_futures)
        self.assertEqual(self.file_cache.current_memory_usage, len(new_contents))
        self.assertEqual(len(self.file_cache.file_access_times), 1)
        self.assertEqual(info[0].name, self.file_cache.file_access_times[0][1])

        # Check that the updated file is returned
        content = self.file_cache.get_file(info[0].name)
        self.assertEqual(content, new_contents)

        self.file_cache.unload_file(info[0].name)
        self.assertFalse(info[0].name in self.file_cache.file_futures)
        self.assertEqual(self.file_cache.current_memory_usage, 0)
        self.assertEqual(len(self.file_cache.file_access_times), 0)

    def test_update_file_multithreaded(self):
        def update_file_thread(file, data, size):
            info = (file, data, size)
            updated = self.file_cache.update_file(info[0].name, info[1])
            self.assertTrue(updated)
            content = self.file_cache.get_file(info[0].name)
            self.assertEqual(content, info[1])

            # Update the file in the cache
            new_contents = gen_data()
            updated = self.file_cache.update_file(info[0].name, new_contents)
            self.assertTrue(updated)

            # Check that the updated file is returned
            content = self.file_cache.get_file(info[0].name)
            self.assertEqual(content, new_contents)

            self.file_cache.unload_file(info[0].name)
            self.assertFalse(info[0].name in self.file_cache.file_futures)

        with ThreadPool() as pool:
            for _ in range(10):
                pool.starmap(update_file_thread, list(self.file_contents.values()))

    def test_update_file_multithreaded_expected(self):
        info = self.file_contents[0]
        name = info[0].name
        self.failed = False
        self.run = True

        # Create a list of expected values for each thread
        expected_values = [gen_data() for _ in range(10)]
        all_read_expected_values = [info[1], *expected_values]

        def update_file_thread(file_name, contents):
            while self.run:
                updated = self.file_cache.update_file(file_name, contents)
                while not updated:
                    updated = self.file_cache.update_file(file_name, contents)

        def get_file_thread():
            while self.run:
                updated_file_1_content = self.file_cache.get_file(name)
                if updated_file_1_content not in all_read_expected_values:
                    self.failed = True
                self.assertIn(updated_file_1_content, all_read_expected_values)

        # Update the test file in multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=update_file_thread, args=(name, expected_values[i]))
            thread.start()
            threads.append(thread)

        for i in range(100):
            thread = threading.Thread(target=get_file_thread)
            thread.start()
            threads.append(thread)

        time.sleep(3)
        self.run = False

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        self.assertFalse(self.failed)

        # Check that the final contents of the file match one of the expected values
        updated_file_1_content = self.file_cache.get_file(name)
        self.assertIn(updated_file_1_content, expected_values)

    def tearDown(self):
        for v in self.file_contents.values():
            os.unlink(v[0].name)

if __name__ == '__main__':
  unittest.main()
