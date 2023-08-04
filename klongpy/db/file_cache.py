import heapq
import os
import time
from concurrent.futures import ThreadPoolExecutor
from .helpers import tinfo
from threading import Lock
import logging


class FileCache:
    def __init__(self, max_memory=None, root_path=None):
        """
        Initializes the FileCache with a maximum memory limit and the root directory for file storage.
        If max_memory is not specified, it defaults to 2**20 bytes.
        If root_path is not specified, it defaults to the current working directory.

        Args:
        - max_memory (int): the maximum amount of memory to use (in bytes)
        - root_path (str): the directory where files will be stored

        Returns:
        None
        """
        self.max_memory = max_memory or 2**20
        self.root_path = root_path or os.getcwd()
        self.current_memory_usage = 0
        self.file_futures = {}
        self.file_access_times = []
        self.file_futures_lock = Lock()
        self.executor = ThreadPoolExecutor()

    def process_contents(self, contents):
        """
        A hook for processing file contents before they are loaded into memory.
        Returns a tuple of the processed contents and the memory usage of the contents.

        Args:
        - contents (Union[str, bytes]): the contents of the file

        Returns:
        - tuple: a tuple of the processed contents and the memory usage of the contents
        """
        return contents, len(contents)

    def _load_file(self, file_name):
        """
        Loads the specified file into memory and updates the memory usage and file future.

        Args:
        - file_name (str): the name of the file to be loaded

        Returns:
        - object: The processed contents of the file
        """
        with open(os.path.join(self.root_path, file_name), 'rb') as file:
            contents, memory_usage = self.process_contents(file.read())
        self.update_file_futures_and_memory(file_name, memory_usage=memory_usage)
        return contents

    def _write_file(self, file_name, new_file_contents, use_fsync):
        """
        Write a file to the filesystem, with option to use fsync to ensure that all data is written to the filesystem.

        Args:
        - file_name (str): the name of the file to be written
        - new_file_contents (Union[str, bytes]): the new contents of the file
        - use_fsync (bool): whether to use fsync to ensure data is written to the filesystem

        Returns:
        - object: The processed contents of the file
        """
        write_fname = os.path.join(self.root_path, file_name)
        write_path = os.path.dirname(write_fname)
        os.makedirs(write_path, exist_ok=True)
        with open(os.path.join(self.root_path, file_name), 'wb') as f:
            f.write(new_file_contents)
            if use_fsync:
                os.fsync(f.fileno())
        contents, memory_usage = self.process_contents(new_file_contents)
        self.update_file_futures_and_memory(file_name, memory_usage=memory_usage)
        return contents

    def update_file_access_time(self, file_name):
        """
        Updates the access time of the specified file and reorders the file access time heap.

        Args:
        - file_name (str): the name of the file to update the access time for

        Returns:
        None
        """
        assert self.file_futures_lock.locked()
        assert self.file_futures.get(file_name) is not None
        self.file_access_times = [(t, fn) for t, fn in self.file_access_times if fn != file_name]
        heapq.heappush(self.file_access_times, (time.time_ns(), file_name))

    def update_file_futures_and_memory(self, file_name, memory_usage):
        """
        Updates the memory usage and file future for the specified file.

        Args:
        - file_name (str): the name of the file to update
        - memory_usage (int): the memory usage of the file

        Returns:
        None
        """
        with self.file_futures_lock:
            can_cache = self.recover_memory(memory_usage)
            if not can_cache:
                logging.warning(f"unable to recover memory for requsted file: {file_name} {memory_usage} {self.max_memory} {self.current_memory_usage}")
            info = self.file_futures.get(file_name)
            # the file can't be unloaded until it's got a file access time
            assert info is not None
            if can_cache:
                self.update_file_access_time(file_name)
                self.current_memory_usage += memory_usage
                self.file_futures[file_name] = (False, memory_usage, info[-1])
            else:
                del self.file_futures[file_name]

    def update_file(self, file_name, new_file_contents, use_fsync=False):
        """
        Update the content of a file.

        Calling thread will block until write is applied, even if it's being done by another thread.
        When the method exits, the caller is notified if their update was applied.  If not, then they
        can read the contents and apply their changes again and resubmit.

        Args:
        - file_name (str): the name of the file to be updated
        - new_data (Union[str, bytes]): the new content of the file
        - use_fsync (bool): use fsync when writing to disk

        Returns:
        bool - True if update was applied.
        """
        claim = len(new_file_contents)
        if claim > self.max_memory:
            raise MemoryError(f"requested file update larger than max_memory: {file_name} {claim} {self.max_memory}")
        with self.file_futures_lock:
            info = self.file_futures.get(file_name)
            if info is None or not info[0]:
                self._unload_file(file_name)
                future = self.executor.submit(self._write_file, file_name, new_file_contents, use_fsync)
                self.file_futures[file_name] = (True, claim, future)
                write_applied = True
            else:
                assert info[0]
                future = info[-1]
                write_applied = False
        future.result()
        return write_applied

    def _unload_file(self, file_name):
        """
        Unload a file from memory.

        Args:
        file_name (str): the name of the file to unload

        Returns:
        None
        """
        assert self.file_futures_lock.locked()
        info = self.file_futures.get(file_name)
        if info is not None:
            self.current_memory_usage -= info[1]
            del self.file_futures[file_name]

    def unload_file(self, file_name):
        """
        Unload a file from memory.

        Args:
        file_name (str): the name of the file to unload

        Returns:
        None
        """
        with self.file_futures_lock:
            self.file_access_times = [(t, fn) for t, fn in self.file_access_times if fn != file_name]
            heapq.heapify(self.file_access_times)
            self._unload_file(file_name)

    def recover_memory(self, claim):
        """
        Recover memory by unloading files from memory until the claim is achieved.

        Args:
        claim (int): amount of memory to claim

        Returns:
        bool: True if the claim is achieved, False otherwise
        """
        assert self.file_futures_lock.locked()
        assert claim <= self.max_memory
        # start_mem = self.current_memory_usage
        writing = None
        while (self.current_memory_usage + claim) > self.max_memory and len(self.file_access_times) > 0:
            data = heapq.heappop(self.file_access_times)
            oldest_file = data[1]
            if self.file_futures[oldest_file][0]:
                if writing is None:
                    writing = []
                writing.append(data)
                continue
            self._unload_file(oldest_file)
        if writing is not None:
            for x in writing:
                heapq.heappush(self.file_access_times, x)
        # recovered_mem = self.current_memory_usage - start_mem
        # if recovered_mem > 0:
        #     tinfo(f"recovered: {recovered_mem}")
        return (self.current_memory_usage + claim) <= self.max_memory

    def get_file(self, file_name):
        """
        Retrieve a file's content from memory.

        Args:
        - file_name (str): the name of the file to retrieve

        Returns:
        bytes: the raw file contents
        """
        full_file_path = os.path.join(self.root_path, file_name)
        if not os.path.exists(full_file_path):
            raise FileNotFoundError(file_name)
        claim = os.path.getsize(full_file_path)
        if claim > self.max_memory:
            raise MemoryError(f"requested file larger than max_memory: {file_name} {claim} {self.max_memory}")
        with self.file_futures_lock:
            info = self.file_futures.get(file_name)
            if info is None:
                tinfo(f"get_file: {file_name}")
                future = self.executor.submit(self._load_file, file_name)
                self.file_futures[file_name] = (False, claim, future)
            else:
                tinfo(f"get_file [cached]: {file_name}")
                future = info[-1]
                if future.done():
                    self.update_file_access_time(file_name)
        return future.result()
