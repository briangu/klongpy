import threading
import weakref

import pandas as pd
from .file_cache import FileCache
from .helpers import deserialize_df, df_memory_usage, serialize_df


class PandasDataFrameCache(FileCache):
    """
    A cache for Pandas DataFrames that uses the FileCache class to store them on disk.

    Args:
        max_memory (int): The maximum amount of memory that the cache should use.
        root_path (str): The root directory for where the cache files should be stored.
    """
    def __init__(self, max_memory=None, root_path=None):
        super().__init__(max_memory=max_memory, root_path=root_path)
        self.append_locks = weakref.WeakValueDictionary()

    def process_contents(self, contents):
        """
        Process the contents of a cache file to return a DataFrame and its memory usage.

        Args:
            contents (bytes): The contents of a cache file.

        Returns:
            tuple: A DataFrame and its memory usage.
        """
        df = pd.DataFrame() if len(contents) == 0 else deserialize_df(contents)
        return df, df_memory_usage(df)

    def get_dataframe(self, file_name, range_start=None, range_end=None, range_type="timestamp"):
        """
        Retrieve a DataFrame from the cache.

        Args:
            file_name (str): The name of the file that contains the DataFrame.
            range_start (int or datetime): The start of the range of rows to retrieve.
            range_end (int or datetime): The end of the range of rows to retrieve.
            range_type (str): The type of the range (either 'timestamp' or 'index')

        Returns:
            DataFrame: The requested DataFrame.
        """
        try:
            df = self.get_file(file_name)
        except FileNotFoundError:
            df = pd.DataFrame()
        if range_start is None:
            if range_end is None:
                return df
            else:
                return df[(df.index <= range_end)] if range_type == "timestamp" else df[:range_end]
        else:
            if range_end is None:
                return df[(df.index >= range_start)] if range_type == "timestamp" else df[range_start:]
            else:
                return df[(df.index >= range_start)&(df.index <= range_end)] if range_type == "timestamp" else df[range_start:range_end]

    def update(self, file_name, new_df):
        """
        Update a DataFrame to the cache file.

        Args:
            file_name (str): The name of the file that should be updated.
            new_df (DataFrame): The DataFrame with the update.

        Returns:
            DataFrame: The new DataFrame.
        """
        with self.file_futures_lock:
            flock = self.append_locks.get(file_name)
            if flock is None:
                flock = threading.Lock()
                self.append_locks[file_name] = flock
        with flock:
            try:
                df = self.get_file(file_name)
                df = pd.concat([df, new_df])
            except FileNotFoundError:
                df = new_df
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='first')]
            update_applied = self.update_file(file_name, serialize_df(df))
            return df if update_applied else self.update(file_name, new_df)
