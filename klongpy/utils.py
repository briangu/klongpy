import collections
import time
from functools import wraps


class ReadonlyDict(collections.abc.Mapping):

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end_time = time.perf_counter_ns()
        total_time = (end_time - start_time) / (10**9)
#        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        print(f'Function {func.__name__} Took {total_time:.6f} s')
        return result
    return timeit_wrapper
