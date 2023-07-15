import collections
# import logging
# import struct
# import threading
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


# def tinfo(msg):
#     logging.info(f"tid: {threading.current_thread().ident}: " + msg)


# def pack_msg(msg):
#     return struct.pack('>I', len(msg)) + msg


# def send_msg(conn, msg):
#     conn.sendall(struct.pack('>I', len(msg)) + msg)


# def recvall(conn, n):
#     data = bytearray()
#     while len(data) < n:
#         packet = conn.recv(n - len(data))
#         if not packet:
#             return None
#         data.extend(packet)
#     return data


# def recv_msg(conn):
#     raw_msglen = recvall(conn, 4)
#     if not raw_msglen:
#         # raise RuntimeError("server error")
#         return None
#     msglen = struct.unpack('>I', raw_msglen)[0]
#     return recvall(conn, msglen)

