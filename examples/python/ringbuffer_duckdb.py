import ctypes
import struct
import time
import multiprocessing as mp
from multiprocessing import shared_memory, Process
import duckdb
import pandas as pd

class MMRingBuffer:
    INT64_SIZE = 8

    def __init__(self, name, size_pow2=1024, create=False):
        if size_pow2 & (size_pow2 - 1):
            raise ValueError("size_pow2 must be a power of two")
        self.size = size_pow2
        self.mask = self.size - 1
        header = 16
        data = self.size * self.INT64_SIZE
        self.shm = shared_memory.SharedMemory(name=name, create=create, size=header + data) if create else shared_memory.SharedMemory(name=name)
        self.buf = self.shm.buf
        self.cur = ctypes.c_long.from_buffer(self.buf, 0)
        self.read = ctypes.c_long.from_buffer(self.buf, 8)
        if create:
            self.cur.value = self.read.value = -1
        self.off = header

    def _ptr(self, s):
        return self.off + ((s & self.mask) << 3)

    def publish(self, v):
        nxt = self.cur.value + 1
        while nxt - self.read.value >= self.size:
            time.sleep(0)
        struct.pack_into('q', self.buf, self._ptr(nxt), v)
        self.cur.value = nxt

    def consume(self):
        nxt = self.read.value + 1
        while nxt > self.cur.value:
            time.sleep(0)
        v, = struct.unpack_from('q', self.buf, self._ptr(nxt))
        self.read.value = nxt
        return v

    def close(self):
        self.shm.close()

NAME = "mmring2"
SIZE = 1024
N = 20_000


def prod():
    r = MMRingBuffer(NAME, SIZE)
    for i in range(N):
        r.publish(i)
    r.publish(-1)
    r.close()


def cons():
    r = MMRingBuffer(NAME, SIZE)
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE events(v INTEGER)")
    buf = []
    while True:
        v = r.consume()
        if v == -1:
            break
        buf.append(v)
        if len(buf) >= 1000:
            df = pd.DataFrame({'v': buf})
            con.execute("INSERT INTO events SELECT * FROM df")
            buf.clear()
    if buf:
        df = pd.DataFrame({'v': buf})
        con.execute("INSERT INTO events SELECT * FROM df")
    count = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    print("rows inserted:", count)
    r.close()


if __name__ == "__main__":
    ring = MMRingBuffer(NAME, SIZE, create=True)
    pc = Process(target=cons)
    pp = Process(target=prod)
    pc.start()
    pp.start()
    pp.join()
    pc.join()
    ring.close()
