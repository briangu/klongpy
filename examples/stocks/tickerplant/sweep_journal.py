"""Local checksummed batch journal and open-day replay for the RAM sweep fixture."""
import argparse
import hmac
import os
from pathlib import Path
import socket
import struct
import zlib

import numpy as np

from klongpy.ipc import Handshake, Message, Session
from .bench import endpoint


HEADER = struct.Struct("<4sQQII")
MAGIC = b"KGT1"
MAX_ROWS = 1048576


def records(path, day):
    with path.open("rb") as stream:
        while header := stream.read(HEADER.size):
            if len(header) != HEADER.size:
                raise ValueError("incomplete journal header")
            magic, written_day, first, n, checksum = HEADER.unpack(header)
            if magic != MAGIC or written_day != day or not 0 < n <= MAX_ROWS:
                raise ValueError("invalid journal header")
            data = stream.read(n * 32)
            if len(data) != n * 32 or zlib.crc32(data) != checksum:
                raise ValueError("incomplete or damaged journal record")
            yield first, np.frombuffer(data, dtype="<i8").reshape(4, n)


class Journal:
    def __init__(self, directory):
        self.directory = directory
        self.day = self.file = self.next = None

    def open_day(self, day):
        if self.file:
            self.file.close()
        self.next = None
        path = self.directory / f"day-{day}.wal"
        new = not path.exists()
        if not new:
            for first, cols in records(path, day):
                if self.next is not None and first != self.next:
                    raise ValueError("discontinuous journal")
                self.next = first + cols.shape[1]
        self.file = path.open("ab", buffering=0)
        if new:
            parent = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        self.day = day

    def append(self, day, first, *columns):
        day, first = int(day), int(first)
        if day < 0 or first < 0:
            raise ValueError("day and sequence must be nonnegative")
        values = [np.asarray(col) for col in columns]
        if len(values) != 4 or any(col.dtype.kind != "i" or col.ndim != 1 for col in values):
            raise ValueError("expected four signed integer vectors")
        n = len(values[0])
        if not 0 < n <= MAX_ROWS or any(len(col) != n for col in values):
            raise ValueError("expected nonempty equal-length columns within one journal record")
        if self.day != day:
            self.open_day(day)
        if self.next is not None and self.next != first:
            raise ValueError("unexpected next sequence")
        cols = [np.ascontiguousarray(col, dtype="<i8") for col in values]
        blocks = [memoryview(col).cast("B") for col in cols]
        checksum = 0
        for block in blocks:
            checksum = zlib.crc32(block, checksum)
        blocks.insert(0, memoryview(HEADER.pack(MAGIC, day, first, n, checksum)))
        fd = self.file.fileno()
        while blocks:
            count = os.writev(fd, blocks)
            if count <= 0:
                raise OSError("short journal write")
            while blocks and count >= len(blocks[0]):
                count -= len(blocks.pop(0))
            if count:
                blocks[0] = blocks[0][count:]
        getattr(os, "fdatasync", os.fsync)(fd)
        self.next = first + n
        return self.next


def serve(directory, port, password=""):
    os.umask(0o077)
    directory.mkdir(parents=True, exist_ok=True)
    journal = Journal(directory)
    with socket.create_server(("127.0.0.1", port)) as server:
        while True:
            conn, _ = server.accept()
            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                session = Session.server()
                while data := conn.recv(1024 * 1024):
                    for event in session.feed(data):
                        if isinstance(event, Handshake):
                            if password and (event.user != "bench" or not hmac.compare_digest(event.password or "", password)):
                                session.reject()
                                conn.shutdown(socket.SHUT_RDWR)
                                break
                            conn.sendall(session.accept())
                        elif isinstance(event, Message):
                            service = session.begin_service(event)
                            name, *args = event.value
                            if str(name) != "append":
                                raise ValueError("expected a journal append")
                            # A failed write stops the fixture before a positive acknowledgment.
                            watermark = journal.append(*args)
                            conn.sendall(session.send_response(service, watermark))
                            session.end_service(service)


def replay(directory, head_addr, hdb_addr):
    with endpoint(head_addr) as head, endpoint(hdb_addr) as hdb:
        if list(head.request("[;dayNo;dayFirst;nextSeq;#subs;replaying]")) != [0, 0, 0, 0, 0]:
            raise ValueError("replay requires a freshly started journaled head with no subscribers")
        day, first = map(int, hdb.request('history("SELECT coalesce(max(dayno)+1,0),coalesce(max(seq)+1,0) FROM trades")'))
        head.request(f'dayNo::{day};dayFirst::{first};nextSeq::{first};recentFirst::{first};'
                     f'live("UPDATE rdb.control SET dayno={day},firstseq={first}");replaying::1')
        path = directory / f"day-{day}.wal"
        cursor = first
        if path.exists():
            for seq, cols in records(path, day):
                if seq != cursor:
                    raise ValueError("unexpected journal sequence")
                cursor += cols.shape[1]
                if head.call("publish", seq, *cols) != cursor:
                    raise ValueError("replay watermark mismatch")
        head.request("replaying::0")
        return day, cursor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    writer = actions.add_parser("serve")
    writer.add_argument("--directory", type=Path, required=True)
    writer.add_argument("--port", type=int, default=5520)
    writer.add_argument("--password-file", type=Path)
    restore = actions.add_parser("replay")
    restore.add_argument("--directory", type=Path, required=True)
    restore.add_argument("--head", default="127.0.0.1:5510")
    restore.add_argument("--hdb", default="127.0.0.1:5511")
    args = parser.parse_args()
    if args.action == "serve":
        serve(args.directory, args.port, args.password_file.read_text().strip() if args.password_file else "")
    else:
        day, cursor = replay(args.directory, args.head, args.hdb)
        print(f"restored day {day} through exclusive watermark {cursor}")


if __name__ == "__main__":
    main()
