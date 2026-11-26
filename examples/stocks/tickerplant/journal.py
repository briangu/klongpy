"""External WAL writer retained for benchmark comparisons with the in-process native writer."""
import argparse
from contextlib import closing
import hmac
import os
from pathlib import Path
import socket
import sqlite3
import struct
import zlib

import numpy as np

from klongpy.ipc import Handshake, MAX_MESSAGE, Message, Session


HEADER = struct.Struct("<4sQQII")
HEADER_CRC = struct.Struct("<I")
MAGIC = b"KGT2"
LEGACY_MAGIC = b"KGT1"
MAX_ROWS = (MAX_MESSAGE - 128) // 32


class TornTail(ValueError):
    """The writer stopped partway through the last record."""


def read_record(stream, day, expected):
    header = stream.read(HEADER.size)
    if not header:
        return None
    prefix = header[:4]
    if not any(magic.startswith(prefix) for magic in (MAGIC, LEGACY_MAGIC)):
        raise ValueError("invalid journal header")
    if len(header) != HEADER.size:
        raise TornTail("incomplete journal header")
    magic, written_day, first, n, checksum = HEADER.unpack(header)
    if magic == MAGIC:
        check = stream.read(HEADER_CRC.size)
        if len(check) != HEADER_CRC.size:
            raise TornTail("incomplete journal header checksum")
        if HEADER_CRC.unpack(check)[0] != zlib.crc32(header):
            raise ValueError("damaged journal header")
    if written_day != day or not 0 < n <= MAX_ROWS:
        raise ValueError("invalid journal header")
    if expected is not None and first != expected:
        raise ValueError(f"discontinuous journal at sequence {first}, expected {expected}")
    data = stream.read(n * 32)
    if len(data) != n * 32:
        if magic == LEGACY_MAGIC:
            raise ValueError("incomplete legacy journal record with unchecked length")
        raise TornTail("incomplete journal record")
    if zlib.crc32(data) != checksum:
        raise ValueError(f"damaged journal record at sequence {first}")
    return first, np.frombuffer(data, dtype="<i8").reshape(4, n)


def records(path, day):
    with path.open("rb") as stream:
        expected = None
        while (record := read_record(stream, day, expected)) is not None:
            first, cols = record
            expected = first + cols.shape[1]
            yield first, cols


def sync_directory(path):
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class Journal:
    def __init__(self, directory, state, initial=None):
        self.directory = directory
        self.state = state
        # Keep one writer for the directory even if another service uses a different port.
        state.execute("PRAGMA locking_mode=EXCLUSIVE")
        state.execute("PRAGMA synchronous=FULL")
        state.execute("BEGIN EXCLUSIVE")
        state.execute("CREATE TABLE IF NOT EXISTS session(id INTEGER PRIMARY KEY CHECK(id=1), "
                      "day INTEGER NOT NULL, first_seq INTEGER NOT NULL, closing_end INTEGER)")
        row = state.execute("SELECT day, first_seq, closing_end FROM session WHERE id=1").fetchone()
        if row is None:
            if initial is None and any(p.name != "day-0.wal" or p.stat().st_size
                                       for p in directory.glob("day-*.wal")):
                raise ValueError("existing WAL needs --initial-day and --first-sequence from the HDB")
            day, first = initial or (0, 0)
            if day < 0 or first < 0:
                raise ValueError("day and first sequence must be nonnegative")
            if any(int(p.stem[4:]) > day for p in directory.glob("day-*.wal")):
                raise ValueError("journal contains a day beyond the supplied HDB checkpoint")
            self.ensure_day(day)
            state.execute("INSERT INTO session VALUES(1, ?, ?, NULL)", (day, first))
            row = (day, first, None)
        state.commit()
        self.day, self.first, self.closing_end = row
        self.stream = None
        self.next = self.first
        self.ready = False

    def path(self, day):
        return self.directory / f"day-{day}.wal"

    def ensure_day(self, day):
        path = self.path(day)
        if not path.exists():
            with path.open("xb") as stream:
                os.fsync(stream.fileno())
            sync_directory(self.directory)

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        self.ready = False

    def open(self):
        self.close()
        self.stream = self.path(self.day).open("r+b", buffering=0)
        self.next = self.first
        return [self.day, self.first, int(self.closing_end is not None)]

    def read(self, cursor):
        if self.stream is None or self.ready or cursor != self.next:
            raise ValueError("unexpected replay cursor")
        offset = self.stream.tell()
        try:
            record = read_record(self.stream, self.day, self.next)
        except TornTail:
            if self.closing_end is not None:
                raise
            # Only the unsealed tail can contain an interrupted append.
            self.stream.truncate(offset)
            self.stream.seek(offset)
            os.fsync(self.stream.fileno())
            record = None
        if record is None:
            if self.closing_end is not None and self.next != self.closing_end:
                raise ValueError("journal does not reach the closing watermark")
            # A previous process may have died after write() but before fsync().
            os.fsync(self.stream.fileno())
            self.ready = True
            return []
        first, cols = record
        self.next += cols.shape[1]
        return [first, *cols]

    def append(self, day, first, columns):
        if not self.ready or self.closing_end is not None or (day, first) != (self.day, self.next):
            raise ValueError("journal is closing, awaiting replay or at a different sequence")
        values = [np.asarray(col) for col in columns]
        if len(values) != 4 or any(col.dtype.kind != "i" or col.ndim != 1 for col in values):
            return -1  # Definite input rejection, before any write.
        n = len(values[0])
        if not 0 < n <= MAX_ROWS or any(len(col) != n for col in values):
            return -1
        blocks = [memoryview(np.ascontiguousarray(col, dtype="<i8")).cast("B") for col in values]
        checksum = 0
        for block in blocks:
            checksum = zlib.crc32(block, checksum)
        header = HEADER.pack(MAGIC, day, first, n, checksum)
        blocks.insert(0, memoryview(header + HEADER_CRC.pack(zlib.crc32(header))))
        fd = self.stream.fileno()
        while blocks:
            count = os.writev(fd, blocks) if hasattr(os, "writev") else os.write(fd, blocks[0])
            if count <= 0:
                raise OSError("short journal write")
            while blocks and count >= len(blocks[0]):
                count -= len(blocks.pop(0))
            if count:
                blocks[0] = blocks[0][count:]
        getattr(os, "fdatasync", os.fsync)(fd)
        self.next += n
        return self.next

    def rollover(self, action, day, end):
        if not self.ready or (day, end) != (self.day, self.next) or end <= self.first:
            raise ValueError("unexpected rollover watermark")
        if action == "beginClose":
            self.state.execute("UPDATE session SET closing_end=?", (end,))
            self.state.commit()
            self.closing_end = end
        else:
            if self.closing_end != end:
                raise ValueError("no matching close intent")
            self.ensure_day(day + 1)
            if self.path(day + 1).stat().st_size:
                raise ValueError("next day already contains trades")
            self.state.execute("UPDATE session SET day=?, first_seq=?, closing_end=NULL", (day + 1, end))
            self.state.commit()
            self.day, self.first, self.closing_end = day + 1, end, None
            self.open()
            self.read(end)
        return end

    def request(self, value):
        name, *args = value
        action = str(name)
        if action == "open" and args == [0]:
            return self.open()
        if action == "read" and len(args) == 1:
            try:
                return self.read(int(args[0]))
            except ValueError:
                self.close()
                raise
        if action == "append" and len(args) == 6:
            return self.append(int(args[0]), int(args[1]), args[2:])
        if action in ("beginClose", "finishClose") and len(args) == 2:
            return self.rollover(action, *map(int, args))
        raise ValueError(f"unknown journal action {action!r} with {len(args)} arguments")


def serve_connection(conn, journal, password):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    session = Session.server()
    while data := conn.recv(1024 * 1024):
        for event in session.feed(data):
            if isinstance(event, Handshake):
                if password and (event.user != "bench" or not hmac.compare_digest(event.password or "", password)):
                    session.reject()
                    return
                conn.sendall(session.accept())
            elif isinstance(event, Message):
                service = session.begin_service(event)
                try:
                    reply = session.send_response(service, journal.request(event.value))
                except (TypeError, ValueError, OverflowError) as exc:
                    reply = session.send_error(service, str(exc))
                conn.sendall(reply)
                session.end_service(service)


def serve(directory, port, password="", initial=None):
    os.umask(0o077)
    directory.mkdir(parents=True, exist_ok=True)
    sync_directory(directory.parent)
    with closing(sqlite3.connect(directory / "session.sqlite3", timeout=0)) as state:
        journal = Journal(directory, state, initial)
        with socket.create_server(("127.0.0.1", port)) as server:
            while True:
                conn, _ = server.accept()
                with conn:
                    try:
                        serve_connection(conn, journal, password)
                    except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                        pass
                    finally:
                        journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["serve"])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5520)
    parser.add_argument("--password-file", type=Path)
    parser.add_argument("--initial-day", type=int, help="starting day when creating a new checkpoint")
    parser.add_argument("--first-sequence", type=int, help="first open-day sequence when creating a new checkpoint")
    args = parser.parse_args()
    if (args.initial_day is None) != (args.first_sequence is None):
        parser.error("supply the initial day and first sequence together")
    initial = None if args.initial_day is None else (args.initial_day, args.first_sequence)
    serve(args.directory, args.port, args.password_file.read_text().strip() if args.password_file else "", initial)


if __name__ == "__main__":
    main()
