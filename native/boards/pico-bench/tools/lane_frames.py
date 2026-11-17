#!/usr/bin/env python3
"""Frames and datagrams of the Pico bench's switch, for a supervisor-side test tool.

The 16-byte frame, the START/DATA/END/ACK datagrams of sub-types 0xD0..0xD3 and a parser with
the 16-byte sliding window, as the nodes do them (native/klong-dgram, the same rules). No
device is opened here: the tool prints hex that another program puts on a lane, and decodes
hex that a lane gave.

    lane_frames.py send DEST TEXT [--src 16] [--chan 1] [--id N] [--ack]   hex of the frames of one message
    lane_frames.py decode HEXFILE|-                                       the frames and messages in a hex dump

TEXT understands \\r \\n \\x04 escapes. Example: ``send 0 '1+2\\r'`` is what a supervisor sends
to type ``1+2`` on node 0's console (channel 1).
"""
import argparse
import sys
import zlib

SYNC = b"\xa5\x5a\x01"
APP_DATA, PING, PONG, STATUS = 9, 1, 2, 3
START, DATA, END, ACK = 0xD0, 0xD1, 0xD2, 0xD3
SUPERVISOR = 16


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: init 0xFFFF, poly 0x1021, no reflection, no xorout."""
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def frame(ftype: int, src: int, dst: int, seq: int, sub: int, payload: bytes = b"") -> bytes:
    if len(payload) > 4:
        raise ValueError("a frame carries at most 4 payload bytes")
    body = bytes([1, ftype, src, dst, seq & 0xFF, (seq >> 8) & 0xFF, sub, len(payload)]) + payload.ljust(4, b"\0")
    return b"\xa5\x5a" + body + crc16(body).to_bytes(2, "little")


def message_frames(src: int, dst: int, mid: int, chan: int, data: bytes, ack: bool = False) -> list:
    """START, DATA 0.., END of one message of 1 to 1024 bytes."""
    if not 1 <= len(data) <= 1024:
        raise ValueError("a message is 1 to 1024 bytes")
    out = [frame(APP_DATA, src, dst, mid, START, len(data).to_bytes(2, "little") + bytes([1 if ack else 0, chan]))]
    for i in range(0, len(data), 4):
        out.append(frame(APP_DATA, src, dst, i // 4, DATA, data[i:i + 4]))
    out.append(frame(APP_DATA, src, dst, mid, END, crc32(data).to_bytes(4, "little")))
    return out


def decode(b: bytes):
    """A 16-byte window as a dict, or None when it is not a valid frame."""
    if len(b) != 16 or b[:3] != SYNC or crc16(b[2:14]) != int.from_bytes(b[14:16], "little"):
        return None
    n = b[9]
    if n > 4 or any(b[10 + n:14]):
        return None
    return {"type": b[3], "src": b[4], "dst": b[5], "seq": b[6] | b[7] << 8, "sub": b[8], "data": bytes(b[10:10 + n])}


def parse(stream: bytes):
    """Every valid frame in a byte stream, found with a 16-byte sliding window."""
    frames, i = [], 0
    while i + 16 <= len(stream):
        f = decode(stream[i:i + 16])
        if f:
            frames.append(f)
            i += 16
        else:
            i += 1
    return frames


class Reassembler:
    """Messages (source, channel, bytes) from datagram frames, one state per source."""

    def __init__(self):
        self.state, self.last = {}, {}

    def feed(self, f):
        if f["type"] != APP_DATA:
            return None
        s, sub = f["src"], f["sub"]
        if sub == START and len(f["data"]) == 4:
            n, flags, chan = int.from_bytes(f["data"][:2], "little"), f["data"][2], f["data"][3]
            self.state[s] = {"id": f["seq"], "n": n, "flags": flags, "chan": chan, "buf": bytearray(), "next": 0,
                             "bad": not 1 <= n <= 1024}
        elif sub == DATA and s in self.state:
            st = self.state[s]
            if f["seq"] != st["next"] or len(st["buf"]) + len(f["data"]) > st["n"]:
                st["bad"] = True
            else:
                st["buf"] += f["data"]
                st["next"] += 1
        elif sub == END and s in self.state and self.state[s]["id"] == f["seq"]:
            st = self.state.pop(s)
            ok = (not st["bad"] and len(st["buf"]) == st["n"] and crc32(bytes(st["buf"])) == int.from_bytes(f["data"], "little"))
            if ok and self.last.get(s) != st["id"]:
                self.last[s] = st["id"]
                return (s, st["chan"], bytes(st["buf"]))
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("send")
    s.add_argument("dest", type=int)
    s.add_argument("text")
    s.add_argument("--src", type=int, default=SUPERVISOR)
    s.add_argument("--chan", type=int, default=1)
    s.add_argument("--id", type=int, default=1)
    s.add_argument("--ack", action="store_true")
    d = sub.add_parser("decode")
    d.add_argument("file")
    a = ap.parse_args(argv)
    if a.cmd == "send":
        data = a.text.encode().decode("unicode_escape").encode("latin-1")
        for f in message_frames(a.src, a.dest, a.id, a.chan, data, a.ack):
            print(f.hex())
        return 0
    text = sys.stdin.read() if a.file == "-" else open(a.file).read()
    raw = bytes.fromhex("".join(text.split()))
    r = Reassembler()
    for f in parse(raw):
        m = r.feed(f)
        print(f"frame type {f['type']} {f['src']}->{f['dst']} seq {f['seq']} sub {f['sub']:#x} {f['data'].hex()}")
        if m:
            print(f"message from {m[0]} channel {m[1]}: {m[2]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
