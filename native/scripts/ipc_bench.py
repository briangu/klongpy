#!/usr/bin/env python3
"""Ticket 106 measurements: IPC throughput over localhost TCP and memory per connection.

    .venv/bin/python native/scripts/ipc_bench.py [--binary native/target/release/klong] [--reps 30]

Throughput: a client sends a vector of 125000 64-bit integers (1 MB on the wire) to `idf::{x}` and gets it
back, `reps` times; the figure is megabytes of payload per second counting both directions. Pairs: native
to native, KlongPy to KlongPy, and the two mixed ones.

Memory: the native server's heap in use (`.mem()`) before and after 50 clients connect and stay idle, and
the same for a native client that opens 50 connections.
"""
import argparse
import os
import socket
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import test_native_ipc as T  # noqa: E402
from ipc_helpers import free_port  # noqa: E402
from klongpy.repl import cleanup_repl, create_repl  # noqa: E402

N = 125_000


def native_server():
    n = T.Native()
    n.run("idf::{x}")
    port = free_port()
    assert n.value(f'.srv("127.0.0.1:{port}")') == "1"
    return n, port


def klongpy_server():
    k, loops = create_repl()
    k("idf::{x}")
    port = free_port()
    k(f".srv({port})")
    return (k, loops), port


def native_client(port, reps):
    c = T.Native()
    try:
        c.run(f"c::.cli({port});.clit(c;0);big::!{N};c(:idf,,big)")
        secs = float(c.value(f"t::.pc();{{c(:idf,,big)}}'!{reps};.pc()-t", 600))
        assert c.value("(c(:idf,,big)~big)") == "1"
        return secs
    finally:
        c.stop()


def klongpy_client(port, reps):
    k, loops = create_repl()
    try:
        k(f"c::.cli({port})")
        k["big"] = np.arange(N)
        k("c(:idf,,big)")
        t = time.perf_counter()
        for _ in range(reps):
            r = k("c(:idf,,big)")
        secs = time.perf_counter() - t
        assert np.array_equal(r, np.arange(N))
        return secs
    finally:
        k[".system"]["closeEvent"].trigger()
        cleanup_repl(loops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=T.BINARY)
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    T.BINARY = a.binary
    mb = 2 * a.reps * N * 8 / 1e6
    print(f"1 MB vector ({N} 64-bit integers) echoed {a.reps} times over localhost TCP")
    for sname, make_server in (("native", native_server), ("KlongPy", klongpy_server)):
        server, port = make_server()
        try:
            for cname, client in (("native", native_client), ("KlongPy", klongpy_client)):
                secs = client(port, a.reps)
                print(f"  {cname:8} client -> {sname:8} server: {secs / a.reps * 1000:7.2f} ms per round trip, {mb / secs:7.0f} MB/s")
        finally:
            if sname == "native":
                server.stop()
            else:
                server[0][".system"]["closeEvent"].trigger()
                cleanup_repl(server[1])

    # memory per connection
    def used(node):
        return int(node.value(".mem()").strip("[]").split()[0])

    server, port = native_server()
    try:
        base = used(server)
        socks = []
        for _ in range(50):
            s = socket.create_connection(("127.0.0.1", port))
            s.sendall(b"\x03\x00")
            assert s.recv(1) == b"\x03"
            socks.append(s)
        time.sleep(0.2)
        idle = used(server)
        print(f"native server: {(idle - base) / 50:.0f} bytes of heap per idle connection (50 connections)")
        for s in socks:
            # one small request each, so that the buffers have been used
            s.sendall(bytes.fromhex("010100000d000000") + bytes.fromhex("f905000000")[:5])
        time.sleep(0.2)
        for s in socks:
            s.recv(64)
        print(f"native server: {(used(server) - base) / 50:.0f} bytes per connection after one request each")
        client = T.Native()
        try:
            client.run("cs::[]")
            b0 = used(client)
            client.run(f"cs::{{.cli({port})}}'!50")
            print(f"native client: {(used(client) - b0) / 50:.0f} bytes of heap per open connection (handle included)")
            client.run("{x(\"1+1\")}'cs")
            print(f"native client: {(used(client) - b0) / 50:.0f} bytes per connection after one call each")
        finally:
            client.stop()
        for s in socks:
            s.close()
        time.sleep(0.3)
        print(f"native server: {used(server) - base} bytes left after all 100 connections closed")
    finally:
        server.stop()
    os._exit(0)


if __name__ == "__main__":
    main()
