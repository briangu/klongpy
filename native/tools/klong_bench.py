#!/usr/bin/env python3
"""A Klong console on the bench's nodes, from a workstation, through the supervisor's bridge.

    klong_bench.py --host HOST[:PORT] repl   --node N [--take]    an interactive session on node N
    klong_bench.py --host HOST[:PORT] eval   --node N EXPR        evaluate EXPR on node N, print its output
    klong_bench.py --host HOST[:PORT] run    --node N FILE.kg     upload FILE and run it on node N
    klong_bench.py --host HOST[:PORT] status                      the nodes: online, Klong or not, load, hello, self-test

Standard library only (Python 3.9+). The bridge is the supervisor's HTTP server (POST /klong/eval,
GET /klong/output, POST /klong/interrupt, GET /klong/nodes, POST /klong/control);
`bench_bridge_sim.py` is a local stand-in for it in front of the simulator. HOST defaults to the
environment variable KLONG_BENCH_HOST (for example `bench.local`); nothing else names a host.

Control. Like the BASIC page, one client at a time types. The first input claims control when nobody
has it, and it is given up when this program ends; it is kept alive by polling the output at least every
10 seconds (the REPL does that while you think). If somebody else has it, the bridge says
"<name> has control" (exit status 2); `--take` asks them for it (they have 15 seconds to answer).

Display. The bridge's transcript already has each typed line as `klong> LINE`, and your terminal shows the
line you typed, so the echo of this client's own input is left out of what is printed; nobody else's is.
Ctrl-C while something runs sends an interrupt to the node (a second one stops waiting). Ctrl-D or `:q`
leaves the REPL. A node's console is the one that last received input: the output of a node that
somebody types at from its USB port, or from another node, does not come here.

`run` sends the file with `raw=1` (the bridge adds no newline and echoes nothing) in datagrams of at most 1024
bytes cut anywhere, so any text file goes, whatever its lines. Still impossible: a program the node cannot hold
(`max_run` of its console, 64 KB, and its free heap), a file that is not valid UTF-8 (the node refuses the program),
and binary data in the answers (the bridge's JSON carries text: NUL becomes a blank, invalid UTF-8 a U+FFFD).

Exit status of `eval` and `run`: 0 ok, 1 an error on the node, 130 interrupted, 2 not delivered or the bridge
says no (409 somebody has control, 503 the node is offline or not a Klong node), 3 other.
"""
import argparse
import http.client
import json
import os
import secrets
import sys
import threading
import time
import zlib
from urllib.parse import quote

CHUNK = 1024  # bytes of console input per datagram
POLL = 0.1
KEEPALIVE = 3.0  # (the bridge forgets a client after 10 s without a poll)


class BridgeError(Exception):
    def __init__(self, code, text):
        super().__init__(text)
        self.code = code
        self.text = text


class Bridge:
    def __init__(self, host, name="klong-bench cli", ident=None, timeout=10):
        # a base URL as the bridge's own pages give it is taken as the host it names (http only: the bridge has no TLS)
        if host.startswith("http://"):
            host = host[len("http://"):]
        host = host.rstrip("/")
        if ":" in host:
            h, p = host.rsplit(":", 1)
            self.host, self.port = h, int(p)
        else:
            self.host, self.port = host, 80
        self.id = ident or secrets.token_hex(4)  # 1..16 letters and digits
        self.name = name
        self.timeout = timeout

    def request(self, method, path, body=None):
        sep = "&" if "?" in path else "?"
        path = "%s%sid=%s&name=%s" % (path, sep, self.id, quote(self.name))
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            # (the body is the bridge's "plain" argument only with this content type)
            conn.request(method, path, body=body, headers={"Content-Type": "text/plain"} if body is not None else {})
            r = conn.getresponse()
            raw = r.read()
        except OSError as e:
            raise BridgeError(0, "cannot reach the bridge at %s:%d: %s" % (self.host, self.port, e))
        finally:
            conn.close()
        try:
            obj = json.loads(raw.decode("utf-8", "replace")) if raw else {}
        except ValueError:
            obj = {"error": raw[:80].decode("utf-8", "replace")}
        if r.status >= 300:
            raise BridgeError(r.status, obj.get("error", "bridge error %d" % r.status))
        return obj

    def eval(self, node, data, raw=False):
        """Queue input; the id of the message. 429 (the queue holds two) is waited out for a few seconds.
        `raw`: the body goes as it is, with no newline added and no echo in the transcript."""
        end = time.time() + 5
        while True:
            try:
                return self.request("POST", "/klong/eval?node=%d%s" % (node, "&raw=1" if raw else ""), data)["id"]
            except BridgeError as e:
                if e.code == 429 and time.time() < end:
                    time.sleep(0.05)
                    continue
                raise

    def interrupt(self, node):
        end = time.time() + 5
        while True:
            try:
                return self.request("POST", "/klong/interrupt?node=%d" % node)
            except BridgeError as e:
                if e.code == 429 and time.time() < end:
                    time.sleep(0.05)
                    continue
                raise

    def output(self, node, frm):
        return self.request("GET", "/klong/output?node=%d&from=%d" % (node, frm))

    def nodes(self):
        return self.request("GET", "/klong/nodes")["nodes"]

    def control(self, action):
        return self.request("POST", "/klong/control?action=%s" % action)


class Transcript:
    """What to show of the bridge's transcript: it has the echo of every line typed (`klong> LINE`), and
    the echo of this client's own lines is left out, wherever it falls in the stream."""

    def __init__(self):
        self.pending = []  # echoes still to come
        self.hold = ""

    def expect(self, data):
        text = data.decode("utf-8", "replace")
        if not text.endswith("\n"):
            text += "\n"
        self.pending.append("".join("klong> " + l + "\n" for l in text.split("\n")[:-1]))

    def feed(self, data, final=False):
        text = self.hold + data
        self.hold = ""
        for e in list(self.pending):
            i = text.find(e)
            if i >= 0:
                text = text[:i] + text[i + len(e):]
                self.pending.remove(e)
        if not final and self.pending:
            # an echo may be split between two responses: keep back what could be its start
            e = self.pending[0]
            for k in range(min(len(e) - 1, len(text)), 0, -1):
                if e.startswith(text[-k:]):
                    self.hold = text[-k:]
                    text = text[:-k]
                    break
        return text


class Session:
    def __init__(self, bridge, node, out=None):
        self.b, self.node = bridge, node
        self.out = out or sys.stdout
        self.lock = threading.RLock()
        # start where the transcript is now (a `from` beyond its end gives its end, and nothing)
        self.pos = bridge.output(node, 2**32 - 1)["next"]
        self.tr = Transcript()
        self.last = None  # the last response
        self.all = []  # what was printed since the last `mark`
        self.keep = None

    def poll(self, show=True):
        """Fetch what is new (all of it, 4 KB at a time); print it. The last response."""
        with self.lock:
            while True:
                o = self.b.output(self.node, self.pos)
                self.last = o
                if o.get("lost"):
                    self.out.write("\n[some output was lost: the bridge keeps 8 KB]\n")
                self.pos = o["next"]
                if o["data"]:
                    text = self.tr.feed(o["data"])
                    self.all.append(text)
                    if show and text:
                        self.out.write(text)
                        self.out.flush()
                if len(o["data"].encode("utf-8", "replace")) < 4000:
                    return o

    def start_keepalive(self):
        def run():
            while not self.keep.wait(KEEPALIVE):
                try:
                    self.poll()
                except BridgeError:
                    pass

        self.keep = threading.Event()
        threading.Thread(target=run, daemon=True).start()

    def stop_keepalive(self):
        if self.keep:
            self.keep.set()

    def take_control(self):
        """`--take`: ask whoever types now, and wait for the answer (or the 15 s after which it is given)."""
        self.b.control("take")
        end = time.time() + 20
        shown = False
        while time.time() < end:
            o = self.poll(show=False)
            you = o.get("control", {}).get("you")
            if you == "driving":
                return
            if not shown:
                who = o.get("control", {}).get("driver", "the driver")
                self.out.write("[asking %s for control]\n" % who)
                shown = True
            time.sleep(0.3)
        raise BridgeError(409, "control was not given")

    @staticmethod
    def done_of(o, my_id):
        """Our input's status from `dones` (the newest 8, newest first), else from `done`."""
        for d in o.get("dones") or []:
            if d["id"] == my_id:
                return d["status"]
        d = o.get("done")
        return d["status"] if d and d["id"] == my_id else None

    def send_and_wait(self, data, show=True, raw=False):
        """Send one input; print its output as it arrives; its status (ok, error, interrupted), `not delivered`
        or `abandoned`."""
        if not raw:
            with self.lock:
                self.tr.expect(data)
        my_id = self.b.eval(self.node, data, raw=raw)
        interrupted = False
        while True:
            try:
                o = self.poll(show=show)
                st = self.done_of(o, my_id)
                if st is not None:
                    # the transcript is complete up to its `done`: one more look for what came with it
                    self.poll(show=show)
                    return st
                sent = o.get("sent") or {}
                if sent.get("id") == my_id and sent.get("state") in ("failed", "refused"):
                    return "not delivered: " + ("the node's console is busy for another sender" if sent["state"] == "refused" else "no acknowledgement from the node")
                time.sleep(POLL)
            except KeyboardInterrupt:
                if interrupted:
                    self.out.write("\n[not waiting any more]\n")
                    return "abandoned"
                interrupted = True
                try:
                    self.b.interrupt(self.node)
                    self.out.write("\n[interrupt sent]\n")
                except BridgeError as e:
                    self.out.write("\n[%s]\n" % e.text)

    def text(self):
        return "".join(self.all)


def finish(sess):
    sess.stop_keepalive()
    try:
        sess.b.control("release")
    except BridgeError:
        pass


def status_code(st):
    if st.startswith("not delivered"):
        print("klong_bench: " + st, file=sys.stderr)
        return 2
    return {"ok": 0, "error": 1, "interrupted": 130}.get(st, 3)


def cmd_eval(b, a):
    s = Session(b, a.node)
    try:
        if a.take:
            s.take_control()
        return status_code(s.send_and_wait(a.expr.encode()))
    finally:
        finish(s)


def cmd_run(b, a):
    prog = open(a.file, "rb").read()
    if not prog:
        print("klong_bench: empty file", file=sys.stderr)
        return 2
    chunks = [prog[i:i + CHUNK] for i in range(0, len(prog), CHUNK)]
    s = Session(b, a.node)
    try:
        if a.take:
            s.take_control()
        header = ("\x01run %d %08x\n" % (len(prog), zlib.crc32(prog) & 0xFFFFFFFF)).encode()
        st = s.send_and_wait(header, show=False, raw=True)
        if st != "ok" or "\x01go" not in s.text():
            print("klong_bench: the node did not accept the upload: %s" % (s.text().replace("\x01", "").strip() or st), file=sys.stderr)
            return 1
        s.all.clear()
        for ch in chunks:
            st = s.send_and_wait(ch, show=False, raw=True)
            if st not in ("ok", "error"):
                return status_code(st)
        text = s.text()
        sys.stdout.write("".join(l + "\n" for l in text.split("\n")[:-1] if not l.startswith("\x01")))
        err = [l for l in text.split("\n") if l.startswith("\x01err")]
        if err:
            print(err[0][1:], file=sys.stderr)
            return 1
        return 0
    finally:
        finish(s)


def cmd_repl(b, a):
    s = Session(b, a.node)
    try:
        for n in b.nodes():
            if n["node"] == a.node and n.get("info"):
                print(n["info"])
        if a.take:
            s.take_control()
        s.start_keepalive()
        while True:
            try:
                line = input("klong[%d]> " % a.node)
            except EOFError:
                print()
                return 0
            except KeyboardInterrupt:
                print()
                continue
            if line.strip() == ":q":
                return 0
            if not line.strip():
                continue
            try:
                st = s.send_and_wait((line + "\n").encode("utf-8"))
                if st.startswith("not delivered"):
                    print("[%s]" % st, file=sys.stderr)
            except BridgeError as e:
                print("[%s]" % (e.text if e.code != 409 else e.text + "; start again with --take to ask for it"), file=sys.stderr)
    finally:
        finish(s)


FLAGS = [(0x01, "alive"), (0x02, "app-image"), (0x04, "app-running"), (0x10, "klong"), (0x20, "selftest-serving"), (0x40, "selftest-pass"), (0x80, "selftest-fail")]


def decode_flags(f):
    """The STATUS flags byte: 0x01 alive, 0x02 an app image, 0x04 an app running (the cube OS), 0x10 not a cube
    node (Klong), 0x20 self-test serving or running, 0x40 passed, 0x80 failed."""
    if f is None:
        return "?"
    return ",".join(n for bit, n in FLAGS if f & bit) or "-"


def cmd_status(b, a):
    print("%-4s %-8s %-5s %-5s %-6s %-26s %s" % ("node", "state", "klong", "load", "flags", "decoded", "what it says"))
    for n in b.nodes():
        state = "online" if n["online"] else "offline"
        says = n.get("selftest") or n.get("info") or ""
        f = n.get("flags")
        print("%-4d %-8s %-5s %-5d %-6s %-26s %s" % (n["node"], state, "yes" if n["klong"] else "no", n["load"], "0x%02x" % f if f is not None and n["online"] else "-", decode_flags(f) if n["online"] else "-", says))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("KLONG_BENCH_HOST"), help="the bridge, HOST or HOST:PORT")
    ap.add_argument("--name", default="klong-bench cli", help="the name the bridge shows for this client")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("repl", "eval", "run"):
        p = sub.add_parser(name)
        p.add_argument("--node", type=int, required=True)
        p.add_argument("--take", action="store_true", help="ask whoever has control for it")
        if name == "eval":
            p.add_argument("expr")
        if name == "run":
            p.add_argument("file")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    if not a.host:
        ap.error("--host (or KLONG_BENCH_HOST) is required")
    b = Bridge(a.host, name=a.name)
    try:
        return {"repl": cmd_repl, "eval": cmd_eval, "run": cmd_run, "status": cmd_status}[a.cmd](b, a)
    except BridgeError as e:
        msg = e.text
        if e.code == 409:
            msg += " (use --take to ask for it)"
        print("klong_bench: %s" % msg, file=sys.stderr)
        return 2 if e.code in (0, 409, 429, 503, 413) else 3


if __name__ == "__main__":
    sys.exit(main())
