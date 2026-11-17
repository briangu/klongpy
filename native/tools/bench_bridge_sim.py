"""A local stand-in for the bench supervisor's Klong console bridge (the one in the esp32_cube
bench firmware: `klong_bridge.h` and the /klong routes of `bench_main.cpp`).

    bench_bridge_sim.py [--port 8080] [--nodes 2] [--baud 115200] [--selftest] [--mute K]
                        [--speed 1] [--bench-bin native/target/release/klong-bench]

It serves the same HTTP routes, answers and errors, in front of the simulator's model of the bench
(N Klong nodes behind the 16-byte-frame switch, in virtual time that follows real time), so that
`klong_bench.py` can be developed and tested without hardware and the page's author has an
executable reference. Where the real bridge was read, this follows it; where it was not, it says so.

Every call carries `id=<1..16 letters and digits>` and `name=<display name>`. Routes (JSON):

    POST /klong/eval?node=N&id&name        body: lines (a newline is added if missing), at most 1024 bytes with it
                                           -> {"id": ID}; 409 {"error": "<name> has control"}; 503 {"error": "that node is offline"
                                           | "that node is not running Klong"}; 413; 429 {"error": "still sending the last one"}
                                           (the queue holds two per node: an evaluation and an interrupt); 400
    GET  /klong/output?node=N&from=X&id&name
                                           -> {"node","next","data","done":{"id","status"}|null,"load","online","klong",
                                              "sent":{"id","state":none|sending|delivered|failed|refused},"control":{...}},
                                              plus "lost": true when X left the window. The window is the newest 8 KB per
                                              node and a response has at most 4 KB, never cut inside a UTF-8 character.
                                              The transcript has each typed line echoed as `klong> LINE`.
                                              Polling with an id keeps that id's control alive (lost after 10 s without a poll).
    POST /klong/interrupt?node=N&id&name   -> {"ok": true}; the same errors as eval
    GET  /klong/nodes                      -> {"nodes": [{"node","online","klong","load","info","selftest"}]}, 4 nodes
    POST /klong/control?action=claim|take|give|keep|release&id&name
                                           -> {"ok": true}, or 409/400 {"error": "..."}

Control is the BASIC page's: the first to type (or `claim`) when nobody drives becomes the driver;
another asks with `take`, and the driver gives or keeps (`give`/`keep`), no answer within 15 s gives it;
a driver not heard from for 10 s is gone; a request that was kept may be repeated after 15 s.

The first line printed is `listening on http://HOST:PORT`, the second `simulator pid N`. The simulator
process is started and stopped by this program (by its own PID, nothing else); SIGINT or SIGTERM
stops both. No token (the real bridge has none).
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BIN = os.path.join(os.path.dirname(HERE), "target", "release", "klong-bench")


class Sim:
    """The klong-bench process and its line protocol (one request, one answer)."""

    def __init__(self, binary, nodes, baud, selftest, mute, speed):
        args = [binary, str(nodes), "--bridge", "--baud", str(baud)]
        if selftest:
            args.append("--selftest")
        for k in mute:
            args += ["--mute", str(k)]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.pid = self.proc.pid
        self.lock = threading.Lock()
        self.speed = speed
        self.stop = threading.Event()
        first = self.proc.stdout.readline().strip()
        if not first.startswith("ready"):
            raise RuntimeError("the simulator did not start: %r" % first)
        self.advancer = threading.Thread(target=self._advance, daemon=True)
        self.advancer.start()

    def cmd(self, line):
        with self.lock:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
            return self.proc.stdout.readline().rstrip("\n")

    def send(self, line):
        """A request that has no answer (time moves; the board may be busy for long)."""
        with self.lock:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()

    def cmd_multi(self, line):
        """A request whose answer is several lines, the last of which is `end`."""
        with self.lock:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
            out = []
            while True:
                l = self.proc.stdout.readline().rstrip("\n")
                if l == "end" or l == "":
                    return out
                out.append(l)

    def _advance(self):
        step = 0.02
        while not self.stop.wait(step):
            try:
                self.send("advance %d" % max(1, int(step * 1000 * self.speed)))
            except (BrokenPipeError, ValueError, OSError):
                return

    def close(self):
        """Stop the simulator we started: ask, then terminate or kill that PID if it lingers."""
        self.stop.set()
        try:
            with self.lock:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


def unhex(s):
    return bytes.fromhex(s) if s else b""


GONE_MS, ANSWER_MS, RETRY_MS = 10000, 15000, 15000
OK, BUSY, TOO_SOON, NOT_DRIVER, BAD = range(5)
ERRORS = ["", "someone else is asking", "wait before asking again", "you do not have control", "bad request"]


def valid_id(i):
    return bool(i) and len(i) <= 16 and re.fullmatch(r"[0-9a-zA-Z]+", i) is not None


def clean_name(n):
    out = ""
    for c in (n or "")[:64]:
        if c == " " and (not out or out[-1] == " "):
            continue
        if c.isascii() and (c.isalnum() or c in " -_.!"):
            out += c
        if len(out) >= 16:
            break
    out = out.rstrip(" ")
    return out or "Someone"


class Control:
    """Who types: the BASIC page's rules (basic_control.h of the real bridge)."""

    def __init__(self):
        self.driver = self.requester = self.denied = None  # (id, name)
        self.driver_seen = self.request_at = self.requester_seen = self.denied_at = 0
        self.lock = threading.Lock()

    @staticmethod
    def now():
        return int(time.monotonic() * 1000)

    def seen(self, id_, name):
        now = self.now()
        if self.driver and self.driver[0] == id_:
            self.driver_seen = now
            if name:
                self.driver = (id_, clean_name(name))
        if self.requester and self.requester[0] == id_:
            self.requester_seen = now
            if name:
                self.requester = (id_, clean_name(name))

    def _hand_over(self, now):
        if self.requester and now - self.requester_seen <= GONE_MS:
            self.driver, self.driver_seen = self.requester, self.requester_seen
        else:
            self.driver = None
        self.requester = None

    def tick(self):
        now = self.now()
        if self.requester and now - self.requester_seen > GONE_MS:
            self.requester = None
        if self.driver and now - self.driver_seen > GONE_MS:
            self._hand_over(now)
        if self.requester and now - self.request_at > ANSWER_MS:
            self._hand_over(now)

    def take(self, id_, name):
        if not valid_id(id_):
            return BAD
        self.tick()
        now = self.now()
        if not self.driver:
            self.driver, self.driver_seen = (id_, clean_name(name)), now
            if self.requester and self.requester[0] == id_:
                self.requester = None
            return OK
        if self.driver[0] == id_:
            return OK
        if self.requester and self.requester[0] != id_:
            return BUSY
        if self.denied and self.denied[0] == id_ and now - self.denied_at < RETRY_MS:
            return TOO_SOON
        if not (self.requester and self.requester[0] == id_):
            self.requester, self.request_at = (id_, clean_name(name)), now
        self.requester_seen = now
        return OK

    def answer(self, id_, give):
        if not (self.driver and self.driver[0] == id_):
            return NOT_DRIVER
        if not self.requester:
            return OK
        if give:
            self._hand_over(self.now())
        else:
            self.denied, self.denied_at, self.requester = self.requester, self.now(), None
        return OK

    def release(self, id_):
        if self.requester and self.requester[0] == id_:
            self.requester = None
            return OK
        if not (self.driver and self.driver[0] == id_):
            return NOT_DRIVER
        self._hand_over(self.now())
        return OK

    def may_type(self, id_, name):
        self.tick()
        if self.driver and self.driver[0] == id_:
            return True
        return not self.driver and self.take(id_, name) == OK

    def json(self, id_):
        now = self.now()
        you = "driving" if self.driver and self.driver[0] == id_ else "asking" if self.requester and self.requester[0] == id_ else "watching" if self.driver else "free"
        c = {"you": you}
        if self.driver:
            c["driver"] = self.driver[1]
        if self.requester:
            c["asker"] = self.requester[1]
            c["left"] = max(0, (ANSWER_MS - (now - self.request_at) + 999) // 1000)
        retry = 0
        if self.denied and self.denied[0] == id_:
            retry = max(0, (RETRY_MS - (now - self.denied_at) + 999) // 1000) if now - self.denied_at < RETRY_MS else 0
        c["retry"] = retry
        return c


def make_handler(sim, control):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def send_json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def params(self, q):
            return q.get("id", [""])[0], q.get("name", [""])[0]

        def node(self, q):
            try:
                k = int(q.get("node", [""])[0])
            except ValueError:
                k = -1
            if not 0 <= k < 4:
                self.send_json(400, {"error": "node=0..3"})
                return None
            return k

        def sim_error(self, r):
            # "err CODE Name"
            parts = r.split(" ", 2)
            code = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 500
            name = parts[2] if len(parts) > 2 else ""
            msg = {"Offline": "that node is offline", "NotKlong": "that node is not running Klong", "TooLong": "at most 1024 bytes at a time",
                   "Full": "still sending the last one", "Empty": "send the lines as the request body", "BadNode": "node=0..3"}.get(name, name)
            self.send_json(code, {"error": msg})

        def may_type(self, q, k):
            id_, name = self.params(q)
            with control.lock:
                ok = control.may_type(id_, name)
                who = control.driver[1] if control.driver else ""
            if not ok:
                self.send_json(409, {"error": who + " has control"})
            return ok

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/klong/output":
                k = self.node(q)
                if k is None:
                    return
                id_, name = self.params(q)
                with control.lock:
                    control.seen(id_, name)
                    control.tick()
                frm = q.get("from", [None])[0]
                try:
                    frm = str(min(int(frm), 2**32 - 1)) if frm is not None else "-"
                except ValueError:
                    frm = "0"
                r = sim.cmd("output %d %s" % (k, frm))
                if not r.startswith("ok "):
                    return self.sim_error(r)
                kv = dict(t.split("=", 1) for t in r[3:].split(" "))
                sid, state = kv["sent"].split(":", 1)
                obj = {"node": k, "next": int(kv["next"]), "data": unhex(kv["data"]).decode("utf-8", "replace").replace("\0", " ")}
                if kv["lost"] == "1":
                    obj["lost"] = True
                if kv["done"] == "-":
                    obj["done"] = None
                else:
                    i, st = kv["done"].split(":", 1)
                    obj["done"] = {"id": int(i), "status": st}
                obj.update({"load": int(kv["load"]), "online": kv["online"] == "1", "klong": kv["klong"] == "1", "sent": {"id": int(sid), "state": state}})
                with control.lock:
                    obj["control"] = control.json(id_)
                return self.send_json(200, obj)
            if u.path == "/klong/nodes":
                out = []
                for l in sim.cmd_multi("nodes"):
                    w = l.split(" ")
                    kv = dict(t.split("=", 1) for t in w[2:])
                    out.append({"node": int(w[1]), "online": kv["online"] == "1", "klong": kv["klong"] == "1", "load": int(kv["load"]),
                                "info": unhex(kv["info"]).decode("utf-8", "replace"), "selftest": unhex(kv["selftest"]).decode("utf-8", "replace")})
                return self.send_json(200, {"nodes": out})
            self.send_json(404, {"error": "no such route"})

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            id_, name = self.params(q)
            if u.path == "/klong/control":
                action = q.get("action", [""])[0]
                with control.lock:
                    r = BAD
                    if action == "claim":
                        r = BUSY if control.driver and control.driver[0] != id_ else control.take(id_, name)
                        if r == BUSY:
                            return self.send_json(409, {"error": control.driver[1] + " has control"})
                    elif action == "take":
                        r = control.take(id_, name)
                    elif action in ("give", "keep"):
                        r = control.answer(id_, action == "give")
                    elif action == "release":
                        r = control.release(id_)
                if r == OK:
                    return self.send_json(200, {"ok": True})
                return self.send_json(400 if r == BAD else 409, {"error": ERRORS[r]})
            if u.path in ("/klong/eval", "/klong/interrupt"):
                k = self.node(q)
                if k is None:
                    return
                # the node must be online and running Klong, then the caller must have control
                r = sim.cmd("check %d" % k)
                if r != "ok":
                    return self.sim_error(r)
                if not self.may_type(q, k):
                    return
                if u.path == "/klong/interrupt":
                    r = sim.cmd("interrupt %d" % k)
                    return self.send_json(200, {"ok": True}) if r.startswith("ok") else self.sim_error(r)
                if not body:
                    return self.send_json(400, {"error": "send the lines as the request body"})
                r = sim.cmd("eval %d %s" % (k, body.hex()))
                if not r.startswith("ok "):
                    return self.sim_error(r)
                return self.send_json(200, {"id": int(r.split()[1])})
            self.send_json(404, {"error": "no such route"})

    return H


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080, help="0 for a free port")
    ap.add_argument("--nodes", type=int, default=2)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--selftest", action="store_true", help="the boot self-test runs on every node")
    ap.add_argument("--mute", type=int, action="append", default=[], help="node K runs other firmware (offline for the bridge)")
    ap.add_argument("--speed", type=float, default=1.0, help="virtual seconds per real second")
    ap.add_argument("--bench-bin", default=DEFAULT_BIN)
    a = ap.parse_args(argv)
    if not os.path.isfile(a.bench_bin):
        print("no %s: build it with `cd native && cargo build --release`" % a.bench_bin, file=sys.stderr)
        return 2
    sim = Sim(a.bench_bin, a.nodes, a.baud, a.selftest, a.mute, a.speed)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(sim, Control()))
    srv.daemon_threads = True

    def stop(*_):
        threading.Thread(target=srv.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print("listening on http://%s:%d" % (a.host, srv.server_address[1]), flush=True)
    print("simulator pid %d" % sim.pid, flush=True)
    try:
        srv.serve_forever()
    finally:
        sim.close()
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
