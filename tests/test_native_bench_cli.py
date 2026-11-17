"""The workstation CLI of the Pico bench (native/tools/klong_bench.py) end to end, through a local stand-in
for the supervisor's bridge (native/tools/bench_bridge_sim.py, in front of the simulator's model of the
bench). No hardware. Skipped unless `cd native && cargo build --release` has built klong-bench.

Every process started here is stopped by its own PID."""
import http.server
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "native", "tools")
BENCH = os.path.join(ROOT, "native", "target", "release", "klong-bench")
BRIDGE = os.path.join(TOOLS, "bench_bridge_sim.py")
CLI = os.path.join(TOOLS, "klong_bench.py")
NEED = unittest.skipUnless(os.access(BENCH, os.X_OK), "klong-bench not built (cd native && cargo build --release)")


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("KLONG_BENCH")}
    env.update(extra)
    return env


class Bridge:
    """A bridge simulator process, started and stopped here."""

    def __init__(self, *args):
        self.proc = subprocess.Popen([sys.executable, "-u", BRIDGE, "--port", "0", "--speed", "4", *args], stdout=subprocess.PIPE, text=True, env=clean_env())
        first = self.proc.stdout.readline().strip()
        assert first.startswith("listening on http://"), first
        self.host = first.split("http://")[1]
        self.sim_pid = int(self.proc.stdout.readline().split()[-1])

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc.stdout.close()

    def call(self, method, path, body=None, who="tester1"):
        sep = "&" if "?" in path else "?"
        req = urllib.request.Request("http://%s%s%sid=%s&name=Tester" % (self.host, path, sep, who), data=body, method=method, headers={"Content-Type": "text/plain"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def get(self, path, who="tester1"):
        return self.call("GET", path, None, who)

    def post(self, path, body=b"", who="tester1"):
        return self.call("POST", path, body, who)

    def cli(self, *args, stdin=None, timeout=60, env=None):
        return subprocess.run([sys.executable, CLI, "--host", self.host, *args], input=stdin, capture_output=True, text=True, timeout=timeout, env=env or clean_env())

    def wait_online(self, n):
        end = time.time() + 20
        while time.time() < end:
            nodes = self.get("/klong/nodes")[1]["nodes"]
            if all(x["online"] and x["klong"] for x in nodes[:n]):
                return
            time.sleep(0.1)
        raise AssertionError("nodes never came online")

    def release(self, who="tester1"):
        self.post("/klong/control?action=release", who=who)


@NEED
class Cli(unittest.TestCase):
    bridge = None

    @classmethod
    def setUpClass(cls):
        cls.bridge = Bridge("--nodes", "3", "--selftest")
        cls.bridge.wait_online(3)

    @classmethod
    def tearDownClass(cls):
        cls.bridge.stop()
        # the simulator process went with it
        with unittest.TestCase().assertRaises(ProcessLookupError):
            os.kill(cls.bridge.sim_pid, 0)

    def setUp(self):
        # control is the page's: a client that is left holding it blocks the next test for 10 s
        for who in ("tester1", "anne"):
            self.bridge.release(who)

    def test_status_shows_the_nodes_and_the_selftest(self):
        deadline = time.time() + 40
        while time.time() < deadline:
            r = self.bridge.cli("status")
            if "klong selftest pass" in r.stdout:
                break
            time.sleep(0.3)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("klong selftest pass", r.stdout)
        self.assertIn("alive,klong", r.stdout)
        self.assertIn("selftest-pass", r.stdout)
        self.assertIn("0x51", r.stdout)
        self.assertEqual(r.stdout.count("online"), 3)
        self.assertEqual(r.stdout.count("offline"), 1)  # a bench has four ports
        # the self-test report is on channel 3, not in the console
        self.assertNotIn("selftest", self.bridge.get("/klong/output?node=1&from=0")[1]["data"])

    def test_eval_prints_the_output_and_the_exit_status(self):
        r = self.bridge.cli("eval", "--node", "1", "1+2")
        self.assertEqual((r.returncode, r.stdout), (0, "3\n"), r.stderr)
        r = self.bridge.cli("eval", "--node", "1", 'a::5\nb::a*2\n.p("b is ",$b);b+1')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("b is 10", r.stdout)
        self.assertNotIn("klong> ", r.stdout)  # our own echo is left out
        r = self.bridge.cli("eval", "--node", "1", '1+"a"')
        self.assertEqual(r.returncode, 1)
        self.assertIn("type error", r.stdout)

    def test_repl_session_and_state_between_lines(self):
        r = self.bridge.cli("repl", "--node", "2", stdin='n::41\nn+1\nf::{x*x}\nf(12)\n1+"a"\n3+4\n:q\n')
        self.assertEqual(r.returncode, 0, r.stderr)
        out = r.stdout
        self.assertIn("klong 0.0.1 node=2", out)
        for want in ["42\n", "144\n", "type error", "7\n"]:
            self.assertIn(want, out)
        self.assertNotIn("klong> ", out)

    def test_a_multi_line_function_in_the_repl(self):
        r = self.bridge.cli("repl", "--node", "2", stdin="g::{[t];t::x+1;\n  t*2}\ng(4)\n:q\n")
        self.assertIn("10\n", r.stdout)

    def test_ctrl_c_interrupts_a_runaway_loop(self):
        p = subprocess.Popen([sys.executable, CLI, "--host", self.bridge.host, "repl", "--node", "1"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=clean_env())
        try:
            p.stdin.write("{1}{x+1}:~0\n")
            p.stdin.flush()
            time.sleep(2.5)
            nodes = self.bridge.get("/klong/nodes")[1]["nodes"]
            self.assertEqual(nodes[1]["load"], 100, nodes)  # STATUS load while it evaluates
            os.kill(p.pid, signal.SIGINT)
            time.sleep(2.5)
            p.stdin.write("3+4\n:q\n")
            p.stdin.flush()
            out, err = p.communicate(timeout=30)
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
        self.assertIn("[interrupt sent]", out)
        self.assertIn("interrupt error: interrupted", out)
        self.assertIn("7\n", out)
        self.assertEqual(p.returncode, 0, err)

    def test_run_uploads_and_runs_a_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".kg", delete=False) as f:
            f.write('sq::{x*x}\n.p("from the file")\nsq(12)\n')
            small = f.name
        with tempfile.NamedTemporaryFile("w", suffix=".kg", delete=False) as f:
            f.write("".join("v%d::%d\n" % (i, i) for i in range(400)) + '.p("big file done")\nv399+1\n')  # about 4 KB: several datagrams
            big = f.name
        try:
            r = self.bridge.cli("run", "--node", "1", small)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("from the file\n", r.stdout)
            self.assertIn("144", r.stdout)
            self.assertNotIn("\x01", r.stdout)
            self.assertEqual(self.bridge.cli("eval", "--node", "1", "sq(3)").stdout, "9\n")
            r = self.bridge.cli("run", "--node", "1", big, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("big file done", r.stdout)
            self.assertIn("400", r.stdout)
        finally:
            os.unlink(small)
            os.unlink(big)

    def test_run_sends_any_text_file_whatever_its_lines(self):
        # raw=1: no newline is added and nothing is echoed, so a chunk may end anywhere: a line of 3000
        # characters and a file without a final newline go through
        with tempfile.NamedTemporaryFile("w", suffix=".kg", delete=False) as f:
            f.write("x::1\ny::[" + " ".join(str(i % 10) for i in range(1500)) + "]\n" + ".p(\"long line ok\")\n+/y")
            path = f.name
        try:
            r = self.bridge.cli("run", "--node", "1", path, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("long line ok", r.stdout)
            self.assertIn("6750", r.stdout)  # 150 * (0+1+..+9) * 1 = 6750
        finally:
            os.unlink(path)

    def test_raw_eval_adds_no_newline_and_leaves_no_echo(self):
        b = self.bridge
        start = b.get("/klong/output?node=1&from=4294967295")[1]["next"]
        st, o = b.post("/klong/eval?node=1&raw=1", b"6*7\n")
        self.assertEqual(st, 200)
        end = time.time() + 20
        while time.time() < end:
            out = b.get("/klong/output?node=1&from=%d" % start)[1]
            if any(d["id"] == o["id"] for d in out["dones"]):
                break
            time.sleep(0.05)
        self.assertEqual(out["data"], "42\n")  # no `klong> 6*7`
        # without raw=1 a newline is added and the line echoed; without text/plain the body is not there
        st, o = b.post("/klong/eval?node=1", b"7*7")
        self.assertEqual(st, 200)
        req = urllib.request.Request("http://%s/klong/eval?node=1&id=tester1&name=T" % b.host, data=b"1\n", method="POST", headers={"Content-Type": "application/x-www-form-urlencoded"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(cm.exception.code, 400)
        time.sleep(1)
        b.release()

    def test_long_output_arrives_whole(self):
        r = self.bridge.cli("eval", "--node", "1", "!1200")
        want = "[" + " ".join(str(i) for i in range(1200)) + "]\n"
        self.assertEqual(r.stdout, want)

    def test_two_nodes_in_turn(self):
        for k in range(3):
            for n in (1, 2):
                self.assertEqual(self.bridge.cli("eval", "--node", str(n), "w::%d" % (n * 10 + k)).returncode, 0)
            self.assertEqual(self.bridge.cli("eval", "--node", "1", "w").stdout, "%d\n" % (10 + k))
            self.assertEqual(self.bridge.cli("eval", "--node", "2", "w").stdout, "%d\n" % (20 + k))

    def test_ipc_between_nodes_while_a_console_is_in_use(self):
        r = self.bridge.cli("eval", "--node", "2", 'h::.cli("node:1");h(:stping1,,5)')
        self.assertEqual((r.returncode, r.stdout.strip().splitlines()[-1]), (0, "5"), r.stderr)
        self.assertEqual(self.bridge.cli("eval", "--node", "2", ".clic(h)").returncode, 0)

    def test_the_http_api_as_the_real_bridge_has_it(self):
        b = self.bridge
        start = b.get("/klong/output?node=1&from=4294967295")[1]["next"]
        st, o = b.post("/klong/eval?node=1", b"6*7")  # no newline: one is added
        self.assertEqual(st, 200)
        self.assertEqual(list(o), ["id"])
        end = time.time() + 20
        while time.time() < end:
            st, out = b.get("/klong/output?node=1&from=%d" % start)
            if out["done"] and out["done"]["id"] == o["id"]:
                break
            time.sleep(0.05)
        self.assertEqual(out["done"], {"id": o["id"], "status": "ok"})
        self.assertLessEqual({"node", "next", "data", "done", "dones", "load", "online", "klong", "sent", "control"}, set(out))
        self.assertEqual(out["dones"][0], out["done"])  # the newest first
        self.assertLessEqual(len(out["dones"]), 8)
        self.assertEqual((out["node"], out["online"], out["klong"], out["load"]), (1, True, True, 0))
        self.assertEqual(out["sent"], {"id": o["id"], "state": "delivered"})
        self.assertEqual(sorted(out["control"]), ["driver", "retry", "you"])
        self.assertEqual((out["control"]["you"], out["control"]["driver"]), ("driving", "Tester"))
        # the transcript has the typed line after a prompt, then the result
        self.assertTrue(out["data"].endswith("klong> 6*7\n42\n"), repr(out["data"][-40:]))
        # no `from`: the oldest byte kept; a `from` past the end: nothing, and the end
        st, first = b.get("/klong/output?node=1")
        self.assertIn("node", first)
        n = out["next"]
        self.assertEqual(b.get("/klong/output?node=1&from=4294967295")[1]["data"], "")
        self.assertEqual(b.post("/klong/interrupt?node=1"), (200, {"ok": True}))
        nodes = b.get("/klong/nodes")[1]["nodes"]
        self.assertEqual(len(nodes), 4)
        self.assertEqual(sorted(nodes[1]), ["flags", "info", "klong", "load", "node", "online", "selftest"])
        self.assertEqual(nodes[1]["flags"] & 0x10, 0x10)
        self.assertTrue(nodes[1]["info"].startswith("klong 0.0.1 node=1 "))
        # the errors and their texts
        self.assertEqual(b.post("/klong/eval?node=7", b"1\n"), (400, {"error": "node=0..3"}))
        self.assertEqual(b.post("/klong/eval?node=1", b""), (400, {"error": "send the lines as the request body"}))
        self.assertEqual(b.post("/klong/eval?node=1", b"1" * 1025), (413, {"error": "at most 1024 bytes at a time"}))
        self.assertEqual(b.post("/klong/eval?node=3", b"1\n"), (503, {"error": "that node is offline"}))
        b.release()

    def test_a_response_has_at_most_4k_and_the_window_is_8k(self):
        b = self.bridge
        self.assertEqual(self.bridge.cli("eval", "--node", "2", "!3000").returncode, 0)  # 14 KB of output
        st, o = b.get("/klong/output?node=2&from=0")
        self.assertEqual(st, 200)
        self.assertTrue(o["lost"])
        self.assertEqual(len(o["data"].encode()), 4096)
        st, o2 = b.get("/klong/output?node=2&from=%d" % o["next"])
        self.assertNotIn("lost", o2)
        self.assertEqual(len(o2["data"].encode()), 4096)
        b.release()

    def test_429_when_two_are_waiting(self):
        b = self.bridge
        # four at once into a queue of two (the lane takes a datagram at a time, and a slow sender is not needed
        # for the third to find it full: the first two are still waiting for their ACK or their turn)
        codes = []
        threads = [threading.Thread(target=lambda: codes.append(b.post("/klong/eval?node=2", b"1\n")[0])) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertTrue(set(codes) <= {200, 429}, codes)
        self.assertIn(200, codes)
        time.sleep(2)
        b.release()

    def test_control_409_take_give_and_release(self):
        b = self.bridge
        b.release("anne")
        b.release("tester1")
        # anne claims by typing; tester1 is told whose turn it is
        self.assertEqual(b.post("/klong/eval?node=1", b"1\n", who="anne")[0], 200)
        self.assertEqual(b.post("/klong/eval?node=1", b"1\n", who="tester1"), (409, {"error": "Tester has control"}))
        self.assertEqual(b.post("/klong/interrupt?node=1", who="tester1")[0], 409)
        self.assertEqual(b.post("/klong/control?action=claim", who="tester1"), (409, {"error": "Tester has control"}))
        self.assertEqual(b.post("/klong/control?action=give", who="tester1"), (409, {"error": "you do not have control"}))
        # tester1 takes it from anne: asking, then anne gives
        self.assertEqual(b.post("/klong/control?action=take", who="tester1"), (200, {"ok": True}))
        c = b.get("/klong/output?node=1&from=4294967295", who="tester1")[1]["control"]
        self.assertEqual((c["you"], c["driver"], c["asker"]), ("asking", "Tester", "Tester"))
        self.assertEqual(b.post("/klong/control?action=give", who="anne"), (200, {"ok": True}))
        self.assertEqual(b.get("/klong/output?node=1&from=4294967295", who="tester1")[1]["control"]["you"], "driving")
        self.assertEqual(b.post("/klong/eval?node=1", b"2\n", who="tester1")[0], 200)
        # keep: the driver may refuse; the one turned down must wait before asking again
        self.assertEqual(b.post("/klong/control?action=take", who="anne"), (200, {"ok": True}))
        self.assertEqual(b.post("/klong/control?action=keep", who="tester1"), (200, {"ok": True}))
        self.assertEqual(b.post("/klong/control?action=take", who="anne"), (409, {"error": "wait before asking again"}))
        b.release("tester1")
        self.assertEqual(b.get("/klong/output?node=1&from=4294967295")[1]["control"]["you"], "free")

    def test_the_cli_reports_and_takes_control(self):
        b = self.bridge
        b.release("anne")
        self.assertEqual(b.post("/klong/eval?node=1", b"1\n", who="anne")[0], 200)
        r = b.cli("eval", "--node", "1", "1")
        self.assertEqual(r.returncode, 2)
        self.assertIn("Tester has control", r.stderr)
        self.assertIn("--take", r.stderr)
        # with --take it asks, anne (polling, as a page does) gives
        done = threading.Event()

        def answer():
            end = time.time() + 20
            while not done.is_set() and time.time() < end:
                c = b.get("/klong/output?node=1&from=4294967295", who="anne")[1]["control"]
                if c.get("asker"):
                    b.post("/klong/control?action=give", who="anne")
                    return
                time.sleep(0.1)

        t = threading.Thread(target=answer)
        t.start()
        try:
            r = b.cli("eval", "--node", "1", "2+2", "--take")
        finally:
            done.set()
            t.join()
        self.assertEqual((r.returncode, r.stdout.strip().splitlines()[-1]), (0, "4"), r.stderr)
        # the CLI let go of control when it ended
        self.assertEqual(b.get("/klong/output?node=1&from=4294967295", who="anne")[1]["control"]["you"], "free")

    def test_a_driver_that_does_not_poll_is_gone_after_10_seconds(self):
        b = self.bridge
        b.release("anne")
        self.assertEqual(b.post("/klong/eval?node=1", b"1\n", who="anne")[0], 200)
        time.sleep(10.5)
        self.assertEqual(b.get("/klong/output?node=1&from=4294967295")[1]["control"]["you"], "free")


@NEED
class Offline(unittest.TestCase):
    def test_503_for_a_node_that_is_offline(self):
        b = Bridge("--nodes", "3", "--mute", "2")
        try:
            b.wait_online(2)
            self.assertEqual(b.post("/klong/eval?node=2", b"1\n"), (503, {"error": "that node is offline"}))
            self.assertEqual(b.post("/klong/interrupt?node=2")[0], 503)
            st, o = b.get("/klong/output?node=2")  # reading is allowed, and says so
            self.assertEqual((st, o["online"], o["klong"]), (200, False, False))
            r = b.cli("eval", "--node", "2", "1")
            self.assertEqual(r.returncode, 2)
            self.assertIn("that node is offline", r.stderr)
            self.assertIn("offline", b.cli("status").stdout)
        finally:
            b.stop()


class FakeBridge(http.server.BaseHTTPRequestHandler):
    """What the CLI does with answers the simulator does not give: delivery that failed, lost output."""

    mode = ""
    seen = []

    def log_message(self, *a):
        pass

    def reply(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        FakeBridge.seen.append(self.path)
        base = {"node": 1, "online": True, "klong": True, "load": 0, "control": {"you": "driving", "retry": 0}}
        if FakeBridge.mode == "refused":
            return self.reply(200, {**base, "next": 0, "data": "", "done": None, "sent": {"id": 7, "state": "refused"}})
        if FakeBridge.mode == "lost":
            if "from=4294967295" in self.path:
                return self.reply(200, {**base, "next": 50000, "data": "", "done": None, "sent": {"id": 0, "state": "none"}})
            if "from=58000" in self.path:
                return self.reply(200, {**base, "next": 58000, "data": "", "done": {"id": 7, "status": "ok"}, "sent": {"id": 7, "state": "delivered"}})
            return self.reply(200, {**base, "next": 58000, "data": "klong> 1\ntail\n", "lost": True, "done": {"id": 7, "status": "ok"}, "sent": {"id": 7, "state": "delivered"}})
        self.reply(200, {**base, "next": 0, "data": "", "done": {"id": 7, "status": "ok"}, "sent": {"id": 7, "state": "delivered"}})

    def do_POST(self):
        FakeBridge.seen.append(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        self.reply(200, {"id": 7} if "/klong/eval" in self.path else {"ok": True})


class WithFake(unittest.TestCase):
    def run_cli(self, mode, *args, env=None):
        FakeBridge.mode = mode
        FakeBridge.seen = []
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeBridge)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            host = "127.0.0.1:%d" % srv.server_address[1]
            e = clean_env(**(env or {}))
            argv = [sys.executable, CLI] + ([] if env and "KLONG_BENCH_HOST" in env else ["--host", host]) + list(args)
            if env and "KLONG_BENCH_HOST" in env:
                e["KLONG_BENCH_HOST"] = host
            return subprocess.run(argv, capture_output=True, text=True, timeout=30, env=e, stdin=subprocess.DEVNULL)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_refused_is_not_delivered(self):
        r = self.run_cli("refused", "eval", "--node", "1", "1+1")
        self.assertEqual(r.returncode, 2)
        self.assertIn("not delivered", r.stderr)
        self.assertIn("busy for another sender", r.stderr)

    def test_lost_output_is_noticed_and_our_echo_is_left_out(self):
        r = self.run_cli("lost", "eval", "--node", "1", "1")
        self.assertEqual(r.returncode, 0)
        self.assertIn("output was lost", r.stdout)
        self.assertIn("tail", r.stdout)
        self.assertNotIn("klong> 1", r.stdout)

    def test_every_call_names_the_client_and_the_host_may_come_from_the_environment(self):
        r = self.run_cli("ok", "eval", "--node", "1", "1", env={"KLONG_BENCH_HOST": "set"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(FakeBridge.seen and all("id=" in p and "name=klong-bench%20cli" in p for p in FakeBridge.seen), FakeBridge.seen)
        self.assertTrue(any("/klong/control?action=release" in p for p in FakeBridge.seen), "control is given up at the end")

    def test_a_base_url_is_a_host(self):
        sys.path.insert(0, os.path.join(os.path.dirname(CLI)))
        import importlib
        kb = importlib.import_module("klong_bench")
        for text, want in [("bench.local", ("bench.local", 80)), ("bench.local:8080", ("bench.local", 8080)), ("http://bench.local", ("bench.local", 80)), ("http://bench.local:81/", ("bench.local", 81))]:
            b = kb.Bridge(text)
            self.assertEqual((b.host, b.port), want, text)

    def test_no_host_is_written_into_the_tools(self):
        for path in (CLI, BRIDGE):
            text = open(path).read()
            self.assertNotRegex(text, r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b(?<!127\.0\.0\.1)(?<!0\.0\.0\.0)")


if __name__ == "__main__":
    unittest.main()
