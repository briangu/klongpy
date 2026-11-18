"""
.web handlers: the response a handler chooses, HEAD, 405, failures, and what a handler may start
(a timer, .webc); the HTTP-request recipe of docs/web_server.md. Needs aiohttp.
"""
import io
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stderr

try:
    import aiohttp  # noqa: F401
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from klongpy.repl import cleanup_repl, create_repl


def free_port():
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    except PermissionError as exc:
        raise unittest.SkipTest("socket bind not permitted") from exc
    finally:
        s.close()


def wait_listening(port, seconds=5):
    """.web returns once the server task is created; the socket is bound a moment later."""
    end = time.time() + seconds
    while time.time() < end:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            return
        except OSError:
            time.sleep(0.02)
    raise AssertionError("the server did not start listening")


def request(method, port, path, data=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method)
    try:
        r = urllib.request.urlopen(req, timeout=5)
        return r.status, r.headers.get("Content-Type"), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read().decode()


PROGRAM = '''
.py("klongpy.web")
ok(q)::{"fine"}
bad(q)::{1+"a"}
nf(q)::{:{;"status";404;"type";"application/json";"body";.json(:{;"error";"none"})}}
st(q)::{:{;"status";201}}
unk(q)::{:{;"colour";"red"}}
badst(q)::{:{;"status";"x"}}
num(q)::{42}
stats(q)::{:{;"type";"application/json";"body";.json(:{;"count";3;"values";[1 2 3]})}}
echo(q)::{.json(q)}
get:::{;"/ok";ok;"/bad";bad;"/nf";nf;"/st";st;"/unk";unk;"/badst";badst;"/num";num;"/stats";stats}
post:::{;"/bad";bad;"/echo";echo}
srv::.web("127.0.0.1:%d";get;post)
'''


@unittest.skipUnless(HAS_AIOHTTP, "requires aiohttp")
class TestWebHandlers(unittest.TestCase):
    def setUp(self):
        self.klong, self.loops = create_repl()
        self.port = free_port()
        self.err = io.StringIO()
        with redirect_stderr(self.err):
            self.klong(PROGRAM % self.port)
        wait_listening(self.port)

    def tearDown(self):
        h = self.klong['srv']
        if getattr(h, "runner", None) is not None:
            h.close()
        cleanup_repl(self.loops)

    def test_text_is_200_plain(self):
        self.assertEqual(request("GET", self.port, "/ok"), (200, "text/plain; charset=utf-8", "fine"))
        self.assertEqual(request("GET", self.port, "/num")[2], "42")

    def test_head_is_get_without_body(self):
        status, ctype, body = request("HEAD", self.port, "/ok")
        self.assertEqual((status, body), (200, ""))

    def test_wrong_method_is_405(self):
        self.assertEqual(request("POST", self.port, "/ok", b"a=1")[0], 405)
        self.assertEqual(request("GET", self.port, "/echo")[0], 405)
        self.assertEqual(request("GET", self.port, "/missing")[0], 404)

    def test_dictionary_response(self):
        self.assertEqual(request("GET", self.port, "/nf"),
                         (404, "application/json; charset=utf-8", '{"error":"none"}'))
        self.assertEqual(request("GET", self.port, "/st")[0::2], (201, ""))
        status, ctype, body = request("GET", self.port, "/stats")
        self.assertEqual((status, body), (200, '{"count":3,"values":[1,2,3]}'))
        self.assertTrue(ctype.startswith("application/json"))

    def test_post_receives_form_fields(self):
        self.assertEqual(request("POST", self.port, "/echo", b"a=1&b=x")[2], '{"a":"1","b":"x"}')

    def test_failures_are_500_with_one_logged_line(self):
        buf = io.StringIO()
        with redirect_stderr(buf):
            for method, path, data in [("GET", "/bad", None), ("POST", "/bad", b"a=1"),
                                       ("GET", "/unk", None), ("GET", "/badst", None)]:
                status, ctype, body = request(method, self.port, path, data)
                self.assertEqual((status, body), (500, "handler failed"), path)
                self.assertTrue(ctype.startswith("text/plain"))
        lines = buf.getvalue().strip().splitlines()
        self.assertEqual(len(lines), 4, lines)
        self.assertIn("web GET /bad: handler failed: type error:", lines[0])
        self.assertIn("web POST /bad: handler failed: type error:", lines[1])
        self.assertIn('unknown response key "colour"', lines[2])
        self.assertIn("status", lines[3])
        self.assertFalse(any("Traceback" in ln for ln in lines))

    def test_traceback_with_the_flag(self):
        self.klong['.system']['traceback'] = True
        buf = io.StringIO()
        with redirect_stderr(buf):
            request("GET", self.port, "/bad")
        self.assertIn("Traceback", buf.getvalue())


@unittest.skipUnless(HAS_AIOHTTP, "requires aiohttp")
class TestWhatAHandlerMayStart(unittest.TestCase):
    def setUp(self):
        self.klong, self.loops = create_repl()
        self.port = free_port()

    def tearDown(self):
        h = self.klong['srv']
        if getattr(h, "runner", None) is not None:
            h.close()
        cleanup_repl(self.loops)

    def test_timer_created_in_a_handler_fires(self):
        self.klong('.py("klongpy.web")')
        self.klong('fired::0')
        self.klong('tick()::{fired::1;0}')
        self.klong('go(q)::{.timer("t";0.1;tick);"ok"}')
        self.klong('post:::{;"/go";go}')
        self.klong(f'srv::.web("127.0.0.1:{self.port}";:{{}};post)')
        wait_listening(self.port)
        self.assertEqual(request("POST", self.port, "/go", b"a=1")[2], "ok")
        deadline = time.time() + 5
        while time.time() < deadline and self.klong['fired'] != 1:
            time.sleep(0.05)
        self.assertEqual(self.klong['fired'], 1)

    def test_webc_in_a_handler_finishes_the_reply_then_stops(self):
        self.klong('.py("klongpy.web")')
        self.klong('stop(q)::{.webc(srv);"bye"}')
        self.klong('post:::{;"/quit";stop}')
        self.klong(f'srv::.web("127.0.0.1:{self.port}";:{{}};post)')
        wait_listening(self.port)
        self.assertEqual(request("POST", self.port, "/quit", b"a=1"), (200, "text/plain; charset=utf-8", "bye"))
        handle = self.klong['srv']
        deadline = time.time() + 5
        while time.time() < deadline and handle.is_listening():
            time.sleep(0.05)
        self.assertFalse(handle.is_listening())
        with self.assertRaises(Exception):
            request("GET", self.port, "/")

    def test_webc_from_a_timer_stops_the_server(self):
        self.klong('.py("klongpy.web")')
        self.klong('stop()::{.webc(srv);0}')
        self.klong(f'srv::.web("127.0.0.1:{self.port}";:{{}};:{{}})')
        self.klong('th::.timer("q";0.1;stop)')
        handle = self.klong['srv']
        deadline = time.time() + 5
        while time.time() < deadline and handle.is_listening():
            time.sleep(0.05)
        self.assertFalse(handle.is_listening())


@unittest.skipUnless(HAS_AIOHTTP, "requires aiohttp")
class TestRequestRecipe(unittest.TestCase):
    """The recipe of docs/web_server.md, "Requesting a server from Klong", against a local .web server."""

    def test_fetch_json_with_pyf(self):
        server, sloops = create_repl()
        client, cloops = create_repl()
        port = free_port()
        try:
            server('.py("klongpy.web")')
            server('stats(q)::{.json(:{;"count";3;"values";[1 2 3]})}')
            server('get:::{;"/stats";stats}')
            server(f'srv::.web("127.0.0.1:{port}";get;:{{}})')
            wait_listening(port)
            client('.pyf("urllib.request";"urlopen")')
            client('fetch(url)::{[r b];r::urlopen(url);b::.pyc(r,"read";[];:{});.pyc(b,"decode";[];:{})}')
            client(f'text::fetch("http://127.0.0.1:{port}/stats")')
            self.assertEqual(client['text'], '{"count":3,"values":[1,2,3]}')
            client('d::.unjson(text)')
            self.assertEqual(client('d@"count"'), 3)
            self.assertEqual(list(client('d@"values"')), [1, 2, 3])
        finally:
            h = server['srv']
            if getattr(h, "runner", None) is not None:
                h.close()
            cleanup_repl(cloops)
            cleanup_repl(sloops)


if __name__ == "__main__":
    unittest.main()
