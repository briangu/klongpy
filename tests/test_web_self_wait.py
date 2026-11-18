"""
A program that waits, on the Klong loop, for an answer from its own .web server gets a 503 at once instead of
hanging for ever; a slow handler, or a client that waits on something else, is not touched. Needs aiohttp.
"""
import http.server
import io
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr

from tests.test_web_handlers import HAS_AIOHTTP, free_port, request, wait_listening

from klongpy.repl import cleanup_repl, create_repl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

PROGRAM = '''
.py("klongpy.web")
.pyf("urllib.request";"urlopen")
.pyf("time";"sleep")
fetch(url)::{[r b];r::urlopen(url);b::.pyc(r,"read";[];:{});.pyc(b,"decode";[];:{})}
ok(q)::{"fine"}
slow(q)::{sleep(2.5);"slow done"}
self(q)::{fetch("http://127.0.0.1:%(port)d/ok")}
ext(q)::{fetch("http://127.0.0.1:%(ext)d/")}
get:::{;"/ok";ok;"/slow";slow;"/self";self;"/ext";ext}
srv::.web("127.0.0.1:%(port)d";get;:{})
'''


class SlowServer:
    """A server in this process (another thread) that answers after `delay` seconds."""
    def __init__(self, delay):
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                time.sleep(delay)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"external")

            def log_message(self, *a):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        outer.httpd = self.httpd

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@unittest.skipUnless(HAS_AIOHTTP, "requires aiohttp")
class TestWebSelfWait(unittest.TestCase):
    def setUp(self):
        self.klong, self.loops = create_repl()
        self.port = free_port()
        self.ext = SlowServer(2.5)
        self.addCleanup(self.ext.close)
        self.err = io.StringIO()
        with redirect_stderr(self.err):
            self.klong(PROGRAM % {"port": self.port, "ext": self.ext.port})
        wait_listening(self.port)

    def tearDown(self):
        h = self.klong['srv']
        if getattr(h, "runner", None) is not None:
            h.close()
        cleanup_repl(self.loops)

    def test_handler_that_requests_its_own_server_fails_at_once(self):
        buf = io.StringIO()
        t0 = time.time()
        with redirect_stderr(buf):
            status, ctype, body = request("GET", self.port, "/self")
        self.assertLess(time.time() - t0, 10)
        self.assertEqual((status, body), (500, "handler failed"))      # the handler got HTTP Error 503
        text = buf.getvalue()
        self.assertEqual(text.count("refused with 503"), 1, text)
        self.assertIn("cannot wait for its own server", text)
        self.assertIn("503", text)
        # the server goes on
        self.assertEqual(request("GET", self.port, "/ok")[0::2], (200, "fine"))

    def test_slow_handler_is_not_refused_for_a_client_in_this_process(self):
        results = {}
        buf = io.StringIO()

        def second():
            time.sleep(0.3)
            results["second"] = request("GET", self.port, "/ok")

        t = threading.Thread(target=second)
        with redirect_stderr(buf):
            t.start()
            slow = request("GET", self.port, "/slow")
            t.join(10)
        self.assertEqual(slow[0::2], (200, "slow done"))
        self.assertEqual(results["second"][0::2], (200, "fine"))          # waited more than a second, answered
        self.assertNotIn("refused", buf.getvalue())

    def test_loop_blocked_on_another_server_is_not_refused(self):
        results = {}
        buf = io.StringIO()

        def second():
            time.sleep(0.3)
            results["second"] = request("GET", self.port, "/ok")

        t = threading.Thread(target=second)
        with redirect_stderr(buf):
            t.start()
            ext = request("GET", self.port, "/ext")
            t.join(10)
        self.assertEqual(ext[0::2], (200, "external"))
        self.assertEqual(results["second"][0::2], (200, "fine"))
        self.assertNotIn("refused", buf.getvalue())


@unittest.skipUnless(HAS_AIOHTTP, "requires aiohttp")
class TestProgramWaitsForItsOwnServer(unittest.TestCase):
    def test_file_run(self):
        port = free_port()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "self.kg")
            with open(path, "w") as f:
                f.write('.py("klongpy.web")\n.pyf("urllib.request";"urlopen")\nok::{x;"fine"}\n'
                        'fetch(url)::{[r b];r::urlopen(url);b::.pyc(r,"read";[];:{});.pyc(b,"decode";[];:{})}\n'
                        'get:::{;"/ok";ok}\nsrv::.web("127.0.0.1:%d";get;:{})\n.p("asking")\n'
                        'r::fetch("http://127.0.0.1:%d/ok")\n.p("never")\n' % (port, port))
            env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
            t0 = time.time()
            r = subprocess.run([sys.executable, "-W", "ignore", "-m", "klongpy.cli", path], cwd=d, env=env,
                               stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
        self.assertLess(time.time() - t0, 30)
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("never", r.stdout)
        self.assertEqual(r.stderr.count("refused with 503"), 1, r.stderr)
        self.assertIn("HTTP Error 503", r.stderr)


if __name__ == "__main__":
    unittest.main()
