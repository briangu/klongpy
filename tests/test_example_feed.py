"""examples/stocks/feed/demo.kg: the simulated feed, a server and a consumer in one process.

The run is seeded, so its output is exact and must not depend on the backend.
The program ends by itself after the summary: the test checks a clean exit.
"""
import os
import socket
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEED = os.path.join(ROOT, "examples", "stocks", "feed")

EXPECTED = """\
feed: published 200 trades
summary of 200 trades published
AAPL     last   187.78 vwap   189.79 sma5   187.47 volume  27000 trades 53
NVDA     last   119.93 vwap   120.32 sma5   120.30 volume  19800 trades 44
TSLA     last   248.62 vwap   249.19 sma5   248.69 volume  25000 trades 49"""


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_demo(extra=(), seconds=60):
    """Run the demo to its end; return (output lines, exit status). A run that does not end in time fails."""
    port = free_port()
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "-W", "ignore", "-m", "klongpy.cli", "--strict=2", *extra, "demo.kg", "--", str(port)]
    p = subprocess.Popen(cmd, cwd=FEED, env=env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out, _ = p.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        p.kill()
        out, _ = p.communicate()
        raise AssertionError(f"the demo did not end within {seconds} s; output:\n{out}")
    return out.splitlines(), p.returncode


class TestFeedDemo(unittest.TestCase):

    def test_output_is_exact(self):
        backend = os.environ.get("_KLONGPY_TEST_BACKEND", "numpy")
        extra = []
        if backend != "numpy":
            extra = ["--backend", backend]
            if os.environ.get("_KLONGPY_TEST_DEVICE"):
                extra += ["--device", os.environ["_KLONGPY_TEST_DEVICE"]]
        lines, status = run_demo(extra)
        self.assertEqual(status, 0, lines)
        banner = "feed on port"
        self.assertTrue(lines and lines[0].startswith(banner), lines)
        self.assertEqual(EXPECTED.splitlines(), lines[1:])

    def test_readme_shows_the_output(self):
        with open(os.path.join(FEED, "README.md")) as f:
            self.assertIn(EXPECTED, f.read())

    def test_runs_with_the_same_seed_give_the_same_output(self):
        self.assertEqual(run_demo()[0][1:], run_demo()[0][1:])

    def test_feed_and_consumer_as_two_programs(self):
        """`kgpy feed.kg` stays up for its server; the consumer ends by itself after its summary."""
        port = str(free_port())
        env = dict(os.environ, PYTHONPATH=ROOT, PYTHONUNBUFFERED="1")
        base = [sys.executable, "-W", "ignore", "-m", "klongpy.cli", "--strict=2"]
        feed = subprocess.Popen(base + ["feed.kg", "--", port, "400", "200", "7"], cwd=FEED, env=env,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            banner = feed.stdout.readline()
            self.assertTrue(banner.startswith("feed on port"), banner)
            done = subprocess.run(base + ["consumer.kg", "--", port, "AAPL", "NVDA", "TSLA"], cwd=FEED, env=env,
                                  stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertEqual(EXPECTED.splitlines()[1:], done.stdout.splitlines())
            self.assertIsNone(feed.poll(), "the feed ended: its server must keep it up")
        finally:
            feed.terminate()                      # SIGTERM: a clean stop with status 143
            try:
                status = feed.wait(timeout=20)
            except subprocess.TimeoutExpired:
                feed.kill()
                status = feed.wait()
                self.fail("the feed did not end on SIGTERM")
            finally:
                feed.stdout.close()
        self.assertEqual(status, 143)


if __name__ == "__main__":
    unittest.main()
