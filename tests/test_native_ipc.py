"""
IPC between the native runtime (native/, ticket 106) and KlongPy, and optionally PeachQ, over localhost TCP.

Skipped unless the `klong` binary is built:

    cd native && cargo build --release

The binary is `native/target/release/klong`, or the path in the environment variable KLONG_NATIVE.
The PeachQ part is opt-in as in `test_ipc_peachq.py`: set KLONG_TEST_PEACHQ to the PeachQ `q` executable.

A native node is the `klong` read-eval-print loop in a child process: lines go in on its standard input,
and what it prints comes back. It serves IPC and runs timers while it waits for the next line, so one node
can be a server, a client, or both. Every step is bounded in time.
"""
import math
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import unittest

import numpy as np

from ipc_helpers import bounded, free_port, wait_until
from peachq_helpers import ENV, Q_PRELUDE, TESTED_VERSION, PeachQServer, parse_results, peachq_exe, run_q

from klongpy.core import KGChar, KGSym, KLONG_UNDEFINED
from klongpy.errors import (KlongError, KlongNameError, KlongRangeError, KlongRemoteError, KlongSyntaxError,
                            KlongSystemError, KlongTypeError)
from klongpy.repl import cleanup_repl, create_repl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BINARY = os.environ.get("KLONG_NATIVE") or os.path.join(ROOT, "native", "target", "release", "klong")
BUILT = os.path.isfile(BINARY) and os.access(BINARY, os.X_OK)
NEED_NATIVE = unittest.skipUnless(BUILT, "native klong binary not built (cd native && cargo build --release)")
EXE = peachq_exe()
NEED_PEACHQ = unittest.skipUnless(BUILT and EXE, f"PeachQ interop is opt-in: set {ENV} to a PeachQ q executable "
                                                 f"(tested with {TESTED_VERSION}), and build the native binary")
TIMEOUT = 20
S = KGSym
ERROR_LINE = re.compile(r"^(?:\d+:\d+: )?(syntax|name|arity|type|shape|range|system|memory|internal|remote|interrupt) error: (.*)$")


class NativeError(Exception):
    def __init__(self, kind, message, text):
        super().__init__(f"{kind} error: {message}")
        self.kind, self.message, self.text = kind, message, text


class Native:
    """The native REPL in a child process."""

    DONE = "@@done"

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="klong-native-")
        self.proc = subprocess.Popen([BINARY], cwd=self.dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
        self.lines = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for raw in self.proc.stdout:
            self.lines.put(raw.decode("utf-8", "replace").rstrip("\n"))
        self.lines.put(None)

    def send(self, line):
        assert "\n" not in line
        self.proc.stdin.write(line.encode("utf-8") + b"\n")
        self.proc.stdin.flush()

    def output(self, timeout=TIMEOUT):
        """Everything printed up to the echo of the marker line."""
        self.send(f'.p("{self.DONE}")')
        out, end = [], time.monotonic() + timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.01, end - time.monotonic()))
            except queue.Empty:
                raise AssertionError(f"native node did not answer within {timeout}s; so far: {out!r}") from None
            if line is None:
                raise AssertionError(f"native node exited; output: {out!r}")
            while line.startswith("  "):            # prompts
                line = line[2:]
            if line == f'"{self.DONE}"':
                return [x for x in out if x != self.DONE]
            out.append(line)

    def run(self, line, timeout=TIMEOUT):
        """Evaluate one line; the lines it printed (the last is the value). An error raises NativeError."""
        self.send(line)
        out = self.output(timeout)
        for i, text in enumerate(out):
            m = ERROR_LINE.match(text)
            if m:
                raise NativeError(m.group(1), m.group(2), "\n".join(out[i:]))
        return out

    def value(self, expr, timeout=TIMEOUT):
        out = self.run(expr, timeout)
        return out[-1] if out else ""

    def alive(self):
        return self.proc.poll() is None

    def stop(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(5)
        for f in (self.proc.stdin, self.proc.stdout):
            try:
                f.close()
            except OSError:
                pass
        try:
            os.rmdir(self.dir)
        except OSError:
            pass


def as_np(v):
    """A backend array (a torch tensor when the session backend is torch) as a numpy array; anything else unchanged."""
    if not isinstance(v, np.ndarray) and hasattr(v, "detach") and hasattr(v, "cpu"):
        return v.detach().cpu().numpy()
    return v


def norm(v):
    """A value as plain Python data that says what kind each part is."""
    v = as_np(v)
    if isinstance(v, np.ndarray) and v.ndim == 0:
        v = v.item()
    if isinstance(v, np.ndarray):
        return [norm(x) for x in v.tolist()] if v.dtype != object else [norm(x) for x in v]
    if isinstance(v, (list, tuple)):
        return [norm(x) for x in v]
    if isinstance(v, KGSym):
        return ("sym", str(v))
    if isinstance(v, KGChar):
        return ("char", str(v))
    if isinstance(v, dict):
        return ("dict", [(norm(k), norm(x)) for k, x in v.items()])
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return ("nan",) if math.isnan(v) else ("float", float(v))
    if v is KLONG_UNDEFINED:
        return ("undefined",)
    return v


#: name, the Klong literal the native node holds, the value KlongPy holds for the same thing
VALUES = [
    ("vint", "5", 5), ("vneg", "-7", -7), ("vbig", "9223372036854775807", 2 ** 63 - 1),
    ("vfloat", "1.5", 1.5), ("vfneg", "-0.25", -0.25),
    ("vchar", "0ca", KGChar("a")), ("vstr", '"abc"', "abc"), ("vempty", '""', ""), ("vuni", '"héllo 世"', "héllo 世"),
    ("vsym", ":abc", S("abc")),
    ("vivec", "[1 2 3]", np.array([1, 2, 3])), ("vfvec", "[1.5 2.5]", np.array([1.5, 2.5])),
    ("vsvec", "[:a :b]", [S("a"), S("b")]), ("vstrs", '["ab" "cd"]', ["ab", "cd"]),
    ("vmixed", '[1 :a "s"]', [1, S("a"), "s"]), ("vnest", "[[1 2] [3 4]]", [[1, 2], [3, 4]]),
    ("vrag", "[[1 2] [3]]", [[1, 2], [3]]), ("vdeep", '[1 ["two" [3.5 :four]]]', [1, ["two", [3.5, S("four")]]]),
    ("vlist0", "[]", []),
    ("vdict", ":{[:a 1] [:b 2]}", {S("a"): 1, S("b"): 2}),
    ("vdict2", ':{[:a [1 2]] [:b "x"]}', {S("a"): [1, 2], S("b"): "x"}),
    ("vdict3", ':{[1 "one"] [2 "two"]}', {1: "one", 2: "two"}),
]

SETUP = ("idf::{x};sq::{x*x};add::{x+y};nil::{42};fail::{1+\"a\"};"
         "wait::{[t];t::.pc()+x;{.pc()<t}{x}:~0;x};" + ";".join(f"{n}::{lit}" for n, lit, _ in VALUES))


class Case(unittest.TestCase):
    def repl(self):
        klong, loops = create_repl()

        def shutdown():
            bounded(klong['.system']['closeEvent'].trigger, 20)
            bounded(lambda: cleanup_repl(loops), 20)
        self.addCleanup(shutdown)
        return klong

    def kg(self, klong, text, timeout=TIMEOUT):
        return bounded(lambda: klong(text), timeout)

    def native(self, setup=None):
        n = Native()
        self.addCleanup(n.stop)
        if setup:
            n.run(setup)
        return n

    def native_server(self, setup=SETUP):
        n = self.native(setup)
        port = free_port()
        self.assertEqual(n.value(f'.srv("127.0.0.1:{port}")'), "1")
        return n, port

    def assertSame(self, got, want, msg=None):
        self.assertEqual(norm(got), norm(want), msg)

    def native_error(self, node, line, kind, contains=None, timeout=TIMEOUT):
        with self.assertRaises(NativeError) as cm:
            node.run(line, timeout)
        self.assertEqual(cm.exception.kind, kind, cm.exception.text)
        if contains:
            self.assertIn(contains, cm.exception.message)
        return cm.exception


# ----------------------------------------------------------------------------------------------------------
# KlongPy client -> native server
# ----------------------------------------------------------------------------------------------------------

@NEED_NATIVE
class TestKlongPyClientNativeServer(Case):
    def setUp(self):
        self.server, self.port = self.native_server()
        self.klong = self.repl()
        self.kg(self.klong, f"c::.cli({self.port})")

    def req(self, *items, timeout=TIMEOUT):
        self.klong["req"] = items[0] if len(items) == 1 else list(items)
        return self.kg(self.klong, "c(req)", timeout)

    def test_every_mapped_type_arrives(self):
        for name, _, want in VALUES:
            self.assertSame(self.req(S(name)), want, name)

    def test_every_mapped_type_round_trips(self):
        for name, lit, want in VALUES:
            self.assertSame(self.req(S("idf"), want), want, name)
            # and the native node sees what it would have read from its own source
            self.klong["v"] = want
            self.kg(self.klong, "d::.clid(c);d,:got,,v")
            self.assertEqual(self.server.value(f"got~{lit}"), "1", name)

    def test_a_function_of_four_parameters_round_trips(self):
        # it travels as its display form, ":fn/4", and the server keeps it as a function
        self.kg(self.klong, "f4(a;b;c;d)::{a+b+c+d};d::.clid(c);d,:ff,,f4")
        self.assertEqual(self.server.value("ff"), ":kdb-lambda")
        self.assertNotIsInstance(self.kg(self.klong, "d?:ff"), Exception)

    def test_special_floats_and_undefined(self):
        self.assertTrue(math.isnan(self.req(S("idf"), float("nan"))))
        self.assertEqual(self.req(S("idf"), math.inf), math.inf)
        self.assertIs(self.req("xx::1"), 1)
        self.assertIs(self.kg(self.klong, "d::.clid(c);c(:set,:zz,,5)"), KLONG_UNDEFINED)
        self.assertEqual(self.req(S("zz")), 5)

    def test_calls(self):
        self.assertEqual(self.req("1+2"), 3)
        self.assertEqual(self.req(S("sq"), 7), 49)
        self.assertEqual(self.req(S("add"), 3, 4), 7)
        self.assertEqual(self.req("sq", 7), 49)                       # the callee as a string
        self.assertEqual(self.kg(self.klong, "f::c(:sq);f(9)"), 81)   # a proxy
        self.assertEqual(self.kg(self.klong, "g::c(:add);g(1;2)"), 3)
        self.assertEqual(self.kg(self.klong, "h::c(:nil);h()"), 42)
        self.assertSame(self.kg(self.klong, "f'[1 2 3]"), [1, 4, 9])
        self.assertEqual(self.kg(self.klong, "d::.clid(c);d,[:q 11];d?:q"), 11)
        self.assertEqual(self.server.value("q"), "11")
        # a Klong function sent as a value is defined there and can be called
        self.kg(self.klong, "d,:twice,,{x*2}")
        self.assertEqual(self.server.value("twice(21)"), "42")
        self.assertEqual(self.req(S("twice"), 4), 8)

    def test_errors_arrive_as_the_right_kind(self):
        for request, cls, text in [("1+\"a\"", KlongTypeError, "cannot apply + (Plus) to integer and string"),
                                   ("fail(1)", KlongTypeError, "cannot apply + (Plus)"),
                                   ("1+", KlongSyntaxError, None),
                                   ("[1 2 3]@9", KlongRangeError, None)]:
            with self.assertRaises(cls) as cm:
                self.req(request)
            if text:
                self.assertIn(text, str(cm.exception))
        with self.assertRaises(KlongNameError) as cm:
            self.req(S("nosuch"), 1)
        self.assertIn("symbol not found: nosuch", str(cm.exception))
        self.assertEqual(self.req("2+2"), 4)                          # the connection is still good

    def test_async_and_hooks(self):
        self.server.run("opened::0;closed::0;.srv.o::{opened::opened+1};.srv.c::{closed::closed+1}")
        self.kg(self.klong, f"c2::.cli({self.port});got::[];cb::{{got::got,x}};af::.async(c2(:sq);cb);af(5)")
        self.assertTrue(wait_until(lambda: list(self.klong["got"]) == [25]))
        self.assertEqual(self.server.value("opened,closed"), "[1 0]")
        self.kg(self.klong, ".clic(c2)")
        self.assertTrue(wait_until(lambda: self.server.value("opened,closed") == "[1 1]"))

    def test_server_calls_back_during_a_client_call(self):
        self.server.run('f::{.cli.h("g()")};h::{40+x};f2::{.cli.h("g2()")};up::{.cli.h(:upd,,[:ibm 2.5])}')
        self.kg(self.klong, "g::{7};g2::{c(:h,,2)};upd::{x}")
        self.assertEqual(self.req("f()"), 7)
        self.assertEqual(self.req("f2()"), 42)                        # two levels: A->B, B->A, A->B
        self.assertSame(self.req("up()"), [S("ibm"), 2.5])

    def test_nesting_limit_answers_with_an_error(self):
        self.server.run("sf::{.cli.h(:cf,,x+1)}")
        self.kg(self.klong, "cf::{c(:sf,,x+1)}")
        with self.assertRaises(KlongError) as cm:
            self.kg(self.klong, "c(:sf,,0)")
        self.assertIn("nested calls deeper than", str(cm.exception))
        self.assertEqual(self.req("1+1"), 2)

    def test_call_timeout(self):
        self.assertEqual(self.kg(self.klong, ".clit(c;0.5)"), 1)
        t0 = time.monotonic()
        with self.assertRaises(TimeoutError):
            self.req(S("wait"), 2)
        self.assertLess(time.monotonic() - t0, 1.9)
        with self.assertRaises(KlongSystemError):
            self.req("1+1")
        self.kg(self.klong, f"c::.cli({self.port})", 15)              # once the server is done waiting
        self.assertEqual(self.req("1+1"), 2)

    def test_large_vector(self):
        n = 200_000
        self.klong["big"] = np.arange(n)
        r = as_np(self.kg(self.klong, "c(:idf,,big)", 60))
        self.assertEqual(r.dtype.kind, "i")
        np.testing.assert_array_equal(r, np.arange(n))
        self.klong["bigf"] = np.arange(1_000_000) * 0.5               # 8 MB each way
        r = as_np(self.kg(self.klong, "c(:idf,,bigf)", 60))
        self.assertEqual(r.dtype.kind, "f")
        np.testing.assert_array_equal(r, np.arange(1_000_000) * 0.5)
        self.assertEqual(self.server.value("#c2::0.5*!1000000"), "1000000")
        np.testing.assert_array_equal(as_np(self.kg(self.klong, "c(:c2)", 60)), np.arange(1_000_000) * 0.5)

    def test_authentication(self):
        self.server.run('.srv.auth::{(x~"ann")&y~"pw"}')
        with self.assertRaises(KlongSystemError):
            self.kg(self.klong, f'.cli("localhost:{self.port}:bob:no")')
        self.kg(self.klong, f'c3::.cli("localhost:{self.port}:ann:pw")')
        self.klong["req"] = "1+1"
        self.assertEqual(self.kg(self.klong, "c3(req)"), 2)


# ----------------------------------------------------------------------------------------------------------
# native client -> KlongPy server
# ----------------------------------------------------------------------------------------------------------

@NEED_NATIVE
class TestNativeClientKlongPyServer(Case):
    def setUp(self):
        self.server = self.repl()
        for name, _, value in VALUES:
            self.server[name] = value
        self.server["slow"] = lambda x: (time.sleep(x), x)[1]
        self.kg(self.server, 'idf::{x};sq::{x*x};add::{x+y};nil::{42};fail::{1+"a"}')
        self.port = free_port()
        self.assertEqual(self.kg(self.server, f".srv({self.port})"), 1)
        self.client = self.native(f"c::.cli({self.port})")

    def test_every_mapped_type_arrives(self):
        for name, lit, _ in VALUES:
            self.assertEqual(self.client.value(f"c(:{name})~{lit}"), "1", f"{name}: {self.client.value(f'c(:{name})')}")

    def test_every_mapped_type_round_trips(self):
        for name, lit, want in VALUES:
            self.assertEqual(self.client.value(f"c([:idf {lit}])~{lit}"), "1", name)
            self.client.run(f"d::.clid(c);d,[:got {lit}]")
            self.assertSame(self.server["got"], want, name)

    def test_calls(self):
        c = self.client
        self.assertEqual(c.value('c("1+2")'), "3")
        self.assertEqual(c.value("c(:sq,,7)"), "49")
        self.assertEqual(c.value("c(:add,3,4)"), "7")
        self.assertEqual(c.value("f::c(:sq);f(9)"), "81")
        self.assertEqual(c.value("g::c(:add);g(1;2)"), "3")
        self.assertEqual(c.value("h::c(:nil);h()"), "42")
        self.assertEqual(c.value("f'[1 2 3]"), "[1 4 9]")
        self.assertEqual(c.value("d::.clid(c);d,[:q 11];d?:q"), "11")
        self.assertEqual(self.server["q"], 11)
        c.run("d,:twice,,{x*2}")
        self.assertEqual(self.kg(self.server, "twice(21)"), 42)
        self.assertEqual(c.value("c(:twice,,4)"), "8")

    def test_errors_arrive_as_the_right_kind(self):
        c = self.client
        self.native_error(c, 'c("1+""a""")', "type", "cannot apply + (Plus) to integer and string")
        self.native_error(c, "c(:fail,,1)", "type")
        self.native_error(c, 'c("1+")', "syntax")
        self.native_error(c, "c(:nosuch,,1)", "name", "symbol not found: nosuch")
        self.native_error(c, 'c("[1 2 3]@9")', "range")
        self.assertEqual(c.value('c("2+2")'), "4")

    def test_async(self):
        c = self.client
        c.run("got::[];cb::{got::got,x};af::.async(c(:sq);cb)")
        self.assertEqual(c.value("af(5)"), "1")
        self.assertTrue(wait_until(lambda: c.value("got") == "[25]"))
        c.run('ac::.async(c;cb);ac("100+1")')
        self.assertTrue(wait_until(lambda: c.value("got") == "[25 101]"))

    def test_server_calls_back_during_a_client_call(self):
        self.kg(self.server, 'f::{.cli.h("g()")};h::{40+x};f2::{.cli.h("g2()")};up::{.cli.h(:upd,,[:ibm 2.5])}')
        self.client.run("g::{7};g2::{c(:h,,2)};upd::{x}")
        self.assertEqual(self.client.value('c("f()")'), "7")
        self.assertEqual(self.client.value('c("f2()")'), "42")
        self.assertEqual(self.client.value('c("up()")'), "[:ibm 2.5]")

    def test_nesting_limit_answers_with_an_error(self):
        self.kg(self.server, "sf::{.cli.h(:cf,,x+1)}")
        self.client.run("cf::{c(:sf,,x+1)}")
        with self.assertRaises(NativeError) as cm:
            self.client.run("c(:sf,,0)")
        self.assertIn("nested calls deeper than", cm.exception.message)
        self.assertEqual(self.client.value('c("1+1")'), "2")

    def test_push_to_an_idle_client(self):
        self.kg(self.server, "hh::0;grab::{hh::.cli.h;1}")
        self.client.run('got::0;rec::{got::x;x*2};c("grab()")')
        self.assertEqual(self.kg(self.server, "hh(:rec,,5)"), 10)
        self.assertEqual(self.client.value("got"), "5")

    def test_call_timeout(self):
        c = self.client
        self.assertEqual(c.value(".clit(c;0.5)"), "1")
        t0 = time.monotonic()
        self.native_error(c, "c(:slow,,2)", "system", "timed out")
        self.assertLess(time.monotonic() - t0, 1.9)
        self.native_error(c, 'c("1+1")', "system")
        self.assertEqual(c.value(f'c::.cli({self.port});c("1+1")'), "2")

    def test_large_vector(self):
        n = 200_000
        self.assertEqual(self.client.value(f"big::!{n};r::c(:idf,,big);(r~big),#r", 60), f"[1 {n}]")
        # a 1,000,000-element float array, 8 MB each way
        self.assertEqual(self.client.value("bigf::0.5*!1000000;r::c(:idf,,bigf);(r~bigf),#r", 60), "[1 1000000]")
        self.client.run("d::.clid(c);d,:got,,bigf", 60)
        np.testing.assert_array_equal(self.server["got"], np.arange(1_000_000) * 0.5)

    def test_connection_refused_and_server_gone(self):
        c = self.client
        self.native_error(c, f".cli({free_port()})", "system", "cannot connect")
        self.kg(self.server, ".srv(0)")
        self.native_error(c, 'c("1+1")', "system")


# ----------------------------------------------------------------------------------------------------------
# native <-> native
# ----------------------------------------------------------------------------------------------------------

@NEED_NATIVE
class TestNativeToNative(Case):
    def setUp(self):
        self.server, self.port = self.native_server()
        self.client = self.native(f"c::.cli({self.port})")

    def test_types_calls_and_errors(self):
        c = self.client
        for name, lit, _ in VALUES:
            self.assertEqual(c.value(f"(c(:{name})~{lit})&c([:idf {lit}])~{lit}"), "1", name)
        self.assertEqual(c.value("f::c(:sq);f'[1 2 3]"), "[1 4 9]")
        self.native_error(c, "c(:fail,,1)", "type")
        self.native_error(c, "c(:nosuch,,1)", "name")

    def test_callbacks_two_levels_and_the_nesting_limit(self):
        self.server.run('f::{.cli.h("g()")};h::{40+x};f2::{.cli.h("g2()")};sf::{.cli.h(:cf,,x+1)}')
        self.client.run("g::{7};g2::{c(:h,,2)};cf::{c(:sf,,x+1)}")
        self.assertEqual(self.client.value('c("f()")'), "7")
        self.assertEqual(self.client.value('c("f2()")'), "42")
        with self.assertRaises(NativeError) as cm:
            self.client.run("c(:sf,,0)")
        self.assertIn("nested calls deeper than", cm.exception.message)
        self.assertEqual(self.client.value('c("1+1")'), "2")

    def test_a_peer_that_dies_fails_the_pending_call(self):
        threading.Timer(0.5, self.server.proc.kill).start()
        t0 = time.monotonic()
        self.native_error(self.client, "c(:wait,,10)", "system")
        self.assertLess(time.monotonic() - t0, 5)
        self.native_error(self.client, 'c("1+1")', "system")

    def test_timers_run_while_the_node_serves(self):
        self.server.run('n::0;tick::{n::n+1;1};th::.timer("tick";0.05;tick)')
        time.sleep(0.4)
        n = int(self.client.value("c(:n)"))
        self.assertTrue(4 <= n <= 9, n)
        self.assertEqual(self.client.value('c(".timerc(th)")'), "1")
        n = int(self.client.value("c(:n)"))
        time.sleep(0.2)
        self.assertEqual(int(self.client.value("c(:n)")), n)

    def test_large_vector(self):
        n = 200_000
        self.assertEqual(self.client.value(f"big::!{n};r::c(:idf,,big);(r~big),#r", 60), f"[1 {n}]")


# ----------------------------------------------------------------------------------------------------------
# hostile bytes
# ----------------------------------------------------------------------------------------------------------

@NEED_NATIVE
class TestNativeServerSurvivesBadPeers(Case):
    """Seeded: damaged test vectors, garbage, lies about lengths, peers that stop half way."""

    ROUNDS = 300

    def test_fuzzed_connections_do_not_stop_the_server(self):
        import json
        import random
        import socket
        with open(os.path.join(HERE, "ipc_vectors", "vectors.json"), encoding="utf-8") as f:
            vectors = [bytes.fromhex(v["hex"]) for v in json.load(f)]
        server, port = self.native_server()
        rng = random.Random(106)
        answered = closed = 0
        for i in range(self.ROUNDS):
            msg = bytearray(rng.choice(vectors))
            how = rng.randrange(8)
            if how == 0:
                msg = bytearray(rng.randbytes(rng.randrange(1, 64)))
            elif how == 1:
                for _ in range(rng.randrange(1, 4)):
                    msg[rng.randrange(len(msg))] = rng.randrange(256)
            elif how == 2:
                msg = msg[:rng.randrange(1, len(msg))]                       # stops half way
            elif how == 3 and len(msg) >= 8:
                msg[4:8] = rng.choice([b"\xff\xff\xff\x7f", b"\x00\x00\x00\x80", b"\x00\x00\x00\x00", b"\xff\xff\xff\x0f"])
            elif how == 4 and len(msg) >= 9:
                msg[1] = 1                                                   # a sync request
                msg[8 + rng.randrange(len(msg) - 8)] ^= 1 << rng.randrange(8)
            elif how == 5:
                msg = msg * rng.randrange(2, 5)
            elif how == 6 and len(msg) >= 2:
                msg[1] = rng.choice([0, 1, 2, 3, 255])                       # a response nobody asked for, a bad type
            hello = b"\x03\x00" if rng.randrange(6) else bytes(rng.randbytes(rng.randrange(0, 300)))
            s = socket.create_connection(("127.0.0.1", port), timeout=5)
            try:
                s.settimeout(0.2)
                s.sendall(hello)
                try:
                    s.recv(1)
                    s.sendall(bytes(msg))
                    if rng.randrange(3) == 0:
                        s.shutdown(socket.SHUT_WR)
                    if s.recv(65536):
                        answered += 1
                    else:
                        closed += 1
                except (socket.timeout, OSError):
                    pass
            finally:
                s.close()
            if i % 50 == 49:
                self.assertTrue(server.alive(), f"the server died in round {i}")
        self.assertTrue(server.alive())
        # still serving, and no connection is left behind
        client = self.native(f"c::.cli({port})")
        self.assertEqual(client.value('c("1+2")'), "3")
        self.assertGreater(answered, 0)
        self.assertGreater(closed, 0)
        mem = [int(x) for x in server.value(".mem()").strip("[]").split()]
        self.assertLess(mem[0], 4 * 1024 * 1024, mem)


# ----------------------------------------------------------------------------------------------------------
# PeachQ
# ----------------------------------------------------------------------------------------------------------

@NEED_PEACHQ
class TestNativeClientPeachQServer(Case):
    @classmethod
    def setUpClass(cls):
        cls.peachq = PeachQServer(EXE)
        cls.addClassCleanup(cls.peachq.stop)

    def setUp(self):
        self.assertTrue(self.peachq.alive(), self.peachq.stdout())
        self.c = self.native(f"c::.cli({self.peachq.port})")

    def q(self, text):
        return self.c.value('c("' + text.replace('"', '""') + '")')

    def test_values_from_q(self):
        for text, want in [("1+1", "2"), ("42", "42"), ("42i", "42"), ("42h", "42"), ("1b", "1"), ("0x2a", "42"),
                           ("2.5", "2.5"), ("2.5e", "2.5"), ('"a"', "0ca"), ('"abc"', '"abc"'), ("`abc", ":abc"),
                           ("1 2 3", "[1 2 3]"), ("1 2 3i", "[1 2 3]"), ("101b", "[1 0 1]"), ("1.5 2.5", "[1.5 2.5]"),
                           ("`a`b", "[:a :b]"), ('(1;`a;"s")', '[1 :a 0cs]'), ('(1;`a;"st")', '[1 :a "st"]'), ('("ab";"cd")', '["ab" "cd"]'),
                           ("(1 2;3 4)", "[[1 2] [3 4]]"), ("()", "[]"), ("til 5", "[0 1 2 3 4]"),
                           ("`a`b!1 2", ":{[:a 1] [:b 2]}"), ("2000.01.02", "1"), ("0D00:00:00.000000001", "1")]:
            self.assertEqual(self.q(text), want, text)
        self.assertEqual(self.c.value('t::c("([]a:1 2;b:`x`y)");(t?:a),t?:b'), "[1 2 :x :y]")   # a table is its dictionary of columns

    def test_values_to_q(self):
        for name, lit, _ in VALUES:
            if name == "vdict3":
                continue
            self.assertEqual(self.c.value(f"d::.clid(c);d,[:kv {lit}];(d?:kv)~{lit}"), "1", name)
        for lit, shown in [("5", "-7|5"), ("1.5", "-9|1.5"), ('"abc"', '10|"abc"'), (":abc", "-11|`abc"),
                           ("[1 2 3]", "7|1 2 3"), ("[1.5 2.5]", "9|1.5 2.5"), ("[:a :b]", "11|`a`b"),
                           (":{[:a 1] [:b 2]}", "99|`a`b!1 2"), ("0ca", '-10|"a"')]:
            self.c.run(f"d::.clid(c);d,[:kv {lit}]")
            self.assertEqual(self.q('(string type kv),"|",-3!kv'), '"' + shown.replace('"', '""') + '"', lit)

    def test_calls(self):
        c = self.c
        self.q("sq:{x*x}; add:{x+y}; nil:{42}")
        self.assertEqual(c.value("c(:sq,,7)"), "49")
        self.assertEqual(c.value("c(:add,3,4)"), "7")
        self.assertEqual(c.value("c([:nil])"), "42")
        self.assertEqual(c.value('c("til",,4)'), "[0 1 2 3]")
        self.assertEqual(c.value('c("sq",,5)'), "25")
        self.assertEqual(c.value("d::.clid(c);d,[:zz 42];d?:zz"), "42")
        # a q function is opaque here and can be sent back to be called
        self.assertEqual(c.value('qf::c("{x+100}");c(qf,,1)'), "101")

    def test_q_signals(self):
        # a bare q signal is not "<wire name>: <message>": kind remote, as in KlongPy
        e = self.native_error(self.c, 'c("1+`a")', "remote")
        self.assertEqual(e.message, "type")
        self.assertEqual(self.native_error(self.c, 'c("1 2 3+1 2")', "remote").message, "length")
        e = self.native_error(self.c, 'c("\'custom")', "remote")
        self.assertIn("custom", e.message)
        self.native_error(self.c, 'c("nosuchname")', "remote")
        self.assertEqual(self.q("1+1"), "2")

    def test_async(self):
        c = self.c
        c.run('got::[];cb::{got::got,x};ac::.async(c;cb);ac("til 3")')
        self.assertTrue(wait_until(lambda: c.value("got") == "[0 1 2]"))

    def test_large_vector(self):
        self.assertEqual(self.c.value('r::c("til 200000");(r~!200000),#r', 60), "[1 200000]")
        self.assertEqual(self.c.value("big::!200000;d::.clid(c);d,:bigv,,big;c(\"sum bigv\")", 60), str(sum(range(200000))))


@NEED_PEACHQ
class TestPeachQClientNativeServer(Case):
    def setUp(self):
        self.server, self.port = self.native_server()

    def run_q(self, body, timeout=40):
        script = f"h:hopen {self.port};\n" + Q_PRELUDE + body + "\n\\\\\n"
        code, out, err = run_q(script, timeout)
        return parse_results(out), out + err

    def test_requests(self):
        r, out = self.run_q('''
rep["sum";"1+1"];
rep["fn";"sq(7)"];
rep["call";(`sq;7)];
rep["call2";(`add;3;4)];
rep["callstr";("sq";7)];
rep["nil";(`nil;::)];
rep["lookup";`vint];
rep["data";1 2 3];
rep["define";"newfn::{x+100}"];
rep["use";"newfn(1)"];
rep["set";(`set;`gset;1 2 3)];
rep["got";`gset];
''')
        self.assertEqual(r["sum"], ("ok", -7, "2"), out)
        self.assertEqual(r["fn"], ("ok", -7, "49"), out)
        self.assertEqual(r["call"], ("ok", -7, "49"), out)
        self.assertEqual(r["call2"], ("ok", -7, "7"), out)
        self.assertEqual(r["callstr"], ("ok", -7, "49"), out)
        self.assertEqual(r["nil"], ("ok", -7, "42"), out)
        self.assertEqual(r["lookup"], ("ok", -7, "5"), out)
        self.assertEqual(r["data"], ("ok", 7, "1 2 3"), out)
        self.assertEqual(r["use"], ("ok", -7, "101"), out)
        self.assertEqual(r["got"], ("ok", 7, "1 2 3"), out)
        self.assertEqual(self.server.value("newfn(1),gset"), "[101 1 2 3]")

    def test_values_as_q_sees_them(self):
        shown = {"vint": "-7|5", "vfloat": "-9|1.5", "vchar": '-10|"a"', "vstr": '10|"abc"', "vempty": '10|""',
                 "vsym": "-11|`abc", "vivec": "7|1 2 3", "vfvec": "9|1.5 2.5", "vsvec": "11|`a`b",
                 "vmixed": '0|(1;`a;,"s")', "vstrs": '0|("ab";"cd")', "vnest": "0|(1 2;3 4)", "vrag": "0|(1 2;,3)",
                 "vlist0": "0|()", "vdict": "99|`a`b!1 2", "vbig": "-7|0W"}
        r, out = self.run_q("\n".join(f'rep["{n}";`{n}];' for n in shown))
        for n, want in shown.items():
            t, text = want.split("|", 1)
            self.assertEqual(r.get(n), ("ok", int(t), text), f"{n}: {out}")

    def test_errors_are_q_signals_with_the_wire_name(self):
        r, out = self.run_q('''
rep["type";"1+\\"a\\""];
rep["name call";(`nosuchfn;1)];
rep["parse";"1+"];
rep["after";"1+1"];
''')
        self.assertEqual(r["type"][0], "err", out)
        self.assertTrue(r["type"][1].startswith("type: cannot apply + (Plus) to integer and string"), r["type"])
        self.assertEqual(r["name call"], ("err", "name: symbol not found: nosuchfn"), out)
        self.assertEqual(r["parse"][0], "err", out)
        self.assertTrue(r["parse"][1].startswith("parse: "), r["parse"])
        self.assertEqual(r["after"], ("ok", -7, "2"), out)

    def test_async_messages_keep_their_order(self):
        r, out = self.run_q('''
asy["ga::5"];
asy["ga::ga+1"];
asy[(`set;`gb;`abc)];
asy["1+\\"a\\""];
rep["order";"ga"];
''')
        self.assertEqual(r["order"], ("ok", -7, "6"), out)
        self.assertEqual(self.server.value("gb"), ":abc")

    def test_large_vector(self):
        r, out = self.run_q('rep["big";"#!200000"];\nrep["sum";(`idf;til 200000)~til 200000];\nv:h(`idf;til 200000);\n'
                            '-1 "R|same|ok|0|",string v~til 200000;')
        self.assertEqual(r["big"], ("ok", -7, "200000"), out)
        self.assertEqual(r["same"], ("ok", 0, "1"), out)


if __name__ == "__main__":
    unittest.main()
