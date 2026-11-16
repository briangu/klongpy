"""
The serial console of the native runtime, end to end on the host (ticket 104): a simulated
board (`native/target/release/klong-sim --pty`) behind a pseudo-terminal, driven by the upload
tool (`native/tools/kgupload.py`) exactly as a board on a serial port is.

    cd native && cargo build --release -p klong-sim
    .venv/bin/python -m unittest discover -s native/tests

Skipped unless klong-sim is built (or KLONG_SIM names it). The classes are also imported by
tests/test_native_conformance.py, so the repository's own test run includes them.
"""
import importlib.util
import os
import pty
import random
import select
import subprocess
import sys
import tempfile
import threading
import time
import tty
import unittest
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SIM = os.environ.get("KLONG_SIM") or os.path.join(ROOT, "native", "target", "release", "klong-sim")
TOOL = os.path.join(ROOT, "native", "tools", "kgupload.py")
BUILT = os.path.isfile(SIM) and os.access(SIM, os.X_OK)
TAGS = set(
    (os.environ.get("KLONG_NATIVE_TAGS")
     or "core,strings,dicts,unicode,sysio,float,string-fmt,width64,strict1,strict2,numpy-only,device").split(",")
)


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


kg = _load("kgupload", TOOL)


class SimBoard:
    """A klong-sim process and the device path of its pseudo-terminal."""

    def __init__(self, *args):
        self.proc = subprocess.Popen([SIM, "--pty", *args], stdout=subprocess.PIPE)
        line = self.proc.stdout.readline().decode()
        assert line.startswith("pty: "), line
        self.path = line[5:].strip()

    def close(self):
        self.proc.kill()
        self.proc.wait()
        self.proc.stdout.close()


class Link:
    """
    A serial line with faults: a second pseudo-terminal in front of the board's. What the tool
    sends passes through `damage(offset, data) -> data`, where `offset` counts the bytes sent
    since `arm()`; what the board sends passes unchanged.
    """

    def __init__(self, board_path):
        self.master, self.slave = pty.openpty()
        tty.setraw(self.slave)
        self.path = os.ttyname(self.slave)
        self.board = os.open(board_path, os.O_RDWR | os.O_NOCTTY)
        self.damage = None
        self.offset = 0
        self.stop = False
        self.thread = threading.Thread(target=self._relay, daemon=True)
        self.thread.start()

    def arm(self, damage):
        self.offset = 0
        self.damage = damage

    def _relay(self):
        while not self.stop:
            r, _, _ = select.select([self.master, self.board], [], [], 0.05)
            try:
                if self.master in r:
                    data = os.read(self.master, 4096)
                    if self.damage:
                        out = self.damage(self.offset, data)
                        self.offset += len(data)
                        data = out
                    if data:
                        os.write(self.board, data)
                if self.board in r:
                    os.write(self.master, os.read(self.board, 4096))
            except OSError:
                return

    def close(self):
        self.stop = True
        self.thread.join(1.0)
        for fd in (self.master, self.slave, self.board):
            try:
                os.close(fd)
            except OSError:
                pass


def cut(start, end):
    """Damage: the bytes [start, end) of the stream never arrive."""
    def damage(offset, data):
        lo, hi = max(start - offset, 0), min(end - offset, len(data))
        return data[:lo] + data[hi:] if lo < hi else data
    return damage


def flip(at, once=True):
    """Damage: one bit of the byte at `at` is turned over."""
    def damage(offset, data):
        i = at - offset
        if 0 <= i < len(data):
            return data[:i] + bytes([data[i] ^ 0x01]) + data[i + 1:]
        return data
    return damage


@unittest.skipUnless(BUILT, f"simulated board not built: {SIM}")
class TestCheckComplete(unittest.TestCase):
    """klong-console's port of check_complete against the cases of tests/test_repl_complete.py and against KlongPy itself."""

    def verdicts(self, texts):
        lines = "".join(t.encode("utf-8").hex() + "\n" for t in texts)
        p = subprocess.run([SIM, "--check-complete"], input=lines.encode(), capture_output=True, timeout=60)
        self.assertEqual(p.returncode, 0)
        out = p.stdout.decode().splitlines()
        self.assertEqual(len(out), len(texts))
        return out

    def test_shared_cases(self):
        sys.path.insert(0, os.path.join(ROOT, "tests"))
        try:
            cases = _load("test_repl_complete_cases", os.path.join(ROOT, "tests", "test_repl_complete.py")).CASES
        except Exception as e:  # KlongPy itself is not importable here
            self.skipTest(f"tests/test_repl_complete.py does not load: {e}")
        finally:
            sys.path.pop(0)
        self.assertGreater(len(cases), 100)
        got = self.verdicts([text for text, _ in cases])
        for (text, want), verdict in zip(cases, got):
            with self.subTest(text=text):
                self.assertEqual(verdict, want)

    def test_agrees_with_klongpy_on_noise(self):
        sys.path.insert(0, ROOT)
        try:
            from klongpy.repl_input import check_text
        except Exception as e:
            self.skipTest(f"klongpy does not import: {e}")
        finally:
            sys.path.pop(0)
        rng = random.Random(104)
        alphabet = list('()[]{}:"\\ ;\n+-0c.ab1X') + [':[', ':{', ':(', '.comment("X")', ':"', '""', 'é', '.comment("XX")', 'XX']
        texts = ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 16))) for _ in range(6000)]
        got = self.verdicts(texts)
        differ = [(t, v, check_text(t)) for t, v in zip(texts, got) if v != check_text(t)]
        self.assertFalse(differ, f"{len(differ)} of {len(texts)} differ, for example {differ[:5]}")


@unittest.skipUnless(BUILT, f"simulated board not built: {SIM}")
class TestUploadTool(unittest.TestCase):
    """The upload tool against a simulated board, through a pseudo-terminal."""

    def setUp(self):
        self.sims = []
        self.links = []
        self.ports = []

    def tearDown(self):
        for p in self.ports:
            p.close()
        for l in self.links:
            l.close()
        for s in self.sims:
            s.close()

    def sim(self, *args):
        s = SimBoard(*args)
        self.sims.append(s)
        return s

    def board(self, path, **kw):
        port = kg.Port.open(path)
        self.ports.append(port)
        b = kg.Board(port, **{"timeout": 5.0, **kw})
        b.sync()
        return b

    def link(self, sim):
        l = Link(sim.path)
        self.links.append(l)
        return l

    def tool(self, sim, *args, data=None):
        p = subprocess.run([sys.executable, TOOL, sim.path, *args], input=data, capture_output=True, timeout=60)
        return p.returncode, p.stdout.decode(), p.stderr.decode()

    # ----- the library ------------------------------------------------------------------------

    def test_version(self):
        b = self.board(self.sim().path)
        v = b.ver()
        self.assertEqual((v["ver"], v["platform"], v["fs"]), ("0.0.1", "sim", "1"))
        self.assertIn(v["int"], ("32", "64"))
        self.assertIn("files", v["features"].split(","))
        self.assertGreater(int(v["heap_used"]), 0)
        # a board with a heap limit says how much is free
        v = self.board(self.sim("--heap-limit", "1M").path).ver()
        self.assertTrue(900_000 < int(v["heap_free"]) <= 1024 * 1024, v)

    def test_run(self):
        b = self.board(self.sim().path)
        self.assertEqual(b.run("+/!100"), ("4950", ""))
        self.assertEqual(b.run('.d("ab");.p("c");[1 2 3]'), ("[1 2 3]", "abc\n"))
        self.assertEqual(b.run(""), (None, ""))
        self.assertEqual(b.run('.p("only output")'), ('"only output"', "only output\n"))
        self.assertEqual(b.run('"two\nlines"'), ('"two\nlines"', ""))
        self.assertEqual(b.run("f::{x*2}\nf(21)"), ("42", ""))
        self.assertEqual(b.run("f(4)"), ("8", ""))
        with self.assertRaises(kg.BoardError) as e:
            b.run('.p("before");[1 2 3]@5')
        self.assertEqual((e.exception.kind, e.exception.output), ("range", "before\n"))
        with self.assertRaises(kg.BoardError) as e:
            b.run("(1]")
        self.assertEqual(e.exception.kind, "syntax")
        with self.assertRaises(kg.BoardError) as e:
            b.run("1+1;" * 20000)
        self.assertEqual(e.exception.kind, "range")
        self.assertEqual(b.run("héllo::\"wörld\";#héllo"), ("5", ""))

    def test_upload_text_and_load_it(self):
        b = self.board(self.sim().path)
        self.assertEqual(b.put(b"sq::{x*x}\ncube::{x*sq(x)}\n", "lib.kg"), "")
        self.assertEqual(b.run('.l("lib.kg");cube(3)'), ("27", ""))
        # a syntax error is a remark, and the file is there
        remark = b.put(b"f::{x+\n", "bad.kg")
        self.assertTrue(remark.startswith("warn syntax: "), remark)
        with self.assertRaises(kg.BoardError) as e:
            b.run('.l("bad.kg")')
        self.assertEqual(e.exception.kind, "syntax")
        # the same through Intel HEX
        self.assertEqual(b.hex(b"k::7\n", "k.kg"), "")
        self.assertEqual(b.run('.l("k.kg");k'), ("7", ""))
        # refusals are the board's words, and are not retried
        with self.assertRaises(kg.BoardError) as e:
            b.put(b"x", "has blank.kg")
        self.assertEqual(e.exception.kind, "syntax")
        b = self.board(self.sim("--no-fs").path)
        for send in (b.put, b.hex):
            with self.assertRaises(kg.BoardError) as e:
                send(b"x", "f")
            self.assertEqual((e.exception.kind, e.exception.message), ("system", "this board has no file system"))
        self.assertEqual(b.run("1+1"), ("2", ""))
        b = self.board(self.sim("--max-file", "1K", "--fs-bytes", "600").path)
        with self.assertRaises(kg.BoardError) as e:
            b.put(bytes(2000), "f")
        self.assertEqual(e.exception.kind, "range")
        with self.assertRaises(kg.BoardError) as e:
            b.put(bytes(1000), "f")
        self.assertEqual((e.exception.kind, e.exception.message), ("system", "no space left"))
        self.assertEqual(b.run("1+1"), ("2", ""))

    def test_a_megabyte_of_binary_reads_back_identical(self):
        b = self.board(self.sim().path)
        rng = random.Random(1)
        data = bytes(rng.getrandbits(8) for _ in range(1024 * 1024 + 3))
        # the board answers the end-of-file record with the length and CRC-32 of what it reads
        # back from its file system; hex() compares them with the file
        self.assertEqual(b.hex(data, "big.bin"), "")
        self.assertEqual(b.put(data, "big2.bin"), "")
        # and Klong can open them
        self.assertEqual(b.run('.cc(.ic("big.bin"));.cc(.ic("big2.bin"));1'), ("1", ""))
        # larger records, a smaller window
        b.record, b.window = 255, 4
        self.assertEqual(b.hex(data[:300_000], "big3.bin"), "")

    def test_reset_gives_a_fresh_interpreter(self):
        b = self.board(self.sim().path)
        b.run("a::42")
        b.put(b"x", "f.txt")
        self.assertEqual(b.run("a"), ("42", ""))
        v = b.reset()
        self.assertEqual(v["platform"], "sim")
        self.assertNotEqual(b.run("a"), ("42", ""))
        self.assertEqual(b.run("1+1"), ("2", ""))

    def test_crlf_board(self):
        b = self.board(self.sim("--crlf").path)
        self.assertEqual(b.run('.p("a");.p("b");"c\nd"'), ('"c\nd"', "a\nb\n"))
        self.assertEqual(b.hex(bytes(range(256)), "f.bin"), "")

    def test_heap_exhaustion_is_an_error_and_the_board_lives(self):
        b = self.board(self.sim("--heap-limit", "1M").path)
        for program in ("!100000000", "a::[];{1}{a::a,[[1 2 3]]}:~0"):
            with self.assertRaises(kg.BoardError) as e:
                b.run(program)
            self.assertEqual(e.exception.kind, "memory")
        self.assertEqual(b.run("a::0;+/!1000"), ("499500", ""))

    def test_a_small_stack_is_a_range_error(self):
        b = self.board(self.sim("--stack-budget", "32K").path)
        with self.assertRaises(kg.BoardError) as e:
            b.run("f::{:[x;f(x-1);0]};f(100000)")
        self.assertEqual(e.exception.kind, "range")
        self.assertEqual(b.run("f(5)"), ("0", ""))

    # ----- a terminal -------------------------------------------------------------------------

    def read_until(self, port, marker, timeout=5.0):
        end = time.monotonic() + timeout
        while marker not in port.buf:
            if time.monotonic() > end:
                self.fail(f"{marker!r} never came: {bytes(port.buf)!r}")
            port._fill(0.05)
        data = bytes(port.buf)
        port.buf.clear()
        return data

    def test_typing_at_the_prompt(self):
        s = self.sim()
        port = kg.Port.open(s.path)
        self.ports.append(port)
        self.assertEqual(self.read_until(port, b"klong> "), b"Klong 0.0.1 (sim, int64 float64)\nklong> ")
        for key in b"1+3\x7f2\r":
            port.write(bytes([key]))
        self.assertEqual(self.read_until(port, b"klong> "), b"1+3\x08 \x082\n3\nklong> ")
        port.write(b"f::{x*\r")
        self.assertEqual(self.read_until(port, b"     > "), b"f::{x*\n     > ")
        port.write(b"2}\rf(4)\r")
        self.assertEqual(self.read_until(port, b"8\nklong> "), b"2}\n:monad\nklong> f(4)\n8\nklong> ")
        # Ctrl-A in the middle of a line is ordinary input
        port.write(b"1\x01ver\r")
        out = self.read_until(port, b"klong> ")
        self.assertNotIn(b"\x01ok", out)
        self.assertIn(b"syntax error", out)

    def test_ctrl_c_stops_a_runaway_loop(self):
        s = self.sim()
        port = kg.Port.open(s.path)
        self.ports.append(port)
        self.read_until(port, b"klong> ")
        port.write(b"{1}{x+1}:~0\r")
        time.sleep(0.3)
        port.write(b"2+2\r")
        time.sleep(0.1)
        self.assertNotIn(b"klong> ", port.drain())
        port.write(b"\x03")
        out = self.read_until(port, b"klong> ")
        self.assertIn(b"interrupt error", out)
        # what was typed ahead before the Ctrl-C is discarded (docs/native/console.md); what comes after is run
        time.sleep(0.2)
        self.assertNotIn(b"4\n", port.drain())
        port.write(b"2+2\r")
        out = self.read_until(port, b"4\nklong> ")
        # the board is as alive as before
        self.assertEqual(kg.Board(port, timeout=5.0).sync()["platform"], "sim")

    def test_a_stalled_transfer_times_out_on_the_board(self):
        s = self.sim("--timeout-ms", "300")
        port = kg.Port.open(s.path)
        self.ports.append(port)
        self.read_until(port, b"klong> ")
        port.write(b"\x01put f.bin 100 00000000\n")
        self.read_until(port, b"\x01go\n")
        port.write(b"only some of it")
        start = time.monotonic()
        out = self.read_until(port, b"klong> ")
        self.assertEqual(out, b"\x01err system: transfer timed out\nklong> ")
        self.assertGreater(time.monotonic() - start, 0.25)
        b = kg.Board(port, timeout=5.0)
        with self.assertRaises(kg.BoardError) as e:
            b.run('.ic("f.bin")')
        self.assertEqual(e.exception.kind, "system")
        with self.assertRaises(kg.BoardError):
            b.run('.ic("f.bin.tmp")')

    # ----- a line with faults -----------------------------------------------------------------

    def test_hex_a_corrupted_record_is_sent_again(self):
        s = self.sim()
        l = self.link(s)
        b = self.board(l.path)
        data = bytes(range(256)) * 4
        log = []
        b.log = type("Log", (), {"write": lambda self, t: log.append(t)})()
        # the stream after arming: "\x01hex f.bin\n" (11 bytes), an address record (16), then
        # records of 44 bytes; one bit of the fourth data record is turned over
        l.arm(flip(11 + 16 + 3 * 44 + 20))
        self.assertEqual(b.hex(data, "f.bin"), "")
        text = "".join(log)
        self.assertIn("refused: checksum", text)
        self.assertIn("sending 1 records again", text)
        self.assertEqual(b.run('.cc(.ic("f.bin"));1'), ("1", ""))

    def test_hex_a_lost_record_is_noticed(self):
        s = self.sim()
        l = self.link(s)
        b = self.board(l.path)
        data = bytes(range(256)) * 4
        log = []
        b.log = type("Log", (), {"write": lambda self, t: log.append(t)})()
        # the third data record never arrives: one answer is missing, everything is sent again
        l.arm(cut(11 + 16 + 2 * 44, 11 + 16 + 3 * 44))
        self.assertEqual(b.hex(data, "f.bin"), "")
        self.assertIn("sending all records again", "".join(log))
        # half a record lost: it runs into the next one and both are refused or lost
        l.arm(cut(11 + 16 + 5 * 44 + 10, 11 + 16 + 5 * 44 + 30))
        self.assertEqual(b.hex(data, "g.bin"), "")

    def test_put_is_retried_after_damage(self):
        s = self.sim("--timeout-ms", "400")
        l = self.link(s)
        if True:
            b = self.board(l.path, board_timeout=0.4)
            data = b"sq::{x*x}\n" * 50
            log = []
            b.log = type("Log", (), {"write": lambda self, t: log.append(t)})()
            header = len(f"\x01put sq.kg {len(data)} {zlib.crc32(data):08x}\n")
            # a byte of the data changed: the board's check sum does not agree
            l.arm(flip(header + 100))
            self.assertEqual(b.put(data, "sq.kg"), "")
            self.assertIn("check sum", "".join(log))
            self.assertEqual(b.run('.l("sq.kg");sq(5)'), ("25", ""))
            # bytes of the data lost: the board waits, gives up, and the tool starts again
            del log[:]
            l.arm(cut(header + 10, header + 20))
            self.assertEqual(b.put(data, "sq2.kg"), "")
            self.assertIn("retry 1", "".join(log))
            # the command line itself damaged
            l.arm(flip(3))
            self.assertEqual(b.put(data, "sq3.kg"), "")
            self.assertEqual(b.run('.l("sq2.kg");.l("sq3.kg");sq(6)'), ("36", ""))
            # a line that damages everything: a clear failure after the retries
            l.arm(lambda offset, data: b"")
            b.retries = 1
            b.timeout = 0.5
            with self.assertRaises(kg.ProtocolError) as e:
                b.put(data, "never.kg")
            self.assertIn("does not answer", str(e.exception))

    # ----- the command line -------------------------------------------------------------------

    def test_command_line(self):
        s = self.sim()
        code, out, err = self.tool(s, "ver")
        self.assertEqual(code, 0, err)
        self.assertIn("ver=0.0.1", out)
        self.assertIn("platform=sim", out)
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "area.kg")
            with open(src, "w") as f:
                f.write("area::{x*y}\n")
            binary = os.path.join(d, "table.bin")
            with open(binary, "wb") as f:
                f.write(bytes(range(256)) * 20)
            program = os.path.join(d, "main.kg")
            with open(program, "w") as f:
                f.write('.l("area.kg")\n.p("computing")\narea(6;7)\n')
            code, out, err = self.tool(s, "upload", src)
            self.assertEqual((code, err), (0, ""))
            self.assertRegex(out, r"^area\.kg: 12 bytes, crc32 [0-9a-f]{8}, put, ")
            code, out, err = self.tool(s, "upload", binary, "data/table.bin")
            self.assertEqual((code, err), (0, ""))
            self.assertRegex(out, r"^data/table\.bin: 5120 bytes, crc32 [0-9a-f]{8}, hex, ")
            code, out, err = self.tool(s, "upload", "--hex", src, "area2.kg")
            self.assertEqual(code, 0, err)
            self.assertIn(", hex, ", out)
            code, out, err = self.tool(s, "upload", "--raw", binary)
            self.assertEqual(code, 0, err)
            self.assertIn(", put, ", out)
            code, out, err = self.tool(s, "run", program)
            self.assertEqual((code, out, err), (0, "computing\n42\n", ""))
            code, out, err = self.tool(s, "upload", os.path.join(d, "missing.kg"))
            self.assertEqual(code, 2)
        code, out, err = self.tool(s, "run", "-e", '.p("x");[1 2 3]@9')
        self.assertEqual((code, out), (1, "x\n"))
        self.assertTrue(err.startswith("error: range: "), err)
        code, out, err = self.tool(s, "run", "-e", "a::7;a*a")
        self.assertEqual((code, out), (0, "49\n"))
        code, out, err = self.tool(s, "-v", "reset")
        self.assertEqual(code, 0, err)
        self.assertIn("-> run 5 ", err)
        code, out, err = self.tool(s, "run", "-e", "a")
        self.assertNotEqual(out, "49\n")
        # no board there
        p = subprocess.run([sys.executable, TOOL, "/dev/does-not-exist", "ver"], capture_output=True, timeout=30)
        self.assertEqual(p.returncode, 3)
        self.assertIn(b"cannot open", p.stderr)

    def test_options_come_before_or_after_the_subcommand(self):
        ap = kg.build_parser()
        before = ap.parse_args(["PORT", "--chunk", "64", "--chunk-delay", "2", "-v", "upload", "f.kg", "r.kg"])
        after = ap.parse_args(["PORT", "upload", "--chunk", "64", "--chunk-delay", "2", "-v", "f.kg", "r.kg"])
        last = ap.parse_args(["PORT", "upload", "f.kg", "r.kg", "--chunk", "64", "--chunk-delay", "2", "-v"])
        self.assertEqual(vars(before), vars(after))
        self.assertEqual(vars(before), vars(last))
        self.assertEqual((before.chunk, before.chunk_delay, before.verbose, before.window), (64, 2.0, True, 32))
        # a value given before the subcommand is not overwritten by the subcommand's default
        both = ap.parse_args(["PORT", "--timeout", "3", "run", "-e", "1", "--retries", "5"])
        self.assertEqual((both.timeout, both.retries, both.run_timeout, both.expr), (3.0, 5, 60.0, "1"))
        for cmd in ("ver", "reset"):
            self.assertEqual(ap.parse_args(["PORT", cmd, "--baud", "9600"]).baud, 9600)
        # defaults are unchanged
        d = ap.parse_args(["PORT", "ver"])
        self.assertEqual((d.baud, d.timeout, d.chunk, d.chunk_delay, d.record, d.window, d.verbose), (None, 10.0, 1024, 0.0, 16, 32, False))
        # and it works end to end, with the options after the subcommand
        s = self.sim()
        with tempfile.TemporaryDirectory() as t:
            src = os.path.join(t, "big.kg")
            with open(src, "w") as f:
                f.write("x::7\n" * 100)
            code, out, err = self.tool(s, "upload", "--chunk", "100", "--chunk-delay", "1", src)
            self.assertEqual((code, err), (0, ""))
            self.assertRegex(out, r"^big\.kg: 500 bytes, ")
        code, out, err = self.tool(s, "ver", "--timeout", "20")
        self.assertEqual(code, 0, err)

    def test_a_board_that_does_not_answer(self):
        # a pseudo-terminal with nobody behind it
        master, slave = pty.openpty()
        tty.setraw(slave)
        try:
            p = subprocess.run([sys.executable, TOOL, os.ttyname(slave), "--timeout", "0.3", "--board-timeout", "0.3", "ver"], capture_output=True, timeout=60,
                               env={**os.environ, "PYTHONPATH": ""})
            self.assertEqual(p.returncode, 3)
            self.assertIn(b"does not answer", p.stderr)
        finally:
            os.close(master)
            os.close(slave)


@unittest.skipUnless(BUILT, f"simulated board not built: {SIM}")
class TestConformanceOverConsole(unittest.TestCase):
    """Every conformance case through `^A run` on a simulated board gives the result the suite expects."""

    def test_cases(self):
        runner = _load("console_conformance", os.path.join(ROOT, "native", "scripts", "console_conformance.py"))
        conf = _load("conformance_run", os.path.join(ROOT, "conformance", "run.py"))
        want = sum(len(conf.selected(conf.load_file(p), TAGS)) for p in conf.case_files(conf.CASES_DIR))
        passed, total, failures, _ = runner.run(TAGS, sim=SIM, show=lambda *_: None)
        self.assertEqual(total, want)
        self.assertGreater(total, 1000)
        self.assertFalse(failures, f"{len(failures)} of {total} cases failed through the console:\n" + "\n".join(failures[:40]))
        self.assertEqual(passed, total)


if __name__ == "__main__":
    unittest.main()
