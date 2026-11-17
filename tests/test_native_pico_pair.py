"""Two Pico bench nodes as host processes behind a switch: the boot self-test with the firmware's own allocator.

`native/klong-ffi/tests/pico_node.c` is one node: the runtime of klong-ffi with the lane and the self-test, the firmware's
allocator (`native/boards/pico-bench/klong_heap.h`) over a model of newlib's malloc with an arena of a given size, and the
firmware's heap limits. This file is the switch and the supervisor: it routes the 16-byte frames of two such nodes (2 and 3: the
higher one runs the self-test as a client of the lower one, as on the boards), pings them, reads the reports they send to
address 16 on channel 3 and types console lines at them on channel 1 (the way the bridge does).

What the first image that failed on the boards did is reproduced with `old`: the sizes counted as asked, the limit 4% under
the arena, and a NULL from malloc a `panic("Out of memory")` of the SDK's pico_malloc, which spins: the node goes quiet in the
self-test's management step. The fixed policy counts every block as malloc holds it and takes memory from `__real_malloc`, so
that running out is a memory error: over a range of arena sizes it must never go quiet, and when it fits the test passes and
the console works after it.
"""
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.join(os.path.dirname(HERE), "native")
sys.path.insert(0, os.path.join(NATIVE, "boards", "pico-bench", "tools"))
import lane_frames as lf  # noqa: E402

BIG = 4 * 1024 * 1024
_binary = None


def build_node():
    global _binary
    if _binary:
        return _binary
    if not (shutil.which("cargo") and shutil.which("cc")):
        raise unittest.SkipTest("needs cargo and a C compiler")
    lib = os.path.join(NATIVE, "target", "pico-pair", "release", "libklong_ffi.a")
    r = subprocess.run(["cargo", "build", "--release", "-p", "klong-ffi", "--features", "lane", "--target-dir", "target/pico-pair"], cwd=NATIVE, capture_output=True, text=True)
    if r.returncode:
        raise unittest.SkipTest("cannot build klong-ffi with the lane: " + r.stderr[-300:])
    out = os.path.join(tempfile.gettempdir(), "kg_pico_node_%d" % os.getpid())
    r = subprocess.run(["cc", "-std=c11", "-O1", "-w", "-DKLONG_STACK_BYTES=1u", "-DKH_TESTED_ARENA=%du" % BIG, "-I" + os.path.join(NATIVE, "klong-ffi", "include"),
                        "-I" + os.path.join(NATIVE, "boards", "pico-bench"), "-I" + os.path.join(NATIVE, "klong-ffi", "tests"),
                        os.path.join(NATIVE, "klong-ffi", "tests", "pico_node.c"), lib, "-o", out], capture_output=True, text=True)
    if r.returncode:
        raise unittest.SkipTest("cannot compile pico_node.c: " + r.stderr[-300:])
    _binary = out
    return out


class Node:
    def __init__(self, binary, arena, old, note=None):
        r_rx, w_rx = os.pipe()   # switch -> node
        r_tx, w_tx = os.pipe()   # node -> switch
        env = dict(os.environ)
        if note:
            env["PICO_NOTE"] = note
        args = [binary, str(arena)] + (["old"] if old else [])
        self.errfile = tempfile.TemporaryFile()  # a file: a dump of the heap is longer than a pipe holds
        self.p = subprocess.Popen(args, pass_fds=(3, 4), stderr=self.errfile, env=env,
                                  preexec_fn=lambda: (os.dup2(r_rx, 3), os.dup2(w_tx, 4)))
        os.close(r_rx)
        os.close(w_tx)
        self.rx_w, self.tx_r = w_rx, r_tx
        os.set_blocking(self.tx_r, False)
        self.buf = b""
        self.status = []     # (t, flags)
        self.last_status = 0.0

    def send(self, data):
        try:
            os.write(self.rx_w, data)
        except OSError:
            pass

    def pump(self):
        try:
            d = os.read(self.tx_r, 65536)
        except BlockingIOError:
            return []
        except OSError:
            return []
        self.buf += d
        out, i = [], 0
        while i + 16 <= len(self.buf):
            f = lf.decode(self.buf[i:i + 16])
            if f:
                out.append(f)
                i += 16
            else:
                i += 1
        self.buf = self.buf[i:]
        return out

    def alive(self):
        return self.p.poll() is None

    def stop(self):
        if self.p.poll() is None:
            self.p.kill()
        self.p.wait()
        self.errfile.seek(0)
        err = self.errfile.read().decode("utf-8", "replace")
        self.errfile.close()
        for fd in (self.rx_w, self.tx_r):
            try:
                os.close(fd)
            except OSError:
                pass
        return err


class Bench:
    """The switch and the supervisor of two nodes."""

    def __init__(self, arena, old=False, notes=None):
        binary = build_node()
        self.nodes = {2: Node(binary, arena, old), 3: Node(binary, arena, old, (notes or {}).get(3))}
        self.t0 = time.time()
        self.reports = []   # (t, src, text) on channel 3
        self.console = {}   # (src, chan) -> [text]
        self.re = lf.Reassembler()
        self.seq = 0
        self.mid = 100
        self.next_ping = 0.0
        self.silent = {}

    def now(self):
        return time.time() - self.t0

    def spin(self, seconds, until=None):
        end = time.time() + seconds
        while time.time() < end:
            t = self.now()
            if t >= self.next_ping:
                self.next_ping = t + 0.5
                for n, node in self.nodes.items():
                    self.seq += 1
                    node.send(lf.frame(lf.PING, 16, n, self.seq, 0))
            rd = [n.tx_r for n in self.nodes.values()]
            select.select(rd, [], [], 0.02)
            for src, node in self.nodes.items():
                for f in node.pump():
                    if f["type"] == lf.STATUS and f["dst"] == 16:
                        node.status.append((self.now(), f["data"][1] if len(f["data"]) > 1 else None))
                        node.last_status = self.now()
                    elif f["dst"] in self.nodes and f["type"] != lf.STATUS:
                        self.nodes[f["dst"]].send(lf.encode(f) if hasattr(lf, "encode") else raw_frame(f))
                    elif f["dst"] == 16:
                        m = self.re.feed(f)
                        if m:
                            s, chan, data = m
                            text = data.decode("utf-8", "replace")
                            if chan == 3:
                                self.reports.append((self.now(), s, text))
                            else:
                                self.console.setdefault((s, chan), []).append(text)
            if until and until():
                return True
        return False

    def last_report(self, node):
        r = [x for x in self.reports if x[1] == node]
        return r[-1][2] if r else None

    def type(self, node, text, timeout=20):
        """A console line to a node as the bridge sends it; its output and the `done` status."""
        self.mid += 1
        mid = self.mid
        text = text if text.endswith("\n") else text + "\n"  # an input datagram ends at a line end
        for fr in lf.message_frames(16, node, mid, 1, text.encode(), ack=True):
            self.nodes[node].send(fr)
        self.console[(node, 1)] = []
        self.console[(node, 2)] = []
        ok = self.spin(timeout, until=lambda: any(c.startswith("done %d " % mid) for c in self.console.get((node, 2), [])))
        self.spin(0.3)  # the output datagrams of a line can come after its `done`
        out = "".join(self.console.get((node, 1), []))
        done = [c for c in self.console.get((node, 2), []) if c.startswith("done %d " % mid)]
        return out.strip(), (done[0].split()[-1] if done else None) if ok else None

    def close(self):
        return {n: node.stop() for n, node in self.nodes.items()}


def raw_frame(f):
    return lf.frame(f["type"], f["src"], f["dst"], f["seq"], f["sub"], f["data"])


def run_selftest(arena, old, seconds=40):
    """Run until node 3 has a verdict, or has gone quiet (no STATUS for 6 s: a node that spins in panic() answers no PING),
    or has exited; what the supervisor saw."""
    b = Bench(arena, old)
    try:
        def verdict():
            r = b.last_report(3)
            return r is not None and (r.startswith("klong selftest pass") or r.startswith("klong selftest fail"))

        def quiet():
            n = b.nodes[3]
            return bool(n.status) and b.now() - n.last_status > 6.0

        b.spin(seconds, until=lambda: verdict() or quiet() or not b.nodes[3].alive())
        res = {"report": b.last_report(3), "alive": b.nodes[3].alive(), "quiet": quiet(), "t": b.now(), "after": None}
        if not res["quiet"] and res["report"] and res["report"].startswith("klong selftest pass"):
            res["after"] = [b.type(3, "1+1"), b.type(3, "f::{:[x;1+f(x-1);0]};f(40)"), b.type(3, ".mem()"), b.type(3, "#!500")]
    finally:
        errs = b.close()
    res["stderr"] = errs
    return res


MGMT_NAMES = None


def mgmt_drop_lines():
    """What the self-test does to drop the management library: its definitions set to 0, twelve to a line."""
    global MGMT_NAMES
    if MGMT_NAMES is None:
        import re
        with open(os.path.join(os.path.dirname(HERE), "klongpy", "lib", "mgmt.kg")) as f:
            MGMT_NAMES = [m.group(1) for line in f for m in [re.match(r"([A-Za-z][A-Za-z0-9._]*)::", line)] if m]
    return [";".join(n + "::0" for n in MGMT_NAMES[i:i + 12]) for i in range(0, len(MGMT_NAMES), 12)]


def largest(b, node, hi=400000):
    """The largest `#!n` the node makes, by bisection through its console."""
    lo = 0
    while hi - lo > 128:
        mid = (lo + hi) // 2
        out, _ = b.type(node, "#!%d" % mid, timeout=15)
        lo, hi = (mid, hi) if out.strip() == str(mid) else (lo, mid)
    return lo


def counted(b, node):
    out, _ = b.type(node, ".mem()")
    return int(out.strip("[]").split()[0])


class TestPair(unittest.TestCase):
    def test_the_fixed_policy_with_room_passes_and_the_console_works_after(self):
        r = run_selftest(BIG, False)
        self.assertTrue(r["alive"], r)
        self.assertTrue(r["report"].startswith("klong selftest pass"), r)
        self.assertEqual(r["after"][0], ("2", "ok"), r)
        self.assertEqual(r["after"][1], ("40", "ok"), r)
        self.assertEqual(r["after"][3], ("500", "ok"), r)

    def test_the_image_that_failed_on_the_boards_goes_quiet_in_the_management_step(self):
        # sizes counted as asked, the limit 4% under the arena, a NULL from malloc the SDK's panic: the arena of the host is
        # larger than the board's (64-bit values, a 278 KB peak for the management library), the policy is the same
        r = run_selftest(240 * 1024, True)
        self.assertTrue(r["quiet"], r)
        self.assertIn("PANIC Out of memory", r["stderr"][3], r)
        self.assertFalse((r["report"] or "").startswith("klong selftest pass"), r)

    def test_the_fixed_policy_never_goes_quiet_whatever_the_arena(self):
        # from an arena too small to start in to one with room: a pass, a failure that says memory, or a node that
        # cannot connect (serving): never a node that stops answering, never a panic, never the abort of a frame
        outcomes = set()
        for kb in (60, 90, 112, 143, 174, 240):
            r = run_selftest(kb * 1024, False, seconds=20)
            self.assertFalse(r["quiet"], (kb, r))
            self.assertTrue(r["alive"], (kb, r))
            for n, err in r["stderr"].items():
                self.assertNotIn("PANIC", err, (kb, r))
                self.assertNotIn("FATAL", err, (kb, r))
            outcomes.add((r["report"] or "none").split(" node=")[0][:34])
        self.assertIn("klong selftest pass", " ".join(outcomes), outcomes)

    def test_the_client_of_the_selftest_has_the_memory_of_the_server_and_the_library_can_be_loaded_and_dropped(self):
        # The nodes of the boards measured on 2026-11-17: the server's largest vector 7,390 elements, the client's 4,000 to 4,300 straight after
        # its self-test; one connect and close recovered 8 KB; and `.l("mgmt")` on the client failed with a memory error. Three causes: a closed
        # connection kept by the table of handles until the next one pruned it (its 4 KB buffer), the symbols and tables that grow while the
        # library loads and pin the middle of malloc's arena, and a library that could not be loaded a second time. The arena is the
        # host's (the management library takes a megabyte on the 64-bit host, 45 KB on the board): the proportions are what is asserted.
        b = Bench(1200 * 1024, False)
        try:
            b.spin(40, until=lambda: (b.last_report(3) or "").startswith("klong selftest pass"))
            self.assertTrue((b.last_report(3) or "").startswith("klong selftest pass"), b.reports[-3:])
            b.spin(1)
            l2, l3 = largest(b, 2), largest(b, 3)
            self.assertGreaterEqual(l3, 0.985 * l2, (l2, l3))
            # a connection closed is gone at once: the memory after one cycle is the memory after the next
            b.type(3, 'h::.cli("node:2");h("1+1");.clic(h);h::0')
            m1 = counted(b, 3)
            b.type(3, 'h::.cli("node:2");h("1+1");.clic(h);h::0')
            m2 = counted(b, 3)
            self.assertLess(abs(m2 - m1), 400, (m1, m2))
            # loaded: it works, and what is left is what the library leaves
            out, st = b.type(3, '.l("mgmt")', timeout=60)
            self.assertEqual(st, "ok", out)
            with_lib = largest(b, 3)
            self.assertGreater(with_lib, 0.5 * l2, (l2, with_lib))
            self.assertLess(with_lib, l3, (l3, with_lib))
            # dropped: the memory of before, and the library again
            for line in mgmt_drop_lines():
                self.assertEqual(b.type(3, line)[1], "ok")
            dropped = largest(b, 3)
            self.assertGreaterEqual(dropped, 0.985 * l3, (l3, dropped))
            self.assertEqual(b.type(3, '.l("mgmt")', timeout=60)[1], "ok")
            for line in mgmt_drop_lines():
                self.assertEqual(b.type(3, line)[1], "ok")
            self.assertGreaterEqual(largest(b, 3), 0.985 * l3)
        finally:
            b.close()

    def test_a_node_that_restarted_says_why_in_its_reports(self):
        b = Bench(BIG, False, notes={3: "hardfault pc=10002d04 lr=10002d01"})
        try:
            b.spin(25, until=lambda: (b.last_report(3) or "").startswith("klong selftest pass"))
            self.assertIn("boot=hardfault pc=10002d04 lr=10002d01", b.last_report(3) or "", b.reports[-3:])
        finally:
            b.close()


if __name__ == "__main__":
    unittest.main()
