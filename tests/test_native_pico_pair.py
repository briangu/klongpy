"""Two Pico bench nodes as host processes behind a switch: the boot self-test with the firmware's own allocator.

`native/klong-ffi/tests/pico_node.c` is one node: the runtime of klong-ffi with the lane and the self-test, the firmware's
allocator (`native/boards/pico-bench/klong_heap.h`: one pool, an address-ordered free list, no malloc) over a static arena of a
given size, the firmware's heap limits and its diagnostic, `.heap`. This file is the switch and the supervisor: it routes the
16-byte frames of two such nodes (2 and 3: the higher one runs the self-test as a client of the lower one, as on the boards), pings
them, reads the reports they send to address 16 on channel 3 and types console lines at them on channel 1 (the way the bridge does).

What the host can and cannot show. The runtime here is the 64-bit build: its objects are about twice the Pico's and the management
library takes several times what it takes on the board, so sizes mean nothing and every assertion below is a ratio: the largest vector
of a node against what its counted use leaves (`(limit - reserve - in use) / 8`, which is what a heap with no fragmentation gives), and
the client of the self-test against the server. The arena is chosen so that the library's peak is about the same share of it as on
the board (about 80%): that is what makes the allocator's placement matter. The previous image's allocator, over a model of newlib's
malloc, is built with `legacy=True` (`pico_node.c -DPICO_V1`): at this proportion it shows the direction of what the boards showed (the
client's largest vector below what its use allows; the management library refused after the self-test too, until it was loaded statement by statement) but not the size of it (the
boards lost 30%, the model 4 to 15%): the model is a check that the harness can see the problem, not a measurement of the board.
The real numbers come from `.heap` on the boards.
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
_binaries = {}


def build_node(legacy=False):
    """The node program: the firmware's allocator (`legacy`: an earlier one over a model of newlib's malloc)."""
    if legacy in _binaries:
        return _binaries[legacy]
    if not (shutil.which("cargo") and shutil.which("cc")):
        raise unittest.SkipTest("needs cargo and a C compiler")
    target = "pico-pair-legacy" if legacy else "pico-pair"
    feats = "lane" if legacy else "lane,heapdiag"
    lib = os.path.join(NATIVE, "target", target, "release", "libklong_ffi.a")
    r = subprocess.run(["cargo", "build", "--release", "-p", "klong-ffi", "--features", feats, "--target-dir", "target/" + target], cwd=NATIVE, capture_output=True, text=True)
    if r.returncode:
        raise unittest.SkipTest("cannot build klong-ffi with the lane: " + r.stderr[-300:])
    out = os.path.join(tempfile.gettempdir(), "kg_pico_node_%s_%d" % ("v1" if legacy else "v2", os.getpid()))
    cmd = ["cc", "-std=c11", "-O1", "-w", "-DKLONG_STACK_BYTES=1u", "-DKH_TESTED_ARENA=%du" % BIG, "-I" + os.path.join(NATIVE, "klong-ffi", "include"),
           "-I" + os.path.join(NATIVE, "boards", "pico-bench"), "-I" + os.path.join(NATIVE, "klong-ffi", "tests")]
    if legacy:
        cmd.append("-DPICO_V1")
    r = subprocess.run(cmd + [os.path.join(NATIVE, "klong-ffi", "tests", "pico_node.c"), lib, "-o", out], capture_output=True, text=True)
    if r.returncode:
        raise unittest.SkipTest("cannot compile pico_node.c: " + r.stderr[-300:])
    _binaries[legacy] = out
    return out


class Node:
    def __init__(self, binary, arena, note=None):
        r_rx, w_rx = os.pipe()   # switch -> node
        r_tx, w_tx = os.pipe()   # node -> switch
        env = dict(os.environ)
        if note:
            env["PICO_NOTE"] = note
        args = [binary, str(arena)]
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

    def __init__(self, arena, legacy=False, notes=None):
        binary = build_node(legacy)
        self.nodes = {2: Node(binary, arena), 3: Node(binary, arena, (notes or {}).get(3))}
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


def run_selftest(arena, legacy=False, seconds=40):
    """Run until node 3 has a verdict, or has gone quiet (no STATUS for 6 s: a node that spins in panic() answers no PING),
    or has exited; what the supervisor saw."""
    b = Bench(arena, legacy)
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




def mgmt_drop_lines():
    """What the self-test does to drop the management library: the module is released when its name is cleared."""
    return ["mgmt::0"]


def largest(b, node, hi=400000):
    """The largest `#!n` the node makes, by bisection through its console."""
    lo = 0
    while hi - lo > 64:
        mid = (lo + hi) // 2
        out, _ = b.type(node, "#!%d" % mid, timeout=15)
        lo, hi = (mid, hi) if out.strip() == str(mid) else (lo, mid)
    return lo


def mem(b, node):
    """(in use, peak, limit) as `.mem()` says."""
    out, _ = b.type(node, ".mem()")
    return tuple(int(x) for x in out.strip("[]").split())


def ideal(b, node, reserve=6144):
    """The largest `#!n` a heap with no fragmentation would give: what the counted use leaves, less the reserve, in elements (and the
    evaluation of `#!n` itself, a few hundred bytes: so this is the upper bound)."""
    used, _, limit = mem(b, node)
    return (limit - reserve - used) // 8


def heap(b, node, mode):
    out, _ = b.type(node, ".heap(%d)" % mode)
    return [int(x) for x in out.strip("[]").split()]


def heap_map(b, node):
    """The pool as runs [offset, size, kind (0 free, 1 used), kinds of the blocks that start in it], from `.heap(1..)`."""
    runs, page = [], 1
    while True:
        v = heap(b, node, page)
        if not v:
            return runs
        runs += [v[i:i + 4] for i in range(0, len(v), 4)]
        page += 1


class TestPair(unittest.TestCase):
    def test_with_room_it_passes_and_the_console_works_after(self):
        r = run_selftest(BIG)
        self.assertTrue(r["alive"], r)
        self.assertTrue(r["report"].startswith("klong selftest pass"), r)
        self.assertEqual(r["after"][0], ("2", "ok"), r)
        self.assertEqual(r["after"][1], ("40", "ok"), r)
        self.assertEqual(r["after"][3], ("500", "ok"), r)

    def test_it_never_goes_quiet_whatever_the_arena(self):
        # from an arena too small to start in to one with room: a pass, a failure that says memory, or a node that cannot connect
        # (serving): never a node that stops answering, never the abort of a frame (`fatal`)
        outcomes = set()
        for kb in (60, 90, 112, 143, 174, 240):
            r = run_selftest(kb * 1024)
            self.assertFalse(r["quiet"], (kb, r))
            self.assertTrue(r["alive"], (kb, r))
            for n, err in r["stderr"].items():
                self.assertNotIn("FATAL", err, (kb, r))
            outcomes.add((r["report"] or "none").split(" node=")[0][:34])
        self.assertIn("klong selftest pass", " ".join(outcomes), outcomes)

    def test_the_model_of_the_previous_allocator_shows_the_direction_of_what_the_boards_showed(self):
        # That firmware on the boards (2026-11-17): the server's largest vector 6,421 against 7,633 that its use of 22,232 bytes allowed
        # (84%), the client's 4,359 against 6,178 for 33,872 (71%), `.l("mgmt")` refused on the client after its self-test, and both
        # unchanged by later activity. The model of newlib's malloc under that allocator, with the library's peak the same share of the
        # arena, loses less (a few per cent, not a third) but in the same places: the client's below the server's.
        # This is the check that the harness can see the problem; the sizes are not the board's.
        b = Bench(400 * 1024, legacy=True)
        try:
            b.spin(40, until=lambda: (b.last_report(3) or "").startswith("klong selftest pass"))
            self.assertTrue((b.last_report(3) or "").startswith("klong selftest pass"), b.reports[-3:])
            b.spin(1)
            l2, l3 = largest(b, 2), largest(b, 3)
            self.assertLess(l3, 0.97 * l2, (l2, l3))
            self.assertLess(l3, 0.97 * ideal(b, 3), (l3, ideal(b, 3)))
            out, st = b.type(3, '.l("mgmt")', timeout=60)
            # (it was refused here and on the boards while a load held the whole file's tokens and tree at once; loaded statement by
            # statement it fits even this allocator: the refusal was the peak, the smaller largest vector was the placement)
            self.assertEqual(st, "ok", out)
        finally:
            b.close()

    def test_the_client_of_the_selftest_has_the_memory_its_use_allows_and_the_library_can_be_loaded_and_dropped(self):
        # What the diagnostic found in the model and the boards were meant to confirm: after the self-test no block is left in the
        # middle of the heap, so the largest vector of each node is what its counted use leaves; the client's counted use is more
        # than the server's (the symbols the library interned stay), and that is all the difference. The library loads on the client,
        # and dropped it leaves the same, twice; and the library loads on either node in the 300 KB arena too, where the whole file in memory at once
        # (the old load, 3.5 times what it keeps) did not.
        for arena in (400 * 1024, 300 * 1024):
            b = Bench(arena)
            try:
                b.spin(40, until=lambda: (b.last_report(3) or "").startswith("klong selftest pass"))
                self.assertTrue((b.last_report(3) or "").startswith("klong selftest pass"), b.reports[-3:])
                b.spin(1)
                # (first, before the largest vectors, whose own peak would hide it)
                # the library is loaded statement by statement: what the load held at once, above what was in use, is what it keeps and one
                # statement, not the tokens and the tree of the whole file (the whole file peaked at three times what stayed)
                base, _, _ = mem(b, 2)
                out, st = b.type(2, '.l("mgmt")', timeout=60)
                self.assertEqual(st, "ok", out)
                used, peak, _ = mem(b, 2)
                # (host figures, 64-bit: the whole file kept 91 KB and peaked at 297; the function literals are kept as the place of their text now)
                self.assertLess(used - base, 45000, (arena, base, used, peak))
                self.assertLess(peak - base, 70000, (arena, base, used, peak))
                for line in mgmt_drop_lines():
                    self.assertEqual(b.type(2, line)[1], "ok")
                for n in (2, 3):
                    got, want = largest(b, n), ideal(b, n)
                    self.assertGreaterEqual(got, 0.98 * want, (arena, n, got, want))  # (a kept tree of a few KB at the bottom costs up to 2%)
                    self.assertLessEqual(got, want + 300, (arena, n, got, want))  # (`ideal` reads `.mem()` while the evaluation of `.mem()` is in use)
                l3 = largest(b, 3)
                # a connection closed is gone at once: the memory after one cycle is the memory after the next
                b.type(3, 'h::.cli("node:2");h("1+1");.clic(h);h::0')
                m1 = mem(b, 3)[0]
                b.type(3, 'h::.cli("node:2");h("1+1");.clic(h);h::0')
                m2 = mem(b, 3)[0]
                self.assertLess(abs(m2 - m1), 400, (m1, m2))
                out, st = b.type(3, '.l("mgmt")', timeout=60)
                self.assertEqual(st, "ok", out)
                # and it works: the management node calls the other node through the functions that were kept as text
                self.assertEqual(b.type(3, 'mgmt.add("p";"node:2")', timeout=30)[1], "ok")
                self.assertEqual(b.type(3, 'mgmt.call("p";"1+1";[])', timeout=30), ("2", "ok"))
                b.type(3, '.clic((mgmt.nodes?"p")?"h")')  # (as the self-test does, before it drops the library)
                for line in mgmt_drop_lines():
                    self.assertEqual(b.type(3, line)[1], "ok")
                b.spin(1.0)  # (the node lets go of the functions it parsed 300 ms after it has nothing to do)
                dropped = largest(b, 3)
                self.assertGreaterEqual(dropped, 0.97 * l3, (l3, dropped))  # (the names the library interned stay, counted)
                self.assertEqual(b.type(3, '.l("mgmt")', timeout=60)[1], "ok")
                for line in mgmt_drop_lines():
                    self.assertEqual(b.type(3, line)[1], "ok")
                b.spin(1.0)
                self.assertGreaterEqual(largest(b, 3), 0.97 * l3)
                self.assertGreaterEqual(largest(b, 3), 0.98 * ideal(b, 3))  # (the kept trees at the bottom, a few KB, cost up to 2%)
            finally:
                b.close()

    def test_the_diagnostic_describes_the_pool_exactly(self):
        b = Bench(400 * 1024)
        try:
            b.spin(40, until=lambda: (b.last_report(3) or "").startswith("klong selftest pass"))
            b.spin(1)
            for n in (2, 3):
                s = heap(b, n, 0)
                used, peak, limit = mem(b, n)
                self.assertGreaterEqual(len(s), 21, s)
                base, size, taken, taken_peak, cached, drains, drained, free, largest_free, chunks, first_off, first_size, last_off, last_size, nlarge, large_bytes, lost, fail_small, fail_large, grown, missed = s[:21]
                runs = heap_map(b, n)
                # the map: runs in address order that tile the pool, free and used alternating
                self.assertEqual(runs[0][0], 0)
                for (o1, s1, k1, _), (o2, s2, k2, _) in zip(runs, runs[1:]):
                    self.assertEqual(o1 + s1, o2)
                    self.assertNotEqual(k1, k2)
                self.assertEqual(runs[-1][0] + runs[-1][1], size)
                frees = [r for r in runs if r[2] == 0]
                # (each `.heap` is an evaluation, which takes and gives back a few hundred bytes: the pages are not one instant, so the
                # figures of the summary and the sums of the map agree within that, and the summary agrees with itself exactly)
                self.assertLess(abs(sum(r[1] for r in frees) - free), 4096, (n, s))
                self.assertLess(abs(max(r[1] for r in frees) - largest_free), 4096)
                self.assertLessEqual(abs(len(frees) - chunks), 12)
                self.assertEqual(size - free, taken)
                # idle: only what the evaluation of `.heap` itself has cached, and every counted byte is in a block: what is taken is at least what is counted
                self.assertLessEqual(cached, 4096)
                self.assertGreaterEqual(taken, used - 4096)  # (the console's own evaluation of .mem is in `used`)
                self.assertGreaterEqual(taken_peak, taken)
                # the live blocks of 512 bytes or more: sorted by address, inside the pool, each kind a known one, and what they hold is in the map
                blocks, page = [], 100
                while True:
                    v = heap(b, n, page)
                    if not v:
                        break
                    blocks += [v[i:i + 3] for i in range(0, len(v), 3)]
                    page += 1
                self.assertEqual(len(blocks), nlarge, (n, s))
                self.assertEqual(sum(x[1] for x in blocks), large_bytes)
                self.assertEqual([x[0] for x in blocks], sorted(x[0] for x in blocks))
                for off, sz, tag in blocks:
                    self.assertTrue(0 <= tag <= 7 and off + sz <= size)
                    used_run = [r for r in runs if r[2] == 1 and r[0] <= off < r[0] + r[1]]
                    self.assertTrue(used_run and used_run[0][3] & (1 << tag), (n, off, sz, tag))
                self.assertEqual(len(heap(b, n, 200)), 16)  # the cache by size class
            # the client holds the library's symbols and the evaluator's tables, tagged: there are blocks of the globals' and symbols' kinds
            kinds = {t for (_, _, t) in [heap(b, 3, 100)[i:i + 3] for i in range(0, len(heap(b, 3, 100)), 3)]}
            self.assertTrue(kinds, kinds)
        finally:
            b.close()

    def test_a_node_with_the_library_loaded_and_called_keeps_its_largest_vector(self):
        # The boards (2026-11-17, node 2 with the library loaded, `mgmt.add` and calls from the bridge): 34.6 KB in use, a count that allowed
        # about 6,100 elements and a largest vector of 5,000, 58 KB free in 56 chunks, the largest 42 KB: a 2.7 KB table of the globals with 5.9 KB
        # free right above it, small holes in the top. The model, with the arena where the library is the board's share of it, loses less (about
        # 4 to 7%: the table was moved, the holes of the top zone are what they are) but in the same place; this keeps it from growing. The calls
        # are separated by more than the 300 ms after which the node lets go of the functions it parsed.
        b = Bench(200 * 1024)
        try:
            b.spin(40, until=lambda: (b.last_report(3) or "").startswith("klong selftest pass"))
            b.spin(1)
            self.assertEqual(b.type(2, '.l("mgmt")', timeout=60)[1], "ok")
            self.assertEqual(b.type(2, 'mgmt.add("p";"node:3")', timeout=30)[1], "ok")
            for _ in range(6):
                self.assertEqual(b.type(2, 'mgmt.call("p";"1+1";[])', timeout=30), ("2", "ok"))
                b.spin(0.5)
            got, want = largest(b, 2), ideal(b, 2)
            self.assertGreaterEqual(got, 0.93 * want, (got, want))
            s = heap(b, 2, 0)
            free, largest_free = s[7], s[8]
            self.assertGreater(largest_free, 0.8 * free, (free, largest_free))
        finally:
            b.close()

    def test_a_node_that_restarted_says_why_in_its_reports(self):
        b = Bench(BIG, notes={3: "hardfault pc=10002d04 lr=10002d01"})
        try:
            b.spin(25, until=lambda: (b.last_report(3) or "").startswith("klong selftest pass"))
            self.assertIn("boot=hardfault pc=10002d04 lr=10002d01", b.last_report(3) or "", b.reports[-3:])
        finally:
            b.close()


if __name__ == "__main__":
    unittest.main()
