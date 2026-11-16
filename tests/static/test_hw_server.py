"""
The card server and its client (ticket 111), without a card: the protocol, the buffer bookkeeping, the rules of
``load``, the client's handling of a dropped connection. The device is FakeDevice (numpy), in the same process
or in a server process started with ``--fake``. What reaches the real card is tests/static/test_hw_card.py.
"""
import ast
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import time
import unittest

import numpy as np

from klongpy.hw import client, gate, server
from klongpy.hw.client import HwConnectionError, HwError, HwRefused, HwSession, HwTrap
from klongpy.hw.server import FakeDevice, Server, ServerError, encode_frame, read_frame, serve

from .test_hw_build import FAIL_RPT, PASS_RPT

HW_DIR = os.path.dirname(os.path.abspath(server.__file__))

ENTRY_SUM = {"name": "total", "c_name": "total", "top": "t_total_kernel", "in_words": 4, "out_words": 1, "in_type": "kg_float",
             "out_type": "kg_float", "in_bytes": 16, "out_bytes": 4, "signature": "float[4] -> float",
             "params": [{"name": "x", "elem": "float", "shape": [4], "offset": 0, "words": 4}],
             "result": {"elem": "float", "shape": [], "words": 1}}
ENTRY_ACC = {"name": "acc", "c_name": "acc", "top": "t_acc_kernel", "in_words": 1, "out_words": 1, "in_type": "kg_float",
             "out_type": "kg_float", "in_bytes": 4, "out_bytes": 4, "signature": "float -> float",
             "params": [{"name": "x", "elem": "float", "shape": [], "offset": 0, "words": 1}],
             "result": {"elem": "float", "shape": [], "words": 1}}


def k_sum(words, n, reset, state):
    x = words[:4 * n].view(np.float32).reshape(n, 4)
    return x.sum(axis=1, dtype=np.float32), 0


def k_acc(words, n, reset, state):
    """A running sum kept between calls (state), cleared by reset; a negative input sets the trap bits."""
    if reset or "acc" not in state:
        state["acc"] = np.float32(0)
    x = words[:n].view(np.float32)
    out = np.empty(n, np.float32)
    status = 0
    for i in range(n):
        if x[i] < 0:
            status |= 3
        state["acc"] = np.float32(state["acc"] + x[i])
        out[i] = state["acc"]
    return out, status


def make_image(root, passing=True, name="t.xclbin"):
    """A build directory as the build flow leaves it (the gate does the writing): returns the path of the image."""
    d = tempfile.mkdtemp(prefix="kgimg_", dir=root)
    os.makedirs(os.path.join(d, "_x/reports/link/imp"))
    with open(os.path.join(d, "_x/reports/link/imp/impl_1_hw_bb_locked_timing_summary_routed.rpt"), "w") as f:
        f.write(PASS_RPT if passing else FAIL_RPT)
    with open(os.path.join(d, name), "wb") as f:
        f.write(b"not a bitstream")
    kern = os.path.join(d, "kernels.json")
    with open(kern, "w") as f:
        json.dump([{"entries": [ENTRY_SUM, ENTRY_ACC]}], f)
    gate.main([d, "--xclbin", name, "--key", "k", "--clock-mhz", "300", "--kernels", kern])
    produced = [n for n in os.listdir(d) if n.endswith(".xclbin")]
    return os.path.join(d, produced[0])


class Pipe:
    """An in-process connection to a Server: requests go in as frames, replies come back as (header, data)."""
    def __init__(self, srv):
        self.srv = srv
        self.next = 1

    def call(self, cmd, data=b"", **args):
        args.update(cmd=cmd, id=self.next)
        self.next += 1
        stdin = io.BytesIO(encode_frame(args, data))
        stdout = io.BytesIO()
        serve(stdin, stdout, self.srv)
        out = io.BytesIO(stdout.getvalue())
        hello = read_frame(out)
        reply = read_frame(out)
        return reply


class SyntaxTest(unittest.TestCase):
    def test_server_and_gate_are_python38_and_standard(self):
        for name in ("server.py", "gate.py"):
            with open(os.path.join(HW_DIR, name)) as f:
                src = f.read()
            tree = ast.parse(src, feature_version=(3, 8))
            mods = set()
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    mods.update(a.name.split(".")[0] for a in n.names)
                elif isinstance(n, ast.ImportFrom) and n.level == 0:
                    mods.add((n.module or "").split(".")[0])
            self.assertNotIn("klongpy", mods)
            self.assertNotIn("torch", mods)
            self.assertLessEqual(mods - set(sys.stdlib_module_names), {"numpy", "pyxrt", "gate"}, name)


class FramingTest(unittest.TestCase):
    def test_roundtrip_and_little_endian(self):
        raw = encode_frame({"cmd": "put", "n": 3}, b"\x01\x02\x03")
        (length,) = struct.unpack_from("<I", raw, 0)
        self.assertEqual(length, len(raw) - 4)
        h, d = read_frame(io.BytesIO(raw))
        self.assertEqual((h, d), ({"cmd": "put", "n": 3}, b"\x01\x02\x03"))
        self.assertEqual(read_frame(io.BytesIO(encode_frame({"a": 1})))[1], b"")

    def test_malformed(self):
        with self.assertRaises(EOFError):
            read_frame(io.BytesIO(b""))
        with self.assertRaises(EOFError):
            read_frame(io.BytesIO(struct.pack("<I", 100) + b"short"))
        with self.assertRaises(ServerError):
            read_frame(io.BytesIO(struct.pack("<I", 2) + b"xx"))
        with self.assertRaises(ServerError):
            read_frame(io.BytesIO(struct.pack("<II", 8, 100) + b"abcd"))
        with self.assertRaises(ServerError):
            read_frame(io.BytesIO(struct.pack("<II", 4 + 5, 5) + b"notjs"))
        with self.assertRaises(ServerError):
            read_frame(io.BytesIO(struct.pack("<II", 4 + 2, 2) + b"[]"))

    def test_a_bad_frame_gets_an_error_frame(self):
        srv = Server(FakeDevice({}))
        out = io.BytesIO()
        serve(io.BytesIO(struct.pack("<I", 2) + b"xx"), out, srv)
        o = io.BytesIO(out.getvalue())
        read_frame(o)
        h, _ = read_frame(o)
        self.assertEqual((h["ok"], h["kind"]), (False, "BadRequest"))

    def test_unknown_command_and_many_requests_in_one_stream(self):
        srv = Server(FakeDevice({}))
        stream = (encode_frame({"cmd": "nope", "id": 1}) + encode_frame({"cmd": "status", "id": 2}) +
                  encode_frame({"cmd": "bye", "id": 3}) + encode_frame({"cmd": "status", "id": 4}))
        out = io.BytesIO()
        self.assertEqual(serve(io.BytesIO(stream), out, srv), 3)
        o = io.BytesIO(out.getvalue())
        read_frame(o)
        r = [read_frame(o)[0] for _ in range(3)]
        self.assertEqual([x["ok"] for x in r], [False, True, True])
        self.assertEqual([x["id"] for x in r], [1, 2, 3])
        self.assertEqual(r[0]["kind"], "BadRequest")


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="kgsrv_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.dev = FakeDevice({"t_total_kernel": k_sum, "t_acc_kernel": k_acc})
        self.srv = Server(self.dev)
        self.p = Pipe(self.srv)
        self.image = make_image(self.tmp)

    def ok(self, cmd, data=b"", **a):
        h, d = self.p.call(cmd, data, **a)
        self.assertTrue(h["ok"], h)
        return h, d

    def err(self, cmd, data=b"", **a):
        h, d = self.p.call(cmd, data, **a)
        self.assertFalse(h["ok"], h)
        return h["kind"], h["message"]

    def test_nothing_before_load(self):
        self.assertEqual(self.err("put", b"abcd")[0], "NoImage")
        self.assertEqual(self.err("run", kernel="t_total_kernel", n=1, **{"in": 1})[0], "NoImage")

    def test_info_reports_the_card_and_what_is_loaded(self):
        h, _ = self.ok("info")
        self.assertEqual(h["uuid"], "00000000-0000-0000-0000-000000000000")
        self.assertEqual(h["kind"], "fake")
        self.assertIsNone(h["session_image"])

    def test_load_says_what_it_replaces(self):
        h, _ = self.ok("load", path=self.image)
        self.assertIn("replaces the image that was on the card", h["message"])
        self.assertEqual(h["replaced"]["uuid"], "00000000-0000-0000-0000-000000000000")
        self.assertEqual(h["kernels"], ["t_acc_kernel", "t_total_kernel"])
        self.assertEqual(self.dev.loaded_paths, [self.image])
        self.assertEqual(self.ok("info")[0]["session_image"]["path"], self.image)

    def test_load_refuses_timing_failed_images_and_missing_records(self):
        bad = make_image(self.tmp, passing=False)
        self.assertTrue(bad.endswith(".timing-failed.xclbin"))
        kind, msg = self.err("load", path=bad)
        self.assertEqual(kind, "ImageRefused")
        self.assertIn("failed routed timing", msg)
        # the same bits under a good name, without a build.json
        d = tempfile.mkdtemp(dir=self.tmp)
        shutil.copy(bad, os.path.join(d, "plain.xclbin"))
        self.assertEqual(self.err("load", path=os.path.join(d, "plain.xclbin"))[0], "ImageRefused")
        # the failed image renamed by hand, with the failed build.json next to it
        d2 = os.path.dirname(bad)
        shutil.copy(bad, os.path.join(d2, "renamed.xclbin"))
        self.assertEqual(self.err("load", path=os.path.join(d2, "renamed.xclbin"))[0], "ImageRefused")
        self.assertEqual(self.dev.loaded_paths, [])
        self.assertEqual(self.err("load", path=os.path.join(self.tmp, "absent.xclbin"))[0], "ImageRefused")

    def test_load_refuses_a_card_in_use(self):
        self.dev.busy_users = [(4242, "someone")]
        kind, msg = self.err("load", path=self.image)
        self.assertEqual(kind, "CardBusy")
        self.assertIn("someone (pid 4242)", msg)
        self.assertEqual(self.dev.loaded_paths, [])
        self.dev.busy_users = []
        self.dev.was = {"uuid": "u", "compute_units": ["a:a_1"], "busy": True}
        self.assertEqual(self.err("load", path=self.image)[0], "CardBusy")
        self.assertEqual(self.dev.loaded_paths, [])

    def test_put_get_free_bookkeeping(self):
        self.ok("load", path=self.image)
        a = np.arange(8, dtype=np.float32)
        h1, _ = self.ok("put", a.tobytes(), nbytes=32)
        h2, _ = self.ok("put", b"x" * 100, nbytes=100)
        self.assertNotEqual(h1["handle"], h2["handle"])
        st = self.ok("status")[0]
        self.assertEqual(st["bytes_in_use"], 132)
        _, d = self.ok("get", handle=h1["handle"])
        np.testing.assert_array_equal(np.frombuffer(d, np.float32), a)
        _, d = self.ok("get", handle=h1["handle"], offset=8, nbytes=8)
        np.testing.assert_array_equal(np.frombuffer(d, np.float32), a[2:4])
        self.assertEqual(self.err("get", handle=h1["handle"], offset=30, nbytes=8)[0], "BadRequest")
        self.ok("free", handle=h1["handle"])
        self.assertEqual(self.ok("status")[0]["bytes_in_use"], 100)
        self.assertEqual(self.err("get", handle=h1["handle"])[0], "NoSuchHandle")
        self.assertEqual(self.err("free", handle=h1["handle"])[0], "NoSuchHandle")
        self.assertEqual(self.err("put", b"abc", nbytes=5)[0], "BadRequest")
        self.assertEqual(self.ok("free", handle="all")[0]["freed"], 1)
        self.assertEqual(self.ok("status")[0]["bytes_in_use"], 0)

    def test_run_checks_and_results(self):
        self.ok("load", path=self.image)
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        hin, _ = self.ok("put", x.tobytes(), nbytes=x.nbytes)
        h, _ = self.ok("run", kernel="t_total_kernel", n=3, reset=False, **{"in": hin["handle"]})
        self.assertEqual(h["status"], 0)
        self.assertEqual(h["out_nbytes"], 12)
        _, d = self.ok("get", handle=h["out"])
        np.testing.assert_array_equal(np.frombuffer(d, np.float32), x.sum(axis=1))
        self.assertEqual(self.err("run", kernel="t_total_kernel", n=4, **{"in": hin["handle"]})[0], "TooBig")
        self.assertEqual(self.err("run", kernel="nope", n=1, **{"in": hin["handle"]})[0], "NoSuchKernel")
        self.assertEqual(self.err("run", kernel="t_total_kernel", n=0, **{"in": hin["handle"]})[0], "BadRequest")
        self.assertEqual(self.err("run", kernel="t_total_kernel", n=1, **{"in": 999})[0], "NoSuchHandle")
        small, _ = self.ok("put", b"\0" * 4, nbytes=4)
        self.assertEqual(self.err("run", kernel="t_total_kernel", n=3, out=small["handle"], **{"in": hin["handle"]})[0], "TooBig")
        st = self.ok("status")[0]
        self.assertEqual((st["counters"]["runs"], st["counters"]["records"]), (1, 3))

    def test_state_persists_resets_and_traps(self):
        self.ok("load", path=self.image)
        xs = np.array([1, 2, 3], np.float32)
        hin, _ = self.ok("put", xs.tobytes(), nbytes=12)

        def run(reset):
            h, _ = self.ok("run", kernel="t_acc_kernel", n=3, reset=reset, **{"in": hin["handle"]})
            _, d = self.ok("get", handle=h["out"])
            return np.frombuffer(d, np.float32).tolist(), h["status"]
        self.assertEqual(run(True), ([1, 3, 6], 0))
        self.assertEqual(run(False), ([7, 9, 12], 0))          # the state continued
        self.assertEqual(run(True), ([1, 3, 6], 0))            # reset clears it
        bad, _ = self.ok("put", np.array([-1], np.float32).tobytes(), nbytes=4)
        h, _ = self.ok("run", kernel="t_acc_kernel", n=1, reset=True, **{"in": bad["handle"]})
        self.assertEqual(h["status"], 3)

    def test_loading_frees_the_buffers_of_the_old_image(self):
        self.ok("load", path=self.image)
        self.ok("put", b"abcd", nbytes=4)
        self.ok("load", path=make_image(self.tmp))
        self.assertEqual(self.ok("status")[0]["bytes_in_use"], 0)


class ClientTest(unittest.TestCase):
    """The client against a server process (``--fake``) and against processes that misbehave."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="kgcli_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def image_with_copy(self):
        entry = dict(ENTRY_ACC, top="copy", name="copy", c_name="copy", in_words=1, out_words=1)
        d = tempfile.mkdtemp(dir=self.tmp)
        os.makedirs(os.path.join(d, "_x/reports/link/imp"))
        with open(os.path.join(d, "_x/reports/link/imp/impl_1_hw_bb_locked_timing_summary_routed.rpt"), "w") as f:
            f.write(PASS_RPT)
        with open(os.path.join(d, "c.xclbin"), "wb") as f:
            f.write(b"bits")
        with open(os.path.join(d, "k.json"), "w") as f:
            json.dump([{"entries": [entry]}], f)
        gate.main([d, "--xclbin", "c.xclbin", "--kernels", os.path.join(d, "k.json")])
        return os.path.join(d, "c.xclbin")

    def test_session_over_a_real_pipe(self):
        with HwSession(None, fake=True, timeout=30) as hw:
            self.assertEqual(hw.hello["protocol"], 1)
            self.assertEqual(hw.hello["device"], "fake")
            self.assertEqual(hw.info()["kind"], "fake")
            r = hw.load(self.image_with_copy())
            self.assertIn("replaces", r["message"])
            x = np.arange(1000, dtype=np.float32)
            out = hw.kernel("copy").run(x)
            np.testing.assert_array_equal(out.reshape(-1), x)
            self.assertEqual(hw.kernel("copy").e["top"], "copy")
            h, _ = hw.put(x)
            np.testing.assert_array_equal(hw.get(h), x)
            k = hw.kernel("copy")
            res = k.run(h, keep=True)
            np.testing.assert_array_equal(hw.get(res), x)
            self.assertEqual(hw.status()["counters"]["runs"], 2)
            with self.assertRaises(HwError) as cm:
                hw.kernel("missing")
            self.assertEqual(cm.exception.kind, "NoSuchKernel")
            with self.assertRaises(HwError) as cm:
                hw.get(client.Handle(hw, 99, 4))
            self.assertEqual(cm.exception.kind, "NoSuchHandle")

    def test_refusal_is_an_exception(self):
        with HwSession(None, fake=True, timeout=30) as hw:
            with self.assertRaises(HwRefused):
                hw.load(os.path.join(self.tmp, "nothing.xclbin"))

    def test_pack_checks_shapes(self):
        k = client.HwKernel(None, ENTRY_SUM)
        a, n = k.pack([np.zeros((5, 4))])
        self.assertEqual((a.shape, n, a.dtype), ((5, 4), 5, np.dtype("<f4")))
        a, n = k.pack([np.zeros(4)])
        self.assertEqual((a.shape, n), ((1, 4), 1))
        with self.assertRaises(HwError):
            k.pack([np.zeros((5, 3))])
        with self.assertRaises(HwError):
            k.pack([np.zeros(4), np.zeros(4)])

    def test_a_dropped_connection_is_a_clean_error(self):
        s = HwSession(None, command=["bash", "-c", "echo 'cannot connect' >&2; exit 255"], timeout=10)
        with self.assertRaises(HwConnectionError) as cm:
            s.start()
        self.assertIn("cannot connect", str(cm.exception))
        # the server dies after the hello
        code = ("import sys, struct, json\n"
                "h=json.dumps({'ok':True,'protocol':1}).encode()\n"
                "sys.stdout.buffer.write(struct.pack('<II',4+len(h),len(h))+h); sys.stdout.buffer.flush()\n"
                "sys.stdin.buffer.read(1)\n"
                "sys.exit(7)\n")
        s = HwSession(None, command=[sys.executable, "-c", code], timeout=10)
        s.start()
        t0 = time.time()
        with self.assertRaises(HwConnectionError):
            s.info()
        self.assertLess(time.time() - t0, 8)
        s.close()

    def test_a_silent_server_times_out(self):
        code = ("import sys, struct, json, time\n"
                "h=json.dumps({'ok':True,'protocol':1}).encode()\n"
                "sys.stdout.buffer.write(struct.pack('<II',4+len(h),len(h))+h); sys.stdout.buffer.flush()\n"
                "time.sleep(30)\n")
        s = HwSession(None, command=[sys.executable, "-c", code], timeout=1)
        s.start()
        t0 = time.time()
        with self.assertRaises(HwConnectionError) as cm:
            s.info()
        self.assertIn("no answer", str(cm.exception))
        self.assertLess(time.time() - t0, 10)
        s.close()


if __name__ == "__main__":
    unittest.main()
