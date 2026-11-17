"""
The Alveo U250: the kernels of ``mavg``, ``fir`` and the trap program run on the card and equal the
interpreter. Skipped unless KLONG_HW_HOST names the machine with the card (user@host) and an image built from exactly
this source passed routed timing there:

    kgpy --hw-build examples/embedded/mavg.kg examples/embedded/fir.kg tests/static/hw/trap.kg --wait
    KLONG_HW_HOST=user@fpga-host python -m pytest tests/static/test_hw_card.py

Loading the image replaces whatever user image the card holds; the tests skip when the card is in use. A result here
is "working on hardware" for that image only (design: the image passed timing and its outputs matched the interpreter).
"""
import os
import unittest

import numpy as np

from klongpy.hw.build import StaleImage, find_image
from klongpy.hw.client import HwConnectionError, HwError, HwRefused, HwSession, HwTrap
from klongpy.static.signature import Ty

from .harness import Host, same
from .test_emit_hls import program, special_floats
from .test_ir import ROOT

HOST = os.environ.get("KLONG_HW_HOST", "").strip() or None
BAD_ROOT = os.environ.get("KLONG_HW_ROOT", "").strip() or "~/klong-hw"      # a root on the card host: only names matter here
CLOCK = int(os.environ.get("KLONG_HW_CLOCK", "300"))
TRAP = os.path.join(ROOT, "tests", "static", "hw", "trap.kg")
SOURCES = [program("mavg"), program("fir"), TRAP]


def float_rules(test, want, got, shape):
    """Float32 results within 1e-5 relative, any nan equals any nan."""
    ty = Ty("float", tuple(shape))
    for i, (w, g) in enumerate(zip(want, got)):
        d = same(w, g if shape else g.item(), ty, 32)
        test.assertIsNone(d, "record %d: %s" % (i, d))


def interpreter(path, entry, calls):
    with open(path) as f:
        host = Host(f.read(), path=path)
    out = []
    for args in calls:
        kind, v = host.call(entry, *args)
        assert kind == "ok", v
        out.append(v)
    return out


def _locate():
    if not HOST:
        return None, "set KLONG_HW_HOST=user@host (the machine with the U250)"
    try:
        return find_image(SOURCES, host=HOST, clock_mhz=CLOCK), ""
    except StaleImage as e:
        return None, str(e)
    except Exception as e:                      # the machine does not answer
        return None, "cannot look for the image on %s: %s" % (HOST, e)


FOUND, WHY = _locate()


@unittest.skipUnless(FOUND, WHY)
class TestOnCard(unittest.TestCase):
    hw = None

    @classmethod
    def setUpClass(cls):
        cls.path, cls.build = FOUND
        cls.hw = HwSession(HOST, timeout=300)
        try:
            cls.hw.start()
            cls.loaded = cls.hw.load(cls.path)
        except HwRefused as e:
            cls.hw.close()
            raise unittest.SkipTest("the card cannot be loaded now: %s" % e)

    @classmethod
    def tearDownClass(cls):
        if cls.hw:
            cls.hw.close()

    def test_the_image_passed_routed_timing(self):
        t = self.build["timing"]
        self.assertTrue(self.build["passed"] and t["met"] and t["wns_ns"] >= 0, t)
        self.assertNotIn("timing-failed", self.path)
        self.assertEqual(sorted(self.loaded["kernels"]), ["fir_run_kernel", "mavg_step_kernel", "trap_f_kernel", "trap_g_kernel"])
        self.assertIn("replaces", self.loaded["message"])

    def test_mavg_over_100000_samples_equals_the_interpreter(self):
        rng = np.random.default_rng(111)
        x = special_floats(rng, (100000,), rate=0.0005)
        got = self.hw.kernel("mavg_step_kernel").run(x, reset=True)
        self.assertEqual(got.shape, (100000,))
        float_rules(self, interpreter(program("mavg"), "step", [[float(v)] for v in x]), got, [])

    def test_fir_over_1000_blocks_equals_the_interpreter(self):
        rng = np.random.default_rng(112)
        x = special_floats(rng, (1000, 256), rate=0.001)
        got = self.hw.kernel("fir_run_kernel").run(x)
        self.assertEqual(got.shape, (1000, 241))
        float_rules(self, interpreter(program("fir"), "run", [[b] for b in x]), got, [241])

    def test_state_persists_between_runs_and_resets_on_request(self):
        k = self.hw.kernel("mavg_step_kernel")
        x = np.arange(1, 41, dtype=np.float32)
        whole = k.run(x, reset=True)
        a = k.run(x[:15], reset=True)
        b = k.run(x[15:], reset=False)
        np.testing.assert_array_equal(np.concatenate([a, b]), whole)
        again = k.run(x[15:], reset=True)             # the window starts empty: not what continuing gave
        self.assertFalse(np.array_equal(again, b))
        np.testing.assert_array_equal(k.run(x[:15], reset=True), a)

    def test_the_trap_flag_reaches_the_host(self):
        f = self.hw.kernel("trap_f_kernel")
        table = np.array([1, 2, 3, 4], dtype=np.int32)
        ok = f.run(np.array([table, table]), np.array([3, 0]))
        self.assertEqual(ok.reshape(-1).tolist(), [4, 1])
        self.assertEqual(f.last["status"], 0)
        with self.assertRaises(HwTrap) as cm:
            f.run(np.array([table, table, table]), np.array([1, 9, 2]))
        self.assertEqual(cm.exception.status, 1 | 2)
        g = self.hw.kernel("trap_g_kernel")
        self.assertEqual(g.run(np.array([7]), np.array([3])).reshape(-1).tolist(), [1])
        with self.assertRaises(HwTrap) as cm:
            g.run(np.array([7, 7]), np.array([3, 0]))
        self.assertEqual(cm.exception.status, 1 | 4)
        # the next run starts clean
        self.assertEqual(g.run(np.array([9]), np.array([4])).reshape(-1).tolist(), [1])

    def test_two_entries_of_one_image_both_work(self):
        m = self.hw.kernel("mavg_step_kernel").run(np.array([4.0, 8.0], np.float32), reset=True)
        self.assertEqual(m.tolist(), [1.0, 3.0])
        y = self.hw.kernel("fir_run_kernel").run(np.ones((1, 256), np.float32))
        np.testing.assert_allclose(y, np.ones((1, 241)), rtol=1e-5)
        # and again, the other way round: neither disturbed the other
        self.assertEqual(self.hw.kernel("mavg_step_kernel").run(np.array([4.0], np.float32)).tolist(), [4.0])

    def test_data_stays_resident(self):
        k = self.hw.kernel("fir_run_kernel")
        x = np.random.default_rng(1).random((20, 256), dtype=np.float32)
        h, _ = self.hw.put(x)
        a, b = k.run(h), k.run(h)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, k.run(x))
        out = k.run(h, keep=True)
        np.testing.assert_array_equal(self.hw.get(out).reshape(20, 241), a)
        self.hw.free(out)
        self.hw.free(h)

    def test_a_timing_failed_image_is_refused_by_name_and_without_a_record(self):
        for bad in (BAD_ROOT + "/builds/none/x.timing-failed.xclbin", BAD_ROOT + "/builds/none/x.xclbin"):
            with self.assertRaises(HwRefused):
                self.hw.load(bad)
        self.hw.kernel("mavg_step_kernel")          # the image loaded before is still the session's

    def test_a_stale_image_is_refused(self):
        with open(program("mavg")) as f:
            text = f.read()
        import tempfile
        d = tempfile.mkdtemp(prefix="kgstale_")
        p = os.path.join(d, "mavg.kg")
        with open(p, "w") as f:
            f.write(text.replace("buf::[0.0 0.0 0.0 0.0]", "buf::[0.0 0.0 0.0 0.0 0.0]"))
        with self.assertRaises(StaleImage) as cm:
            find_image([p, program("fir"), TRAP], host=HOST, clock_mhz=CLOCK)
        self.assertIn("kgpy --hw-build", str(cm.exception))


@unittest.skipUnless(HOST, "set KLONG_HW_HOST=user@host")
class TestHostSideRefusals(unittest.TestCase):
    """What the server on the card host does before it touches the card."""

    def test_the_loader_refuses_a_timing_failed_name_without_loading(self):
        with HwSession(HOST, timeout=120) as hw:
            with self.assertRaises(HwRefused) as cm:
                hw.load(BAD_ROOT + "/builds/none/x.timing-failed.xclbin")
            self.assertIn("failed routed timing", str(cm.exception))
            self.assertEqual(hw.info()["session_image"], None)


if __name__ == "__main__":
    unittest.main()
