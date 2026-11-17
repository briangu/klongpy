"""
The strategy kernel of examples/hardware/crossover.kg on the Alveo U250. Skipped unless
KLONG_HW_HOST names the machine with the card and an image of the program with 32 lanes passed routed timing there:

    kgpy --hw-build examples/hardware/crossover.kg --lanes 32 --clock-mhz 250 --wait
    KLONG_HW_HOST=brian@xilinx python -m pytest tests/static/test_crossover_card.py

(the clock is the one the image was built for; the tests find an image of any clock). Loading it replaces whatever
user image the card holds. The two hand-checked series and a history of 200000 ticks are compared with the numpy
version of the strategy, which equals the interpreter stand-in; the full 10-million-tick replay and its timing
are examples/hardware/crossover_bench.py.
"""
import os
import sys
import unittest

import numpy as np

from klongpy.hw import binding

from .test_crossover import HAND, PROGRAM, as_float, raw_ticks
from .test_ir import ROOT

sys.path.insert(0, os.path.join(ROOT, "examples", "hardware"))
import crossover_bench as cb          # noqa: E402

HOST = os.environ.get("KLONG_HW_HOST", "").strip() or None
LANES = 32


def _locate():
    if not HOST:
        return False, "set KLONG_HW_HOST=user@host (the machine with the U250)"
    try:
        binding.find_image_for([PROGRAM], HOST, None, LANES)
        return True, ""
    except Exception as e:
        return False, str(e)


FOUND, WHY = _locate()
_SAVED = {}


def setUpModule():
    _SAVED["KLONG_HW"] = os.environ.pop("KLONG_HW", None)


def tearDownModule():
    binding.close_all()
    if _SAVED.get("KLONG_HW") is not None:
        os.environ["KLONG_HW"] = _SAVED["KLONG_HW"]


@unittest.skipUnless(FOUND, WHY)
class TestStrategyOnCard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from klongpy import KlongInterpreter
        cls.k = KlongInterpreter()
        try:
            cls.k('h::.hw.lanes("%s";%d)' % (PROGRAM, LANES))
        except Exception as e:
            raise unittest.SkipTest("cannot use the card: %s" % e)
        cls.fast, cls.slow, cls.stop = cb.param_grid(LANES)

    def set_params(self, fast, slow, stop):
        for name, v in (("fast", fast.astype(np.int64)), ("slow", slow.astype(np.int64)), ("stop", stop.astype(np.float64) / 65536.0)):
            self.k["pv"] = self.k._backend.kg_asarray(np.asarray(v))
            self.k('.hw.set(h;"%s";pv)' % name)

    def run_ticks(self, raw):
        self.k["t"] = self.k._backend.kg_asarray(raw.astype(np.float64) / 65536.0)
        out = np.asarray(self.k("crossover.step(t)"))
        return (out * 2.0 ** 32).astype(np.int64)

    def test_the_hand_checked_series_in_every_lane(self):
        self.set_params(np.full(LANES, 2), np.full(LANES, 4), np.full(LANES, int(1.5 * 65536)))
        for name, (prices, want) in HAND.items():
            self.k(".hw.reset(crossover.step)")
            got = self.run_ticks(raw_ticks(prices))
            self.assertEqual(got.shape, (LANES, 3))
            for lane in range(LANES):
                self.assertEqual(as_float(got[lane]), want, (name, lane))

    def test_a_history_equals_the_numpy_version_in_every_lane_and_the_state_continues(self):
        self.set_params(self.fast, self.slow, self.stop)
        self.k(".hw.reset(crossover.step)")
        h = cb.history(200000, seed=11)
        want = cb.numpy_run(h, self.fast, self.slow, self.stop)
        self.assertTrue(np.array_equal(self.run_ticks(h), want))
        # the same history in two calls: the second continues from the state of the first
        self.k(".hw.reset(crossover.step)")
        self.run_ticks(h[:77777])
        self.assertTrue(np.array_equal(self.run_ticks(h[77777:]), want))
