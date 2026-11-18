"""
examples/hardware/crossover.kg, the strategy kernel, on every tier that needs no card: the 20-tick
series checked by hand, the interpreter stand-in, the emitted C, the generated HLS kernel in C simulation, a numpy
version of the strategy, and the kernel through ``.fpga.lanes`` on the simulator. All of them give the same account,
bit for bit. The card itself is tests/static/test_crossover_card.py.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from klongpy.static import call_entry, load_program, set_param
from klongpy.static import fixed as fx

from .harness import find_cc
from .test_emit_hls import CC, CXX, HlsBuild, pack
from .test_hls_stream import regs_bytes

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROGRAM = os.path.join(ROOT, "examples", "hardware", "crossover.kg")
sys.path.insert(0, os.path.join(ROOT, "examples", "hardware"))
import crossover_bench as cb          # noqa: E402

SPEC = fx.spec_of("fixed<32,16>")

# Twenty ticks, fast = 2, slow = 4, stop = 1.5. Worked by hand (a price in the comments is the tick, 1 to 20):
#   ticks 1-4 are flat at 10; the signal needs 4 ticks, and at tick 4 both averages are 10 (no signal).
#   5 (11): fast 10.5 > slow 10.25: long at 11 (trade 1).  6 to 8: 12, 13, 14, each +1: profit 3, peak 3.
#   9 (13): -1, profit 2 (drawdown 1).   10 (12): -1, profit 1 (drawdown 2); the averages cross: short at 12 (trade 2).
#   11 to 14 (11, 10, 9, 8): +1 each, profit 5, peak 5.  15 (9): -1, profit 4.  16 (10): -1, profit 3 (drawdown 2);
#   the averages cross again: long at 10 (trade 3).  17 to 20 (11 to 14): +1 each, profit 7.  No stop is hit.
RAMP = [10, 10, 10, 10, 11, 12, 13, 14, 13, 12, 11, 10, 9, 8, 9, 10, 11, 12, 13, 14]
RAMP_RESULT = (7.0, 2.0, 3.0)
# The same start, then a stop and a flip:
#   5 (11): long at 11 (trade 1).  6 (9.25): -1.75, profit -1.75, drawdown 1.75; 11 > 9.25 + 1.5: the stop is hit and the
#   position closes (fast 10.125 is still above slow 10.0625, so it is not reopened).  7 (9.5): the signal turns down:
#   short at 9.5 (trade 2).  8 (9.75): -0.25, profit -2.  9 (11.5): -1.75, profit -3.75 (drawdown 3.75); 11.5 - 1.5 > 9.5:
#   the stop is hit, and the signal is up: long at 11.5 (trade 3).  10 to 20: +0.5 a tick, 11 ticks: profit 1.75.
STOP = [10, 10, 10, 10, 11, 9.25, 9.5, 9.75, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17]
STOP_RESULT = (1.75, 3.75, 3.0)
HAND = {"ramp": (RAMP, RAMP_RESULT), "stop": (STOP, STOP_RESULT)}
PARAMS = (2, 4, 1.5)


def raw_ticks(prices):
    return np.asarray(fx.from_float(np.asarray(prices, dtype=np.float64), SPEC), dtype="<i4")


def interpreter(raw, fast, slow, stop_raw):
    """The interpreter stand-in: step once per tick, then summary. Returns int64 raw [profit, drawdown, trades]."""
    k = load_program(PROGRAM, strict=2, backend="numpy")
    set_param(k, "fast", int(fast))
    set_param(k, "slow", int(slow))
    set_param(k, "stop", fx.FixedArray(np.int64(stop_raw), SPEC))
    for x in raw:
        assert call_entry(k, "step", fx.FixedArray(np.int64(x), SPEC)) is None
    return np.asarray(call_entry(k, "summary", raw=True))


def as_float(raw_summary):
    return tuple(np.asarray(raw_summary, dtype=np.float64) / 2.0 ** 32)


class TestHandChecked(unittest.TestCase):
    def test_the_interpreter_gives_the_hand_worked_results(self):
        for name, (prices, want) in HAND.items():
            with self.subTest(series=name):
                got = interpreter(raw_ticks(prices), PARAMS[0], PARAMS[1], int(PARAMS[2] * 65536))
                self.assertEqual(as_float(got), want)

    def test_the_numpy_version_gives_them_too(self):
        for name, (prices, want) in HAND.items():
            with self.subTest(series=name):
                got = cb.numpy_run(raw_ticks(prices), [2], [4], [int(1.5 * 65536)])
                self.assertEqual(as_float(got[0]), want)

    def test_nothing_is_traded_before_the_slow_window_is_full_and_the_parameters_are_clamped(self):
        # slow = 9: the first signal can only come at tick 9; fast/slow outside 1..32 act as the nearest end
        prices = [10, 10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        got = interpreter(raw_ticks(prices), 1, 9, 65536)
        self.assertEqual(as_float(got)[2], 1.0)
        a = interpreter(raw_ticks(prices), 0, 100, 65536)
        b = interpreter(raw_ticks(prices), 1, 32, 65536)
        self.assertTrue(np.array_equal(a, b))


@unittest.skipUnless(CC and CXX, "needs a C and a C++ compiler")
class TestTiers(unittest.TestCase):
    """Every tier on the same history and the same parameter sets: equal to the numpy version, which equals the interpreter."""
    LANES = [0, 5, 22, 41, 63]
    TICKS = 6000

    @classmethod
    def setUpClass(cls):
        cls.history = cb.history(cls.TICKS, seed=7)
        f, s, p = cb.param_grid(64)
        cls.fast, cls.slow, cls.stop = f[cls.LANES], s[cls.LANES], p[cls.LANES]
        cls.want = cb.numpy_run(cls.history, cls.fast, cls.slow, cls.stop)

    def test_the_interpreter_equals_numpy(self):
        for i in (0, 3):
            got = interpreter(self.history[:1500], self.fast[i], self.slow[i], self.stop[i])
            want = cb.numpy_run(self.history[:1500], self.fast[i:i + 1], self.slow[i:i + 1], self.stop[i:i + 1])[0]
            self.assertTrue(np.array_equal(got, want), (i, got, want))

    def test_the_emitted_c_equals_numpy(self):
        d = tempfile.mkdtemp(prefix="kgxo_")
        r = subprocess.run([sys.executable, "-W", "ignore", "-m", "klongpy.cli", "compile", "c", PROGRAM, "-o", d], capture_output=True,
                           text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stderr)
        cb.main(["cfile", d])
        subprocess.run([find_cc(), "-O3", "-std=c99", "-I" + d, os.path.join(d, "crossover.c"), os.path.join(d, "cpu_bench.c"), "-o",
                        os.path.join(d, "cpu_bench")], check=True, capture_output=True)
        self.history.tofile(os.path.join(d, "h.bin"))
        np.stack([self.fast, self.slow, self.stop], axis=1).astype("<i4").tofile(os.path.join(d, "p.bin"))
        subprocess.run([os.path.join(d, "cpu_bench"), os.path.join(d, "h.bin"), os.path.join(d, "p.bin"), "0", str(len(self.LANES)),
                        os.path.join(d, "o.bin")], check=True, capture_output=True)
        rec = np.fromfile(os.path.join(d, "o.bin"), dtype=np.uint8).reshape(len(self.LANES), 32)
        got = rec[:, :24].copy().view("<i8").reshape(len(self.LANES), 3)
        self.assertTrue(np.array_equal(got, self.want))

    def test_the_hls_kernel_in_c_simulation_equals_numpy_for_one_call_and_for_chunks(self):
        lanes = len(self.LANES)
        b = HlsBuild(path=PROGRAM, name="crossover", wrapped=False, lanes=lanes)
        e = b.entry
        self.assertEqual((e["output"], e["out_words"], e["kernel_args"][-2:]), ("summary", 3 * lanes, ["params", "status"]))
        import json
        with open(os.path.join(b.dir, "manifest.json")) as f:
            m = json.load(f)
        regs = regs_bytes(m, {"fast": self.fast, "slow": self.slow, "stop": self.stop / 65536.0})
        data = self.history.tobytes()
        for per in (0, 999):
            out, status = b.run(data, per=per, regs=regs)
            self.assertEqual(status, 0)
            got = np.frombuffer(out, dtype="<i8").reshape(lanes, 3)
            self.assertTrue(np.array_equal(got, self.want), per)

    def test_the_binding_on_the_simulator_equals_numpy_and_set_changes_the_lanes(self):
        from klongpy import KlongInterpreter
        from klongpy.hw import binding
        lanes = len(self.LANES)
        with mock.patch.dict(os.environ, {"KLONG_FPGA": "sim"}):
            os.environ.pop("KLONG_FPGA_HOST", None)
            k = KlongInterpreter()
            k('h::.fpga.lanes("%s";%d)' % (PROGRAM, lanes))
            k['t'] = self.history.astype(np.float64) / 65536.0
            def setp(name, values):
                k["pv"] = np.asarray(values)
                return int(k('.fpga.set(h;"%s";pv)' % name))
            self.assertEqual(setp("fast", self.fast.astype(np.int64)), 1)
            setp("slow", self.slow.astype(np.int64))
            setp("stop", self.stop.astype(np.float64) / 65536.0)
            got = np.asarray(k._backend.to_numpy(k("crossover.step(t)"))) * 2.0 ** 32
            self.assertTrue(np.array_equal(got.astype(np.int64), self.want))
            again = np.asarray(k._backend.to_numpy(k("crossover.summary()"))) * 2.0 ** 32
            self.assertTrue(np.array_equal(again.astype(np.int64), self.want), "summary() reads the state, it does not run ticks")
            # other parameters, the same history, a fresh start
            k(".fpga.reset(crossover.step)")
            setp("fast", self.fast[::-1].astype(np.int64))
            setp("slow", self.slow[::-1].astype(np.int64))
            setp("stop", self.stop[::-1].astype(np.float64) / 65536.0)
            got = np.asarray(k._backend.to_numpy(k("crossover.step(t)"))) * 2.0 ** 32
            want = cb.numpy_run(self.history, self.fast[::-1], self.slow[::-1], self.stop[::-1])
            self.assertTrue(np.array_equal(got.astype(np.int64), want))
            binding.close_all()


class TestCheckers(unittest.TestCase):
    def test_it_is_inside_the_profile_and_streamable(self):
        r = subprocess.run([sys.executable, "-W", "ignore", "-m", "klongpy.cli", "check", PROGRAM, "--streamable"],
                           capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("nothing keeps a kernel from taking one record per clock cycle", r.stdout)

    def test_history_is_inside_the_band_and_reproducible(self):
        a = cb.history(5000, seed=3)
        self.assertTrue(np.array_equal(a, cb.history(5000, seed=3)))
        self.assertFalse(np.array_equal(a, cb.history(5000, seed=4)))
        self.assertGreaterEqual(int(a.min()), 5 * 65536)
        self.assertLessEqual(int(a.max()), 15 * 65536)
