"""
.fpga: Klong code calls a kernel like a function (klongpy/sys_fn_hw.py, klongpy/hw/binding.py), on the simulator
(KLONG_FPGA=sim), so this runs wherever a C and a C++ compiler exist: argument checks, streams, state, traps, resident
buffers, the info record, the errors, and the same answers as the interpreted program on the numpy and torch backends.
The card itself is tests/static/test_hw_binding_card.py.
"""
import numbers
import os
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np

from klongpy import KlongInterpreter
from klongpy.errors import (KlongArityError, KlongError, KlongRangeError, KlongShapeError, KlongStaticError, KlongSystemError,
                            KlongTypeError)
from klongpy.hw import binding
from klongpy.hw.client import HwConnectionError, HwRefused
from klongpy.hw.sim import SimSession, SimUnavailable, compilers
from klongpy.sys_fn_hw import KlongHwBusyError, KlongHwConnectionError, KlongHwError, KlongHwImageError

from tests.static.harness import Host, same
from tests.static.hostutil import to_np
from klongpy.static.signature import Ty

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAVG = os.path.join(ROOT, "examples", "embedded", "mavg.kg")
FIR = os.path.join(ROOT, "examples", "embedded", "fir.kg")
PID = os.path.join(ROOT, "examples", "embedded", "pid.kg")
TRAP = os.path.join(ROOT, "tests", "static", "hw", "trap.kg")
PYTHON = sys.executable

try:
    compilers()
    WHY = ""
except SimUnavailable as e:
    WHY = str(e)


_SAVED = {}


def setUpModule():
    for k in ("KLONG_FPGA", "KLONG_FPGA_HOST"):
        _SAVED[k] = os.environ.get(k)
    os.environ["KLONG_FPGA"] = "sim"
    os.environ.pop("KLONG_FPGA_HOST", None)


def tearDownModule():
    binding.close_all()
    for k, v in _SAVED.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def klong():
    return KlongInterpreter()


def reference(path, entry, calls):
    """The interpreted program (numpy backend, strict 2) on a list of argument lists."""
    with open(path) as f:
        host = Host(f.read(), path=path)
    out = []
    for args in calls:
        kind, v = host.call(entry, *args)
        assert kind == "ok", v
        out.append(v)
    return out


def close(test, want, got, shape):
    ty = Ty("float", tuple(shape))
    for i, (w, g) in enumerate(zip(want, got)):
        d = same(w, g if shape else float(g), ty, 32)
        test.assertIsNone(d, "call %d: %s" % (i, d))


class TestNoHardware(unittest.TestCase):
    def test_unset_means_a_clear_error(self):
        env = {k: v for k, v in os.environ.items() if k not in ("KLONG_FPGA", "KLONG_FPGA_HOST")}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("importlib.util.find_spec", return_value=None):
            with self.assertRaises(KlongHwError) as cm:
                klong()('.fpga("%s")' % MAVG)
        m = str(cm.exception)
        self.assertIn("KLONG_FPGA_HOST", m)
        self.assertIn("KLONG_FPGA=sim", m)
        self.assertEqual(cm.exception.kind, "system")

    def test_nothing_of_the_hardware_is_imported_at_start(self):
        code = ("import sys; import klongpy; from klongpy import KlongInterpreter; k = KlongInterpreter(); "
                "bad = [m for m in sys.modules if m.startswith('klongpy.hw') or m in ('torch', 'pyxrt')]; "
                "print(bad)")
        env = {k: v for k, v in os.environ.items() if k not in ("KLONG_FPGA", "KLONG_FPGA_HOST", "_KLONGPY_TEST_BACKEND")}
        r = subprocess.run([PYTHON, "-W", "ignore", "-c", code], capture_output=True, text=True, cwd=ROOT, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "[]")

    def test_info_without_hardware_is_an_error(self):
        env = {k: v for k, v in os.environ.items() if k not in ("KLONG_FPGA", "KLONG_FPGA_HOST")}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("importlib.util.find_spec", return_value=None):
            with self.assertRaises(KlongHwError):
                klong()(".fpga.info()")


@unittest.skipIf(WHY, WHY)
class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.k = klong()

    def run_kg(self, text):
        return self.k(text)

    def arr(self, v):
        return to_np(self.k, v) if hasattr(v, "shape") else v

    # -- binding --------------------------------------------------------------------------
    def test_bind_returns_the_entries_and_binds_program_dot_entry(self):
        h = self.run_kg('fir::.fpga("%s")' % FIR)
        self.assertEqual(sorted(h), ["run"])
        y = self.run_kg('fir.run(256#1.0)')
        self.assertEqual(tuple(y.shape), (241,))
        # the dictionary value is the same function
        self.run_kg('g::fir@"run"')
        self.assertEqual(tuple(self.run_kg("g(256#1.0)").shape), (241,))

    def test_several_programs_in_one_image(self):
        self.k['paths'] = [MAVG, FIR, TRAP]
        self.run_kg('h::.fpga(paths)')
        self.assertEqual(sorted(self.run_kg("h")), ["f", "g", "run", "step"])
        self.assertEqual(self.run_kg("trap.g(7;3)"), 1)
        self.assertEqual(self.run_kg("mavg.step(4.0)"), 1.0)

    def test_a_program_outside_the_profile_or_with_an_extern_is_refused(self):
        with self.assertRaises(KlongStaticError) as cm:
            self.run_kg('.fpga("%s")' % PID)
        self.assertIn("extern", str(cm.exception))

    def test_missing_file_and_wrong_argument(self):
        with self.assertRaises(KlongSystemError):
            self.run_kg('.fpga("/no/such/program.kg")')
        with self.assertRaises(KlongTypeError):
            self.run_kg(".fpga(3)")

    # -- the same answers as the interpreter -------------------------------------------------
    def test_fir_blocks_equal_the_interpreter(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(40, 256))
        self.run_kg('.fpga("%s")' % FIR)
        self.k['x'] = x
        got = to_np(self.k, self.run_kg("fir.run(x)"))
        self.assertEqual(got.shape, (40, 241))
        close(self, reference(FIR, "run", [[b] for b in x]), got, [241])
        # one call equals the same block inside the stream
        self.k['b3'] = x[3]
        close(self, [got[3]], [to_np(self.k, self.run_kg("fir.run(b3)"))], [241])

    def test_mavg_ticks_equal_the_interpreter_and_state_persists(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=300)
        self.run_kg('.fpga("%s")' % MAVG)
        self.k['x'] = x
        got = to_np(self.k, self.run_kg("mavg.step(x)"))
        close(self, reference(MAVG, "step", [[float(v)] for v in x]), got, [])
        # state persists between calls: the window holds the last four ticks
        a = float(self.run_kg("mavg.step(8.0)"))
        self.assertAlmostEqual(a, (x[-3] + x[-2] + x[-1] + 8.0) / 4, places=4)
        # and the first call after .fpga starts from the initial state
        self.run_kg('.fpga("%s")' % MAVG)
        self.assertEqual(self.run_kg("mavg.step(4.0)"), 1.0)
        self.assertEqual(self.run_kg("mavg.step(4.0)"), 2.0)
        self.run_kg("v::.fpga.reset(mavg.step)")
        self.assertEqual(self.run_kg("mavg.step(4.0)"), 1.0)

    def test_integers_and_two_arguments(self):
        self.run_kg('.fpga("%s")' % TRAP)
        self.assertEqual(self.run_kg("trap.f([1 2 3 4];3)"), 4)
        self.assertEqual(self.run_kg("trap.g(7;3)"), 1)
        r = self.run_kg("trap.g([7 9 10];[3 4 5])")
        np.testing.assert_array_equal(self.arr(r), [1, 1, 0])
        self.k['tabs'] = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        self.k['idx'] = np.array([0, 3])
        np.testing.assert_array_equal(self.arr(self.run_kg("trap.f(tabs;idx)")), [1, 8])

    def test_the_result_is_an_array_of_the_session_backend(self):
        self.run_kg('.fpga("%s")' % FIR)
        y = self.run_kg("fir.run(256#1.0)")
        self.assertTrue(self.k._backend.is_array(y))
        s = self.run_kg("fir.run(256#1.0)@0")
        self.assertAlmostEqual(float(s), 1.0, places=5)
        self.run_kg('.fpga("%s")' % MAVG)
        self.assertIsInstance(self.run_kg("mavg.step(4.0)"), numbers.Real)       # the backend's float type (float32 on torch)
        self.run_kg('.fpga("%s")' % TRAP)
        self.assertIsInstance(self.run_kg("trap.g(7;3)"), int)

    # -- the checks before anything is sent -------------------------------------------------------
    def test_argument_checks_are_the_entry_signature_checks(self):
        self.run_kg('.fpga("%s")' % FIR)
        for text, cls in (("fir.run([1.0 2.0])", KlongShapeError), ('fir.run("abc")', KlongTypeError), ("fir.run(3.0)", KlongTypeError),
                          ("fir.run(1;2)", KlongArityError), ("fir.run()", KlongArityError)):
            with self.assertRaises(cls, msg=text) as cm:
                self.run_kg(text)
            self.assertIn("float[256] -> float[241]", str(cm.exception), text)
        self.run_kg('.fpga("%s")' % TRAP)
        with self.assertRaises(KlongTypeError):
            self.run_kg("trap.g(1.5;2)")
        with self.assertRaises(KlongRangeError) as cm:
            self.run_kg("trap.g(2147483648;2)")
        self.assertIn("32-bit", str(cm.exception))
        with self.assertRaises(KlongShapeError):
            self.run_kg("trap.g([7 8];[1 2 3])")

    def test_the_checks_agree_with_call_entry(self):
        from klongpy.static import call_entry, load_program
        host = load_program(FIR, strict=2, backend="numpy")
        self.run_kg('.fpga("%s")' % FIR)
        for bad in ("[1.0 2.0]", '"abc"', "3.0"):
            with self.assertRaises(KlongError) as a:
                self.run_kg("fir.run(%s)" % bad)
            with self.assertRaises(KlongError) as b:
                call_entry(host, "run", {"[1.0 2.0]": [1.0, 2.0], '"abc"': "abc", "3.0": 3.0}[bad])
            self.assertEqual(a.exception.kind, b.exception.kind, bad)

    # -- traps ------------------------------------------------------------------------------
    def test_traps_raise_and_name_the_entry_and_the_next_run_is_clean(self):
        self.run_kg('.fpga("%s")' % TRAP)
        with self.assertRaises(KlongRangeError) as cm:
            self.run_kg("trap.f([1 2 3 4];9)")
        self.assertIn("trap.f", str(cm.exception))
        self.assertIn("index out of range", str(cm.exception))
        self.assertIn("not returned", str(cm.exception))
        with self.assertRaises(KlongRangeError) as cm:
            self.run_kg("trap.g(7;0)")
        self.assertIn("trap.g", str(cm.exception))
        self.assertIn("division by zero", str(cm.exception))
        self.assertEqual(self.run_kg("trap.g(7;3)"), 1)
        # one bad call among several makes the whole run fail
        with self.assertRaises(KlongRangeError):
            self.run_kg("trap.g([7 7 7];[3 0 3])")

    # -- buffers ----------------------------------------------------------------------------
    def test_resident_buffers(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(6, 256))
        self.run_kg('.fpga("%s")' % FIR)
        self.k['x'] = x
        direct = to_np(self.k, self.run_kg("fir.run(x)"))
        self.run_kg("b::.fpga.put(fir.run;x)")
        self.assertEqual(int(self.run_kg('.fpga.info()@"buffers"')), 1)
        np.testing.assert_array_equal(to_np(self.k, self.run_kg("fir.run(b)")), direct)
        np.testing.assert_array_equal(to_np(self.k, self.run_kg("fir.run(b)")), direct)     # again: it is still there
        back = to_np(self.k, self.run_kg(".fpga.get(b)"))
        np.testing.assert_allclose(back, x, rtol=1e-6)
        self.assertEqual(self.run_kg(".fpga.free(b)"), 1)
        self.assertEqual(int(self.run_kg('.fpga.info()@"buffers"')), 0)
        with self.assertRaises(KlongTypeError) as cm:
            self.run_kg("fir.run(b)")
        self.assertIn("freed", str(cm.exception))
        with self.assertRaises(KlongTypeError):
            self.run_kg(".fpga.free(b)")
        with self.assertRaises(KlongTypeError):
            self.run_kg(".fpga.get(b)")

    def test_buffer_of_two_arguments_and_the_wrong_entry(self):
        self.run_kg('.fpga(["%s" "%s"])' % (TRAP, FIR))
        # a kernel with several arguments takes them as a list
        from klongpy.sys_fn_hw import eval_sys_hw_put
        buf = eval_sys_hw_put(self.k, self.k['trap.f'], [np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), np.array([1, 2])])
        self.k['b'] = buf
        np.testing.assert_array_equal(self.arr(self.run_kg("trap.f(b)")), [2, 7])
        with self.assertRaises(KlongTypeError) as cm:
            self.run_kg("fir.run(b)")
        self.assertIn("trap.f", str(cm.exception))
        with self.assertRaises(KlongTypeError):
            self.run_kg("trap.f(b;1)")
        self.run_kg(".fpga.free(b)")

    def test_put_needs_a_kernel(self):
        self.run_kg('.fpga("%s")' % FIR)
        with self.assertRaises(KlongTypeError):
            self.run_kg(".fpga.put(3;4)")
        with self.assertRaises(KlongShapeError):
            self.run_kg(".fpga.put(fir.run;[1.0 2.0])")
        with self.assertRaises(KlongTypeError):
            self.run_kg(".fpga.free(3)")

    # -- info -------------------------------------------------------------------------------
    def test_info(self):
        self.run_kg('.fpga("%s")' % FIR)
        info = self.run_kg(".fpga.info()")
        self.assertEqual(info["mode"], "sim")
        self.assertIn("simulator", info["note"])
        self.assertIn("fir_run_kernel", info["kernels"])
        self.assertTrue(info["image"].endswith(".sim"))
        self.assertEqual(len(info["key"]), 64)
        self.assertEqual(info["build"]["passed"], 1)
        self.assertEqual(info["buffers"], 0)

    # -- a binding is tied to its image ------------------------------------------------------------
    def test_a_replaced_image_is_an_error_not_a_silent_reload(self):
        self.run_kg('.fpga("%s")' % FIR)
        self.run_kg('.fpga("%s")' % MAVG)
        with self.assertRaises(KlongHwImageError) as cm:
            self.run_kg("fir.run(256#1.0)")
        self.assertIn("call .fpga again", str(cm.exception))
        self.run_kg('.fpga("%s")' % FIR)
        self.assertEqual(tuple(self.run_kg("fir.run(256#1.0)").shape), (241,))

    def test_same_program_in_two_interpreters_shares_the_session(self):
        a, b = klong(), klong()
        a('.fpga("%s")' % MAVG)
        b('.fpga("%s")' % MAVG)
        self.assertEqual(a("mavg.step(4.0)"), 1.0)
        self.assertEqual(b("mavg.step(4.0)"), 1.0)      # the binding of b started the program afresh

    def test_lint_knows_the_binding(self):
        from klongpy.lint import lint_source
        text = 'fir::.fpga("fir.kg")\ny::fir.run(256#1.0)\n'
        names = [d for d in lint_source(text) if "fir.run" in str(d)]
        self.assertEqual(names, [])


class TestErrors(unittest.TestCase):
    """The errors of a card session, produced by stand-ins for the parts that need a machine."""

    def setUp(self):
        os.environ["KLONG_FPGA"] = "sim"
        self.k = klong()

    def test_no_image_names_the_build_command(self):
        from klongpy.hw.binding import NoImage
        with mock.patch.object(binding, "choose_mode", return_value="card"), \
                mock.patch.object(binding, "session_for", return_value=SimSession()), \
                mock.patch.object(binding, "find_image_for", side_effect=NoImage("no usable image for this source (key abc): the build does not exist")):
            with self.assertRaises(KlongHwImageError) as cm:
                self.k('.fpga("%s")' % FIR)
        self.assertIn("kgpy fpga build " + FIR, str(cm.exception))

    def test_timing_failed_and_stale_images(self):
        from klongpy.hw.binding import NoImage
        for why in ("failed routed timing", "does not exist"):
            msg = "no usable image for this source (key abc): the build %s. Build it with: kgpy fpga build x.kg" % why
            with mock.patch.object(binding, "choose_mode", return_value="card"), \
                    mock.patch.object(binding, "session_for", return_value=SimSession()), \
                    mock.patch.object(binding, "find_image_for", side_effect=NoImage(msg)):
                with self.assertRaises(KlongHwImageError) as cm:
                    self.k('.fpga("%s")' % FIR)
            self.assertIn(why, str(cm.exception))
            self.assertIn("kgpy fpga build", str(cm.exception))

    def test_find_image_for_searches_other_images_and_reports_the_command(self):
        """The exact image first; otherwise a passing image that holds this program; otherwise an error with the command."""
        from klongpy.hw import build, remote
        from klongpy.hw.build import StaleImage

        class FakeBuilder:
            root = "/r"
            listing = ""

            def __init__(self, remote=None):
                pass

            def sh(self, script, timeout=120):
                if script.startswith("cat "):
                    return 0, '{"xclbin": "x.xclbin", "passed": true}', ""
                return 0, FakeBuilder.listing, ""

        keys = build.plan_for([FIR], 32, 300).program_keys
        import json
        good = json.dumps({"dir": "/r/builds/aa", "xclbin": "x.xclbin", "passed": True, "met": True, "target": "hw",
                           "built_at": "2026-11-16", "programs": keys + ["other"]})
        failed = json.dumps({"dir": "/r/builds/bb", "xclbin": "y.xclbin", "passed": False, "met": False, "target": "hw",
                             "built_at": "2026-11-17", "programs": keys})
        with mock.patch.object(build, "find_image", side_effect=StaleImage("no usable image for this source: x")), \
                mock.patch.object(build, "Builder", FakeBuilder), mock.patch.object(remote, "Remote", lambda target=None: None):
            FakeBuilder.listing = good + "\n" + failed + "\n"
            path, meta = binding.find_image_for([FIR], "h")
            self.assertEqual(path, "/r/builds/aa/x.xclbin")
            FakeBuilder.listing = failed + "\n"
            with self.assertRaises(binding.NoImage):
                binding.find_image_for([FIR], "h")

    def test_card_busy_and_refused_images(self):
        with mock.patch.object(SimSession, "load", side_effect=HwRefused("CardBusy", "the card is in use by x (pid 1); nothing was loaded")):
            with self.assertRaises(KlongHwBusyError) as cm:
                self.k('.fpga("%s")' % MAVG)
        self.assertIn("in use", str(cm.exception))
        with mock.patch.object(SimSession, "load", side_effect=HwRefused("ImageRefused", "a.timing-failed.xclbin: timing failed")):
            with self.assertRaises(KlongHwImageError) as cm:
                self.k('.fpga("%s")' % MAVG)
        self.assertIn("timing", str(cm.exception))

    @unittest.skipIf(WHY, WHY)
    def test_connection_lost(self):
        self.k('.fpga("%s")' % MAVG)
        with mock.patch.object(SimSession, "call", side_effect=HwConnectionError("the connection to the server was lost")):
            with self.assertRaises(KlongHwConnectionError) as cm:
                self.k("mavg.step(4.0)")
        self.assertIn("lost", str(cm.exception))
        self.assertEqual(cm.exception.kind, "system")

    def test_a_server_that_dies_is_a_connection_error(self):
        from klongpy.hw.client import HwSession
        dead = HwSession(None, command=[PYTHON, "-c", "import sys; sys.exit(3)"], timeout=10)
        with mock.patch.object(binding, "choose_mode", return_value="card"), mock.patch.object(binding, "session_for", return_value=dead), \
                mock.patch.object(binding, "find_image_for", return_value=("/x/y.xclbin", {"key": "k"})):
            with self.assertRaises(KlongHwConnectionError):
                self.k('.fpga("%s")' % MAVG)

    def test_error_kinds(self):
        for cls in (KlongHwError, KlongHwImageError, KlongHwBusyError, KlongHwConnectionError):
            self.assertTrue(issubclass(cls, KlongError))
            self.assertEqual(cls.kind, "system")


if __name__ == "__main__":
    unittest.main()


REGS = os.path.join(ROOT, "tests", "static", "hw", "regs.kg")


def klist(values):
    """Klong source for a list of numbers."""
    return "[" + " ".join(repr(float(v)) if isinstance(v, float) else str(int(v)) for v in values) + "]"


def interpreted(ticks, gain=2.0, inc=3, off=0.5):
    """regs.kg run by the interpreter stand-in: the summary after the ticks."""
    from klongpy.static import call_entry, load_program, set_param
    k = load_program(REGS, strict=2, backend="numpy")
    set_param(k, "gain", gain)
    set_param(k, "inc", inc)
    set_param(k, "off", off)
    for x in ticks:
        call_entry(k, "step", float(x))
    return list(np.asarray(call_entry(k, "summary")))


@unittest.skipIf(WHY, WHY)
class TestRegistersAndLanes(unittest.TestCase):
    """.fpga.set and .fpga.lanes: parameters between runs, one value per lane, and the summary of a pair."""
    TICKS = [1.0, 2.0, 3.5, -4.0, 0.25]

    def setUp(self):
        self.k = klong()

    def values(self, v):
        return [float(x) for x in np.asarray(to_np(self.k, v)).reshape(-1)]

    def test_a_summary_pair_runs_the_calls_and_returns_the_summary_and_reads_the_state(self):
        self.k('h::.fpga("%s")' % REGS)
        self.assertEqual(sorted(self.k("h")), ["step", "summary"])
        got = self.values(self.k("regs.step(%s)" % klist(self.TICKS)))
        self.assertEqual(got, interpreted(self.TICKS))
        self.assertEqual(self.values(self.k("regs.summary()")), got, "summary() runs no ticks")
        more = self.values(self.k("regs.step(1.0)"))
        self.assertEqual(more, interpreted(self.TICKS + [1.0]), "the state continues")
        self.k(".fpga.reset(regs.step)")
        self.assertEqual(self.values(self.k("regs.summary()")), interpreted([]), "a reset empties the state")

    def test_set_changes_scalar_registers_of_every_type_and_survives_a_reset(self):
        self.k('h::.fpga("%s")' % REGS)
        self.assertEqual(int(self.k('.fpga.set(h;"gain";0.5)')), 1)
        self.k('.fpga.set(h;"inc";-2)')
        self.k('.fpga.set(h;"off";0.3)')
        self.assertEqual(self.values(self.k("regs.step(%s)" % klist(self.TICKS))), interpreted(self.TICKS, 0.5, -2, 0.3))
        self.k(".fpga.reset(regs.step)")
        self.assertEqual(self.values(self.k("regs.step(%s)" % klist(self.TICKS))), interpreted(self.TICKS, 0.5, -2, 0.3))
        # a kernel handle works as well as the dictionary
        self.k('.fpga.set(regs.step;"inc";7)')
        self.k(".fpga.reset(regs.step)")
        self.assertEqual(self.values(self.k("regs.step(%s)" % klist(self.TICKS))), interpreted(self.TICKS, 0.5, 7, 0.3))

    def test_lanes_take_one_value_each_and_the_result_has_a_lane_dimension(self):
        self.k('h::.fpga.lanes("%s";3)' % REGS)
        self.k('.fpga.set(h;"gain";%s)' % klist([1.0, 2.0, -0.5]))
        self.k('.fpga.set(h;"inc";%s)' % klist([1, 2, 3]))
        self.k('.fpga.set(h;"off";%s)' % klist([0.0, 0.25, -1.5]))
        got = self.k("regs.step(%s)" % klist(self.TICKS))
        self.assertEqual(tuple(got.shape), (3, 3))
        want = [interpreted(self.TICKS, g, n, o) for g, n, o in ((1.0, 1, 0.0), (2.0, 2, 0.25), (-0.5, 3, -1.5))]
        self.assertEqual(np.asarray(to_np(self.k, got)).tolist(), want)
        again = np.asarray(to_np(self.k, self.k("regs.summary()"))).tolist()
        self.assertEqual(again, want)

    def test_set_errors(self):
        self.k('h::.fpga.lanes("%s";3)' % REGS)
        from klongpy.errors import KlongNameError
        with self.assertRaises(KlongNameError) as cm:
            self.k('.fpga.set(h;"nothing";[1 2 3])')
        self.assertIn("gain, inc, off", str(cm.exception))
        with self.assertRaises(KlongShapeError):                       # one value per lane
            self.k('.fpga.set(h;"inc";[1 2])')
        with self.assertRaises(KlongTypeError):                        # a scalar is not an array of lanes
            self.k('.fpga.set(h;"inc";5)')
        with self.assertRaises(KlongTypeError):                        # an int register takes ints
            self.k('.fpga.set(h;"inc";[1.5 2 3])')
        with self.assertRaises(KlongRangeError):                       # the kernel's int is 32 bits
            self.k('.fpga.set(h;"inc";[1 2 3000000000])')
        with self.assertRaises(KlongTypeError):
            self.k('.fpga.set(h;7;[1 2 3])')
        with self.assertRaises(KlongTypeError):
            self.k('.fpga.set(5;"inc";[1 2 3])')
        with self.assertRaises(KlongTypeError):
            self.k('.fpga.lanes("%s";0)' % REGS)

    def test_lanes_and_scalars_do_not_mix_up(self):
        self.k('h::.fpga("%s")' % REGS)
        with self.assertRaises(KlongTypeError):                        # no lane dimension: a scalar
            self.k('.fpga.set(h;"inc";[1 2 3])')
        binding.close_all()
        k2 = klong()
        k2('g::.fpga.lanes("%s";2)' % REGS)
        self.assertEqual(tuple(k2("regs.step(1.0)").shape), (2, 3))

    def test_no_image_for_lanes_names_the_lanes_in_the_build_command(self):
        from klongpy.hw.binding import build_command
        self.assertEqual(build_command(["a.kg"], 300, 64), "kgpy fpga build a.kg --lanes 64")
        self.assertEqual(build_command(["a.kg"], 250, 64), "kgpy fpga build a.kg --clock-mhz 250 --lanes 64")
        self.assertEqual(build_command(["a.kg"], 300, None), "kgpy fpga build a.kg")


@unittest.skipIf(WHY, WHY)
class TestResidentSummary(unittest.TestCase):
    """The client's side of a summary kernel: a resident history, a run of its first n calls, a read without calls."""
    def test_a_resident_buffer_runs_its_first_n_calls_and_a_read_runs_none(self):
        os.environ["KLONG_FPGA"] = "sim"
        b = binding.bind([REGS])
        k = b.session.kernel(b.by_name["step"][0].top)
        ticks = np.array([1.0, 2.0, 3.5, -4.0, 0.25], dtype="<f4")
        h, _ = b.session.put(ticks)
        self.assertEqual(list(k.run(h, reset=True, n=3)), interpreted(ticks[:3]))
        self.assertEqual(list(k.run(h, reset=False, n=2)), interpreted(list(ticks[:3]) + list(ticks[:2])), "n calls continue the state")
        self.assertEqual(list(k.run()), interpreted(list(ticks[:3]) + list(ticks[:2])), "no argument: a read, no calls")
        self.assertEqual(list(k.run(h, reset=True)), interpreted(ticks), "all the calls the buffer holds")
        with self.assertRaises(Exception) as cm:
            k.run(h, n=6)
        self.assertIn("holds 5 calls", str(cm.exception))
        binding.close_all()
