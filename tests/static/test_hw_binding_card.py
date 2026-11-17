"""
.hw on the Alveo U250 (klongpy/sys_fn_hw.py): Klong code calls the kernels of a built image and the answers equal the
interpreter. Skipped unless KLONG_HW_HOST names the machine with the card (user@host) and an image holding
examples/embedded/mavg.kg, fir.kg and tests/static/hw/trap.kg passed routed timing there:

    kgpy --hw-build examples/embedded/mavg.kg examples/embedded/fir.kg tests/static/hw/trap.kg --wait
    KLONG_HW_HOST=brian@xilinx python -m pytest tests/static/test_hw_binding_card.py

Loading the image replaces whatever user image the card holds; the tests skip when the card is in use. The session
that these tests open is closed when the module ends, so that tests/static/test_hw_card.py can use the card after it.
"""
import os
import tempfile
import unittest

import numpy as np

from klongpy import KlongInterpreter
from klongpy.errors import KlongRangeError, KlongTypeError
from klongpy.hw import binding
from klongpy.hw.build import StaleImage, find_image
from klongpy.static.signature import Ty
from klongpy.sys_fn_hw import KlongHwImageError

from tests.test_hw_binding import close, reference
from .hostutil import to_np
from .test_emit_hls import program
from .test_ir import ROOT

HOST = os.environ.get("KLONG_HW_HOST", "").strip() or None
TRAP = os.path.join(ROOT, "tests", "static", "hw", "trap.kg")
MAVG, FIR = program("mavg"), program("fir")
SOURCES = [MAVG, FIR, TRAP]


def _locate():
    if not HOST:
        return None, "set KLONG_HW_HOST=user@host (the machine with the U250)"
    try:
        return find_image(SOURCES, host=HOST), ""
    except StaleImage as e:
        return None, str(e)
    except Exception as e:                      # the machine does not answer
        return None, "cannot look for the image on %s: %s" % (HOST, e)


FOUND, WHY = _locate()
_SAVED = {}


def setUpModule():
    for k in ("KLONG_HW",):
        _SAVED[k] = os.environ.get(k)
    os.environ.pop("KLONG_HW", None)


def tearDownModule():
    binding.close_all()
    for k, v in _SAVED.items():
        if v is not None:
            os.environ[k] = v


@unittest.skipUnless(FOUND, WHY)
class TestBindingOnCard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.k = KlongInterpreter()
        try:
            cls.k('h::.hw(["%s" "%s" "%s"])' % (MAVG, FIR, TRAP))
        except Exception as e:
            if "in use" in str(e):
                raise unittest.SkipTest("the card cannot be loaded now: %s" % e)
            raise

    def test_bound_to_the_image_that_passed_timing(self):
        info = self.k(".hw.info()")
        self.assertEqual(info["mode"], "card")
        self.assertTrue(info["key"].startswith(FOUND[1]["key"][:12]))
        self.assertEqual(info["build"]["timing"]["met"], 1)
        self.assertGreaterEqual(info["build"]["timing"]["wns_ns"], 0)
        self.assertNotIn("timing-failed", info["image"])
        self.assertIn("xilinx_u250", info["shell"])

    def test_a_single_program_finds_the_image_it_was_built_in(self):
        k = KlongInterpreter()
        h = k('.hw("%s")' % FIR)
        self.assertEqual(sorted(h), ["run"])
        self.k('.hw(["%s" "%s" "%s"])' % (MAVG, FIR, TRAP))      # leave the module's binding as it was

    def test_mavg_over_20000_ticks_equals_the_interpreter(self):
        rng = np.random.default_rng(1121)
        x = rng.normal(size=20000)
        self.k["x"] = x
        got = to_np(self.k, self.k("mavg.step(x)"))
        self.assertEqual(got.shape, (20000,))
        close(self, reference(MAVG, "step", [[float(v)] for v in x]), got, [])

    def test_fir_over_200_blocks_equals_the_interpreter_and_one_call_equals_the_stream(self):
        rng = np.random.default_rng(1122)
        x = rng.normal(size=(200, 256))
        self.k["x"] = x
        got = to_np(self.k, self.k("fir.run(x)"))
        self.assertEqual(got.shape, (200, 241))
        close(self, reference(FIR, "run", [[b] for b in x]), got, [241])
        self.k["b7"] = x[7]
        close(self, [got[7]], [to_np(self.k, self.k("fir.run(b7)"))], [241])

    def test_state_persists_and_resets(self):
        self.k(".hw.reset(mavg.step)")
        self.assertEqual(self.k("mavg.step(4.0)"), 1.0)
        self.assertEqual(self.k("mavg.step(8.0)"), 3.0)
        self.k(".hw.reset(mavg.step)")
        self.assertEqual(self.k("mavg.step(8.0)"), 2.0)

    def test_traps_raise_and_the_next_run_is_clean(self):
        self.assertEqual(self.k("trap.f([1 2 3 4];3)"), 4)
        with self.assertRaises(KlongRangeError) as cm:
            self.k("trap.f([1 2 3 4];9)")
        self.assertIn("trap.f", str(cm.exception))
        with self.assertRaises(KlongRangeError) as cm:
            self.k("trap.g(7;0)")
        self.assertIn("division by zero", str(cm.exception))
        self.assertEqual(self.k("trap.g(7;3)"), 1)

    def test_resident_buffer(self):
        rng = np.random.default_rng(1123)
        x = rng.normal(size=(50, 256))
        self.k["x"] = x
        direct = to_np(self.k, self.k("fir.run(x)"))
        self.k("b::.hw.put(fir.run;x)")
        self.assertEqual(int(self.k('.hw.info()@"buffers"')), 1)
        np.testing.assert_array_equal(to_np(self.k, self.k("fir.run(b)")), direct)
        np.testing.assert_allclose(to_np(self.k, self.k(".hw.get(b)")), x, rtol=1e-6)
        self.assertEqual(self.k(".hw.free(b)"), 1)
        self.assertEqual(int(self.k('.hw.info()@"buffers"')), 0)
        with self.assertRaises(KlongTypeError):
            self.k("fir.run(b)")

    def test_source_without_an_image_names_the_build_command(self):
        with open(MAVG) as f:
            text = f.read().replace("0.0 0.0 0.0 0.0", "1.0 0.0 0.0 0.0")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mavg2.kg")
            with open(path, "w") as f:
                f.write(text)
            with self.assertRaises(KlongHwImageError) as cm:
                KlongInterpreter()('.hw("%s")' % path)
        self.assertIn("kgpy --hw-build " + path, str(cm.exception))


if __name__ == "__main__":
    unittest.main()
