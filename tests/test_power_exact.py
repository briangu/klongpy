import unittest

from klongpy import KlongInterpreter


class TestExactIntegerPower(unittest.TestCase):
    """A scalar integer to a non-negative integer power is exact."""

    def setUp(self):
        self.klong = KlongInterpreter()

    def test_scalar_exact_beyond_float_precision(self):
        self.assertEqual(self.klong("7^22"), 3909821048582988049)
        self.assertEqual([int(v) for v in self.klong("[7]^[22]")], [3909821048582988049])

    def test_scalar_exact_beyond_64_bits(self):
        self.assertEqual(self.klong("7^40"), 7 ** 40)
        self.assertEqual(int(self.klong("(-3)^41")), (-3) ** 41)

    def test_compiled_and_interpreted_agree(self):
        self.klong("f::{x^y}")
        self.assertEqual(self.klong("f(7;22)"), 7 ** 22)
        self.assertEqual(self.klong("{x^22}(7)"), 7 ** 22)
        self.assertEqual(self.klong("+/{x^22}'[7 7]"), 2 * 7 ** 22)

    def test_negative_or_fractional_exponent_is_a_float(self):
        self.assertEqual(float(self.klong("2^-2")), 0.25)
        self.assertAlmostEqual(float(self.klong("2^0.5")), 2 ** 0.5)
        self.assertEqual(float(self.klong("2.5^2")), 6.25)


class TestHugeIntegerText(unittest.TestCase):
    """An exact power can have far more than 4300 digits (Python 3.11+ refuses str() of
    that); every display path must still print it."""

    def setUp(self):
        self.klong = KlongInterpreter()

    def test_writer_matches_python_around_the_limit(self):
        from klongpy.writer import kg_write_integer
        for v in (10 ** 4299, 10 ** 4300, 7 ** 5000 - 1, -(3 ** 9000), 2 ** 12000, 2 ** 12001, 10 ** 100000):
            self.assertEqual(kg_write_integer(v), _py_str(v))

    def test_every_display_path(self):
        want = "1" + "0" * 5000
        self.assertEqual(self.klong("$10^5000"), want)
        self.assertEqual(self.klong("$-(10^5000)"), "-" + want)
        from klongpy.writer import kg_write
        self.assertEqual(kg_write(self.klong("10^5000"), self.klong._backend), want)
        self.assertEqual(kg_write(self.klong._backend.kg_asarray([10 ** 5000, 2]), self.klong._backend), "[" + want + " 2]")

    def test_near_the_exact_limit(self):
        s = self.klong("$2^524288")
        self.assertEqual(s, _py_str(2 ** 524288))

    def test_command_line(self):
        import os, subprocess, sys, tempfile
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        env = dict(os.environ, PYTHONPATH=root)
        base = [sys.executable, "-m", "klongpy.cli"] if False else [sys.executable, "-c", "import sys; from klongpy.cli import main; sys.argv=['kgpy']+sys.argv[1:]; main()"]
        r = subprocess.run(base + ["-e", "10^5000"], capture_output=True, text=True, env=env, cwd=root)
        self.assertEqual((r.returncode, r.stdout.strip(), r.stderr), (0, "1" + "0" * 5000, ""))
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "big.kg")
            with open(f, "w") as fh:
                fh.write(".p(10^5000)\n")
            r = subprocess.run(base + [f], capture_output=True, text=True, env=env, cwd=root)
            self.assertEqual((r.stdout.strip(), r.stderr), ("1" + "0" * 5000, ""))
        r = subprocess.run(base, input="10^5000\n", capture_output=True, text=True, env=env, cwd=root)
        self.assertIn("1" + "0" * 5000, r.stdout)
        self.assertNotIn("internal error", r.stdout + r.stderr)


def _py_str(v):
    import sys
    old = getattr(sys, "get_int_max_str_digits", None)
    if old is None:
        return str(v)
    prev = old()
    sys.set_int_max_str_digits(0)
    try:
        return str(v)
    finally:
        sys.set_int_max_str_digits(prev)


class TestTorchIntegerPowerIsExact(unittest.TestCase):
    """On mps pow is exp/log in float32: [2 4]^-1 gave [0.49999994 0.24999997] and 10^3
    999.99994. Exactly representable powers must be exact on every torch device."""

    def devices(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        devs = ["cpu"]
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devs.append("mps")
        return devs

    def test_exact_powers_on_each_device(self):
        cases = {
            "[2 4]^-1": [0.5, 0.25],
            "[2 4]^[1 -1]": [2.0, 0.25],
            "[2 4]^-2": [0.25, 0.0625],
            "[2 4]^[-3 -3]": [0.125, 0.015625],
            "[2.0 10.0 5.0]^3": [8.0, 1000.0, 125.0],
            "[0 2]^-1": [float("inf"), 0.5],
        }
        for dev in self.devices():
            for src, expected in cases.items():
                with self.subTest(device=dev, src=src):
                    r = KlongInterpreter(backend="torch", device=dev)(src)
                    self.assertEqual(r.cpu().tolist() if hasattr(r, "cpu") else list(r), expected)


if __name__ == "__main__":
    unittest.main()
