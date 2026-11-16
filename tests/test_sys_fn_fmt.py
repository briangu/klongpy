import io
import unittest

from klongpy import KlongInterpreter, KlongException
from klongpy.core import KGChannel, KGChannelDir
from tests.backend_compat import requires_strings, numeric_only


def _capture(klong):
    buf = io.StringIO()
    klong['.sys.cout'] = KGChannel(buf, KGChannelDir.OUTPUT)
    return buf


@requires_strings
class TestFmt(unittest.TestCase):
    def setUp(self):
        self.klong = KlongInterpreter()

    def fmt(self, s):
        return self.klong(s)

    def test_scalar_and_auto_index(self):
        self.assertEqual(self.fmt('.fmt("Spot: {}";3.5)'), "Spot: 3.5")
        self.assertEqual(self.fmt('.fmt("{} {}";[;1;"a"])'), "1 a")

    def test_indexed(self):
        self.assertEqual(self.fmt('.fmt("{1} {0}";[;1;2])'), "2 1")
        self.assertEqual(self.fmt('.fmt("{0}{0}";[;7])'), "77")

    def test_escaped_braces(self):
        self.assertEqual(self.fmt('.fmt("{{}}";[])'), "{}")
        self.assertEqual(self.fmt('.fmt("{{{}}}";5)'), "{5}")

    def test_widths(self):
        self.assertEqual(self.fmt('.fmt("{:8}|";"ab")'), self.fmt('(8$"ab"),"|"'))
        self.assertEqual(self.fmt('.fmt("{0:-8}";"ab")'), self.fmt('(0-8)$"ab"'))
        self.assertEqual(self.fmt('.fmt("{:3}|";12)'), "12 |")

    def test_fractional(self):
        self.assertEqual(self.fmt('.fmt("{:5.2}";3.14159)'), "    3.14")
        self.assertEqual(self.fmt('.fmt("{:5.2}";3.14159)'), self.fmt('5.2$3.14159'))
        # integer converted to float first; width follows Format2 (integer digits)
        self.assertEqual(self.fmt('.fmt("{:5.2}";3)'), self.fmt('5.2$3.0'))
        self.assertEqual(self.fmt('.fmt("{:5.2}";3)'), "    3.00")

    def test_fractional_needs_number(self):
        with self.assertRaises(KlongException):
            self.fmt('.fmt("{:5.2}";"ab")')

    def test_rendering_matches_display(self):
        for v in ['[1 2 3]', '"str"', '0cx', ':{[1 2]}', ':undefined', '[1 [2 3]]', '1.5']:
            buf = _capture(self.klong)
            self.klong(f'.d({v})')
            self.assertEqual(self.fmt(f'.fmt("{{}}";,{v})'), buf.getvalue(), v)

    def test_list_args_and_wrapped_list(self):
        self.assertEqual(self.fmt('.fmt("{}";,[1 2 3])'), "[1 2 3]")
        self.assertEqual(self.fmt('.fmt("{} {}";1,2)'), "1 2")
        self.assertEqual(self.fmt('.fmt("{} {}";[;[1 2];"x"])'), "[1 2] x")
        self.assertEqual(self.fmt('.fmt("{}";"abc")'), "abc")

    def test_errors(self):
        for src in ['.fmt("{} {}";1)', '.fmt("{}";[;1;2])', '.fmt("}";1)',
                    '.fmt("{} {0}";[;1;2])', '.fmt("{2}";[;1;2])', '.fmt("{:x}";1)',
                    '.fmt("{a}";1)', '.fmt(5;1)']:
            with self.assertRaises(KlongException, msg=src):
                self.fmt(src)

    def test_unmatched_open_brace(self):
        # the Klong parser cannot read a string literal holding a lone "{",
        # so pass the template through a variable
        self.klong['t'] = "{"
        with self.assertRaises(KlongException):
            self.fmt('.fmt(t;1)')
        self.klong['t'] = "a{b"
        with self.assertRaises(KlongException) as ctx:
            self.fmt('.fmt(t;1)')
        self.assertIn("a{b", str(ctx.exception))

    def test_error_names_template(self):
        with self.assertRaises(KlongException) as ctx:
            self.fmt('.fmt("a}b";1)')
        self.assertIn("a}b", str(ctx.exception))

    def test_pf(self):
        buf = _capture(self.klong)
        r = self.klong('.pf("a={}";1)')
        self.assertEqual(r, "a=1")
        self.assertEqual(buf.getvalue(), "a=1\n")

    def test_d_and_p_unchanged(self):
        buf = _capture(self.klong)
        self.klong('.d("x");.p(1)')
        self.assertEqual(buf.getvalue(), "x1\n")


class TestFmtNumeric(unittest.TestCase):
    @numeric_only
    def test_numeric_args(self):
        klong = KlongInterpreter()
        self.assertEqual(klong('.fmt("{} {}";[1 2])'), "1 2")
        self.assertEqual(klong('.fmt("{}";,[1 2 3])'), "[1 2 3]")


if __name__ == "__main__":
    unittest.main()
