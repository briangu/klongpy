"""A 0-d value from a reduction is a number to `$` (Format2), `.fmt` and the address of .srv (torch backend)."""
import unittest

from klongpy import KlongInterpreter
from klongpy.core import kg_write


class TestZeroDimensionalNumbers(unittest.TestCase):

    def run_kg(self, text):
        k = KlongInterpreter()
        k("v::[1.5 2.5]\nn::+/[1 2]\nw::-(+/[3 4])")
        return kg_write(k(text), k._backend, display=True)

    def test_fmt_with_a_computed_real(self):
        self.assertEqual(self.run_kg('.fmt("{:-8.2}";+/v)'), "    4.00")
        self.assertEqual(self.run_kg('.fmt("{:8.2}";+/v)'), "       4.00")

    def test_fmt_with_a_computed_integer(self):
        self.assertEqual(self.run_kg('.fmt("{:5.1}";n)'), "    3.0")
        self.assertEqual(self.run_kg('.fmt("{:5}";n)'), "3    ")
        self.assertEqual(self.run_kg('.fmt("{} and {:6.2}";[;n;+/v])'), "3.0 and      4.00")

    def test_format2_with_a_computed_real(self):
        self.assertEqual(self.run_kg('(-8.2)$+/v'), "    4.00")
        self.assertEqual(self.run_kg('(-8.2)$187.78'), "  187.78")
        self.assertEqual(self.run_kg('8.2$187.78'), "     187.78")

    def test_format2_with_a_computed_width(self):
        self.assertEqual(self.run_kg('w$n'), "      3")
        self.assertEqual(self.run_kg('(-w)$n'), "3      ")

    def test_pf_prints_the_same(self):
        self.assertEqual(self.run_kg('.pf("{:-8.2}";+/v)'), "    4.00")

    def test_a_computed_port_is_a_number(self):
        k = KlongInterpreter()
        from klongpy.sys_fn_ipc import _plain
        k("p::+/[1000 2000]")
        self.assertEqual(str(_plain(k, k("p"))), "3000")


if __name__ == "__main__":
    unittest.main()
