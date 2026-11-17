"""
kgpy --check-static --streamable (ticket 113 part 2): the constructs that keep a hardware kernel from taking one record per
clock cycle, each with its span and what to change. Fixtures: tests/static/accept/stream-*.kg and reject/stream-*.kg.
"""
import os
import re
import subprocess
import sys
import unittest

from klongpy.static.rules import STREAM_RULES
from klongpy.static.streamable import check_streamable

from .test_ir import EMBEDDED, ROOT

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECT = re.compile(r':" expect: static line (\d+) (\S+) "')


def found(text, name="t.kg"):
    return [(d.line, d.rule) for d in check_streamable(name, text=text)]


class TestFixtures(unittest.TestCase):
    def test_every_streamable_rule_has_an_accept_and_a_reject_fixture(self):
        for rule in STREAM_RULES:
            with self.subTest(rule=rule):
                self.assertEqual(check_streamable(os.path.join(HERE, "accept", rule + ".kg")), [])
                path = os.path.join(HERE, "reject", rule + ".kg")
                with open(path) as f:
                    m = EXPECT.match(f.readline())
                got = [(d.line, d.rule) for d in check_streamable(path)]
                self.assertEqual(got, [(int(m.group(1)), rule)], "one construct, one rule")

    def test_a_diagnostic_has_a_span_and_says_what_to_change(self):
        for rule in STREAM_RULES:
            d = check_streamable(os.path.join(HERE, "reject", rule + ".kg"))[0]
            self.assertTrue(d.span[1] > d.span[0] or rule == "stream-block")
            self.assertTrue(d.hint and len(d.hint) > 30)
            self.assertIn(f"[{rule}]", d.render())
            self.assertIn("hint:", d.render())


class TestRules(unittest.TestCase):
    def test_the_reference_programs(self):
        self.assertEqual(check_streamable(os.path.join(EMBEDDED, "mavg.kg")), [])
        self.assertEqual(check_streamable(os.path.join(EMBEDDED, "fir.kg")), [])           # a sliding window streams
        got = check_streamable(os.path.join(EMBEDDED, "pid.kg"), externs={"read.adc": lambda c: 0.3, "write.pwm": lambda c, d: 0})
        self.assertEqual({d.rule for d in got}, {"stream-state"})      # integ is read and rewritten through a float add

    def test_iterate_with_a_constant_count_is_unrolled_but_a_runtime_count_is_not(self):
        self.assertEqual(found('f::{3{x*2}:*x}\n.entry("f";"int -> int")'), [])
        self.assertEqual(found('f::{y{x*2}:*x}\n.entry("f";"int int -> int")'), [(1, "stream-loop")])

    def test_state_through_a_float_function_is_found_inside_a_callee(self):
        text = 'acc::0.0\ng::{acc*(0.5+x)}\nf::{acc::g(x)+x;acc}\n.entry("f";"float -> float")'
        self.assertEqual(found(text), [(3, "stream-state")])

    def test_float_state_that_is_only_stored_and_summed_is_fine_but_a_fixed_accumulator_is_too(self):
        self.assertEqual(found('b::[0.0 0.0 0.0 0.0]\nf::{b::b:=x,0;+/b}\n.entry("f";"float -> float")'), [])
        text = 'a::.fixed("fixed<32,16>";0)\nf::{a::a+x;a}\n.entry("f";"fixed<32,16> -> fixed<32,16>")'
        self.assertEqual(found(text), [])

    def test_a_divider_in_the_loop_is_found_but_a_constant_modulus_is_not(self):
        self.assertEqual(found('c::0\nf::{c::(c+1)!x;c}\n.entry("f";"int -> int")'), [(2, "stream-state")])
        self.assertEqual(found('c::0\nf::{c::(c+x)!8;c}\n.entry("f";"int -> int")'), [])
        text = 'c::.fixed("fixed<32,16>";1)\nf::{c::c%x;c}\n.entry("f";"fixed<32,16> -> fixed<32,16>")'
        self.assertEqual(found(text), [(2, "stream-state")])
        text = 'c::.fixed("fixed<32,16>";1)\nf::{c::c*x;c}\n.entry("f";"fixed<32,16> -> fixed<32,16>")'
        self.assertEqual(found(text), [])           # a fixed-point multiply is one DSP stage: measured II 1

    def test_a_dynamic_index_into_a_small_table_is_fine(self):
        self.assertEqual(found('t::0.5*!512\nf::{t@x}\n.entry("f";"int -> float")'), [])
        self.assertEqual(found('t::0.5*!2048\nf::{t@x}\n.entry("f";"int -> float")'), [(2, "stream-table")])
        self.assertEqual(found('t::0.5*!2048\nf::{x+t@3}\n.entry("f";"float -> float")'), [])     # a constant index is wiring


class TestCli(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run([sys.executable, "-W", "ignore", "-m", "klongpy.cli", *args], capture_output=True, text=True, cwd=ROOT)

    def test_streamable_flag(self):
        r = self.cli("--check-static", os.path.join(EMBEDDED, "mavg.kg"), "--streamable")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("streamable: nothing keeps a kernel", r.stdout)
        r = self.cli("--check-static", os.path.join(HERE, "reject", "stream-loop.kg"), "--streamable")
        self.assertEqual(r.returncode, 1)
        self.assertIn("[stream-loop]", r.stderr)
        self.assertIn("reject/stream-loop.kg:3:", r.stderr)
        # without the flag the same file is inside the profile
        self.assertEqual(self.cli("--check-static", os.path.join(HERE, "reject", "stream-loop.kg")).returncode, 0)
        self.assertEqual(self.cli("--streamable").returncode, 2)


if __name__ == "__main__":
    unittest.main()
