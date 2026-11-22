"""
The typed IR, lowering and inference, through the extension: the text form of the typed IR (dump_ir), the driver (check, evaluate
stage one, check again, lower), the typing rules as a user sees them (result types and rule ids of rejections), the fixtures.
The IR itself is the engine's (klong_static); it comes out as text (Compiled.ir_text) and as a structure (ir_json).

Golden files live in tests/static/golden/; KG_UPDATE_GOLDEN=1 rewrites them.
"""
import glob
import os
import unittest

from klongpy.errors import KlongError, KlongStaticError
from klongpy.static.native import c_prefix, compile_program, ir_json
from klongpy.static.signature import Ty

from .common import EMBEDDED, EXPECT, HERE, STUBS, golden


def compile_text(text, **kw):
    return compile_program("t.kg", text=text, **kw)


def entry_result(body, sig, pre=""):
    """The type the typed IR gives the result of f entered with sig."""
    j = ir_json("t.kg", text=f'{pre}f::{{{body}}}\n.entry("f";"{sig}")')
    label = next(e["fn"] for e in j["entries"] if e["name"] == "f")
    r = next(f for f in j["fns"] if f["label"] == label)["result"]
    return str(Ty(r["elem"], tuple(r["shape"])))


def rule_of(body, sig, pre=""):
    with unittest.TestCase().assertRaises(KlongStaticError) as cm:
        compile_text(f'{pre}f::{{{body}}}\n.entry("f";"{sig}")')
    return cm.exception.rule


def nodes_of(node, kind, out=None):
    """Every node of a kind in an IR node of ir_json."""
    out = [] if out is None else out
    if isinstance(node, dict):
        if node.get("k") == kind:
            out.append(node)
        for v in node.values():
            nodes_of(v, kind, out)
    elif isinstance(node, list):
        for v in node:
            nodes_of(v, kind, out)
    return out


def entry_fn(text, name="f"):
    j = ir_json("t.kg", text=text)
    label = next(e["fn"] for e in j["entries"] if e["name"] == name)
    return next(f for f in j["fns"] if f["label"] == label)


class TestTextForm(unittest.TestCase):
    def test_the_text_form_is_stable(self):
        text = 'acc::0\nf::{acc::acc+x;acc}\n.entry("f";"int -> int")'
        self.assertEqual(compile_text(text).ir_text, compile_text(text).ir_text)
        self.assertEqual(compile_text(text).ir_text, (
            "program t (kg_int: 32 bits)\n"
            "global state acc: int = 0\n"
            "entry f: int -> int -> f#0\n"
            "\n"
            "fn f#0 [fn] (x: int) -> int\n"
            "  (set:int (state acc:int) (add:int (state acc:int) (param 0 x:int)))\n"
            "  (state acc:int)\n"
            "  effects: yes\n"))

    def test_golden_dumps_of_the_reference_programs(self):
        for name in ("mavg", "pid", "fir"):
            with self.subTest(program=name):
                golden(self, name + ".ir", compile_program(os.path.join(EMBEDDED, name + ".kg")).ir_text)

    def test_fir_has_its_tables_as_constants_and_no_raw(self):
        j = ir_json(os.path.join(EMBEDDED, "fir.kg"))
        g = {x["name"]: x for x in j["globals"]}
        self.assertEqual(sorted(g), ["idx", "taps"])
        self.assertEqual((g["taps"]["ty"], g["taps"]["state"]), ({"elem": "float", "shape": [16]}, False))
        self.assertEqual((g["idx"]["ty"], g["idx"]["state"]), ({"elem": "int", "shape": [16, 241]}, False))
        text = compile_program(os.path.join(EMBEDDED, "fir.kg")).ir_text
        self.assertIn("global const taps: float[16]", text)
        self.assertIn("global const idx: int[16,241]", text)
        self.assertNotIn("raw", text)
        self.assertEqual([f["name"] for f in j["fns"] if f["kind"] == "fn"], ["run"])


class TestDriver(unittest.TestCase):
    def test_stage_one_values_decide_the_shapes(self):
        j = ir_json("t.kg", text='n::1+2\ntab::{x*x}\'!n\nf::{tab@x}\n.entry("f";"int -> int")')
        self.assertEqual({g["name"]: g["ty"] for g in j["globals"]}, {"tab": {"elem": "int", "shape": [3]}})
        self.assertIn("global const tab: int[3] = [0 1 4]", compile_text('n::1+2\ntab::{x*x}\'!n\nf::{tab@x}\n.entry("f";"int -> int")').ir_text)

    def test_a_program_without_an_entry_is_an_error(self):
        with self.assertRaises(KlongStaticError) as cm:
            compile_text("f::{x+1}")
        self.assertEqual(cm.exception.rule, "entry")

    def test_a_compile_time_extern_call_without_a_stand_in_names_the_extern(self):
        # engine needs: compile_program takes no stand-ins, so the program that needs one at compile time cannot be compiled
        text = 'adc::.extern("read_adc";"int -> float")\ncal::adc(0)\nf::{x+cal}\n.entry("f";"float -> float")'
        with self.assertRaises(KlongError) as cm:
            compile_text(text)
        self.assertIn("read_adc", str(cm.exception))

    def test_the_default_prefix_is_a_c_identifier(self):
        self.assertEqual(c_prefix("examples/embedded/mavg.kg"), "mavg")
        self.assertEqual(c_prefix("my-prog.2.kg"), "my_prog_2")
        self.assertEqual(c_prefix("9lives.kg"), "prog_9lives")
        self.assertEqual(c_prefix("kg_x.kg"), "prog_kg_x")

    def test_int_range_follows_int_bits(self):
        text = 'big::3000000000\nf::{x+big}\n.entry("f";"int -> int")'
        with self.assertRaises(KlongStaticError) as cm:
            compile_text(text)
        self.assertEqual(cm.exception.rule, "int-range")
        self.assertIn("kg_int: 64 bits", compile_text(text, int_bits=64).ir_text)


class TestFixtures(unittest.TestCase):
    def test_every_accept_fixture_lowers(self):
        for path in sorted(glob.glob(os.path.join(HERE, "accept", "*.kg"))):
            with self.subTest(fixture=os.path.basename(path)):
                self.assertIn("entry ", compile_program(path, externs=STUBS).ir_text)

    def test_every_reject_fixture_fails_with_its_rule_and_line(self):
        for path in sorted(glob.glob(os.path.join(HERE, "reject", "*.kg"))):
            if os.path.basename(path).startswith("stream-"):
                continue            # inside the profile: the streamable tier rejects them (test_streamable.py)
            with self.subTest(fixture=os.path.basename(path)):
                with open(path) as f:
                    m = EXPECT.match(f.read())
                line, rule = int(m.group(1)), m.group(2)
                with self.assertRaises(KlongStaticError) as cm:
                    compile_program(path, externs=STUBS)
                found = [(d.rule, d.line) for d in cm.exception.diagnostics]
                self.assertIn((rule, line), found)
                self.assertEqual(cm.exception.kind, "static")


class TestTypingRules(unittest.TestCase):
    def result(self, body, sig, pre=""):
        return entry_result(body, sig, pre)

    def rule(self, body, sig, pre=""):
        return rule_of(body, sig, pre)

    def test_atomic_dyads_pair_leading_dimensions(self):
        self.assertEqual(str(self.result("x+y", "int[2,3] int[2] -> int[2,3]")), "int[2,3]")
        self.assertEqual(self.rule("x+y", "int[3] int[2,3] -> int[2,3]"), "shape-mismatch")

    def test_power_is_int_only_for_a_non_negative_constant_exponent(self):
        self.assertEqual(str(self.result("x^3", "int -> int")), "int")
        self.assertEqual(str(self.result("x^y", "int int -> float")), "float")
        self.assertEqual(str(self.result("x^(0-1)", "int -> float")), "float")
        self.assertEqual(str(self.result("x^2", "float -> float")), "float")

    def test_divide_floor_integer_divide_and_remainder(self):
        self.assertEqual(str(self.result("x%y", "int int -> float")), "float")
        self.assertEqual(str(self.result("_x", "float[3] -> int[3]")), "int[3]")
        self.assertEqual(self.rule("x:%2", "float -> int"), "operand")
        self.assertEqual(str(self.result("x!y", "int int -> int")), "int")
        self.assertEqual(str(self.result("x!y", "float int -> float")), "float")

    def test_scan_and_each_pair_shapes(self):
        self.assertEqual(str(self.result("+\\x", "int[4] -> int[4]")), "int[4]")
        self.assertEqual(str(self.result("0+\\x", "int[4] -> int[5]")), "int[5]")
        self.assertEqual(str(self.result("{x+y}:'x", "int[4] -> int[3]")), "int[3]")
        self.assertEqual(str(self.result("{x,y}:'x", "int[4] -> int[3,2]")), "int[3,2]")

    def test_take_drop_and_negative_counts(self):
        self.assertEqual(str(self.result("(-2)#x", "int[5] -> int[2]")), "int[2]")
        self.assertEqual(str(self.result("7#x", "int[5] -> int[7]")), "int[7]")
        self.assertEqual(self.rule("2#x", "int[2,2] -> int[2,2]"), "take-rank")
        self.assertEqual(self.rule("y#x", "int[5] int -> int[2]"), "const-count")
        self.assertEqual(str(self.result("((#x)-2)_x", "int[5] -> int[2]")), "int[2]")

    def test_gather_gives_the_index_shape_then_the_item_shape(self):
        self.assertEqual(str(self.result("x@y", "float[256] int[16,241] -> float[16,241]")), "float[16,241]")
        self.assertEqual(str(self.result("x@y", "float[5,2] int[3] -> float[3,2]")), "float[3,2]")

    def test_a_constant_index_is_checked_and_made_non_negative(self):
        self.assertEqual(self.rule("x@5", "int[5] -> int"), "index-range")
        f = entry_fn('f::{x@-1}\n.entry("f";"int[5] -> int")')
        (ix,) = nodes_of(f["body"], "Index")
        self.assertTrue(ix["checked"])
        self.assertIn("4", str(ix))
        f = entry_fn('f::{x@y}\n.entry("f";"int[5] int -> int")')
        self.assertFalse(nodes_of(f["body"], "Index")[0]["checked"])

    def test_amend_forms(self):
        self.assertEqual(str(self.result("x:=1.5,0", "int[3] -> float[3]")), "float[3]")
        self.assertEqual(self.rule("x:=y,0", "int[3] int[2] -> int[3]"), "amend-form")
        self.assertEqual(self.rule("x:=9,0", "int[2,2] -> int[2,2]"), "amend-form")

    def test_reshape_of_a_matrix_needs_a_list_of_dimensions(self):
        self.assertEqual(str(self.result("[2 3]:^x", "int[3,2] -> int[2,3]")), "int[2,3]")
        self.assertEqual(str(self.result("[4]:^x", "int[3,2] -> int[4]")), "int[4]")
        self.assertEqual(self.rule("4:^x", "int[3,2] -> int[4]"), "operand")
        self.assertEqual(str(self.result("4:^x", "int[2] -> int[4]")), "int[4]")

    def test_conversions_are_explicit(self):
        f = entry_fn('f::{x+y}\n.entry("f";"int float -> float")')
        (b,) = nodes_of(f["body"], "Binary")
        self.assertEqual(b["a"]["k"], "Convert")
        self.assertEqual(b["a"]["ty"], {"elem": "float", "shape": []})
        self.assertEqual(b["b"]["ty"], {"elem": "float", "shape": []})

    def test_a_local_has_the_join_of_its_assignments(self):
        f = entry_fn('f::{[t];t::0;t::x%2;t}\n.entry("f";"int -> float")')
        self.assertEqual(f["locals"], [{"name": "t", "ty": {"elem": "float", "shape": []}}])
        self.assertEqual(self.rule("[t];t::[1 2];t::[1 2 3];t", "int -> int[3]"), "shape-change")

    def test_a_global_never_changes_type(self):
        self.assertEqual(self.rule("g::x%2;g", "int -> float", pre="g::0\n"), "shape-change")

    def test_constants_fold(self):
        f = entry_fn('n::4\nf::{(n*2)+#x}\n.entry("f";"int[3] -> int")')
        self.assertEqual(f["body"]["stmts"][0]["k"], "Const")
        self.assertIn("11", str(f["body"]["stmts"][0]))

    def test_accumulator_promotes_once(self):
        f = entry_fn('f::{{x+y%2}/x}\n.entry("f";"int[4] -> float")')
        (r,) = nodes_of(f["body"], "Reduce")
        self.assertEqual(r["ty"], {"elem": "float", "shape": []})
        self.assertEqual(self.rule("{x,y}/x", "int[4] -> int[4]"), "loop-state")


if __name__ == "__main__":
    unittest.main()
