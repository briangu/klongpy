"""
The typed IR, lowering and inference (ticket 091): the node classes and the
text form, the driver (check, evaluate stage one, check again, lower), the
rules of the operator table applied to every row, specialisation,
reachability, and the data the storage planner starts from.

Golden files live in tests/static/golden/; KG_UPDATE_GOLDEN=1 rewrites them.
"""
import glob
import os
import re
import unittest

import numpy as np

from klongpy.errors import KlongError, KlongStaticError
from klongpy.static import FORMS, OPTABLE, parse_signature
from klongpy.static import ir
from klongpy.static.compile import c_prefix, compile_program
from klongpy.static.host import load_program
from klongpy.static.ir import dump, storage_inputs
from klongpy.static.signature import Ty

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EMBEDDED = os.path.join(ROOT, "examples", "embedded")
GOLDEN = os.path.join(HERE, "golden")
EXPECT = re.compile(r':" expect: static line (\d+) ([a-z0-9-]+) "\n')
STUBS = {"read.adc": lambda ch: 0.5, "log.msg": lambda s, n: None}


def golden(test, name, text):
    path = os.path.join(GOLDEN, name)
    if os.environ.get("KG_UPDATE_GOLDEN"):
        with open(path, "w") as f:
            f.write(text)
    with open(path) as f:
        want = f.read()
    test.assertEqual(text, want, f"{name} differs from the golden file (KG_UPDATE_GOLDEN=1 rewrites it)")


def compile_text(text, **kw):
    return compile_program("t.kg", text=text, **kw).program


def entry_fn(program, name="f"):
    return next(e.fn for e in program.entries if e.decl.name == name)


def nodes(fn, cls):
    return [n for n in ir.walk(fn.body) if isinstance(n, cls)]


class TestNodesAndPrinter(unittest.TestCase):
    def test_nodes_are_frozen_and_carry_type_and_span(self):
        c = ir.Const(Ty("int"), (3, 4), value=7)
        with self.assertRaises(Exception):
            c.value = 8
        self.assertEqual(c.ty, Ty("int"))
        self.assertEqual(c.span, (3, 4))
        self.assertEqual(c.cval, 7)

    def test_every_node_class_of_the_design_exists(self):
        for name in ("Const GlobalRef LocalRef ParamRef Unary Binary Convert Index Gather Take Drop Rotate Reverse "
                     "Transpose Reshape Join Iota MakeList Amend Cond Call ExternCall Map Map2 Reduce Scan EachPair "
                     "Loop Assign Seq FnSpec Program").split():
            self.assertTrue(hasattr(ir, name), name)

    def test_evaluation_order_is_part_of_the_node(self):
        self.assertEqual(ir.Binary.ORDER, ("b", "a"))               # the right operand of a dyad first
        self.assertEqual(ir.Call.ORDER, ("args",))                  # arguments left to right
        self.assertEqual(ir.Amend.ORDER, ("idx", "value", "a"))     # a:=v,i evaluates i, then v, then a
        self.assertEqual(ir.Reduce.ORDER[:2], ("seed", "a"))        # the operands of an adverb left to right

    def test_the_text_form_is_stable(self):
        text = 'acc::0\nf::{acc::acc+x;acc}\n.entry("f";"int -> int")'
        self.assertEqual(dump(compile_text(text)), dump(compile_text(text)))
        self.assertEqual(dump(compile_text(text)), (
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
                p = compile_program(os.path.join(EMBEDDED, name + ".kg")).program
                golden(self, name + ".ir", dump(p))

    def test_fir_has_its_tables_as_constants_and_no_raw(self):
        p = compile_program(os.path.join(EMBEDDED, "fir.kg")).program
        g = {x.name: x for x in p.globals}
        self.assertEqual(sorted(g), ["idx", "taps"])
        self.assertEqual((g["taps"].ty, g["taps"].state), (Ty("float", (16,)), False))
        self.assertEqual((g["idx"].ty, g["idx"].state), (Ty("int", (16, 241)), False))
        text = dump(p)
        self.assertIn("global const taps: float[16]", text)
        self.assertIn("global const idx: int[16,241]", text)
        self.assertNotIn("raw", text)
        self.assertEqual([f.name for f in p.fns if f.kind == "fn"], ["run"])


class TestDriver(unittest.TestCase):
    def test_stage_one_values_decide_the_shapes(self):
        p = compile_text('n::1+2\ntab::{x*x}\'!n\nf::{tab@x}\n.entry("f";"int -> int")')
        self.assertEqual({g.name: str(g.ty) for g in p.globals}, {"tab": "int[3]"})
        np.testing.assert_array_equal(p.globals[0].value, [0, 1, 4])

    def test_a_program_without_an_entry_is_an_error(self):
        with self.assertRaises(KlongStaticError) as cm:
            compile_text("f::{x+1}")
        self.assertEqual(cm.exception.rule, "entry")

    def test_a_compile_time_extern_call_without_a_stand_in_names_the_extern(self):
        text = '.extern("read.adc";"int -> float")\ncal::read.adc(0)\nf::{x+cal}\n.entry("f";"float -> float")'
        with self.assertRaises(KlongError) as cm:
            compile_text(text)
        self.assertIn("read.adc", str(cm.exception))
        p = compile_text(text, externs={"read.adc": lambda ch: 0.25})
        self.assertEqual(nodes(entry_fn(p), ir.Const)[0].value, 0.25)

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
        self.assertEqual(compile_text(text, int_bits=64).int_bits, 64)


class TestFixtures(unittest.TestCase):
    def test_every_accept_fixture_lowers(self):
        for path in sorted(glob.glob(os.path.join(HERE, "accept", "*.kg"))):
            with self.subTest(fixture=os.path.basename(path)):
                p = compile_program(path, externs=STUBS).program
                self.assertTrue(p.entries)
                self.assertTrue(dump(p))

    def test_every_reject_fixture_fails_with_its_rule_and_line(self):
        for path in sorted(glob.glob(os.path.join(HERE, "reject", "*.kg"))):
            with self.subTest(fixture=os.path.basename(path)):
                with open(path) as f:
                    m = EXPECT.match(f.read())
                line, rule = int(m.group(1)), m.group(2)
                with self.assertRaises(KlongStaticError) as cm:
                    compile_program(path, externs=STUBS)
                found = [(d.rule, d.line) for d in cm.exception.diagnostics]
                self.assertIn((rule, line), found)
                self.assertEqual(cm.exception.kind, "static")


def host_type(v):
    if isinstance(v, np.ndarray):
        return ("float" if v.dtype.kind == "f" else "int"), tuple(v.shape)
    return ("float" if isinstance(v, (float, np.floating)) else "int"), ()


def sample(ty, seed):
    if not ty.shape:
        return seed + 2 if ty.elem == "int" else seed + 1.5
    n = int(np.prod(ty.shape))
    flat = np.arange(1, n + 1) + seed
    return (flat if ty.elem == "int" else flat * 0.5 + 0.25).reshape(ty.shape)


class TestRowsOfTheOperatorTable(unittest.TestCase):
    """Per row: the inferred type and shape are the declared ones, and those of KlongPy's own result."""

    HOST_ELEM_DIFFERS = {"pow", "conditional"}      # the host's element type depends on the value there

    def programs(self):
        for op in OPTABLE:
            for ex in op.examples:
                yield op.id, f'f::{{{ex[0]}}}\n.entry("f";"{ex[1]}")', ex[1], (ex[2] if len(ex) > 2 else None)
        for form in FORMS:
            for prog, sig in form.examples:
                yield form.id, f'{prog}\n.entry("f";"{sig}")', sig, None

    def test_every_allowed_row_has_an_example_that_lowers_to_its_declared_type(self):
        seen = set()
        for row, text, sig, _ in self.programs():
            with self.subTest(row=row, prog=text):
                want = parse_signature(sig)
                p = compile_text(text, externs=STUBS)
                f = entry_fn(p)
                self.assertEqual(f.param_tys, want.params)
                self.assertEqual(f.result_ty, want.result)
                self.assertEqual(f.body.ty, f.result_ty)
                seen.add(row)
        allowed = {op.id for op in OPTABLE if op.status == "allowed"} - {"define"}
        self.assertEqual(allowed - seen, set())

    def test_the_inferred_type_is_the_type_of_klongpys_result(self):
        for row, text, sig, inputs in self.programs():
            for seed in (0, 3):
                with self.subTest(row=row, prog=text, seed=seed):
                    want = parse_signature(sig)
                    f = entry_fn(compile_text(text, externs=STUBS))
                    klong = load_program("t.kg", text=text, externs=STUBS, backend="numpy")
                    args = inputs if inputs is not None else [sample(p, i + seed) for i, p in enumerate(want.params)]
                    value = klong["f"](*[np.asarray(a) if isinstance(a, list) else a for a in args]) if want.arity else klong("f()")
                    elem, shape = host_type(value)
                    self.assertEqual(tuple(f.result_ty.shape), shape)
                    if row not in self.HOST_ELEM_DIFFERS:
                        self.assertEqual(f.result_ty.elem, elem)

    def test_every_node_has_a_type_and_array_shapes_are_positive(self):
        for row, text, sig, _ in self.programs():
            p = compile_text(text, externs=STUBS)
            for f in p.fns:
                for n in ir.walk(f.body):
                    self.assertIsInstance(n.ty, Ty, f"{row}: {type(n).__name__}")
                    self.assertTrue(all(d > 0 for d in n.ty.shape))
                    self.assertLessEqual(len(n.ty.shape), 4)


class TestTypingRules(unittest.TestCase):
    def result(self, body, sig, pre=""):
        return entry_fn(compile_text(f'{pre}f::{{{body}}}\n.entry("f";"{sig}")')).result_ty

    def rule(self, body, sig, pre=""):
        with self.assertRaises(KlongStaticError) as cm:
            compile_text(f'{pre}f::{{{body}}}\n.entry("f";"{sig}")')
        return cm.exception.rule

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
        f = entry_fn(compile_text('f::{x@-1}\n.entry("f";"int[5] -> int")'))
        (ix,) = nodes(f, ir.Index)
        self.assertTrue(ix.checked)
        self.assertEqual(ix.i.value, 4)
        f = entry_fn(compile_text('f::{x@y}\n.entry("f";"int[5] int -> int")'))
        self.assertFalse(nodes(f, ir.Index)[0].checked)

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
        f = entry_fn(compile_text('f::{x+y}\n.entry("f";"int float -> float")'))
        (b,) = nodes(f, ir.Binary)
        self.assertIsInstance(b.a, ir.Convert)
        self.assertEqual(b.a.ty, Ty("float"))
        self.assertEqual(b.b.ty, Ty("float"))

    def test_a_local_has_the_join_of_its_assignments(self):
        f = entry_fn(compile_text('f::{[t];t::0;t::x%2;t}\n.entry("f";"int -> float")'))
        self.assertEqual(f.locals, (Ty("float"),))
        first = f.body.stmts[0]
        self.assertEqual(first.value.ty, Ty("float"))       # the int 0 is stored as a float
        self.assertEqual(self.rule("[t];t::[1 2];t::[1 2 3];t", "int -> int[3]"), "shape-change")

    def test_a_global_never_changes_type(self):
        self.assertEqual(self.rule("g::x%2;g", "int -> float", pre="g::0\n"), "shape-change")

    def test_an_assigned_parameter_becomes_a_local(self):
        f = entry_fn(compile_text('f::{x::x*2;x+1}\n.entry("f";"int -> int")'))
        self.assertEqual(f.local_names, ("x",))
        self.assertIsInstance(f.body.stmts[0].value, ir.ParamRef)

    def test_constants_fold(self):
        f = entry_fn(compile_text('n::4\nf::{(n*2)+#x}\n.entry("f";"int[3] -> int")'))
        self.assertIsInstance(f.body.stmts[0], ir.Const)
        self.assertEqual(f.body.stmts[0].value, 11)

    def test_accumulator_promotes_once(self):
        f = entry_fn(compile_text('f::{{x+y%2}/x}\n.entry("f";"int[4] -> float")'))
        (r,) = nodes(f, ir.Reduce)
        self.assertEqual(r.ty, Ty("float"))
        self.assertEqual(r.fn.param_tys, (Ty("float"), Ty("int")))
        self.assertEqual(self.rule("{x,y}/x", "int[4] -> int[4]"), "loop-state")


class TestSpecialisation(unittest.TestCase):
    def test_one_spec_per_tuple_of_argument_types(self):
        p = compile_text('sq::{x*x}\nf::{sq(x)+sq(1.5)+sq(x+1)}\n.entry("f";"int -> float")')
        sqs = [f for f in p.fns if f.name == "sq"]
        self.assertEqual(sorted(str(f.param_tys[0]) for f in sqs), ["float", "int"])
        self.assertEqual({f.result_ty.elem for f in sqs}, {"int", "float"})

    def test_callees_come_before_callers_and_indices_are_dense(self):
        p = compile_text('a::{x+1}\nb::{a(x)*2}\nf::{b(x)}\n.entry("f";"int -> int")')
        self.assertEqual([f.name for f in p.fns], ["a", "b", "f"])
        self.assertEqual([f.index for f in p.fns], [0, 1, 2])

    def test_unreachable_functions_and_globals_are_left_out(self):
        p = compile_text('t::[1 2 3]\nunused::[4 5 6]\nhelper::{x+unused}\nf::{x+t}\n.entry("f";"int -> int[3]")')
        self.assertEqual([f.name for f in p.fns], ["f"])
        self.assertEqual([g.name for g in p.globals], ["t"])

    def test_recursion_is_rejected(self):
        for text in ('f::{:[x>0;f(x-1);0]}\n.entry("f";"int -> int")',
                     'a::{b(x)}\nb::{a(x)}\n.entry("a";"int -> int")'):
            with self.assertRaises(KlongStaticError) as cm:
                compile_text(text)
            self.assertEqual(cm.exception.rule, "recursion")

    def test_a_literal_verb_gets_the_variables_it_reads_as_captures(self):
        p = compile_text('f(a;b)::{[k];k::2;{x*k+b}\'a}\n.entry("f";"int[3] int -> int[3]")')
        (m,) = nodes(entry_fn(p), ir.Map)
        self.assertEqual(m.fn.kind, "lambda")
        self.assertEqual(m.fn.n_own, 1)
        self.assertEqual(m.fn.param_names, ("x", "b", "k"))       # x*k+b reads b first
        self.assertEqual([type(c).__name__ for c in m.captures], ["ParamRef", "LocalRef"])

    def test_an_operator_verb_is_a_spec_of_kind_op(self):
        p = compile_text('f::{+/x}\n.entry("f";"float[4] -> float")')
        (r,) = nodes(entry_fn(p), ir.Reduce)
        self.assertEqual((r.fn.kind, r.fn.name), ("op", "+"))
        self.assertIsInstance(r.fn.body.stmts[0], ir.Binary)

    def test_several_entries_of_one_function_get_a_type_suffix(self):
        p = compile_text('f::{+/x}\n.entry("f";"float[4] -> float")\n.entry("f";"int[2,3] -> int[3]")\n'
                         'g::{x}\n.entry("g";"int -> int")')
        self.assertEqual([e.c_name for e in p.entries], ["f__f4", "f__i2x3", "g"])

    def test_entry_result_must_fit_the_declaration(self):
        with self.assertRaises(KlongStaticError) as cm:
            compile_text('f::{x%2}\n.entry("f";"int -> int")')
        self.assertEqual(cm.exception.rule, "entry")


class TestStoragePlanningInputs(unittest.TestCase):
    def test_state_and_constants_are_told_apart(self):
        p = compile_program(os.path.join(EMBEDDED, "mavg.kg")).program
        self.assertEqual({g.name: g.state for g in p.globals}, {"buf": True, "pos": True})
        p = compile_program(os.path.join(EMBEDDED, "pid.kg")).program
        self.assertEqual(sorted(g.name for g in p.globals if g.state), ["integ", "prev"])
        self.assertEqual([x.name for x in p.externs], ["read.adc", "write.pwm"])

    def test_array_intermediates_are_listed_as_fused_or_slot(self):
        p = compile_program(os.path.join(EMBEDDED, "fir.kg")).program
        plan = storage_inputs(entry_fn(p, "run"))
        self.assertEqual(plan["arrays"], ["reduce:float[241] slot", "binary:float[16,241] fused", "gather:float[16,241] fused"])
        self.assertEqual(plan["calls"], ["+#0"])

    def test_effects_are_marked(self):
        p = compile_program(os.path.join(EMBEDDED, "pid.kg")).program
        by = {f.name: f for f in p.fns}
        self.assertFalse(ir.effects(by["clamp"]))
        self.assertTrue(ir.effects(by["step"]))
        calls = nodes(by["step"], ir.ExternCall)
        self.assertEqual([c.ext.c_name for c in calls], ["read_adc", "write_pwm"])

    def test_the_package_does_not_need_a_c_compiler_or_optional_modules(self):
        import subprocess
        import sys
        code = ("import sys; import klongpy.static.ir, klongpy.static.infer, klongpy.static.emit_c, klongpy.static.compile; "
                "bad = [m for m in ('torch', 'hypothesis', 'pandas') if m in sys.modules]; sys.exit(1 if bad else 0)")
        self.assertEqual(subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=ROOT).returncode, 0)


if __name__ == "__main__":
    unittest.main()
