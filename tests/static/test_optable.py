"""
The operator table (klongpy/static/optable.py): it covers every operator the
interpreter has, its vocabularies are closed, and every example of an allowed
row is inside the profile with the result type the row names, both for the
checker and, where the design says the two agree, for KlongPy itself.
"""
import unittest

import numpy as np

from klongpy import KlongInterpreter
from klongpy.static import FORMS, OPTABLE, RULES, check_source, parse_signature
from klongpy.static.host import call_entry, load_program
from klongpy.static.optable import BY_ID, OPERAND_KINDS, SHAPE_RULES, TYPE_RULES, lookup
from klongpy.types import is_adverb

from .hostutil import to_np

# Rows whose element type on the host depends on the value:
# KlongPy's ^ returns an int whenever the result is whole, and a conditional
# whose branches are an int and a float has the type of the branch taken
# (the profile's type is float; the value is the same).
HOST_ELEM_DIFFERS = {"pow", "conditional"}

ADVERBS = ["'", ":\\", ":'", ":/", "/", ":~", ":*", "\\", "\\~", "\\*", "@'"]


def sample(ty, seed):
    """A value of a type: small positive numbers that differ from one another."""
    if not ty.shape:
        return seed + 2 if ty.elem == "int" else seed + 1.5
    n = int(np.prod(ty.shape))
    flat = np.arange(1, n + 1) + seed
    return (flat if ty.elem == "int" else flat * 0.5 + 0.25).reshape(ty.shape)


def host_type(v):
    """(elem, shape) of a KlongPy value."""
    if isinstance(v, np.ndarray):
        return ("float" if v.dtype.kind == "f" else "int"), tuple(v.shape)
    return ("float" if isinstance(v, (float, np.floating)) else "int"), ()


class TestCoverage(unittest.TestCase):
    def test_every_operator_of_the_interpreter_has_a_row(self):
        klong = KlongInterpreter()
        for glyph in klong._vm:
            self.assertIsNotNone(lookup(glyph, "monad"), f"monad {glyph}")
        for glyph in klong._vd:
            self.assertIsNotNone(lookup(glyph, "dyad"), f"dyad {glyph}")

    def test_every_adverb_has_a_row(self):
        for glyph in ADVERBS:
            self.assertTrue(is_adverb(glyph))
            self.assertTrue(lookup(glyph, "adverb1") or lookup(glyph, "adverb2"), glyph)

    def test_no_row_for_an_operator_that_does_not_exist(self):
        klong = KlongInterpreter()
        for op in OPTABLE:
            if op.valence == "monad":
                self.assertIn(op.glyph, klong._vm, op.id)
            elif op.valence == "dyad":
                self.assertIn(op.glyph, klong._vd, op.id)
            else:
                self.assertIn(op.glyph, ADVERBS, op.id)

    def test_vocabularies_are_closed(self):
        for op in OPTABLE:
            with self.subTest(row=op.id):
                self.assertIn(op.status, ("allowed", "rejected"))
                if op.status == "allowed":
                    self.assertIn(op.type_rule, TYPE_RULES)
                    self.assertIn(op.shape_rule, SHAPE_RULES)
                    self.assertIsNone(op.rule)
                    for k in op.operands:
                        self.assertIn(k, OPERAND_KINDS)
                    if op.id != "define":
                        self.assertTrue(op.examples, "an allowed row shows at least one example")
                else:
                    self.assertIn(op.rule, RULES)
                    self.assertTrue(op.note)
        self.assertEqual(len(BY_ID), len(OPTABLE))


class TestExamples(unittest.TestCase):
    """Each example (body, signature): f::{body} entered with the signature is accepted and has that result type."""

    def programs(self):
        for op in OPTABLE:
            for ex in op.examples:
                body, sig = ex[0], ex[1]
                yield op.id, f'f::{{{body}}}\n.entry("f";"{sig}")', sig, (ex[2] if len(ex) > 2 else None)
        for form in FORMS:
            for prog, sig in form.examples:
                yield form.id, f'{prog}\n.entry("f";"{sig}")', sig, None

    def test_the_checker_accepts_each_example_with_the_declared_result_type(self):
        for row, prog, sig, _ in self.programs():
            with self.subTest(row=row, prog=prog):
                r = check_source(prog, "t.kg")
                self.assertEqual([str(d) for d in r.diagnostics], [])
                want = parse_signature(sig)
                (args, result), = r.functions["f"].specializations.items()
                self.assertEqual(tuple(args), want.params)
                self.assertEqual(result, want.result, "the inferred result type is the declared one, exactly")

    def test_klongpy_gives_a_value_of_that_type(self):
        stubs = {"read.adc": lambda ch: 0.5, "log.msg": lambda s, n: None}
        for row, prog, sig, inputs in self.programs():
            with self.subTest(row=row, prog=prog):
                want = parse_signature(sig)
                klong = load_program("t.kg", text=prog, externs=stubs)
                args = inputs if inputs is not None else [sample(p, i) for i, p in enumerate(want.params)]
                value = klong["f"](*[np.asarray(a) if isinstance(a, list) else a for a in args]) if want.arity else klong("f()")
                elem, shape = host_type(to_np(klong, value))
                self.assertEqual(shape, tuple(want.result.shape))
                if row not in HOST_ELEM_DIFFERS:
                    self.assertEqual(elem, want.result.elem)
                # and call_entry, which checks arguments and result against the signature, agrees
                call_entry(klong, "f", *args)

    def test_rejected_rows_are_rejected_with_their_rule(self):
        written = {"monad": "{0}x", "dyad": "x{0}y", "adverb1": "{{x}}{0}x", "adverb2": "x{{x}}{0}y"}
        for op in OPTABLE:
            if op.status == "allowed":
                continue
            with self.subTest(row=op.id):
                body = written[op.valence].format(op.glyph)
                if op.id == "each-left" or op.id == "each-right":
                    body = f"x+{op.glyph}y"
                r = check_source(f"f::{{{body}}}", "t.kg")
                self.assertIn(op.rule, [d.rule for d in r.diagnostics], r.render())


if __name__ == "__main__":
    unittest.main()
