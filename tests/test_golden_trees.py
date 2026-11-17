"""Golden parse-tree snapshots and unit tests of klongpy.dump."""
import os
import re
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden"))

import regen  # noqa: E402
from klongpy import KlongInterpreter  # noqa: E402
from klongpy.dump import dump, dump_text, parse_program  # noqa: E402
from klongpy.types import KGQSym, KGSym  # noqa: E402
from tests.backend_compat import is_torch_backend  # noqa: E402


_FLOAT_NUM = re.compile(r"\(num (-?(?:\d+\.\d*(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+))\)")


def _as_float32(text):
    return _FLOAT_NUM.sub(lambda m: "(num %r)" % np.float32(float(m.group(1))).item(), text)


class TestGoldenTrees(unittest.TestCase):
    def test_goldens_exist_for_every_source(self):
        n = 0
        for group, name, _, golden in regen.cases():
            n += 1
            self.assertTrue(os.path.exists(golden), f"missing golden {group}/{name}.sexp")
        self.assertGreater(n, 100)

    def test_no_orphan_goldens(self):
        expected = {golden for _, _, _, golden in regen.cases()}
        for group in regen.GROUPS:
            d = os.path.join(regen.HERE, group)
            for fn in os.listdir(d):
                if fn.endswith(".sexp"):
                    self.assertIn(os.path.join(d, fn), expected, f"orphan golden {group}/{fn}")

    def test_goldens_match(self):
        for group, name, src, golden in regen.cases():
            with self.subTest(group=group, name=name):
                with open(golden, "r") as f:
                    want = f.read()
                got = regen.render(src)
                if is_torch_backend():
                    # A float inside a list literal is an element of a float32 tensor on the
                    # torch backend, so the dump shows its float32 value. The trees are the
                    # same otherwise: compare both with every float rounded to float32.
                    got, want = _as_float32(got), _as_float32(want)
                self.assertEqual(got, want)

    def test_dump_is_deterministic(self):
        for group, name, src, _ in regen.cases():
            if group == "fixtures":
                self.assertEqual(regen.render(src), regen.render(src))


def D(text, compat=False):
    return dump(parse_program(KlongInterpreter(), text), compat).rstrip("\n")


class TestDumpForms(unittest.TestCase):
    def test_num(self):
        self.assertEqual(D("1"), "(num 1)")
        self.assertEqual(D("1.5"), "(num 1.5)")
        self.assertEqual(D("1e3"), "(num 1000.0)")

    def test_char_str(self):
        self.assertEqual(D("0cx"), '(char "x")')
        self.assertEqual(D('"a""b"'), '(str "a\\"b")')  # doubled quote is one quote
        self.assertEqual(D('"a b"'), '(str "a b")')

    def test_str_escapes(self):
        self.assertEqual(dump('a"b\\'), '(str "a\\"b\\\\")\n')

    def test_sym_and_module(self):
        self.assertEqual(D("a"), "(sym a)")
        self.assertEqual(D(".module(:m);a").splitlines()[1], "(sym a`m)")

    def test_qsym_and_compat(self):
        self.assertEqual(D(":a"), "(qsym a)")
        self.assertEqual(D(":a", compat=True), "(sym a)")
        self.assertEqual(dump(KGQSym("a")), "(qsym a)\n")
        self.assertEqual(dump(KGSym("a")), "(sym a)\n")

    def test_arg(self):
        self.assertEqual(D("x"), "(arg x)")
        self.assertEqual(D("z"), "(arg z)")

    def test_list_data_form(self):
        self.assertEqual(D('[1 a "s" 0cx]'), '(list (num 1) (sym a) (str "s") (char "x"))')
        self.assertEqual(D("[1 + 2]"), '(list (num 1) (op "+") (num 2))')
        self.assertEqual(D("[1;2]"), '(list (num 1) (str ";") (num 2))')
        self.assertEqual(D("[1 [2]]"), "(list (num 1) (list (num 2)))")
        self.assertEqual(D("[]"), "(list)")
        self.assertEqual(D("[x]"), "(list (sym x))")

    def test_dict(self):
        self.assertEqual(D(':{[1 2] ["k" [1 2]]}'), '(dict ((num 1) (num 2)) ((str "k") (list (num 1) (num 2))))')

    def test_elist(self):
        self.assertEqual(D("[;1;2+3]"), '(elist (num 1) (d "+" (num 2) (num 3)))')

    def test_monad(self):
        self.assertEqual(D("-x"), '(m "-" (arg x))')

    def test_dyad(self):
        self.assertEqual(D("1+2"), '(d "+" (num 1) (num 2))')
        self.assertEqual(D("1-2"), '(d "-" (num 1) (num 2))')
        self.assertEqual(D("1+2*3"), '(d "+" (num 1) (d "*" (num 2) (num 3)))')

    def test_def(self):
        self.assertEqual(D("a::1"), "(def (sym a) (num 1))")

    def test_fn_and_locals(self):
        self.assertEqual(D("{x+y}"), '(fn 2 (d "+" (arg x) (arg y)))')
        self.assertEqual(dump_text("{}"), "(error UnexpectedChar)\n")  # legacy: empty body is a parse error
        self.assertEqual(D("{[a b];a::x;a}"), "(fn 1 (locals a b) (def (sym a) (arg x)) (sym a))")
        self.assertEqual(D("{[a];1}"), "(fn 0 (locals a) (num 1))")
        self.assertEqual(D("{[a 1];1}"), "(fn 0 (list (sym a) (num 1)) (num 1))")

    def test_call(self):
        self.assertEqual(D("f()"), "(call (sym f))")
        self.assertEqual(D("f(1;2)"), "(call (sym f) (num 1) (num 2))")
        self.assertEqual(D("{x+1}(1)"), '(call (d "+" (arg x) (num 1)) (num 1))')

    def test_proj(self):
        self.assertEqual(D("f(1;)"), "(proj (sym f) (num 1) _)")
        self.assertEqual(D("f(;2)"), "(proj (sym f) _ (num 2))")
        self.assertEqual(D("{x+y}(;2)"), '(proj (d "+" (arg x) (arg y)) _ (num 2))')

    def test_infix(self):
        self.assertEqual(D("1 f 2"), "(infix (sym f) (num 1) (num 2))")
        self.assertEqual(D("1 {x+y} 2"), '(infix (fn 2 (d "+" (arg x) (arg y))) (num 1) (num 2))')

    def test_adv(self):
        self.assertEqual(D("+/x"), '(adv (verb "+") (adverbs "/") (arg x))')
        self.assertEqual(D(",/\\~x"), '(adv (verb ",") (adverbs "/" "\\\\~") (arg x))')
        self.assertEqual(D("f'x"), '(adv (verb (sym f)) (adverbs "\'") (arg x))')

    def test_adv2(self):
        self.assertEqual(D("x+/y"), '(adv2 (verb "+") (adverbs "/") (arg x) (arg y))')
        self.assertEqual(D("1,'2"), '(adv2 (verb ",") (adverbs "\'") (num 1) (num 2))')

    def test_cond(self):
        self.assertEqual(D(":[1;2;3]"), "(cond (num 1) (num 2) (num 3))")
        self.assertEqual(D(":[a;1:|b;2;3]"), "(cond (sym a) (num 1) (cond (sym b) (num 2) (num 3)))")

    def test_opval(self):
        self.assertEqual(D(":!1"), '(opval ":!")')

    def test_missing_operand(self):
        self.assertEqual(D("-"), '(m "-" (missing))')
        self.assertEqual(D("x@"), '(d "@" (arg x) (missing))')

    def test_program_one_line_per_statement(self):
        self.assertEqual(dump(parse_program(KlongInterpreter(), "1;2\n3")), "(num 1)\n(num 2)\n(num 3)\n")

    def test_comment_is_consumed(self):
        self.assertEqual(D('.comment("E")\nfoo\nE\n1'), "(num 1)")

    def test_parse_program_does_not_evaluate_or_leak_module(self):
        k = KlongInterpreter()
        parse_program(k, ".module(:m);a::1")
        self.assertIsNone(k.current_module())
        with self.assertRaises(KeyError):
            k["a"]

    def test_dump_text_reports_error_class(self):
        self.assertEqual(dump_text("(1+2"), "(error UnexpectedChar)\n")


if __name__ == "__main__":
    unittest.main()
