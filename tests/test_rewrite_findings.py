"""
Small bugs found by rewriting real programs: .rs and .r on bad or empty input,
the display of ragged nested lists, .fmt and .pf next to .d, .pf with a
template alone, the error for a { inside a list in a named-parameter
definition, the lint cascade after E001, the order of --check-static flags and
the message of .l for a file that is not found.
"""
import io
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from klongpy import KlongInterpreter
from klongpy.core import KGChannel, KGChannelDir, KlongException
from klongpy.errors import KlongRangeError, KlongTypeError
from klongpy.lint import lint_source

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def capture(klong):
    buf = io.StringIO()
    klong.console(write=buf.write)
    return buf


def is_undefined(klong, v):
    klong['v'] = v
    return bool(klong(':_v'))


class TestReadString(unittest.TestCase):
    def setUp(self):
        self.klong = KlongInterpreter()

    def test_a_non_string_is_a_type_error(self):
        for src in ('.rs(5)', '.rs([1 2])', '.rs(0cx)', '.rs(:a)'):
            with self.assertRaises(KlongTypeError, msg=src) as cm:
                self.klong(src)
            self.assertIn(".rs: x must be a string, got", str(cm.exception))

    def test_empty_or_blank_text_is_undefined(self):
        for src in ('.rs("")', '.rs("   ")'):
            self.assertTrue(is_undefined(self.klong, self.klong(src)), src)

    def test_text_still_reads(self):
        self.assertEqual(self.klong('.rs("7")'), 7)
        self.assertEqual(list(self.klong('.rs("[1 2]")')), [1, 2])

    def test_r_at_end_of_input_is_undefined(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "e.txt")
            open(path, "w").close()
            self.klong(f'.fc(.ic("{path}"))')
            try:
                self.assertTrue(is_undefined(self.klong, self.klong('.r()')))
            finally:
                self.klong('.cc(.fc(0))')


class TestRaggedDisplay(unittest.TestCase):
    CASES = [
        ('[;[[1 2 3]];[4]]', '[[[1 2 3]] [4]]'),
        ('[;[1 2];[3]]', '[[1 2] [3]]'),
        ('[;[[1 2] [3 4]];[5]]', '[[[1 2] [3 4]] [5]]'),
        ('[;[;[1 2];[3]];[;4]]', '[[[1 2] [3]] [4]]'),
        ('[;[[1 2 3]];[[4 5 6]]]', '[[[1 2 3]] [[4 5 6]]]'),
        ('[;[;[[1 2 3]];[4]];"a"]', '[[[[1 2 3]] [4]] a]'),
    ]

    def test_display_print_and_format(self):
        for src, want in self.CASES:
            k = KlongInterpreter()
            out = capture(k)
            k(f'.d({src})')
            self.assertEqual(out.getvalue(), want, src)
            out.truncate(0)
            out.seek(0)
            k(f'.p({src})')
            self.assertEqual(out.getvalue(), want + "\n", src)
            self.assertEqual(k(f'.fmt("{{}}";,{src})'), want, src)


class TestFormatLikeDisplay(unittest.TestCase):
    def shown(self, k, name):
        out = capture(k)
        k(f'.d({name})')
        return out.getvalue()

    def test_a_float32_scalar_shows_as_display_does(self):
        k = KlongInterpreter()
        k['f'] = np.float32(68.76)
        self.assertEqual(self.shown(k, 'f'), "68.76")
        self.assertEqual(k('.fmt("{}";f)'), "68.76")
        k['l'] = ['e', np.float32(68.76)]
        self.assertEqual(k('.fmt("{} {}";l)'), "e 68.76")

    def test_a_float32_array_shows_each_member_short(self):
        k = KlongInterpreter()
        k['f'] = np.array([68.76, 1.2], dtype=np.float32)
        self.assertEqual(self.shown(k, 'f'), "[68.76 1.2]")
        self.assertEqual(k('.fmt("{}";,f)'), "[68.76 1.2]")

    def test_fmt_agrees_with_display_for_computed_values(self):
        k = KlongInterpreter()
        k('a::1.2')
        k('g::[1 2 3]∇{+/x^2}')
        for expr in ('a*57.3', 'a+1'):
            self.assertEqual(k(f'.fmt("{{}}";{expr})'), self.shown_expr(k, expr), expr)
        # the members of a list are the arguments, each shown as .d shows it
        # (enlisting g first would narrow it to float32 on the torch backend)
        want = " ".join(self.shown_expr(k, f'g@{i}') for i in range(3))
        self.assertEqual(k('.fmt("{} {} {}";g)'), want)

    def shown_expr(self, k, expr):
        k(f'v::{expr}')
        return self.shown(k, 'v')


class TestFormatTemplateAlone(unittest.TestCase):
    def setUp(self):
        self.klong = KlongInterpreter()

    def test_no_placeholders_needs_no_arguments(self):
        self.assertEqual(self.klong('.fmt("done")'), "done")
        self.assertEqual(self.klong('.fmt("a {{b}}")'), "a {b}")
        out = capture(self.klong)
        self.assertEqual(self.klong('.pf("text")'), "text")
        self.assertEqual(out.getvalue(), "text\n")

    def test_placeholders_without_arguments_is_an_error(self):
        for src in ('.fmt("{}")', '.pf("{} {}")', '.fmt("{1} {0}")'):
            with self.assertRaises(KlongRangeError, msg=src) as cm:
                self.klong(src)
            self.assertIn("but no arguments", str(cm.exception))

    def test_arity_is_still_checked(self):
        with self.assertRaises(KlongTypeError):
            self.klong('.fmt("a";1;2)')
        with self.assertRaises(KlongTypeError):
            self.klong('.pf()')

    def test_projection_and_assignment_still_work(self):
        self.assertEqual(self.klong('f::.fmt("{}";);f(5)'), "5")
        self.assertEqual(self.klong('g::.fmt;g("a")'), "a")
        self.assertEqual(self.klong('.fmt("{} {}";[;3;2.5])'), "3.0 2.5")
        self.assertEqual(self.klong('.fmt("{} {}";[;$3;2.5])'), "3 2.5")


class TestNamedDefinitionError(unittest.TestCase):
    def message(self, src):
        with self.assertRaises(KlongException) as cm:
            KlongInterpreter()(src)
        return cm.exception

    def test_a_brace_in_a_list_is_named_as_the_cause(self):
        src = "f(a;b)::{[{:[x=1;[];a]}'!2;b]}"
        e = self.message(src)
        self.assertIn("a { inside a [ ] list is not a function", str(e))
        self.assertEqual(e.span, (src.index("{:["), src.index("{:[") + 1))
        self.assertIn("[;", e.hint)

    def test_other_bad_right_sides_keep_the_old_message(self):
        for src in ("f(a;b)::5", "f(a;b)::a+b", "f(a;b)::"):
            self.assertIn("needs a function body in braces", str(self.message(src)), src)

    def test_evaluated_lists_and_good_definitions_are_untouched(self):
        k = KlongInterpreter()
        k("f(a;b)::{a+b}")
        self.assertEqual(k("f(1;2)"), 3)
        k("h(a;b)::{[;{x};a]}")
        k("h(1;2)")


class TestLintCascade(unittest.TestCase):
    def test_a_failed_definition_defines_its_name_and_parameters(self):
        text = 'left(m)::vis\'m\nnums(s)::.rs("[",s,"]")\nq::left([1 2])\nr::nums("1 2")\n'
        ds = lint_source(text, "t.kg")
        self.assertEqual([d.code for d in ds if d.code[0] == 'E'], ['E001', 'E001'])
        w002 = [(d.line, d.message) for d in ds if d.code == 'W002']
        self.assertEqual(w002, [(1, "'vis' is never defined in this file")])

    def test_other_names_are_still_reported(self):
        ds = lint_source('left(m)::vis\'m\nz::m+other\n', "t.kg")
        names = [d.message.split("'")[1] for d in ds if d.code == 'W002']
        self.assertEqual(names, ['vis', 'm', 'other'])


class TestLoadMessage(unittest.TestCase):
    def test_the_error_lists_the_places_it_looked(self):
        with tempfile.TemporaryDirectory() as d:
            prog = os.path.join(d, "prog.kg")
            with open(prog, "w") as f:
                f.write('.l("common.kg")\n')
            r = subprocess.run([sys.executable, "-m", "klongpy.cli", prog], capture_output=True, text=True,
                               cwd=tempfile.gettempdir(), timeout=120,
                               env={**os.environ, "KLONGPATH": "/nowhere/a:/nowhere/b"})
        self.assertIn("file does not exist: common.kg", r.stderr)
        self.assertIn("/nowhere/a/common.kg", r.stderr)
        self.assertIn("/nowhere/b/common.kg", r.stderr)
        # the directory of the program is the last place tried
        self.assertIn(os.path.join(os.path.basename(d), "common.kg"), r.stderr)


if __name__ == "__main__":
    unittest.main()
