"""
Tests for klongpy.lint and `kgpy lint` (design in
docs/lint.md).

Layout: the rule table (one positive and one negative case per code, with the
line:col of every finding), suppression, select/ignore, cross-file names,
"never evaluates", the CLI by subprocess, robustness (mutated sources, no
optional packages) and the shipped .kg files.
"""
import glob
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import unittest
import warnings

from klongpy.lexer import Source
from klongpy.lint import RULES, Diagnostic, lint_file, lint_main, lint_source, parse_codes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def found(text, code=None, **kw):
    """[(line, col)] of the findings (of one code), or the codes when code is None."""
    ds = lint_source(text, "t.kg", **kw)
    if code is None:
        return [d.code for d in ds]
    return [(d.line, d.col) for d in ds if d.code == code]


# (code, source, [(line, col) of every finding of that code]). [] is a negative case.
CASES = [
    # W001 _ between name characters
    ("W001", "my_var::5", [(1, 1)]),
    ("W001", "a::1;b::2;a+b_c", [(1, 13)]),
    ("W001", "x_a", []),                    # x_a is "drop x from a": an argument on one side
    ("W001", "n::5;n_x", []),
    ("W001", "a _ b", []),
    ("W001", "a_1", []),
    ("W001", "1_x", []),
    ("W001", "my.var::5", []),
    # W002 names: use before definition, never defined
    ("W002", "b::a;a::1", [(1, 4)]),
    ("W002", "g::{zz}", [(1, 5)]),
    ("W002", "zz", [(1, 1)]),
    ("W002", "a::1;b::a", []),
    ("W002", "f::{a};a::1", []),            # a use inside a body before a later definition is fine
    ("W002", "s::[3 1 2];s@<s::s", []),     # right to left: s is defined in the statement that uses it
    ("W002", '.p("x")', []),                # a system function
    ("W002", "f::{[a];a::x;a}", []),        # a declared local
    ("W002", "f::{x};f(1)", []),
    ("W002", "[a b]::[1 2];a+b", []),       # destructuring defines a and b
    ("W002", "f(a;b)::{a+b};f(1;2)", []),   # named parameters
    ('W002', '.bkf(["exp" "sin"]);exp(1.0)', []),
    ("W002", '.py("m");zz', []),            # .py defines unknown names: W002 is off for the file
    ("W002", '.extern("read.adc";"int -> float");read.adc(0)', []),        # .extern binds its name
    ("W002", '.extern("write.pwm";"int float -> void";"hw");write.pwm(0;1.0)', []),
    ("W002", 'read.adc(0);.extern("read.adc";"int -> float")', [(1, 1)]),   # used before the declaration
    ("W002", 'setup::{.extern("read.adc";"int -> float")};f::{read.adc(x)}', []),   # declared inside a function
    ("W002", '.extern("read.adc";"int -> float");read.dac(0)', [(1, 36)]),  # another name is still undefined
    ("W002", 'n::"a";.extern(n;"int -> float");zz(0)', []),                # a computed name: W002 is off for the file
    ("W002", 'f::{x};.entry("f";"int -> int");f(1)', []),
    # W003 declared local never used
    ("W003", "f::{[a;b];a::1}", [(1, 8)]),
    ("W003", "f::{[a b]::x;a}", [(1, 8)]),
    ("W003", "f::{[a];[b c]::x;b+c}", [(1, 6)]),
    ("W003", "f::{[a];a::1;a}", []),
    ("W003", "f::{[a];{a}(1)}", []),        # used by a nested function (dynamic scope)
    ("W003", "f::{[a];a::x;g()};g::{a+1}", []),     # read by a function that f calls (dynamic scope)
    ("W003", "f::{[a;b];a::1};g::{[b];b}", [(1, 8)]),   # another function's own local b is not a reader
    # W010 leading - then a number and a dyad
    ("W010", "-1+2", [(1, 1)]),
    ("W010", "a::-1+2", [(1, 4)]),
    ("W010", "f::{-1<x}", [(1, 5)]),
    ("W010", "x@-1+2", [(1, 3)]),
    ("W010", "1-1+2", []),                  # a dyadic minus
    ("W010", "a-1+2", []),
    ("W010", "-1*2", []),                   # the sign commutes with * and %: nothing to warn about
    ("W010", "-1", []),
    ("W010", "-(1+2)", []),
    ("W010", "[-1+2]", []),                 # inside a list: W017
    # W011 function uses y without x (or z without y), and is called short
    ("W011", "f::{y};f(5)", [(1, 4)]),
    ("W011", "{y}(5)", [(1, 1)]),
    ("W011", "f::{x+z};f(1;2)", [(1, 4)]),
    ("W011", "f::{y};f(1;5)", []),
    ("W011", "{y}/[1 2 3]", []),            # an adverb operand
    ("W011", "f::{y}", []),                 # never called short: a fixed-signature callback
    ("W011", "{-x}(1)", []),
    ("W011", "{x+y}(1)", []),
    ("W011", "f::{[y];y::x;y};f(1)", []),   # y is a local here, not the argument
    # W012 bare name in a list is a symbol
    ("W012", "a::1;[a b]", [(1, 7)]),
    ("W012", "f::{[a];a::1;[a 2]}", [(1, 15)]),
    ("W012", "[a b]", []),                  # not defined: nothing to confuse
    ("W012", "a::1;[;a;2]", []),            # the evaluated form
    ("W012", 'a::1;:{;"k";[;a;2]}', []),
    ("W012", "a::1;[:a 2]", []),
    ("W012", "a::1;:{[a 1]}", []),          # a dictionary entry: the key is a symbol
    ("W012", 'r::1;.pyc([r "x"];[];:{})', []),   # .pyc looks the object up by name
    ("W012", "a::1;f:>[a a]", []),          # :> takes variable names
    # W013 :[ as the else branch of :[
    ("W013", ":[1;2;:[3;4;5]]", [(1, 7)]),
    ("W013", ":[1;2;\n :[3;4;5]\n]", [(2, 2)]),
    ("W013", ":[1;2:|3;4;5]", []),
    ("W013", ":[1;:[2;3;4];5]", []),        # in the then branch
    ("W013", ":[1;2;:[3;4;5]+1]", []),      # the else branch is more than the nested :[
    # W014 whitespace before ( or an adverb
    ("W014", "f (1)", [(1, 3)]),
    ("W014", "{x} '[1 2]", [(1, 5)]),
    ("W014", "a::{x};a (1)", [(1, 10)]),
    ("W014", "a::f (1)", [(1, 6)]),
    ("W014", "{x} (1)", [(1, 5)]),
    ("W014", "f(1)", []),
    ("W014", "{x}'[1 2]", []),
    ("W014", "1 f (2)", []),                # f is the infix verb and (2) its operand
    # W015 newline after a comment
    ("W015", 'f::{x :"c"\n+2}', [(1, 7)]),
    ("W015", 'f::{:[1 :"c"\n;2;3]}', [(1, 9)]),
    ("W015", 'f(1 :"c"\n;2)', [(1, 5)]),
    ("W015", '[;1 :"c"\n+2;3]', [(1, 5)]),
    ("W015", "f::{x\n+2}", []),
    ("W015", '[1 2 :"c"\n3]', []),          # not in a data list
    ("W015", 'a::1 :"c"\nb::2', []),        # at top level a newline is a separator anyway
    # W016 : before a digit
    ("W016", ":123", [(1, 1)]),
    ("W016", "a:1", [(1, 2)]),
    ("W016", "[1 :2]", [(1, 4)]),
    ("W016", "x::123", []),
    ("W016", ":abc", []),
    # W017 ; bracket or operator in a data list
    ("W017", "[1;2]", [(1, 3)]),
    ("W017", "[1 + 2]", [(1, 4)]),
    ("W017", "[1 (2)]", [(1, 4)]),
    ("W017", "[1 {2}]", [(1, 4)]),
    ("W017", ":{[1;2]}", [(1, 5)]),
    ("W017", "[1 2 3]", []),
    ("W017", "[1 -2]", []),                 # -2 is one number
    ("W017", "f::{[a;b];a}", []),           # a locals declaration
    ("W017", "[;1;2]", []),                 # the evaluated list
    ("W017", ":{[1 2]}", []),
    ("W017", ':{;"a";[;1;2]}', []),         # the evaluated dictionary :{;k;v}: its body is code
    # W018 number immediately followed by a name
    ("W018", "10c", [(1, 1)]),
    ("W018", "2x", [(1, 1)]),
    ("W018", "[2x]", [(1, 2)]),
    ("W018", "10 c", []),
    ("W018", "1e5", []),
    ("W018", "0cx", []),
    ("W018", "a1", []),
    # W019 assignment to x, y or z
    ("W019", "{x::5;x}(1)", [(1, 2)]),
    ("W019", "{[a];z::1;a}", [(1, 6)]),
    ("W019", "{y::2}", [(1, 2)]),
    ("W019", "y::2", []),                       # top level: an ordinary variable
    ("W019", "{[x];x::5;x}(1)", []),        # a declared local named x
    ("W019", "{a::x}", []),
    # W020 a two-element list joined to a dictionary
    ("W020", "[1 2],:{[3 4]}", [(1, 6)]),
    ("W020", "[1 -2],:{[3 4]}", [(1, 7)]),
    ("W020", "a::[1 2];a,:{[3 4]}", [(1, 11)]),
    ("W020", "[1 2 3],:{[3 4]}", []),
    ("W020", "[1 2],,:{[3 4]}", []),
    ("W020", "a::[1 2 3];a,:{[3 4]}", []),
    ("W020", "[],:{[3 4]}", []),
    # E001 the parser rejects it at strict level 1
    ("E001", "1+", [(1, 2)]),
    ("E001", "f (1)", [(1, 3)]),
    ("E001", "1::5", [(1, 1)]),             # a level-1 error that level 0 accepts silently
    ("E001", '"abc', [(1, 1)]),
    ("E001", "[1 2", [(1, 1)]),
    ("E001", "(1+2", [(1, 5)]),
    ("E001", "1+2", []),
    # E002 more arguments than the callee takes
    ("E002", "f::{x};f(1;2)", [(1, 8)]),
    ("E002", "{x}(1;2)", [(1, 1)]),
    ("E002", "f(a;b)::{a+b};f(1;2;3)", [(1, 15)]),
    ("E002", "f::{x};f(1;)", [(1, 8)]),
    ("E002", "f::{x+z};f(1;2;3)", []),      # z is the highest argument used: it takes three
    ("E002", "f::{x};f(1)", []),
    ("E002", "f::{{x+y}};f(1;2)", []),      # a body that is a function takes its arguments on
    ("E002", "f::{x};f::{x+y};f(1;2)", []), # two definitions: the larger wins
    ("E002", "g::{[f];f::{x};f(1;2)}", []), # f is a local here
]


class TestRuleTable(unittest.TestCase):
    def test_every_code_has_a_positive_and_a_negative_case(self):
        for code in RULES:
            pos = [c for c in CASES if c[0] == code and c[2]]
            neg = [c for c in CASES if c[0] == code and not c[2]]
            self.assertTrue(pos, f"{code}: no positive case")
            self.assertTrue(neg, f"{code}: no negative case")
        self.assertEqual(set(c[0] for c in CASES) - set(RULES), set())

    def test_cases(self):
        for code, text, want in CASES:
            with self.subTest(code=code, text=text):
                self.assertEqual(found(text, code), want)

    def test_the_observed_inputs(self):
        """The observed inputs give the codes it names."""
        for text, code in [("my_var::5", "W001"), ("-1+2", "W010"), ("f::{y};f(5)", "W011"),
                           ("a::1;[a b]", "W012"), ("f (1)", "W014"), ("{x} '[1 2]", "W014"),
                           ('{x :"c"\n+2}(1)', "W015"), (":123", "W016"), ("[1;2]", "W017"),
                           ("[1 + 2]", "W017"), ("10c", "W018"), ("{x::5;x}(1)", "W019")]:
            with self.subTest(text=text):
                self.assertIn(code, found(text))

    def test_acceptance_of_w002_and_e002(self):
        self.assertEqual(found("a::1;b::a", "W002"), [])
        self.assertEqual(found("b::a;a::1", "W002"), [(1, 4)])
        self.assertEqual(found("f::{a};a::1", "W002"), [])
        self.assertEqual(found("g::{zz}", "W002"), [(1, 5)])
        self.assertEqual(found("f::{x};f(1;2)", "E002"), [(1, 8)])
        self.assertEqual(found("f::{x+z};f(1;2;3)", "E002"), [])

    def test_a_clean_program_has_no_findings(self):
        text = ("sq::{x*x}\n"
                "sum::{[a];a::0;{a::a+x}'x;a}\n"
                "r::sum([1 2 3])\n"
                "s::sq(r)\n"
                ":[r>1;s;0]\n")
        self.assertEqual(lint_source(text, "t.kg"), [])

    def test_a_syntax_error_does_not_stop_the_token_rules(self):
        codes = found("my_var::5\n1+\nx::10c\n")
        self.assertIn("E001", codes)
        self.assertIn("W001", codes)
        self.assertIn("W018", codes)

    def test_parsing_resumes_after_an_error(self):
        # three separate problems on three lines are three findings
        self.assertEqual(found("a::1\nf (1)\ng (2)\n1+\n", "E001"), [(2, 3), (3, 3), (4, 2)])

    def test_messages_and_hints_are_one_line(self):
        for code, text, want in CASES:
            if not want:
                continue
            for d in lint_source(text, "t.kg", select=[code]):
                self.assertNotIn("\n", d.message)
                self.assertTrue(d.hint is None or "\n" not in d.hint)
                self.assertEqual(d.severity, RULES[code].severity)


class TestDiagnostic(unittest.TestCase):
    def test_fields(self):
        (d,) = lint_source("a::1\nb::-1+2", "dir/x.kg", select="W010")
        self.assertIsInstance(d, Diagnostic)
        self.assertEqual((d.code, d.severity, d.line, d.col), ("W010", "warning", 2, 4))
        self.assertEqual(d.span, (8, 11))
        self.assertIsInstance(d.source, Source)
        self.assertEqual(d.source.name, "dir/x.kg")
        self.assertTrue(d.hint)
        self.assertTrue(str(d).startswith("dir/x.kg:2:4: W010 "))

    def test_severities(self):
        self.assertEqual({c: r.severity for c, r in RULES.items() if c[0] == "E"},
                         {"E001": "error", "E002": "error"})
        self.assertTrue(all(r.severity == "warning" for c, r in RULES.items() if c[0] == "W"))
        self.assertEqual(set(RULES), {"W001", "W002", "W003", "W010", "W011", "W012", "W013", "W014", "W015",
                                      "W016", "W017", "W018", "W019", "W020", "E001", "E002"})

    def test_sorted_by_position(self):
        ds = lint_source("[1;2]\nmy_var::5\n-1+2\n", "t.kg")
        pos = [d.span[0] for d in ds]
        self.assertEqual(pos, sorted(pos))

    def test_empty_and_comment_only(self):
        self.assertEqual(lint_source("", "t.kg"), [])
        self.assertEqual(lint_source(':" nothing here "\n', "t.kg"), [])

    def test_comment_directive_text_is_skipped(self):
        # the body of .comment("X") is not code
        self.assertEqual(lint_source('.comment("END")\nmy_var::5 [1;2]\nEND\na::1\n', "t.kg"), [])


class TestSuppression(unittest.TestCase):
    def test_removes_that_code_on_that_line_only(self):
        text = 'a::1;[a b] :"lint:ignore W012"\nb::1;[b a]\n'
        self.assertEqual(found(text, "W012"), [(2, 7), (2, 9)])

    def test_other_codes_on_the_line_stay(self):
        text = 'a::1;[a;b] :"lint:ignore W012"\n'
        self.assertEqual(found(text), ["W017"])
        text = 'a::1;[a;b] :"lint:ignore W017"\n'
        self.assertEqual(found(text), ["W012"])

    def test_several_codes(self):
        text = 'a::1;[a;b] :"lint:ignore W012 W017"\n'
        self.assertEqual(found(text), [])
        text = 'a::1;[a;b] :"lint:ignore W012, W017"\n'
        self.assertEqual(found(text), [])

    def test_the_comment_does_not_reach_the_next_line(self):
        text = ':"lint:ignore W016"\n:123\n'
        self.assertEqual(found(text), ["W016"])

    def test_can_suppress_errors(self):
        self.assertEqual(found('1+ :"lint:ignore E001"\n'), [])

    def test_a_comment_that_merely_mentions_it_does_nothing(self):
        self.assertEqual(found(':"lint:ignore"\n:123\n'), ["W016"])
        self.assertEqual(found(':"remember to lint:ignore W016 someday"\n:123\n'), ["W016"])

    def test_inside_a_function_body_and_after_a_multiline_statement(self):
        text = 'f::{[a];a::1; :123 :"lint:ignore W016"\na}\n'
        self.assertEqual(found(text, "W016"), [])


class TestSelectIgnore(unittest.TestCase):
    TEXT = "my_var::5\n[1;2]\n-1+2\n"

    def test_select(self):
        self.assertEqual(found(self.TEXT, select="W001"), ["W001"])
        self.assertEqual(sorted(found(self.TEXT, select="W001,W010")), ["W001", "W010"])
        self.assertEqual(found(self.TEXT, select=["W017"]), ["W017"])

    def test_ignore(self):
        self.assertNotIn("W017", found(self.TEXT, ignore="W017"))
        self.assertIn("W001", found(self.TEXT, ignore="W017"))
        self.assertEqual(found(self.TEXT, ignore="W001,W002,W010,W017"), [])

    def test_ignore_wins_over_select(self):
        self.assertEqual(found(self.TEXT, select="W001", ignore="W001"), [])

    def test_unknown_code(self):
        with self.assertRaises(ValueError):
            lint_source("1", "t.kg", select="W999")
        with self.assertRaises(ValueError):
            parse_codes("W001,nope")
        self.assertEqual(parse_codes(" w001 , e002 "), {"W001", "E002"})
        self.assertIsNone(parse_codes(None))


class TestCrossFile(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def write(self, name, text):
        path = os.path.join(self.dir.name, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
        return path

    def lint(self, name, **kw):
        path = os.path.join(self.dir.name, name)
        return [(d.code, d.line, d.col) for d in lint_file(path, **kw)]

    def test_names_of_a_loaded_file_are_defined(self):
        self.write("lib.kg", "helper::{x+1}\nconst::5\n")
        self.write("main.kg", '.l("lib.kg")\nhelper(const)\n')
        self.assertEqual(self.lint("main.kg"), [])

    def test_the_extension_is_optional_and_subdirectories_work(self):
        self.write("sub/lib.kg", "helper::{x+1}\n")
        self.write("main.kg", '.l("sub/lib")\nhelper(1)\n')
        self.assertEqual(self.lint("main.kg"), [])

    def test_a_use_before_the_load_is_flagged(self):
        self.write("lib.kg", "helper::{x+1}\n")
        self.write("main.kg", 'helper(1)\n.l("lib.kg")\n')
        self.assertEqual(self.lint("main.kg"), [("W002", 1, 1)])

    def test_a_name_the_loaded_file_does_not_define_is_still_flagged(self):
        self.write("lib.kg", "helper::{x+1}\n")
        self.write("main.kg", '.l("lib.kg")\nhelper(other)\n')
        self.assertEqual(self.lint("main.kg"), [("W002", 2, 8)])

    def test_loads_are_followed(self):
        self.write("a.kg", '.l("b.kg")\nfa::{x}\n')
        self.write("b.kg", "fb::{x}\n")
        self.write("main.kg", '.l("a.kg")\nfa(fb(1))\n')
        self.assertEqual(self.lint("main.kg"), [])

    def test_arity_of_a_function_from_a_loaded_file(self):
        self.write("lib.kg", "helper::{x+1}\n")
        self.write("main.kg", '.l("lib.kg")\nhelper(1;2)\n')
        self.assertEqual(self.lint("main.kg"), [("E002", 2, 1)])

    def test_a_cycle_terminates(self):
        self.write("a.kg", '.l("b.kg")\nfa::{x}\n')
        self.write("b.kg", '.l("a.kg")\nfb::{x}\n')
        self.assertEqual(self.lint("a.kg"), [])

    def test_a_non_literal_load_turns_w002_off(self):
        self.write("main.kg", 'n::"lib.kg"\n.l(n)\nunknown(1)\n')
        self.assertEqual(self.lint("main.kg"), [])

    def test_a_missing_file_turns_w002_off(self):
        self.write("main.kg", '.l("nowhere.kg")\nunknown(1)\n')
        self.assertEqual(self.lint("main.kg"), [])

    def test_python_imports_turn_w002_off(self):
        for imp in ('.py("m")', '.pyf("m";"f")', '.pya(x;"a")'):
            self.write("main.kg", imp + "\nunknown(1)\n")
            self.assertEqual(self.lint("main.kg"), [], imp)

    def test_resolve_false_never_reads_files(self):
        self.write("lib.kg", "helper::{x+1}\n")
        self.write("main.kg", '.l("lib.kg")\nunknown(1)\n')
        self.assertEqual(self.lint("main.kg", resolve=False), [])

    def test_the_library_is_found(self):
        # klongpy/lib/csv.kg defines csv.make (a dotted name) and csv.tbl
        self.write("main.kg", '.l("csv")\ncsv.make(1)\ncsv.nosuch(2)\n')
        self.assertEqual(self.lint("main.kg"), [("W002", 3, 1)])

    def test_a_syntax_error_in_the_loaded_file_is_not_reported_in_the_loader(self):
        self.write("lib.kg", "helper::{x+1}\n1+\n")
        self.write("main.kg", '.l("lib.kg")\nhelper(1)\n')
        self.assertEqual(self.lint("main.kg"), [])


class TestNeverEvaluates(unittest.TestCase):
    def test_exit_is_not_run(self):
        self.assertEqual(found(".x(3)"), [])

    def test_a_loaded_file_is_not_run(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "boom.kg"), "w") as f:
                f.write(".x(3)\n{1}{1}:~0\n")
            with open(os.path.join(d, "main.kg"), "w") as f:
                f.write('.l("boom.kg")\n.x(4)\n')
            self.assertEqual(lint_file(os.path.join(d, "main.kg")), [])

    def test_nothing_is_written(self):
        with tempfile.TemporaryDirectory() as d:
            marker = os.path.join(d, "marker")
            with open(os.path.join(d, "main.kg"), "w") as f:
                f.write(f'.sys.cout::0;.pya("x";"y");.sys("touch {marker}")\n.rs("{marker}")\n')
            lint_file(os.path.join(d, "main.kg"))
            self.assertFalse(os.path.exists(marker))

    def test_python_modules_are_not_imported(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "evil.py"), "w") as f:
                f.write("raise SystemExit(7)\n")
            text = f'.py("{os.path.join(d, "evil")}")\n'
            self.assertEqual(lint_source(text, "t.kg"), [])


class Cli(unittest.TestCase):
    """kgpy lint by subprocess. Time-bounded; stdin is closed so a REPL would end at once, and is detected by its output."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def write(self, name, text):
        path = os.path.join(self.dir.name, name)
        with open(path, "w") as f:
            f.write(text)
        return path

    def run_cli(self, *args, stdin=None):
        env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
        env.pop("KLONGPATH", None)
        p = subprocess.run([sys.executable, "-W", "ignore", "-m", "klongpy.cli", "lint", *args],
                           capture_output=True, text=True, timeout=120, cwd=self.dir.name, env=env,
                           input=stdin if stdin is not None else "")
        return p.returncode, p.stdout, p.stderr

    LINE = re.compile(r"^(?P<file>.+):(?P<line>\d+):(?P<col>\d+): (?P<code>[WE]\d{3}) (?P<msg>.+)$")

    def test_format_and_exit_status_zero_for_warnings(self):
        path = self.write("w.kg", "a::1\nmy_var::5\n[1;2]\n")
        rc, out, err = self.run_cli(path)
        self.assertEqual(rc, 0, err)
        lines = out.splitlines()
        self.assertEqual(len(lines), 3)
        m = [self.LINE.match(x) for x in lines[:2]]
        self.assertTrue(all(m), lines)
        self.assertEqual([(x["file"], x["line"], x["col"], x["code"]) for x in m],
                         [(path, "2", "1", "W001"), (path, "3", "3", "W017")])
        self.assertEqual(lines[2], "0 errors, 2 warnings in 1 file")
        self.assertEqual(err, "")

    def test_exit_status_one_for_an_error(self):
        path = self.write("e.kg", "f::{x}\nf(1;2)\n")
        rc, out, err = self.run_cli(path)
        self.assertEqual(rc, 1, err)
        self.assertIn(f"{path}:2:1: E002 ", out)
        self.assertTrue(out.endswith("1 error, 0 warnings in 1 file\n"), out)

    def test_clean_file(self):
        path = self.write("ok.kg", "f::{x+1}\nf(2)\n")
        rc, out, err = self.run_cli(path)
        self.assertEqual((rc, out, err), (0, "0 errors, 0 warnings in 1 file\n", ""))

    def test_several_files_in_one_run(self):
        a = self.write("a.kg", "my_var::5\n")
        b = self.write("b.kg", "1+\n")
        c = self.write("c.kg", "f::{x::1}\n")
        rc, out, err = self.run_cli(a, b, c)
        self.assertEqual(rc, 1, err)
        lines = out.splitlines()
        self.assertTrue(lines[0].startswith(f"{a}:1:1: W001 "), lines)
        self.assertTrue(any(x.startswith(f"{b}:1:2: E001 ") for x in lines), lines)
        self.assertEqual(lines[-1], "1 error, 2 warnings in 3 files")

    def test_a_file_with_a_syntax_error(self):
        path = self.write("bad.kg", "a::1\nf::{x\nmy_var::5\n")
        rc, out, err = self.run_cli(path)
        self.assertEqual(rc, 1, err)
        self.assertIn("E001 ", out)
        # the token rules still ran on the broken file
        self.assertIn("W001 ", out)
        self.assertNotIn("Traceback", err)

    def test_select_and_ignore(self):
        path = self.write("s.kg", "my_var::5\n[1;2]\n-1+2\n")
        rc, out, _ = self.run_cli(path, "--select", "W001")
        self.assertEqual(out.count(": W"), 1)
        self.assertIn("W001", out)
        rc, out, _ = self.run_cli(path, "--ignore", "W001,W002")
        self.assertNotIn("W001", out)
        self.assertIn("W017", out)
        self.assertIn("W010", out)
        rc, out, _ = self.run_cli(path, "--select=W010")
        self.assertEqual(out.count(": W"), 1)

    def test_ignoring_the_errors_changes_the_exit_status(self):
        path = self.write("e.kg", "1+\n")
        self.assertEqual(self.run_cli(path)[0], 1)
        self.assertEqual(self.run_cli(path, "--ignore", "E001")[0], 0)

    def test_an_unknown_code_is_a_usage_error(self):
        path = self.write("a.kg", "1\n")
        rc, out, err = self.run_cli(path, "--select", "W999")
        self.assertEqual(rc, 2)
        self.assertIn("W999", err)

    def test_an_unreadable_file(self):
        good = self.write("a.kg", "my_var::5\n")
        rc, out, err = self.run_cli(os.path.join(self.dir.name, "missing.kg"), good)
        self.assertEqual(rc, 2)
        self.assertIn("missing.kg: cannot read", err)
        self.assertIn("W001", out)      # the other file was still checked
        self.assertTrue(out.endswith("0 errors, 1 warning in 1 file\n"), out)

    def test_hints_with_verbose(self):
        path = self.write("a.kg", "my_var::5\n")
        _, out, _ = self.run_cli(path, "-v")
        self.assertIn("\n  hint: ", out)
        _, out, _ = self.run_cli(path)
        self.assertNotIn("hint", out)

    def test_nothing_is_run_and_no_repl_starts(self):
        path = self.write("x.kg", ".x(3)\n")
        rc, out, err = self.run_cli(path)
        self.assertEqual((rc, out, err), (0, "0 errors, 0 warnings in 1 file\n", ""))
        # a file that would loop for ever, and one that loads it
        self.write("loop.kg", "{1}{1}:~0\n")
        path = self.write("main.kg", '.l("loop.kg")\n')
        rc, out, err = self.run_cli(path)
        self.assertEqual(rc, 0, err)

    def test_define(self):
        path = self.write("t.kg", 't("x";1;1)\n')
        _, out, _ = self.run_cli(path)
        self.assertIn("W002", out)
        _, out, _ = self.run_cli(path, "--define", "t")
        self.assertNotIn("W002", out)

    def test_stdin(self):
        rc, out, err = self.run_cli("-", stdin="my_var::5\n")
        self.assertEqual(rc, 0, err)
        self.assertIn("<stdin>:1:1: W001 ", out)

    def test_in_process_entry_point(self):
        import io
        path = self.write("a.kg", "my_var::5\n")
        out, err = io.StringIO(), io.StringIO()
        self.assertEqual(lint_main([path], stdout=out, stderr=err), 0)
        self.assertIn("W001", out.getvalue())


class TestRobustness(unittest.TestCase):
    def test_no_optional_packages(self):
        code = ("import sys\n"
                "for m in ('torch', 'pandas', 'jax'):\n"
                "    sys.modules[m] = None\n"
                "from klongpy.lint import lint_source\n"
                "assert [d.code for d in lint_source('my_var::5', 't.kg')] == ['W001']\n"
                "assert 'torch' not in [k for k, v in sys.modules.items() if v is not None]\n")
        p = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True,
                           timeout=120, cwd=ROOT)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_strange_input_does_not_crash(self):
        texts = ["", "\n\n", ":", ":\"", "[", "]", "{", "}", ")", "(", ":[", ":{", ":(", ":|", "0c", "[;", "{[", "{[a",
                 "{[a];", "{[a b]::", "f(a;b)::", "[a b]::", "\"", ":\"abc", "é::1", "a::²", "1e", "1.2.3",
                 ".comment(\"X\")", ".l(", ".l(\"", ".bkf(", ".bkf([\"a\"", "x::{", ":[1;2;:[", ",:{", "[1 2],:{",
                 "a\x00b", "\t\r\n", "}{", "][", "){", "f(\n", "{x}(1;2;3;4;5;6)", "∇", "[a b]∇f"]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for t in texts:
                with self.subTest(text=t):
                    for d in lint_source(t, "t.kg", resolve=False):
                        self.assertIn(d.code, RULES)

    def test_mutated_library_sources(self):
        files = sorted(glob.glob(os.path.join(ROOT, "klongpy", "lib", "*.kg")))
        texts = [open(f).read() for f in files]
        rnd = random.Random(52)
        chars = list("[](){};:\"'/\\ \n.,_-+*%@#{[|0123456789abxyz") + [":[", ":{", ":(", ":|", "0c", "::", ":\"", "\\~"]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for k in range(250):
                t = rnd.choice(texts)
                s = rnd.randrange(0, max(1, len(t) - 2000))
                t = list(t[s:s + rnd.randrange(50, 2000)])
                for _ in range(rnd.randrange(1, 6)):
                    i = rnd.randrange(0, len(t) + 1)
                    if rnd.random() < 0.4 and t:
                        del t[min(i, len(t) - 1)]
                    else:
                        t.insert(i, rnd.choice(chars))
                lint_source("".join(t), "fuzz.kg", resolve=False)

    def test_unknown_token_kinds_are_neutral(self):
        # a token kind the linter has never heard of must not break the frame scan
        from klongpy import lint
        toks, trivia = lint._tokenize(Source("t", "a::1"))
        odd = toks[:1] + [toks[0]._replace(kind="NEWKIND")] + toks[1:]
        frames, ctx, _ = lint._scan(odd)
        self.assertEqual(len(ctx), len(odd))
        self.assertEqual([f.kind for f in frames], ["top"])


def _corpus():
    out = []
    for pat in ("klongpy/lib/*.kg", "tests/kgtests/**/*.kg", "examples/**/*.kg"):
        out.extend(sorted(glob.glob(os.path.join(ROOT, pat), recursive=True)))
    return out


class TestShippedSources(unittest.TestCase):
    """
    The linter over the .kg files of the repository. The findings that remain
    were each read and judged true; rules
    that would indicate a false positive or a real hazard must not appear.
    """
    # codes that fire on shipped code, each reviewed: W003 unused locals, W012/W018/W020 in the test files that
    # test exactly that, E002 extra arguments in the stock-feed examples
    REVIEWED = {"W003", "W012", "W018", "W020", "E002"}

    def test_corpus(self):
        files = _corpus()
        self.assertGreater(len(files), 100)
        bad = []
        for f in files:
            for d in lint_file(f, define="t"):
                if d.code not in self.REVIEWED:
                    bad.append(f"{os.path.relpath(f, ROOT)}:{d.line}:{d.col}: {d.code} {d.message}")
        self.assertEqual(bad, [])

    def test_library_files_have_no_errors(self):
        for f in sorted(glob.glob(os.path.join(ROOT, "klongpy", "lib", "*.kg"))):
            with self.subTest(file=os.path.basename(f)):
                self.assertEqual([d for d in lint_file(f) if d.severity == "error"], [])

    def test_the_locals_found_on_tokens_are_the_ones_the_parser_found(self):
        from klongpy import lint
        for f in _corpus():
            an = lint._Analysis(Source(f, open(f).read()), os.path.dirname(f), False)
            for fr in an.frames:
                if fr.kind == "fn" and fr.meta is not None:
                    want = tuple(str(x).split("`")[0] for x in fr.meta.locals)
                    self.assertEqual(want, tuple(fr.locals), f"{f}:{an.source.line_col(an.toks[fr.open][1])}")

    def test_speed(self):
        path = os.path.join(ROOT, "tests", "kgtests", "language", "test_suite.kg")
        lint_source("1", "warm.kg")
        t = time.perf_counter()
        lint_file(path, define="t")
        self.assertLess(time.perf_counter() - t, 1.0)


if __name__ == "__main__":
    unittest.main()
