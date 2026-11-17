"""
The static profile checker (docs/static-profile.md).

Table-driven. For every rule id in klongpy.static.rules there is one program
inside the profile (tests/static/accept/<id>.kg: the nearest legal form) and
one outside it (tests/static/reject/<id>.kg, whose first line names the line
and the rule expected). CASES adds further programs per rule, with the
(line, rule) of every diagnostic.

The checker never evaluates a program: TestNeverEvaluates runs it with the
interpreter's evaluator disabled.
"""
import os
import re
import unittest

from klongpy import KlongInterpreter
from klongpy.errors import KlongError, KlongStaticError
from klongpy.static import RULES, check_file, check_source

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECT = re.compile(r':" expect: static line (\d+) (\S+) "')


def found(text, **kw):
    """[(line, rule)] of every diagnostic, in order."""
    return [(d.line, d.rule) for d in check_source(text, "t.kg", **kw).diagnostics]


def entry(sig, name="f"):
    return f'\n.entry("{name}";"{sig}")'


# (rule, program, [(line, rule), ...]); [] means the program is inside the profile.
CASES = [
    # -- program structure ---------------------------------------------------
    ("toplevel-effect", "f::{x}\nf(1)", [(2, "toplevel-effect")]),
    ("toplevel-effect", "f::{x}\n42", [(2, "toplevel-effect")]),
    ("toplevel-effect", "f::{x}\nt::{x*2}'!8", []),                 # an adverb applied at top level is a value
    ("toplevel-effect", "x::5", [(1, "toplevel-effect")]),
    ("toplevel-effect", ".module(:m)\nf::{x}", [(1, "toplevel-effect")]),
    ("decl-literal", '.extern("a.b";"int -> int";3)', [(1, "decl-literal")]),
    ("decl-literal", '.extern("a.b")', [(1, "decl-literal")]),
    ("decl-literal", 'n::"a"\n.l(n)', [(1, "global-value"), (2, "decl-literal")]),
    ("signature", 'f::{x}\n.entry("f";"int -> flot")', [(2, "signature")]),
    ("signature", 'f::{x}\n.entry("f";"str -> int")', [(2, "signature")]),
    ("signature", '.extern("a.b";"int -> str")', [(1, "signature")]),
    ("signature", '.extern("a.b";"int[0] -> int")', [(1, "signature")]),
    ("signature", '.extern("a.b";"int[1,2,3,4,5] -> int")', [(1, "signature")]),
    ("signature", '.extern("a.b";"str int float[2,3] -> void";"dsp")', []),
    ("extern", '.extern("while";"int -> int")', [(1, "extern")]),
    ("extern", '.extern("a_b";"int -> int")', [(1, "extern")]),
    ("extern", '.extern("x";"int -> int")', [(1, "extern")]),
    ("extern", '.extern("a.b";"int -> int")\n.extern("a.b";"int -> float")', [(2, "extern")]),
    ("extern", '.extern("a.b";"int -> int")\n.extern("a.b";"int->int")', []),
    ("entry", '.entry("f";"int -> int")\nf::{x}', [(1, "entry")]),
    ("entry", 'init::{x}\n.entry("init";"int -> int")', [(2, "entry")]),
    ("entry", 'g::3\n.entry("g";"-> int")', [(2, "entry")]),
    ("entry", "f::{x%2}" + entry("int -> int"), [(2, "entry")]),                # returns float
    ("entry", "f::{x,x}" + entry("int -> int[3]"), [(2, "entry")]),             # returns int[2]
    ("entry", "f::{x+1}" + entry("int -> float"), []),                          # int converts to float
    ("entry", "f::{x+1}" + entry("int -> void"), []),
    ("entry", "f::{x}" + entry("int -> int") + entry("float[3] -> float[3]"), []),   # several entries
    ("entry", "f(a;b;c;d)::{a+b+c+d}" + entry("int int int int -> int"), []),
    ("redefinition", "f::{x}\nf::3", [(2, "redefinition")]),
    ("redefinition", 'f::{x}\n.extern("f";"int -> int")', [(2, "redefinition")]),
    ("redefinition", "a::1\na::a+1\nf::{x+a}", []),                             # a value may be computed in steps
    ("redefinition", "g::{x}\nf::{g::3;x}", [(2, "redefinition")]),
    # -- compile-time values -----------------------------------------------------
    ("global-value", "a:::{[1 2]}", [(1, "global-value")]),
    ("global-value", "a:::sym", [(1, "global-value")]),
    ("global-value", "a::0cx", [(1, "global-value")]),
    ("global-value", "g::{x}\na::g", [(2, "global-value")]),
    ("global-value", "g::{x+y}\na::g(1;)", [(2, "global-value")]),
    ("global-value", "a::[1 2 3]\nb::2.5\nc::-7\nd::[[1.5 2] [3 4]]", []),
    ("global-array", "a::[]", [(1, "global-array")]),
    ("global-array", "a::[1 [2 3]]", [(1, "global-array")]),
    ("global-array", 'a::[1 "s"]', [(1, "global-array")]),
    ("int-range", "a::[1 2 99999999999]", [(1, "int-range")]),
    ("int-range", "a::-2147483648", []),
    ("int-range", "a::-2147483649", [(1, "int-range")]),
    ("int-range", "f::{x+3000000000}", [(1, "int-range")]),
    # -- values ----------------------------------------------------------------------
    ("string", "f::{x;0cA}", [(1, "string")]),
    ("string", 'f::{x+["a" "b"]}', [(1, "string")]),
    ("string", "f::{x+[1;2]}", [(1, "string")]),                    # ; inside [...] is the string ";"
    ("symbol", "f::{x;:foo}", [(1, "symbol")]),
    ("dict", "f::{:{;1;x}}", [(1, "dict")]),
    ("list-literal", "f::{x+[1 + 2]}", [(1, "list-literal")]),
    ("list-literal", "f::{[;x;[1 2]]}" + entry("int -> int[2]"), [(1, "list-literal")]),
    ("list-literal", "f::{[;x;1.5]}" + entry("int -> float[2]"), []),
    ("fn-value", "g::{x}\nf::{g}", [(2, "fn-value")]),
    ("fn-value", "f::{{x+1}}", [(1, "fn-value")]),
    ("fn-value", "g::{x}\nh::{x}\nf::{h(g)}", [(3, "fn-value")]),
    ("fn-value", "g::{x}\nf::{g@x}", [(2, "fn-value")]),
    ("fn-value", "f::{{x+1}(x)}", [(1, "fn-value")]),
    ("fn-value", "f::{x(1)}", [(1, "fn-value")]),
    ("fn-value", "g::{x+1}\nf::{g'x}" + entry("int[3] -> int[3]"), []),         # a function name as a verb
    ("arity", "g::{x+1}\nf::{x g'y}", [(2, "arity")]),
    ("projection", "g::{x+y}\nf::{g(x)}", [(2, "projection")]),
    ("void-value", '.extern("v";"-> void")\nf::{[a];a::v();x}', [(2, "void-value")]),
    ("void-value", '.extern("v";"-> void")\nf::{x;v()}' + entry("int -> void"), []),
    ("entry", '.extern("v";"-> void")\nf::{x;v()}' + entry("int -> int"), [(3, "entry")]),
    # -- result length -------------------------------------------------------------------
    ("grade", "f::{>x}", [(1, "grade")]),
    ("const-count", "f::{y_x}", [(1, "const-count")]),
    ("const-count", "f::{y:^x}", [(1, "const-count")]),
    ("const-count", "f::{!x}", [(1, "const-count")]),
    ("const-count", "f::{(#x)#y}", []),                             # #a is a constant
    ("const-count", "f::{((#x)-1)_y}", []),
    ("const-count", "n::4\nf::{n::n+1;n#x}", [(2, "const-count")]),             # an assigned global is not constant
    ("const-count", "g::{x#y}\nf::{g(2;x)}", [(1, "const-count")]),             # nor is a constant argument
    ("scan-dynamic", "f::{3{x*2}\\*x}", [(1, "scan-dynamic")]),
    ("scan-dynamic", "f::{{x%2}\\~x}", [(1, "scan-dynamic")]),
    # -- forms -------------------------------------------------------------------------------
    ("take-rank", "f::{3#x}" + entry("int -> int[3]"), [(1, "take-rank")]),
    ("amend-form", "f::{x:=y,0}" + entry("int[2,2] int -> int[2,2]"), [(1, "amend-form")]),
    ("amend-form", "f::{x:=y,0}" + entry("int[4] int[2] -> int[4]"), [(1, "amend-form")]),
    ("amend-form", "f::{x:=y}", [(1, "amend-form")]),
    ("amend-form", "buf::[0.0 0.0]\nf::{buf::buf:=x,[0 1];x}" + entry("float -> float"), []),
    ("size-of-atom", "f::{^x}" + entry("float -> int"), [(1, "size-of-atom")]),
    ("unsupported-op", "f::{+/'x}", [(1, "unsupported-op")]),       # an adverb chain
    ("unsupported-op", "f::{x:@y}", [(1, "unsupported-op")]),
    ("unsupported-op", "f::{@x}", [(1, "unsupported-op")]),
    ("unsupported-op", "f::{x+:/y}", [(1, "unsupported-op")]),
    ("recursion", "f::{.f(x-1)}", [(1, "recursion")]),
    ("recursion", "g::{h(x)}\nh::{g(x)}\nf::{g(x)}" + entry("int -> int"), [(1, "recursion"), (2, "recursion")]),
    ("recursion", "g::{g(x)}\nf::{x+1}" + entry("int -> int"), []),             # no entry reaches g: host only
    ("sys-fn", "f::{.E(x)}", [(1, "sys-fn")]),
    ("sys-fn", "f::{x+.e}", [(1, "sys-fn")]),
    ("sys-fn", 'f::{.l("math");x}', [(1, "sys-fn")]),
    ("sys-fn", "f::{.pyc(x;1;2)}", [(1, "sys-fn")]),
    ("undefined-name", "f::{g(x)}", [(1, "undefined-name")]),
    ("undefined-name", "f::{k::x;k}", [(1, "undefined-name"), (1, "undefined-name")]),
    ("undefined-name", "g::{x;a+1}\nf::{[a];a::x;g(0)}", [(1, "undefined-name")]),   # no dynamic scope
    ("unset-local", "f::{[a];:[x;a::1;0];a}", [(1, "unset-local")]),
    ("unset-local", "f::{[a];:[x;a::1;a::2];a}", []),
    ("closure-assign", "f::{[k];k::2;{x*k}'x}" + entry("int[3] -> int[3]"), []),  # reading is allowed
    ("arity", "f::{{x+y}'x}", [(1, "arity")]),
    ("arity", "f::{{x}/x}", [(1, "arity")]),
    # -- types and shapes ---------------------------------------------------------------------
    ("operand", "f::{x@y}" + entry("int[3] float -> int"), [(1, "operand")]),
    ("operand", "f::{x@0}" + entry("int -> int"), [(1, "operand")]),
    ("operand", "f::{+x}" + entry("int[3] -> int[3]"), [(1, "operand")]),
    ("operand", "f::{1:+x}" + entry("int[2,2] -> int[2,2]"), [(1, "operand")]),
    ("operand", "f::{{x+1}'x}" + entry("int -> int"), [(1, "operand")]),
    ("operand", "f::{x{x+1}:*y}" + entry("float int -> int"), [(1, "operand")]),
    ("shape-mismatch", "f::{x,y}" + entry("int[2,3] int[3] -> int[3,3]"), [(1, "shape-mismatch")]),
    ("shape-mismatch", "f::{x,,y}" + entry("int[2,3] int[3] -> int[3,3]"), []),
    ("shape-mismatch", "f::{x~y}" + entry("int[2] int[3] -> int"), [(1, "shape-mismatch")]),
    ("shape-mismatch", "f::{x{x+y}'y}" + entry("int[2] int[3] -> int[2]"), [(1, "shape-mismatch")]),
    ("shape-mismatch", "f::{[a b]::x;a+b}" + entry("int[3] -> int"), [(1, "shape-mismatch")]),
    ("shape-mismatch", "f::{x+y}" + entry("int[2,3] int[2] -> int[2,3]"), []),  # a prefix
    ("shape-mismatch", "f::{x+y}" + entry("int[2,3] int[3] -> int[2,3]"), [(1, "shape-mismatch")]),
    ("shape-change", "buf::[0 0 0]\nf::{buf::buf:=x,0;x}" + entry("float -> float"), [(2, "shape-change")]),
    ("shape-change", "c::0.0\nf::{c::1;x}" + entry("int -> int"), [(2, "shape-change")]),
    ("shape-change", "c::0\nf::{c::x%2;x}" + entry("int -> int"), [(2, "shape-change")]),
    ("shape-change", "c::[1 2]\nf::{c::x;x}" + entry("int[3] -> int[3]"), [(2, "shape-change")]),
    ("shape-change", "f::{[a];a::0;a::x%2;a}" + entry("int -> float"), []),     # a local joins int with float
    ("cond-shape", "f::{:[x;1;[1 2]]}", [(1, "cond-shape")]),
    ("cond-shape", "f::{:[x;1;2.5]}" + entry("int -> float"), []),
    ("cond-scalar", "f::{{x}{x+1}:~x}" + entry("int[2] -> int[2]"), [(1, "cond-scalar")]),
    ("div-zero", "f::{x:%0}" + entry("int -> int"), [(1, "div-zero")]),
    ("div-zero", "f::{1%0}", [(1, "div-zero")]),
    ("div-zero", "f::{x%0}" + entry("float -> float"), []),         # IEEE at run time
    ("index-range", "f::{x@[0 5]}" + entry("int[4] -> int[2]"), [(1, "index-range")]),
    ("index-range", "f::{x@-4}" + entry("int[4] -> int"), []),      # counts from the end
    ("index-range", "f::{x@-5}" + entry("int[4] -> int"), [(1, "index-range")]),
    ("index-range", "f::{x:=0,4}" + entry("int[4] -> int[4]"), [(1, "index-range")]),
    ("index-range", "f::{x@#x}" + entry("int[4] -> int"), [(1, "index-range")]),
    ("array-shape", "f::{x+[]}", [(1, "array-shape")]),
    ("array-shape", "f::{4_x}" + entry("int[4] -> int"), [(1, "array-shape")]),
    ("array-shape", "f::{!0}", [(1, "array-shape")]),
    ("array-shape", "f::{,,x}" + entry("int[2,2,2] -> int"), [(1, "array-shape")]),
    ("array-shape", "f::{-:'x}" + entry("int[1] -> int"), [(1, "array-shape")]),
    ("loop-state", "f::{,/x}" + entry("int[3,2] -> int[6]"), [(1, "loop-state")]),
    ("loop-state", "f::{{x<9}{x+0.5}:~0}", [(1, "loop-state")]),
    ("loop-state", "f::{{x+y%2}/x}" + entry("int[4] -> float"), []),            # one promotion is allowed
    ("extern-call", '.extern("r";"int -> int")\nf::{r(x;1)}', [(2, "extern-call")]),
    ("extern-call", '.extern("r";"float[3] -> int")\nf::{r(x)}' + entry("float[4] -> int"), [(2, "extern-call")]),
    ("extern-call", '.extern("r";"float -> int")\nf::{r(x)}' + entry("int -> int"), []),   # an int converts to float
    ("extern-call", '.extern("r";"str -> int")\nf::{[s];r(x)}', [(2, "extern-call")]),
    ("extern-call", '.extern("r";"int -> float")\nf::{r\'x}' + entry("int[3] -> float[3]"), []),
]


class TestRuleTable(unittest.TestCase):
    """One accepted and one rejected program for every rule id."""

    def test_every_rule_is_accepted_in_its_nearest_legal_form(self):
        for rule in RULES:
            with self.subTest(rule=rule):
                r = check_file(os.path.join(HERE, "accept", rule + ".kg"))
                self.assertEqual([str(d) for d in r.diagnostics], [])
                self.assertTrue(r.ok)
                self.assertTrue(r.entries, "an accept fixture declares an entry, so its types are known")

    def test_every_rule_rejects_its_fixture_at_the_expected_line(self):
        for rule in RULES:
            with self.subTest(rule=rule):
                path = os.path.join(HERE, "reject", rule + ".kg")
                with open(path) as f:
                    m = EXPECT.match(f.readline())
                self.assertIsNotNone(m, "the first line names the expected error")
                self.assertEqual(m.group(2), rule)
                r = check_file(path)
                self.assertFalse(r.ok)
                got = [(d.line, d.rule) for d in r.diagnostics]
                self.assertEqual(got[0], (int(m.group(1)), rule))
                self.assertEqual({g[1] for g in got}, {rule}, "a reject fixture breaks one rule only")

    def test_cases(self):
        for rule, text, want in CASES:
            with self.subTest(rule=rule, text=text):
                self.assertEqual(found(text), want)
                if want:
                    self.assertIn(rule, [w[1] for w in want])

    def test_every_rule_has_a_case_or_fixture_and_every_case_names_a_rule(self):
        for rule, _, want in CASES:
            self.assertIn(rule, RULES)
            for _, got in want:
                self.assertIn(got, RULES)


class TestDiagnostics(unittest.TestCase):
    def test_a_diagnostic_is_a_positioned_static_error_that_names_its_rule(self):
        for rule in RULES:
            with self.subTest(rule=rule):
                r = check_file(os.path.join(HERE, "reject", rule + ".kg"))
                d = r.diagnostics[0]
                e = d.error()
                self.assertIsInstance(e, KlongStaticError)
                self.assertIsInstance(e, KlongError)
                self.assertEqual(e.kind, "static")
                self.assertEqual(e.rule, rule)
                self.assertTrue(str(e).endswith(f"[{rule}]"))
                self.assertFalse(e.message.endswith("."))
                self.assertNotIn("\n", e.message)
                start, end = e.span
                self.assertTrue(0 <= start < end <= len(e.source.text), "the span lies in the source and is not empty")
                lines = e.render().split("\n")
                self.assertRegex(lines[0], rf"reject/{re.escape(rule)}\.kg:\d+:\d+: static error: .* \[{re.escape(rule)}\]$")
                self.assertIn("^", lines[2])
                if e.hint:
                    self.assertTrue(lines[-1].startswith("  hint: "))

    def test_the_span_is_the_offending_expression(self):
        cases = [
            ("f::{x+&y}", "&y"),
            ("f::{[a];a::x?1;a}", "x?1"),
            ("g::{x}\nf::{1+g}", "g"),
            ('f::{x;"abc"}', '"abc"'),
            ("f::{x+[1 [2]]}", "[1 [2]]"),
            ('.extern("r";"int -> flot")', "flot"),
            ('f::{x}\n.entry("f";"int int -> int")', '"int int -> int"'),
            ("f::{:[x;1;[1 2]]}", ":[x;1;[1 2]]"),
            ("a::[1 [2]]", "[1 [2]]"),
        ]
        for text, want in cases:
            with self.subTest(text=text):
                d = check_source(text, "t.kg").diagnostics[0]
                self.assertEqual(text[d.span[0]:d.span[1]], want)

    def test_all_problems_are_reported_not_only_the_first(self):
        text = "\n".join([
            'name::"x"',                    # 1 global-value
            "f::{&x}",                      # 2 where
            "g::{x+}",                      # 3 strict-syntax
            "h::{.p(x);x?1}",               # 4 sys-fn, find
            ".p(1)",                        # 5 toplevel-effect
            'k::{:[x;[1 2];[1 2 3]]}',      # 6 cond-shape
        ])
        self.assertEqual(found(text), [(1, "global-value"), (2, "where"), (3, "strict-syntax"), (4, "sys-fn"), (4, "find"),
                                       (5, "toplevel-effect"), (6, "cond-shape")])

    def test_one_diagnostic_per_place_however_often_a_function_is_specialised(self):
        text = 'g::{&x}\nf::{g(x)+g(1.5)+g([1 2])}' + entry("int -> int")
        self.assertEqual(found(text), [(1, "where")])

    def test_a_name_whose_definition_did_not_parse_is_not_also_undefined(self):
        self.assertEqual(found("g::{x+}\nf::{g(x)}" + entry("int -> int")), [(1, "strict-syntax")])

    def test_raise_first(self):
        r = check_source("f::{&x;?x}", "t.kg")
        with self.assertRaises(KlongStaticError) as cm:
            r.raise_first()
        self.assertEqual(cm.exception.rule, "where")
        self.assertEqual(len(cm.exception.diagnostics), 2)
        self.assertIs(check_source("f::{x}", "t.kg").raise_first().ok, True)

    def test_diagnostics_name_the_function_they_lie_in(self):
        r = check_source("g::{&x}\nf::{g(x)}\nh::{x+1}\n.p(1)", "t.kg")
        self.assertEqual([(d.rule, d.function) for d in r.diagnostics], [("where", "g"), ("toplevel-effect", None)])
        self.assertEqual({n: f.in_profile for n, f in r.functions.items()}, {"g": False, "f": False, "h": True})


class TestStages(unittest.TestCase):
    """what each kind of top-level statement is."""

    def test_statements_are_sorted_by_the_root_of_their_tree(self):
        text = "\n".join([
            "tab::{x*x}'!8",                        # a value: the root is an adverb, not a function literal
            "f::{tab@x}",
            "g(a;b)::{a+b}",
            '.extern("read.adc";"int -> float";"io")',
            '.entry("f";"int -> int")',
            "[p q]::[1 2]",
        ])
        r = check_source(text, "t.kg")
        self.assertTrue(r.ok, r.render())
        self.assertEqual([(k, n) for k, n, *_ in r.statements],
                         [("global", "tab"), ("function", "f"), ("function", "g"), ("extern", "read.adc"),
                          ("entry", "f"), ("global", "p"), ("global", "q")])
        self.assertEqual(r.externs["read.adc"].module, "io")
        self.assertEqual(str(r.externs["read.adc"].signature), "int -> float")
        self.assertEqual(r.functions["g"].params, ("a", "b"))
        self.assertEqual(r.functions["f"].params, ("x",))
        self.assertEqual(str(r.globals["p"].ty), "int")
        self.assertEqual(str(r.globals["q"].ty), "int")
        self.assertIsNone(r.globals["tab"].ty)
        self.assertEqual(r.unresolved, ["tab"])

    def test_a_name_assigned_inside_a_definition_is_a_global_too(self):
        r = check_source("a::b::3\nf::{x+a+b}" + entry("int -> int"), "t.kg")
        self.assertTrue(r.ok, r.render())
        self.assertEqual(sorted(r.globals), ["a", "b"])
        self.assertEqual(r.unresolved, ["a", "b"])

    def test_only_functions_an_entry_reaches_are_checked(self):
        text = 'show::{.p(x)}\nhelper::{x*2}\nf::{helper(x)}' + entry("int -> int")
        r = check_source(text, "t.kg")
        self.assertTrue(r.ok, r.render())
        self.assertEqual(r.host_only, ["show"])
        self.assertEqual([d.rule for d in check_source(text, "t.kg", all_functions=True).diagnostics], ["sys-fn"])

    def test_a_program_without_entries_is_checked_as_a_library(self):
        r = check_source("a::{x+1}\nb::{&x}", "t.kg")
        self.assertEqual([(d.rule, d.function) for d in r.diagnostics], [("where", "b")])
        self.assertEqual(r.host_only, [])

    def test_state_and_constants(self):
        r = check_source("gain::3\ncount::0\nf::{count::count+1;x*gain}" + entry("int -> int"), "t.kg")
        self.assertTrue(r.ok, r.render())
        self.assertEqual({n: g.assigned for n, g in r.globals.items()}, {"gain": False, "count": True})

    def test_functions_are_specialised_per_argument_types(self):
        r = check_source("sq::{x*x}\nf::{sq(x)+sq(1.5)+*sq([1 2])}" + entry("int -> float"), "t.kg")
        self.assertTrue(r.ok, r.render())
        got = {tuple(str(a) for a in k): str(v) for k, v in r.functions["sq"].specializations.items()}
        self.assertEqual(got, {("int",): "int", ("float",): "float", ("int[2]",): "int[2]"})

    def test_values_handed_in_by_the_driver_decide_what_literals_cannot(self):
        import numpy as np
        text = "taps::mk(16)\nrun::{+/taps*x}" + entry("float[16] -> float", "run")
        r = check_source(text, "t.kg")
        self.assertTrue(r.ok)
        self.assertEqual(r.unresolved, ["taps"])
        r = check_source(text, "t.kg", globals={"taps": np.ones(16)})
        self.assertTrue(r.ok, r.render())
        self.assertEqual(str(r.globals["taps"].ty), "float[16]")
        self.assertEqual([str(v) for v in r.functions["run"].specializations.values()], ["float"])
        self.assertEqual(found(text, globals={"taps": np.ones(8)}), [(2, "shape-mismatch")])
        self.assertEqual(found(text, globals={"taps": "abc"}), [(1, "global-value")])
        self.assertEqual(found(text, globals={"taps": np.array([1, 2 ** 40])}), [(1, "int-range")])
        self.assertEqual(found(text, globals={"taps": np.array([1, 2 ** 40])}, int_bits=64), [(2, "shape-mismatch")])
        self.assertEqual(found(text, globals={"taps": np.array([[1, 2], 3], dtype=object)}), [(1, "global-array")])

    def test_int64(self):
        self.assertEqual(found("a::3000000000"), [(1, "int-range")])
        self.assertEqual(found("a::3000000000", int_bits=64), [])

    def test_load_follows_the_file_under_the_same_rules(self):
        r = check_file(os.path.join(HERE, "accept", "load.kg"))
        self.assertTrue(r.ok, r.render())
        self.assertIn("scale", r.functions)
        self.assertTrue(r.functions["scale"].source.name.endswith("scale.kg"))
        self.assertEqual(found('.l("support/scale")\nf::{scale(x)}', resolve=False), [(2, "undefined-name")])


class TestNeverEvaluates(unittest.TestCase):
    def test_the_checker_runs_with_the_evaluator_disabled(self):
        def boom(*a, **k):
            raise AssertionError("the checker evaluated something")
        saved = {n: getattr(KlongInterpreter, n) for n in ("__call__", "eval", "call", "exec", "apply")}
        for n in saved:
            setattr(KlongInterpreter, n, boom)
        try:
            for rule in RULES:
                check_file(os.path.join(HERE, "accept", rule + ".kg"))
                check_file(os.path.join(HERE, "reject", rule + ".kg"))
            for rule, text, want in CASES:
                self.assertEqual(found(text), want)
        finally:
            for n, f in saved.items():
                setattr(KlongInterpreter, n, f)

    def test_nothing_in_the_program_happens(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            mark = os.path.join(d, "mark")
            prog = f'f::{{x}}\nt::.sys("touch {mark}")\n.sys("touch {mark}")\nu::.E("1+1")\n.py("os")'
            r = check_source(prog, os.path.join(d, "p.kg"))
            self.assertEqual([d_.rule for d_ in r.diagnostics], ["toplevel-effect", "toplevel-effect"])
            self.assertFalse(os.path.exists(mark))


if __name__ == "__main__":
    unittest.main()
