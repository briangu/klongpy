"""
Dictionary ergonomics (design:
docs/language-additions.md): the evaluated literal
:{;k1;v1;k2;v2}, nested literals as values, merge with `,`, lookup with `@`,
and `$` of a dictionary. Every rule is run at strict levels 0, 1 and 2. The
evaluated literal and nested literals are on the token parser only.
"""
import unittest
import warnings

from klongpy import KlongInterpreter
from klongpy.core import KGSym, KLONG_UNDEFINED
from tests.treedump import dump_text
from klongpy.errors import KlongError, KlongNameError, KlongSyntaxError, KlongSyntaxWarning, KlongTypeError

LEVELS = (0, 1, 2)


def interp(level, *lines):
    k = KlongInterpreter(strict=level)
    for line in lines:
        k(line)
    return k


def run(level, *lines):
    k = KlongInterpreter(strict=level)
    r = None
    for line in lines:
        r = k(line)
    return r


def error(level, *lines):
    try:
        run(level, *lines)
    except KlongError as e:
        return e
    raise AssertionError(f"no error at level {level}: {lines}")


def num(v):
    return v.item() if hasattr(v, "item") else v


def nums(v):
    return [num(q) for q in v]


def plain(d):
    """A dictionary with numbers as Python numbers (a torch value is a 0-d tensor)."""
    return {k: (plain(v) if isinstance(v, dict) else num(v) if getattr(v, "ndim", 0) == 0 else nums(v)) for k, v in d.items()}


class TestEvaluatedLiteral(unittest.TestCase):
    def test_keys_and_values_are_expressions(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(num(run(level, ':{;"a";1+1;"b";2*3}?"b"')), 6)
                self.assertEqual(num(run(level, 'a::5;d:::{;"k";a};d?"k"')), 5)
                self.assertEqual(plain(run(level, 'k::"key";v::[1 2 3];:{;k;v;k,"2";#v}')), {"key": [1, 2, 3], "key2": 3})
                self.assertEqual(plain(run(level, ":{;1;2;3;4}")), {1: 2, 3: 4})
                self.assertEqual(plain(run(level, ":{;:a;1;:b;2}")), {KGSym("a"): 1, KGSym("b"): 2})

    def test_empty(self):
        for level in LEVELS:
            self.assertEqual(run(level, ":{;}"), {})
            self.assertEqual(run(level, ":{;\n}"), {})

    def test_a_fresh_dictionary_each_time(self):
        for level in LEVELS:
            k = interp(level, "mk::{:{;:n;x}}", "p::mk(1)", "q::mk(2)", "p,[:m 9]")
            self.assertEqual(plain(k("p")), {KGSym("n"): 1, KGSym("m"): 9})
            self.assertEqual(plain(k("q")), {KGSym("n"): 2})

    def test_evaluated_left_to_right_each_time(self):
        for level in LEVELS:
            k = interp(level, "n::0", "t::{n::n+1;n}", "f::{:{;t();t();t();t()}}")
            self.assertEqual(plain(k("f()")), {1: 2, 3: 4})
            self.assertEqual(plain(k("f()")), {5: 6, 7: 8})

    def test_a_later_key_replaces_an_earlier_one(self):
        self.assertEqual(plain(run(0, ":{;1;2;1;3}")), {1: 3})

    def test_nested(self):
        for level in LEVELS:
            self.assertEqual(plain(run(level, 'a::1;:{;"in";:{;"a";a};"l";[;a;a+1]}')), {"in": {"a": 1}, "l": [1, 2]})
            self.assertEqual(num(run(level, 'd:::{;"GET";:{;"/";2}};(d?"GET")?"/"')), 2)

    def test_newlines(self):
        for level in LEVELS:
            self.assertEqual(plain(run(level, ':{;"a";1\n  ;"b";2\n}')), {"a": 1, "b": 2})
            self.assertEqual(plain(run(level, ':{;\n "a";1;\n "b";2}')), {"a": 1, "b": 2})

    def test_odd_count_is_a_syntax_error(self):
        for level in LEVELS:
            for text in (":{;1;2;3}", ":{;1}", ':{;"a";1;"b"}'):
                with self.subTest(level=level, text=text):
                    e = error(level, text)
                    self.assertIsInstance(e, KlongSyntaxError)
                    self.assertIn("value", str(e))
                    self.assertEqual(e.span[0], 0)
                    self.assertIn(":{;key1;value1", e.hint)

    def test_empty_slot_is_a_syntax_error(self):
        for level in LEVELS:
            for text in (":{;1;;2}", ":{;;1;2}", ":{;1;2;}", ":{;;}"):
                with self.subTest(level=level, text=text):
                    self.assertIsInstance(error(level, text), KlongSyntaxError)

    def test_unterminated(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.assertEqual(plain(run(0, ":{;1;2")), {1: 2})
        self.assertTrue(any("unterminated dictionary" in str(q.message) for q in w))
        for level in (1, 2):
            self.assertIsInstance(error(level, ":{;1;2"), KlongSyntaxError)

    def test_a_space_makes_it_the_plain_literal(self):
        # only ":{;" starts the evaluated form, as only "[;" starts an evaluated list
        for level in LEVELS:
            self.assertIsInstance(error(level, ":{ ;1;2}"), KlongSyntaxError)

    def test_a_list_cannot_be_a_key(self):
        for level in LEVELS:
            e = error(level, ":{;[1 2];3}")
            self.assertIsInstance(e, KlongTypeError)
            self.assertIn("key", str(e))
            self.assertIsNotNone(e.span)

    def test_undefined_name_inside(self):
        e = error(2, ":{;1;nope}")
        self.assertIsInstance(e, KlongNameError)
        self.assertEqual(e.span, (5, 9))

    def test_plain_literal_is_unchanged(self):
        for level in LEVELS:
            k = interp(level, "mode::7")
            self.assertEqual(k(':{["mode" mode]}'), {"mode": KGSym("mode")})
            self.assertEqual(plain(k(":{[1 2] [3 4]}")), {1: 2, 3: 4})
            self.assertEqual(k(":{}"), {})

    def test_dump(self):
        D = lambda t: dump_text(t).rstrip("\n")
        self.assertEqual(D(':{;"a";1+1;b;2}'), '(edict (str "a") (d "+" (num 1) (num 1)) (sym b) (num 2))')
        self.assertEqual(D(":{;}"), "(edict)")
        self.assertEqual(D(":{;1;:{;2;3}}"), "(edict (num 1) (edict (num 2) (num 3)))")
        self.assertEqual(D(":{[1 2]}"), "(dict ((num 1) (num 2)))")


class TestNestedLiterals(unittest.TestCase):
    def test_nested_in_a_dictionary(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(num(run(level, 'c:::{["GET" :{["/" 2]}]};(c?"GET")?"/"')), 2)
                r = run(level, ":{[1 :{[2 3]}]}")
                self.assertIsInstance(r[1], dict)
                self.assertEqual(plain(r), {1: {2: 3}})
                self.assertEqual(plain(run(level, ":{[1 :{[2 :{[3 4]}]}]}")), {1: {2: {3: 4}}})

    def test_nested_in_a_list(self):
        for level in LEVELS:
            with self.subTest(level=level):
                r = run(level, "[1 :{[2 3]} 4]")
                self.assertEqual(len(r), 3)
                self.assertEqual(plain(r[1]), {2: 3})
                self.assertEqual(num(run(level, "l::[:{[1 2]} :{[1 3]}];(l@1)?1")), 3)
                self.assertEqual(num(run(level, "#[:{} :{}]")), 2)

    def test_each_evaluation_gives_a_fresh_copy(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = interp(level, "mk::{:{[1 :{[2 3]}]}}", "p::mk()", "(p?1),[9 9]")
                self.assertEqual(plain(k("p")), {1: {2: 3, 9: 9}})
                self.assertEqual(plain(k("mk()")), {1: {2: 3}})
                k = interp(level, "ml::{[:{[1 2]}]}", "p::ml()", "(p@0),[9 9]")
                self.assertEqual(plain(k("p@0")), {1: 2, 9: 9})
                self.assertEqual(plain(k("(ml())@0")), {1: 2})

    def test_writes_as_it_reads(self):
        k = KlongInterpreter()
        self.assertEqual(k("$:{[1 :{[2 3]}]}"), ":{[1 :{[2 3]}]}")

    def test_dump_is_what_it_was(self):
        self.assertEqual(dump_text(":{[1 :{[2 3]}]}"), "(dict ((num 1) (dict ((num 2) (num 3)))))\n")
        self.assertEqual(dump_text("[:{[1 2]} 3]"), "(list (dict ((num 1) (num 2))) (num 3))\n")


class TestMerge(unittest.TestCase):
    def test_merge_in_place(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = interp(level, "b:::{[1 2]}", "c:::{[3 4]}")
                self.assertEqual(plain(k("b,c")), {1: 2, 3: 4})
                self.assertEqual(plain(k("b")), {1: 2, 3: 4})       # b shows the merge
                self.assertEqual(plain(k("c")), {3: 4})

    def test_right_wins(self):
        for level in LEVELS:
            self.assertEqual(plain(run(level, ":{[1 2] [3 4]},:{[3 5] [6 7]}")), {1: 2, 3: 5, 6: 7})

    def test_empty(self):
        for level in LEVELS:
            self.assertEqual(plain(run(level, ":{},:{[1 2]}")), {1: 2})
            self.assertEqual(plain(run(level, ":{[1 2]},:{}")), {1: 2})
            self.assertEqual(run(level, ":{},:{}"), {})

    def test_merge_over_a_list_of_dictionaries(self):
        for level in LEVELS:
            self.assertEqual(plain(run(level, "l::[:{[1 2]} :{[3 4]} :{[1 5]}];,/l")), {1: 5, 3: 4})

    def test_pair_join_is_unchanged(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(plain(run(level, "[1 2],:{[1 0]}")), {1: 2})
                self.assertEqual(plain(run(level, ":{[1 0]},[1 2]")), {1: 2})
                self.assertEqual(plain(run(level, 'd:::{};d,"k",,5;d')), {"k": 5})
                self.assertEqual(plain(run(level, "d:::{[1 2]};e::d;d,1,,5;e")), {1: 5})
                # a list that is not a pair appends the dictionary, as before
                self.assertEqual(len(run(level, "[1 2 3],:{[1 0]}")), 4)
                self.assertEqual(num(run(level, "A::[:{} :{}];A::A,,:{};#A")), 3)

    def test_anything_else_is_a_type_error(self):
        for level in LEVELS:
            for text in ("d,5", 'd,"k"', "d,[1 2 3]", 'd,"ab"', "d,:sym", "d,[]", "d,[1]"):
                with self.subTest(level=level, text=text):
                    e = error(level, "d:::{[1 2]}", text)
                    self.assertIsInstance(e, KlongTypeError)
                    self.assertEqual(e.kind, "type")


class TestLookupWithAt(unittest.TestCase):
    def test_at_is_find(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = interp(level, 'd:::{["k" 1] [:s 2] [3 4] [1.5 6]}')
                self.assertEqual(num(k('d@"k"')), 1)
                self.assertEqual(num(k("d@:s")), 2)
                self.assertEqual(num(k("d@3")), 4)
                self.assertEqual(num(k("d@1.5")), 6)
                for key in ('"z"', ":t", "7", "2.5", "0cx"):
                    self.assertIs(k(f"d@{key}"), KLONG_UNDEFINED, key)
                    self.assertIs(k(f"d?{key}"), KLONG_UNDEFINED, key)

    def test_acceptance(self):
        k = interp(0, 'd:::{["k" 1]}')
        self.assertEqual(num(k('d@"k"')), 1)
        self.assertIs(k('d@"z"'), KLONG_UNDEFINED)
        self.assertIs(k('d?"z"'), KLONG_UNDEFINED)

    def test_values_of_every_kind(self):
        k = interp(0, 'd:::{;"l";[1 2 3];"s";"str";"d";:{[1 2]};"f";{x+1}}')
        self.assertEqual(nums(k('d@"l"')), [1, 2, 3])
        self.assertEqual(k('d@"s"'), "str")
        self.assertEqual(plain(k('d@"d"')), {1: 2})
        self.assertEqual(num(k('g::d@"f";g(1)')), 2)

    def test_a_list_of_keys(self):
        # d@[k1 k2] worked for keys that are present; it still does, and a missing key is :undefined
        for level in LEVELS:
            k = interp(level, "d:::{[1 10] [2 20]}")
            self.assertEqual(nums(k("d@[1 2 1]")), [10, 20, 10])
            r = k("d@[1 7]")
            self.assertEqual(num(r[0]), 10)
            self.assertIs(r[1], KLONG_UNDEFINED)
            self.assertEqual(len(k("d@[]")), 0)

    def test_at_on_other_things_is_unchanged(self):
        k = KlongInterpreter()
        self.assertEqual(num(k("[1 2 3]@1")), 2)
        self.assertEqual(k('"abc"@1'), "b")
        self.assertEqual(num(k("{x+1}@1")), 2)


class TestFormat(unittest.TestCase):
    def test_format_of_a_dictionary_is_its_written_form(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(run(level, "$:{[1 2]}"), ":{[1 2]}")
                self.assertEqual(run(level, '$:{["a" 1] [:b [1 2]]}'), ':{["a" 1] [:b [1 2]]}')
                self.assertEqual(run(level, "$:{}"), ":{}")
                self.assertEqual(run(level, '$:{;"k";1+1}'), ':{["k" 2]}')


if __name__ == "__main__":
    unittest.main()
