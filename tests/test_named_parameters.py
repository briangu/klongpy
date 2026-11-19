"""
Named parameters and more than three arguments, f(a;b;c;d)::{...} (design: docs/language-additions.md).

The left side of :: is a call of a plain name with plain names as arguments;
the right side is a function literal. The function takes exactly that many
arguments, bound by position to those names. Every rule is run at strict
levels 0, 1 and 2. The syntax exists on the token parser only.
"""
import copy
import pickle
import unittest
import warnings

from klongpy import KlongInterpreter
from klongpy.core import KGCall, KGFn, KGFnWrapper, KGSym
from tests.treedump import dump_text
from klongpy.errors import (KlongArityError, KlongError, KlongNameError, KlongSyntaxError, KlongSyntaxWarning)

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


class TestDefinitionAndCall(unittest.TestCase):
    def test_four_parameters(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(num(run(level, "f(a;b;c;d)::{a+b+c+d};f(1;2;3;4)")), 10)
                self.assertEqual(num(run(level, "f(a;b;c;d)::{(a*1000)+(b*100)+(c*10)+d};f(1;2;3;4)")), 1234)

    def test_zero_one_two_three_and_six(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(run(level, "f()::{42};f()"), 42)
                self.assertEqual(num(run(level, "f(a)::{a*2};f(21)")), 42)
                self.assertEqual(num(run(level, "f(a;b)::{a-b};f(5;3)")), 2)
                self.assertEqual(num(run(level, "f(a;b;c)::{a-b-c};f(9;5;3)")), 7)
                self.assertEqual(nums(run(level, "f(a;b;c;d;e;g)::{[;g;e;d;c;b;a]};f(1;2;3;4;5;6)")), [6, 5, 4, 3, 2, 1])

    def test_arity_is_the_number_of_parameters(self):
        k = interp(0, "f(a;b;c;d)::{a}", "g()::{1}", "h(a;b)::{1}")
        self.assertEqual([k[n].arity for n in "fgh"], [4, 0, 2])
        f = k["f"].fn
        self.assertEqual(f.params, (KGSym("a"), KGSym("b"), KGSym("c"), KGSym("d")))

    def test_the_definition_returns_the_function(self):
        k = KlongInterpreter()
        r = k("f(a;b)::{a+b}")
        self.assertIsInstance(r, KGFn)
        self.assertEqual(r.arity, 2)

    def test_arguments_are_evaluated_once_in_order(self):
        for level in LEVELS:
            k = interp(level, "n::0", "tick()::{n::n+1;n}", "f(a;b;c;d)::{[;a;b;c;d]}")
            self.assertEqual(nums(k("f(tick();tick();tick();tick())")), [1, 2, 3, 4])
            self.assertEqual(k("n"), 4)

    def test_values_of_every_kind(self):
        for level in LEVELS:
            k = interp(level, 'f(s;l;d;g;q)::{[;s;#l;d?1;g(2);q]}')
            r = k('f("str";[1 2 3];:{[1 2]};{x*10};:sym)')
            self.assertEqual(r[0], "str")
            self.assertEqual([num(r[1]), num(r[2]), num(r[3])], [3, 2, 20])
            self.assertEqual(r[4], KGSym("sym"))

    def test_a_symbol_argument_is_not_looked_up(self):
        for level in LEVELS:
            self.assertEqual(run(level, "foo::5", "f(a;b)::{a}", "v::[:foo :bar]", "f(v@0;1)"), KGSym("foo"))

    def test_parameters_are_local(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = interp(level, "a::100;b::200", "f(a;b)::{a+b}")
                self.assertEqual(num(k("f(1;2)")), 3)
                self.assertEqual((k("a"), k("b")), (100, 200))
                # assigning a parameter assigns the local
                k("g(a)::{a::a+1;a}")
                self.assertEqual(num(k("g(1)")), 2)
                self.assertEqual(k("a"), 100)

    def test_locals_list(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = interp(level, "f(a;b)::{[t];t::a;t+b}")
                self.assertEqual(num(k("f(1;2)")), 3)
                if level < 2:
                    self.assertEqual(k("t"), KGSym("t"))        # not defined
                else:
                    with self.assertRaises(KlongNameError):
                        k("t")
                self.assertEqual(tuple(k["f"].fn.locals), (KGSym("t"),))

    def test_a_local_named_like_a_parameter_does_not_hide_the_argument(self):
        for level in LEVELS:
            self.assertEqual(num(run(level, "f(a;b)::{[a t];t::a;t+b};f(1;2)")), 3)

    def test_head_destructuring_inside(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = interp(level, "p::7", "f(v;k)::{[p q]::v;k*p+q}")
                self.assertEqual(num(k("f([1 2];10)")), 30)
                self.assertEqual(k("p"), 7)

    def test_x_in_a_nested_literal(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(num(run(level, "f(a;b)::{{x}(a)+b};f(1;2)")), 3)
                self.assertEqual(nums(run(level, "f(a;b)::{{x+b}'a};f([1 2 3];10)")), [11, 12, 13])
                self.assertEqual(num(run(level, "f(a;b)::{{x*y}/a,b};f(3;4)")), 12)

    def test_called_from_a_function_of_x(self):
        # the caller's x is not the callee's business
        for level in LEVELS:
            self.assertEqual(num(run(level, "f(a;b)::{a-b}", "g::{f(x;1)+x}", "g(10)")), 19)

    def test_recursion_by_name_and_by_dot_f(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(num(run(level, "fact(n;acc)::{:[n<1;acc;fact(n-1;acc*n)]};fact(5;1)")), 120)
                self.assertEqual(num(run(level, "fact(n;acc)::{:[n<1;acc;.f(n-1;acc*n)]};fact(5;1)")), 120)
                self.assertEqual(num(run(level, "fib(n)::{:[n<2;n;fib(n-1)+fib(n-2)]};fib(15)")), 610)
                self.assertEqual(num(run(level, "sum(n;a;b;c)::{[t];t::a+b+c;:[n<1;t;.f(n-1;a+1;b;c)]};sum(3;0;0;0)")), 3)

    def test_redefinition(self):
        for level in LEVELS:
            k = interp(level, "f(a;b)::{a+b}", "f(a;b;c)::{a*b*c}")
            self.assertEqual(num(k("f(2;3;4)")), 24)
            k("f::{x+1}")
            self.assertEqual(num(k("f(1)")), 2)
            k("f(p)::{p+10}")
            self.assertEqual(num(k("f(1)")), 11)

    def test_module_qualified(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = KlongInterpreter(strict=level)
                k('.module(:m);k::3;mf(a;b)::{k*a+b};mg::{mf(x;1)};.module(0)')
                self.assertEqual(num(k("mg(2)")), 9)
                self.assertEqual(num(k("mf(1;1)")), 6)

    def test_function_value_can_be_passed_and_stored(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = interp(level, "f(a;b;c;d)::{a+b+c+d}", "g::f", "ap(fn;v)::{fn(v;v;v;v)}", "d:::{}", "d,:k,f")
                self.assertEqual(num(k("g(1;2;3;4)")), 10)
                self.assertEqual(num(k("ap(f;2)")), 8)
                self.assertEqual(num(k("h::d?:k;h(1;1;1;1)")), 4)

    def test_function_returned_from_a_function(self):
        for level in LEVELS:
            k = interp(level, "f(a;b;c;d)::{a+b+c+d}", "mk()::{f}", "g::mk()")
            self.assertEqual(num(k("g(1;2;3;4)")), 10)

    def test_newlines_in_the_definition(self):
        k = interp(0, "f(a;\n  b)::{\n a+\n b}")
        self.assertEqual(num(k("f(1;2)")), 3)

    def test_copy_and_pickle(self):
        k = interp(0, "f(a;b;c;d)::{[t];t::a+b;t+c+d}")
        f = k["f"].fn
        for g in (copy.deepcopy(f), pickle.loads(pickle.dumps(f))):
            k["g"] = g
            self.assertEqual(num(k("g(1;2;3;4)")), 10)
            with self.assertRaises(KlongArityError):
                k("g(1;2;3;4;5)")


class TestArity(unittest.TestCase):
    def test_too_many_arguments_is_an_arity_error_at_every_level(self):
        for level in LEVELS:
            with self.subTest(level=level):
                e = error(level, "f(a;b)::{a+b};f(1;2;3)")
                self.assertIsInstance(e, KlongArityError)
                self.assertEqual(str(e), "f takes 2 arguments, got 3")
                self.assertEqual(e.kind, "arity")
                e = error(level, "f()::{1};f(1)")
                self.assertEqual(str(e), "f takes no arguments, got 1")
                e = error(level, "f(a)::{a};f(1;2;3;4;5)")
                self.assertEqual(str(e), "f takes 1 argument, got 5")

    def test_too_many_has_a_position(self):
        e = error(0, "f(a;b)::{a+b}", "1+f(1;2;3)")
        self.assertEqual(e.span, (2, 10))
        self.assertIn("arity error: f takes 2 arguments, got 3", e.render())

    def test_legacy_functions_keep_dropping_extras_below_level_2(self):
        for level in (0, 1):
            self.assertEqual(run(level, "f::{x};f(1;2;3;4)"), 1)
        self.assertIsInstance(error(2, "f::{x};f(1;2;3;4)"), KlongArityError)

    def test_fewer_arguments_is_not_a_call(self):
        # as for a function of x, y, z: the unapplied call is the value
        for level in LEVELS:
            r = run(level, "f(a;b;c;d)::{a+b+c+d};f(1;2)")
            self.assertIsInstance(r, KGCall)

    def test_through_dot_f(self):
        for level in LEVELS:
            e = error(level, "f(a;b)::{:[a;.f(0;1;2);b]};f(1;2)")
            self.assertIsInstance(e, KlongArityError)
            e = error(level, "f(a;b)::{:[a;.f(0);b]};f(1;2)")
            self.assertIsInstance(e, KlongArityError)
            self.assertEqual(str(e), "the function takes 2 arguments, got 1")


class TestProjection(unittest.TestCase):
    def test_holes_by_position(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = interp(level, "f(a;b;c;d)::{(a*1000)+(b*100)+(c*10)+d}")
                self.assertEqual(num(k("g::f(1;;;4);g(2;3)")), 1234)
                self.assertEqual(num(k("g::f(;2;3;);g(1;4)")), 1234)
                self.assertEqual(num(k("g::f(;;;4);g(1;2;3)")), 1234)
                self.assertEqual(num(k("g::f(1;2;3;);g(4)")), 1234)

    def test_acceptance(self):
        for level in LEVELS:
            self.assertEqual(num(run(level, "f(a;b;c;d)::{a+b+c+d};g::f(1;;;4);g(2;3)")), 10)

    def test_projection_of_a_projection(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = interp(level, "f(a;b;c;d;e)::{(a*10000)+(b*1000)+(c*100)+(d*10)+e}")
                k("g::f(1;;;;5)")
                k("h::g(2;;)")
                k("i::h(3;)")
                self.assertEqual(num(k("i(4)")), 12345)
                self.assertEqual(num(k("h(3;4)")), 12345)

    def test_five_levels_of_projection(self):
        # a function of x, y, z can be projected twice at most; n parameters allow n-1 times
        for level in LEVELS:
            k = interp(level, "f(a;b;c;d;e;g)::{[;a;b;c;d;e;g]}", "p1::f(1;;;;;)", "p2::p1(2;;;;)", "p3::p2(3;;;)",
                       "p4::p3(4;;)", "p5::p4(5;)")
            self.assertEqual(nums(k("p5(6)")), [1, 2, 3, 4, 5, 6])

    def test_projection_arguments_can_be_lists(self):
        for level in LEVELS:
            self.assertEqual(nums(run(level, "f(a;b;c;d)::{a,b,c,d};g::f([1 2];;[3 4];);g(0;9)")), [1, 2, 0, 3, 4, 9])

    def test_too_many_for_a_projection(self):
        for level in LEVELS:
            with self.subTest(level=level):
                e = error(level, "f(a;b;c;d)::{a+b+c+d};g::f(1;;;4);g(2;3;5)")
                self.assertIsInstance(e, KlongArityError)

    def test_projection_under_each(self):
        for level in LEVELS:
            self.assertEqual(nums(run(level, "f(a;b;c;d)::{a+b+c+d};g::f(1;2;3;);g'[1 2 3]")), [7, 8, 9])
            self.assertEqual(nums(run(level, "f(a;b;c;d)::{a+b+c+d};f(1;2;3;)'[1 2 3]")), [7, 8, 9])


class TestApplyAndAdverbs(unittest.TestCase):
    def test_at_spreads_a_list(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(num(run(level, "f(a;b;c;d)::{a+b+c+d};f@[1 2 3 4]")), 10)
                self.assertEqual(num(run(level, "f(a)::{a*2};f@21")), 42)
                self.assertEqual(nums(run(level, "f(a;b)::{a,b};f@[;[1 2];[3 4]]")), [1, 2, 3, 4])

    def test_at_with_the_wrong_number(self):
        for level in LEVELS:
            with self.subTest(level=level):
                e = error(level, "f(a;b;c;d)::{a+b+c+d};f@[1 2 3 4 5]")
                self.assertIsInstance(e, KlongArityError)
                # @ receives the function value, which does not know its name
                self.assertEqual(str(e), "the function takes 4 arguments, got 5")
                e = error(level, "f(a;b;c;d)::{a+b+c+d};f@[1 2]")
                self.assertIsInstance(e, KlongArityError)
                self.assertEqual(str(e), "the function takes 4 arguments, got 2")

    def test_monad_and_dyad_under_adverbs(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = interp(level, "sq(a)::{a*a}", "add(a;b)::{a+b}")
                self.assertEqual(nums(k("sq'[1 2 3]")), [1, 4, 9])
                self.assertEqual(num(k("add/[1 2 3 4]")), 10)
                self.assertEqual(nums(k("add\\[1 2 3 4]")), [1, 3, 6, 10])
                self.assertEqual(nums(k("[1 2 3]add'[10 20 30]")), [11, 22, 33])
                self.assertEqual(nums(k("1 add:\\[10 20 30]")), [11, 21, 31])
                self.assertEqual(num(k("1 add 2")), 3)

    def test_wrong_arity_under_an_adverb_is_an_arity_error(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = interp(level, "f(a;b;c;d)::{a+b+c+d}", "sq(a)::{a*a}", "add(a;b)::{a+b}")
                for text in ("f'[1 2 3]", "f/[1 2 3]", "[1 2]f'[3 4]", "sq/[1 2 3]", "add'[1 2 3]"):
                    with self.assertRaises(KlongArityError, msg=text):
                        k(text)
                e = error(level, "f(a;b;c;d)::{a+b+c+d};f'[1 2 3]")
                self.assertEqual(str(e), "f takes 4 arguments, got 1")
                self.assertIsNotNone(e.span)


class TestFromPython(unittest.TestCase):
    def test_positional_and_keyword(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = interp(level, "f(a;b)::{a-b}", "g(a;b;c;d;e)::{[;a;b;c;d;e]}")
                f = k["f"]
                self.assertIsInstance(f, KGFnWrapper)
                self.assertEqual(num(f(5, 3)), 2)
                self.assertEqual(num(f(5, b=3)), 2)
                self.assertEqual(num(f(b=3, a=5)), 2)
                self.assertEqual(nums(k["g"](1, 2, e=5, d=4, c=3)), [1, 2, 3, 4, 5])
                self.assertEqual(nums(k["g"](1, 2, 3, 4, 5)), [1, 2, 3, 4, 5])

    def test_acceptance(self):
        k = interp(0, "f(a;b)::{a+b}")
        self.assertEqual(num(k["f"](1, b=2)), 3)
        for args, kwargs in (((1,), {}), ((1, 2, 3), {}), ((1,), {"c": 2}), ((1,), {"a": 2})):
            with self.subTest(args=args, kwargs=kwargs):
                with self.assertRaises(KlongArityError):
                    k["f"](*args, **kwargs)

    def test_messages(self):
        k = interp(0, "f(a;b)::{a+b}")
        cases = {
            ((1,), ()): "f takes 2 arguments, got 1",
            ((1, 2, 3), ()): "f takes 2 arguments, got 3",
            ((1,), (("c", 2),)): "f has no parameter named c",
            ((1,), (("a", 2),)): "f got its parameter a twice",
            ((), (("a", 2),)): "f takes 2 arguments, got 1",
        }
        for (args, kwargs), msg in cases.items():
            with self.assertRaises(KlongArityError) as c:
                k["f"](*args, **dict(kwargs))
            self.assertEqual(str(c.exception), msg)
            self.assertIsInstance(c.exception, RuntimeError)      # what the wrapper always raised

    def test_functions_of_x_y_z_take_those_keywords(self):
        k = interp(0, "f::{x-y}")
        self.assertEqual(num(k["f"](3, 2)), 1)
        self.assertEqual(num(k["f"](3, y=2)), 1)
        self.assertEqual(num(k["f"](y=2, x=3)), 1)
        for args, kwargs in (((1,), {"z": 2}), ((1,), {"w": 2}), ((1, 2), {"y": 2}), ((), {"x": 1})):
            with self.assertRaises(KlongArityError):
                k["f"](*args, **kwargs)
        with self.assertRaises(RuntimeError) as c:
            k["f"](1)
        self.assertEqual(str(c.exception), "Klong function called with 1 but expected 2")

    def test_redefinition_is_seen_by_the_wrapper(self):
        k = interp(0, "f(a;b)::{a+b}")
        f = k["f"]
        k("f(a;b;c)::{a*b*c}")
        self.assertEqual(num(f(2, 3, c=4)), 24)

    def test_lists_become_arrays(self):
        k = interp(0, "f(a;b)::{(+/a)*b}")
        self.assertEqual(num(k["f"]([1, 2, 3], b=2)), 12)

    def test_wrapper_without_a_name(self):
        k = interp(0, "f(a;b;c;d)::{a+b+c+d}")
        w = KGFnWrapper(k, k["f"].fn, sym=None)
        self.assertEqual(num(w(1, 2, 3, d=4)), 10)

    def test_apply_with_the_body(self):
        # IPC and timer callbacks hold the body of a function and call apply with it
        k = interp(0, "f(a;b;c;d)::{a+b+c+d}")
        body = k["f"].fn.a
        self.assertEqual(num(k.apply(body, [1, 2, 3, 4])), 10)
        with self.assertRaises(KlongArityError):
            k.apply(body, [1, 2, 3])
        self.assertEqual(num(k.apply(KGSym("f"), [1, 2, 3, 4])), 10)


class TestSyntax(unittest.TestCase):
    def test_right_side_must_be_a_function_literal(self):
        for level in LEVELS:
            for text in ("f(a;b)::5", "f(a;b)::a+b", "f(a;b)::g", "f(a;b)::{a}+1", "f(a;b)::{a+b}(1;2)", "f(a;b)::[1 2]",
                         "f(a;b)::{a}'b", "f()::1"):
                with self.subTest(level=level, text=text):
                    e = error(level, text)
                    self.assertIsInstance(e, KlongSyntaxError)
                    self.assertEqual(str(e), "named-parameter definition needs a function body in braces")
                    self.assertIn("f(a;b)::{", e.hint)
                    self.assertIsNotNone(e.span)

    def test_missing_right_side(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", KlongSyntaxWarning)
            for level in LEVELS:
                self.assertIsInstance(error(level, "f(a;b)::"), KlongSyntaxError)

    def test_x_y_z_in_the_body(self):
        for level in LEVELS:
            for text, name in (("f(a;b)::{x}", "x"), ("f(a;b)::{a+y}", "y"), ("f(a)::{[t];t::z;t}", "z"),
                               ("f(a)::{:[a;x;1]}", "x"), ("f(a)::{[;a;x]}", "x"), ("f(a)::{g(x)}", "x"),
                               ("f(a)::{x'a}", "x"), ("f(a)::{:{;1;y}}", "y"), ("f(a)::{[x y];{x}(z)}", "z")):
                at = text.index(name, text.index("::{"))
                if "[x y]" in text:
                    at = text.rindex(name)
                with self.subTest(level=level, text=text):
                    e = error(level, text)
                    self.assertIsInstance(e, KlongSyntaxError)
                    self.assertEqual(str(e), f"{name} is not available in a function with named parameters")
                    self.assertEqual(e.span, (at, at + 1))
                    self.assertIn("nested", e.hint)

    def test_x_as_data_or_in_a_nested_literal_is_fine(self):
        for level in LEVELS:
            for text in ("f(a)::{[x y z]}", "f(a)::{{x}(a)}", "f(a)::{{x+y}/a}", "f(a)::{:{[x 1]}}", "f(a)::{:x}",
                         "f(a)::{a,{[t];t::x;t}(1)}"):
                with self.subTest(level=level, text=text):
                    run(level, text)

    def test_what_is_not_this_form_keeps_its_meaning(self):
        # an argument that is not a name: an assignment to a non-name, as before
        for text in ("f(a;1)::{a}", "f(a;)::{a}", "f(a;:b)::{a}", "f(a;b+1)::{a}", "f(1)::5", "{f}(a)::{a}"):
            with self.subTest(text=text):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", KlongSyntaxWarning)
                    k = KlongInterpreter()
                    k(text)                                             # accepted silently at level 0
                    with self.assertRaises(KlongError):
                        k("f(1;2)")
                for level in (1, 2):
                    e = error(level, text)
                    self.assertIsInstance(e, KlongSyntaxError)
                    self.assertIn("left side of ::", str(e))

    def test_hint_for_a_call_on_the_left(self):
        e = error(1, "f(a;1)::{a}")
        self.assertIn("f(a;b)::{", e.hint)

    def test_reserved_and_duplicate_parameter_names(self):
        cases = {"f(a;x)::{a}": "x, y and z", "f(a;a)::{a}": "a is named twice", "f(x)::{1}": "x, y and z",
                 "f(a;b;a)::{a}": "a is named twice"}
        for text, word in cases.items():
            with self.subTest(text=text):
                for level in (1, 2):
                    e = error(level, text)
                    self.assertIsInstance(e, KlongSyntaxError)
                    self.assertIn(word, str(e))
                    self.assertEqual(e.span[0], 0)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    k = KlongInterpreter()
                    k(text)
                w = [q for q in w if issubclass(q.category, KlongSyntaxWarning)]
                self.assertEqual(len(w), 1)
                self.assertIn(word, str(w[0].message))
                self.assertIn("nothing is defined", str(w[0].message))
                self.assertEqual(w[0].message.code, "not-a-definition")

    def test_no_warning_for_a_definition(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for level in LEVELS:
                self.assertEqual(num(run(level, "f(a;b)::{a+b};f(1;2)")), 3)


class TestTree(unittest.TestCase):
    def D(self, text):
        return dump_text(text).rstrip("\n")

    def test_dump(self):
        self.assertEqual(self.D("f(a;b)::{a+b}"), '(def (sym f) (fn 2 (params a b) (d "+" (sym a) (sym b))))')
        self.assertEqual(self.D("f()::{1}"), "(def (sym f) (fn 0 (params) (num 1)))")
        self.assertEqual(self.D("f(a;b;c;d)::{[t];t::a;t}"),
                         "(def (sym f) (fn 4 (params a b c d) (locals t) (def (sym t) (sym a)) (sym t)))")
        self.assertEqual(self.D("f(v)::{[p q]::v;p}"),
                         "(def (sym f) (fn 1 (params v) (locals p q) (destructure (p q) (sym v)) (sym p)))")
        self.assertEqual(self.D("f(a)::{a;1}"), "(def (sym f) (fn 1 (params a) (sym a) (num 1)))")
        self.assertEqual(self.D(".module(:m);f(a)::{a}"),
                         "(call (sym .module) (qsym m))\n(def (sym f`m) (fn 1 (params a`m) (sym a`m)))")

    def test_unchanged_forms(self):
        self.assertEqual(self.D("f::{x+y}"), '(def (sym f) (fn 2 (d "+" (arg x) (arg y))))')
        self.assertEqual(self.D("f(a;b)"), "(call (sym f) (sym a) (sym b))")
        self.assertEqual(self.D("g::f(a;b)"), "(def (sym g) (call (sym f) (sym a) (sym b)))")
        self.assertEqual(self.D("f(a;1)::{a}"), "(def (call (sym f) (sym a) (num 1)) (fn 0 (sym a)))")


class TestCompiledPath(unittest.TestCase):
    """The expression compiler sees a parameter as it sees any variable; it never binds one."""

    def test_arithmetic_on_parameters_of_changing_kind(self):
        for level in LEVELS:
            k = interp(level, "f(a;b;c;d)::{(a*b)+(c*d)}")
            self.assertEqual(num(k("f(1;2;3;4)")), 14)
            self.assertEqual(num(k("f(1.5;2;3;4)")), 15.0)
            self.assertEqual(nums(k("f([1 2];2;3;4)")), [14, 16])
            self.assertEqual(num(k("f(1;2;3;4)")), 14)

    def test_reduce_on_a_parameter(self):
        for level in LEVELS:
            k = interp(level, "s(v;k)::{k*+/v}")
            self.assertEqual(num(k("s([1 2 3];2)")), 12)
            self.assertEqual(num(k("s([1.5 2.5];2)")), 8.0)

    def test_a_global_of_the_same_name_does_not_leak_into_the_compiled_body(self):
        k = interp(0, "a::100", "b::200", "f(a;b)::{a+b}")
        self.assertEqual(num(k("a+b")), 300)
        self.assertEqual(num(k("f(1;2)")), 3)
        self.assertEqual(num(k("a+b")), 300)
        self.assertEqual(num(k("f(1;2)+a+b")), 303)


class TestIPCFunctionText(unittest.TestCase):
    def test_a_named_parameter_function_goes_with_its_parameter_list(self):
        # the parameter list is outside the braces, so the text sent carries it:
        # "{a+b}" alone would arrive as a function of no arguments
        from klongpy.ipc.endpoint import function_source_text, lambda_value
        from klongpy.ipc import codec
        k = KlongInterpreter()
        k("f(a;b)::{a+b}")
        k("g::{x+y}")
        f = k["f"].fn
        self.assertEqual(function_source_text(f), "fn(a;b)::{a+b}")
        self.assertEqual(function_source_text(k["g"].fn), "{x+y}")
        got = codec.decode(codec.encode(f))
        self.assertEqual(got.text, "fn(a;b)::{a+b}")
        k("h(a;b;c;d;e)::{a}")
        got = codec.decode(codec.encode(k["h"].fn))
        self.assertEqual(got.text, "fn(a;b;c;d;e)::{a}")
        ref = lambda_value(KlongInterpreter(), got)
        self.assertEqual(ref.arity, 5)


class TestDisplayOfManyParameters(unittest.TestCase):
    def test_four_or_more_parameters_display_as_fn_n(self):
        k = KlongInterpreter()
        k("f(a;b;c;d)::{a+b+c+d}")
        k("g(a;b;c;d;e;f)::{a}")
        k("h(a;b;c)::{a}")
        self.assertEqual(str(k("f")), ":fn/4")
        self.assertEqual(str(k("g")), ":fn/6")
        self.assertEqual(str(k("h")), ":triad")


if __name__ == "__main__":
    unittest.main()
