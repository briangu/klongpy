import unittest

from klongpy import KlongInterpreter
from klongpy.parser import UnexpectedChar
from klongpy.utils import CallbackEvent


class TestReviewRegressions(unittest.TestCase):
    def test_trailing_unmatched_brace_raises(self):
        klong = KlongInterpreter()
        with self.assertRaises(UnexpectedChar):
            klong("a::1}}")

    def test_postfix_indexing_works_inside_and_outside_parentheses(self):
        klong = KlongInterpreter()
        klong("q::[3 8]")

        self.assertTrue(klong._backend.kg_equal(klong("q[0],q[-1]"), [3, 8]))
        self.assertTrue(klong._backend.kg_equal(klong("(q[0],q[-1])"), [3, 8]))

    def test_semicolon_string_argument_is_not_treated_as_projection(self):
        klong = KlongInterpreter()
        klong('f::{x,y}')
        self.assertEqual(klong('f("hello";";")'), "hello;")

    def test_join_merges_dictionaries(self):
        klong = KlongInterpreter()
        klong("b:::{[1 2]}")
        klong("c:::{[3 4]}")
        self.assertEqual(klong("b,c"), {1: 2, 3: 4})

    def test_join_can_append_dictionary_values(self):
        klong = KlongInterpreter()
        result = klong("A::[];A::A,:{};A::A,:{};A::A,:{};A")
        self.assertEqual(result.tolist(), [{}, {}, {}])

    def test_nested_dictionary_literal_materializes_inner_dictionary(self):
        klong = KlongInterpreter()
        self.assertEqual(klong(':{[1 :{[2 3]}]}'), {1: {2: 3}})

        klong('c:::{["GET" :{["/" 2]}]}')
        self.assertEqual(klong('(c?"GET")?"/"'), 2)

    def test_take_repeats_nested_arrays_as_elements(self):
        klong = KlongInterpreter()
        result = klong("(4)#[[0 0]]")
        self.assertTrue(klong._backend.kg_equal(result, [[0, 0], [0, 0], [0, 0], [0, 0]]))

    def test_callback_event_allows_self_unsubscribe_during_trigger(self):
        event = CallbackEvent()
        called = []

        def cb():
            called.append(1)
            event.unsubscribe(cb)

        event.subscribe(cb)
        event.trigger()

        self.assertEqual(called, [1])
        self.assertNotIn(cb, event.subscribers)


if __name__ == "__main__":
    unittest.main()
