"""
The fixtures of the static profile: every rule id has one
accept and one reject file, every file belongs to a rule, and every reject
file starts with a valid ``expect`` line. (tests/static/test_profile.py runs
the checker on them.)
"""
import os
import re
import unittest

from klongpy import KlongInterpreter
from klongpy.static import RULES
from klongpy.static.rules import STREAM_RULES
from klongpy.static.profile_doc import generated_block

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECT = re.compile(r':" expect: static line (\d+) ([a-z0-9-]+) "\n')


def names(sub):
    return sorted(f[:-3] for f in os.listdir(os.path.join(HERE, sub)) if f.endswith(".kg"))


class TestFixtures(unittest.TestCase):
    def test_every_rule_has_both_fixtures_and_nothing_else_is_there(self):
        self.assertEqual(names("accept"), sorted(list(RULES) + list(STREAM_RULES)))
        self.assertEqual(names("reject"), sorted(list(RULES) + list(STREAM_RULES)))

    def test_every_reject_file_has_a_valid_expect_line(self):
        for rule in list(RULES) + list(STREAM_RULES):
            with self.subTest(rule=rule):
                with open(os.path.join(HERE, "reject", rule + ".kg")) as f:
                    lines = f.readlines()
                m = EXPECT.fullmatch(lines[0])
                self.assertIsNotNone(m)
                self.assertEqual(m.group(2), rule)
                self.assertTrue(1 < int(m.group(1)) <= len(lines), "the expected line exists")
                self.assertTrue(lines[int(m.group(1)) - 1].strip())

    def test_every_accept_file_says_what_it_shows(self):
        for rule in list(RULES) + list(STREAM_RULES):
            with self.subTest(rule=rule):
                with open(os.path.join(HERE, "accept", rule + ".kg")) as f:
                    self.assertTrue(f.readline().startswith(f':" accept: {rule} - '))

    def test_accept_files_are_valid_klong_at_strict_level_2(self):
        # parse only: some call externs, which have no stand-in here
        for rule in list(RULES) + list(STREAM_RULES):
            with self.subTest(rule=rule):
                with open(os.path.join(HERE, "accept", rule + ".kg")) as f:
                    KlongInterpreter(strict=2).prog(f.read())

    def test_every_rule_id_is_printed_in_the_profile_document(self):
        block = generated_block()
        for rule in list(RULES) + list(STREAM_RULES):
            self.assertIn(f"`{rule}`", block)

    def test_rule_ids_are_short_stable_words(self):
        for rule, r in RULES.items():
            self.assertRegex(rule, r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
            self.assertEqual(r.id, rule)
            self.assertTrue(r.title and not r.title.endswith("."))


if __name__ == "__main__":
    unittest.main()
