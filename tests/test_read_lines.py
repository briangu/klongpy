import os
import tempfile
import unittest

from klongpy import KlongInterpreter


class TestReadLines(unittest.TestCase):
    def test_rls_strips_the_line_separator_as_rl_does(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.txt")
            with open(path, "w") as f:
                f.write("ab\ncde\n\nlast")
            klong = KlongInterpreter()
            klong["path"] = path
            lines = klong('.fc(.ic(path));.rls()')
            self.assertEqual([str(s) for s in lines], ["ab", "cde", "", "last"])
            self.assertEqual(klong('.fc(.ic(path));.rl()'), "ab")


if __name__ == "__main__":
    unittest.main()
