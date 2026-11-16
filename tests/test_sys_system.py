import unittest

from klongpy import KlongInterpreter


class TestSysSystem(unittest.TestCase):

    def setUp(self):
        self.klong = KlongInterpreter()

    def test_sys_true_returns_0(self):
        self.assertEqual(self.klong('.sys("true")'), 0)

    def test_sys_false_returns_1(self):
        self.assertEqual(self.klong('.sys("false")'), 1)

    def test_sys_exit_code(self):
        self.assertEqual(self.klong('.sys("exit 3")'), 3)

    def test_sys_echo_writes_output(self):
        import subprocess, sys
        # run in a child so the inherited stdout can be captured reliably
        code = ("from klongpy import KlongInterpreter;"
                "print('rc=%s' % KlongInterpreter()('.sys(\"echo hi\")'))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
        self.assertEqual(out.split(), ["hi", "rc=0"])

    def test_sys_args_split(self):
        self.assertEqual(self.klong('.sys("ls -l /")'), 0)


if __name__ == '__main__':
    unittest.main()
