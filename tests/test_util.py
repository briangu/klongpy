import unittest
from klongpy.core import *
from utils import die, array_equal

class TestUtil(unittest.TestCase):

    def test_die(self):
        with self.assertRaises(RuntimeError):
            die("test")

    def test_read_shifted_comment(self):
        t = ':"hello"'
        i = 0
        i = read_shifted_comment(t,i+2)
        self.assertEqual(i,len(t))

    def test_read_shifted_comment_embedded(self):
        t = 'this is a :"hello" test'
        i = t.index(':"')+2
        i = read_shifted_comment(t,i)
        self.assertEqual(i,t.index(" test"))

    def test_read_shifted_comment_double_quote_embedded(self):
        t = 'this is a :"hello""world" test'
        i = t.index(':"')+2
        i = read_shifted_comment(t,i)
        self.assertEqual(i,t.index(" test"))

    def test_read_string(self):
        t = '"hello"'
        i = 0
        i,s = read_string(t,i+1)
        self.assertEqual(i,len(t))
        self.assertEqual(s,"hello")

    def test_read_substring(self):
        t = 'this is "hello" test'
        i = t.index('"')+1
        i,s = read_string(t,i)
        self.assertEqual(i,t.index(" test"))
        self.assertEqual(s,"hello")

    def test_read_quoted_substring(self):
        t = 'this is "hello""""world" test'
        i = t.index('"')+1
        i,s = read_string(t,i)
        self.assertEqual(i,t.index(" test"))
        self.assertEqual(s,'hello""world')

    def test_read_double_quoted_substring(self):
        t = 'this is "hello""""""world" test'
        i = t.index('"')+1
        i,s = read_string(t,i)
        self.assertEqual(i,t.index(" test"))
        self.assertEqual(s,'hello"""world')

    def test_read_empty_string(self):
        t = '""'
        i = 0
        i,s = read_string(t,i+1)
        self.assertEqual(i,len(t))
        self.assertEqual(s,'')

    def test_read_quoted_empty_string(self):
        t = '""""'
        i = 0
        i,s = read_string(t,i+1)
        self.assertEqual(i,len(t))
        # TODO: this seems like a bug in kg parsing strings since it starts on i+1
        self.assertEqual(s,'"')

    def test_read_double_quoted_empty_string(self):
        t = '""""""'
        i = 0
        i,s = read_string(t,i+1)
        self.assertEqual(i,len(t))
        # TODO: this seems like a bug in kg parsing strings since it starts on i+1
        self.assertEqual(s,'""')


    def test_read_quoted_embedded_string(self):
        t = '":["""";1;2]"'
        i = 0
        i,s = read_string(t,i+1)
        self.assertEqual(i,len(t))
        self.assertEqual(s,':["";1;2]')

    def test_embedded_string(self):
        t = '"A::""hello, world!"""'
        i = 0
        i,s = read_string(t,i+1)
        self.assertEqual(i,len(t))
        self.assertEqual(s,'A::"hello, world!"')

    def test_is_list(self):
        self.assertTrue(is_list([]))
        self.assertTrue(is_list(np.array([])))
        self.assertFalse(is_list(()))
        self.assertFalse(is_list(1))
        self.assertFalse(is_list(None))

    def test_is_iterable(self):
        self.assertTrue(is_iterable(""))
        self.assertTrue(is_iterable(''))
        self.assertTrue(is_iterable([]))
        self.assertFalse(is_iterable(None))
        self.assertFalse(is_iterable(1))

    def test_is_empty(self):
        self.assertTrue(is_empty(""))
        self.assertTrue(is_empty(''))
        self.assertTrue(is_empty([]))
        self.assertTrue(is_empty(np.array([])))
        self.assertFalse(is_empty(np.array([1])))
        self.assertFalse(is_empty([1]))
        self.assertFalse(is_empty(None))
        self.assertFalse(is_empty(1))

    def test_to_list(self):
        self.assertTrue(isinstance(to_list([]), list))
        self.assertEqual(to_list([]), [])
        self.assertTrue(isinstance(to_list(1), list))
        self.assertEqual(to_list([1]), [1])
        self.assertTrue(isinstance(to_list(np.array([])), list))
        self.assertEqual(to_list(np.array([])), [])
        self.assertTrue(isinstance(to_list(np.array([1])), list))
        self.assertEqual(to_list(np.array([1])), np.array([1]))

    def test_is_float(self, fn=is_float):
        self.assertFalse(fn(None))
        self.assertFalse(fn([]))
        self.assertFalse(fn([1]))
        self.assertFalse(fn({}))
        self.assertFalse(fn(""))
        self.assertFalse(fn("a"))
        self.assertFalse(fn("0"))
        self.assertFalse(fn("1"))
        self.assertFalse(fn("0.0"))
        self.assertFalse(fn("1.0"))
        self.assertTrue(fn(0))
        self.assertTrue(fn(1))
        self.assertTrue(fn(-1))
        self.assertTrue(fn(0.0))
        self.assertTrue(fn(-0.0))
        self.assertTrue(fn(1.0))
        self.assertTrue(fn(-1.0))

    def test_is_int(self, fn=is_integer, allow_floats=False):
        self.assertFalse(fn(None))
        self.assertFalse(fn([]))
        self.assertFalse(fn([1]))
        self.assertFalse(fn({}))
        self.assertFalse(fn(""))
        self.assertFalse(fn("a"))
        self.assertFalse(fn("0"))
        self.assertFalse(fn("1"))
        self.assertFalse(fn("0.0"))# if not allow_floats else self.assertTrue(fn("0.0"))
        self.assertFalse(fn("1.0"))# if not allow_floats else self.assertTrue(fn("1.0"))
        self.assertTrue(fn(0))
        self.assertTrue(fn(1))
        self.assertTrue(fn(-1))
        self.assertTrue(fn(0))
        self.assertTrue(fn(-0))
        self.assertTrue(fn(1))
        self.assertTrue(fn(-1))

    def test_is_number(self):
        self.test_is_float(fn=is_number)
        self.test_is_int(fn=is_number, allow_floats=True)

    def test_in_map(self):
        self.assertFalse(in_map(1,None))
        self.assertFalse(in_map(1,{}))
        self.assertTrue(in_map(1,{1:2}))

    def test_is_char(self):
        self.assertFalse(is_char(None))
        self.assertFalse(is_char(0))
        self.assertFalse(is_char([]))
        self.assertFalse(is_char({}))
        self.assertFalse(is_char(()))
        self.assertTrue(is_char(KGChar('1')))
        self.assertTrue(is_char(KGChar("1")))
        self.assertTrue(is_char(KGChar("-")))
        self.assertTrue(is_char(KGChar('a')))
        self.assertTrue(is_char(KGChar("a")))
        self.assertFalse(is_char("ab"))

    def test_cmatch(self):
        arr = "abc"
        self.assertFalse(cmatch(arr, 4, 'a'))
        [self.assertFalse(cmatch(arr, i, x)) for i,x in enumerate("def")]
        [self.assertTrue(cmatch(arr, i, x)) for i,x in enumerate(arr)]

    def test_cexpect(self):
        arr = "abc"
        with self.assertRaises(UnexpectedChar) as cm:
            cexpect(arr, 4, 'a')
        e = cm.exception
        self.assertEqual(e.args, ('t: abc pos: 4 char: a',))
        with self.assertRaises(UnexpectedChar):
            cexpect(arr, 0, 'e')
        for i,x in enumerate(arr):
            self.assertEqual(cexpect(arr, i, x), i+1)

    def test_safe_eq(self):
        self.assertFalse(safe_eq(1,[]))
        self.assertTrue(safe_eq(1,1))

    def test_merge_projections(self):
        self.assertEqual(merge_projections([]), [])
        self.assertEqual(merge_projections([[1]]), [1])
        self.assertTrue(np.equal(merge_projections([[10,20,30]]), [10,20,30]).all())
        self.assertTrue(np.equal(merge_projections([[10,None,None],[20,30]]), np.array([10,20,30])).all())
        self.assertTrue(np.equal(merge_projections([[10,None,None],[20,None],[30]]), np.array([10,20,30])).all())

    def test_is_adverb(self):
        a = ["'",':\\',":'",':/','/',':~',':*','\\','\\~','\\*']
        for x in a:
            self.assertTrue(is_adverb(x))
        self.assertFalse(is_adverb(":"))
        self.assertFalse(is_adverb("dog"))

    def test_argsort(self):
        a = [1,3,4,2]
        self.assertTrue(array_equal(kg_argsort(a), [0, 3, 1, 2]))
        self.assertTrue(array_equal(kg_argsort(a,descending=True), [2, 1, 3, 0]))
        a = [[1],[2],[],[3]]
        self.assertTrue(array_equal(kg_argsort(a,descending=True), [3, 1, 0, 2]))


if __name__ == '__main__':
  unittest.main()
