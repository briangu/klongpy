import unittest
from klongpy.utils import ReadonlyDict, CallbackEvent


class TestReadonlyDict(unittest.TestCase):
    def test_getitem(self):
        data = {'a': 1, 'b': 2}
        rd = ReadonlyDict(data)
        self.assertEqual(rd['a'], 1)
        self.assertEqual(rd['b'], 2)

    def test_len(self):
        data = {'a': 1, 'b': 2, 'c': 3}
        rd = ReadonlyDict(data)
        self.assertEqual(len(rd), 3)

    def test_iter(self):
        data = {'a': 1, 'b': 2}
        rd = ReadonlyDict(data)
        keys = list(rd)
        self.assertEqual(set(keys), {'a', 'b'})

    def test_empty(self):
        rd = ReadonlyDict({})
        self.assertEqual(len(rd), 0)
        self.assertEqual(list(rd), [])


class TestCallbackEvent(unittest.TestCase):
    def test_subscribe(self):
        event = CallbackEvent()
        callback = lambda: None
        event.subscribe(callback)
        self.assertIn(callback, event.subscribers)

    def test_unsubscribe(self):
        event = CallbackEvent()
        callback = lambda: None
        event.subscribe(callback)
        self.assertIn(callback, event.subscribers)
        event.unsubscribe(callback)
        self.assertNotIn(callback, event.subscribers)

    def test_unsubscribe_not_found(self):
        event = CallbackEvent()
        callback = lambda: None
        # Should not raise even if callback not subscribed
        event.unsubscribe(callback)
        self.assertNotIn(callback, event.subscribers)

    def test_trigger(self):
        event = CallbackEvent()
        results = []

        def callback1():
            results.append(1)

        def callback2():
            results.append(2)

        event.subscribe(callback1)
        event.subscribe(callback2)
        event.trigger()

        self.assertEqual(set(results), {1, 2})

    def test_trigger_empty(self):
        event = CallbackEvent()
        # Should not raise when no subscribers
        event.trigger()

    def test_multiple_subscribe(self):
        event = CallbackEvent()
        callback = lambda: None
        event.subscribe(callback)
        event.subscribe(callback)  # Adding same callback again
        # Set should only contain one instance
        self.assertEqual(len(event.subscribers), 1)


if __name__ == '__main__':
    unittest.main()
