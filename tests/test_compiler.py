import importlib.util
import unittest


needs_torch = unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires torch")


if __name__ == '__main__':
    unittest.main()
