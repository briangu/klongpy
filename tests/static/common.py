"""Paths, stand-ins and the golden-file helper that the tests of the static profile share."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EMBEDDED = os.path.join(ROOT, "examples", "embedded")
GOLDEN = os.path.join(HERE, "golden")
EXPECT = re.compile(r':" expect: static line (\d+) ([a-z0-9-]+) "\n')
STUBS = {"read.adc": lambda ch: 0.5, "log.msg": lambda s, n: None}


def golden(test, name, text):
    """Compare text with tests/static/golden/<name>; KG_UPDATE_GOLDEN=1 rewrites the file."""
    path = os.path.join(GOLDEN, name)
    if os.environ.get("KG_UPDATE_GOLDEN"):
        with open(path, "w") as f:
            f.write(text)
    with open(path) as f:
        want = f.read()
    test.assertEqual(text, want, f"{name} differs from the golden file (KG_UPDATE_GOLDEN=1 rewrites it)")
