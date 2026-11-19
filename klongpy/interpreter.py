"""
KlongInterpreter: KlongPy 2 evaluates on the native (Rust) engine (klongpy/rust_engine.py) and needs its compiled extension: a missing
extension is an ImportError that says how to build it, never a silent fallback.

KLONGPY_ENGINE=python selects the Python engine (klongpy/py_interpreter.py), the reference implementation until it is removed.
"""
import os

from .errors import KlongException  # noqa: F401  (re-exported: klongpy.interpreter.KlongException is where callers import it from)
from .utils import STRICT_LEVELS, check_strict_level  # noqa: F401

_ENGINE = os.environ.get('KLONGPY_ENGINE', 'rust').strip().lower() or 'rust'
if _ENGINE not in ('rust', 'python'):
    raise ImportError(f"KLONGPY_ENGINE={_ENGINE!r}: it is 'rust' (the default) or 'python'")
if _ENGINE == 'rust':
    from .rust_engine import RustInterpreter as KlongInterpreter
else:
    from .py_interpreter import KlongInterpreter, KlongContext  # noqa: F401


def __getattr__(name):
    # The Python engine's names (PyKlongInterpreter, KlongContext, ...) for the code that is to be deleted with it (the old static
    # tools, the tests of its internals). Nothing in the native engine's path asks for one.
    if name.startswith('__'):
        raise AttributeError(name)
    from . import py_interpreter
    if name == 'PyKlongInterpreter':
        return py_interpreter.KlongInterpreter
    try:
        return getattr(py_interpreter, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
