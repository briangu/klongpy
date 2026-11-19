"""
KlongInterpreter: KlongPy 2 evaluates on the native (Rust) engine (klongpy/rust_engine.py) and needs its compiled extension: a missing
extension is an ImportError that says how to build it, never a silent fallback.
"""
from .errors import KlongException  # noqa: F401  (re-exported: klongpy.interpreter.KlongException is where callers import it from)
from .utils import STRICT_LEVELS, check_strict_level  # noqa: F401
from .rust_engine import KlongInterpreter
