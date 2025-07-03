try:
    from klongpy_rs import *  # compiled fast-path
except ImportError:
    from .numpy_backend import *  # pure-Python fallback

from .interpreter import KlongInterpreter, KlongException
__all__ = [
    "KlongInterpreter",
    "KlongException",
    "add",
    "subtract",
    "multiply",
    "divide",
    "map",
    "set_dtype",
    "get_dtype",
    "to_pandas",
    "PyArrayF64",
]
