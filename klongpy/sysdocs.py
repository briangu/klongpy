"""
The help text of the system functions the native engine implements (.p .d .l .extern ...): data, not code
(klongpy/data/system_function_help.json, name -> the text, in the form of the docstring of an ``eval_sys_*`` function:
a header line ``.name(x)   [Title]``, then the description). The functions that are Python's (.py .srv .fpga ...) keep
theirs as the docstring of the function; the help index and the generated system-function page read both.
"""
import json
import os

__all__ = ["engine_docs"]

_docs = None


def engine_docs():
    """{system function name: help text} of the functions the engine implements."""
    global _docs
    if _docs is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "system_function_help.json")
        with open(path, encoding="utf-8") as f:
            _docs = json.load(f)
    return _docs
