"""
The canonical parse tree of a program (conformance/README.md), as the native engine prints it: `kgpy ast` and the golden .sexp
files. `dump_text(text)` is the text, or `(error ClassName)` for a program that does not parse.
"""
from klongpy.native_ext import native


def dump_text(text, name="<string>"):
    ext = native()
    try:
        return ext.ast_text(text, name)
    except ext.NativeError as e:
        from klongpy.rust_engine import _error
        return f"(error {type(_error(e)).__name__})\n"
