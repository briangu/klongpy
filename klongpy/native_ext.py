"""
The compiled extension of the native (Rust) engine (native/klong-py), `klongpy._native` in a wheel or an editable install, `klong_native` when built by hand: the interpreter and the tools (lint, the static
checker, the C and FPGA-source compiler, the tree dump, REPL input and IPC function text). Importing it is an ImportError that says how
to build it when it is missing; there is no fallback.
"""
import os
import sys

_module = None


def load_extension():
    """The compiled extension: KLONGPY_NATIVE_PATH (a directory that holds it) when set, else the module `klongpy._native`, else
    `klong_native` on the import path."""
    ext = os.environ.get("KLONGPY_NATIVE_PATH")
    if ext:
        sys.path.insert(0, ext)
    try:
        if not ext:
            try:
                from . import _native as mod
                return mod
            except ImportError:
                pass
        import klong_native as mod
        return mod
    except ImportError as e:
        raise ImportError(
            "KlongPy 2 evaluates on the native (Rust) engine and needs its compiled extension, which was not found "
            "(neither klongpy._native nor klong_native is importable, and KLONGPY_NATIVE_PATH is "
            + (f"set to {ext!r}, which does not hold it" if ext else "not set")
            + "). An installed wheel carries it as klongpy._native: reinstall with `pip install --force-reinstall klongpy`. "
            "From a source tree, `pip install -e .` builds it (needs a Rust toolchain, https://rustup.rs); or build it by hand with "
            "`cd native/klong-py && cargo build --release`, copy target/release/libklong_native.dylib (Linux: .so, Windows: .dll) "
            "to a directory as klong_native.abi3.so (Windows: klong_native.pyd), and put that directory on KLONGPY_NATIVE_PATH.") from e
    finally:
        if ext:
            try:
                sys.path.remove(ext)
            except ValueError:
                pass


def native():
    """The extension module, loaded once."""
    global _module
    if _module is None:
        _module = load_extension()
    return _module
