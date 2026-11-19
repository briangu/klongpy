"""
Static Klong: the subset of the language that can be compiled ahead of time
(docs/static-profile.md). The checker, the compiler and the emitters are the native engine's
(klong_static, through the extension klong_native); klongpy/static/native.py is their Python API.

    check_source, check_file, check_main     the profile checker (kgpy check)
    StaticReport, StaticDiagnostic           what it returns
    parse_signature, Signature, Ty           signatures of .extern and .entry (klongpy/static/signature.py: the engine's parser)
    compile_program, emit_program            check, evaluate the compile-time definitions, lower; emit C
    emit_hls_program                         the C and the Vitis HLS files around it
    load_program, call_entry, set_param      run a static program with Python functions bound as its externs
    RULES, OPTABLE, FORMS, ...               the profile's tables (klongpy/static/tables.py: the engine's)
    fixed                                    fixed<W,I> arrays (klongpy/static/fixed.py: the host's value type)

The checker never evaluates a program. This package imports nothing beyond numpy and klongpy itself.
"""
import importlib as _importlib

# The names are loaded on first use, so importing klongpy (which needs only
# .signature) does not import the checker, the host or the rest of the package.
_SOURCES = {
    "native": ("FunctionInfo", "GlobalInfo", "StaticReport", "check_file", "check_main",
               "check_source", "check_streamable", "check_report", "ir_json", "validate_global", "call_entry", "load_program", "set_param", "compile_program",
               "emit_program", "emit_hls_program"),
    "diagnostics": ("StaticDiagnostic",),
    # the rule and operator tables are data the Python checker decides by; the engine's own are not exposed (an engine need), so these
    # are the Python ones, until klongpy/static/optable.py and rules.py go
    "tables": ("RULES", "STREAM_RULES", "GROUPS", "OPTABLE", "BY_ID", "FORMS", "OPERAND_KINDS", "TYPE_RULES", "SHAPE_RULES", "LINT_RULES"),
    "signature": ("Signature", "SignatureError", "Ty", "c_identifier", "parse_signature", "parse_type", "EntryDecl", "ExternDecl"),
}
_HOME = {name: module for module, names in _SOURCES.items() for name in names}


def __getattr__(name):
    module = _HOME.get(name)
    if module is None:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    value = getattr(_importlib.import_module("." + module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_HOME))


__all__ = [
    "check_source", "check_file", "check_main", "validate_global", "compile_program", "emit_program", "emit_hls_program",
    "StaticReport", "StaticDiagnostic", "FunctionInfo", "GlobalInfo", "ExternDecl", "EntryDecl",
    "parse_signature", "parse_type", "Signature", "SignatureError", "Ty", "c_identifier", "RULES", "STREAM_RULES", "OPTABLE", "FORMS",
    "check_report", "ir_json",
    "load_program", "call_entry", "set_param",
]
