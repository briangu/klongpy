"""
Static Klong: the subset of the language that can be compiled ahead of time
(docs/static-profile.md).

    check_source, check_file, check_main     the profile checker (kgpy check)
    StaticReport, StaticDiagnostic           what it returns
    RULES                                    rule id -> Rule: every rejection the checker can report
    OPTABLE, FORMS                           the operator table and the allowed forms, as data
    parse_signature, Signature, Ty           signatures of .extern and .entry
    validate_global                          is a compile-time value acceptable as a global?
    c_identifier                             the C name of a Klong name
    load_program, call_entry                 run a static program under KlongPy with Python
                                             functions bound as its externs (klongpy.static.host)
    set_param                                set a .param register between runs
    fixed, eval_ir, streamable               fixed<W,I> semantics, the evaluator behind call_entry for programs that use
                                             them, and kgpy check --streamable

The checker never evaluates a program. This package imports nothing beyond
numpy and klongpy itself.
"""
import importlib as _importlib

# The names are loaded on first use, so importing klongpy (which needs only
# .signature) does not import the checker, the host or the rest of the package.
_SOURCES = {
    "checker": ("EntryDecl", "ExternDecl", "FunctionInfo", "GlobalInfo", "StaticReport", "check_file", "check_main",
                "check_source", "validate_global"),
    "diagnostics": ("StaticDiagnostic",),
    "host": ("call_entry", "load_program", "set_param"),
    "optable": ("FORMS", "OPTABLE"),
    "rules": ("RULES",),
    "signature": ("Signature", "SignatureError", "Ty", "c_identifier", "parse_signature", "parse_type"),
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
    "check_source", "check_file", "check_main", "validate_global",
    "StaticReport", "StaticDiagnostic", "FunctionInfo", "GlobalInfo", "ExternDecl", "EntryDecl",
    "RULES", "OPTABLE", "FORMS",
    "parse_signature", "parse_type", "Signature", "SignatureError", "Ty", "c_identifier",
    "load_program", "call_entry", "set_param",
]
