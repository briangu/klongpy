"""
Static Klong: the subset of the language that can be compiled ahead of time
(tickets/design/static-compile.md).

    check_source, check_file, check_main     the profile checker (kgpy --check-static)
    StaticReport, StaticDiagnostic           what it returns
    RULES                                    rule id -> Rule: every rejection the checker can report
    OPTABLE, FORMS                           the operator table and the allowed forms, as data
    parse_signature, Signature, Ty           signatures of .extern and .entry
    validate_global                          is a compile-time value acceptable as a global?
    c_identifier                             the C name of a Klong name
    load_program, call_entry                 run a static program under KlongPy with Python
                                             functions bound as its externs (klongpy.static.host)

The checker never evaluates a program. This package imports nothing beyond
numpy and klongpy itself.
"""
from .checker import (EntryDecl, ExternDecl, FunctionInfo, GlobalInfo, StaticReport, check_file, check_main,
                      check_source, validate_global)
from .diagnostics import StaticDiagnostic
from .host import call_entry, load_program
from .optable import FORMS, OPTABLE
from .rules import RULES
from .signature import Signature, SignatureError, Ty, c_identifier, parse_signature, parse_type

__all__ = [
    "check_source", "check_file", "check_main", "validate_global",
    "StaticReport", "StaticDiagnostic", "FunctionInfo", "GlobalInfo", "ExternDecl", "EntryDecl",
    "RULES", "OPTABLE", "FORMS",
    "parse_signature", "parse_type", "Signature", "SignatureError", "Ty", "c_identifier",
    "load_program", "call_entry",
]
