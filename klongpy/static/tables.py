"""
The rule and operator tables of the static profile, as Python names over the native engine's own (the extension's static_tables
and lint_rules: the source of truth, also what docs/static-profile.md is generated from).

    RULES         id -> Rule(id, group, title, section, hint)      every rule the checker reports, the streamable tier's too
    STREAM_RULES  id -> Rule                                        the streamable tier's alone
    GROUPS        ((id, title), ...)                                the groups of the error table, in order
    OPTABLE       (Op(id, glyph, valence, name, status, operands, type_rule, shape_rule, rule, note, examples), ...)
    BY_ID         id -> Op
    FORMS         (Form(id, form, note, examples), ...)
    OPERAND_KINDS, TYPE_RULES, SHAPE_RULES                          name -> description
    LINT_RULES    ((code, severity, title), ...)                    the rules of `kgpy lint`

An example is Example(body, signature, inputs): ``f::{body}`` entered with ``signature`` is inside the profile with the result type
the signature names; ``inputs`` is JSON text of the arguments a test calls it with, or None. The names are built on first use.
"""
from collections import namedtuple

from ..native_ext import native

Rule = namedtuple("Rule", "id group title section hint")
Op = namedtuple("Op", "id glyph valence name status operands type_rule shape_rule rule note examples")
Form = namedtuple("Form", "id form note examples")
Example = namedtuple("Example", "body signature inputs")

_built = None


def _tables():
    global _built
    if _built is None:
        t = native().static_tables()
        ex = lambda rows: tuple(Example(*e) for e in rows)
        rules = {r[0]: Rule(*r) for r in t["rules"]}
        stream = {r[0]: Rule(*r) for r in t["stream_rules"]}
        ops = tuple(Op(o[0], o[1], o[2], o[3], "allowed" if o[4] is True else "rejected" if o[4] is False else o[4], tuple(o[5]), o[6], o[7], o[8], o[9], ex(o[10])) for o in t["optable"])
        _built = {
            "RULES": rules,
            "STREAM_RULES": stream,
            "GROUPS": tuple(tuple(g) for g in t["groups"]),
            "OPTABLE": ops,
            "BY_ID": {o.id: o for o in ops},
            "FORMS": tuple(Form(f[0], f[1], f[2], ex(f[3])) for f in t["forms"]),
            "OPERAND_KINDS": dict(t["operand_kinds"]),
            "TYPE_RULES": dict(t["type_rules"]),
            "SHAPE_RULES": dict(t["shape_rules"]),
            "LINT_RULES": tuple(tuple(r) for r in native().lint_rules()),
        }
    return _built


_NAMES = ("RULES", "STREAM_RULES", "GROUPS", "OPTABLE", "BY_ID", "FORMS", "OPERAND_KINDS", "TYPE_RULES", "SHAPE_RULES", "LINT_RULES")
__all__ = list(_NAMES) + ["Rule", "Op", "Form", "Example"]


def __getattr__(name):
    if name in _NAMES:
        return _tables()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
