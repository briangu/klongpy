"""
The help text of the operators and adverbs: klongpy/data/operator_help.json, a list of {module, name, doc} in the order the
text was written (monads, dyads, adverbs). The names are those of the evaluator functions the text once documented
(eval_monad_*, eval_dyad_*, eval_adverb_*, eval_dyad_adverb_*); the evaluator is the native engine's, the text is data.
"""
import json
import os

_DOCS = None


def operator_docs():
    """{module: {function name: help text}}, module being monads, dyads or adverbs; the dicts are in file order."""
    global _DOCS
    if _DOCS is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "operator_help.json")
        with open(path, "r", encoding="utf-8") as f:
            docs = {}
            for item in json.load(f):
                docs.setdefault(item["module"], {})[item["name"]] = item["doc"]
        _DOCS = docs
    return _DOCS
