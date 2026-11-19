"""
Help index for the REPL. The text of the operators and adverbs is data
(klongpy/data/operator_help.json); that of the system functions and variables
is the docstring of their ``eval_sys_*`` and ``eval_sys_var_*`` functions.
Each whose text starts with a header line of the form ``name   [Title]``
becomes one entry. A few forms that have no function
(conditional, evaluated list, dictionary literal, comment) are described in a
small hand-written table.
"""

import inspect
import re
import textwrap

from .operator_help import operator_docs

CATEGORIES = [
    ("operators", "Operators"),
    ("adverbs", "Adverbs"),
    ("system functions", "System functions"),
    ("variables", "System variables"),
    ("extensions", "KlongPy extensions"),
    ("forms", "Language forms"),
]

_EXTENSION_GLYPHS = {"∇", "∂", "˙", ":>"}
_EXTENSION_NAMES = {".strict", ".unsafe", ".py", ".pya", ".pyc", ".pyf",
                    ".bkd", ".bkf", ".bkp", ".module"}

_HEADER = re.compile(r"^(?P<head>.*?)\s{2,}\[(?P<title>[^\]]+)\]\s*$")
_ARG_LETTER = re.compile(r"(?<![A-Za-z0-9_])[abfxyz](?![A-Za-z0-9_])")

# (header, title, keys, text)
_FORMS = [
    (":[c;a;b]", "Conditional", [":[", "conditional", "cond", "if", ":|"],
     "If c is true, evaluate a, otherwise evaluate b. A number equal to 0 and\n"
     "any empty value ([] or \"\") are false; everything else is true.\n\n"
     "Else-if chain: :[c1;a:|c2;b;d] is short for :[c1;a;:[c2;b;d]].\n\n"
     "Examples:  :[0;1:|1;2;3]  -->  2\n"
     "           :[0;1:|0;2;3]  -->  3"),
    ("[a;b;...]", "Evaluated-List", ["[;", "[;...]", "evaluated list", "list"],
     "A list written with ';' separators ([a;b;c]) evaluates each element and\n"
     "returns the list of results. A list written with spaces ([1 2 3]) holds\n"
     "its elements as written.\n\n"
     "Example:  [1+1;2*3]  -->  [2 6]"),
    ("{a,b}", "Dictionary", ["dictionary", "dict", ":{"],
     "A dictionary is written :{[k1 v1] [k2 v2]}. Use a:=... style amendment\n"
     "or a,[k v] to add entries and d@k to look one up.\n\n"
     "Example:  :{[1 2] [3 4]}@3  -->  4"),
    (":\"text\"", "Comment", ["comment", ":\""],
     "A comment is a colon followed by a string: :\"this is ignored\". It can\n"
     "stand alone or follow an expression on the same line."),
]


class HelpEntry:
    def __init__(self, category, header, title, text, keys):
        self.category = category
        self.header = header
        self.title = title
        self.text = text
        self.keys = keys

    def summary(self):
        return f"{self.header:<20}[{self.title}]"

    def full(self):
        return f"{self.summary()}\n\n{self.text}"


def _glyph(header):
    h = header.strip()
    m = re.match(r"^\.\w+", h)
    if m:
        return m.group(0)
    return _ARG_LETTER.sub("", h).strip()


_CONTROL = {"\x07": "\\a", "\x08": "\\b"}
_CONTROL.update({chr(i): "\\%d" % i for i in range(1, 7)})


def _unescape(doc):
    """Restore backslashes that Python read as escapes (\\a, \\b, \\1) in a plain docstring."""
    return "".join(_CONTROL.get(c, c) for c in doc)


def _docs():
    """(category, function name, docstring) of every operator, adverb, system function and variable that has help, in the
    order of the function names within each group. The operators' and adverbs' text is data (klongpy/data/operator_help.json,
    read by operator_docs); the system functions' and variables' is the docstring of their eval_* function."""
    out = []
    operator_text = operator_docs()
    for category, module, prefix in (("operators", "monads", "eval_monad_"), ("operators", "dyads", "eval_dyad_"),
                                     ("adverbs", "adverbs", "eval_adverb_")):
        out.extend((category, name, doc) for name, doc in sorted(operator_text.get(module, {}).items())
                   if name.startswith(prefix))
    from . import sys_fn, sys_var
    for category, mod, prefix in (("system functions", sys_fn, "eval_sys_"), ("variables", sys_var, "eval_sys_var_")):
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if name.startswith(prefix) and fn.__module__ == mod.__name__:
                out.append((category, name, fn.__doc__))
    return out


def build_entries():
    """Build the list of HelpEntry objects from the help text and the forms table."""
    entries = []
    seen = set()
    for category, name, doc in _docs():
        if category == "system functions" and name.startswith("eval_sys_var_"):
            continue
        if not doc:
            continue
        # a few docstrings contain an unescaped backslash that Python reads as an escape
        doc = _unescape(doc)
        lines = textwrap.dedent(doc).strip("\n").splitlines()
        first = next((ln for ln in lines if ln.strip()), "")
        m = _HEADER.match(first.strip())
        if not m:
            continue
        header, title = m.group("head").strip(), m.group("title")
        glyph = _glyph(header)
        body = "\n".join(lines[lines.index(first) + 1:]).strip("\n")
        body = textwrap.indent(textwrap.dedent(body), "    ") if body else ""
        if glyph in _EXTENSION_GLYPHS or glyph in _EXTENSION_NAMES:
            cat = "extensions"
        else:
            cat = category
        keys = [glyph, title.lower()]
        if glyph.startswith("."):
            keys.append(glyph[1:])
        sig = (header, title)
        if sig in seen:
            continue
        seen.add(sig)
        entries.append(HelpEntry(cat, header, title, body, keys))
    for header, title, keys, text in _FORMS:
        entries.append(HelpEntry("forms", header, title,
                                 textwrap.indent(text, "    "),
                                 [k.lower() for k in keys] + [title.lower()]))
    return entries


_cache = None


def entries():
    global _cache
    if _cache is None:
        _cache = build_entries()
    return _cache


def index_text():
    out = []
    for cat, label in CATEGORIES:
        rows = [e for e in entries() if e.category == cat]
        if not rows:
            continue
        out.append(f"{label}:")
        out.extend(f"  {e.summary()}" for e in rows)
        out.append("")
    out.append("]h TOPIC shows an entry by glyph (]h /), name (]h Over, ]h .p) or part of a name.")
    out.append("]h all shows every entry; ]h adverbs (or operators, variables, extensions, forms) shows one group.")
    out.append("]a TEXT lists the entries whose name or title contains TEXT.")
    return "\n".join(out)


_CATEGORY_WORDS = {
    "operators": "operators", "adverbs": "adverbs",
    "system": "system functions", "functions": "system functions",
    "variables": "variables", "extensions": "extensions", "forms": "forms",
}


def _by_category_word(topic):
    return _CATEGORY_WORDS.get(topic.lower())


def help_text(topic):
    """Return the help text for topic (never raises for any string)."""
    topic = topic.strip()
    if not topic:
        return index_text()
    low = topic.lower()
    if low == "all":
        return "\n\n".join(e.full() for e in entries())
    cat = _by_category_word(topic)
    if cat:
        return "\n".join(e.summary() for e in entries() if e.category == cat)
    exact = [e for e in entries() if topic in e.keys or low in e.keys]
    if exact:
        return "\n\n".join(e.full() for e in exact)
    part = _search(low)
    if len(part) == 1:
        return part[0].full()
    if part:
        return "\n".join(e.summary() for e in part)
    return f"no help for '{topic}'; ]h lists all topics"


def _search(low):
    return [e for e in entries()
            if low in e.title.lower() or low in e.header.lower()
            or any(low in k for k in e.keys if len(k) > 1)]


def apropos_text(topic):
    topic = topic.strip()
    if not topic:
        return "usage: ]a topic"
    found = _search(topic.lower())
    if not found:
        return f"nothing found for '{topic}'"
    return "\n".join(e.summary() for e in found)
