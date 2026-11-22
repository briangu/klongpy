"""
kgpy lint: a static check for the classic Klong traps.

    lint_source(text, name="<string>") -> [Diagnostic, ...]
    lint_file(path)                    -> [Diagnostic, ...]
    lint_main(files, select, ignore)   -> exit status (the body of ``kgpy lint``; klongpy/tools.py)
    RULES                              code -> Rule(code, severity, title)
    parse_codes(value)                 a set of codes, checked by the engine

The linter is the native engine's (klong_lint, through the extension klong_native: lint_files, lint_source). This module is its
Python face: it reads a program, never runs it, and returns Diagnostic objects.

Output of ``kgpy lint FILE...``: one line per finding,

    file:line:col: CODE message

then a summary line. Exit status 1 if any E finding is reported, 2 if a file
could not be read, else 0. See docs/lint.md for the rule table.

Rules, by code
--------------

    W001  _ between name characters (my_var); not x_a, a_x (Drop with an argument)
    W002  name used before its definition, or never defined
    W003  declared local never used (and not read by a function it calls)
    W010  leading - followed by a number and a dyad (-1+2); not for * and %
    W011  function uses y without x, or z without y, and is called with too few
    W012  bare name in a [...] list that is a defined variable or local
    W013  :[ in the else position of :[ (the :| chain)
    W014  whitespace between a name or } and ( or an adverb
    W015  newline directly after a comment inside a body or conditional
    W016  : directly before a digit
    W017  ; a bracket or an operator inside a [...] data list
    W018  number immediately followed by a name (10c, 2x)
    W019  assignment to x, y or z (that is not declared a local)
    W020  a two-element list joined to a dictionary literal with ,
    W021  a.b::v where a is never a dictionary (a variable named with a dot in Klong 1)
    W022  a module bound by .l and never used
    W023  a local file that shadows a bundled library file of the same name
    E001  anything the parser rejects at strict level 1
    E002  call with more arguments than the callee takes
    E003  assignment to an entry of a module (it is read-only)
    E004  qualified name, or .l entry, that the module does not export
    E005  .extern whose value is discarded

The notes in docs/lint.md say where a rule is narrower than the table
and why. Rules that cannot be decided from the text are reduced to the part
that can: W002 and W012 compare textual order with the order statements run
(top level only, right to left inside a statement), W011 needs a call that
shows the problem, W020 needs a list of two elements written out, and E002
knows only the functions bound once to a literal ``name::{...}``.

Suppression: a comment ``:"lint:ignore W012"`` on a line removes that code
on that line (several codes may follow, separated by spaces or commas).
"""
import os
import re
from dataclasses import dataclass

from .native_ext import native
from .source import Source
from .tools import builtin_names, lint_main, library_dir  # noqa: F401  (lint_main: the body of ``kgpy lint``)

__all__ = ["Rule", "RULES", "Diagnostic", "format_diagnostic", "lint_source", "lint_file", "lint_main", "parse_codes", "parse_names"]


@dataclass(frozen=True)
class Rule:
    code: str
    severity: str       # "warning" or "error"
    title: str


def _rules():
    from .native_ext import native
    return {code: Rule(code, severity, title) for code, severity, title in native().lint_rules()}


# the rules of the linter, from the engine's own table (lint_rules of the extension); the text of docs/lint.md
RULES = _rules()


@dataclass(frozen=True)
class Diagnostic:
    """
    One finding.

        code      "W001" ... "E002"
        severity  "warning" or "error"
        span      (start, end) offsets into source.text
        message   one line, no position
        hint      one line of advice, or None
        source    the klongpy.source.Source the span refers to

    line and col (1-based) are derived from the span.
    """
    code: str
    severity: str
    span: tuple
    message: str
    hint: object
    source: object

    @property
    def line(self):
        return self.source.line_col(self.span[0])[0]

    @property
    def col(self):
        return self.source.line_col(self.span[0])[1]

    def __str__(self):
        return format_diagnostic(self)


def format_diagnostic(d, hint=False):
    """``file:line:col: CODE message`` (and, with hint=True, an indented ``hint:`` line)."""
    line, col = d.source.line_col(d.span[0])
    s = f"{d.source.name}:{line}:{col}: {d.code} {d.message}"
    if hint and d.hint:
        s += f"\n  hint: {d.hint}"
    return s


def parse_codes(value):
    """A comma-separated string (or an iterable) of codes as a set of upper-case codes; ValueError on an unknown code. The
    engine decides what a code is."""
    if value is None:
        return None
    if isinstance(value, str):
        value = [v for v in re.split(r'[,\s]+', value) if v]
    value = list(value)
    native().lint_source("<codes>", "", value, None, None, None, False, None, None)    # raises ValueError for an unknown code
    return {v.strip().upper() for v in value}


def parse_names(value):
    """A comma-separated string (or an iterable) of names as a set of names."""
    if not value:
        return set()
    if isinstance(value, str):
        value = re.split(r'[,\s]+', value)
    return {v for v in value if v}


def lint_source(text, name="<string>", *, select=None, ignore=None, define=None, base_dir=None, resolve=True):
    """
    Lint a program text. Returns the list of Diagnostic, sorted by position.

        text      the program
        name      its name in the findings; also where ``.l("file")`` is looked up
                  (the directory of name, when that is a file)
        select    only these codes (a set, list or comma-separated string)
        ignore    leave these codes out
        define    names the host defines before the program runs (a set, list or
                  comma-separated string): W002 does not report them
        base_dir  directory for ``.l`` lookups, instead of the directory of name
        resolve   False: never read other files; every ``.l`` then counts as one
                  that defines unknown names (W002 off)

    Nothing in the program is evaluated and no other file is run.
    """
    source = Source(name, text)
    found = native().lint_source(name, text, None if select is None else sorted(parse_codes(select)),
                                 None if ignore is None else sorted(parse_codes(ignore)),
                                 None if define is None else sorted(parse_names(define)),
                                 base_dir, resolve, library_dir(), builtin_names())
    return [Diagnostic(code, severity, (a, b), message, hint, source) for code, severity, a, b, _line, _col, message, hint in found]


def lint_file(path, **kw):
    """Lint a file. OSError and UnicodeDecodeError propagate."""
    with open(path, 'r') as f:
        text = f.read()
    return lint_source(text, path, **kw)
