"""
Diagnostics of the static profile checker.

A StaticDiagnostic is one profile violation: the rule id (klongpy.static.tables.RULES),
the span and Source it points at, a one-line message and an optional hint.
``d.error()`` is the same thing as a KlongStaticError (kind "static"), whose
``render()`` prints it like every other KlongPy error:

    prog.kg:3:5: static error: &a (Expand/Where) has a result whose length depends on the data [where]
      f::{&x}
          ^^
      hint: ...

The rule id is the bracketed word at the end of the message and the ``rule``
attribute of the error.
"""
from ..errors import KlongStaticError


class StaticDiagnostic:
    __slots__ = ("rule", "span", "source", "message", "hint", "function")

    def __init__(self, rule, span, source, message, hint=None, function=None):
        self.rule = rule
        self.span = span
        self.source = source
        self.message = message
        self.hint = hint
        self.function = function    # name of the top-level function the span lies in, or None

    @property
    def line(self):
        return self.source.line_col(self.span[0])[0]

    @property
    def col(self):
        return self.source.line_col(self.span[0])[1]

    def error(self):
        """The diagnostic as a KlongStaticError (with a ``rule`` attribute)."""
        e = KlongStaticError(f"{self.message} [{self.rule}]", span=self.span, source=self.source, hint=self.hint)
        e.rule = self.rule
        return e

    def render(self, color=False):
        return self.error().render(color)

    def __str__(self):
        return f"{self.source.name}:{self.line}:{self.col}: static error: {self.message} [{self.rule}]"

    def __repr__(self):
        return f"<StaticDiagnostic {self}>"
