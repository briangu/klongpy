"""
Source: a piece of program text and where it came from. No dependency on any engine.

    Source(name, text)        the text being read; line_col(offset) -> (line, col), 1-based
"""
from bisect import bisect_right

__all__ = ['Source']


class Source:
    """
    A piece of program text and where it came from.

    name is a file path or a tag such as "<string>" or "<repl:12>". Spans are
    (start, end) offsets into text; line and column are derived on demand.
    """
    __slots__ = ('name', 'text', '_lines')

    def __init__(self, name, text):
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'text', text)
        object.__setattr__(self, '_lines', None)

    def __setattr__(self, k, v):
        raise AttributeError("Source is immutable")

    def __eq__(self, o):
        return isinstance(o, Source) and self.name == o.name and self.text == o.text

    def __hash__(self):
        return hash((self.name, self.text))

    def __repr__(self):
        return f"Source({self.name!r}, {len(self.text)} chars)"

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (Source, (self.name, self.text))

    def line_starts(self):
        """Offsets at which each line starts (the first is 0)."""
        ls = self._lines
        if ls is None:
            ls = [0]
            find = self.text.find
            i = find('\n')
            while i >= 0:
                ls.append(i + 1)
                i = find('\n', i + 1)
            object.__setattr__(self, '_lines', ls)
        return ls

    def line_col(self, offset):
        """(line, column) of an offset, both 1-based. The offset is clamped to [0, len(text)]."""
        offset = max(0, min(offset, len(self.text)))
        ls = self.line_starts()
        k = bisect_right(ls, offset) - 1
        return k + 1, offset - ls[k] + 1

    def line_text(self, line):
        """The text of a 1-based line, without its newline."""
        ls = self.line_starts()
        if line < 1 or line > len(ls):
            return ""
        s = ls[line - 1]
        e = ls[line] - 1 if line < len(ls) else len(self.text)
        return self.text[s:e]
