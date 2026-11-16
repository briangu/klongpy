"""
KlongPy lexer: source text to tokens with source spans.

    Source(name, text)        the text being read; line_col(offset) -> (line, col), 1-based
    Tok                       one token: kind, [start, end) offsets, decoded value, flags
    Lexer(source, pos=0)      pull-based token stream: next(), peek(k), seek(offset), trivia
    tokenize(source)          the whole token list (skips .comment("...") bodies itself)

The lexer is context free and never raises. Whatever the legacy character
readers in klongpy/parser.py rejected with an exception is an ERROR token
(a malformed number, ``0c`` at end of input) or an ``unterminated`` flag (a
string or a comment that runs to end of input). Everything that depends on
context is left to the parser: ``-`` is always an operator (a negative number
exists only inside a list), a newline is always a NEWLINE token (it is a
statement separator only at top level), adjacency is the ``ws`` flag.

Token kinds

    NUM       digits with ./e/e+/e- runs, optionally a leading ':'    int or float
    CHAR      0c plus one character, optionally a leading ':'         KGChar
    STR       "..." with "" for a quote                               str
    NAME      run of letters, digits and '.', not starting with a digit   the text
    QNAME     ':' then a name                                         the text without ':'
    OP        one character, or \\~ or \\*                              the text
    COP       ':' plus one character (any not listed elsewhere)       the text
    LBRACK RBRACK LBRACE RBRACE LPAREN RPAREN SEMI                    the text
    COND ':['   ELIF ':|'   DICT ':{'   CALLP ':('                    the text
    NEWLINE   one '\\n'                                                '\\n'
    ERROR     a malformed number or 0c at end of input                the text; .message
    EOF       end of input (start == end == len(text))
    COMMENT   :"..." (trivia only, never in the token stream)         the text

Whitespace is whatever str.isspace() accepts, except '\\n'. Letters and digits
are str.isalpha()/isdigit()/isnumeric() as in the legacy readers, so Unicode
letters are name characters and any other glyph is a one-character OP.
"""
import re
from bisect import bisect_right
from typing import NamedTuple

from .types import KGChar

__all__ = ['Source', 'Tok', 'Lexer', 'tokenize', 'ERR_CHAR_EOF']


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


class Tok(NamedTuple):
    kind: str
    start: int
    end: int
    value: object           # decoded value for literals, text for everything else
    ws: bool = False        # whitespace or a comment between the previous token and this one
    nl_comment: bool = False    # NEWLINE only: a :"..." comment sits between the previous token and this newline
    unterminated: bool = False  # STR and COMMENT: end of input came before the closing quote
    message: object = None      # ERROR only


ERR_CHAR_EOF = "character literal 0c at end of input"

_PUNCT = {';': 'SEMI', '(': 'LPAREN', ')': 'RPAREN', '{': 'LBRACE', '}': 'RBRACE',
          '[': 'LBRACK', ']': 'RBRACK'}
_COLON = {':[': 'COND', ':{': 'DICT', ':|': 'ELIF', ':(': 'CALLP'}

# One token, after any run of whitespace other than a newline. The order of the
# alternatives is the order of the tests in the legacy kg_read. Every
# alternative starts with a character that is not whitespace, so the pattern
# fails only at end of input. No alternative contains a nested capturing group
# (lastgroup must name the alternative).
_MASTER = re.compile(
    r'[^\S\n]*(?:'
    r'(?P<NAME>[A-Za-z.][A-Za-z0-9.]*)'
    r'|(?P<NL>\n)'
    r'|(?P<P>[;(){}\[\]])'
    r'|(?P<CH>:?0c[\s\S]?)'
    r'|(?P<NUM>:?[0-9](?:[0-9.]+|e[+\-][\s\S]?|e)*)'
    r'|(?P<STR>"(?:[^"]+|"")*"?)'
    r'|(?P<COM>:"(?:[^"]+|"")*"?)'
    r'|(?P<QN>:[A-Za-z.][A-Za-z0-9.]*)'
    r'|(?P<CX>:[\[{|(])'
    r'|(?P<COP>:[\s\S])'
    r'|(?P<OP>\\[~*]|\S)'
    r')'
)


def _is_symbolic(c):
    return c.isalpha() or c.isdigit() or c == '.'


def _num_end(text, i, n):
    """Continue the legacy read_num loop from i (used when a non-ASCII character follows)."""
    while i < n:
        c = text[i]
        if c == '.':
            pass
        elif c == 'e':
            if text[i+1:i+2] in ('-', '+'):
                i += 2
        elif not c.isnumeric():
            break
        i += 1
    return i if i < n else n


def _lex(text, pos, out, trivia, one=False):
    """
    Append the tokens of text[pos:] to out (ending with EOF) and its comments
    to trivia. With one=True stop after the first token.
    """
    n = len(text)
    match = _MASTER.match
    new = tuple.__new__
    T = Tok
    append = out.append
    gap = pos       # where the whitespace before the next token starts
    com = False     # a comment was seen in that whitespace
    while True:
        m = match(text, pos)
        if m is None:
            append(new(T, ('EOF', n, n, '', n > gap, False, False, None)))
            return
        g = m.lastgroup
        s = m.start(g)
        e = m.end()
        if g == 'NAME':
            if e < n and text[e] > '\x7f':
                while e < n and _is_symbolic(text[e]):
                    e += 1
            tok = new(T, ('NAME', s, e, text[s:e], s > gap, False, False, None))
        elif g == 'OP':
            c = text[s]
            if c > '\x7f' and (c.isnumeric() or c.isalpha()):
                # a non-ASCII digit or letter: the legacy readers test isnumeric() first
                if c.isnumeric():
                    e = _num_end(text, s, n)
                    tok = _num_tok(text, s, e, s, s > gap)
                else:
                    while e < n and _is_symbolic(text[e]):
                        e += 1
                    tok = new(T, ('NAME', s, e, text[s:e], s > gap, False, False, None))
            else:
                tok = new(T, ('OP', s, e, text[s:e], s > gap, False, False, None))
        elif g == 'NUM':
            if e < n and text[e] > '\x7f':
                e = _num_end(text, e, n)
            d = text[s+1:e] if text[s] == ':' else text[s:e]
            try:
                v = float(d) if ('.' in d or 'e' in d) else int(d)
                tok = new(T, ('NUM', s, e, v, s > gap, False, False, None))
            except Exception as ex:
                tok = new(T, ('ERROR', s, e, d, s > gap, False, False, str(ex)))
        elif g == 'P':
            c = text[s]
            tok = new(T, (_PUNCT[c], s, e, c, s > gap, False, False, None))
        elif g == 'NL':
            tok = new(T, ('NEWLINE', s, e, '\n', s > gap, com, False, None))
        elif g == 'STR':
            body = text[s+1:e]
            q = body.count('"')
            if q & 1:
                body = body[:-1]
            if q > 1:
                body = body.replace('""', '"')
            tok = new(T, ('STR', s, e, body, s > gap, False, not (q & 1), None))
        elif g == 'COM':
            trivia.append(new(T, ('COMMENT', s, e, text[s:e], False, False,
                                  not (text.count('"', s + 2, e) & 1), None)))
            com = True
            pos = e
            continue
        elif g == 'COP':
            c = text[s+1]
            if c > '\x7f' and (c.isalpha() or c.isnumeric()):
                if c.isalpha():
                    e = s + 2
                    while e < n and _is_symbolic(text[e]):
                        e += 1
                    tok = new(T, ('QNAME', s, e, text[s+1:e], s > gap, False, False, None))
                else:
                    e = _num_end(text, s + 1, n)
                    tok = _num_tok(text, s, e, s + 1, s > gap)
            else:
                tok = new(T, ('COP', s, e, text[s:e], s > gap, False, False, None))
        elif g == 'QN':
            if e < n and text[e] > '\x7f':
                while e < n and _is_symbolic(text[e]):
                    e += 1
            tok = new(T, ('QNAME', s, e, text[s+1:e], s > gap, False, False, None))
        elif g == 'CX':
            v = text[s:e]
            tok = new(T, (_COLON[v], s, e, v, s > gap, False, False, None))
        else:  # CH
            d = text[s+1:e] if text[s] == ':' else text[s:e]
            if len(d) == 3:
                tok = new(T, ('CHAR', s, e, KGChar(d[2]), s > gap, False, False, None))
            else:
                tok = new(T, ('ERROR', s, e, d, s > gap, False, False, ERR_CHAR_EOF))
        append(tok)
        if one:
            return
        pos = gap = e
        com = False


def _num_tok(text, s, e, ds, ws):
    """A NUM or ERROR token for text[s:e] whose digits start at ds."""
    d = text[ds:e]
    try:
        v = float(d) if ('.' in d or 'e' in d) else int(d)
        return Tok('NUM', s, e, v, ws)
    except Exception as ex:
        return Tok('ERROR', s, e, d, ws, False, False, str(ex))


class Lexer:
    """
    Pull-based token stream over a Source.

        next()        consume and return the next token (EOF is returned forever)
        peek(k=0)     the k-th token ahead, without consuming
        seek(offset)  drop the lookahead and continue from a character offset
        trivia        the comments seen so far (kind COMMENT), in source order

    seek exists because the parser, not the lexer, knows that ``.comment("END")``
    hides raw text up to the marker: text the lexer cannot be expected to
    tokenise sensibly.

    The parser reads ``toks`` and ``pos`` directly: toks holds every token from
    the current position to EOF (always the last element), pos is the index of
    the next token. seek keeps the list object and edits it in place.
    """
    __slots__ = ('source', 'text', 'toks', 'pos', 'trivia', 'start', 'at_end', '_base')

    def __init__(self, source, pos=0):
        if isinstance(source, str):
            source = Source("<string>", source)
        self.source = source
        self.text = source.text
        self.toks = []
        self.trivia = []
        self.pos = 0
        self.start = pos        # parser scratch: start offset of the expression just parsed
        self.at_end = False     # parser scratch: a read ran into end of input
        self._base = (0, pos)   # (token index, offset): where lexing last (re)started
        _lex(self.text, pos, self.toks, self.trivia)

    def next(self):
        tok = self.toks[self.pos]
        if tok[0] != 'EOF':
            self.pos += 1
        return tok

    def peek(self, k=0):
        toks = self.toks
        j = self.pos + k
        return toks[j] if j < len(toks) else toks[-1]

    def offset(self):
        """
        The character offset of the read position: the end of the last token
        consumed (the whitespace before the next token is not consumed), or
        len(text) once a read has run into end of input.
        """
        p = self.pos
        if self.at_end and p == len(self.toks) - 1:
            return len(self.text)
        if p == self._base[0]:
            return self._base[1]
        return self.toks[p - 1][2]

    def end(self):
        """The end offset of the last token consumed."""
        p = self.pos
        return self.toks[p - 1][2] if p else self._base[1]

    def seek(self, offset):
        """Drop the lookahead and continue lexing from offset."""
        text = self.text
        offset = max(0, min(offset, len(text)))
        toks = self.toks
        p = self.pos
        lo = self.offset()
        first = []
        triv = []
        _lex(text, offset, first, triv, one=True)
        first = first[0]
        s = first[1]
        # Lexing is a function of the position alone, so if the tokens already
        # buffered include one that starts where the first token after offset
        # starts, everything after it is what lexing from offset would give.
        a, b = p, len(toks)
        while a < b:
            mid = (a + b) // 2
            if toks[mid][1] < s:
                a = mid + 1
            else:
                b = mid
        if a < len(toks) and toks[a][1] == s and toks[a][2] == first[2] and toks[a][0] == first[0]:
            toks[p:a+1] = [first]
            self.trivia = [c for c in self.trivia if c[1] < lo or c[1] >= s] + triv
            self.trivia.sort(key=_start)
        else:
            del toks[p:]
            self.trivia = [c for c in self.trivia if c[1] < lo]
            _lex(text, offset, toks, self.trivia)
        self._base = (p, offset)
        self.at_end = False


def _start(tok):
    return tok[1]


def _comment_end(text, i, marker):
    """End offset of a .comment body starting at i, or -1 (the legacy read_sys_comment rule)."""
    j = text.find(marker, i)
    if j < 0 or not marker:
        return -1
    while text.startswith(marker, j + 1):
        j += 1
    return j + len(marker)


def tokenize(source):
    """
    Every token of a source, ending with EOF. Never raises.

    ``.comment("MARK")`` hides the text up to and including the marker from the
    token stream, as it does when a program is parsed. If the marker is not
    found the rest of the text is tokenised as it stands.
    """
    lx = Lexer(source)
    out = []
    text = lx.text
    while True:
        tok = lx.next()
        out.append(tok)
        k = tok[0]
        if k == 'EOF':
            return out
        if k == 'RPAREN' and not tok[4] and len(out) >= 4:
            a, b, c = out[-4], out[-3], out[-2]
            if (a[0] in ('NAME', 'QNAME') and a[3] == '.comment' and b[0] in ('LPAREN', 'CALLP') and not b[4]
                    and c[0] == 'STR' and not c[6]):
                end = _comment_end(text, tok[2], c[3])
                if end >= 0:
                    lx.seek(end)
