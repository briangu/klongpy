"""
REPL input helpers that need no terminal: the completeness check for multi-line
entry, the tab completer and the persistent history.

    check_complete(tokens, source=None) -> "complete" | "incomplete" | "error"
    Completer(klong, sys_cmds)         .matches(line) -> (start, [words]); callable as a readline completer
    History(readline_module, path)     load() / save() / replace_entry(mark, text)
"""
import glob
import os
import re

from .lexer import Lexer, Source, _comment_end, tokenize

__all__ = ['check_complete', 'check_text', 'Completer', 'History', 'history_path', 'HISTORY_LIMIT']

_OPENERS = {'LPAREN': ')', 'CALLP': ')', 'LBRACK': ']', 'COND': ']', 'LBRACE': '}', 'DICT': '}'}
_CLOSERS = {'RPAREN': ')', 'RBRACK': ']', 'RBRACE': '}'}


def _unterminated_comment(tokens, text):
    """True when the text between the last token and the end holds a :" comment with no closing quote."""
    last = tokens[-1] if tokens else None
    start = 0
    for t in reversed(tokens):
        if t[0] != 'EOF':
            start = t[2]
            break
    tail = Source("<tail>", text[start:])
    lx = Lexer(tail)
    while lx.next()[0] != 'EOF':
        pass
    return any(c[0] == 'COMMENT' and c[6] for c in lx.trivia)


def _open_comment_marker(tokens, text):
    """True when a .comment("X") call has no end marker X after it."""
    for i in range(len(tokens) - 3):
        a, b, c, d = tokens[i:i + 4]
        if (a[0] in ('NAME', 'QNAME') and a[3] == '.comment' and b[0] in ('LPAREN', 'CALLP') and not b[4]
                and c[0] == 'STR' and not c[6] and d[0] == 'RPAREN' and not d[4]
                and c[3] and _comment_end(text, d[2], c[3]) < 0):
            return True
    return False


def check_complete(tokens, source=None):
    """
    Decide whether the tokens of a REPL entry form a whole entry.

    "incomplete": an opener (( [ { :[ :{ :() is still open at EOF, the last token
    is an unterminated string, or (when source is given) the text ends in an
    unterminated :" comment or a .comment("X") without its end marker.
    "error": a closer does not match the innermost opener, or closes nothing.
    "complete": anything else; a trailing dyad (1+) is complete.

    source is the Source or the text the tokens came from. The token list does
    not carry comments (they are trivia) nor the text a .comment("X") would
    hide, so those two cases need the text; without it they read as complete.
    """
    stack = []
    for t in tokens:
        k = t[0]
        if k in _OPENERS:
            stack.append(_OPENERS[k])
        elif k in _CLOSERS:
            if not stack or stack.pop() != _CLOSERS[k]:
                return "error"
    if stack:
        return "incomplete"
    real = [t for t in tokens if t[0] != 'EOF']
    if real and real[-1][0] == 'STR' and real[-1][6]:
        return "incomplete"
    if source is not None:
        text = source.text if isinstance(source, Source) else source
        if _unterminated_comment(tokens, text) or _open_comment_marker(tokens, text):
            return "incomplete"
    return "complete"


def check_text(text, name="<repl>"):
    """Tokenise text and run check_complete on it."""
    src = Source(name, text)
    return check_complete(tokenize(src), src)


# ---------------------------------------------------------------- completion

_NAME_TAIL = re.compile(r'[A-Za-z0-9.]*$')
_LOAD_STR = re.compile(r'\.l\("([^"]*)$')
_PATH_ARG = re.compile(r'^\s*\]l\s+(.*)$')


def _kg_files(prefix):
    """File names completing prefix: entries of the directory it names, plus .kg files on KLONGPATH."""
    out = set()
    for p in glob.glob(glob.escape(prefix) + "*"):
        out.add(p + "/" if os.path.isdir(p) else p)
    if os.sep not in prefix:
        for d in (os.environ.get("KLONGPATH") or "").split(":"):
            if d and os.path.isdir(d):
                for f in os.listdir(d):
                    if f.startswith(prefix) and f.endswith(".kg"):
                        out.add(f)
    return sorted(out)


class Completer:
    """
    Tab completion of defined names (including system functions), ]-commands
    and file names. Read-only: nothing is evaluated.

    matches(line) takes the text before the cursor and returns (start, words):
    each word replaces line[start:].
    """

    def __init__(self, klong, sys_cmds=None):
        self.klong = klong
        self.sys_cmds = sys_cmds or {}
        self._state = []

    def names(self):
        try:
            names = getattr(self.klong, "names", None)
            if names is not None:       # the native engine lists its globals itself
                return [str(n) for n in names()]
            return [str(k) for k, _ in self.klong._context]
        except Exception:
            return []

    def matches(self, line):
        m = _LOAD_STR.search(line)
        if m:
            return m.start(1), _kg_files(m.group(1))
        s = line.lstrip()
        if s.startswith("]"):
            off = len(line) - len(s)
            if not any(c.isspace() for c in s):
                return off, sorted("]" + k for k in self.sys_cmds if ("]" + k).startswith(s))
            m = _PATH_ARG.match(line)
            if m and s[1:2] == "l":
                return m.start(1), _kg_files(m.group(1))
        start = _NAME_TAIL.search(line).start()
        prefix = line[start:]
        if not prefix:
            return start, []
        return start, sorted(n for n in set(self.names()) if n.startswith(prefix))

    def complete_names(self, prefix):
        """The words completing prefix as a name, sorted."""
        return self.matches(prefix)[1]

    def __call__(self, text, state, line=None, begidx=None):
        """The readline completer protocol; line and begidx default to the readline buffer."""
        if state == 0:
            if line is None:
                import readline
                line = readline.get_line_buffer()[:readline.get_endidx()]
                begidx = readline.get_begidx()
            start, words = self.matches(line)
            begidx = len(line) - len(text) if begidx is None else begidx
            if start >= begidx:
                words = [line[begidx:start] + w for w in words]
            else:
                words = [w[begidx - start:] for w in words]
            self._state = words
        return self._state[state] if state < len(self._state) else None


# ------------------------------------------------------------------- history

HISTORY_LIMIT = 1000


def history_path(environ=None):
    """The history file: $KGPY_HISTORY (empty disables, returns None) or ~/.kgpy_history."""
    env = os.environ if environ is None else environ
    if "KGPY_HISTORY" in env:
        v = env["KGPY_HISTORY"]
        return v or None
    return os.path.join(os.path.expanduser("~"), ".kgpy_history")


def _encode(item):
    return item.replace("\\", "\\\\").replace("\n", "\\n")


def _decode(line):
    out, i = [], 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            out.append("\n" if line[i + 1] == "n" else line[i + 1])
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


class History:
    """
    Persistent history over a readline-like module (add_history,
    get_history_item, get_current_history_length, remove_history_item,
    clear_history). One line per item, newlines in an item written as \\n, so
    a multi-line entry is one item. Every method is a no-op or swallows OSError
    when there is no readline or no writable file.
    """

    def __init__(self, rl, path):
        self.rl = rl
        self.path = path

    @property
    def active(self):
        return self.rl is not None

    def load(self):
        if self.rl is None or not self.path:
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                lines = f.read().split("\n")
        except (OSError, UnicodeError):
            return
        for ln in [x for x in lines if x][-HISTORY_LIMIT:]:
            self.rl.add_history(_decode(ln))

    def save(self):
        if self.rl is None or not self.path:
            return
        try:
            n = self.rl.get_current_history_length()
            items = [self.rl.get_history_item(i) for i in range(1, n + 1)]
            items = [x for x in items if x][-HISTORY_LIMIT:]
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("".join(_encode(x) + "\n" for x in items))
        except (OSError, UnicodeError, ValueError):
            pass

    def mark(self):
        return self.rl.get_current_history_length() if self.rl is not None else 0

    def replace_entry(self, mark, text):
        """Replace the history items added since mark by one item holding text."""
        if self.rl is None:
            return
        try:
            for i in range(self.rl.get_current_history_length(), mark, -1):
                self.rl.remove_history_item(i - 1)
            if text.strip():
                self.rl.add_history(text)
        except (ValueError, OSError):
            pass
