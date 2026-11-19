"""
REPL input helpers that need no terminal: the completeness check for multi-line
entry, the tab completer and the persistent history.

    check_text(text) -> "complete" | "incomplete" | "error"       (the native engine's repl_check)
    Completer(klong, sys_cmds)         .matches(line) -> (start, [words]); callable as a readline completer
    History(readline_module, path)     load() / save() / replace_entry(mark, text)

The completeness check and the matching of names, ]-commands and file names are the extension's (klong_tools::repl); what is here is
the readline protocol around them and the history file.
"""
import os

from .native_ext import native

__all__ = ['check_text', 'Completer', 'History', 'history_path', 'HISTORY_LIMIT']


def check_text(text, name="<repl>", comments=True):
    """Whether text is a whole REPL entry: "incomplete" when an opener ( ( [ { :[ :{ :( ) is still open, a string is
    unterminated, or (comments) a :" comment or a .comment("X") is left open; "error" when a closer does not match the
    innermost opener or closes nothing; else "complete" (a trailing dyad, 1+, is complete). `name` is accepted for the callers
    that give the entry one."""
    return native().repl_check(text, comments)


# ---------------------------------------------------------------- completion

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
            return [str(n) for n in self.klong.names()]
        except Exception:
            return []

    def matches(self, line):
        """(start, words) for the text before the cursor; each word replaces line[start:]."""
        return native().repl_complete(line, self.names(), [str(k) for k in self.sys_cmds], os.environ.get("KLONGPATH") or "")

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
