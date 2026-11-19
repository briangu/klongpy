import collections
import collections.abc


class ReadonlyDict(collections.abc.Mapping):

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


class CallbackEvent:
    def __init__(self):
        self.subscribers = set()

    def subscribe(self, callback):
        self.subscribers.add(callback)

    def unsubscribe(self, callback):
        try:
            self.subscribers.discard(callback)
        except ValueError:
            # Callback was not found in the list of subscribers
            pass

    def trigger(self):
        for callback in list(self.subscribers):
            callback()


STRICT_LEVELS = (0, 1, 2)


def check_strict_level(n):
    """
    The strict level n if it is one (the integer 0, 1 or 2), else ValueError.

        0   the default: everything the language always accepted
        1   malformed input is a syntax error; inside a function, assigning
            to or reading an undeclared name is a name error
        2   also: reading an undefined name anywhere, reading a local before
            it is assigned and passing too many arguments are errors, and a
            quoted symbol is always the symbol

    The table is docs/errors.md.
    """
    if type(n) is not int or n not in STRICT_LEVELS:
        raise ValueError(f"strict level must be 0, 1 or 2, not {n!r}")
    return n
