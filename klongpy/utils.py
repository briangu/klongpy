import collections


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
        self.subscribers = []

    def subscribe(self, callback):
        self.subscribers.append(callback)

    def unsubscribe(self, callback):
        try:
            self.subscribers.remove(callback)
        except ValueError:
            # Callback was not found in the list of subscribers
            pass

    def trigger(self):
        for callback in self.subscribers:
            callback()
