class RequestCache:
    def __init__(self, entries=[]):
        self.entries = entries

    def remember(self, key, value):
        self.entries.append((key, value))

    def get(self, key):
        for entry_key, entry_value in self.entries:
            if entry_key == key:
                return entry_value
        return None
