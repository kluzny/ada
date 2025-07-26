from typing import List
from ada.entry import Entry
from ada.formatter import block


class Conversation:
    """
    A history of Agent interactions
    """

    history: List[Entry] = []

    def __init__(self):
        pass

    def append_entry(self, entry: Entry):
        self.history.append(entry)

    def append(self, author: str, body: str):
        entry = Entry(author, body)
        self.append_entry(entry)

    def clear(self):
        self.history = []
        print(block("HISTORY CLEARED"))

    def __str__(self):
        output = ""
        output += block("HISTORY START")
        for entry in self.history:
            output += str(entry) + "\n"
        output += block("HISTORY END")
        return output.strip()
