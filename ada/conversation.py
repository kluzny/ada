from typing import List

from ada.entry import Entry
from ada.formatter import block
from ada.logger import build_logger
from ada.response import Response

logger = build_logger(__name__)


class Conversation:
    """
    A history of Agent interactions
    """

    history: List[Entry] = []

    def __init__(self):
        logger.info("initializing conversation")

    def append(self, author: str, body: str):
        entry = Entry(author, body)
        self.history.append(entry)

    def append_response(self, author: str, response: Response):
        entry = Entry(
            author,
            response.body,
            role=response.role,
            content=response.content,
        )
        self.history.append(entry)

    def clear(self):
        self.history = []
        print(block("HISTORY CLEARED").strip())

    def messages(self) -> List[dict]:
        return [entry.message() for entry in self.history]

    def __str__(self):
        output = ""
        output += block("HISTORY START")
        for entry in self.history:
            output += str(entry) + "\n"
        output += block("HISTORY END")
        return output.strip()
