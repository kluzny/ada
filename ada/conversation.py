from typing import List
from pydantic import BaseModel, Field

from ada.entry import Entry
from ada.formatter import block
from ada.logger import build_logger
from ada.response import Response

logger = build_logger(__name__)


class Conversation(BaseModel):
    """
    A history of Agent interactions
    """

    history: List[Entry] = Field(
        default_factory=list,
        description="List of conversation entries",
    )

    def append(self, author: str, body: str):
        entry = Entry(author=author, body=body)
        self.history.append(entry)

    def append_response(self, author: str, response: Response):
        entry = Entry(
            author=author,
            body=response.body,
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
